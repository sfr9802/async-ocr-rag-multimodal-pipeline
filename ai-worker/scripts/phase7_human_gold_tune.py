"""Phase 7.x — human-weighted gold seed 50 + silver 500 tuning CLI.

Drives the gold-50 + silver-500 pair through one *baseline* and zero
or more *candidate* retrieval variants, then writes the comparison
artefact bundle the spec asks for. Two execution modes:

  * **Live mode** (``--live``): builds a real :class:`Retriever` for
    every variant from a pre-built FAISS cache + the matching
    ``rag_chunks_*.jsonl`` file, runs every gold + silver query, and
    scores the results.

  * **Replay mode** (default): consumes pre-computed retrieval result
    JSONL files (``--baseline-results`` / ``--candidate-results``).
    The replay payload is the same shape ``write_*_results`` produces;
    keeping replay first-class lets the test suite, sweeps, and
    re-runs use the *same* CLI without paying the embedder cost twice.

The CLI never imports the production retriever code at import time —
the live-mode imports happen inside ``run_live_variant`` so the test
suite can exercise the replay path without :class:`SentenceTransformerEmbedder`
loading.

Outputs (under ``--report-dir``):

  - ``baseline_gold_summary.json``
  - ``baseline_silver_summary.json``
  - ``candidate_results.jsonl``       (one entry per variant)
  - ``comparison_summary.json``
  - ``comparison_report.md``
  - ``failure_audit_gold.jsonl`` + ``failure_audit_gold.md``
  - ``failure_audit_silver.jsonl`` + ``failure_audit_silver.md``
    (silver audit only includes top-10 misses, not all rows)
  - ``best_config.json``
  - ``best_config.env``
  - ``manifest.json``
  - per-variant retrieval JSONL: ``retrieval_<variant>_gold.jsonl`` /
    ``retrieval_<variant>_silver.jsonl`` (only in live mode — replay
    mode reuses the inputs in place)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from eval.harness.phase7_human_gold_tune import (
    HUMAN_FOCUS_DISCLAIMER,
    ComparisonResult,
    FailureAuditRow,
    GoldQueryEvalRow,
    GoldSeedDataset,
    GoldSeedValidationError,
    GoldSummary,
    GuardrailWarning,
    RetrievedDoc,
    SilverDataset,
    SilverQueryEvalRow,
    SilverSummary,
    VariantResult,
    build_failure_audit_row,
    classify_failure,
    compare_variants,
    comparison_to_dict,
    evaluate_gold,
    evaluate_silver,
    gold_summary_to_dict,
    load_human_gold_seed_50,
    load_llm_silver_500,
    render_comparison_report,
    render_failure_audit_md,
    silver_summary_to_dict,
    summarize_gold,
    summarize_silver,
)


log = logging.getLogger("scripts.phase7_human_gold_tune")


# ---------------------------------------------------------------------------
# Variant configuration — declarative
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSpec:
    """One retrieval variant definition the CLI knows how to instantiate.

    ``cache_dir_relative`` is resolved against ``--index-root``. The
    Retriever is constructed with ``top_k`` / ``candidate_k`` /
    ``use_mmr`` / ``mmr_lambda`` overrides; everything not set falls
    back to the production defaults.

    ``rag_chunks_path_relative`` is the matching ``rag_chunks_*.jsonl``
    on disk — required so the harness can join doc_id → title +
    chunk_id → section_path metadata onto the retrieved chunks. That
    metadata isn't in the FAISS chunks.jsonl so the join lives at the
    eval boundary, not inside the production retriever.
    """

    name: str
    cache_dir_relative: str
    rag_chunks_path_relative: str
    top_k: int = 10
    candidate_k: Optional[int] = None
    use_mmr: bool = False
    mmr_lambda: float = 0.7
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default variant set: baseline = production default (retrieval_title_section
# with top_k=10), and a small grid of candidate sweeps that cover the
# axes the spec calls out (candidate_k, MMR, top_k).
#
# Path conventions:
#   - ``cache_dir_relative`` is resolved against ``--index-root``.
#   - ``rag_chunks_path_relative`` is resolved against
#     ``--rag-chunks-root`` (defaulting to the working directory). Use
#     a path the CLI can find verbatim from the project root, e.g.
#     ``eval/reports/phase7/silver500/retrieval/<chunks>.jsonl``.
_DEFAULT_RAG_CHUNKS_RTS = (
    "eval/reports/phase7/silver500/retrieval/"
    "rag_chunks_retrieval_title_section.jsonl"
)
_DEFAULT_RAG_CHUNKS_TS = (
    "eval/reports/phase7/silver500/retrieval/"
    "rag_chunks_title_section.jsonl"
)
DEFAULT_VARIANTS: Tuple[VariantSpec, ...] = (
    VariantSpec(
        name="baseline_retrieval_title_section_top10",
        cache_dir_relative="namu-v4-2008-2026-04-retrieval-title-section-mseq512",
        rag_chunks_path_relative=_DEFAULT_RAG_CHUNKS_RTS,
        top_k=10,
        description="Production default — retrieval_title_section + top_k=10.",
    ),
    VariantSpec(
        name="cand_top10_candk30",
        cache_dir_relative="namu-v4-2008-2026-04-retrieval-title-section-mseq512",
        rag_chunks_path_relative=_DEFAULT_RAG_CHUNKS_RTS,
        top_k=10,
        candidate_k=30,
        description="Wide candidate pool (30) feeding the same top_k=10.",
    ),
    VariantSpec(
        name="cand_top10_mmr_lambda07",
        cache_dir_relative="namu-v4-2008-2026-04-retrieval-title-section-mseq512",
        rag_chunks_path_relative=_DEFAULT_RAG_CHUNKS_RTS,
        top_k=10,
        candidate_k=30,
        use_mmr=True,
        mmr_lambda=0.7,
        description="Wide candidate pool + MMR(λ=0.7) diversification.",
    ),
    VariantSpec(
        name="cand_title_section_top10",
        cache_dir_relative="namu-v4-2008-2026-04-title-section-mseq512",
        rag_chunks_path_relative=_DEFAULT_RAG_CHUNKS_TS,
        top_k=10,
        description=(
            "title_section variant (Phase 7.0 baseline before retrieval_title "
            "promotion) — sanity check that the promotion still wins on "
            "human-weighted gold."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Chunk metadata join — FAISS chunks → (title, section_path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkMetadata:
    """The doc_id → title + chunk_id → section_path map row.

    Built once per variant from the ``rag_chunks_*.jsonl`` export so
    the runner can attach human-meaningful titles + section paths to
    the bare ``RetrievedChunk`` instances the production retriever
    returns.
    """

    chunk_id: str
    doc_id: str
    title: str
    section_path: Tuple[str, ...]


def load_chunk_metadata(rag_chunks_path: Path) -> Dict[str, ChunkMetadata]:
    """Load the rag_chunks JSONL keyed by chunk_id.

    The export carries ``title``, ``section_path``, ``chunk_id``,
    ``doc_id`` per row; we only keep what the failure audit + section
    metric need so the in-memory map stays small (~135k chunks).
    """
    out: Dict[str, ChunkMetadata] = {}
    with Path(rag_chunks_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_id = str(rec.get("chunk_id") or "")
            if not chunk_id:
                continue
            sp = rec.get("section_path") or []
            if isinstance(sp, str):
                sp = [sp]
            out[chunk_id] = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=str(rec.get("doc_id") or ""),
                title=str(rec.get("title") or ""),
                section_path=tuple(str(x) for x in sp if str(x)),
            )
    log.info("loaded %d chunks of metadata from %s", len(out), rag_chunks_path)
    return out


def doc_title_map(metadata: Mapping[str, ChunkMetadata]) -> Dict[str, str]:
    """Reduce the chunk-id keyed map to doc_id → title (mode-of-titles).

    Inside one document chunks share a title, but a defensive caller
    might pass a heterogeneous map. We pick the most-frequent non-empty
    title per doc_id so a single mislabelled chunk doesn't pollute the
    audit display.
    """
    by_doc: Dict[str, Dict[str, int]] = {}
    for cm in metadata.values():
        if not cm.doc_id or not cm.title:
            continue
        by_doc.setdefault(cm.doc_id, {})
        by_doc[cm.doc_id][cm.title] = by_doc[cm.doc_id].get(cm.title, 0) + 1
    out: Dict[str, str] = {}
    for doc_id, counts in by_doc.items():
        title, _ = max(counts.items(), key=lambda kv: kv[1])
        out[doc_id] = title
    return out


# ---------------------------------------------------------------------------
# Retrieval result container — also used as the replay JSONL row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    """One (variant, query_id) retrieval result row.

    The replay JSONL is one of these per line. ``docs`` is a list of
    serialized :class:`RetrievedDoc` dicts (rank/chunk_id/page_id/title/
    section_path/score). The CLI accepts this same row shape on
    ``--baseline-results`` / ``--candidate-results``.
    """

    variant: str
    query_id: str
    query: str
    elapsed_ms: float
    docs: Tuple[RetrievedDoc, ...]

    def to_jsonl_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "query_id": self.query_id,
            "query": self.query,
            "elapsed_ms": self.elapsed_ms,
            "docs": [
                {
                    "rank": d.rank,
                    "chunk_id": d.chunk_id,
                    "page_id": d.page_id,
                    "title": d.title,
                    "section_path": list(d.section_path),
                    "score": d.score,
                }
                for d in self.docs
            ],
        }


def retrieval_result_from_dict(payload: Mapping[str, Any]) -> RetrievalResult:
    docs_raw = payload.get("docs") or []
    docs: List[RetrievedDoc] = []
    for d in docs_raw:
        docs.append(RetrievedDoc(
            rank=int(d.get("rank") or 0),
            chunk_id=str(d.get("chunk_id") or ""),
            page_id=str(d.get("page_id") or ""),
            title=str(d.get("title") or ""),
            section_path=tuple(str(x) for x in (d.get("section_path") or [])),
            score=(
                float(d.get("score"))
                if d.get("score") is not None else None
            ),
        ))
    return RetrievalResult(
        variant=str(payload.get("variant") or ""),
        query_id=str(payload.get("query_id") or ""),
        query=str(payload.get("query") or ""),
        elapsed_ms=float(payload.get("elapsed_ms") or 0.0),
        docs=tuple(docs),
    )


def write_retrieval_jsonl(
    rows: Iterable[RetrievalResult], out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")
    return out_path


def read_retrieval_jsonl(path: Path) -> List[RetrievalResult]:
    out: List[RetrievalResult] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            out.append(retrieval_result_from_dict(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# Live execution — builds a :class:`Retriever` and runs every query.
# ---------------------------------------------------------------------------


@dataclass
class LiveResources:
    """Cached embedder + per-cache-dir retriever / chunk-metadata.

    The CLI builds a single :class:`LiveResources` instance and threads
    it through every variant — variants that share an ``index_root /
    cache_dir_relative`` reuse the loaded :class:`Retriever` (the
    Phase 7.x default set has three variants on the
    ``retrieval-title-section`` index, so reusing saves three
    14-second index loads).

    The retriever's ``top_k`` / ``candidate_k`` are mutable post-
    construction; ``apply_variant`` flips them per-variant before the
    retrieve loop runs. MMR toggling uses the same private-attr
    poke the production registry does.
    """

    embedder: Any = None
    retrievers: Dict[str, Any] = field(default_factory=dict)
    chunk_metas: Dict[str, Dict[str, ChunkMetadata]] = field(default_factory=dict)
    doc_titles: Dict[str, Dict[str, str]] = field(default_factory=dict)
    embedding_model: str = "BAAI/bge-m3"
    max_seq_length: int = 512


def _ensure_embedder(res: LiveResources) -> Any:
    if res.embedder is None:
        from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
        res.embedder = SentenceTransformerEmbedder(
            model_name=res.embedding_model,
            max_seq_length=res.max_seq_length,
        )
    return res.embedder


def _ensure_retriever(
    res: LiveResources, cache_dir: Path, *, default_top_k: int = 10,
) -> Any:
    """Load a variant's index once and cache it on the resources object."""
    key = str(cache_dir)
    if key in res.retrievers:
        return res.retrievers[key]
    from eval.harness.embedding_text_reindex import load_variant_dense_stack
    embedder = _ensure_embedder(res)
    retriever, info, _ = load_variant_dense_stack(
        cache_dir, embedder=embedder, top_k=default_top_k,
    )
    log.info(
        "live: loaded retriever cache=%s chunks=%d dim=%d",
        cache_dir, info.chunk_count, info.dimension,
    )
    res.retrievers[key] = retriever
    return retriever


def _ensure_chunk_metadata(
    res: LiveResources, rag_chunks_path: Path,
) -> Tuple[Dict[str, ChunkMetadata], Dict[str, str]]:
    key = str(rag_chunks_path)
    if key in res.chunk_metas:
        return res.chunk_metas[key], res.doc_titles[key]
    cm = load_chunk_metadata(rag_chunks_path)
    dt = doc_title_map(cm)
    res.chunk_metas[key] = cm
    res.doc_titles[key] = dt
    return cm, dt


def run_live_variant(
    *,
    spec: VariantSpec,
    index_root: Path,
    queries: Sequence[Tuple[str, str]],   # (query_id, query)
    resources: Optional[LiveResources] = None,
    embedding_model: str = "BAAI/bge-m3",
    max_seq_length: int = 512,
    rag_chunks_root: Optional[Path] = None,
) -> List[RetrievalResult]:
    """Run a single variant over ``queries`` and return result rows.

    When ``resources`` is supplied, the embedder + retriever loaded for
    a previous variant on the same cache_dir are reused; pass ``None``
    to get a fresh load (matches the original test contract).
    """
    res = resources or LiveResources(
        embedding_model=embedding_model,
        max_seq_length=max_seq_length,
    )

    cache_dir = Path(index_root) / spec.cache_dir_relative
    rcr = Path(rag_chunks_root) if rag_chunks_root is not None else Path.cwd()
    chunks_arg = Path(spec.rag_chunks_path_relative)
    if chunks_arg.is_absolute():
        rag_chunks_path = chunks_arg
    else:
        rag_chunks_path = (rcr / chunks_arg).resolve()

    log.info(
        "live[%s]: cache=%s top_k=%d candidate_k=%s use_mmr=%s mmr_lambda=%.2f",
        spec.name, cache_dir, spec.top_k, spec.candidate_k,
        spec.use_mmr, spec.mmr_lambda,
    )
    log.info("live[%s]: rag_chunks=%s", spec.name, rag_chunks_path)

    chunk_meta, doc_titles = _ensure_chunk_metadata(res, rag_chunks_path)
    retriever = _ensure_retriever(res, cache_dir, default_top_k=spec.top_k)

    # Patch retriever knobs per-variant. ``_top_k`` / ``_candidate_k`` /
    # ``_use_mmr`` / ``_mmr_lambda`` are private but the production
    # registry pokes them the same way during config refresh.
    retriever._top_k = int(spec.top_k)
    if spec.candidate_k is None or int(spec.candidate_k) <= 0:
        retriever._candidate_k = retriever._top_k
    else:
        retriever._candidate_k = max(retriever._top_k, int(spec.candidate_k))
    retriever._use_mmr = bool(spec.use_mmr)
    retriever._mmr_lambda = float(max(0.0, min(1.0, spec.mmr_lambda)))

    rows: List[RetrievalResult] = []
    for i, (qid, query) in enumerate(queries, start=1):
        t0 = time.perf_counter()
        report = retriever.retrieve(query)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        docs = []
        for rank, c in enumerate(report.results[: spec.top_k], start=1):
            cm = chunk_meta.get(c.chunk_id)
            title = (cm.title if cm else "") or doc_titles.get(c.doc_id, "")
            section_path: Tuple[str, ...] = ()
            if cm:
                section_path = cm.section_path
            elif c.section:
                section_path = tuple(s.strip() for s in c.section.split(">") if s.strip())
            docs.append(RetrievedDoc(
                rank=rank,
                chunk_id=c.chunk_id,
                page_id=c.doc_id,
                title=title,
                section_path=section_path,
                score=float(c.score) if c.score is not None else None,
            ))
        rows.append(RetrievalResult(
            variant=spec.name,
            query_id=qid,
            query=query,
            elapsed_ms=elapsed_ms,
            docs=tuple(docs),
        ))
        if i % 50 == 0:
            log.info("live[%s]: %d/%d queries scored", spec.name, i, len(queries))

    return rows


# ---------------------------------------------------------------------------
# Pipeline glue: rows + retrievals → :class:`VariantResult`
# ---------------------------------------------------------------------------


def _retrievals_by_id(
    rows: Sequence[RetrievalResult],
) -> Dict[str, List[RetrievedDoc]]:
    out: Dict[str, List[RetrievedDoc]] = {}
    for r in rows:
        out[r.query_id] = list(r.docs)
    return out


def build_variant_result(
    *,
    variant: str,
    config: Mapping[str, Any],
    gold: GoldSeedDataset,
    silver: SilverDataset,
    gold_retrievals: Sequence[RetrievalResult],
    silver_retrievals: Sequence[RetrievalResult],
) -> VariantResult:
    """Compute per-query rows + summaries + audit for one variant."""
    gold_map = _retrievals_by_id(gold_retrievals)
    silver_map = _retrievals_by_id(silver_retrievals)
    gold_rows = evaluate_gold(gold, gold_map)
    silver_rows = evaluate_silver(silver, silver_map)
    gold_summary = summarize_gold(gold_rows)
    silver_summary = summarize_silver(silver_rows)
    audit = [build_failure_audit_row(r) for r in gold_rows]
    return VariantResult(
        variant=variant,
        gold_summary=gold_summary,
        silver_summary=silver_summary,
        gold_per_query=gold_rows,
        silver_per_query=silver_rows,
        failure_audit=audit,
        config=dict(config),
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_gold_summary(path: Path, summary: GoldSummary) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gold_summary_to_dict(summary)
    payload["human_focus_disclaimer"] = HUMAN_FOCUS_DISCLAIMER
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_silver_summary(path: Path, summary: SilverSummary) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = silver_summary_to_dict(summary)
    payload["human_focus_disclaimer"] = HUMAN_FOCUS_DISCLAIMER
    payload["note"] = (
        "Silver-500 is LLM-generated. Treat these metrics as a "
        "regression sanity check, NOT the primary tuning objective."
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_candidate_results_jsonl(
    path: Path, candidates: Sequence[VariantResult],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for c in candidates:
            payload = {
                "variant": c.variant,
                "config": c.config,
                "gold_summary": gold_summary_to_dict(c.gold_summary),
                "silver_summary": silver_summary_to_dict(c.silver_summary),
            }
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def write_comparison_summary(
    path: Path, comp: ComparisonResult,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(comparison_to_dict(comp), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_comparison_md(path: Path, comp: ComparisonResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_comparison_report(comp), encoding="utf-8")
    return path


def write_failure_audit(
    md_path: Path, jsonl_path: Path,
    rows: Sequence[FailureAuditRow], *, header: str,
) -> Tuple[Path, Path]:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_failure_audit_md(rows, header=header),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return md_path, jsonl_path


def write_best_config(
    *,
    json_path: Path,
    env_path: Path,
    comp: ComparisonResult,
    variants_index: Mapping[str, VariantSpec],
) -> Tuple[Path, Path]:
    """Persist the recommended config in both JSON and env-snippet form.

    The env snippet is shaped so a reviewer can paste the relevant
    lines into ``.env`` / ``settings.toml`` after the silver guardrail
    passes — we deliberately do NOT auto-apply, the spec is explicit
    that production code only changes via a normal PR.
    """
    best = comp.best_variant
    best_spec = variants_index.get(best)
    payload: Dict[str, Any] = {
        "best_variant": best,
        "best_reason": comp.best_reason,
        "primary_score": comp.baseline.gold_summary.primary_score,
        "primary_score_baseline": comp.baseline.gold_summary.primary_score,
        "deltas": comp.deltas.get(best, {}),
        "guardrails": [w.to_dict() for w in comp.guardrails.get(best, [])],
        "config": (best_spec.to_dict() if best_spec is not None else {}),
        "human_focus_disclaimer": HUMAN_FOCUS_DISCLAIMER,
    }
    if best_spec is not None:
        # Find the matching candidate to read its score.
        for c in comp.candidates:
            if c.variant == best:
                payload["primary_score"] = c.gold_summary.primary_score
                break
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env_lines: List[str] = [
        "# Phase 7.x human-weighted gold-50 best config snippet",
        "# DO NOT auto-apply. Promote via the standard config-change PR.",
        f"# {HUMAN_FOCUS_DISCLAIMER}",
        f"# best_variant={best}",
        f"# best_reason={comp.best_reason}",
    ]
    if best_spec is not None:
        env_lines.append(f"AIPIPELINE_WORKER_RAG_TOP_K={best_spec.top_k}")
        if best_spec.candidate_k is not None:
            env_lines.append(
                f"AIPIPELINE_WORKER_RAG_CANDIDATE_K={best_spec.candidate_k}"
            )
        env_lines.append(
            f"AIPIPELINE_WORKER_RAG_USE_MMR="
            f"{'true' if best_spec.use_mmr else 'false'}"
        )
        env_lines.append(
            f"AIPIPELINE_WORKER_RAG_MMR_LAMBDA={best_spec.mmr_lambda:.4f}"
        )
        env_lines.append(
            f"# index variant cache: {best_spec.cache_dir_relative}"
        )
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    return json_path, env_path


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    variants: Sequence[VariantSpec],
    comp: ComparisonResult,
    gold_path: Path,
    silver_path: Path,
    issues_gold: Sequence[Any],
    issues_silver: Sequence[Any],
) -> Path:
    payload: Dict[str, Any] = {
        "phase": "7.x_human_gold_seed_50_tuning",
        "human_focus_disclaimer": HUMAN_FOCUS_DISCLAIMER,
        "gold_path": str(gold_path),
        "silver_path": str(silver_path),
        "report_dir": str(args.report_dir),
        "index_root": str(args.index_root) if args.index_root else None,
        "embedding_model": args.embedding_model,
        "max_seq_length": int(args.max_seq_length),
        "live": bool(args.live),
        "variants": [v.to_dict() for v in variants],
        "best_variant": comp.best_variant,
        "best_reason": comp.best_reason,
        "primary_score_baseline": comp.baseline.gold_summary.primary_score,
        "deltas": comp.deltas,
        "guardrails": {
            v: [w.to_dict() for w in ws] for v, ws in comp.guardrails.items()
        },
        "validation_issues_gold": [
            i.to_dict() if hasattr(i, "to_dict") else i for i in issues_gold
        ],
        "validation_issues_silver": [
            i.to_dict() if hasattr(i, "to_dict") else i for i in issues_silver
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Replay-mode entry: read pre-computed retrieval JSONL files
# ---------------------------------------------------------------------------


def load_replay_results(
    paths: Sequence[Path],
) -> Dict[str, List[RetrievalResult]]:
    """Read multiple replay JSONL files keyed by their inferred variant.

    Each line in a replay file carries a ``variant`` field; rows are
    grouped by that variant so the caller doesn't have to rely on file
    naming.
    """
    by_variant: Dict[str, List[RetrievalResult]] = {}
    for p in paths:
        for r in read_retrieval_jsonl(p):
            by_variant.setdefault(r.variant, []).append(r)
    return by_variant


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(
        "Phase 7.x human-weighted gold seed 50 + silver 500 tuning. "
        "NOT a generic benchmark — focus set on subpage / named-subpage "
        "retrieval failures."
    ))
    p.add_argument(
        "--gold-path", type=Path, required=True,
        help="Path to phase7_human_gold_seed_50.csv (with eval_use + eval_weight).",
    )
    p.add_argument(
        "--silver-path", type=Path, required=True,
        help="Path to queries_v4_llm_silver_500.jsonl.",
    )
    p.add_argument(
        "--report-dir", type=Path, required=True,
        help="Output directory for the bundle.",
    )
    p.add_argument(
        "--index-root", type=Path, default=None,
        help=(
            "FAISS index cache root (e.g. eval/indexes/). Required when "
            "--live is set."
        ),
    )
    p.add_argument(
        "--rag-chunks-root", type=Path, default=None,
        help=(
            "Root the variant's rag_chunks_*.jsonl path is resolved "
            "against. Defaults to the variant's cache_dir, then the "
            "value here. Set this to the eval reports root if your "
            "rag_chunks paths are relative to the report tree."
        ),
    )
    p.add_argument(
        "--embedding-model", default="BAAI/bge-m3",
        help="Embedding model — must match the index build.",
    )
    p.add_argument(
        "--max-seq-length", type=int, default=512,
        help="Embedder max_seq_length — must match the index build.",
    )
    p.add_argument(
        "--live", action="store_true",
        help=(
            "Run real retrievers via SentenceTransformerEmbedder. Default "
            "is replay mode (consume --baseline-results / "
            "--candidate-results)."
        ),
    )
    p.add_argument(
        "--variants", default="default",
        help=(
            "Comma-separated variant names to run (subset of "
            "DEFAULT_VARIANTS by name) or 'default' for the full "
            "default set. Use 'baseline_only' to only score the "
            "baseline (useful for first-look reports)."
        ),
    )
    p.add_argument(
        "--baseline-variant", default="baseline_retrieval_title_section_top10",
        help=(
            "Name of the variant treated as the baseline; remainder are "
            "candidates."
        ),
    )
    p.add_argument(
        "--baseline-results", type=Path, default=None,
        help=(
            "(Replay mode) JSONL of retrieval results for the baseline "
            "variant, written by a prior --live run."
        ),
    )
    p.add_argument(
        "--candidate-results", type=Path, action="append", default=[],
        help=(
            "(Replay mode) JSONL of retrieval results for one candidate "
            "variant. May be passed multiple times."
        ),
    )
    p.add_argument(
        "--include-silver-guardrail", action="store_true",
        help=(
            "Run the full silver-500 set through every variant. When "
            "off, silver evaluation is skipped and only gold metrics + "
            "audits are produced (faster smoke-test)."
        ),
    )
    p.add_argument(
        "--silver-guardrail-fraction", type=float, default=1.0,
        help=(
            "Fraction (0,1] of silver-500 to score. Defaults to the full "
            "set; set lower for fast iteration when you trust the "
            "subset to be representative."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the silver-fraction subsample.",
    )
    p.add_argument(
        "--audit-include-hits", action="store_true",
        help=(
            "Include hit@10 == 1 rows in the failure audit MD as well. "
            "Default keeps the audit focused on misses."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _resolve_variants(
    variants_arg: str,
) -> List[VariantSpec]:
    if variants_arg == "default":
        return list(DEFAULT_VARIANTS)
    if variants_arg == "baseline_only":
        return [DEFAULT_VARIANTS[0]]
    requested = {v.strip() for v in variants_arg.split(",") if v.strip()}
    out = [v for v in DEFAULT_VARIANTS if v.name in requested]
    missing = requested - {v.name for v in out}
    if missing:
        raise SystemExit(
            f"unknown variant(s): {sorted(missing)}. Known: "
            f"{[v.name for v in DEFAULT_VARIANTS]}"
        )
    return out


def _silver_subsample(
    silver: SilverDataset, fraction: float, seed: int,
) -> SilverDataset:
    """Down-sample the silver dataset deterministically by query_id."""
    if fraction >= 1.0:
        return silver
    if fraction <= 0.0:
        raise SystemExit("--silver-guardrail-fraction must be in (0, 1].")
    import random as _random
    rng = _random.Random(seed)
    rows = list(silver.rows)
    rng.shuffle(rows)
    keep = max(1, int(round(fraction * len(rows))))
    return SilverDataset(rows=rows[:keep], issues=list(silver.issues))


def _filter_results_by_ids(
    rows: Sequence[RetrievalResult], keep_ids: Iterable[str],
) -> List[RetrievalResult]:
    keep = set(keep_ids)
    return [r for r in rows if r.query_id in keep]


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading gold seed: %s", args.gold_path)
    gold = load_human_gold_seed_50(args.gold_path)
    log.info(
        "gold rows=%d issues=%d (errors=%d, warnings=%d)",
        len(gold.rows), len(gold.issues),
        sum(1 for i in gold.issues if i.severity == "error"),
        sum(1 for i in gold.issues if i.severity == "warning"),
    )

    log.info("loading silver: %s", args.silver_path)
    silver_full = load_llm_silver_500(args.silver_path)
    log.info(
        "silver rows=%d issues=%d", len(silver_full.rows),
        len(silver_full.issues),
    )

    silver = _silver_subsample(
        silver_full,
        fraction=float(args.silver_guardrail_fraction),
        seed=int(args.seed),
    )
    if len(silver.rows) != len(silver_full.rows):
        log.info(
            "silver downsampled to %d / %d rows (fraction=%s)",
            len(silver.rows), len(silver_full.rows),
            args.silver_guardrail_fraction,
        )

    variants = _resolve_variants(args.variants)
    if not variants:
        raise SystemExit("no variants selected.")
    variants_index = {v.name: v for v in variants}
    if args.baseline_variant not in variants_index:
        raise SystemExit(
            f"--baseline-variant={args.baseline_variant!r} not in selected "
            f"variants {[v.name for v in variants]}."
        )

    # ------------------------------------------------------------------
    # Gather retrievals (live mode runs them; replay mode reads JSONL)
    # ------------------------------------------------------------------
    gold_retrievals: Dict[str, List[RetrievalResult]] = {}
    silver_retrievals: Dict[str, List[RetrievalResult]] = {}

    if args.live:
        if args.index_root is None:
            raise SystemExit("--index-root is required when --live is set.")
        gold_queries = [(r.query_id, r.query) for r in gold.rows]
        silver_queries = [(r.query_id, r.query) for r in silver.rows] \
            if args.include_silver_guardrail else []
        # Share a single embedder + per-cache-dir retriever across
        # variants. Variants sharing an index pay the load cost once.
        resources = LiveResources(
            embedding_model=args.embedding_model,
            max_seq_length=int(args.max_seq_length),
        )
        for spec in variants:
            log.info("=== live retrieval: variant=%s ===", spec.name)
            gold_rows = run_live_variant(
                spec=spec,
                index_root=args.index_root,
                queries=gold_queries,
                resources=resources,
                rag_chunks_root=args.rag_chunks_root,
            )
            gold_retrievals[spec.name] = gold_rows
            write_retrieval_jsonl(
                gold_rows,
                report_dir / f"retrieval_{spec.name}_gold.jsonl",
            )
            if silver_queries:
                silver_rows = run_live_variant(
                    spec=spec,
                    index_root=args.index_root,
                    queries=silver_queries,
                    resources=resources,
                    rag_chunks_root=args.rag_chunks_root,
                )
                silver_retrievals[spec.name] = silver_rows
                write_retrieval_jsonl(
                    silver_rows,
                    report_dir / f"retrieval_{spec.name}_silver.jsonl",
                )
            else:
                silver_retrievals[spec.name] = []
    else:
        # Replay: pull from --baseline-results / --candidate-results.
        replay_paths: List[Path] = []
        if args.baseline_results is not None:
            replay_paths.append(args.baseline_results)
        replay_paths.extend(args.candidate_results or [])
        if not replay_paths:
            raise SystemExit(
                "replay mode requires --baseline-results and at least one "
                "--candidate-results, OR set --live to actually run "
                "retrievers."
            )
        all_replay = load_replay_results(replay_paths)
        for spec in variants:
            rows = all_replay.get(spec.name, [])
            if not rows:
                log.warning(
                    "variant %s has zero replay rows; treating as all-misses.",
                    spec.name,
                )
            gold_ids = {g.query_id for g in gold.rows}
            silver_ids = {s.query_id for s in silver.rows}
            gold_retrievals[spec.name] = _filter_results_by_ids(rows, gold_ids)
            silver_retrievals[spec.name] = _filter_results_by_ids(rows, silver_ids)

    # ------------------------------------------------------------------
    # Score every variant
    # ------------------------------------------------------------------
    variant_results: List[VariantResult] = []
    for spec in variants:
        vr = build_variant_result(
            variant=spec.name,
            config=spec.to_dict(),
            gold=gold,
            silver=silver,
            gold_retrievals=gold_retrievals.get(spec.name, []),
            silver_retrievals=silver_retrievals.get(spec.name, []),
        )
        variant_results.append(vr)
        log.info(
            "variant=%s gold.primary=%.6f silver.hit@5=%.4f",
            spec.name, vr.gold_summary.primary_score,
            vr.silver_summary.hit_at_5,
        )

    baseline = next(
        v for v in variant_results if v.variant == args.baseline_variant
    )
    candidates = [v for v in variant_results if v.variant != args.baseline_variant]
    comp = compare_variants(baseline=baseline, candidates=candidates)
    log.info("best_variant=%s reason=%s", comp.best_variant, comp.best_reason)

    # ------------------------------------------------------------------
    # Write the bundle
    # ------------------------------------------------------------------
    write_gold_summary(
        report_dir / "baseline_gold_summary.json",
        baseline.gold_summary,
    )
    write_silver_summary(
        report_dir / "baseline_silver_summary.json",
        baseline.silver_summary,
    )
    write_candidate_results_jsonl(
        report_dir / "candidate_results.jsonl",
        candidates,
    )
    write_comparison_summary(report_dir / "comparison_summary.json", comp)
    write_comparison_md(report_dir / "comparison_report.md", comp)

    # Failure audit (gold) — focused on misses by default.
    audit_rows = baseline.failure_audit
    if not args.audit_include_hits:
        audit_rows = [
            r for r in audit_rows
            if r.failure_reason != "UNKNOWN"
            or r.hit_at_10 == 0
        ]
    write_failure_audit(
        report_dir / "failure_audit_gold.md",
        report_dir / "failure_audit_gold.jsonl",
        audit_rows,
        header=(
            f"Failure audit — gold-50 (baseline={baseline.variant}). "
            "Each row is one human-reviewed query."
        ),
    )

    # Failure audit (silver) — top-10 misses only, lighter per-row info.
    silver_audit_rows: List[FailureAuditRow] = []
    for sr in baseline.silver_per_query:
        if sr.expected_not_in_corpus or not sr.expected_page_id:
            continue
        if sr.hit_at_10 == 1:
            continue
        # Build a lightweight failure-audit row from the silver eval row
        # — silver doesn't carry retrieved docs; we only render the bare
        # row so the reviewer sees which silver IDs missed.
        silver_audit_rows.append(FailureAuditRow(
            query_id=sr.query_id,
            query=sr.query,
            normalized_eval_group="SILVER",
            eval_weight=0.0,
            bucket=sr.bucket,
            query_type=sr.query_type,
            expected_title="",
            expected_page_id=sr.expected_page_id,
            expected_section_path=(),
            top1_title=None,
            top1_page_id=None,
            top1_score=None,
            hit_at_1=sr.hit_at_1,
            hit_at_3=sr.hit_at_3,
            hit_at_5=sr.hit_at_5,
            hit_at_10=sr.hit_at_10,
            top_k_titles=(),
            top_k_page_ids=(),
            top_k_section_paths=(),
            failure_reason="UNKNOWN",
        ))
    write_failure_audit(
        report_dir / "failure_audit_silver.md",
        report_dir / "failure_audit_silver.jsonl",
        silver_audit_rows,
        header=(
            f"Failure audit — silver-500 misses (baseline={baseline.variant}). "
            "Diagnostic only: silver labels are LLM-generated, not human-verified."
        ),
    )

    write_best_config(
        json_path=report_dir / "best_config.json",
        env_path=report_dir / "best_config.env",
        comp=comp,
        variants_index=variants_index,
    )
    write_manifest(
        report_dir / "manifest.json",
        args=args,
        variants=variants,
        comp=comp,
        gold_path=args.gold_path,
        silver_path=args.silver_path,
        issues_gold=gold.issues,
        issues_silver=silver_full.issues,
    )

    log.info("wrote bundle to %s", report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
