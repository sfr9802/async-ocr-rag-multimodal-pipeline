"""Confirm sweep — rerank-input cap policy comparison.

Phase 2 follow-up to ``confirm_reranker_input_format``. The verdict
from that run was ``ADOPT_RERANKER_TITLE_PREFIX`` on the
``title_section`` index × ``title_plus_chunk`` format. The audit
flagged ``gold_chunk_not_in_rerank_input_pool`` as the dominant
remaining failure, with ``character_relation`` queries plateaued at
hit@5 ≈ 0.38–0.40. The hypothesis: ``title_cap_rerank_input=1`` is
clipping the gold chunk before the cross-encoder sees it.

Independent variable: cap policy on the rerank-input slice. Held
constant: title_section index, title_plus_chunk reranker format,
candidate_k=100, final_top_k=8, rerank_in=16, MMR λ=0.65, mmr_k=48,
title_cap_final=2.

Cap policies under test (ordered as the spec lists them):

  1. ``title_cap_1``        — anchor (current best from prior verdict).
  2. ``title_cap_2``        — relax title cap to 2.
  3. ``title_cap_3``        — relax title cap to 3.
  4. ``no_cap``             — no rerank-input cap; final cap stays at 2.
  5. ``doc_id_cap``         — strict per-doc_id cap (cap=2).
  6. ``section_path_cap``   — per-(doc_id, section) cap (cap=2).

Eval-only / report-only. Production code (``app/``) is **not** modified.
The existing FAISS ``title_section`` cache is reused as-is — this run
does NOT regenerate any embedding.

Outputs (under ``eval/reports/retrieval-rerank-input-cap-policy-
confirm-<TIMESTAMP>/``):

  - ``summary.csv``               — flat headline metrics per policy
  - ``summary.json``              — full RetrievalEvalSummary per policy
  - ``comparison_report.md``      — narrative + final verdict
  - ``per_query_results.jsonl``   — one row per (policy, query)
  - ``per_query_diffs.jsonl``     — improved/regressed query lists vs anchor
  - ``cap_audit.jsonl``           — per-query gold-rank audit per policy
  - ``regression_guard.md``       — per-policy pass/fail vs anchor
  - ``config_dump.json``          — frozen run knobs
  - ``index_manifest.json``       — variant cache dir + manifest
  - ``bucket_metrics.json``       — character_relation / plot_event slices

Run::

    python -m scripts.confirm_rerank_input_cap_policy
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("confirm_rerank_input_cap_policy")


_DEFAULT_DATASET = Path("eval/eval_queries/anime_silver_200.jsonl")
_DEFAULT_CORPUS = Path(
    "eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl"
)
_DEFAULT_REPORTS_ROOT = Path("eval/reports")
_DEFAULT_QUERY_TYPE_DRAFT = Path(
    "eval/eval_queries/anime_silver_200.query_type_draft.jsonl"
)
_DEFAULT_CACHE_ROOT = Path("eval/agent_loop_ab/_indexes")

# Held-constant — these mirror the prior format-confirm sweep so the
# cap-policy axis is the only independent variable.
_VARIANT = "title_section"
_RERANKER_INPUT_FORMAT = "title_plus_chunk"
_CANDIDATE_K = 100
_FINAL_TOP_K = 8
_RERANK_IN = 16
_USE_MMR = True
_MMR_LAMBDA = 0.65
_MMR_K = 48
_TITLE_CAP_FINAL = 2

# Policy comparison cap (used by title_cap_2/3, doc_id_cap, section_path_cap).
_POLICY_TITLE_CAP_2 = 2
_POLICY_TITLE_CAP_3 = 3
_POLICY_DOC_ID_CAP_VALUE = 2
_POLICY_SECTION_CAP_VALUE = 2

# Bucket keys used in the character_relation / plot_event breakdown.
_BUCKET_CHARACTER_RELATION = "character_relation"
_BUCKET_PLOT_EVENT = "plot_event"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _default_out_dir() -> Path:
    return (
        _DEFAULT_REPORTS_ROOT
        / f"retrieval-rerank-input-cap-policy-confirm-{_now_stamp()}"
    )


def _f(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_signed(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:+.4f}"


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Policy roster — matches the order in the module docstring + spec.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicySpec:
    label: str
    description: str
    builder_kind: str  # "title_cap" / "doc_id_cap" / "no_cap" / "section_path_cap"
    cap: Optional[int]


def default_policy_specs() -> List[PolicySpec]:
    """Return the 6-policy roster called out by the cap-policy spec."""
    return [
        PolicySpec(
            label="title_cap_1",
            description=(
                "Anchor — title cap=1 on rerank input (current best "
                "from the prior reranker-format sweep)."
            ),
            builder_kind="title_cap",
            cap=1,
        ),
        PolicySpec(
            label="title_cap_2",
            description=(
                "Title cap relaxed to 2 — does relaxing the cap let "
                "more gold chunks through?"
            ),
            builder_kind="title_cap",
            cap=_POLICY_TITLE_CAP_2,
        ),
        PolicySpec(
            label="title_cap_3",
            description="Title cap relaxed to 3 — further relaxation.",
            builder_kind="title_cap",
            cap=_POLICY_TITLE_CAP_3,
        ),
        PolicySpec(
            label="no_cap",
            description=(
                "No rerank-input cap. Final cap stays at "
                f"{_TITLE_CAP_FINAL} so output diversity is bounded."
            ),
            builder_kind="no_cap",
            cap=None,
        ),
        PolicySpec(
            label="doc_id_cap",
            description=(
                f"Strict per-doc_id cap={_POLICY_DOC_ID_CAP_VALUE}. "
                "Same-titled different-doc rows do NOT collapse."
            ),
            builder_kind="doc_id_cap",
            cap=_POLICY_DOC_ID_CAP_VALUE,
        ),
        PolicySpec(
            label="section_path_cap",
            description=(
                f"Per-(doc_id, section) cap={_POLICY_SECTION_CAP_VALUE}. "
                "Different sections of the same doc survive separately."
            ),
            builder_kind="section_path_cap",
            cap=_POLICY_SECTION_CAP_VALUE,
        ),
    ]


def build_cap_policy(spec: PolicySpec, *, title_provider: Any):
    from eval.harness.cap_policy import (
        DocIdCapPolicy,
        NoCapPolicy,
        SectionPathCapPolicy,
        TitleCapPolicy,
    )

    if spec.builder_kind == "title_cap":
        return TitleCapPolicy(spec.cap, title_provider=title_provider)
    if spec.builder_kind == "doc_id_cap":
        return DocIdCapPolicy(spec.cap)
    if spec.builder_kind == "no_cap":
        return NoCapPolicy()
    if spec.builder_kind == "section_path_cap":
        return SectionPathCapPolicy(spec.cap)
    raise ValueError(f"Unknown policy builder_kind: {spec.builder_kind!r}")


# ---------------------------------------------------------------------------
# Stack management — load existing variant cache, never re-index here.
# ---------------------------------------------------------------------------


@dataclass
class VariantStack:
    variant: str
    cache_dir: Path
    retriever: Any
    base_reranker: Any
    info: Any
    manifest: Any


def _load_variant_stack(
    *, args: argparse.Namespace,
) -> VariantStack:
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.capabilities.rag.reranker import CrossEncoderReranker
    from app.core.config import get_settings

    from eval.harness.embedding_text_reindex import (
        default_cache_dir_for_variant,
        load_variant_dense_stack,
    )

    settings = get_settings()
    if args.title_section_cache_dir_arg:
        cache_dir = Path(args.title_section_cache_dir_arg)
    else:
        cache_dir = default_cache_dir_for_variant(
            cache_root=Path(args.cache_root),
            embedding_model=settings.rag_embedding_model,
            max_seq_length=int(args.max_seq_length),
            corpus_path=Path(args.corpus),
            variant=_VARIANT,
        )
    log.info("variant=%s cache_dir=%s", _VARIANT, cache_dir)

    if not (
        (cache_dir / "faiss.index").exists()
        and (cache_dir / "build.json").exists()
        and (cache_dir / "chunks.jsonl").exists()
    ):
        raise FileNotFoundError(
            f"Variant cache {cache_dir} is incomplete; this script "
            "does not re-index. Run confirm_embedding_text_variant.py "
            "first to build the cache."
        )

    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
        max_seq_length=int(args.max_seq_length),
        batch_size=int(args.embed_batch_size),
        show_progress_bar=False,
        cuda_alloc_conf=settings.rag_embedding_cuda_alloc_conf or None,
    )
    base_reranker = CrossEncoderReranker(
        model_name=str(args.reranker_model),
        max_length=int(args.reranker_max_length),
        batch_size=int(args.reranker_batch_size),
        text_max_chars=int(args.reranker_text_max_chars),
        device=args.reranker_device or None,
        collect_stage_timings=False,
    )
    retriever, info, manifest = load_variant_dense_stack(
        cache_dir,
        embedder=embedder,
        top_k=10,
        reranker=base_reranker,
        candidate_k=50,
    )
    return VariantStack(
        variant=_VARIANT,
        cache_dir=cache_dir,
        retriever=retriever,
        base_reranker=base_reranker,
        info=info,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Per-policy eval loop
# ---------------------------------------------------------------------------


@dataclass
class PolicyRun:
    policy: str
    description: str
    spec: PolicySpec
    summary: Any
    rows: List[Any]
    audit_rows: List[Dict[str, Any]]
    started_at: str
    finished_at: str


class _AuditCapturer:
    """Iterator hook — capture last_audit per query_id during eval."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._query_to_id: Dict[str, str] = {}
        self.events: Dict[str, Any] = {}

    def name_for_query(self, query: str, query_id: str) -> None:
        self._query_to_id[query] = query_id

    def retrieve(self, query: str) -> Any:
        report = self._adapter.retrieve(query)
        qid = self._query_to_id.get(query, "")
        if qid:
            self.events[qid] = self._adapter.last_audit
        return report

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)


def _run_policy(
    *,
    spec: PolicySpec,
    stack: VariantStack,
    dataset: List[Dict[str, Any]],
    args: argparse.Namespace,
    title_provider: Any,
    query_type_by_id: Optional[Dict[str, str]] = None,
) -> PolicyRun:
    from eval.harness.cap_policy_audit import (
        CapPolicyAuditAdapter,
        CapPolicyAuditConfig,
        score_audit_event,
    )
    from eval.harness.retrieval_eval import (
        DEFAULT_CANDIDATE_KS, DEFAULT_DIVERSITY_KS, run_retrieval_eval,
    )
    from eval.harness.reranker_input_format import FormattingRerankerWrapper

    candidate_ks = tuple(sorted(set(list(DEFAULT_CANDIDATE_KS) + [200])))

    cap_policy = build_cap_policy(spec, title_provider=title_provider)
    final_cap_policy = build_cap_policy(
        PolicySpec(
            label="final_title_cap_2",
            description="Final cap=2 (held constant across runs).",
            builder_kind="title_cap",
            cap=_TITLE_CAP_FINAL,
        ),
        title_provider=title_provider,
    )

    formatted_reranker = FormattingRerankerWrapper(
        stack.base_reranker,
        fmt=_RERANKER_INPUT_FORMAT,
        title_provider=title_provider,
        record_input_previews=False,
    )

    audit_config = CapPolicyAuditConfig(
        candidate_k=_CANDIDATE_K,
        final_top_k=_FINAL_TOP_K,
        rerank_in=_RERANK_IN,
        cap_policy_rerank_input=cap_policy,
        cap_policy_final=final_cap_policy,
        use_mmr=_USE_MMR,
        mmr_lambda=_MMR_LAMBDA,
        mmr_k=_MMR_K,
    )

    adapter = CapPolicyAuditAdapter(
        stack.retriever,
        config=audit_config,
        final_reranker=formatted_reranker,
        title_provider=title_provider,
        name=f"{_VARIANT}/{_RERANKER_INPUT_FORMAT}/{spec.label}",
    )
    capturer = _AuditCapturer(adapter)
    for row in dataset:
        capturer.name_for_query(
            str(row.get("query") or ""), str(row.get("id") or ""),
        )

    log.info(
        "[policy=%s] cap=%s start (cand_k=%d top_k=%d rerank_in=%d "
        "mmr=%s λ=%.2f cap_final=%d)",
        spec.label, spec.cap, _CANDIDATE_K, _FINAL_TOP_K, _RERANK_IN,
        _USE_MMR, _MMR_LAMBDA, _TITLE_CAP_FINAL,
    )

    started_at = datetime.now().isoformat(timespec="seconds")
    summary, rows, _, _ = run_retrieval_eval(
        list(dataset),
        retriever=capturer,
        top_k=_FINAL_TOP_K,
        mrr_k=10,
        ndcg_k=10,
        candidate_ks=candidate_ks,
        diversity_ks=DEFAULT_DIVERSITY_KS,
        dataset_path=str(args.dataset),
        corpus_path=str(args.corpus),
    )
    finished_at = datetime.now().isoformat(timespec="seconds")
    log.info(
        "  [policy=%s] hit@5=%.4f mrr@10=%.4f cand@50=%s p95=%.1fms",
        spec.label,
        (summary.mean_hit_at_5 or 0.0),
        (summary.mean_mrr_at_10 or 0.0),
        (summary.candidate_hit_rates or {}).get("50"),
        float(
            summary.p95_total_retrieval_ms
            or summary.p95_retrieval_ms
            or 0.0
        ),
    )

    audit_rows: List[Dict[str, Any]] = []
    for row in rows:
        rid = getattr(row, "id", None)
        if not rid:
            continue
        event = capturer.events.get(str(rid))
        if event is None:
            continue
        scored = score_audit_event(
            event,
            expected_doc_ids=list(getattr(row, "expected_doc_ids", []) or []),
            cap_policy=cap_policy,
        )
        qt = getattr(row, "query_type", None)
        if not qt and query_type_by_id is not None:
            qt = query_type_by_id.get(str(rid))
        audit_rows.append({
            "policy": spec.label,
            "query_id": str(rid),
            "query": str(getattr(row, "query", "")),
            "query_type": qt,
            "expected_doc_ids": list(getattr(row, "expected_doc_ids", []) or []),
            "retrieved_top5_doc_ids": list(
                getattr(row, "retrieved_doc_ids", []) or []
            )[:5],
            **scored,
        })

    return PolicyRun(
        policy=spec.label,
        description=spec.description,
        spec=spec,
        summary=summary,
        rows=rows,
        audit_rows=audit_rows,
        started_at=started_at,
        finished_at=finished_at,
    )


# ---------------------------------------------------------------------------
# Bucket metrics — character_relation / plot_event slices.
# ---------------------------------------------------------------------------


def _load_query_types(path: Path) -> Tuple[Dict[str, str], Dict[str, float]]:
    qt: Dict[str, str] = {}
    conf: Dict[str, float] = {}
    if not path.exists():
        return qt, conf
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(obj.get("id") or "").strip()
            if not qid:
                continue
            qt[qid] = str(obj.get("query_type") or "unknown")
            try:
                conf[qid] = float(obj.get("query_type_confidence") or 0.0)
            except (TypeError, ValueError):
                conf[qid] = 0.0
    return qt, conf


def _bucket_metrics(
    runs: List[PolicyRun], *, query_type_path: Path,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    qt_by_id, _conf = _load_query_types(query_type_path)
    if not qt_by_id:
        return {}
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run in runs:
        per_bucket: Dict[str, Dict[str, Any]] = {}
        for row in run.rows:
            rid = getattr(row, "id", None)
            if not rid:
                continue
            bucket = qt_by_id.get(str(rid))
            if not bucket:
                continue
            h5 = getattr(row, "hit_at_5", None)
            mrr = getattr(row, "mrr_at_10", None)
            if h5 is None or mrr is None:
                continue
            entry = per_bucket.setdefault(bucket, {
                "count": 0, "h5_sum": 0.0, "mrr_sum": 0.0,
            })
            entry["count"] += 1
            entry["h5_sum"] += float(h5)
            entry["mrr_sum"] += float(mrr)
        bucket_payload: Dict[str, Dict[str, Any]] = {}
        for bucket, entry in per_bucket.items():
            n = max(1, entry["count"])
            bucket_payload[bucket] = {
                "count": entry["count"],
                "mean_hit_at_5": round(entry["h5_sum"] / n, 4),
                "mean_mrr_at_10": round(entry["mrr_sum"] / n, 4),
            }
        out[run.policy] = bucket_payload
    return out


def _audit_summary(audit_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate gold_was_capped_out / gold_chunk_not_in_rerank_input_pool counts."""
    capped_out = 0
    not_in_rerank_input = 0
    not_in_dense_pool = 0
    not_in_mmr_pool = 0
    char_relation_capped_out = 0
    plot_event_capped_out = 0
    capped_out_by_query_type: Dict[str, int] = {}
    for row in audit_rows:
        if row.get("gold_was_capped_out"):
            capped_out += 1
            qt = str(row.get("query_type") or "unknown")
            capped_out_by_query_type[qt] = (
                capped_out_by_query_type.get(qt, 0) + 1
            )
            if qt == _BUCKET_CHARACTER_RELATION:
                char_relation_capped_out += 1
            elif qt == _BUCKET_PLOT_EVENT:
                plot_event_capped_out += 1
        if not row.get("gold_was_in_rerank_input"):
            not_in_rerank_input += 1
        if not row.get("gold_was_in_dense_pool"):
            not_in_dense_pool += 1
        if not row.get("gold_was_in_mmr_pool"):
            not_in_mmr_pool += 1
    return {
        "total_audit_rows": len(audit_rows),
        "gold_was_capped_out_count": capped_out,
        "gold_chunk_not_in_rerank_input_pool_count": not_in_rerank_input,
        "gold_not_in_dense_pool_count": not_in_dense_pool,
        "gold_not_in_mmr_pool_count": not_in_mmr_pool,
        "character_relation_capped_out_count": char_relation_capped_out,
        "plot_event_capped_out_count": plot_event_capped_out,
        "capped_out_by_query_type": capped_out_by_query_type,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_summary_csv(
    out_dir: Path,
    runs: List[PolicyRun],
    deltas_by_policy: Dict[str, Any],
    audit_summary_by_policy: Dict[str, Dict[str, Any]],
    bucket_metrics_by_policy: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    headers = [
        "policy", "cap",
        "row_count",
        "hit@1", "hit@3", "hit@5",
        "mrr@10", "ndcg@10",
        "candidateHit@10", "candidateHit@20",
        "candidateHit@50", "candidateHit@100",
        "duplicateDocRatio@5", "duplicateDocRatio@10",
        "uniqueDocCount@10",
        "avgTotalRetrievalMs",
        "p50ms", "p95ms", "p99ms",
        "avgDenseRetrievalMs", "avgRerankMs",
        "rerankUpliftHit@5", "rerankUpliftMrr@10",
        "qualityScore", "efficiencyScore",
        "grade",
        "Δhit@5_vs_anchor", "Δmrr@10_vs_anchor",
        "Δcand@50_vs_anchor", "latencyRatioP95_vs_anchor",
        "gold_was_capped_out", "gold_not_in_rerank_input",
        "gold_not_in_dense_pool",
        "char_relation_hit@5", "char_relation_mrr@10",
        "plot_event_hit@5", "plot_event_mrr@10",
    ]
    with (out_dir / "summary.csv").open(
        "w", encoding="utf-8", newline="",
    ) as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for run in runs:
            s = run.summary
            cand = s.candidate_hit_rates or {}
            dup = s.duplicate_doc_ratios or {}
            udc = s.unique_doc_counts or {}
            d = deltas_by_policy.get(run.policy)
            audit = audit_summary_by_policy.get(run.policy, {})
            buckets = bucket_metrics_by_policy.get(run.policy, {})
            cr = buckets.get(_BUCKET_CHARACTER_RELATION, {}) or {}
            pe = buckets.get(_BUCKET_PLOT_EVENT, {}) or {}
            writer.writerow([
                run.policy, run.spec.cap,
                s.row_count,
                _f(s.mean_hit_at_1), _f(s.mean_hit_at_3),
                _f(s.mean_hit_at_5),
                _f(s.mean_mrr_at_10), _f(s.mean_ndcg_at_10),
                _f(cand.get("10")), _f(cand.get("20")),
                _f(cand.get("50")), _f(cand.get("100")),
                _f(dup.get("5")), _f(dup.get("10")),
                _f(udc.get("10")),
                _f(s.avg_total_retrieval_ms or s.mean_retrieval_ms),
                _f(s.p50_retrieval_ms),
                _f(s.p95_total_retrieval_ms or s.p95_retrieval_ms),
                _f(s.p99_retrieval_ms),
                _f(s.mean_dense_retrieval_ms),
                _f(s.mean_rerank_ms),
                _f(s.rerank_uplift_hit_at_5),
                _f(s.rerank_uplift_mrr_at_10),
                _f(s.quality_score),
                _f(s.efficiency_score),
                "" if d is None else d.grade,
                _f(None if d is None else d.delta_hit_at_5),
                _f(None if d is None else d.delta_mrr_at_10),
                _f(None if d is None else d.delta_candidate_hit_at_50),
                _f(None if d is None else d.latency_ratio_p95),
                audit.get("gold_was_capped_out_count", ""),
                audit.get("gold_chunk_not_in_rerank_input_pool_count", ""),
                audit.get("gold_not_in_dense_pool_count", ""),
                _f(cr.get("mean_hit_at_5")),
                _f(cr.get("mean_mrr_at_10")),
                _f(pe.get("mean_hit_at_5")),
                _f(pe.get("mean_mrr_at_10")),
            ])


def _write_summary_json(
    out_dir: Path,
    *,
    runs: List[PolicyRun],
    deltas_by_policy: Dict[str, Any],
    audit_summary_by_policy: Dict[str, Dict[str, Any]],
    bucket_metrics_by_policy: Dict[str, Dict[str, Dict[str, Any]]],
    stack: VariantStack,
    args: argparse.Namespace,
    settings: Any,
    verdict: str,
    rationale: str,
) -> None:
    payload = {
        "schema": "phase2-rerank-input-cap-policy-confirm.v1",
        "run": {
            "dataset": str(args.dataset),
            "corpus_path": str(args.corpus),
            "embedding_model": settings.rag_embedding_model,
            "reranker_model": str(args.reranker_model),
            "anchor": "title_cap_1",
            "held_constants": {
                "variant": _VARIANT,
                "reranker_input_format": _RERANKER_INPUT_FORMAT,
                "candidate_k": _CANDIDATE_K,
                "final_top_k": _FINAL_TOP_K,
                "rerank_in": _RERANK_IN,
                "use_mmr": _USE_MMR,
                "mmr_lambda": _MMR_LAMBDA,
                "mmr_k": _MMR_K,
                "title_cap_final": _TITLE_CAP_FINAL,
            },
            "policies": [r.policy for r in runs],
            "started_at": runs[0].started_at if runs else None,
            "finished_at": runs[-1].finished_at if runs else None,
        },
        "verdict": {"label": verdict, "rationale": rationale},
        "stack": {
            "variant": _VARIANT,
            "cache_dir": str(stack.cache_dir),
            "embedding_model": getattr(stack.info, "embedding_model", None),
            "index_version": getattr(stack.info, "index_version", None),
            "chunk_count": getattr(stack.info, "chunk_count", None),
            "document_count": getattr(stack.info, "document_count", None),
        },
        "runs": [
            {
                "policy": r.policy,
                "description": r.description,
                "spec": asdict(r.spec),
                "summary": asdict(r.summary),
                "deltas": (
                    None
                    if r.policy not in deltas_by_policy
                    else asdict(deltas_by_policy[r.policy])
                ),
                "audit_summary": audit_summary_by_policy.get(r.policy, {}),
                "bucket_metrics": bucket_metrics_by_policy.get(r.policy, {}),
                "started_at": r.started_at,
                "finished_at": r.finished_at,
            }
            for r in runs
        ],
        "caveats": [
            "Production code (app/) is not modified — every policy "
            "swaps the cap rule via the eval-only CapPolicyAuditAdapter "
            "over the existing title_section FAISS cache.",
            "The reranker input format is held at title_plus_chunk for "
            "every policy; the only independent variable is the cap "
            "policy on the rerank-input slice.",
            "FAISS indexes are reused as-is; cand@K rates differ by less "
            "than EPS_CANDIDATE across policies — non-zero deltas signal "
            "MMR diversity drift, not bi-encoder pool drift.",
            "p95/p99 latency is sensitive to GPU thermal state on a 200-"
            "row dataset; treat single-digit-percent latency deltas as "
            "noise.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_per_query_jsonl(out_dir: Path, runs: List[PolicyRun]) -> None:
    from eval.harness.retrieval_eval import row_to_dict

    with (out_dir / "per_query_results.jsonl").open(
        "w", encoding="utf-8",
    ) as fp:
        for run in runs:
            for row in run.rows:
                payload = row_to_dict(row)
                payload["policy"] = run.policy
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_per_query_diffs(
    out_dir: Path,
    diffs_by_policy: Dict[str, Tuple[List[Any], List[Any]]],
) -> None:
    with (out_dir / "per_query_diffs.jsonl").open(
        "w", encoding="utf-8",
    ) as fp:
        for policy, (improved, regressed) in diffs_by_policy.items():
            for entry in improved:
                payload = {
                    "policy": policy,
                    "direction": "improved",
                    **asdict(entry),
                }
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            for entry in regressed:
                payload = {
                    "policy": policy,
                    "direction": "regressed",
                    **asdict(entry),
                }
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_cap_audit_jsonl(
    out_dir: Path, runs: List[PolicyRun],
) -> None:
    with (out_dir / "cap_audit.jsonl").open(
        "w", encoding="utf-8",
    ) as fp:
        for run in runs:
            for entry in run.audit_rows:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_config_dump(
    out_dir: Path,
    *,
    runs: List[PolicyRun],
    stack: VariantStack,
    args: argparse.Namespace,
) -> None:
    payload = {
        "schema": "phase2-rerank-input-cap-policy-confirm.config.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "dataset": str(args.dataset),
            "corpus": str(args.corpus),
            "out_dir": str(args.out_dir) if args.out_dir else None,
            "limit": args.limit,
            "cache_root": str(args.cache_root),
            "title_section_cache_dir": (
                str(args.title_section_cache_dir_arg)
                if args.title_section_cache_dir_arg else None
            ),
            "max_seq_length": args.max_seq_length,
            "embed_batch_size": args.embed_batch_size,
            "reranker_model": args.reranker_model,
            "reranker_max_length": args.reranker_max_length,
            "reranker_batch_size": args.reranker_batch_size,
            "reranker_text_max_chars": args.reranker_text_max_chars,
        },
        "anchor": "title_cap_1",
        "held_constants": {
            "variant": _VARIANT,
            "reranker_input_format": _RERANKER_INPUT_FORMAT,
            "candidate_k": _CANDIDATE_K,
            "final_top_k": _FINAL_TOP_K,
            "rerank_in": _RERANK_IN,
            "use_mmr": _USE_MMR,
            "mmr_lambda": _MMR_LAMBDA,
            "mmr_k": _MMR_K,
            "title_cap_final": _TITLE_CAP_FINAL,
        },
        "policies": [asdict(r.spec) for r in runs],
        "stack": {
            "variant": _VARIANT,
            "cache_dir": str(stack.cache_dir),
        },
    }
    (out_dir / "config_dump.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_index_manifest(out_dir: Path, stack: VariantStack) -> None:
    payload = {
        "schema": "phase2-rerank-input-cap-policy-confirm.index-manifest.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "variant": _VARIANT,
        "cache_dir": str(stack.cache_dir),
        "embedding_model": getattr(stack.info, "embedding_model", None),
        "index_version": getattr(stack.info, "index_version", None),
        "chunk_count": getattr(stack.info, "chunk_count", None),
        "document_count": getattr(stack.info, "document_count", None),
        "dimension": getattr(stack.info, "dimension", None),
        "manifest": (
            None if stack.manifest is None
            else {
                "variant": stack.manifest.variant,
                "variant_slug": stack.manifest.variant_slug,
                "max_seq_length": stack.manifest.max_seq_length,
                "embed_text_sha256": stack.manifest.embed_text_sha256,
                "embed_text_samples": list(
                    stack.manifest.embed_text_samples or []
                ),
            }
        ),
    }
    (out_dir / "index_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_regression_guard(
    out_dir: Path,
    runs: List[PolicyRun],
    deltas_by_policy: Dict[str, Any],
) -> None:
    from eval.harness.confirm_wide_mmr_helpers import (
        EPS_HIT, EPS_MRR, GRADE_REGRESSION,
    )
    from eval.harness.variant_comparison import EPS_CANDIDATE

    md: List[str] = []
    md.append("# Regression guard — rerank-input cap policy confirm sweep")
    md.append("")
    md.append(
        "Anchor: policy=`title_cap_1`. Each policy is checked against the "
        "anchor on the held-constant cell (title_section index × "
        "title_plus_chunk reranker input). A regression fires when "
        "``Δhit@5 ≤ -0.005`` OR ``Δmrr@10 ≤ -0.005`` OR "
        "``Δcand@50 ≤ -0.005``."
    )
    md.append("")
    md.append(
        f"Epsilon contract: `EPS_HIT={EPS_HIT}`, `EPS_MRR={EPS_MRR}`, "
        f"`EPS_CANDIDATE={EPS_CANDIDATE}`."
    )
    md.append("")
    md.append(
        "| policy | grade | Δhit@5 | Δmrr@10 | Δcand@50 | latRatioP95 | passes |"
    )
    md.append("|---|---|---:|---:|---:|---:|---|")
    failures = 0
    for run in runs:
        d = deltas_by_policy.get(run.policy)
        if d is None:
            continue
        passes = "OK" if d.grade != GRADE_REGRESSION else "FAIL"
        if d.grade == GRADE_REGRESSION:
            failures += 1
        md.append(
            f"| {run.policy} | {d.grade} | "
            f"{_fmt_signed(d.delta_hit_at_5)} | "
            f"{_fmt_signed(d.delta_mrr_at_10)} | "
            f"{_fmt_signed(d.delta_candidate_hit_at_50)} | "
            f"{_fmt(d.latency_ratio_p95)} | {passes} |"
        )
    md.append("")
    if failures == 0:
        md.append(
            "**Result: PASS** — no policy regresses against the anchor "
            "beyond epsilon."
        )
    else:
        md.append(
            f"**Result: {failures} regressing policy / policies flagged.** "
            "Review individual ``Δ*`` columns and the comparison report "
            "before adopting any policy."
        )
    md.append("")
    (out_dir / "regression_guard.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


def _write_bucket_metrics(
    out_dir: Path,
    bucket_metrics_by_policy: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    (out_dir / "bucket_metrics.json").write_text(
        json.dumps(
            {
                "schema": "phase2-rerank-input-cap-policy-confirm.bucket.v1",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "buckets_per_policy": bucket_metrics_by_policy,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Comparison report (markdown narrative + verdict)
# ---------------------------------------------------------------------------


def _write_comparison_report(
    out_dir: Path,
    *,
    runs: List[PolicyRun],
    deltas_by_policy: Dict[str, Any],
    diffs_by_policy: Dict[str, Tuple[List[Any], List[Any]]],
    audit_summary_by_policy: Dict[str, Dict[str, Any]],
    bucket_metrics_by_policy: Dict[str, Dict[str, Dict[str, Any]]],
    stack: VariantStack,
    miss_summary: Dict[str, Dict[str, int]],
    sample_audit_rows: List[Dict[str, Any]],
    verdict: str,
    rationale: str,
    args: argparse.Namespace,
) -> None:
    md: List[str] = []
    md.append(
        "# Rerank-input cap policy confirm sweep "
        "— title cap=1 vs cap=2/3/no_cap/doc_id/section_path"
    )
    md.append("")
    md.append(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    md.append("- anchor: policy=`title_cap_1`")
    md.append(
        f"- held constant: variant=`{_VARIANT}`, format="
        f"`{_RERANKER_INPUT_FORMAT}`, candidate_k={_CANDIDATE_K}, "
        f"final_top_k={_FINAL_TOP_K}, rerank_in={_RERANK_IN}, "
        f"use_mmr={_USE_MMR}, mmr_lambda={_MMR_LAMBDA}, mmr_k={_MMR_K}, "
        f"title_cap_final={_TITLE_CAP_FINAL}"
    )
    md.append(f"- corpus: {args.corpus}")
    md.append(f"- dataset: {args.dataset}")
    md.append("")

    # 1. Headline metrics ---------------------------------------------------
    md.append("## Headline metrics")
    md.append("")
    md.append(
        "| policy | cap | hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 | "
        "cand@50 | cand@100 | dup@10 | uniq@10 | p50ms | p95ms | p99ms |"
    )
    md.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for run in runs:
        s = run.summary
        cand = s.candidate_hit_rates or {}
        dup = s.duplicate_doc_ratios or {}
        udc = s.unique_doc_counts or {}
        md.append(
            f"| {run.policy} | {run.spec.cap if run.spec.cap is not None else '-'} | "
            f"{_fmt(s.mean_hit_at_1)} | {_fmt(s.mean_hit_at_3)} | "
            f"{_fmt(s.mean_hit_at_5)} | {_fmt(s.mean_mrr_at_10)} | "
            f"{_fmt(s.mean_ndcg_at_10)} | "
            f"{_fmt(cand.get('50'))} | {_fmt(cand.get('100'))} | "
            f"{_fmt(dup.get('10'))} | {_fmt(udc.get('10'))} | "
            f"{_fmt_ms(s.p50_retrieval_ms)} | "
            f"{_fmt_ms(s.p95_total_retrieval_ms or s.p95_retrieval_ms)} | "
            f"{_fmt_ms(s.p99_retrieval_ms)} |"
        )
    md.append("")

    # 2. Deltas vs anchor ---------------------------------------------------
    md.append("## Deltas vs anchor (`title_cap_1`)")
    md.append("")
    md.append(
        "| policy | grade | Δhit@1 | Δhit@3 | Δhit@5 | Δmrr@10 | "
        "Δndcg@10 | Δcand@50 | Δcand@100 | Δdup@10 | Δuniq@10 | "
        "latRatioP95 | reason |"
    )
    md.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for run in runs:
        d = deltas_by_policy.get(run.policy)
        if d is None:
            continue
        md.append(
            f"| {run.policy} | {d.grade} | "
            f"{_fmt_signed(d.delta_hit_at_1)} | "
            f"{_fmt_signed(d.delta_hit_at_3)} | "
            f"{_fmt_signed(d.delta_hit_at_5)} | "
            f"{_fmt_signed(d.delta_mrr_at_10)} | "
            f"{_fmt_signed(d.delta_ndcg_at_10)} | "
            f"{_fmt_signed(d.delta_candidate_hit_at_50)} | "
            f"{_fmt_signed(d.delta_candidate_hit_at_100)} | "
            f"{_fmt_signed(d.delta_duplicate_ratio_at_10)} | "
            f"{_fmt_signed(d.delta_unique_doc_count_at_10)} | "
            f"{_fmt(d.latency_ratio_p95)} | {d.reason} |"
        )
    md.append("")

    # 3. Cap audit summary --------------------------------------------------
    md.append("## Cap audit — gold-rank trace")
    md.append("")
    md.append(
        "Counts come from `cap_audit.jsonl`. ``gold_was_capped_out`` "
        "fires when the gold doc was in the MMR pool but the cap "
        "policy dropped it before the rerank-input slice. "
        "``gold_chunk_not_in_rerank_input_pool`` is the broader "
        "miss — gold not present in the slice for any reason "
        "(dropped by MMR, dropped by cap, or never in pool)."
    )
    md.append("")
    md.append(
        "| policy | gold_was_capped_out | gold_not_in_rerank_input | "
        "gold_not_in_dense_pool | char_relation_capped_out | "
        "plot_event_capped_out |"
    )
    md.append("|---|---:|---:|---:|---:|---:|")
    for run in runs:
        a = audit_summary_by_policy.get(run.policy, {})
        md.append(
            f"| {run.policy} | "
            f"{a.get('gold_was_capped_out_count', '-')} | "
            f"{a.get('gold_chunk_not_in_rerank_input_pool_count', '-')} | "
            f"{a.get('gold_not_in_dense_pool_count', '-')} | "
            f"{a.get('character_relation_capped_out_count', '-')} | "
            f"{a.get('plot_event_capped_out_count', '-')} |"
        )
    md.append("")

    # 4. Character_relation / plot_event bucket -----------------------------
    md.append("## byQueryType breakdown — character_relation / plot_event")
    md.append("")
    md.append(
        "Joined with ``anime_silver_200.query_type_draft.jsonl`` (auto-"
        "tagged, **not manually reviewed**). Treat the bucket numbers "
        "as directional. The `character_relation` bucket is the spec's "
        "stress test: prior plateau ≈ 0.38–0.40 hit@5 even after the "
        "best reranker format was adopted."
    )
    md.append("")
    md.append(
        "| policy | char_relation hit@5 | char_relation mrr@10 | "
        "plot_event hit@5 | plot_event mrr@10 |"
    )
    md.append("|---|---:|---:|---:|---:|")
    for run in runs:
        buckets = bucket_metrics_by_policy.get(run.policy, {})
        cr = buckets.get(_BUCKET_CHARACTER_RELATION, {}) or {}
        pe = buckets.get(_BUCKET_PLOT_EVENT, {}) or {}
        md.append(
            f"| {run.policy} | "
            f"{_fmt(cr.get('mean_hit_at_5'))} | "
            f"{_fmt(cr.get('mean_mrr_at_10'))} | "
            f"{_fmt(pe.get('mean_hit_at_5'))} | "
            f"{_fmt(pe.get('mean_mrr_at_10'))} |"
        )
    md.append("")

    # 5. Miss counters ------------------------------------------------------
    md.append("## Miss counters vs anchor")
    md.append("")
    md.append(
        "*unrecoverable*: gold doc never appears in the candidate "
        "pool (pure dense miss). *recoverable*: gold is in the pool "
        "but the reranker dropped it from the final top-5. The cap-"
        "policy axis should chip away at the *recoverable* counter; "
        "if it doesn't, the bottleneck is reranker model capacity or "
        "dataset schema, not cap tuning."
    )
    md.append("")
    md.append(
        "| policy | unrecoverable | recoverable | "
        "Δunrecoverable_vs_anchor | Δrecoverable_vs_anchor |"
    )
    md.append("|---|---:|---:|---:|---:|")
    for run in runs:
        ms = miss_summary.get(run.policy, {})
        md.append(
            f"| {run.policy} | "
            f"{ms.get('unrecoverable', 'n/a')} | "
            f"{ms.get('recoverable', 'n/a')} | "
            f"{_fmt_signed(ms.get('unrecoverable_delta'))} | "
            f"{_fmt_signed(ms.get('recoverable_delta'))} |"
        )
    md.append("")

    # 6. Per-query diffs vs anchor -----------------------------------------
    md.append("## Per-query diffs vs anchor")
    md.append("")
    md.append(
        "Lists below enumerate query IDs whose hit@5 flipped between "
        "the anchor (`title_cap_1`) and the named policy. ``improved``: "
        "anchor=miss → policy=hit; ``regressed``: anchor=hit → policy=miss."
    )
    md.append("")
    for policy, (improved, regressed) in diffs_by_policy.items():
        if policy == "title_cap_1":
            continue
        md.append(f"### `{policy}`")
        md.append("")
        md.append(f"- improved vs anchor: **{len(improved)}** queries")
        md.append(f"- regressed vs anchor: **{len(regressed)}** queries")
        if improved:
            md.append("")
            md.append("Improved query IDs (up to 10): " + ", ".join(
                e.id for e in improved[:10]
            ))
        if regressed:
            md.append("")
            md.append("Regressed query IDs (up to 10): " + ", ".join(
                e.id for e in regressed[:10]
            ))
        md.append("")

    # 7. Cap audit samples --------------------------------------------------
    md.append("## Cap audit samples (gold-was-capped-out)")
    md.append("")
    md.append(
        "Up to 30 sample rows from ``cap_audit.jsonl`` where "
        "``gold_was_capped_out=True``. Useful for spotting which "
        "(query, gold_doc) pairs the cap policy is dropping."
    )
    md.append("")
    md.append(
        "| policy | query_id | query_type | gold_doc | "
        "capped_out_group_key | group_size | dense_rank | mmr_rank | "
        "rerank_input_rank | final_rank |"
    )
    md.append(
        "|---|---|---|---|---|---:|---:|---:|---:|---:|"
    )
    for sample in sample_audit_rows[:30]:
        gold_doc_str = ", ".join(
            (sample.get("expected_doc_ids") or [])[:1]
        )[:18]
        group_key = (sample.get("capped_out_by_group_key") or "")[:30]
        md.append(
            f"| {sample.get('policy', '')} | "
            f"`{sample.get('query_id', '')}` | "
            f"{sample.get('query_type', '') or '-'} | "
            f"`{gold_doc_str}` | "
            f"`{group_key}` | "
            f"{sample.get('capped_out_group_size', '-')} | "
            f"{sample.get('gold_dense_rank', '-')} | "
            f"{sample.get('gold_after_mmr_rank', '-')} | "
            f"{sample.get('gold_rerank_input_rank', '-')} | "
            f"{sample.get('gold_final_rank', '-')} |"
        )
    md.append("")

    # 8. Verdict + next-step ------------------------------------------------
    md.append("## Verdict")
    md.append("")
    md.append(f"**{verdict}** — {rationale}")
    md.append("")
    md.append("## Next-step recommendation")
    md.append("")
    if verdict == "ADOPT_TITLE_CAP_RERANK_INPUT_2":
        md.append(
            "1. Adopt ``title_cap_rerank_input=2`` as the production "
            "cap. Update the production retriever's title cap (or the "
            "wide-MMR config that wraps it) to ``cap=2`` at the rerank-"
            "input boundary while keeping the final cap at 2."
        )
        md.append(
            "2. Re-run the full retrieval eval (silver_200 + a smoke "
            "battery) with the new cap to confirm no latency regression."
        )
    elif verdict == "ADOPT_TITLE_CAP_RERANK_INPUT_3":
        md.append(
            "1. Adopt ``title_cap_rerank_input=3`` — the loosest title "
            "cap that still beats the anchor. Watch dup@10 / latency."
        )
    elif verdict == "ADOPT_DOC_ID_LEVEL_CAP":
        md.append(
            "1. Switch the rerank-input cap from title-level to doc_id-"
            "level grouping. The eval-only ``DocIdCapPolicy`` is the "
            "reference; mirror it in the production retriever."
        )
        md.append(
            "2. The production helper currently caps by title; adding a "
            "``cap_grouping=doc_id`` knob to the wide-MMR config is the "
            "minimal change."
        )
    elif verdict == "ADOPT_NO_CAP_RERANK_INPUT":
        md.append(
            "1. Drop the rerank-input cap entirely. Retain the final "
            "cap (cap=2) so output diversity stays bounded."
        )
        md.append(
            "2. Audit dup@10 in production traffic — the no-cap pool "
            "feeds the cross-encoder more near-duplicates per query, "
            "which can shift the latency profile in ways the 200-row "
            "silver dataset doesn't surface."
        )
    elif verdict == "NEED_SCHEMA_ENRICHMENT":
        md.append(
            "1. Cap policy is **not** the bottleneck. The "
            "``character_relation`` bucket stays below 0.45 hit@5 "
            "regardless of cap relaxation, and ``gold_was_capped_out`` "
            "doesn't drop materially across policies — the dataset is "
            "missing the disambiguation signals (work_title, "
            "entity_name, entity_type, section_path, source_doc_id) "
            "the reranker would need."
        )
        md.append(
            "2. Next axis: dataset schema enrichment. Add per-chunk "
            "entity / character / arc tags so the reranker can "
            "disambiguate same-franchise / same-character queries from "
            "structural metadata, not just chunk text."
        )
    else:  # KEEP_TITLE_CAP_RERANK_INPUT_1
        md.append(
            "1. Keep ``title_cap_rerank_input=1``. No alternative "
            "cap policy clears EPS on hit@5 / MRR over the anchor."
        )
        md.append(
            "2. Two next directions: (a) reranker model headroom — "
            "``bge-reranker-v2-m3`` may saturate on this dataset; try "
            "a larger reranker. (b) chunking / section-level dedup "
            "before the dense pool, to reduce the reranker's same-"
            "title load without touching the cap."
        )
    md.append("")

    # 9. Caveats -----------------------------------------------------------
    md.append("## Caveats")
    md.append("")
    md.append(
        "- Production code (``app/``) is not modified. Each policy "
        "runs through ``CapPolicyAuditAdapter`` over the existing "
        f"FAISS cache for variant=`{_VARIANT}`."
    )
    md.append(
        "- FAISS indexes are reused as-is; the cap policy is post-"
        "MMR / pre-rerank, so candidate@K rates are constant per "
        "policy modulo MMR diversity drift."
    )
    md.append(
        "- ``cap_audit.jsonl`` reports gold ranks at every stage. "
        "``gold_was_capped_out`` requires gold to be in the MMR pool "
        "but missing from the after-cap rerank-input pool; a chunk "
        "that was already MMR-rejected counts as ``gold_not_in_mmr_"
        "pool``, not capped-out."
    )
    md.append(
        "- The ``query_type_draft`` join is heuristic — diagnostic "
        "only. Manual review of bucket assignments is open work."
    )
    md.append(
        "- p95/p99 latency on 200 rows is sensitive to GPU thermal "
        "state; treat single-digit-percent latency deltas as noise. "
        "Cap relaxation should not change the latency profile in "
        "isolation — cap work happens in pure Python over <300 chunks."
    )
    md.append("")

    (out_dir / "comparison_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument(
        "--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-text-max-chars", type=int, default=800)
    parser.add_argument("--reranker-device", type=str, default=None)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on the dataset row count (smoke runs).",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=_DEFAULT_CACHE_ROOT,
        help="Root directory for variant FAISS caches.",
    )
    parser.add_argument(
        "--title-section-cache-dir", dest="title_section_cache_dir_arg",
        type=Path, default=None,
    )
    parser.add_argument(
        "--query-type-draft", type=Path, default=_DEFAULT_QUERY_TYPE_DRAFT,
    )
    parser.add_argument(
        "--audit-sample-limit", type=int, default=30,
        help="Per-policy gold_was_capped_out sample cap for the "
             "comparison_report.md table.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    if out_dir.exists():
        log.error(
            "Refusing to overwrite existing out-dir %s — pick a new "
            "path or wait a minute for the timestamp to roll.",
            out_dir,
        )
        return 2
    out_dir.mkdir(parents=True, exist_ok=False)
    log.info("Output dir: %s", out_dir)

    from eval.harness.cap_policy_comparison import (
        ANCHOR_POLICY,
        compute_cap_policy_deltas,
        decide_cap_policy_verdict,
    )
    from eval.harness.io_utils import load_jsonl
    from eval.harness.variant_comparison import (
        candidate_pool_recoverable_miss_count,
        candidate_pool_unrecoverable_miss_count,
        variant_per_query_diff,
    )
    from eval.harness.wide_retrieval_helpers import DocTitleResolver

    title_resolver = DocTitleResolver.from_corpus(args.corpus)
    title_provider = title_resolver.title_provider()

    dataset = list(load_jsonl(args.dataset))
    if args.limit is not None and args.limit > 0:
        dataset = dataset[: int(args.limit)]
    log.info(
        "Loaded %d query rows (limit=%s) from %s",
        len(dataset), args.limit, args.dataset,
    )

    stack = _load_variant_stack(args=args)
    log.info(
        "[variant=%s] cache=%s chunks=%s dim=%s",
        _VARIANT, stack.cache_dir,
        getattr(stack.info, "chunk_count", None),
        getattr(stack.info, "dimension", None),
    )

    qt_by_id, _ = _load_query_types(Path(args.query_type_draft))

    runs: List[PolicyRun] = []
    for spec in default_policy_specs():
        run = _run_policy(
            spec=spec,
            stack=stack,
            dataset=dataset,
            args=args,
            title_provider=title_provider,
            query_type_by_id=qt_by_id,
        )
        runs.append(run)

    runs_by_policy: Dict[str, PolicyRun] = {r.policy: r for r in runs}
    anchor_run = runs_by_policy.get(ANCHOR_POLICY)
    if anchor_run is None:
        log.error("Anchor policy %s missing — cannot compute deltas.", ANCHOR_POLICY)
        return 4

    deltas_by_policy: Dict[str, Any] = {}
    for run in runs:
        deltas_by_policy[run.policy] = compute_cap_policy_deltas(
            policy_label=run.policy,
            policy_summary=run.summary,
            anchor_summary=anchor_run.summary,
        )

    diffs_by_policy: Dict[str, Tuple[List[Any], List[Any]]] = {}
    from eval.harness.retrieval_eval import row_to_dict
    anchor_rows_dict = [row_to_dict(r) for r in anchor_run.rows]
    for run in runs:
        if run.policy == ANCHOR_POLICY:
            continue
        improved, regressed = variant_per_query_diff(
            cell_label="rerank_input_cap_policy",
            variant=run.policy,
            raw_rows=anchor_rows_dict,
            variant_rows=[row_to_dict(r) for r in run.rows],
        )
        diffs_by_policy[run.policy] = (improved, regressed)

    miss_summary: Dict[str, Dict[str, int]] = {}
    anchor_unrec = candidate_pool_unrecoverable_miss_count(anchor_rows_dict)
    anchor_rec = candidate_pool_recoverable_miss_count(anchor_rows_dict)
    for run in runs:
        rows_dict = [row_to_dict(r) for r in run.rows]
        unrec = candidate_pool_unrecoverable_miss_count(rows_dict)
        rec = candidate_pool_recoverable_miss_count(rows_dict)
        miss_summary[run.policy] = {
            "unrecoverable": unrec,
            "recoverable": rec,
            "unrecoverable_delta": unrec - anchor_unrec,
            "recoverable_delta": rec - anchor_rec,
        }

    bucket_metrics_by_policy = _bucket_metrics(
        runs, query_type_path=Path(args.query_type_draft),
    )
    audit_summary_by_policy: Dict[str, Dict[str, Any]] = {
        run.policy: _audit_summary(run.audit_rows) for run in runs
    }

    verdict, rationale = decide_cap_policy_verdict(
        deltas_by_policy=deltas_by_policy,
        bucket_metrics_by_policy=bucket_metrics_by_policy,
        audit_summary_by_policy=audit_summary_by_policy,
    )
    log.info("Verdict: %s — %s", verdict, rationale)

    sample_audit_rows: List[Dict[str, Any]] = []
    for run in runs:
        capped = [
            row for row in run.audit_rows if row.get("gold_was_capped_out")
        ][: int(args.audit_sample_limit)]
        sample_audit_rows.extend(capped)

    from app.core.config import get_settings
    settings = get_settings()

    _write_summary_csv(
        out_dir, runs, deltas_by_policy,
        audit_summary_by_policy, bucket_metrics_by_policy,
    )
    _write_summary_json(
        out_dir,
        runs=runs,
        deltas_by_policy=deltas_by_policy,
        audit_summary_by_policy=audit_summary_by_policy,
        bucket_metrics_by_policy=bucket_metrics_by_policy,
        stack=stack,
        args=args,
        settings=settings,
        verdict=verdict,
        rationale=rationale,
    )
    _write_per_query_jsonl(out_dir, runs)
    _write_per_query_diffs(out_dir, diffs_by_policy)
    _write_cap_audit_jsonl(out_dir, runs)
    _write_config_dump(out_dir, runs=runs, stack=stack, args=args)
    _write_index_manifest(out_dir, stack)
    _write_regression_guard(out_dir, runs, deltas_by_policy)
    _write_bucket_metrics(out_dir, bucket_metrics_by_policy)
    _write_comparison_report(
        out_dir,
        runs=runs,
        deltas_by_policy=deltas_by_policy,
        diffs_by_policy=diffs_by_policy,
        audit_summary_by_policy=audit_summary_by_policy,
        bucket_metrics_by_policy=bucket_metrics_by_policy,
        stack=stack,
        miss_summary=miss_summary,
        sample_audit_rows=sample_audit_rows,
        verdict=verdict,
        rationale=rationale,
        args=args,
    )

    log.info(
        "Cap policy confirm sweep finished — verdict=%s artifacts in %s",
        verdict, out_dir,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
