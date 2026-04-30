"""Phase 7.4 — controlled recovery loop evaluator.

The loop reads Phase 7.3's ``per_query_confidence.jsonl`` (or the
narrower ``recommended_recovery_queries.jsonl``), routes each row
through the Phase 7.4 ``recovery_policy`` to obtain a typed decision,
and — for the ATTEMPT_* decisions — re-runs retrieval to measure
whether gold's rank improves.

The retrieval for the recovery is *eval-only* and *deterministic*:

  - Dense side is **frozen**. We do not have a live FAISS in this
    harness; we treat Phase 7.0's ``candidate.top_results`` as the
    deterministic dense top-N for the original query. This is the same
    contract Phase 7.3 consumed — the dense ordering is replay-safe.
  - Sparse side is **fresh**. We build a BM25 index over the chunks
    JSONL (the same artefact Phase 7.3 used to enrich titles) once
    per run, and run BM25 against either the original query (HYBRID
    path) or the rewritten query (REWRITE path). The BM25 index is
    cheap to build (~50k chunks, in-memory inverted index) and the
    re-run is a single dot-product per term.
  - Fusion is **RRF** with the same constants the Phase 2 hybrid
    retriever uses (``DEFAULT_K_RRF = 60``). RRF on (frozen-dense,
    fresh-sparse) is cheap and produces a deterministic top-K.

Why we do not re-run dense for the rewrite: Phase 7.4's stated goal is
to size the *recovery loop's upper bound* against the existing
retrieval stack — we want to learn how much of the gold-miss tail BM25
+ rewriting can recover when stacked on top of frozen dense. A live
dense re-query would also be informative but is out of scope for the
deterministic-first phase. The harness is structured so swapping in a
live-dense path later is a single retriever swap.

Production code is NOT touched. The only production imports are read
through ``app.capabilities.rag.generation.RetrievedChunk`` (the dense
chunk dataclass we *deserialise into*, already used by every other
harness module), and through whatever the BM25 index reads via
``build_bm25_index``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from app.capabilities.rag.generation import RetrievedChunk

from eval.harness.bm25_retriever import (
    BM25EvalRetriever,
    BM25Index,
    build_bm25_index,
    tokenize_for_bm25,
)
from eval.harness.embedding_text_builder import VARIANT_RAW
from eval.harness.hybrid_retriever import (
    DEFAULT_K_RRF,
    rrf_fuse_ranked_lists,
)
from eval.harness.recovery_policy import (
    RECOVERY_ACTION_ATTEMPT_HYBRID,
    RECOVERY_ACTION_ATTEMPT_REWRITE,
    RECOVERY_ACTION_NOOP,
    RECOVERY_ACTION_SKIP_CAUTION,
    RECOVERY_ACTION_SKIP_DEFER,
    RECOVERY_ACTION_SKIP_REFUSE,
    RECOVERY_ACTIONS,
    REWRITE_MODE_BOTH,
    REWRITE_MODE_ORACLE,
    REWRITE_MODE_PRODUCTION_LIKE,
    REWRITE_MODES,
    LabelLeakageError,
    RecoveryAttempt,
    RecoveryDecision,
    RecoveryResult,
    classify_attempt,
    decide_recovery,
)

log = logging.getLogger(__name__)


_RANK_MISS = -1
_DEFAULT_FINAL_K = 10
_DEFAULT_HYBRID_TOP_K = 10
_DEFAULT_BM25_POOL_SIZE = 100


# ---------------------------------------------------------------------------
# Chunk loader for chunks_jsonl
# ---------------------------------------------------------------------------


def load_chunks_for_bm25(chunks_jsonl: Path) -> List[Any]:
    """Load chunks JSONL into duck-typed objects ``build_bm25_index`` can read.

    The Phase 7.0 export shape (``rag_chunks_*.jsonl``) carries
    ``chunk_id``, ``doc_id``, ``title``, ``retrieval_title``, ``aliases``,
    ``section_path``, ``chunk_text``, ``embedding_text``. We map them
    to the duck-typed ``.text``, ``.title``, ``.section``, ``.keywords``
    surface ``build_bm25_index`` reads, preferring ``embedding_text``
    when present (that is what the dense embedder would have seen).
    Otherwise we fall back to ``chunk_text``.
    """
    out: List[Any] = []
    with Path(chunks_jsonl).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_id = str(rec.get("chunk_id") or "")
            if not chunk_id:
                continue
            text = rec.get("embedding_text") or rec.get("chunk_text") or ""
            title = (
                rec.get("retrieval_title")
                or rec.get("display_title")
                or rec.get("title")
            )
            section_path = rec.get("section_path")
            if isinstance(section_path, list) and section_path:
                section = " > ".join(str(s) for s in section_path)
            else:
                section = ""
            aliases = rec.get("aliases")
            if isinstance(aliases, list):
                keywords = tuple(str(a) for a in aliases if a)
            else:
                keywords = ()
            out.append(SimpleNamespace(
                chunk_id=chunk_id,
                doc_id=str(rec.get("doc_id") or ""),
                section=section,
                text=str(text),
                title=str(title) if title else None,
                keywords=keywords,
            ))
    return out


# ---------------------------------------------------------------------------
# Per-query frozen dense state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenDenseRow:
    """The dense top-N from Phase 7.0 for one query, kept in retriever order.

    Carries the data we need to (a) report the before-rank, (b) feed
    the dense list into the RRF fuse step, and (c) trace the row back
    to its origin if a reviewer wants to look at the exact dense
    ordering Phase 7.4 fused against.
    """

    qid: str
    query_text: str
    expected_doc_ids: Tuple[str, ...]
    top_chunk_ids: Tuple[str, ...]
    top_doc_ids: Tuple[str, ...]
    top1_score: Optional[float]


def load_frozen_dense_state(
    per_query_path: Path,
    *,
    side: str = "candidate",
    final_k: int = _DEFAULT_FINAL_K,
) -> Dict[str, FrozenDenseRow]:
    """Index Phase 7.0's per_query_comparison.jsonl by qid.

    ``side`` is ``baseline`` or ``candidate``; the loop normally runs
    ``candidate`` (Phase 7.0 promoted that variant). The dense top-N
    is sliced at ``final_k`` because we only RRF-fuse against the same
    final_k window the candidate was producing.
    """
    if side not in ("baseline", "candidate"):
        raise ValueError(f"side must be baseline / candidate, got {side!r}.")
    out: Dict[str, FrozenDenseRow] = {}
    with Path(per_query_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("qid") or "")
            if not qid:
                continue
            side_block = rec.get(side) or {}
            top_results = side_block.get("top_results") or []
            top_chunk_ids: List[str] = []
            top_doc_ids: List[str] = []
            top1_score: Optional[float] = None
            for i, row in enumerate(top_results[:final_k]):
                cid = str(row.get("chunk_id") or "")
                did = str(row.get("doc_id") or "")
                top_chunk_ids.append(cid)
                top_doc_ids.append(did)
                if i == 0 and "score" in row and row["score"] is not None:
                    top1_score = float(row["score"])
            expected = tuple(
                str(d) for d in (rec.get("expected_doc_ids") or [])
            )
            out[qid] = FrozenDenseRow(
                qid=qid,
                query_text=str(rec.get("query") or ""),
                expected_doc_ids=expected,
                top_chunk_ids=tuple(top_chunk_ids),
                top_doc_ids=tuple(top_doc_ids),
                top1_score=top1_score,
            )
    return out


# ---------------------------------------------------------------------------
# Rank computation utilities
# ---------------------------------------------------------------------------


def _gold_rank_in(
    chunk_ids: Sequence[str],
    doc_ids: Sequence[str],
    *,
    gold_doc_id: Optional[str],
    gold_page_id: Optional[str],
) -> int:
    """1-indexed rank of the first gold-matching position in two parallel lists.

    Mirrors the v4 confidence detector's ``_gold_rank`` but runs over
    the (chunk_id, doc_id) parallel lists materialised from a hybrid
    fuse — the loop never has chunk objects, only their IDs.
    """
    if not gold_doc_id and not gold_page_id:
        return _RANK_MISS
    for i, (_cid, did) in enumerate(zip(chunk_ids, doc_ids), start=1):
        if gold_doc_id and did == gold_doc_id:
            return i
    return _RANK_MISS


# ---------------------------------------------------------------------------
# Hybrid fuse (frozen-dense × fresh-BM25)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedTopK:
    """The top-K result of fusing frozen dense and fresh BM25."""

    chunk_ids: Tuple[str, ...]
    doc_ids: Tuple[str, ...]
    fused_scores: Tuple[float, ...]
    bm25_top1_score: Optional[float]


def fuse_dense_and_bm25(
    *,
    dense_chunk_ids: Sequence[str],
    dense_doc_ids: Sequence[str],
    bm25_results: Sequence[RetrievedChunk],
    final_k: int = _DEFAULT_FINAL_K,
    k_rrf: int = DEFAULT_K_RRF,
) -> FusedTopK:
    """Run RRF on (frozen dense, fresh BM25) and return top-K.

    Doc IDs are looked up off the BM25 results when a chunk_id is
    contributed only by BM25; the dense list carries chunk_id+doc_id
    pairs so we can't rebuild a doc_id from chunk_id alone.
    """
    chunk_to_doc: Dict[str, str] = {}
    for cid, did in zip(dense_chunk_ids, dense_doc_ids):
        if cid:
            chunk_to_doc[cid] = did
    for c in bm25_results:
        cid = c.chunk_id
        if cid and cid not in chunk_to_doc:
            chunk_to_doc[cid] = c.doc_id

    bm25_top1_score: Optional[float] = None
    if bm25_results:
        bm25_top1_score = (
            float(bm25_results[0].score)
            if bm25_results[0].score is not None
            else None
        )

    dense_keys = [(cid, None) for cid in dense_chunk_ids if cid]
    bm25_keys = [(c.chunk_id, None) for c in bm25_results if c.chunk_id]
    fused = rrf_fuse_ranked_lists(dense_keys, bm25_keys, k_rrf=k_rrf)
    sliced = fused[: max(1, int(final_k))]
    chunk_ids = tuple(k for k, _ in sliced)
    doc_ids = tuple(chunk_to_doc.get(k, "") for k in chunk_ids)
    fused_scores = tuple(float(s) for _, s in sliced)
    return FusedTopK(
        chunk_ids=chunk_ids,
        doc_ids=doc_ids,
        fused_scores=fused_scores,
        bm25_top1_score=bm25_top1_score,
    )


# ---------------------------------------------------------------------------
# Loop config + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlledRecoveryConfig:
    """Knobs the loop needs to reproduce a run.

    The CLI maps argparse → this dataclass → JSON summary. A reviewer
    can tell from one block exactly which final_k / pool size / mode
    produced a given report.
    """

    rewrite_mode: str = REWRITE_MODE_PRODUCTION_LIKE
    final_k: int = _DEFAULT_FINAL_K
    hybrid_top_k: int = _DEFAULT_HYBRID_TOP_K
    bm25_pool_size: int = _DEFAULT_BM25_POOL_SIZE
    k_rrf: int = DEFAULT_K_RRF
    top_n_for_production: int = 5
    strict_label_leakage: bool = True
    side: str = "candidate"

    def validate(self) -> "ControlledRecoveryConfig":
        if self.rewrite_mode not in REWRITE_MODES:
            raise ValueError(
                f"unknown rewrite_mode={self.rewrite_mode!r}; "
                f"expected one of {REWRITE_MODES}."
            )
        if self.final_k <= 0:
            raise ValueError(f"final_k must be > 0, got {self.final_k}.")
        if self.hybrid_top_k <= 0:
            raise ValueError(
                f"hybrid_top_k must be > 0, got {self.hybrid_top_k}."
            )
        if self.bm25_pool_size <= 0:
            raise ValueError(
                f"bm25_pool_size must be > 0, got {self.bm25_pool_size}."
            )
        if self.k_rrf < 1:
            raise ValueError(f"k_rrf must be >= 1, got {self.k_rrf}.")
        if self.top_n_for_production < 1:
            raise ValueError(
                f"top_n_for_production must be >= 1, "
                f"got {self.top_n_for_production}."
            )
        if self.side not in ("baseline", "candidate"):
            raise ValueError(
                f"side must be 'baseline' / 'candidate', got {self.side!r}."
            )
        return self


@dataclass
class ControlledRecoveryResult:
    """Container for the loop's outputs.

    Lists are kept separate (not zipped) because the writer emits each
    as its own JSONL — keeping the model parallel to the Phase 7.3
    writer.
    """

    config: ControlledRecoveryConfig
    decisions: List[RecoveryDecision] = field(default_factory=list)
    attempts: List[RecoveryAttempt] = field(default_factory=list)
    results: List[RecoveryResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Verdict row loading
# ---------------------------------------------------------------------------


def load_verdict_rows(confidence_jsonl: Path) -> List[Dict[str, Any]]:
    """Load Phase 7.3's per-query confidence JSONL into row dicts.

    Skips empty lines. Returns the full row, including the embedded
    ``input`` block — the policy uses both halves.
    """
    out: List[Dict[str, Any]] = []
    with Path(confidence_jsonl).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Per-query attempt execution
# ---------------------------------------------------------------------------


def _run_bm25(
    retriever: BM25EvalRetriever,
    query: str,
    *,
    pool_size: int,
    clock: Callable[[], float] = time.perf_counter,
) -> Tuple[List[RetrievedChunk], float]:
    """Run BM25 with the configured pool_size and return (results, latency_ms).

    The BM25 retriever is constructed once per loop run; we mutate
    ``_top_k`` only when the caller asks for a different pool size from
    the retriever's default. This is the same low-cost pattern
    ``DenseEvalRetrieverAdapter`` uses on the dense side.
    """
    prev_top_k = retriever.top_k
    desired = max(1, int(pool_size))
    if desired != prev_top_k:
        retriever._top_k = desired  # noqa: SLF001 — local override
    t0 = clock()
    try:
        report = retriever.retrieve(query)
    finally:
        if desired != prev_top_k:
            retriever._top_k = prev_top_k  # noqa: SLF001
    elapsed_ms = round((clock() - t0) * 1000.0, 3)
    return list(report.results), elapsed_ms


def execute_attempt(
    decision: RecoveryDecision,
    *,
    frozen_dense: Optional[FrozenDenseRow],
    bm25_retriever: Optional[BM25EvalRetriever],
    config: ControlledRecoveryConfig,
    clock: Callable[[], float] = time.perf_counter,
) -> RecoveryAttempt:
    """Execute one decision and return a RecoveryAttempt.

    Skips (SKIP_*, NOOP) emit a synthetic attempt with the current
    before-state as the "after" — the metrics layer treats these as
    not-graded (recovered=False, regression=False). ATTEMPT_HYBRID and
    ATTEMPT_REWRITE both call BM25 (rewritten or original) and fuse
    with frozen dense.

    ``frozen_dense`` may be ``None`` for queries that aren't in the
    Phase 7.0 dense bundle — the loop falls back to BM25-only fusion
    (i.e. the BM25 ranked list as the post-recovery list). This keeps
    the loop able to run on a confidence-only JSONL when the operator
    wants a quick sanity check without a per_query dump.
    """
    final_k = config.final_k

    # before-state: gold rank in the frozen dense top-N
    if frozen_dense is not None:
        before_rank = _gold_rank_in(
            frozen_dense.top_chunk_ids,
            frozen_dense.top_doc_ids,
            gold_doc_id=decision.gold_doc_id,
            gold_page_id=decision.gold_page_id,
        )
        before_chunk_ids = frozen_dense.top_chunk_ids
        before_doc_ids = frozen_dense.top_doc_ids
        before_top1 = frozen_dense.top1_score
    else:
        before_rank = _RANK_MISS
        before_chunk_ids = ()
        before_doc_ids = ()
        before_top1 = None
    before_in_top_k = (before_rank != _RANK_MISS) and (before_rank <= final_k)

    if decision.recovery_action in (
        RECOVERY_ACTION_NOOP,
        RECOVERY_ACTION_SKIP_REFUSE,
        RECOVERY_ACTION_SKIP_CAUTION,
        RECOVERY_ACTION_SKIP_DEFER,
    ):
        # No retrieval — after = before.
        return RecoveryAttempt(
            decision=decision,
            before_rank=before_rank,
            before_top_doc_ids=before_doc_ids,
            before_top_chunk_ids=before_chunk_ids,
            before_in_top_k=before_in_top_k,
            before_top1_score=before_top1,
            after_rank=before_rank,
            after_top_doc_ids=before_doc_ids,
            after_top_chunk_ids=before_chunk_ids,
            after_in_top_k=before_in_top_k,
            after_top1_score=before_top1,
            final_k=final_k,
            latency_ms=None,
            error=None,
        )

    if bm25_retriever is None:
        # ATTEMPT_* requested but no BM25 retriever wired — record an
        # error so the row is visible in the unrecovered dump.
        return RecoveryAttempt(
            decision=decision,
            before_rank=before_rank,
            before_top_doc_ids=before_doc_ids,
            before_top_chunk_ids=before_chunk_ids,
            before_in_top_k=before_in_top_k,
            before_top1_score=before_top1,
            after_rank=before_rank,
            after_top_doc_ids=before_doc_ids,
            after_top_chunk_ids=before_chunk_ids,
            after_in_top_k=before_in_top_k,
            after_top1_score=before_top1,
            final_k=final_k,
            latency_ms=None,
            error="bm25_retriever_unavailable",
        )

    # Pick the query the BM25 side runs against.
    if decision.recovery_action == RECOVERY_ACTION_ATTEMPT_HYBRID:
        bm25_query = decision.query_text
    else:
        # ATTEMPT_REWRITE: use the rewritten query when one was
        # produced; fall back to the original query when the rewriter
        # found no usable terms (production_like:noop case).
        bm25_query = decision.rewritten_query or decision.query_text

    try:
        bm25_results, latency_ms = _run_bm25(
            bm25_retriever,
            bm25_query,
            pool_size=config.bm25_pool_size,
            clock=clock,
        )
    except Exception as exc:  # pragma: no cover — defensive
        log.exception("BM25 retrieval failed for qid=%s", decision.query_id)
        return RecoveryAttempt(
            decision=decision,
            before_rank=before_rank,
            before_top_doc_ids=before_doc_ids,
            before_top_chunk_ids=before_chunk_ids,
            before_in_top_k=before_in_top_k,
            before_top1_score=before_top1,
            after_rank=_RANK_MISS,
            after_top_doc_ids=(),
            after_top_chunk_ids=(),
            after_in_top_k=False,
            after_top1_score=None,
            final_k=final_k,
            latency_ms=None,
            error=f"bm25_error:{type(exc).__name__}",
        )

    fused = fuse_dense_and_bm25(
        dense_chunk_ids=before_chunk_ids,
        dense_doc_ids=before_doc_ids,
        bm25_results=bm25_results,
        final_k=config.hybrid_top_k,
        k_rrf=config.k_rrf,
    )
    after_rank = _gold_rank_in(
        fused.chunk_ids,
        fused.doc_ids,
        gold_doc_id=decision.gold_doc_id,
        gold_page_id=decision.gold_page_id,
    )
    after_in_top_k = (after_rank != _RANK_MISS) and (after_rank <= final_k)

    return RecoveryAttempt(
        decision=decision,
        before_rank=before_rank,
        before_top_doc_ids=before_doc_ids,
        before_top_chunk_ids=before_chunk_ids,
        before_in_top_k=before_in_top_k,
        before_top1_score=before_top1,
        after_rank=after_rank,
        after_top_doc_ids=fused.doc_ids,
        after_top_chunk_ids=fused.chunk_ids,
        after_in_top_k=after_in_top_k,
        after_top1_score=fused.bm25_top1_score,
        final_k=final_k,
        latency_ms=latency_ms,
        error=None,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_controlled_recovery(
    *,
    verdict_rows: Sequence[Mapping[str, Any]],
    frozen_dense_by_qid: Mapping[str, FrozenDenseRow],
    bm25_retriever: Optional[BM25EvalRetriever],
    config: ControlledRecoveryConfig,
    progress_log_every: int = 50,
    clock: Callable[[], float] = time.perf_counter,
) -> ControlledRecoveryResult:
    """Drive the recovery loop over a list of Phase 7.3 verdict rows.

    When ``config.rewrite_mode == 'both'``, every QUERY_REWRITE row is
    fanned out into two attempts: one oracle, one production-like. Both
    rows share the same ``query_id`` but carry distinct
    ``rewrite_mode`` / ``oracle_upper_bound`` flags so the writer can
    split them into separate JSONLs. The non-QUERY_REWRITE rows
    (HYBRID, SKIP_*) are emitted exactly once regardless of mode.

    The loop never sees an LLM rewriter — every attempt is reproducible
    from (verdict_row, chunks_jsonl, per_query.jsonl) alone.
    """
    config = config.validate()

    decisions: List[RecoveryDecision] = []
    attempts: List[RecoveryAttempt] = []
    results: List[RecoveryResult] = []

    fan_out_modes: Tuple[str, ...]
    if config.rewrite_mode == REWRITE_MODE_BOTH:
        fan_out_modes = (REWRITE_MODE_ORACLE, REWRITE_MODE_PRODUCTION_LIKE)
    else:
        fan_out_modes = (config.rewrite_mode,)

    for i, row in enumerate(verdict_rows, start=1):
        qid = str(row.get("query_id") or row.get("qid") or "")
        original_action = str(row.get("recommended_action") or "")
        # QUERY_REWRITE rows fan out across modes; everything else
        # only runs once (route under the *first* mode, but the
        # decision body for non-rewrite actions does not depend on it).
        is_rewrite = (original_action == "QUERY_REWRITE")
        per_row_modes = fan_out_modes if is_rewrite else (fan_out_modes[0],)

        for mode in per_row_modes:
            try:
                decision = decide_recovery(
                    row,
                    rewrite_mode=mode,
                    top_n_for_production=config.top_n_for_production,
                    strict_label_leakage=config.strict_label_leakage,
                )
            except LabelLeakageError as exc:
                # Non-strict mode would not raise; in strict mode this
                # is a programming error — every QUERY_REWRITE row in
                # production_like *must not* carry expected_title under
                # the strict guarantee. We surface as a SKIP_DEFER so
                # the run completes and the writer flags it.
                log.warning(
                    "label leakage refusal for qid=%s mode=%s: %s",
                    qid, mode, exc,
                )
                decision = RecoveryDecision(
                    query_id=qid,
                    bucket=str(row.get("bucket") or ""),
                    original_action=original_action,
                    recovery_action=RECOVERY_ACTION_SKIP_DEFER,
                    skip_reason="LABEL_LEAKAGE_REFUSED",
                    rewrite_mode=mode,
                    notes=("label_leakage_refused",),
                )

            decisions.append(decision)
            frozen = frozen_dense_by_qid.get(qid)
            attempt = execute_attempt(
                decision,
                frozen_dense=frozen,
                bm25_retriever=bm25_retriever,
                config=config,
                clock=clock,
            )
            attempts.append(attempt)
            results.append(classify_attempt(attempt, final_k=config.final_k))

        if progress_log_every and (i % progress_log_every == 0):
            log.info("phase7.4 progress: %d/%d", i, len(verdict_rows))

    return ControlledRecoveryResult(
        config=config,
        decisions=decisions,
        attempts=attempts,
        results=results,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _decision_to_dict(d: RecoveryDecision) -> Dict[str, Any]:
    return {
        "query_id": d.query_id,
        "bucket": d.bucket,
        "original_action": d.original_action,
        "recovery_action": d.recovery_action,
        "skip_reason": d.skip_reason,
        "rewrite_mode": d.rewrite_mode,
        "oracle_upper_bound": d.oracle_upper_bound,
        "query_text": d.query_text,
        "rewritten_query": d.rewritten_query,
        "rewrite_terms": list(d.rewrite_terms),
        "rewrite_source": d.rewrite_source,
        "expected_title": d.expected_title,
        "gold_doc_id": d.gold_doc_id,
        "gold_page_id": d.gold_page_id,
        "notes": list(d.notes),
    }


def _attempt_to_dict(a: RecoveryAttempt) -> Dict[str, Any]:
    return {
        "decision": _decision_to_dict(a.decision),
        "before_rank": a.before_rank,
        "before_top_doc_ids": list(a.before_top_doc_ids),
        "before_top_chunk_ids": list(a.before_top_chunk_ids),
        "before_in_top_k": a.before_in_top_k,
        "before_top1_score": a.before_top1_score,
        "after_rank": a.after_rank,
        "after_top_doc_ids": list(a.after_top_doc_ids),
        "after_top_chunk_ids": list(a.after_top_chunk_ids),
        "after_in_top_k": a.after_in_top_k,
        "after_top1_score": a.after_top1_score,
        "final_k": a.final_k,
        "latency_ms": a.latency_ms,
        "error": a.error,
    }


def _result_to_dict(r: RecoveryResult) -> Dict[str, Any]:
    return {
        "query_id": r.query_id,
        "bucket": r.bucket,
        "recovery_action": r.recovery_action,
        "rewrite_mode": r.rewrite_mode,
        "oracle_upper_bound": r.oracle_upper_bound,
        "skipped": r.skipped,
        "recovered": r.recovered,
        "regression": r.regression,
        "gold_newly_entered_candidates": r.gold_newly_entered_candidates,
        "before_rank": r.before_rank,
        "after_rank": r.after_rank,
        "rank_delta": r.rank_delta,
        "final_k": r.final_k,
        "latency_ms": r.latency_ms,
        "notes": list(r.notes),
    }
