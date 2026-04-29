"""Eval-only audit adapter — captures per-stage chunk lists per query.

Why a new adapter instead of instrumenting ``WideRetrievalEvalAdapter``
in place? The audit pipeline needs to surface intermediate chunk lists
(dense pool, MMR-selected, rerank input, reranked, final) so the cap-
policy confirm script can score gold ranks at each stage. Hooking that
into the production adapter would either bloat its return shape or
require a side-channel — both noisier than a sibling adapter that
simply re-runs the same five phases and records each one.

The adapter is functionally identical to ``WideRetrievalEvalAdapter``:

  Phase 1: bi-encoder candidate pool
  Phase 2a: optional MMR selection
  Phase 2b: cap policy on rerank input
  Phase 2c: bound to rerank_in
  Phase 3: cross-encoder rerank
  Phase 4: optional cap policy on final
  Phase 5: truncate to final_top_k

Differences:
  - takes a :class:`CapPolicy` (not a title-cap integer) for both
    rerank-input and final stages, defaulting to a no-cap policy.
  - exposes the per-stage chunk lists on ``last_audit`` after every
    ``retrieve(query)`` call. The driver post-processes that to score
    gold ranks per stage.

Production code (``app/``) is not modified — the adapter mutates only
the ``Retriever`` instance attributes that the existing wide-retrieval
adapter already mutates the same way (``_top_k`` / ``_candidate_k`` /
``_reranker`` / ``_use_mmr`` / ``_mmr_lambda``), and restores them in a
``finally`` block.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from eval.harness.cap_policy import CapPolicy, NoCapPolicy
from eval.harness.wide_retrieval_helpers import (
    DEFAULT_DOC_ID_PENALTY,
    DEFAULT_TITLE_PENALTY,
    TitleProvider,
    mmr_select_score_fallback,
)

log = logging.getLogger(__name__)


@dataclass
class CapPolicyAuditConfig:
    """Knobs for one ``CapPolicyAuditAdapter`` instance.

    Mirrors ``WideRetrievalConfig`` field-for-field except both cap
    inputs are :class:`CapPolicy` instances. ``cap_policy_final``
    defaults to a :class:`NoCapPolicy` so omitting it leaves the final
    stage untouched (matches the legacy adapter's "no cap when None"
    semantics).
    """

    candidate_k: int
    final_top_k: int
    rerank_in: int
    cap_policy_rerank_input: CapPolicy
    cap_policy_final: Optional[CapPolicy] = None
    use_mmr: bool = True
    mmr_lambda: float = 0.65
    mmr_k: int = 48
    doc_id_penalty: float = DEFAULT_DOC_ID_PENALTY
    title_penalty: float = DEFAULT_TITLE_PENALTY


@dataclass
class CapPolicyAuditEvent:
    """Per-call snapshot of the staged chunk lists.

    Lists hold the full chunk objects, in stage order. Empty lists are
    valid (e.g. an empty pool yields all-empty later stages).
    """

    pool: List[Any] = field(default_factory=list)
    mmr_pool: List[Any] = field(default_factory=list)
    after_cap_rerank_input: List[Any] = field(default_factory=list)
    rerank_input: List[Any] = field(default_factory=list)
    reranked: List[Any] = field(default_factory=list)
    after_cap_final: List[Any] = field(default_factory=list)
    final: List[Any] = field(default_factory=list)
    cap_group_sizes_pre: Dict[str, int] = field(default_factory=dict)
    cap_group_sizes_post: Dict[str, int] = field(default_factory=dict)
    capped_out_chunks: Dict[str, List[Any]] = field(default_factory=dict)
    pool_ms: float = 0.0
    rerank_ms: float = 0.0


class CapPolicyAuditAdapter:
    """Wide-retrieval adapter with explicit cap-policy + per-stage audit.

    Pipeline phases mirror :class:`WideRetrievalEvalAdapter` exactly so
    metrics are comparable. The only behavioural difference is that
    cap is applied via a :class:`CapPolicy` (not a title-cap integer),
    and intermediate chunk lists are recorded on ``last_audit`` after
    every ``retrieve()`` call.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        config: CapPolicyAuditConfig,
        final_reranker: Any,
        title_provider: Optional[TitleProvider] = None,
        name: str = "cap-policy-audit",
    ) -> None:
        from app.capabilities.rag.reranker import NoOpReranker

        if config.candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")
        if config.final_top_k <= 0:
            raise ValueError("final_top_k must be > 0")
        if config.rerank_in <= 0:
            raise ValueError("rerank_in must be > 0")
        self._r = retriever
        self._cfg = config
        self._final_reranker = final_reranker
        self._title_provider = title_provider
        self._name = str(name)
        self._noop = NoOpReranker()
        self._last_audit: Optional[CapPolicyAuditEvent] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> CapPolicyAuditConfig:
        return self._cfg

    @property
    def last_audit(self) -> Optional[CapPolicyAuditEvent]:
        return self._last_audit

    def retrieve(self, query: str) -> Any:
        cfg = self._cfg

        prev_top_k = self._r._top_k  # noqa: SLF001
        prev_cand_k = self._r._candidate_k  # noqa: SLF001
        prev_reranker = self._r._reranker  # noqa: SLF001
        prev_use_mmr = getattr(self._r, "_use_mmr", False)
        prev_mmr_lambda = getattr(self._r, "_mmr_lambda", 0.7)

        audit = CapPolicyAuditEvent()
        try:
            self._r._use_mmr = False  # noqa: SLF001
            self._r._top_k = cfg.candidate_k  # noqa: SLF001
            self._r._candidate_k = cfg.candidate_k  # noqa: SLF001
            self._r._reranker = self._noop  # noqa: SLF001

            pool_t0 = time.perf_counter()
            pool_report = self._r.retrieve(query)
            audit.pool_ms = round(
                (time.perf_counter() - pool_t0) * 1000.0, 3
            )

            pool_results: List[Any] = list(
                getattr(pool_report, "results", []) or []
            )
            audit.pool = list(pool_results)

            candidate_doc_ids: List[str] = []
            seen_doc: set = set()
            for chunk in pool_results:
                doc_id = str(getattr(chunk, "doc_id", "") or "")
                if doc_id and doc_id not in seen_doc:
                    candidate_doc_ids.append(doc_id)
                    seen_doc.add(doc_id)

            staged: List[Any] = list(pool_results)
            if cfg.use_mmr:
                staged = mmr_select_score_fallback(
                    staged,
                    top_k=max(cfg.rerank_in, cfg.mmr_k),
                    lambda_val=cfg.mmr_lambda,
                    doc_id_penalty=cfg.doc_id_penalty,
                    title_penalty=cfg.title_penalty,
                    title_provider=self._title_provider,
                )
            audit.mmr_pool = list(staged)

            # Capture cap-group state pre/post + capped-out per group.
            audit.cap_group_sizes_pre = (
                cfg.cap_policy_rerank_input.group_sizes(staged)
            )
            audit.capped_out_chunks = (
                cfg.cap_policy_rerank_input.cap_out_records(staged)
            )
            staged_after_cap = cfg.cap_policy_rerank_input.apply(staged)
            audit.cap_group_sizes_post = (
                cfg.cap_policy_rerank_input.group_sizes(staged_after_cap)
            )
            audit.after_cap_rerank_input = list(staged_after_cap)

            rerank_input = staged_after_cap[: max(1, int(cfg.rerank_in))]
            audit.rerank_input = list(rerank_input)

            rerank_t0 = time.perf_counter()
            reranked = self._final_reranker.rerank(
                query, rerank_input, k=len(rerank_input),
            )
            audit.rerank_ms = round(
                (time.perf_counter() - rerank_t0) * 1000.0, 3
            )
            audit.reranked = list(reranked)

            staged_final: List[Any] = list(reranked)
            if cfg.cap_policy_final is not None:
                staged_final = cfg.cap_policy_final.apply(staged_final)
            audit.after_cap_final = list(staged_final)

            final_results = staged_final[: max(1, int(cfg.final_top_k))]
            audit.final = list(final_results)
        finally:
            self._r._top_k = prev_top_k  # noqa: SLF001
            self._r._candidate_k = prev_cand_k  # noqa: SLF001
            self._r._reranker = prev_reranker  # noqa: SLF001
            self._r._use_mmr = prev_use_mmr  # noqa: SLF001
            self._r._mmr_lambda = prev_mmr_lambda  # noqa: SLF001

        self._last_audit = audit

        return SimpleNamespace(
            results=final_results,
            candidate_doc_ids=candidate_doc_ids,
            index_version=getattr(pool_report, "index_version", None),
            embedding_model=getattr(pool_report, "embedding_model", None),
            reranker_name=getattr(self._final_reranker, "name", self._name),
            rerank_ms=audit.rerank_ms,
            dense_retrieval_ms=audit.pool_ms,
            rerank_breakdown_ms=getattr(
                self._final_reranker, "last_breakdown_ms", None,
            ),
            wide_config=cfg,
            pool_size=cfg.candidate_k,
        )


# ---------------------------------------------------------------------------
# Audit scoring — given a CapPolicyAuditEvent + expected_doc_ids, derive
# the per-stage gold rank fields the cap_audit.jsonl row needs.
# ---------------------------------------------------------------------------


def _first_gold_rank(
    chunks: List[Any], expected_doc_ids: List[str],
) -> Optional[int]:
    """1-based rank of the first chunk whose doc_id matches a gold,
    or None when no such chunk is in ``chunks``.
    """
    if not expected_doc_ids:
        return None
    expected_set = {str(d) for d in expected_doc_ids if d}
    if not expected_set:
        return None
    for i, chunk in enumerate(chunks, start=1):
        doc_id = str(getattr(chunk, "doc_id", "") or "")
        if doc_id and doc_id in expected_set:
            return i
    return None


def _gold_chunk_at_first_match(
    chunks: List[Any], expected_doc_ids: List[str],
) -> Optional[Any]:
    if not expected_doc_ids:
        return None
    expected_set = {str(d) for d in expected_doc_ids if d}
    for chunk in chunks:
        doc_id = str(getattr(chunk, "doc_id", "") or "")
        if doc_id and doc_id in expected_set:
            return chunk
    return None


def score_audit_event(
    event: CapPolicyAuditEvent,
    *,
    expected_doc_ids: List[str],
    cap_policy: CapPolicy,
) -> Dict[str, Any]:
    """Return the cap_audit.jsonl payload fields for one query.

    Computes 1-based gold ranks at each stage, the cap-out boolean (True
    iff gold was in the MMR pool but missing from the after-cap pool),
    and the offending group key + size when capped out.
    """
    pool_rank = _first_gold_rank(event.pool, expected_doc_ids)
    mmr_rank = _first_gold_rank(event.mmr_pool, expected_doc_ids)
    after_cap_rank = _first_gold_rank(
        event.after_cap_rerank_input, expected_doc_ids,
    )
    rerank_input_rank = _first_gold_rank(
        event.rerank_input, expected_doc_ids,
    )
    final_rank = _first_gold_rank(event.final, expected_doc_ids)

    gold_chunk = _gold_chunk_at_first_match(
        event.mmr_pool, expected_doc_ids,
    ) or _gold_chunk_at_first_match(event.pool, expected_doc_ids)

    capped_out = (
        mmr_rank is not None
        and after_cap_rank is None
    )
    group_key = ""
    group_size_pre = 0
    if gold_chunk is not None:
        group_key = cap_policy.key_for(gold_chunk)
        if group_key:
            group_size_pre = event.cap_group_sizes_pre.get(group_key, 0)

    return {
        "gold_dense_rank": pool_rank,
        "gold_after_mmr_rank": mmr_rank,
        "gold_after_cap_rerank_input_rank": after_cap_rank,
        "gold_rerank_input_rank": rerank_input_rank,
        "gold_final_rank": final_rank,
        "gold_was_in_dense_pool": pool_rank is not None,
        "gold_was_in_mmr_pool": mmr_rank is not None,
        "gold_was_in_rerank_input": rerank_input_rank is not None,
        "gold_was_capped_out": bool(capped_out),
        "capped_out_by_group_key": group_key if capped_out else "",
        "capped_out_group_size": group_size_pre if capped_out else 0,
        "cap_policy": cap_policy.name,
        "cap_policy_cap": cap_policy.cap,
    }
