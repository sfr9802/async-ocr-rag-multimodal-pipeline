"""Eval-only adapter that runs a wide-pool retrieval pipeline.

Wraps the production ``Retriever`` to expose four extra knobs the
production code path doesn't take as a single bundle:

  - ``candidate_k``      : how many bi-encoder candidates to fetch
  - ``mmr_*``            : optional eval-only MMR selection (score
                           fallback variant; see
                           ``eval.harness.wide_retrieval_helpers``)
  - ``title_cap``        : cap chunks per title/doc_id at the rerank
                           input AND/OR final top-k stage
  - ``rerank_in``        : how many post-MMR / post-cap candidates the
                           cross-encoder reranker actually scores

The adapter pipeline per query:

    1. mutate ``Retriever._top_k`` / ``_candidate_k`` / ``_reranker``
       to NoOp + candidate_k → fetch the bi-encoder pool.
    2. (optional) MMR-select to ``mmr_k`` candidates, using the score-
       fallback MMR with optional title penalty.
    3. (optional) apply title cap to bound the rerank-input pool.
    4. truncate to ``rerank_in`` candidates and run the supplied
       cross-encoder over that slice.
    5. (optional) apply title cap to the final reranked list.
    6. truncate to ``final_top_k``.

The adapter restores the Retriever's internal state in a ``finally``
block so a failure mid-call doesn't poison subsequent runs. Per-call
candidate_doc_ids surfaces the raw bi-encoder pool (capped at
``candidate_k``) so Phase 1 candidate-hit metrics fire on the actual
wide pool, not the post-MMR sliver.

Production code is *not* modified — the adapter only mutates instance
attributes that the existing minimal sweep already mutates the same
way.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from eval.harness.cap_policy import CapPolicy
from eval.harness.wide_retrieval_helpers import (
    DEFAULT_DOC_ID_PENALTY,
    DEFAULT_TITLE_PENALTY,
    TitleProvider,
    apply_title_cap,
    mmr_select_score_fallback,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WideRetrievalConfig:
    """All knobs for one ``WideRetrievalEvalAdapter`` instance.

    Treat each field as orthogonal to the others — the adapter does
    NOT inter-validate them. The sweep driver is responsible for
    composing sensible (candidate_k, mmr_k, rerank_in, final_top_k)
    tuples; the adapter just executes whatever it's handed.

    The legacy ``title_cap_rerank_input`` / ``title_cap_final``
    integers stay supported for the existing wide-MMR / format-confirm
    sweeps. The newer ``cap_policy_rerank_input`` /
    ``cap_policy_final`` fields override those integers when supplied —
    used by the cap-policy confirm sweep to swap the cap *grouping
    rule* (title / doc_id / no-cap / section_path) without changing
    any of the other knobs.
    """

    candidate_k: int
    final_top_k: int
    rerank_in: int
    use_mmr: bool = False
    mmr_lambda: float = 0.65
    mmr_k: int = 64
    title_cap_rerank_input: Optional[int] = None
    title_cap_final: Optional[int] = None
    doc_id_penalty: float = DEFAULT_DOC_ID_PENALTY
    title_penalty: float = DEFAULT_TITLE_PENALTY
    # Optional policy override. When set, takes precedence over the
    # corresponding ``title_cap_*`` integer at that stage. ``None``
    # means "fall back to the legacy title-cap integer".
    cap_policy_rerank_input: Optional[CapPolicy] = None
    cap_policy_final: Optional[CapPolicy] = None


class WideRetrievalEvalAdapter:
    """Two-phase retrieve() with MMR + title cap + bounded rerank input.

    Phase 1 — candidate pool:
      Mutates the Retriever's ``_top_k`` / ``_candidate_k`` /
      ``_reranker`` to (candidate_k, candidate_k, NoOp) and runs
      ``retrieve()``. The returned list is the bi-encoder candidate
      pool, ordered by raw bi-encoder score. ``candidate_doc_ids`` on
      the emitted report is the deduplicated doc_id list across this
      pool, so Phase 1 candidate@K hit/recall fires on the right pool.

    Phase 2 — MMR / cap / rerank / final cap:
      Optionally MMR-selects ``mmr_k`` from the pool. Optionally caps
      titles at ``title_cap_rerank_input``. Slices to ``rerank_in``
      and runs the supplied reranker over that slice. Optionally caps
      titles again at ``title_cap_final``. Truncates to
      ``final_top_k``.

    Latency surfacing:
      ``dense_retrieval_ms`` is the wall-clock of the bi-encoder pool
      pass. ``rerank_ms`` is the wall-clock of the final reranker
      call. The harness's existing aggregator picks both up unchanged.
      MMR / title-cap wall-clock is tiny (pure Python over <300
      chunks) and is not separately reported; it's bundled into the
      delta between dense_retrieval_ms + rerank_ms and the eval
      harness's own ``retrieval_ms`` (full retrieve() wall-clock).
    """

    def __init__(
        self,
        retriever: Any,
        *,
        config: WideRetrievalConfig,
        final_reranker: Any,
        title_provider: Optional[TitleProvider] = None,
        name: str = "wide-mmr-titlecap",
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

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> WideRetrievalConfig:
        return self._cfg

    def retrieve(self, query: str) -> Any:
        cfg = self._cfg

        # ---- snapshot Retriever internals so we can restore them ----
        prev_top_k = self._r._top_k  # noqa: SLF001
        prev_cand_k = self._r._candidate_k  # noqa: SLF001
        prev_reranker = self._r._reranker  # noqa: SLF001
        prev_use_mmr = getattr(self._r, "_use_mmr", False)
        prev_mmr_lambda = getattr(self._r, "_mmr_lambda", 0.7)

        try:
            # ---- Phase 1: bi-encoder candidate pool, NoOp reranker ----
            # Force production MMR off so the pool is the raw bi-
            # encoder ranking — eval-only MMR runs in Phase 2 over the
            # already-fetched pool, sharing the same lambda/penalty
            # contract regardless of how the production Retriever is
            # configured.
            self._r._use_mmr = False  # noqa: SLF001
            self._r._top_k = cfg.candidate_k  # noqa: SLF001
            self._r._candidate_k = cfg.candidate_k  # noqa: SLF001
            self._r._reranker = self._noop  # noqa: SLF001

            pool_t0 = time.perf_counter()
            pool_report = self._r.retrieve(query)
            pool_elapsed_ms = round(
                (time.perf_counter() - pool_t0) * 1000.0, 3
            )

            pool_results: List[Any] = list(
                getattr(pool_report, "results", []) or []
            )

            # candidate_doc_ids from the bi-encoder pool — deduplicated
            # in pool order so Phase 1 candidate@K metrics fire on the
            # full candidate_k slice.
            candidate_doc_ids: List[str] = []
            seen_doc: set = set()
            for chunk in pool_results:
                doc_id = str(getattr(chunk, "doc_id", "") or "")
                if doc_id and doc_id not in seen_doc:
                    candidate_doc_ids.append(doc_id)
                    seen_doc.add(doc_id)

            # ---- Phase 2a: optional MMR on the pool ----
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

            # ---- Phase 2b: optional cap on rerank input ----
            # Policy override wins. Falling back to legacy title cap
            # keeps the existing wide-MMR sweep exactly reproducing
            # apply_title_cap; the new cap-policy confirm sweep uses
            # cap_policy_rerank_input to swap doc_id / section_path /
            # no_cap variants without re-plumbing the adapter.
            if cfg.cap_policy_rerank_input is not None:
                staged = cfg.cap_policy_rerank_input.apply(staged)
            elif cfg.title_cap_rerank_input is not None:
                staged = apply_title_cap(
                    staged,
                    cap=cfg.title_cap_rerank_input,
                    title_provider=self._title_provider,
                )

            # ---- Phase 2c: bound rerank input slice ----
            rerank_input = staged[: max(1, int(cfg.rerank_in))]

            # ---- Phase 3: real reranker over the bounded slice ----
            rerank_t0 = time.perf_counter()
            reranked = self._final_reranker.rerank(
                query, rerank_input, k=len(rerank_input),
            )
            rerank_elapsed_ms = round(
                (time.perf_counter() - rerank_t0) * 1000.0, 3
            )

            # ---- Phase 4: optional cap on final ----
            staged_final: List[Any] = list(reranked)
            if cfg.cap_policy_final is not None:
                staged_final = cfg.cap_policy_final.apply(staged_final)
            elif cfg.title_cap_final is not None:
                staged_final = apply_title_cap(
                    staged_final,
                    cap=cfg.title_cap_final,
                    title_provider=self._title_provider,
                )

            # ---- Phase 5: final truncation ----
            final_results = staged_final[: max(1, int(cfg.final_top_k))]
        finally:
            # restore Retriever internals — even on exception this
            # leaves the underlying retriever in its pre-call state.
            self._r._top_k = prev_top_k  # noqa: SLF001
            self._r._candidate_k = prev_cand_k  # noqa: SLF001
            self._r._reranker = prev_reranker  # noqa: SLF001
            self._r._use_mmr = prev_use_mmr  # noqa: SLF001
            self._r._mmr_lambda = prev_mmr_lambda  # noqa: SLF001

        return SimpleNamespace(
            results=final_results,
            candidate_doc_ids=candidate_doc_ids,
            index_version=getattr(pool_report, "index_version", None),
            embedding_model=getattr(pool_report, "embedding_model", None),
            reranker_name=getattr(self._final_reranker, "name", self._name),
            rerank_ms=rerank_elapsed_ms,
            dense_retrieval_ms=pool_elapsed_ms,
            rerank_breakdown_ms=getattr(
                self._final_reranker, "last_breakdown_ms", None,
            ),
            wide_config=cfg,
            pool_size=cfg.candidate_k,
        )
