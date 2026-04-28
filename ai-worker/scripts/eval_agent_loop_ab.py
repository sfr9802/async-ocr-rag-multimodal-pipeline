"""Offline A/B eval CLI: legacy ``AgentLoopController`` vs ``AgentLoopGraph``.

Runs both backends in-process against the same query set and emits
``raw_results.jsonl``, ``summary.csv``, and ``comparison_summary.json``
under ``eval/agent_loop_ab/<run-name>/``. Side-effect-free with respect
to the operational stack:

  * No Redis queue / TaskRunner involved.
  * No callback to core-api emitted.
  * No DB write performed (the harness reads metadata via the retriever
    bundle that is already loaded by the registry; nothing else writes).
  * No Spring repo edits.

Two run modes ship out of the box:

  ``--mode=stub``    — uses a deterministic in-memory retriever +
                       extractive generator. Zero network, no FAISS
                       index needed. Fast smoke check that the harness
                       wiring is correct.
  ``--mode=registry`` — pulls a real Retriever + GenerationProvider
                       from ``build_default_registry``. Intended for
                       offline benches with the FAISS index built and
                       (optionally) Ollama / Claude reachable.

Either mode invokes the legacy controller and the graph backend with
exactly the same critic / rewriter / parser / budget so the comparison
is fair.

Usage:

  # quick smoke (no FAISS index needed)
  python -m scripts.eval_agent_loop_ab \\
      --queries ai-worker/fixtures/agent_loop_ab/smoke.jsonl \\
      --mode stub --run-name smoke

  # real bench against the live registry
  python -m scripts.eval_agent_loop_ab \\
      --queries my_eval.csv \\
      --mode registry --run-name 2026-04-28-graph-vs-legacy

The script never imports app/workers/taskrunner or app/clients/redis_*
so a misconfigured run can't accidentally publish a job.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from app.capabilities.agent.critic import (
    AgentCriticProvider,
    NoOpCritic,
    RuleCritic,
)
from app.capabilities.agent.graph_loop import AgentLoopGraph
from app.capabilities.agent.loop import AgentLoopController, LoopBudget
from app.capabilities.agent.rewriter import (
    NoOpQueryRewriter,
    QueryRewriterProvider,
)
from app.capabilities.agent.synthesizer import AgentSynthesizer
from app.capabilities.rag.generation import (
    ExtractiveGenerator,
    GenerationProvider,
    RetrievedChunk,
)
from app.capabilities.rag.query_parser import (
    NoOpQueryParser,
    ParsedQuery,
    QueryParserProvider,
    RegexQueryParser,
)
from eval.harness.agent_loop_ab import (
    AgentLoopABQuery,
    load_query_rows,
    make_default_executor_builder,
    run_ab_eval,
    write_outputs,
)


log = logging.getLogger("scripts.eval_agent_loop_ab")


# Default output root. Overridable via --output-dir; the run-name nests
# the run-level directory underneath so the operator can keep multiple
# runs side-by-side without manual mkdir.
_DEFAULT_OUTPUT_ROOT = Path("eval/agent_loop_ab")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    queries = load_query_rows(args.queries)
    if not queries:
        log.error("No queries loaded from %s — aborting.", args.queries)
        return 2

    if args.max_queries is not None and args.max_queries > 0:
        queries = queries[: args.max_queries]
        log.info("Trimmed query set to first %d rows", len(queries))

    parser = _build_parser(args.parser)
    critic, rewriter, generator, retriever = _build_components(args, parser)
    synthesizer = AgentSynthesizer(generator) if args.synthesize else None
    budget = LoopBudget(
        max_iter=args.max_iter,
        max_total_ms=args.max_total_ms,
        max_llm_tokens=args.max_llm_tokens,
        min_confidence_to_stop=args.min_confidence,
    )

    legacy_runner = AgentLoopController(
        critic=critic,
        rewriter=rewriter,
        parser=parser,
        budget=budget,
    )
    try:
        graph_runner = AgentLoopGraph(
            critic=critic,
            rewriter=rewriter,
            parser=parser,
            budget=budget,
        )
    except Exception as ex:
        log.error(
            "Failed to instantiate AgentLoopGraph (%s: %s). "
            "Install langgraph and try again.",
            type(ex).__name__, ex,
        )
        return 3

    executor_builder = make_default_executor_builder(
        retriever=retriever, generator=generator,
    )

    started = time.perf_counter()
    rows, summary = run_ab_eval(
        queries=queries,
        legacy_runner=legacy_runner,
        graph_runner=graph_runner,
        parser=parser,
        executor_builder=executor_builder,
        legacy_synthesizer=synthesizer,
        graph_synthesizer=synthesizer,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)

    output_dir = (args.output_dir or _DEFAULT_OUTPUT_ROOT) / args.run_name
    paths = write_outputs(
        output_dir=output_dir,
        rows=rows,
        summary=summary,
        metadata={
            "queryFile": str(args.queries),
            "runName": args.run_name,
            "mode": args.mode,
            "parser": args.parser,
            "critic": args.critic,
            "rewriter": args.rewriter,
            "generator": args.generator,
            "retriever": args.retriever,
            "maxIter": args.max_iter,
            "maxTotalMs": args.max_total_ms,
            "maxLlmTokens": args.max_llm_tokens,
            "minConfidence": args.min_confidence,
            "synthesize": args.synthesize,
            "queryCount": len(queries),
            "elapsedMs": elapsed_ms,
        },
    )

    print(json.dumps(
        {
            "queryCount": len(queries),
            "elapsedMs": elapsed_ms,
            "summary": summary,
            "paths": {k: str(v) for k, v in paths.items()},
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline A/B eval comparing legacy AgentLoopController and "
            "AgentLoopGraph (LangGraph). No Redis / DB / callback writes."
        )
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Path to JSONL or CSV query file (see eval/agent_loop_ab/README).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Sub-directory name under --output-dir. Defaults to "
             "'run-<epoch>' so successive invocations don't clobber.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output root directory. Defaults to {_DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--mode",
        choices=("stub", "registry"),
        default="stub",
        help=(
            "Component-wiring mode. 'stub' uses an in-memory retriever + "
            "extractive generator (fastest smoke). 'registry' pulls live "
            "components from build_default_registry."
        ),
    )
    parser.add_argument(
        "--parser",
        choices=("noop", "regex"),
        default="regex",
        help="Query parser variant for the loop. Default: regex.",
    )
    parser.add_argument(
        "--critic",
        choices=("noop", "rule"),
        default="rule",
        help="Critic variant. 'rule' enables real iterations.",
    )
    parser.add_argument(
        "--rewriter",
        choices=("noop",),
        default="noop",
        help=(
            "Rewriter variant for offline mode. Production wiring uses "
            "LlmQueryRewriter; the harness sticks with NoOp so the "
            "comparison stays deterministic."
        ),
    )
    parser.add_argument(
        "--generator",
        choices=("extractive",),
        default="extractive",
        help="Generation provider used by the executor + synthesizer.",
    )
    parser.add_argument(
        "--retriever",
        choices=("stub", "registry"),
        default=None,
        help=(
            "Retriever override. When omitted the harness picks the value "
            "implied by --mode (stub -> stub retriever; registry -> live)."
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="LoopBudget.max_iter (default 3).",
    )
    parser.add_argument(
        "--max-total-ms",
        type=int,
        default=15_000,
        help="LoopBudget.max_total_ms (default 15000).",
    )
    parser.add_argument(
        "--max-llm-tokens",
        type=int,
        default=4_000,
        help="LoopBudget.max_llm_tokens (default 4000).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.75,
        help="LoopBudget.min_confidence_to_stop (default 0.75).",
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Run AgentSynthesizer over outcome.aggregated_chunks (slower).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap the loaded query set (smoke / debug aid).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Emit DEBUG-level logging from the harness.",
    )
    args = parser.parse_args(argv)
    if not args.run_name:
        args.run_name = f"run-{int(time.time())}"
    return args


# ---------------------------------------------------------------------------
# Component wiring
# ---------------------------------------------------------------------------


def _build_parser(name: str) -> QueryParserProvider:
    if name == "noop":
        return NoOpQueryParser()
    return RegexQueryParser()


def _build_components(
    args: argparse.Namespace,
    parser: QueryParserProvider,
) -> Tuple[
    AgentCriticProvider,
    QueryRewriterProvider,
    GenerationProvider,
    Any,
]:
    """Resolve critic, rewriter, generator, and retriever for the run."""
    critic: AgentCriticProvider = (
        NoOpCritic() if args.critic == "noop" else RuleCritic()
    )
    rewriter: QueryRewriterProvider = NoOpQueryRewriter()
    generator: GenerationProvider = ExtractiveGenerator()

    retriever_mode = args.retriever or args.mode
    if retriever_mode == "stub":
        retriever: Any = _StubRetriever()
        log.info("Retriever: stub (in-memory deterministic)")
    elif retriever_mode == "registry":
        retriever = _build_registry_retriever_or_die()
        log.info("Retriever: live registry retriever (%s)", type(retriever).__name__)
    else:  # pragma: no cover - argparse limits this
        raise ValueError(f"Unknown retriever mode: {retriever_mode}")

    return critic, rewriter, generator, retriever


def _build_registry_retriever_or_die() -> Any:
    """Pull a live Retriever from ``build_default_registry`` without
    triggering Redis / callback / Spring writes.

    The registry's RAG capability builds its own retriever bundle and
    caches it in ``_shared_component_cache``; we reach in through the
    ``RagCapability``'s ``_retriever`` attribute. If RAG isn't
    registered the harness fails fast — there's no point continuing
    without retrieval.
    """
    from app.capabilities.registry import build_default_registry
    from app.core.config import get_settings

    settings = get_settings()
    registry = build_default_registry(settings)
    available = registry.available()
    if "RAG" not in available:
        raise RuntimeError(
            "registry mode requires the RAG capability; the registry only "
            f"reported {available}. Build the FAISS index "
            "(python -m scripts.build_rag_index --fixture), check "
            "AIPIPELINE_WORKER_RAG_ENABLED=true, then retry."
        )
    rag = registry.get("RAG")
    retriever = getattr(rag, "_retriever", None) or getattr(rag, "retriever", None)
    if retriever is None:
        raise RuntimeError(
            "RagCapability did not expose a retriever attribute. Update "
            "the harness wiring or use --mode=stub."
        )
    return retriever


# ---------------------------------------------------------------------------
# Stub retriever for offline smoke runs
# ---------------------------------------------------------------------------


class _StubRetriever:
    """In-memory deterministic retriever.

    Returns a fixed list of three chunks whose ``doc_id`` derives from a
    hash of the query so the same query yields the same chunks across
    legacy / graph runs. Score is a function of query length so the
    ``top1_score`` metric varies per row, which lets the aggregator
    produce non-trivial percentile outputs in unit tests.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str):  # noqa: D401 - protocol match
        self.calls.append(query or "")
        seed = abs(hash(query or "")) % 7
        chunks: list[RetrievedChunk] = []
        for i in range(3):
            doc_idx = (seed + i) % 5
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"stub-{doc_idx}-{i}",
                    doc_id=f"stub-doc-{doc_idx}",
                    section=f"section-{i}",
                    text=(
                        f"Stub passage {i} for query={query!r} (doc {doc_idx}). "
                        "The agent loop A/B harness uses this fixed text so "
                        "rule-based critic verdicts are deterministic across "
                        "runs."
                    ),
                    score=round(0.9 - (0.1 * i) - (0.01 * (seed % 3)), 4),
                )
            )

        class _Report:
            def __init__(self, results):
                self.results = results
                # Keep rerank_ms None so the harness counts retrieve-only;
                # the stub makes no rerank pass.
                self.rerank_ms = None
                self.dense_retrieval_ms = 0.0

        return _Report(chunks)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


if __name__ == "__main__":
    sys.exit(main())
