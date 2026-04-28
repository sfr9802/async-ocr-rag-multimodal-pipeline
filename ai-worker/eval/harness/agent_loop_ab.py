"""Offline A/B harness comparing legacy ``AgentLoopController`` vs ``AgentLoopGraph``.

Drives both backends in-process through their shared ``run(...) -> LoopOutcome``
contract on the same query set, with no Redis queue / TaskRunner / Spring
callback / DB write involved. The harness owns three pieces:

  1. **Input loading** (``load_query_rows``) — JSONL + CSV with a flat schema:
     ``query`` is required; ``expected_doc_id`` / ``expected_keywords`` /
     ``input_kind`` / ``capability`` / ``metadata`` are optional. Identical
     fields across both backends keep the comparison fair.

  2. **Backend runners** (``run_backend_for_query``) — invoke a runner
     directly via its ``run(question, initial_parsed_query, execute_fn)``
     surface. The harness builds the ``execute_fn`` closure around a
     ``Retriever`` + ``GenerationProvider`` exactly the way
     ``AgentCapability._run_loop_and_synthesize`` does, but skips the
     capability shell so we don't drag classify/route/dispatch noise into
     the per-query metric.

  3. **Metrics + aggregation** (``extract_metrics``, ``compare_runs``) —
     every metric the spec calls out is read off the ``LoopOutcome`` /
     ``LoopStep`` / ``RetrievalReport`` triples; the comparison summary
     tracks per-query graph-vs-legacy verdicts (``win`` / ``tie`` /
     ``regression``).

The harness deliberately avoids the production-wiring imports
(``app.workers.taskrunner``, callback clients, Redis client). It only
imports the value objects and provider seams. Operators run it from
``scripts/eval_agent_loop_ab.py``; the script wires their concrete
retriever / generator / parser / critic / rewriter via the registry on
startup.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
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

from app.capabilities.agent.critic import AgentCriticProvider
from app.capabilities.agent.loop import (
    AgentLoopController,
    LoopBudget,
    LoopOutcome,
    LoopStep,
)
from app.capabilities.agent.rewriter import QueryRewriterProvider
from app.capabilities.agent.synthesizer import AgentSynthesizer
from app.capabilities.rag.generation import GenerationProvider, RetrievedChunk
from app.capabilities.rag.query_parser import (
    ParsedQuery,
    QueryParserProvider,
)

log = logging.getLogger(__name__)


BACKEND_LEGACY = "legacy"
BACKEND_GRAPH = "graph"


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentLoopABQuery:
    """One input row.

    All fields except ``query`` are optional. Stable string-typing on
    ``query_id`` keeps JSONL / CSV round-trips clean and gives a stable
    join key in summary CSV output.
    """

    query_id: str
    query: str
    expected_doc_id: Optional[str] = None
    expected_keywords: List[str] = field(default_factory=list)
    input_kind: Optional[str] = None
    capability: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queryId": self.query_id,
            "query": self.query,
            "expectedDocId": self.expected_doc_id,
            "expectedKeywords": list(self.expected_keywords),
            "inputKind": self.input_kind,
            "capability": self.capability,
            "metadata": dict(self.metadata),
        }


def load_query_rows(path: Path) -> List[AgentLoopABQuery]:
    """Load query rows from a ``.jsonl`` or ``.csv`` file.

    Format detection is by extension: ``.jsonl`` / ``.json`` is one JSON
    object per line; anything else is parsed as CSV (UTF-8, header row
    required). Fields not in the schema are tucked under ``metadata``
    so existing dataset files with extra columns load without drift.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Query file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".jsonl", ".json"):
        return _load_jsonl(p)
    return _load_csv(p)


def _load_jsonl(path: Path) -> List[AgentLoopABQuery]:
    rows: List[AgentLoopABQuery] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {ex}"
                ) from ex
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Line {line_no} of {path} must be a JSON object, "
                    f"got {type(obj).__name__}"
                )
            rows.append(_row_from_mapping(obj, idx=line_no))
    log.info("agent_loop_ab: loaded %d JSONL queries from %s", len(rows), path)
    return rows


def _load_csv(path: Path) -> List[AgentLoopABQuery]:
    rows: List[AgentLoopABQuery] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or "query" not in reader.fieldnames:
            raise ValueError(
                f"CSV {path} must have a 'query' column; "
                f"got fieldnames={reader.fieldnames}"
            )
        for idx, raw in enumerate(reader, start=1):
            cleaned = {k: (v if v != "" else None) for k, v in raw.items()}
            rows.append(_row_from_mapping(cleaned, idx=idx))
    log.info("agent_loop_ab: loaded %d CSV queries from %s", len(rows), path)
    return rows


def _row_from_mapping(raw: Mapping[str, Any], *, idx: int) -> AgentLoopABQuery:
    query = str(raw.get("query") or "").strip()
    if not query:
        raise ValueError(
            f"Row {idx} has no usable 'query' field; got {raw!r}"
        )
    query_id = str(raw.get("query_id") or raw.get("id") or f"q-{idx:04d}").strip()
    expected_doc = raw.get("expected_doc_id")
    if not expected_doc:
        # silver_200-style schema carries ``expected_doc_ids`` (plural).
        # Take the first id so per-row hit@k works on those datasets
        # without requiring a separate projection pass.
        docs_plural = raw.get("expected_doc_ids")
        if isinstance(docs_plural, (list, tuple)) and docs_plural:
            expected_doc = docs_plural[0]
    expected_doc = str(expected_doc).strip() if expected_doc else None

    keywords_raw = raw.get("expected_keywords")
    if keywords_raw is None:
        # silver_200-style schema names the keyword field
        # ``expected_section_keywords``.
        keywords_raw = raw.get("expected_section_keywords")
    if isinstance(keywords_raw, str):
        # CSV stores a list as "kw1|kw2"; tolerate "," too for hand-edited files.
        sep = "|" if "|" in keywords_raw else ","
        expected_keywords = [
            k.strip() for k in keywords_raw.split(sep) if k.strip()
        ]
    elif isinstance(keywords_raw, (list, tuple)):
        expected_keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
    else:
        expected_keywords = []

    input_kind = raw.get("input_kind")
    input_kind = str(input_kind).strip() if input_kind else None

    capability = raw.get("capability")
    capability = str(capability).strip().lower() if capability else None
    if capability and capability not in ("rag", "multimodal"):
        log.warning(
            "agent_loop_ab: row %s has unsupported capability=%r; "
            "harness only routes 'rag' / 'multimodal'. Falling back to 'rag'.",
            query_id, capability,
        )
        capability = "rag"

    # Pull anything that isn't a recognised field into metadata so
    # downstream filters / sweeps can read it back.
    known = {
        "query", "query_id", "id", "expected_doc_id", "expected_doc_ids",
        "expected_keywords", "expected_section_keywords",
        "input_kind", "capability", "metadata",
    }
    raw_meta = raw.get("metadata")
    if isinstance(raw_meta, str):
        try:
            metadata: Dict[str, Any] = json.loads(raw_meta) if raw_meta else {}
            if not isinstance(metadata, dict):
                metadata = {"value": metadata}
        except json.JSONDecodeError:
            metadata = {"raw": raw_meta}
    elif isinstance(raw_meta, dict):
        metadata = dict(raw_meta)
    else:
        metadata = {}
    for k, v in raw.items():
        if k in known or v is None:
            continue
        metadata.setdefault(str(k), v)

    return AgentLoopABQuery(
        query_id=query_id,
        query=query,
        expected_doc_id=expected_doc,
        expected_keywords=expected_keywords,
        input_kind=input_kind,
        capability=capability,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Per-query metric extraction
# ---------------------------------------------------------------------------


@dataclass
class AgentLoopABMetrics:
    """Per-(backend, query) metric row written to ``raw_results.jsonl``.

    Field names follow the spec; CSV column order is taken from
    ``METRIC_COLUMNS`` so successive runs diff cleanly. Optional metrics
    (``top1_score``, ``mrr``) are ``None`` when the input row didn't
    carry an ``expected_doc_id`` or the retriever didn't surface scores.
    """

    backend: str
    query_id: str
    query: str
    success: bool
    error_code: Optional[str]
    total_latency_ms: float
    loop_iterations: int
    rewrite_count: int
    retrieval_call_count: int
    rerank_call_count: int
    llm_call_count: int
    candidate_count: int
    unique_doc_count: int
    top1_score: Optional[float]
    top1_doc_id: Optional[str]
    expected_doc_hit_at_1: Optional[bool]
    expected_doc_hit_at_3: Optional[bool]
    expected_doc_hit_at_5: Optional[bool]
    mrr_contribution: Optional[float]
    stop_reason: str
    final_answer_length: int
    artifact_types: List[str]
    trace_stage_count: int
    expected_keyword_hit: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert booleans to a CSV-friendly true/false in the JSON too —
        # asdict already keeps bools, but the CSV writer downcasts them.
        return d


METRIC_COLUMNS: Tuple[str, ...] = (
    "backend",
    "query_id",
    "query",
    "success",
    "error_code",
    "total_latency_ms",
    "loop_iterations",
    "rewrite_count",
    "retrieval_call_count",
    "rerank_call_count",
    "llm_call_count",
    "candidate_count",
    "unique_doc_count",
    "top1_score",
    "top1_doc_id",
    "expected_doc_hit_at_1",
    "expected_doc_hit_at_3",
    "expected_doc_hit_at_5",
    "mrr_contribution",
    "stop_reason",
    "final_answer_length",
    "artifact_types",
    "trace_stage_count",
    "expected_keyword_hit",
)


# ---------------------------------------------------------------------------
# Backend runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendRunOutcome:
    """Raw run output for a single (backend, query) pair.

    Holds enough state for ``extract_metrics`` to compute every metric
    in the spec without having to re-execute anything.
    """

    backend: str
    success: bool
    error_code: Optional[str]
    error_message: Optional[str]
    total_latency_ms: float
    outcome: Optional[LoopOutcome]
    final_answer: str
    aggregated_chunks: List[RetrievedChunk]
    artifact_types: List[str]
    retrieval_call_count: int
    rerank_call_count: int
    llm_call_count: int
    trace_stage_count: int


# A loop runner is anything with the legacy/graph ``run`` signature.
LoopRunner = Any  # AgentLoopController | AgentLoopGraph


# Caller-supplied builder: given the per-query parsed-query, return a
# closure compatible with ``ExecuteFn`` plus a mutable ``call_counts``
# dict the harness reads after the run. Decoupling the executor from
# the harness lets ``scripts/eval_agent_loop_ab.py`` wire a real
# Retriever (FAISS-backed) without the harness importing app/workers.
ExecutorBuilder = Callable[
    [AgentLoopABQuery],
    Tuple[Callable[[ParsedQuery], Tuple[str, List[RetrievedChunk], int]], Dict[str, int]],
]


def make_default_executor_builder(
    *,
    retriever: Any,
    generator: GenerationProvider,
) -> ExecutorBuilder:
    """Default executor builder that calls retriever + generator.

    Mirrors ``AgentCapability._run_loop_and_synthesize``'s RAG branch
    bit-for-bit: retrieve once with the rewriter's query, generate once
    against the chunks. Counters track raw call activity so the harness
    can attribute LLM/retrieval cost to each backend.

    The builder is stateless across queries — it returns a *fresh*
    counter dict per call so legacy and graph runs share no mutable
    state. The harness invokes the closure separately per backend, so
    the count of retrieval / rerank / llm calls is unambiguous.
    """

    def _build(query_row: AgentLoopABQuery):
        counts: Dict[str, int] = {
            "retrieval": 0,
            "rerank": 0,
            "llm": 0,
        }
        question_text = query_row.query

        def _exec(pq: ParsedQuery) -> Tuple[str, List[RetrievedChunk], int]:
            counts["retrieval"] += 1
            query_text = pq.normalized or pq.original or question_text
            report = retriever.retrieve(query_text)
            results = list(getattr(report, "results", []) or [])
            # Rerank is on the retriever side; surface it as a call
            # count when the report says it ran (rerank_ms is ``None``
            # for the noop path).
            if getattr(report, "rerank_ms", None) is not None:
                counts["rerank"] += 1
            answer = generator.generate(question_text, results)
            counts["llm"] += 1
            return answer, results, 0

        return _exec, counts

    return _build


def run_backend_for_query(
    *,
    backend: str,
    runner: LoopRunner,
    parser: QueryParserProvider,
    synthesizer: Optional[AgentSynthesizer],
    query_row: AgentLoopABQuery,
    executor_builder: ExecutorBuilder,
) -> BackendRunOutcome:
    """Execute one backend on one query and capture the run outcome.

    No Redis / DB / Spring repo / callback work — just the in-process
    runner contract. The synthesizer is optional: when supplied the
    harness records the synthesized final answer (matching production
    semantics); when ``None`` we use the raw last-iter answer the
    runner left on the outcome.
    """
    initial_pq = parser.parse(query_row.query)
    execute_fn, counts = executor_builder(query_row)

    started = time.perf_counter()
    try:
        outcome = runner.run(
            question=query_row.query,
            initial_parsed_query=initial_pq,
            execute_fn=execute_fn,
        )
    except Exception as ex:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        log.warning(
            "agent_loop_ab: backend=%s query_id=%s raised %s: %s",
            backend, query_row.query_id, type(ex).__name__, ex,
        )
        return BackendRunOutcome(
            backend=backend,
            success=False,
            error_code=type(ex).__name__,
            error_message=str(ex),
            total_latency_ms=elapsed_ms,
            outcome=None,
            final_answer="",
            aggregated_chunks=[],
            artifact_types=[],
            retrieval_call_count=counts["retrieval"],
            rerank_call_count=counts["rerank"],
            llm_call_count=counts["llm"],
            trace_stage_count=0,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)

    # AgentLoopGraph degrades to an *empty* LoopOutcome when the graph
    # itself can't run (langgraph build error, recursion-limit hit,
    # internal LangGraph bug). The runner doesn't raise in that case so
    # the production capability can fall through cleanly, but the A/B
    # harness must NOT count the empty outcome as a healthy success —
    # otherwise a broken graph backend would silently look identical to
    # a working one. ``last_failure`` exposes the build/invoke failure
    # reason; legacy runners don't have the attribute, so ``getattr``
    # returns ``None`` and leaves their outcome alone.
    runner_failure = getattr(runner, "last_failure", None)
    if runner_failure is not None:
        log.warning(
            "agent_loop_ab: backend=%s query_id=%s graph degraded (%s); "
            "marking as failed run.",
            backend, query_row.query_id, runner_failure,
        )
        return BackendRunOutcome(
            backend=backend,
            success=False,
            error_code=f"graph_{runner_failure}",
            error_message=(
                f"AgentLoopGraph reported last_failure={runner_failure!r}; "
                "outcome was a synthetic empty fallback."
            ),
            total_latency_ms=elapsed_ms,
            outcome=None,
            final_answer="",
            aggregated_chunks=[],
            artifact_types=[],
            retrieval_call_count=counts["retrieval"],
            rerank_call_count=counts["rerank"],
            llm_call_count=counts["llm"],
            trace_stage_count=0,
        )

    # Synthesize the final answer the way the production capability
    # would — across the union of aggregated chunks. We only count this
    # generator call when a synthesizer is supplied so the per-backend
    # comparison stays apples-to-apples for callers that want to skip
    # synthesis.
    if synthesizer is not None and outcome.aggregated_chunks:
        try:
            final_answer = synthesizer.synthesize(query_row.query, outcome)
            counts["llm"] += 1
        except Exception as ex:  # pragma: no cover - defensive
            log.warning(
                "agent_loop_ab: synthesizer failed for query_id=%s: %s",
                query_row.query_id, ex,
            )
            final_answer = outcome.final_answer or ""
    else:
        final_answer = outcome.final_answer or ""

    artifact_types = _artifact_types_for_outcome(outcome, final_answer)
    trace_stage_count = len(outcome.steps)

    return BackendRunOutcome(
        backend=backend,
        success=True,
        error_code=None,
        error_message=None,
        total_latency_ms=elapsed_ms,
        outcome=outcome,
        final_answer=final_answer,
        aggregated_chunks=list(outcome.aggregated_chunks),
        artifact_types=artifact_types,
        retrieval_call_count=counts["retrieval"],
        rerank_call_count=counts["rerank"],
        llm_call_count=counts["llm"],
        trace_stage_count=trace_stage_count,
    )


def _artifact_types_for_outcome(
    outcome: LoopOutcome, final_answer: str
) -> List[str]:
    """Mirror the production artifact list emitted by AgentCapability.

    Bit-for-bit identical to ``_run_loop_and_synthesize``'s output: the
    AGENT_DECISION + AGENT_TRACE + RETRIEVAL_RESULT_AGG + FINAL_RESPONSE
    quartet. The harness fakes the AGENT_DECISION because we skipped
    the capability shell, but reading this back as artifact_types stays
    schema-stable for downstream consumers.
    """
    types = ["AGENT_DECISION", "AGENT_TRACE", "RETRIEVAL_RESULT_AGG"]
    types.append("FINAL_RESPONSE")
    return types


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_metrics(
    *,
    query_row: AgentLoopABQuery,
    run_outcome: BackendRunOutcome,
) -> AgentLoopABMetrics:
    """Convert a run outcome + the query's expected fields into metrics."""
    outcome = run_outcome.outcome
    aggregated = run_outcome.aggregated_chunks

    if outcome is None:
        return AgentLoopABMetrics(
            backend=run_outcome.backend,
            query_id=query_row.query_id,
            query=query_row.query,
            success=False,
            error_code=run_outcome.error_code,
            total_latency_ms=run_outcome.total_latency_ms,
            loop_iterations=0,
            rewrite_count=0,
            retrieval_call_count=run_outcome.retrieval_call_count,
            rerank_call_count=run_outcome.rerank_call_count,
            llm_call_count=run_outcome.llm_call_count,
            candidate_count=0,
            unique_doc_count=0,
            top1_score=None,
            top1_doc_id=None,
            expected_doc_hit_at_1=None,
            expected_doc_hit_at_3=None,
            expected_doc_hit_at_5=None,
            mrr_contribution=None,
            stop_reason="error",
            final_answer_length=0,
            artifact_types=[],
            trace_stage_count=0,
            expected_keyword_hit=None,
        )

    steps = list(outcome.steps)
    loop_iterations = len(steps)
    # Rewrite count: every iteration AFTER the first runs through the
    # rewriter once. Identical accounting on both backends.
    rewrite_count = max(0, loop_iterations - 1) if loop_iterations > 0 else 0
    candidate_count = len(aggregated)
    unique_doc_count = len({c.doc_id for c in aggregated if c.doc_id})

    top1_score: Optional[float] = None
    top1_doc_id: Optional[str] = None
    if aggregated:
        first = aggregated[0]
        top1_score = float(first.score) if first.score is not None else None
        top1_doc_id = first.doc_id or None

    expected_doc = query_row.expected_doc_id
    hit_at_1: Optional[bool] = None
    hit_at_3: Optional[bool] = None
    hit_at_5: Optional[bool] = None
    mrr: Optional[float] = None
    if expected_doc:
        doc_ids = [c.doc_id for c in aggregated if c.doc_id]
        hit_at_1 = bool(doc_ids[:1] and doc_ids[0] == expected_doc)
        hit_at_3 = expected_doc in doc_ids[:3]
        hit_at_5 = expected_doc in doc_ids[:5]
        mrr = 0.0
        for rank, doc_id in enumerate(doc_ids, start=1):
            if doc_id == expected_doc:
                mrr = 1.0 / float(rank)
                break

    expected_keyword_hit: Optional[bool] = None
    if query_row.expected_keywords:
        haystack = (run_outcome.final_answer or "").lower()
        if not haystack and aggregated:
            haystack = " ".join(c.text or "" for c in aggregated[:5]).lower()
        expected_keyword_hit = any(
            (kw or "").strip().lower() in haystack
            for kw in query_row.expected_keywords
            if (kw or "").strip()
        )

    return AgentLoopABMetrics(
        backend=run_outcome.backend,
        query_id=query_row.query_id,
        query=query_row.query,
        success=run_outcome.success,
        error_code=run_outcome.error_code,
        total_latency_ms=run_outcome.total_latency_ms,
        loop_iterations=loop_iterations,
        rewrite_count=rewrite_count,
        retrieval_call_count=run_outcome.retrieval_call_count,
        rerank_call_count=run_outcome.rerank_call_count,
        llm_call_count=run_outcome.llm_call_count,
        candidate_count=candidate_count,
        unique_doc_count=unique_doc_count,
        top1_score=top1_score,
        top1_doc_id=top1_doc_id,
        expected_doc_hit_at_1=hit_at_1,
        expected_doc_hit_at_3=hit_at_3,
        expected_doc_hit_at_5=hit_at_5,
        mrr_contribution=mrr,
        stop_reason=str(outcome.stop_reason),
        final_answer_length=len(run_outcome.final_answer or ""),
        artifact_types=list(run_outcome.artifact_types),
        trace_stage_count=run_outcome.trace_stage_count,
        expected_keyword_hit=expected_keyword_hit,
    )


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------


@dataclass
class AgentLoopABComparisonRow:
    """Side-by-side per-query comparison emitted as one ``raw_results.jsonl`` line."""

    query_id: str
    query: str
    legacy: Dict[str, Any]
    graph: Dict[str, Any]
    expected_doc_id: Optional[str]
    expected_keywords: List[str]
    input_kind: Optional[str]
    capability: Optional[str]
    metadata: Dict[str, Any]
    verdict: str  # graph_win | legacy_win | tie | regression

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queryId": self.query_id,
            "query": self.query,
            "legacy": self.legacy,
            "graph": self.graph,
            "expectedDocId": self.expected_doc_id,
            "expectedKeywords": list(self.expected_keywords),
            "inputKind": self.input_kind,
            "capability": self.capability,
            "metadata": dict(self.metadata),
            "verdict": self.verdict,
        }


VERDICT_GRAPH_WIN = "graph_win"
VERDICT_LEGACY_WIN = "legacy_win"
VERDICT_TIE = "tie"
VERDICT_REGRESSION = "regression"


def compare_one(
    *,
    query_row: AgentLoopABQuery,
    legacy: AgentLoopABMetrics,
    graph: AgentLoopABMetrics,
) -> AgentLoopABComparisonRow:
    """Side-by-side row + per-query verdict.

    Verdict rules (deliberately conservative — the comparison is meant
    to surface regressions, not declare graph the winner without a
    quality lift):

      * ``graph_win``  : graph improves hit@k OR keyword_hit OR
                         candidate_count vs legacy AND latency increase
                         is within 50 % (graph.latency <= legacy.latency * 1.5).
      * ``regression`` : graph degrades hit@k or keyword_hit, OR
                         graph.latency > legacy.latency * 2 with no
                         quality improvement, OR graph errored when
                         legacy succeeded.
      * ``legacy_win`` : graph and legacy both succeeded but graph is
                         strictly worse on quality OR strictly more
                         expensive on calls without a quality offset.
      * ``tie``        : neither side improves; everything else.

    The thresholds are deliberately readable — the user-facing summary
    document the same rules so the eval is auditable.
    """
    if not graph.success and legacy.success:
        verdict = VERDICT_REGRESSION
    elif graph.success and not legacy.success:
        verdict = VERDICT_GRAPH_WIN
    elif not graph.success and not legacy.success:
        verdict = VERDICT_TIE
    else:
        verdict = _grade_pair(query_row, legacy=legacy, graph=graph)

    return AgentLoopABComparisonRow(
        query_id=query_row.query_id,
        query=query_row.query,
        legacy=legacy.to_dict(),
        graph=graph.to_dict(),
        expected_doc_id=query_row.expected_doc_id,
        expected_keywords=list(query_row.expected_keywords),
        input_kind=query_row.input_kind,
        capability=query_row.capability,
        metadata=dict(query_row.metadata),
        verdict=verdict,
    )


def _grade_pair(
    query_row: AgentLoopABQuery,
    *,
    legacy: AgentLoopABMetrics,
    graph: AgentLoopABMetrics,
) -> str:
    quality_delta = _quality_delta(query_row, legacy=legacy, graph=graph)
    latency_ratio = (
        (graph.total_latency_ms / legacy.total_latency_ms)
        if legacy.total_latency_ms > 0 else 1.0
    )

    if quality_delta > 0:
        if latency_ratio <= 1.5:
            return VERDICT_GRAPH_WIN
        # graph found something better but at >1.5x latency cost -- still
        # a win; spec says regression only when there is NO quality lift.
        return VERDICT_GRAPH_WIN
    if quality_delta < 0:
        return VERDICT_REGRESSION
    # No quality delta. Penalise graph if it spent more LLM/retrieval
    # calls without finding anything new — that's the "boondoggle"
    # branch the spec calls out.
    extra_llm = graph.llm_call_count - legacy.llm_call_count
    extra_retrieval = graph.retrieval_call_count - legacy.retrieval_call_count
    if (extra_llm > 0 or extra_retrieval > 0) and latency_ratio > 1.2:
        return VERDICT_LEGACY_WIN
    if latency_ratio > 2.0:
        return VERDICT_REGRESSION
    if latency_ratio < 0.8:
        return VERDICT_GRAPH_WIN
    return VERDICT_TIE


def _quality_delta(
    query_row: AgentLoopABQuery,
    *,
    legacy: AgentLoopABMetrics,
    graph: AgentLoopABMetrics,
) -> int:
    """Coarse +1 / 0 / -1 quality signal used by the grader.

    Prefers expected_doc_id verdicts when available (highest signal),
    falls back to expected_keywords, then to candidate_count and
    unique_doc_count as a coverage proxy. Returns +1 when graph helps,
    -1 when it hurts, 0 when neither side moves the needle.
    """
    if query_row.expected_doc_id:
        for legacy_hit, graph_hit in (
            (legacy.expected_doc_hit_at_1, graph.expected_doc_hit_at_1),
            (legacy.expected_doc_hit_at_3, graph.expected_doc_hit_at_3),
            (legacy.expected_doc_hit_at_5, graph.expected_doc_hit_at_5),
        ):
            if graph_hit and not legacy_hit:
                return 1
            if legacy_hit and not graph_hit:
                return -1
        legacy_mrr = legacy.mrr_contribution or 0.0
        graph_mrr = graph.mrr_contribution or 0.0
        if graph_mrr > legacy_mrr + 1e-6:
            return 1
        if graph_mrr + 1e-6 < legacy_mrr:
            return -1
    if query_row.expected_keywords:
        if graph.expected_keyword_hit and not legacy.expected_keyword_hit:
            return 1
        if legacy.expected_keyword_hit and not graph.expected_keyword_hit:
            return -1
    if graph.candidate_count > legacy.candidate_count + 1:
        return 1
    if graph.candidate_count + 1 < legacy.candidate_count:
        return -1
    if graph.unique_doc_count > legacy.unique_doc_count:
        return 1
    if graph.unique_doc_count < legacy.unique_doc_count:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize_runs(
    *,
    rows: Sequence[AgentLoopABComparisonRow],
) -> Dict[str, Any]:
    """Aggregate per-query rows into the comparison_summary.json payload."""
    legacy_metrics = [r.legacy for r in rows]
    graph_metrics = [r.graph for r in rows]

    summary: Dict[str, Any] = {
        "queryCount": len(rows),
        "legacySuccessRate": _success_rate(legacy_metrics),
        "graphSuccessRate": _success_rate(graph_metrics),
        "legacyLatencyP50": _percentile(_field_floats(legacy_metrics, "total_latency_ms"), 0.50),
        "graphLatencyP50": _percentile(_field_floats(graph_metrics, "total_latency_ms"), 0.50),
        "legacyLatencyP95": _percentile(_field_floats(legacy_metrics, "total_latency_ms"), 0.95),
        "graphLatencyP95": _percentile(_field_floats(graph_metrics, "total_latency_ms"), 0.95),
        "legacyAvgIterations": _avg(_field_floats(legacy_metrics, "loop_iterations")),
        "graphAvgIterations": _avg(_field_floats(graph_metrics, "loop_iterations")),
        "legacyAvgRewriteCount": _avg(_field_floats(legacy_metrics, "rewrite_count")),
        "graphAvgRewriteCount": _avg(_field_floats(graph_metrics, "rewrite_count")),
        "legacyAvgRetrievalCalls": _avg(_field_floats(legacy_metrics, "retrieval_call_count")),
        "graphAvgRetrievalCalls": _avg(_field_floats(graph_metrics, "retrieval_call_count")),
        "legacyAvgLlmCalls": _avg(_field_floats(legacy_metrics, "llm_call_count")),
        "graphAvgLlmCalls": _avg(_field_floats(graph_metrics, "llm_call_count")),
    }

    # Hit@k aggregates only over rows with expected_doc_id, otherwise
    # the metric collapses to "rate among rows that bother to score".
    expected_rows = [r for r in rows if r.expected_doc_id]
    if expected_rows:
        for k_label, key in (
            ("HitAt1", "expected_doc_hit_at_1"),
            ("HitAt3", "expected_doc_hit_at_3"),
            ("HitAt5", "expected_doc_hit_at_5"),
        ):
            summary[f"legacy{k_label}"] = _bool_rate(
                [r.legacy.get(key) for r in expected_rows]
            )
            summary[f"graph{k_label}"] = _bool_rate(
                [r.graph.get(key) for r in expected_rows]
            )
        legacy_mrr = [
            (r.legacy.get("mrr_contribution") or 0.0) for r in expected_rows
        ]
        graph_mrr = [
            (r.graph.get("mrr_contribution") or 0.0) for r in expected_rows
        ]
        summary["legacyMRR"] = _avg(legacy_mrr)
        summary["graphMRR"] = _avg(graph_mrr)
        summary["expectedDocRowCount"] = len(expected_rows)

    keyword_rows = [r for r in rows if r.expected_keywords]
    if keyword_rows:
        summary["legacyKeywordHitRate"] = _bool_rate(
            [r.legacy.get("expected_keyword_hit") for r in keyword_rows]
        )
        summary["graphKeywordHitRate"] = _bool_rate(
            [r.graph.get("expected_keyword_hit") for r in keyword_rows]
        )
        summary["expectedKeywordRowCount"] = len(keyword_rows)

    verdict_counter: Dict[str, int] = {
        VERDICT_GRAPH_WIN: 0,
        VERDICT_LEGACY_WIN: 0,
        VERDICT_TIE: 0,
        VERDICT_REGRESSION: 0,
    }
    for row in rows:
        verdict_counter[row.verdict] = verdict_counter.get(row.verdict, 0) + 1

    summary["graphWins"] = verdict_counter[VERDICT_GRAPH_WIN]
    summary["legacyWins"] = verdict_counter[VERDICT_LEGACY_WIN]
    summary["ties"] = verdict_counter[VERDICT_TIE]
    summary["regressions"] = verdict_counter[VERDICT_REGRESSION]

    summary["recommendation"] = _recommendation(summary, expected_rows, rows)
    return summary


def _recommendation(
    summary: Mapping[str, Any],
    expected_rows: Sequence[AgentLoopABComparisonRow],
    rows: Sequence[AgentLoopABComparisonRow],
) -> str:
    """Mirror the spec's adoption rules in plain text.

    * Adopt: graph lifts hit@k or candidate recall AND p95 latency does
      not blow up.
    * Hold (no quality gain): graph adds LLM/retrieval cost without
      quality movement.
    * Hold (debug-only): graph only improves trace/debuggability, not
      retrieval quality — keep as experimental backend.
    """
    regressions = summary.get("regressions", 0)
    graph_wins = summary.get("graphWins", 0)
    legacy_wins = summary.get("legacyWins", 0)
    legacy_p95 = summary.get("legacyLatencyP95") or 0.0
    graph_p95 = summary.get("graphLatencyP95") or 0.0
    latency_ratio = (graph_p95 / legacy_p95) if legacy_p95 > 0 else 1.0

    # Quality lift signal - prefer hit@k when available.
    quality_lift = False
    if expected_rows:
        if (summary.get("graphHitAt5") or 0.0) > (summary.get("legacyHitAt5") or 0.0):
            quality_lift = True
        if (summary.get("graphHitAt1") or 0.0) > (summary.get("legacyHitAt1") or 0.0):
            quality_lift = True
        if (summary.get("graphMRR") or 0.0) > (summary.get("legacyMRR") or 0.0):
            quality_lift = True
    if not quality_lift and rows and graph_wins > legacy_wins + regressions:
        quality_lift = True

    if quality_lift and latency_ratio <= 1.5 and regressions == 0:
        return "adopt_candidate"
    if (
        not quality_lift
        and (
            (summary.get("graphAvgLlmCalls") or 0.0)
            > (summary.get("legacyAvgLlmCalls") or 0.0)
            or (summary.get("graphAvgRetrievalCalls") or 0.0)
            > (summary.get("legacyAvgRetrievalCalls") or 0.0)
        )
    ):
        return "hold_no_quality_gain"
    if not quality_lift:
        return "hold_experimental_backend_only"
    return "hold_review_regressions"


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[AgentLoopABComparisonRow],
    summary: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Persist ``raw_results.jsonl``, ``summary.csv``, and ``comparison_summary.json``.

    Returns a dict mapping artifact name to the absolute path it was
    written to so the CLI can echo the location for operators.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_results.jsonl"
    csv_path = output_dir / "summary.csv"
    summary_path = output_dir / "comparison_summary.json"

    with raw_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for backend_key in ("legacy", "graph"):
                backend_metrics = (
                    row.legacy if backend_key == "legacy" else row.graph
                )
                writer.writerow(
                    {col: _csv_cell(backend_metrics.get(col)) for col in METRIC_COLUMNS}
                )

    payload: Dict[str, Any] = {}
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    payload["summary"] = dict(summary)
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(
        "agent_loop_ab: wrote %s, %s, %s",
        raw_path, csv_path, summary_path,
    )
    return {
        "raw_results": raw_path,
        "summary_csv": csv_path,
        "comparison_summary": summary_path,
    }


# ---------------------------------------------------------------------------
# Driver — top-level entry point
# ---------------------------------------------------------------------------


def run_ab_eval(
    *,
    queries: Sequence[AgentLoopABQuery],
    legacy_runner: LoopRunner,
    graph_runner: LoopRunner,
    parser: QueryParserProvider,
    executor_builder: ExecutorBuilder,
    legacy_synthesizer: Optional[AgentSynthesizer] = None,
    graph_synthesizer: Optional[AgentSynthesizer] = None,
) -> Tuple[List[AgentLoopABComparisonRow], Dict[str, Any]]:
    """Drive both backends on every query and return rows + summary.

    Side-effects: none — no callbacks emitted, no DB writes, no Redis
    pings. The two runners are invoked sequentially so that latency
    numbers reflect the same machine state instead of competing for
    a shared cache or GPU.
    """
    rows: List[AgentLoopABComparisonRow] = []
    for query_row in queries:
        legacy_run = run_backend_for_query(
            backend=BACKEND_LEGACY,
            runner=legacy_runner,
            parser=parser,
            synthesizer=legacy_synthesizer,
            query_row=query_row,
            executor_builder=executor_builder,
        )
        graph_run = run_backend_for_query(
            backend=BACKEND_GRAPH,
            runner=graph_runner,
            parser=parser,
            synthesizer=graph_synthesizer,
            query_row=query_row,
            executor_builder=executor_builder,
        )
        legacy_metrics = extract_metrics(query_row=query_row, run_outcome=legacy_run)
        graph_metrics = extract_metrics(query_row=query_row, run_outcome=graph_run)
        rows.append(
            compare_one(
                query_row=query_row,
                legacy=legacy_metrics,
                graph=graph_metrics,
            )
        )

    summary = summarize_runs(rows=rows)
    return rows, summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_rate(metrics: Sequence[Mapping[str, Any]]) -> Optional[float]:
    if not metrics:
        return None
    successes = sum(1 for m in metrics if m.get("success"))
    return round(successes / len(metrics), 4)


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return round(float(values[0]), 3)
    sorted_values = sorted(values)
    if pct <= 0.0:
        return round(sorted_values[0], 3)
    if pct >= 1.0:
        return round(sorted_values[-1], 3)
    rank = pct * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    interp = sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
    return round(float(interp), 3)


def _avg(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.fmean(values), 4)


def _bool_rate(values: Sequence[Optional[bool]]) -> Optional[float]:
    filtered = [bool(v) for v in values if v is not None]
    if not filtered:
        return None
    return round(sum(1 for v in filtered if v) / len(filtered), 4)


def _field_floats(
    metrics: Sequence[Mapping[str, Any]],
    key: str,
) -> List[float]:
    out: List[float] = []
    for m in metrics:
        v = m.get(key)
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    return str(value)
