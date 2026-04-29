"""Offline A/B harness tests for the agent-loop graph-vs-legacy comparison.

Covers:

  1. Input loaders (JSONL + CSV) and the optional-field schema.
  2. Stub-backed end-to-end run that exercises both legacy and graph
     backends (skipped when langgraph isn't installed).
  3. Metric extraction + per-query verdict logic.
  4. Aggregate summary numbers (success rate, p50/p95, hit@k).
  5. Output writer produces the three artifacts in the right shape.
  6. Side-effect contract: no Redis / DB / callback / Spring imports
     are reachable from the harness module.

The tests deliberately avoid the LangGraph backend in unit-only mode
when langgraph isn't installed — the CLI uses ``importorskip`` for the
end-to-end run, but the harness module itself stays importable.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from app.capabilities.agent.critic import NoOpCritic, RuleCritic
from app.capabilities.agent.loop import (
    AgentLoopController,
    LoopBudget,
    LoopOutcome,
)
from app.capabilities.agent.rewriter import NoOpQueryRewriter
from app.capabilities.rag.generation import (
    ExtractiveGenerator,
    GenerationProvider,
    RetrievedChunk,
)
from app.capabilities.rag.query_parser import (
    ParsedQuery,
    RegexQueryParser,
)
from eval.harness.agent_loop_ab import (
    AgentLoopABComparisonRow,
    AgentLoopABMetrics,
    AgentLoopABQuery,
    BACKEND_GRAPH,
    BACKEND_LEGACY,
    BackendRunOutcome,
    METRIC_COLUMNS,
    QUALITY_LIFT_EPSILON,
    RECOMMENDATION_ADOPT,
    RECOMMENDATION_HOLD_EXPERIMENTAL,
    RECOMMENDATION_HOLD_NO_QUALITY_GAIN,
    RECOMMENDATION_HOLD_REVIEW_ERRORS,
    RECOMMENDATION_HOLD_REVIEW_REGRESSIONS,
    VERDICT_GRAPH_WIN,
    VERDICT_LEGACY_WIN,
    VERDICT_REGRESSION,
    VERDICT_TIE,
    compare_one,
    extract_metrics,
    has_quality_lift,
    load_query_rows,
    make_default_executor_builder,
    run_ab_eval,
    run_backend_for_query,
    summarize_runs,
    write_outputs,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubRetriever:
    """Returns a fixed chunk list per query — deterministic across runs."""

    def __init__(self, by_query: Dict[str, List[RetrievedChunk]]) -> None:
        self._by_query = by_query
        self.calls: List[str] = []

    def retrieve(self, query: str):
        self.calls.append(query)

        class _Report:
            def __init__(self, results):
                self.results = results
                self.rerank_ms = None

        return _Report(list(self._by_query.get(query, [])))


class _RecordingGenerator(GenerationProvider):
    def __init__(self, prefix: str = "answer") -> None:
        self._prefix = prefix
        self.calls: List[Tuple[str, int]] = []

    @property
    def name(self) -> str:
        return "recording"

    def generate(self, query, chunks):
        self.calls.append((query, len(chunks)))
        return f"{self._prefix}: {query} ({len(chunks)} chunks)"


def _chunk(cid: str, doc: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id=doc, section="", text=f"text for {cid}", score=score,
    )


def _runner_legacy(critic=None, rewriter=None, parser=None, budget=None):
    return AgentLoopController(
        critic=critic or NoOpCritic(),
        rewriter=rewriter or NoOpQueryRewriter(),
        parser=parser or RegexQueryParser(),
        budget=budget or LoopBudget(max_iter=2),
    )


# ---------------------------------------------------------------------------
# 1. Input loading: JSONL + CSV
# ---------------------------------------------------------------------------


def test_load_jsonl_basic(tmp_path: Path):
    src = tmp_path / "queries.jsonl"
    src.write_text(
        "\n".join([
            json.dumps({"query": "foo", "expected_doc_id": "d1"}),
            "# comment line ignored",
            "",
            json.dumps({
                "query_id": "q-2", "query": "bar",
                "expected_keywords": ["a", "b"],
                "input_kind": "TEXT",
                "capability": "rag",
                "metadata": {"lang": "ko"},
            }),
        ]),
        encoding="utf-8",
    )
    rows = load_query_rows(src)
    assert len(rows) == 2
    assert rows[0].query_id == "q-0001"
    assert rows[0].query == "foo"
    assert rows[0].expected_doc_id == "d1"
    assert rows[1].query_id == "q-2"
    assert rows[1].expected_keywords == ["a", "b"]
    assert rows[1].metadata == {"lang": "ko"}
    assert rows[1].capability == "rag"


def test_load_csv_basic(tmp_path: Path):
    src = tmp_path / "queries.csv"
    src.write_text(
        "query_id,query,expected_doc_id,expected_keywords,input_kind,capability\n"
        "csv-1,what is bge?,doc-7,bge|embedding,TEXT,rag\n"
        "csv-2,foo,,,TEXT,multimodal\n",
        encoding="utf-8",
    )
    rows = load_query_rows(src)
    assert [r.query_id for r in rows] == ["csv-1", "csv-2"]
    assert rows[0].expected_keywords == ["bge", "embedding"]
    assert rows[1].expected_doc_id is None
    assert rows[1].capability == "multimodal"


def test_load_jsonl_unknown_capability_falls_back_to_rag(tmp_path: Path):
    src = tmp_path / "q.jsonl"
    src.write_text(
        json.dumps({"query": "hi", "capability": "ocr"}) + "\n",
        encoding="utf-8",
    )
    rows = load_query_rows(src)
    assert rows[0].capability == "rag"  # downgrade from unsupported 'ocr'


def test_load_jsonl_rejects_blank_query(tmp_path: Path):
    src = tmp_path / "q.jsonl"
    src.write_text(json.dumps({"query": "  "}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no usable 'query'"):
        load_query_rows(src)


def test_load_jsonl_rejects_invalid_json(tmp_path: Path):
    src = tmp_path / "q.jsonl"
    src.write_text("{not valid", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_query_rows(src)


def test_load_csv_rejects_missing_query_column(tmp_path: Path):
    src = tmp_path / "q.csv"
    src.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must have a 'query' column"):
        load_query_rows(src)


def test_load_jsonl_extra_fields_land_in_metadata(tmp_path: Path):
    src = tmp_path / "q.jsonl"
    src.write_text(
        json.dumps({"query": "hi", "domain": "anime", "language": "ko"}) + "\n",
        encoding="utf-8",
    )
    rows = load_query_rows(src)
    assert rows[0].metadata == {"domain": "anime", "language": "ko"}


def test_load_jsonl_silver200_schema_fallback(tmp_path: Path):
    """``expected_doc_ids`` (plural) + ``expected_section_keywords``
    are accepted as schema-compatible aliases of the singular fields,
    so silver_200-style datasets get per-row hit@k / keyword metrics
    without a separate projection step.
    """
    src = tmp_path / "q.jsonl"
    src.write_text(
        "\n".join([
            json.dumps({
                "id": "anime-silver-0001",
                "query": "템플의 주요 주제",
                "expected_doc_ids": ["e84b316e", "fallback-id"],
                "expected_section_keywords": ["다양한", "템플"],
                "answer_type": "theme_genre",
                "difficulty": "medium",
            }),
            json.dumps({
                "id": "no-keywords",
                "query": "without keywords",
                "expected_doc_ids": [],
            }),
        ]),
        encoding="utf-8",
    )
    rows = load_query_rows(src)
    assert len(rows) == 2
    # First row pulls the *first* id from expected_doc_ids and the
    # silver_200 keyword list; bookkeeping fields land in metadata.
    assert rows[0].query_id == "anime-silver-0001"
    assert rows[0].expected_doc_id == "e84b316e"
    assert rows[0].expected_keywords == ["다양한", "템플"]
    assert rows[0].metadata.get("answer_type") == "theme_genre"
    assert rows[0].metadata.get("difficulty") == "medium"
    # ``expected_doc_ids`` / ``expected_section_keywords`` themselves
    # are recognised fields and must NOT leak into metadata.
    assert "expected_doc_ids" not in rows[0].metadata
    assert "expected_section_keywords" not in rows[0].metadata
    # Empty plural list keeps expected_doc_id None; no fallback noise.
    assert rows[1].expected_doc_id is None
    assert rows[1].expected_keywords == []


def test_load_jsonl_singular_doc_id_wins_over_plural(tmp_path: Path):
    """When BOTH schemas are present, the singular field wins so
    operators can override the silver_200 fallback inline.
    """
    src = tmp_path / "q.jsonl"
    src.write_text(
        json.dumps({
            "query": "hi",
            "expected_doc_id": "primary",
            "expected_doc_ids": ["override-me", "also-ignored"],
            "expected_keywords": ["primary-kw"],
            "expected_section_keywords": ["override-me"],
        }) + "\n",
        encoding="utf-8",
    )
    rows = load_query_rows(src)
    assert rows[0].expected_doc_id == "primary"
    assert rows[0].expected_keywords == ["primary-kw"]


# ---------------------------------------------------------------------------
# 2. Backend runner: legacy controller (no graph dependency)
# ---------------------------------------------------------------------------


def test_run_backend_legacy_records_calls():
    parser = RegexQueryParser()
    retriever = _StubRetriever({"bge-m3 model": [_chunk("c1", "doc-7")]})
    generator = _RecordingGenerator()
    builder = make_default_executor_builder(retriever=retriever, generator=generator)
    runner = _runner_legacy(critic=NoOpCritic(), parser=parser)
    query = AgentLoopABQuery(query_id="q1", query="bge-m3 model")

    result = run_backend_for_query(
        backend=BACKEND_LEGACY,
        runner=runner,
        parser=parser,
        synthesizer=None,
        query_row=query,
        executor_builder=builder,
    )

    assert result.success is True
    assert result.error_code is None
    # Stub retriever called once at iter 0; NoOp critic stops immediately.
    assert result.retrieval_call_count == 1
    assert result.llm_call_count == 1
    assert result.outcome is not None
    assert result.outcome.stop_reason == "converged"
    assert "AGENT_DECISION" in result.artifact_types
    assert "FINAL_RESPONSE" in result.artifact_types


def test_run_backend_records_executor_failure():
    """An exception inside execute_fn must surface as success=False."""
    class _BoomRetriever:
        def retrieve(self, q):
            raise RuntimeError("retriever exploded")

    parser = RegexQueryParser()
    builder = make_default_executor_builder(
        retriever=_BoomRetriever(), generator=_RecordingGenerator(),
    )
    runner = _runner_legacy(parser=parser)
    query = AgentLoopABQuery(query_id="q-boom", query="anything")

    result = run_backend_for_query(
        backend=BACKEND_LEGACY,
        runner=runner,
        parser=parser,
        synthesizer=None,
        query_row=query,
        executor_builder=builder,
    )
    # Legacy controller swallows the iter-0 raise via its caller — it
    # actually re-raises, so the harness records the error outcome.
    assert result.success is False
    assert result.outcome is None
    assert result.error_code == "RuntimeError"
    assert result.retrieval_call_count >= 1


def test_run_backend_marks_graph_last_failure_as_failed_run():
    """Graph backends that report ``last_failure`` after ``run`` must be
    marked success=False — the spec calls this out: build/invoke faults
    are silently degraded into an empty ``LoopOutcome`` so the production
    capability can fall through cleanly, but the A/B harness must not
    count those as healthy successes.
    """
    from app.capabilities.agent.loop import LoopOutcome

    class _DegradedGraphRunner:
        """Mimics ``AgentLoopGraph`` returning a synthetic empty outcome.

        Sets ``last_failure='build_failed'`` exactly the way the real
        adapter does when ``build_agent_loop_graph`` raises.
        """

        def __init__(self) -> None:
            self.last_failure = None

        def run(self, *, question, initial_parsed_query, execute_fn):
            self.last_failure = "build_failed"
            return LoopOutcome(
                steps=[],
                stop_reason="iter_cap",
                final_answer="",
                aggregated_chunks=[],
                total_ms=0.0,
                total_llm_tokens=0,
            )

    parser = RegexQueryParser()
    builder = make_default_executor_builder(
        retriever=_StubRetriever({}), generator=_RecordingGenerator(),
    )
    runner = _DegradedGraphRunner()
    query = AgentLoopABQuery(query_id="q-degraded", query="anything")

    result = run_backend_for_query(
        backend=BACKEND_GRAPH,
        runner=runner,
        parser=parser,
        synthesizer=None,
        query_row=query,
        executor_builder=builder,
    )
    assert result.success is False
    assert result.error_code == "graph_build_failed"
    assert result.outcome is None
    assert result.aggregated_chunks == []


# ---------------------------------------------------------------------------
# 3. Metric extraction: hit@k, MRR, candidate_count
# ---------------------------------------------------------------------------


def test_extract_metrics_hit_at_1_when_expected_doc_first():
    query = AgentLoopABQuery(
        query_id="q1", query="hi", expected_doc_id="doc-7",
    )
    chunks = [_chunk("c1", "doc-7", score=0.95), _chunk("c2", "doc-3", score=0.4)]
    outcome = LoopOutcome(
        steps=[],  # filled separately to keep the test focused
        stop_reason="converged",
        final_answer="answer",
        aggregated_chunks=chunks,
        total_ms=12.5,
        total_llm_tokens=0,
    )
    run_outcome = BackendRunOutcome(
        backend=BACKEND_LEGACY,
        success=True,
        error_code=None,
        error_message=None,
        total_latency_ms=15.0,
        outcome=outcome,
        final_answer="answer",
        aggregated_chunks=chunks,
        artifact_types=["AGENT_DECISION"],
        retrieval_call_count=1,
        rerank_call_count=0,
        llm_call_count=1,
        trace_stage_count=0,
    )
    metrics = extract_metrics(query_row=query, run_outcome=run_outcome)
    assert metrics.expected_doc_hit_at_1 is True
    assert metrics.expected_doc_hit_at_3 is True
    assert metrics.expected_doc_hit_at_5 is True
    assert metrics.mrr_contribution == pytest.approx(1.0)
    assert metrics.candidate_count == 2
    assert metrics.unique_doc_count == 2
    assert metrics.top1_doc_id == "doc-7"
    assert metrics.top1_score == pytest.approx(0.95)


def test_extract_metrics_keyword_hit_against_final_answer():
    query = AgentLoopABQuery(
        query_id="q1", query="hi", expected_keywords=["bge"],
    )
    chunks = [_chunk("c1", "doc-1")]
    outcome = LoopOutcome(
        steps=[], stop_reason="converged",
        final_answer="bge-m3 is a multilingual embedding",
        aggregated_chunks=chunks,
        total_ms=0.0, total_llm_tokens=0,
    )
    run_outcome = BackendRunOutcome(
        backend=BACKEND_GRAPH, success=True, error_code=None, error_message=None,
        total_latency_ms=10.0, outcome=outcome,
        final_answer="bge-m3 is a multilingual embedding",
        aggregated_chunks=chunks,
        artifact_types=[], retrieval_call_count=1, rerank_call_count=0,
        llm_call_count=1, trace_stage_count=0,
    )
    metrics = extract_metrics(query_row=query, run_outcome=run_outcome)
    assert metrics.expected_keyword_hit is True


def test_extract_metrics_handles_error_outcome():
    query = AgentLoopABQuery(query_id="q1", query="hi")
    run_outcome = BackendRunOutcome(
        backend=BACKEND_LEGACY,
        success=False,
        error_code="RuntimeError",
        error_message="boom",
        total_latency_ms=1.0,
        outcome=None,
        final_answer="",
        aggregated_chunks=[],
        artifact_types=[],
        retrieval_call_count=0,
        rerank_call_count=0,
        llm_call_count=0,
        trace_stage_count=0,
    )
    metrics = extract_metrics(query_row=query, run_outcome=run_outcome)
    assert metrics.success is False
    assert metrics.stop_reason == "error"
    assert metrics.candidate_count == 0


# ---------------------------------------------------------------------------
# 4. Verdict logic
# ---------------------------------------------------------------------------


def _metrics(
    backend: str,
    *,
    success=True,
    latency=10.0,
    iters=1,
    rewrites=0,
    retr=1,
    rerank=0,
    llm=1,
    candidates=3,
    unique_docs=2,
    top1_score=0.9,
    top1_doc=None,
    h1=None,
    h3=None,
    h5=None,
    mrr=None,
    stop="converged",
    final_len=10,
    keyword_hit=None,
):
    return AgentLoopABMetrics(
        backend=backend,
        query_id="q",
        query="q",
        success=success,
        error_code=None,
        total_latency_ms=latency,
        loop_iterations=iters,
        rewrite_count=rewrites,
        retrieval_call_count=retr,
        rerank_call_count=rerank,
        llm_call_count=llm,
        candidate_count=candidates,
        unique_doc_count=unique_docs,
        top1_score=top1_score,
        top1_doc_id=top1_doc,
        expected_doc_hit_at_1=h1,
        expected_doc_hit_at_3=h3,
        expected_doc_hit_at_5=h5,
        mrr_contribution=mrr,
        stop_reason=stop,
        final_answer_length=final_len,
        artifact_types=[],
        trace_stage_count=iters,
        expected_keyword_hit=keyword_hit,
    )


def test_compare_one_graph_win_when_graph_finds_expected_doc():
    qrow = AgentLoopABQuery(query_id="q", query="q", expected_doc_id="doc-7")
    legacy = _metrics("legacy", h1=False, h3=False, h5=False, mrr=0.0, latency=10.0)
    graph = _metrics("graph", h1=True, h3=True, h5=True, mrr=1.0, latency=12.0)
    cmp = compare_one(query_row=qrow, legacy=legacy, graph=graph)
    assert cmp.verdict == VERDICT_GRAPH_WIN


def test_compare_one_regression_when_legacy_hits_but_graph_misses():
    qrow = AgentLoopABQuery(query_id="q", query="q", expected_doc_id="doc-7")
    legacy = _metrics("legacy", h1=True, h3=True, h5=True, mrr=1.0)
    graph = _metrics("graph", h1=False, h3=False, h5=False, mrr=0.0, iters=2, llm=2)
    cmp = compare_one(query_row=qrow, legacy=legacy, graph=graph)
    assert cmp.verdict == VERDICT_REGRESSION


def test_compare_one_regression_when_graph_errors():
    qrow = AgentLoopABQuery(query_id="q", query="q")
    legacy = _metrics("legacy")
    graph = _metrics("graph", success=False)
    cmp = compare_one(query_row=qrow, legacy=legacy, graph=graph)
    assert cmp.verdict == VERDICT_REGRESSION


def test_compare_one_legacy_win_when_graph_only_burns_extra_calls():
    qrow = AgentLoopABQuery(query_id="q", query="q")
    legacy = _metrics("legacy", llm=1, retr=1, latency=10.0)
    graph = _metrics("graph", llm=3, retr=3, latency=14.0)
    cmp = compare_one(query_row=qrow, legacy=legacy, graph=graph)
    assert cmp.verdict == VERDICT_LEGACY_WIN


def test_compare_one_tie_when_neither_moves():
    qrow = AgentLoopABQuery(query_id="q", query="q")
    legacy = _metrics("legacy", latency=10.0)
    graph = _metrics("graph", latency=11.0)
    cmp = compare_one(query_row=qrow, legacy=legacy, graph=graph)
    assert cmp.verdict == VERDICT_TIE


# ---------------------------------------------------------------------------
# 5. Aggregate summary numbers
# ---------------------------------------------------------------------------


def test_summarize_runs_basic_aggregates():
    rows = []
    for i, (legacy_h1, graph_h1) in enumerate([(True, True), (False, True), (False, False)]):
        qrow = AgentLoopABQuery(
            query_id=f"q-{i}", query="q", expected_doc_id="doc-7",
        )
        legacy = _metrics(
            "legacy", h1=legacy_h1, h3=legacy_h1, h5=legacy_h1,
            mrr=1.0 if legacy_h1 else 0.0, latency=10.0 + i,
        )
        graph = _metrics(
            "graph", h1=graph_h1, h3=graph_h1, h5=graph_h1,
            mrr=1.0 if graph_h1 else 0.0, latency=11.0 + i, iters=1, llm=1,
        )
        rows.append(compare_one(query_row=qrow, legacy=legacy, graph=graph))

    summary = summarize_runs(rows=rows)
    assert summary["queryCount"] == 3
    assert summary["legacyHitAt1"] == pytest.approx(1 / 3, abs=1e-3)
    assert summary["graphHitAt1"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["graphMRR"] >= summary["legacyMRR"]
    # Recommendation should pick adopt_candidate when graph helps with no
    # latency blowup and no regression.
    assert summary["recommendation"] in (
        "adopt_candidate", "hold_review_regressions",
    )


def test_summarize_runs_keyword_aggregate():
    rows = []
    for i, (legacy_kw, graph_kw) in enumerate([(False, True), (True, True)]):
        qrow = AgentLoopABQuery(
            query_id=f"q-{i}", query="q",
            expected_keywords=["bge"],
        )
        legacy = _metrics("legacy", keyword_hit=legacy_kw, latency=10.0)
        graph = _metrics("graph", keyword_hit=graph_kw, latency=10.0)
        rows.append(compare_one(query_row=qrow, legacy=legacy, graph=graph))
    summary = summarize_runs(rows=rows)
    assert summary["legacyKeywordHitRate"] == pytest.approx(0.5)
    assert summary["graphKeywordHitRate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. End-to-end driver against legacy + a fake "graph" runner
# ---------------------------------------------------------------------------


class _FakeGraphRunner:
    """Stand-in that mirrors AgentLoopGraph.run shape without langgraph."""

    def __init__(self, *, controller: AgentLoopController) -> None:
        self._controller = controller
        self.calls = 0

    def run(self, *, question, initial_parsed_query, execute_fn):
        self.calls += 1
        return self._controller.run(
            question=question,
            initial_parsed_query=initial_parsed_query,
            execute_fn=execute_fn,
        )


def test_run_ab_eval_writes_three_artifacts(tmp_path: Path):
    parser = RegexQueryParser()
    chunks = [_chunk("c1", "doc-7", 0.9), _chunk("c2", "doc-2", 0.5)]
    retriever = _StubRetriever({
        "FAISS 빌드": chunks,
        "bge-m3": [_chunk("c-bge", "doc-2", 0.7)],
    })
    generator = _RecordingGenerator()
    builder = make_default_executor_builder(retriever=retriever, generator=generator)

    legacy = _runner_legacy(critic=NoOpCritic(), parser=parser)
    graph = _FakeGraphRunner(controller=_runner_legacy(critic=NoOpCritic(), parser=parser))

    queries = [
        AgentLoopABQuery(query_id="q1", query="FAISS 빌드", expected_doc_id="doc-7"),
        AgentLoopABQuery(query_id="q2", query="bge-m3", expected_keywords=["bge"]),
    ]
    rows, summary = run_ab_eval(
        queries=queries,
        legacy_runner=legacy,
        graph_runner=graph,
        parser=parser,
        executor_builder=builder,
    )

    paths = write_outputs(
        output_dir=tmp_path / "run", rows=rows, summary=summary,
    )
    assert paths["raw_results"].exists()
    assert paths["summary_csv"].exists()
    assert paths["comparison_summary"].exists()

    # raw_results.jsonl: one JSON object per query with both backends.
    raw_lines = paths["raw_results"].read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2
    parsed = [json.loads(line) for line in raw_lines]
    assert {p["queryId"] for p in parsed} == {"q1", "q2"}
    assert "legacy" in parsed[0]
    assert "graph" in parsed[0]
    assert parsed[0]["verdict"] in (
        VERDICT_GRAPH_WIN, VERDICT_LEGACY_WIN, VERDICT_TIE, VERDICT_REGRESSION,
    )

    # summary.csv: two rows per query (legacy + graph), in METRIC_COLUMNS order.
    with paths["summary_csv"].open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        assert reader.fieldnames == list(METRIC_COLUMNS)
        csv_rows = list(reader)
    assert len(csv_rows) == 4  # 2 queries * 2 backends
    backends = {row["backend"] for row in csv_rows}
    assert backends == {"legacy", "graph"}

    # comparison_summary.json
    summary_obj = json.loads(paths["comparison_summary"].read_text(encoding="utf-8"))
    assert summary_obj["summary"]["queryCount"] == 2
    assert "graphWins" in summary_obj["summary"]
    assert "legacyWins" in summary_obj["summary"]
    assert "ties" in summary_obj["summary"]
    assert "regressions" in summary_obj["summary"]


def test_run_ab_eval_no_callbacks_or_redis_imports():
    """Sanity check the side-effect contract.

    The harness module must not pull in Redis / TaskRunner / Spring repo
    code paths. We assert by importing the harness fresh and confirming
    no module starting with the production-write prefixes leaked into
    sys.modules **as a result of the harness import**.
    """
    forbidden_prefixes = (
        "app.workers.taskrunner",
        "app.workers.callback",
        "app.clients.redis",
    )

    # Drop anything matching the forbidden prefixes in case earlier
    # tests pulled them in. Importing the harness here must not bring
    # them back.
    for mod in list(sys.modules):
        if mod.startswith(forbidden_prefixes):
            sys.modules.pop(mod, None)

    import importlib

    import eval.harness.agent_loop_ab as harness_module

    importlib.reload(harness_module)

    leaked = [
        m for m in sys.modules
        if m.startswith(forbidden_prefixes)
    ]
    assert leaked == [], (
        f"agent_loop_ab harness must not pull in production write modules; "
        f"leaked={leaked}"
    )


# ---------------------------------------------------------------------------
# 7. Integration: real AgentLoopGraph if langgraph is available
# ---------------------------------------------------------------------------


def test_run_ab_eval_with_real_graph_runner_smoke():
    """Round-trip through AgentLoopGraph + AgentLoopController on the
    same query set. Covers the production path the CLI takes.
    """
    pytest.importorskip("langgraph")
    from app.capabilities.agent.graph_loop import AgentLoopGraph

    parser = RegexQueryParser()
    chunks = [_chunk("c1", "doc-1", 0.9)]
    retriever = _StubRetriever({"hello": chunks})
    generator = ExtractiveGenerator()
    builder = make_default_executor_builder(retriever=retriever, generator=generator)

    legacy = AgentLoopController(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=2),
    )
    graph = AgentLoopGraph(
        critic=NoOpCritic(),
        rewriter=NoOpQueryRewriter(),
        parser=parser,
        budget=LoopBudget(max_iter=2),
    )
    queries = [AgentLoopABQuery(query_id="q1", query="hello", expected_doc_id="doc-1")]

    rows, summary = run_ab_eval(
        queries=queries,
        legacy_runner=legacy,
        graph_runner=graph,
        parser=parser,
        executor_builder=builder,
    )
    assert len(rows) == 1
    row = rows[0]
    # Schema parity: both backends must agree on hit@1 / candidate_count
    # for a deterministic run.
    assert row.legacy["expected_doc_hit_at_1"] == row.graph["expected_doc_hit_at_1"]
    assert row.legacy["candidate_count"] == row.graph["candidate_count"]
    assert summary["queryCount"] == 1


# ---------------------------------------------------------------------------
# 8. Conservative recommendation rule
# ---------------------------------------------------------------------------
#
# These tests pin down the priority order of the adoption rule. Each builds
# a synthetic ``rows`` list whose aggregate ``summarize_runs`` shape matches
# one of the spec's five cases, then asserts the recommendation label.
# A latency-only graph_win must NOT trigger ``adopt_candidate`` — that's
# the regression these tests guard against.


def _row_pair(
    *,
    query_id: str,
    expected_doc_id: str = None,
    expected_keywords=None,
    legacy: AgentLoopABMetrics,
    graph: AgentLoopABMetrics,
) -> AgentLoopABComparisonRow:
    qrow = AgentLoopABQuery(
        query_id=query_id,
        query="q",
        expected_doc_id=expected_doc_id,
        expected_keywords=list(expected_keywords or []),
    )
    return compare_one(query_row=qrow, legacy=legacy, graph=graph)


def test_recommendation_latency_only_graph_win_holds_experimental():
    """Case 1: quality identical, graphWins=1 latency-only, regressions=0
    must yield ``hold_experimental_backend_only``. This is the bug case
    where the previous rule promoted graph_wins > legacy_wins to a
    quality_lift and emitted ``adopt_candidate`` — the new rule must
    refuse adoption when no aggregate quality metric moved.
    """
    rows = []
    # 99 ties at identical hit@k.
    for i in range(99):
        legacy = _metrics(
            "legacy", h1=True, h3=True, h5=True, mrr=1.0, latency=1000.0,
        )
        graph = _metrics(
            "graph", h1=True, h3=True, h5=True, mrr=1.0, latency=1010.0,
        )
        rows.append(_row_pair(
            query_id=f"q-{i}", expected_doc_id="doc-7",
            legacy=legacy, graph=graph,
        ))
    # One latency-only graph_win: identical quality, graph noticeably faster.
    legacy = _metrics(
        "legacy", h1=True, h3=True, h5=True, mrr=1.0, latency=1000.0,
    )
    graph = _metrics(
        "graph", h1=True, h3=True, h5=True, mrr=1.0, latency=200.0,
    )
    rows.append(_row_pair(
        query_id="q-cold-start", expected_doc_id="doc-7",
        legacy=legacy, graph=graph,
    ))

    summary = summarize_runs(rows=rows)
    assert summary["graphWins"] >= 1
    assert summary["regressions"] == 0
    # Aggregate hit@k / MRR identical — no quality lift.
    assert summary["legacyHitAt1"] == summary["graphHitAt1"]
    assert summary["legacyHitAt5"] == summary["graphHitAt5"]
    assert summary["legacyMRR"] == summary["graphMRR"]
    assert summary["recommendation"] == RECOMMENDATION_HOLD_EXPERIMENTAL


def test_recommendation_hit_at_3_lift_within_latency_adopts():
    """Case 2: graph improves hit@3 (and MRR) by > epsilon, p95 latency
    stays inside the 1.5x ceiling, no regressions → ``adopt_candidate``.
    """
    rows = []
    # Half the queries: legacy misses, graph hits — drives a hit@3 lift.
    for i in range(20):
        if i < 10:
            legacy = _metrics(
                "legacy", h1=False, h3=False, h5=False, mrr=0.0, latency=1000.0,
            )
            graph = _metrics(
                "graph", h1=False, h3=True, h5=True, mrr=0.5, latency=1100.0,
            )
        else:
            legacy = _metrics(
                "legacy", h1=True, h3=True, h5=True, mrr=1.0, latency=1000.0,
            )
            graph = _metrics(
                "graph", h1=True, h3=True, h5=True, mrr=1.0, latency=1100.0,
            )
        rows.append(_row_pair(
            query_id=f"q-{i}", expected_doc_id="doc-7",
            legacy=legacy, graph=graph,
        ))

    summary = summarize_runs(rows=rows)
    assert summary["regressions"] == 0
    legacy_p95 = summary["legacyLatencyP95"]
    graph_p95 = summary["graphLatencyP95"]
    assert (graph_p95 / legacy_p95) <= 1.5
    assert summary["graphHitAt3"] > summary["legacyHitAt3"] + QUALITY_LIFT_EPSILON
    assert summary["recommendation"] == RECOMMENDATION_ADOPT


def test_recommendation_no_quality_gain_with_extra_calls():
    """Case 3: identical quality but graph spends more LLM/retrieval
    calls on average → ``hold_no_quality_gain``.
    """
    rows = []
    for i in range(10):
        legacy = _metrics(
            "legacy", h1=True, h3=True, h5=True, mrr=1.0,
            latency=1000.0, llm=1, retr=1,
        )
        graph = _metrics(
            "graph", h1=True, h3=True, h5=True, mrr=1.0,
            latency=1100.0, llm=2, retr=2,
        )
        rows.append(_row_pair(
            query_id=f"q-{i}", expected_doc_id="doc-7",
            legacy=legacy, graph=graph,
        ))

    summary = summarize_runs(rows=rows)
    assert summary["regressions"] == 0
    assert summary["graphAvgLlmCalls"] > summary["legacyAvgLlmCalls"]
    assert summary["graphAvgRetrievalCalls"] > summary["legacyAvgRetrievalCalls"]
    assert summary["legacyHitAt5"] == summary["graphHitAt5"]
    assert summary["recommendation"] == RECOMMENDATION_HOLD_NO_QUALITY_GAIN


def test_recommendation_regressions_take_priority():
    """Case 4: at least one per-query regression → ``hold_review_regressions``,
    even if the aggregate has a quality lift (the regression must be
    triaged before any adoption decision).
    """
    rows = []
    # Eight queries where graph improves quality.
    for i in range(8):
        legacy = _metrics(
            "legacy", h1=False, h3=False, h5=False, mrr=0.0, latency=1000.0,
        )
        graph = _metrics(
            "graph", h1=True, h3=True, h5=True, mrr=1.0, latency=1100.0,
        )
        rows.append(_row_pair(
            query_id=f"q-win-{i}", expected_doc_id="doc-7",
            legacy=legacy, graph=graph,
        ))
    # One query where graph regresses on hit@1 (legacy hits, graph misses).
    legacy = _metrics(
        "legacy", h1=True, h3=True, h5=True, mrr=1.0, latency=1000.0,
    )
    graph = _metrics(
        "graph", h1=False, h3=False, h5=False, mrr=0.0, latency=1000.0,
    )
    rows.append(_row_pair(
        query_id="q-regression", expected_doc_id="doc-7",
        legacy=legacy, graph=graph,
    ))

    summary = summarize_runs(rows=rows)
    assert summary["regressions"] >= 1
    # Aggregate quality may still lift; rule still holds for review.
    assert summary["recommendation"] == RECOMMENDATION_HOLD_REVIEW_REGRESSIONS


def test_recommendation_graph_error_rate_takes_top_priority():
    """Case 5: graph success rate is below legacy → ``hold_review_errors``,
    even if regressions or quality signals would otherwise dominate.
    Errors are the highest-priority gate.
    """
    rows = []
    # Eight successful pairs with identical quality.
    for i in range(8):
        legacy = _metrics(
            "legacy", h1=True, h3=True, h5=True, mrr=1.0, latency=1000.0,
        )
        graph = _metrics(
            "graph", h1=True, h3=True, h5=True, mrr=1.0, latency=1000.0,
        )
        rows.append(_row_pair(
            query_id=f"q-{i}", expected_doc_id="doc-7",
            legacy=legacy, graph=graph,
        ))
    # Two queries where graph fails outright (legacy succeeds).
    for i in range(2):
        legacy = _metrics(
            "legacy", h1=True, h3=True, h5=True, mrr=1.0, latency=1000.0,
        )
        graph = _metrics(
            "graph", success=False, h1=None, h3=None, h5=None, mrr=None,
            latency=1000.0,
        )
        rows.append(_row_pair(
            query_id=f"q-err-{i}", expected_doc_id="doc-7",
            legacy=legacy, graph=graph,
        ))

    summary = summarize_runs(rows=rows)
    # Sanity: graph success rate strictly below legacy.
    assert summary["graphSuccessRate"] + QUALITY_LIFT_EPSILON < summary["legacySuccessRate"]
    assert summary["recommendation"] == RECOMMENDATION_HOLD_REVIEW_ERRORS


def test_recommendation_quality_lift_with_p95_blowup_holds_review():
    """Quality lift with p95 latency past the 1.5x ceiling falls into
    ``hold_review_regressions`` rather than ``adopt_candidate`` — the
    quality is real but the cost exceeds the adoption bar.
    """
    rows = []
    # Bulk of queries match in quality with a latency tax just under the
    # per-query regression bar (2x), so per-query verdict stays tie/win
    # but aggregate p95 ratio blows past 1.5x.
    for i in range(10):
        if i < 5:
            legacy = _metrics(
                "legacy", h1=False, h3=False, h5=False, mrr=0.0, latency=500.0,
            )
            graph = _metrics(
                "graph", h1=False, h3=True, h5=True, mrr=0.5, latency=900.0,
            )
        else:
            legacy = _metrics(
                "legacy", h1=True, h3=True, h5=True, mrr=1.0, latency=500.0,
            )
            graph = _metrics(
                "graph", h1=True, h3=True, h5=True, mrr=1.0, latency=900.0,
            )
        rows.append(_row_pair(
            query_id=f"q-{i}", expected_doc_id="doc-7",
            legacy=legacy, graph=graph,
        ))

    summary = summarize_runs(rows=rows)
    legacy_p95 = summary["legacyLatencyP95"]
    graph_p95 = summary["graphLatencyP95"]
    assert (graph_p95 / legacy_p95) > 1.5
    assert summary["graphHitAt3"] > summary["legacyHitAt3"] + QUALITY_LIFT_EPSILON
    assert summary["regressions"] == 0
    assert summary["recommendation"] == RECOMMENDATION_HOLD_REVIEW_REGRESSIONS


def test_has_quality_lift_uses_epsilon():
    """Quality lift requires graph - legacy > epsilon. Sub-epsilon noise
    must not flip the gate.
    """
    # Sub-epsilon delta -- not a lift.
    summary_sub = {
        "legacyHitAt5": 0.700,
        "graphHitAt5": 0.700 + (QUALITY_LIFT_EPSILON / 2.0),
        "legacyMRR": 0.65,
        "graphMRR": 0.65,
    }
    assert has_quality_lift(summary_sub) is False
    # Just over the epsilon -- counts as a lift.
    summary_over = dict(summary_sub)
    summary_over["graphHitAt5"] = 0.700 + (QUALITY_LIFT_EPSILON * 2.0)
    assert has_quality_lift(summary_over) is True


def test_has_quality_lift_keyword_only():
    """Keyword hit rate alone is enough to declare a quality lift when
    there are no expected_doc_id rows.
    """
    summary = {
        "legacyKeywordHitRate": 0.60,
        "graphKeywordHitRate": 0.60 + (QUALITY_LIFT_EPSILON * 5.0),
    }
    assert has_quality_lift(summary) is True


def test_recommendation_no_signals_holds_experimental():
    """Empty edge case: when no expected_doc_id and no keyword rows
    exist, no quality lift can be detected. With no extra cost either,
    the rule lands on ``hold_experimental_backend_only``.
    """
    rows = []
    for i in range(5):
        legacy = _metrics("legacy", latency=1000.0)
        graph = _metrics("graph", latency=1000.0)
        rows.append(_row_pair(
            query_id=f"q-{i}", legacy=legacy, graph=graph,
        ))
    summary = summarize_runs(rows=rows)
    # No expected_doc_id, no expected_keywords → no hit@k aggregates.
    assert "graphHitAt5" not in summary
    assert "graphKeywordHitRate" not in summary
    assert summary["recommendation"] == RECOMMENDATION_HOLD_EXPERIMENTAL


def test_recommendation_applied_to_existing_ab_run_is_experimental():
    """Apply the conservative rule to a synthetic copy of the historical
    A/B summary that previously emitted ``adopt_candidate``: hit@k / MRR
    identical, graphWins=1, regressions=0. The new rule must downgrade
    it to ``hold_experimental_backend_only`` — that's the user-reported
    bug fix this whole change targets.
    """
    rows = []
    # 199 ties on identical quality + tiny latency wobble.
    for i in range(199):
        legacy = _metrics(
            "legacy", h1=True, h3=True, h5=True, mrr=1.0, latency=1000.0,
        )
        graph = _metrics(
            "graph", h1=True, h3=True, h5=True, mrr=1.0, latency=1015.0,
        )
        rows.append(_row_pair(
            query_id=f"q-{i:04d}", expected_doc_id="doc-7",
            legacy=legacy, graph=graph,
        ))
    # One latency-only graph win — graph noticeably faster on a single
    # query (the cold-start case) with quality unchanged.
    legacy = _metrics(
        "legacy", h1=True, h3=True, h5=True, mrr=1.0, latency=1000.0,
    )
    graph = _metrics(
        "graph", h1=True, h3=True, h5=True, mrr=1.0, latency=200.0,
    )
    rows.append(_row_pair(
        query_id="q-cold-start", expected_doc_id="doc-7",
        legacy=legacy, graph=graph,
    ))

    summary = summarize_runs(rows=rows)
    assert summary["graphWins"] == 1
    assert summary["legacyWins"] == 0
    assert summary["regressions"] == 0
    assert summary["recommendation"] == RECOMMENDATION_HOLD_EXPERIMENTAL
