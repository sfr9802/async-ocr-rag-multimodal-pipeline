"""Small golden retrieval harness for SearchUnit-based RAG indexes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from app.capabilities.rag.retrieval_contract import retrieval_result_row
from app.core.config import WorkerSettings
from app.core.logging import configure_logging


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query: str
    category: Optional[str] = None
    query_types: list[str] = field(default_factory=list)
    expected: list[dict[str, Any]] = field(default_factory=list)
    acceptable: list[dict[str, Any]] = field(default_factory=list)
    must_not: list[dict[str, Any]] = field(default_factory=list)
    answer_hint: Optional[str] = None


def load_golden_queries(path: Path) -> list[GoldenQuery]:
    queries: list[GoldenQuery] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        raw = json.loads(stripped)
        queries.append(GoldenQuery(
            id=str(raw["id"]),
            query=str(raw["query"]),
            category=_str_or_none(raw.get("category")),
            query_types=_list_of_str(raw.get("queryTypes") or raw.get("query_types")),
            expected=_list_of_dicts(raw.get("expected")),
            acceptable=_list_of_dicts(raw.get("acceptable")),
            must_not=_list_of_dicts(raw.get("mustNot") or raw.get("must_not")),
            answer_hint=_str_or_none(raw.get("answerHint") or raw.get("answer_hint")),
        ))
        if not queries[-1].expected:
            raise ValueError(f"{path}:{line_no} expected must contain at least one target")
    return queries


def load_source_manifest(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    sources = raw.get("sources") if isinstance(raw, dict) else None
    if not isinstance(sources, list):
        if isinstance(raw, dict):
            return {
                key: value
                for key, value in raw.items()
                if key in {"dataset", "datasetId", "source", "matchingIdentity"}
            }
        return {}
    by_id: dict[str, Any] = {}
    by_file: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        logical_id = _str_or_none(source.get("logicalSourceId"))
        file_name = _str_or_none(source.get("fileName") or source.get("sourceFileName"))
        if logical_id:
            by_id[logical_id] = source
        if file_name:
            by_file[file_name] = source
    manifest = {"by_id": by_id, "by_file": by_file}
    for key in ("dataset", "datasetId", "source", "matchingIdentity"):
        if isinstance(raw, dict) and key in raw:
            manifest[key] = raw[key]
    return manifest


def evaluate_golden_queries(
    queries: list[GoldenQuery],
    *,
    retrieve: Callable[[str, int], list[dict[str, Any]]],
    top_k: int,
    manifest: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    from ai_worker.evals.golden_retrieval.metrics import ndcg_at_k

    manifest = manifest or {}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    metric_totals = {
        "hit@1": 0.0,
        "hit@3": 0.0,
        "hit@5": 0.0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
        "source_file_accuracy@5": 0.0,
        "page_accuracy@5": 0.0,
        "unit_type_accuracy@5": 0.0,
        "citation_match@5": 0.0,
    }
    must_not_violations = 0

    for golden in queries:
        results = retrieve(golden.query, top_k)
        positives = [*golden.expected, *golden.acceptable]
        first_match_rank = _first_match_rank(results, positives, manifest)
        expected_rank = _first_match_rank(results, golden.expected, manifest)
        violations = _must_not_violations(results[:top_k], golden.must_not, manifest)
        must_not_violations += len(violations)

        for k in (1, 3, 5):
            if first_match_rank is not None and first_match_rank <= min(k, top_k):
                metric_totals[f"hit@{k}"] += 1.0
        if first_match_rank is not None:
            metric_totals["mrr"] += 1.0 / float(first_match_rank)
        metric_totals["ndcg@5"] += ndcg_at_k(
            results,
            golden.expected,
            golden.acceptable,
            manifest=manifest,
            k=min(5, top_k),
        )
        if _has_source_match(results[:top_k], positives, manifest):
            metric_totals["source_file_accuracy@5"] += 1.0
        if _has_page_match(results[:top_k], positives, manifest):
            metric_totals["page_accuracy@5"] += 1.0
        if _has_unit_type_match(results[:top_k], positives, manifest):
            metric_totals["unit_type_accuracy@5"] += 1.0
        if expected_rank is not None and expected_rank <= min(5, top_k):
            metric_totals["citation_match@5"] += 1.0

        row = {
            "id": golden.id,
            "category": golden.category,
            "query": golden.query,
            "firstMatchRank": first_match_rank,
            "expectedMatchRank": expected_rank,
            "mustNotViolations": violations,
            "topResult": _summarize_result(results[0]) if results else None,
        }
        rows.append(row)
        if first_match_rank is None or first_match_rank > min(5, top_k) or violations:
            failure = dict(row)
            failure["expected"] = golden.expected
            failure["acceptable"] = golden.acceptable
            failures.append(failure)

    n = len(queries) or 1
    metrics = {key: round(value / n, 6) for key, value in metric_totals.items()}
    metrics["must_not_violation_count"] = must_not_violations
    title = "Golden Retrieval Eval"
    if manifest.get("dataset") == "kovidore-v2-economic-beir" or manifest.get("datasetId") == "kovidore-v2-economic-beir":
        title = "Golden Retrieval Eval - KoViDoRe Economic"
    return {
        "title": title,
        "queries": len(queries),
        "topK": top_k,
        "metrics": metrics,
        "rows": rows,
        "failures": failures,
        "worstCases": failures[:10],
    }


def print_report(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    title = report.get("title") or "Golden Retrieval Eval"
    print(title)
    print("")
    print(f"queries: {report['queries']}")
    print(f"hit@1: {metrics['hit@1']:.2f}")
    print(f"hit@3: {metrics['hit@3']:.2f}")
    print(f"hit@5: {metrics['hit@5']:.2f}")
    print(f"mrr: {metrics['mrr']:.2f}")
    print(f"ndcg@5: {metrics['ndcg@5']:.2f}")
    print(f"source_file_accuracy@5: {metrics['source_file_accuracy@5']:.2f}")
    print(f"page_accuracy@5: {metrics['page_accuracy@5']:.2f}")
    print(f"unit_type_accuracy@5: {metrics['unit_type_accuracy@5']:.2f}")
    print(f"citation_match@5: {metrics['citation_match@5']:.2f}")
    print(f"must_not_violations: {metrics['must_not_violation_count']}")
    if report["worstCases"]:
        print("")
        print("Worst cases:")
        for failure in report["worstCases"]:
            got = failure.get("topResult") or {}
            expected = failure.get("expected") or []
            first_expected = expected[0] if expected else {}
            print(
                "- {id}: expected {exp_type} {exp_key}, got {got_type} {got_key}".format(
                    id=failure["id"],
                    exp_type=first_expected.get("unitType") or "?",
                    exp_key=first_expected.get("unitKey") or first_expected.get("sourceFileName") or "?",
                    got_type=got.get("unitType") or "?",
                    got_key=got.get("unitKey") or got.get("sourceFileName") or "?",
                )
            )


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    queries = load_golden_queries(Path(args.queries))
    manifest = load_source_manifest(Path(args.manifest)) if args.manifest else {}
    retrieve = _build_live_retriever(int(args.top_k))
    report = evaluate_golden_queries(
        queries,
        retrieve=retrieve,
        top_k=int(args.top_k),
        manifest=manifest,
    )
    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    failures_out = Path(args.failures_out) if args.failures_out else out.with_name("eval_failures.jsonl")
    failures_out.parent.mkdir(parents=True, exist_ok=True)
    failures_out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in report["failures"]),
        encoding="utf-8",
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate SearchUnit retrieval against a small golden JSONL set.",
    )
    parser.add_argument("--queries", required=True, help="Path to golden_queries.jsonl.")
    parser.add_argument("--top-k", type=int, default=5, help="Retriever top-k to evaluate.")
    parser.add_argument("--out", required=True, help="Path to write eval_report.json.")
    parser.add_argument("--failures-out", default=None, help="Optional eval_failures.jsonl path.")
    parser.add_argument("--manifest", default=None, help="Optional source_manifest.json path.")
    return parser


def _build_live_retriever(top_k: int) -> Callable[[str, int], list[dict[str, Any]]]:
    from app.capabilities.registry import _get_shared_retriever_bundle  # noqa: WPS433

    settings = WorkerSettings(rag_top_k=top_k)
    retriever, _generator = _get_shared_retriever_bundle(settings)

    def _retrieve(query: str, requested_top_k: int) -> list[dict[str, Any]]:
        report = retriever.retrieve(query)
        return [
            retrieval_result_row(i + 1, chunk)
            for i, chunk in enumerate(report.results[:requested_top_k])
        ]

    return _retrieve


def _first_match_rank(
    results: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> Optional[int]:
    for rank, result in enumerate(results, start=1):
        if any(result_matches_spec(result, spec, manifest) for spec in specs):
            return rank
    return None


def result_matches_spec(
    result: dict[str, Any],
    spec: dict[str, Any],
    manifest: Optional[dict[str, Any]] = None,
) -> bool:
    manifest = manifest or {}
    if not _result_source_matches_spec(result, spec, manifest):
        return False
    if spec.get("sourceFileId") and _value(result, "sourceFileId", "source_file_id") != spec.get("sourceFileId"):
        return False
    if spec.get("searchUnitId") and _value(result, "searchUnitId", "search_unit_id") != spec.get("searchUnitId"):
        return False

    spec_unit_type = _str_or_none(spec.get("unitType"))
    spec_unit_key = _str_or_none(spec.get("unitKey"))
    result_unit_type = _str_or_none(_value(result, "unitType", "unit_type"))
    result_unit_key = _str_or_none(_value(result, "unitKey", "unit_key"))

    if spec_unit_type or spec_unit_key:
        has_result_identity = bool(result_unit_type and result_unit_key)
        if has_result_identity:
            return result_unit_type == spec_unit_type and result_unit_key == spec_unit_key
        if spec_unit_type and result_unit_type and result_unit_type != spec_unit_type:
            return False
        return _page_range_matches(result, spec)

    if "pageStart" in spec or "pageEnd" in spec:
        return _page_range_matches(result, spec)
    return True


def _must_not_violations(
    results: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        for spec in specs:
            if result_matches_spec(result, spec, manifest):
                violations.append({"rank": rank, "spec": spec, "result": _summarize_result(result)})
    return violations


def _has_source_match(
    results: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> bool:
    for result in results:
        result_source_name = _value(result, "sourceFileName", "source_file_name", "originalFilename")
        result_source_id = _value(result, "sourceFileId", "source_file_id")
        for spec in specs:
            source_names = _spec_source_names(spec, manifest)
            if source_names and result_source_name in source_names:
                return True
            if spec.get("sourceFileName") and result_source_name == spec.get("sourceFileName"):
                return True
            if spec.get("sourceFileId") and result_source_id == spec.get("sourceFileId"):
                return True
    return False


def _has_page_match(
    results: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> bool:
    page_specs = [spec for spec in specs if "pageStart" in spec or "pageEnd" in spec]
    if not page_specs:
        return False
    for result in results:
        result_start = _int_or_none(_value(result, "pageStart", "page_start"))
        result_end = _int_or_none(_value(result, "pageEnd", "page_end"))
        if result_start is None or result_end is None:
            continue
        for spec in page_specs:
            if not _result_source_matches_spec(result, spec, manifest):
                continue
            if _page_range_matches(result, spec):
                return True
    return False


def _has_unit_type_match(
    results: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> bool:
    unit_type_specs = [spec for spec in specs if spec.get("unitType")]
    if not unit_type_specs:
        return False
    for result in results:
        result_type = _value(result, "unitType", "unit_type")
        for spec in unit_type_specs:
            if result_type == spec.get("unitType") and _result_source_matches_spec(result, spec, manifest):
                return True
    return False


def _result_source_matches_spec(
    result: dict[str, Any],
    spec: dict[str, Any],
    manifest: dict[str, Any],
) -> bool:
    source_names = _spec_source_names(spec, manifest)
    result_source_name = _value(result, "sourceFileName", "source_file_name", "originalFilename")
    result_source_id = _value(result, "sourceFileId", "source_file_id")
    if source_names:
        return result_source_name in source_names
    if spec.get("sourceFileName"):
        return result_source_name == spec.get("sourceFileName")
    if spec.get("sourceFileId"):
        return result_source_id == spec.get("sourceFileId")
    return True


def _spec_source_names(spec: dict[str, Any], manifest: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    if spec.get("sourceFileName"):
        names.add(str(spec["sourceFileName"]))
    logical_id = spec.get("logicalSourceId")
    if logical_id:
        source = (manifest.get("by_id") or {}).get(str(logical_id))
        if isinstance(source, dict):
            file_name = source.get("fileName") or source.get("sourceFileName")
            if file_name:
                names.add(str(file_name))
    return names


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "sourceFileName": _value(result, "sourceFileName", "source_file_name", "originalFilename"),
        "sourceFileId": _value(result, "sourceFileId", "source_file_id"),
        "searchUnitId": _value(result, "searchUnitId", "search_unit_id"),
        "unitType": _value(result, "unitType", "unit_type"),
        "unitKey": _value(result, "unitKey", "unit_key"),
        "pageStart": _value(result, "pageStart", "page_start"),
        "pageEnd": _value(result, "pageEnd", "page_end"),
        "score": _value(result, "score"),
    }


def _value(result: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in result and result[key] is not None:
            return result[key]
    citation = result.get("citation")
    if isinstance(citation, dict):
        for key in keys:
            if key in citation and citation[key] is not None:
                return citation[key]
    return None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if _str_or_none(item)]


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _page_range_matches(result: dict[str, Any], spec: dict[str, Any]) -> bool:
    if "pageStart" not in spec and "pageEnd" not in spec:
        return False
    result_start = _int_or_none(_value(result, "pageStart", "page_start"))
    result_end = _int_or_none(_value(result, "pageEnd", "page_end"))
    if result_start is None or result_end is None:
        return False
    spec_start = _int_or_none(spec.get("pageStart", spec.get("pageEnd")))
    spec_end = _int_or_none(spec.get("pageEnd", spec.get("pageStart")))
    if spec_start is None or spec_end is None:
        return False
    return result_start <= spec_end and result_end >= spec_start


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
