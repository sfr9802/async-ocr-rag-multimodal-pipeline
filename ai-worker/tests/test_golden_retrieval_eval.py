from __future__ import annotations

import json

import pytest

from ai_worker.evals.golden_retrieval.run import (
    GoldenQuery,
    evaluate_golden_queries,
    load_golden_queries,
    load_source_manifest,
    result_matches_spec,
)


def _row(
    *,
    source="maintenance_contract.pdf",
    unit_type="PAGE",
    unit_key="page:5",
    page_start=5,
    page_end=5,
    search_unit_id="unit-page-5",
):
    return {
        "rank": 1,
        "chunkId": f"chunk-{unit_key}",
        "docId": "source-1",
        "score": 0.9,
        "sourceFileName": source,
        "sourceFileId": "source-1",
        "searchUnitId": search_unit_id,
        "unitType": unit_type,
        "unitKey": unit_key,
        "pageStart": page_start,
        "pageEnd": page_end,
        "citation": {
            "sourceFileName": source,
            "sourceFileId": "source-1",
            "searchUnitId": search_unit_id,
            "unitType": unit_type,
            "unitKey": unit_key,
            "pageStart": page_start,
            "pageEnd": page_end,
        },
    }


def test_load_golden_queries_and_manifest(tmp_path):
    queries_path = tmp_path / "golden_queries.jsonl"
    queries_path.write_text(
        json.dumps({
            "id": "q001",
            "query": "SLA 응답 시간",
            "expected": [{"sourceFileName": "maintenance_contract.pdf"}],
            "acceptable": [],
            "mustNot": [{"sourceFileName": "sales_proposal.pdf"}],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "sources": [
                {
                    "logicalSourceId": "maintenance_contract_2024",
                    "fileName": "maintenance_contract.pdf",
                }
            ]
        }),
        encoding="utf-8",
    )

    queries = load_golden_queries(queries_path)
    manifest = load_source_manifest(manifest_path)

    assert queries[0].id == "q001"
    assert queries[0].must_not[0]["sourceFileName"] == "sales_proposal.pdf"
    assert manifest["by_id"]["maintenance_contract_2024"]["fileName"] == "maintenance_contract.pdf"


def test_result_matching_uses_stable_search_unit_identity_and_manifest():
    manifest = {
        "by_id": {
            "maintenance_contract_2024": {"fileName": "maintenance_contract.pdf"}
        }
    }
    result = _row(unit_type="TABLE", unit_key="page:5:table:1")

    assert result_matches_spec(
        result,
        {
            "logicalSourceId": "maintenance_contract_2024",
            "unitType": "TABLE",
            "unitKey": "page:5:table:1",
        },
        manifest,
    )
    assert result_matches_spec(
        result,
        {
            "sourceFileName": "maintenance_contract.pdf",
            "pageStart": 5,
            "pageEnd": 5,
        },
        manifest,
    )
    assert not result_matches_spec(
        result,
        {"sourceFileName": "sales_proposal.pdf"},
        manifest,
    )


def test_eval_metrics_and_failure_report_are_computed():
    queries = [
        GoldenQuery(
            id="q001",
            query="SLA 응답 시간",
            expected=[
                {
                    "sourceFileName": "maintenance_contract.pdf",
                    "unitType": "TABLE",
                    "unitKey": "page:5:table:1",
                    "pageStart": 5,
                    "pageEnd": 5,
                }
            ],
            acceptable=[
                {
                    "sourceFileName": "maintenance_contract.pdf",
                    "unitType": "PAGE",
                    "unitKey": "page:5",
                    "pageStart": 5,
                    "pageEnd": 5,
                }
            ],
        ),
        GoldenQuery(
            id="q002",
            query="장애 보고서 제출 기한",
            expected=[
                {
                    "sourceFileName": "operations_policy.pdf",
                    "unitType": "PAGE",
                    "unitKey": "page:3",
                    "pageStart": 3,
                    "pageEnd": 3,
                }
            ],
            must_not=[{"sourceFileName": "policy_2023.pdf"}],
        ),
    ]

    def retrieve(query: str, top_k: int):
        if "SLA" in query:
            return [
                _row(unit_type="PAGE", unit_key="page:5"),
                _row(unit_type="TABLE", unit_key="page:5:table:1", search_unit_id="unit-table-1"),
            ][:top_k]
        return [
            _row(
                source="policy_2023.pdf",
                unit_type="PAGE",
                unit_key="page:2",
                page_start=2,
                page_end=2,
            )
        ]

    report = evaluate_golden_queries(queries, retrieve=retrieve, top_k=5)

    assert report["metrics"]["hit@1"] == pytest.approx(0.5)
    assert report["metrics"]["hit@3"] == pytest.approx(0.5)
    assert report["metrics"]["hit@5"] == pytest.approx(0.5)
    assert report["metrics"]["mrr"] == pytest.approx(0.5)
    assert report["metrics"]["source_file_accuracy@5"] == pytest.approx(0.5)
    assert report["metrics"]["page_accuracy@5"] == pytest.approx(0.5)
    assert report["metrics"]["unit_type_accuracy@5"] == pytest.approx(0.5)
    assert report["metrics"]["citation_match@5"] == pytest.approx(0.5)
    assert report["metrics"]["must_not_violation_count"] == 1
    assert report["failures"][0]["id"] == "q002"
