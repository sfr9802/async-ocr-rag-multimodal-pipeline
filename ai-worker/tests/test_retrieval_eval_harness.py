from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ai-worker" / "eval" / "harness" / "rag_ingestion_retrieval_eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rag_ingestion_retrieval_eval_for_harness", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_module = load_module()


def test_retrieval_eval_scores_xlsx_location_hits():
    rows = [_row()]

    report = eval_module.evaluate_gold_rows(rows, search_fn=_fake_search, top_k=10)

    assert report["metrics"]["Hit@1"] == 1.0
    assert report["metrics"]["MRR@10"] == 1.0
    assert report["metrics"]["xlsx_citation_location_accuracy"] == 1.0
    assert report["metrics"]["citation_accuracy"] == 1.0
    assert report["metrics"]["hidden_content_leakage_count"] == 0
    assert report["metrics"]["xlsx_file_hit@10"] == 1.0
    assert report["metrics"]["xlsx_sheet_hit@10"] == 1.0
    assert report["metrics"]["xlsx_range_contains@10"] == 1.0
    assert report["bucket_metrics"]["xlsx_lookup"]["Hit@10"] == 1.0
    query_result = report["query_results"][0]
    assert query_result["expected_file_name"] == "sales.xlsx"
    assert query_result["top_k_results"][0]["rank"] == 1
    assert query_result["top_k_results"][0]["match_breakdown"]["xlsx_range_contains"] is True
    assert query_result["final_match_outcome"] == "matched"
    assert query_result["failure_reason"] is None


def test_retrieval_eval_counts_identity_hit_and_location_accuracy_separately():
    report = eval_module.evaluate_gold_rows([_row()], search_fn=_fake_wrong_location_search, top_k=10)

    assert report["metrics"]["Hit@1"] == 1.0
    assert report["metrics"]["MRR@10"] == 1.0
    assert report["metrics"]["citation_accuracy"] == 0.0
    assert report["metrics"]["xlsx_citation_location_accuracy"] == 0.0
    assert report["per_query"][0]["location_match"] is False
    assert report["query_results"][0]["failure_reason"] == "expected_range_not_found"
    assert report["metrics"]["overall_failure_reason_counts"]["expected_range_not_found"] == 1


def test_retrieval_eval_keeps_scanning_for_location_match_after_identity_hit():
    report = eval_module.evaluate_gold_rows([_row()], search_fn=_fake_wrong_then_right_search, top_k=10)

    assert report["metrics"]["Hit@1"] == 1.0
    assert report["metrics"]["MRR@10"] == 1.0
    assert report["metrics"]["citation_accuracy"] == 1.0
    assert report["per_query"][0]["hit_rank"] == 1
    assert report["per_query"][0]["location_match"] is True


def test_retrieval_eval_scores_pdf_bbox_location_hits():
    report = eval_module.evaluate_gold_rows([_pdf_row()], search_fn=_fake_pdf_search, top_k=10)

    assert report["metrics"]["Hit@1"] == 1.0
    assert report["metrics"]["pdf_citation_location_accuracy"] == 1.0
    assert report["metrics"]["pdf_file_hit@10"] == 1.0
    assert report["metrics"]["pdf_page_hit@10"] == 1.0
    assert report["metrics"]["pdf_bbox_overlap@10"] == 1.0
    assert report["bucket_metrics"]["pdf_page_lookup"]["MRR@10"] == 1.0


def test_retrieval_eval_classifies_pdf_page_and_bbox_failures():
    page_report = eval_module.evaluate_gold_rows([_pdf_row()], search_fn=_fake_pdf_wrong_page_search, top_k=10)
    bbox_report = eval_module.evaluate_gold_rows([_pdf_row()], search_fn=_fake_pdf_wrong_bbox_search, top_k=10)

    assert page_report["query_results"][0]["failure_reason"] == "expected_page_not_found"
    assert page_report["metrics"]["pdf_page_hit@10"] == 0.0
    assert bbox_report["query_results"][0]["failure_reason"] == "bbox_mismatch"
    assert bbox_report["metrics"]["pdf_page_hit@10"] == 1.0
    assert bbox_report["metrics"]["pdf_bbox_overlap@10"] == 0.0


def test_retrieval_eval_counts_hidden_content_leakage():
    row = _row()
    row["hidden_policy"] = "exclude_hidden"
    row["must_not_contain_terms"] = "secret"

    report = eval_module.evaluate_gold_rows([row], search_fn=_fake_hidden_search, top_k=10)

    assert report["metrics"]["hidden_content_leakage_count"] == 1


def test_retrieval_eval_counts_negative_hidden_policy_leakage():
    row = _row()
    row["hidden_policy"] = "negative"
    row["must_not_contain_terms"] = "secret"

    report = eval_module.evaluate_gold_rows([row], search_fn=_fake_hidden_search, top_k=10)

    assert report["metrics"]["hidden_content_leakage_count"] == 1


def test_retrieval_eval_filters_unembedded_hits_by_default():
    report = eval_module.evaluate_gold_rows([_row()], search_fn=_fake_pending_search, top_k=10)

    assert report["metrics"]["Hit@1"] == 0.0
    assert report["metrics"]["embedding_filtered_eval"] is True
    assert report["metrics"]["indexing_filtered_hit_count"] == 1
    assert report["per_query"][0]["failure_reason"] == "embedding_status_mismatch"
    assert report["metrics"]["embedding_status_mismatch_count"] == 1


def test_retrieval_eval_classifies_required_index_version_mismatch():
    report = eval_module.evaluate_gold_rows(
        [_row()],
        search_fn=_fake_search,
        top_k=10,
        required_embedding_status=None,
        required_index_version="candidate-v2",
    )

    assert report["metrics"]["Hit@1"] == 0.0
    assert report["per_query"][0]["failure_reason"] == "required_index_version_mismatch"
    assert report["metrics"]["required_index_version_mismatch_count"] == 1


def test_retrieval_eval_scores_negative_hidden_policy_without_positive_denominator():
    row = _row()
    row["hidden_policy"] = "negative"
    row["must_not_contain_terms"] = "secret"

    report = eval_module.evaluate_gold_rows([row], search_fn=lambda _query, _top_k: [], top_k=10)

    assert report["metrics"]["hidden_negative_pass_count"] == 1
    assert report["metrics"]["Hit@10"] == 0.0
    assert report["query_results"][0]["final_match_outcome"] == "hidden_negative_pass"
    assert report["query_results"][0]["failure_reason"] is None


def test_retrieval_eval_reports_invalid_gold_row_without_searching():
    row = _row()
    row["expected_sheet_name"] = ""

    def fail_search(_query: str, _top_k: int):
        raise AssertionError("invalid rows must not call search")

    report = eval_module.evaluate_gold_rows([row], search_fn=fail_search, top_k=10)

    assert report["status"] == "COMPLETED_WITH_INVALID_GOLD"
    assert report["metrics"]["gold_label_invalid_count"] == 1
    assert report["query_results"][0]["label_status"] == "invalid"
    assert report["query_results"][0]["failure_reason"] == "gold_label_invalid"


def test_retrieval_eval_classifies_empty_results():
    report = eval_module.evaluate_gold_rows([_row()], search_fn=lambda _query, _top_k: [], top_k=10)

    assert report["metrics"]["result_empty_count"] == 1
    assert report["query_results"][0]["failure_reason"] == "search_result_empty"
    assert report["metrics"]["overall_failure_reason_counts"]["search_result_empty"] == 1


def test_retrieval_eval_writes_report_for_invalid_gold_csv(tmp_path):
    gold = tmp_path / "invalid_gold.csv"
    report_path = tmp_path / "report.json"
    gold.write_text("query_id,bucket\nq1,xlsx_lookup\n", encoding="utf-8")

    exit_code = eval_module.main(["--gold", str(gold), "--report", str(report_path)])

    assert exit_code == 1
    payload = eval_module.json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "VALIDATION_FAILED"
    assert payload["validation"]["ok"] is False
    assert any("missing required columns" in error for error in payload["validation"]["errors"])


def _row() -> dict[str, str]:
    return {
        column: ""
        for column in eval_module.REQUIRED_COLUMNS
    } | {
        "query_id": "q1",
        "bucket": "xlsx_lookup",
        "query": "1호선",
        "expected_file_name": "sales.xlsx",
        "expected_document_version_id": "docv_1",
        "expected_chunk_type": "row_group",
        "expected_location_type": "xlsx",
        "expected_sheet_name": "철도",
        "expected_cell_range": "B2:C3",
        "range_match_policy": "contains_expected",
        "hidden_policy": "exclude_hidden",
        "requires_formula_value": "false",
        "requires_formatted_value": "true",
        "requires_aggregation": "false",
        "label_status": "bound",
    }


def _pdf_row() -> dict[str, str]:
    return {
        column: ""
        for column in eval_module.REQUIRED_COLUMNS
    } | {
        "query_id": "q-pdf",
        "bucket": "pdf_page_lookup",
        "query": "계약 해지 조건",
        "expected_file_name": "contract.pdf",
        "expected_document_version_id": "docv_pdf_1",
        "expected_chunk_type": "paragraph",
        "expected_location_type": "pdf",
        "expected_physical_page_index": "2",
        "expected_page_no": "3",
        "expected_page_label": "iii",
        "expected_bbox": "[72, 120, 510, 680]",
        "range_match_policy": "none",
        "hidden_policy": "include_hidden",
        "requires_formula_value": "false",
        "requires_formatted_value": "false",
        "requires_aggregation": "false",
        "label_status": "bound",
    }


def _fake_search(_query: str, _top_k: int):
    return [
        {
            "sourceFile": {"originalFileName": "sales.xlsx"},
            "searchUnit": {
                "searchUnitId": "unit-1",
                "embeddingStatus": "EMBEDDED",
                "indexVersion": "idx-v1",
                "chunkType": "row_group",
                "locationType": "xlsx",
                "locationJson": (
                    '{"type":"xlsx","document_version_id":"docv_1",'
                    '"sheet_name":"철도","cell_range":"A1:D10"}'
                ),
                "citationText": "sales.xlsx > 철도 > A1:D10",
            },
        }
    ]


def _fake_wrong_location_search(_query: str, _top_k: int):
    return [
        {
            "sourceFile": {"originalFileName": "sales.xlsx"},
            "searchUnit": {
                "searchUnitId": "unit-wrong",
                "embeddingStatus": "EMBEDDED",
                "indexVersion": "idx-v1",
                "chunkType": "row_group",
                "locationType": "xlsx",
                "locationJson": (
                    '{"type":"xlsx","document_version_id":"docv_1",'
                    '"sheet_name":"철도","cell_range":"Z100:Z101"}'
                ),
                "citationText": "sales.xlsx > 철도 > Z100:Z101",
            },
        }
    ]


def _fake_wrong_then_right_search(_query: str, _top_k: int):
    right = _fake_search(_query, _top_k)[0]
    return _fake_wrong_location_search(_query, _top_k) + [right]


def _fake_pdf_search(_query: str, _top_k: int):
    return [
        {
            "sourceFile": {"originalFileName": "contract.pdf"},
            "searchUnit": {
                "searchUnitId": "unit-pdf",
                "embeddingStatus": "EMBEDDED",
                "indexVersion": "idx-v1",
                "chunkType": "paragraph",
                "locationType": "pdf",
                "locationJson": (
                    '{"type":"pdf","document_version_id":"docv_pdf_1",'
                    '"physical_page_index":2,"page_no":3,"page_label":"iii",'
                    '"bbox":[70,118,500,650]}'
                ),
                "citationText": "contract.pdf > p.iii",
            },
        }
    ]


def _fake_pdf_wrong_page_search(_query: str, _top_k: int):
    hit = _fake_pdf_search(_query, _top_k)[0]
    hit["searchUnit"]["locationJson"] = (
        '{"type":"pdf","document_version_id":"docv_pdf_1",'
        '"physical_page_index":9,"page_no":10,"page_label":"10",'
        '"bbox":[70,118,500,650]}'
    )
    return [hit]


def _fake_pdf_wrong_bbox_search(_query: str, _top_k: int):
    hit = _fake_pdf_search(_query, _top_k)[0]
    hit["searchUnit"]["locationJson"] = (
        '{"type":"pdf","document_version_id":"docv_pdf_1",'
        '"physical_page_index":2,"page_no":3,"page_label":"iii",'
        '"bbox":[1,1,10,10]}'
    )
    return [hit]


def _fake_hidden_search(_query: str, _top_k: int):
    hit = _fake_search(_query, _top_k)[0]
    hit["searchUnit"]["citation"] = {"locationJson": '{"note":"SECRET value"}'}
    return [hit]


def _fake_pending_search(_query: str, _top_k: int):
    hit = _fake_search(_query, _top_k)[0]
    hit["searchUnit"]["embeddingStatus"] = "PENDING"
    return [hit]
