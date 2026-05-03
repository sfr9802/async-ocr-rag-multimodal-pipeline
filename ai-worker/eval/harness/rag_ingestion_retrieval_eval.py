"""Retrieval eval harness for xlsx/pdf RAG ingestion v2.

The harness validates the gold CSV schema, calls the current library search
endpoint by default, scores location-aware hits, and writes a report JSON. It
does not introduce hybrid retrieval, reranking, or answer generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx


REQUIRED_COLUMNS = [
    "query_id",
    "bucket",
    "query",
    "expected_file_name",
    "expected_document_version_id",
    "expected_chunk_type",
    "expected_location_type",
    "expected_sheet_name",
    "expected_cell_range",
    "expected_table_id",
    "expected_physical_page_index",
    "expected_page_no",
    "expected_page_label",
    "expected_bbox",
    "expected_answer_text",
    "must_contain_terms",
    "must_not_contain_terms",
    "range_match_policy",
    "hidden_policy",
    "requires_formula_value",
    "requires_formatted_value",
    "requires_aggregation",
    "source_sample_id",
    "label_status",
    "notes",
]

SUPPORTED_BUCKETS = {
    "xlsx_lookup",
    "xlsx_header_ambiguous",
    "xlsx_formula_value",
    "xlsx_date_number_format",
    "xlsx_hidden_policy",
    "xlsx_aggregation",
    "pdf_page_lookup",
    "pdf_section_question",
    "pdf_table_lookup",
    "pdf_ocr_noise",
    "mixed_text_table",
}
SUPPORTED_RANGE_POLICIES = {"exact_match", "contains_expected", "overlaps_expected", "none"}
SUPPORTED_HIDDEN_POLICIES = {"", "include_hidden", "exclude_hidden", "negative"}
BOOL_COLUMNS = {"requires_formula_value", "requires_formatted_value", "requires_aggregation"}
FAILURE_REASONS = {
    "search_result_empty",
    "expected_file_not_found",
    "expected_sheet_not_found",
    "expected_table_not_found",
    "expected_range_not_found",
    "expected_page_not_found",
    "bbox_mismatch",
    "hidden_content_returned",
    "gold_label_invalid",
    "candidate_index_mismatch",
    "embedding_status_mismatch",
    "required_index_version_mismatch",
    "unsupported_bucket",
    "match_policy_error",
    "unknown",
}


@dataclass(frozen=True)
class GoldValidationResult:
    row_count: int
    errors: list[str]
    bucket_counts: dict[str, int]
    row_errors: dict[str, list[str]]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_gold_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_gold_rows(rows: list[dict[str, str]], *, require_live_bound: bool = False) -> GoldValidationResult:
    errors: list[str] = []
    row_errors: dict[str, list[str]] = {}
    if not rows:
        errors.append("gold CSV must contain at least one row")
        return GoldValidationResult(0, errors, {}, row_errors)

    columns = set(rows[0].keys())
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        errors.append("missing required columns: " + ", ".join(missing_columns))

    seen_ids: set[str] = set()
    bucket_counts: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows, start=2):
        current_errors: list[str] = []
        query_id = _cell(row, "query_id")
        if not query_id:
            current_errors.append("query_id is required")
        elif query_id in seen_ids:
            current_errors.append(f"duplicate query_id {query_id}")
        seen_ids.add(query_id)

        bucket = _cell(row, "bucket")
        if bucket not in SUPPORTED_BUCKETS:
            current_errors.append(f"unsupported bucket {bucket!r}")
        else:
            bucket_counts[bucket] += 1

        if not _cell(row, "query"):
            current_errors.append("query is required")
        hidden_policy = _cell(row, "hidden_policy")
        is_negative = hidden_policy == "negative"
        if not is_negative and not _cell(row, "expected_file_name"):
            current_errors.append("expected_file_name is required")
        if require_live_bound and not _cell(row, "expected_document_version_id"):
            current_errors.append("expected_document_version_id is required for live-bound eval")

        policy = _cell(row, "range_match_policy") or "none"
        if policy not in SUPPORTED_RANGE_POLICIES:
            current_errors.append(f"unsupported range_match_policy {policy!r}")
        if hidden_policy not in SUPPORTED_HIDDEN_POLICIES:
            current_errors.append(f"unsupported hidden_policy {hidden_policy!r}")

        location_type = _cell(row, "expected_location_type")
        if not is_negative and location_type not in {"xlsx", "pdf", "ocr"}:
            current_errors.append("expected_location_type must be xlsx, pdf, or ocr")
        if location_type == "xlsx" and not is_negative and policy != "none":
            if not _cell(row, "expected_sheet_name"):
                current_errors.append("expected_sheet_name is required for XLSX range policy")
            expected_range = _cell(row, "expected_cell_range")
            if not expected_range:
                current_errors.append("expected_cell_range is required for XLSX range policy")
            elif _cell_range(expected_range) is None:
                current_errors.append("expected_cell_range must be an A1-style range")
        if location_type in {"pdf", "ocr"} and not is_negative and not (
            _cell(row, "expected_physical_page_index") or _cell(row, "expected_page_no")
        ):
            current_errors.append("PDF rows require expected page fields")
        if location_type in {"pdf", "ocr"} and _cell(row, "expected_bbox") and _bbox(_cell(row, "expected_bbox")) is None:
            current_errors.append("expected_bbox must contain four numeric coordinates")

        for column in BOOL_COLUMNS:
            value = _cell(row, column).lower()
            if value and value not in {"true", "false"}:
                current_errors.append(f"{column} must be true or false")

        label_status = _cell(row, "label_status")
        if label_status not in {"pending", "bound", "verified", "rejected", "invalid"}:
            current_errors.append("label_status must be pending, bound, verified, rejected, or invalid")

        if current_errors:
            row_key = query_id or f"row_{index}"
            row_errors[row_key] = current_errors
            errors.extend(f"row {index}: {error}" for error in current_errors)

    return GoldValidationResult(len(rows), errors, dict(bucket_counts), row_errors)


def evaluate_gold_rows(
    rows: list[dict[str, str]],
    *,
    search_fn: Callable[[str, int], list[dict[str, Any]]],
    top_k: int = 10,
    candidate_index_version: str | None = None,
    baseline_index_version: str | None = None,
    required_embedding_status: str | None = "EMBEDDED",
    required_index_version: str | None = None,
) -> dict[str, Any]:
    validation = validate_gold_rows(rows, require_live_bound=False)
    row_errors = validation.row_errors
    per_query: list[dict[str, Any]] = []
    query_results: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    location_hits: list[bool] = []
    xlsx_location_hits: list[bool] = []
    pdf_location_hits: list[bool] = []
    xlsx_file_hits: list[bool] = []
    xlsx_sheet_hits: list[bool] = []
    xlsx_table_hits: list[bool] = []
    xlsx_range_overlap_hits: list[bool] = []
    xlsx_range_contains_hits: list[bool] = []
    xlsx_exact_range_hits: list[bool] = []
    pdf_file_hits: list[bool] = []
    pdf_page_hits: list[bool] = []
    pdf_bbox_overlap_hits: list[bool] = []
    pdf_exact_bbox_hits: list[bool] = []
    hidden_leakage_count = 0
    result_empty_count = 0
    gold_label_invalid_count = 0
    candidate_index_mismatch_count = 0
    indexing_filtered_hit_count = 0
    search_error_count = 0
    hidden_negative_pass_count = 0
    bucket_ranks: dict[str, list[int | None]] = defaultdict(list)
    bucket_stats: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    bucket_failure_reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    overall_failure_reason_counts: Counter[str] = Counter()

    for row in rows:
        query_id = _cell(row, "query_id")
        bucket = _cell(row, "bucket")
        invalid_errors = row_errors.get(query_id) or []
        is_negative = _cell(row, "hidden_policy") == "negative"
        label_status = "invalid" if invalid_errors else (_cell(row, "label_status") or "pending")
        if invalid_errors:
            gold_label_invalid_count += 1
            failure_reason = _invalid_failure_reason(invalid_errors)
            query_result = _base_query_result(
                row,
                candidate_index_version=candidate_index_version,
                baseline_index_version=baseline_index_version,
                label_status=label_status,
            ) | {
                "top_k_results": [],
                "final_match_outcome": "invalid_gold",
                "failure_reason": failure_reason,
                "validation_errors": invalid_errors,
            }
            query_results.append(query_result)
            per_query.append(_legacy_per_query(row, None, False, False, None, failure_reason))
            bucket_failure_reason_counts[bucket][failure_reason] += 1
            overall_failure_reason_counts[failure_reason] += 1
            continue

        search_error = None
        try:
            hits = search_fn(row["query"], top_k)
        except Exception as exc:  # pragma: no cover - live service resilience
            search_error = f"{type(exc).__name__}: {exc}"
            hits = []
            search_error_count += 1
        hit_details = [
            _summarize_ranked_hit(
                row,
                hit,
                rank=rank,
                candidate_index_version=candidate_index_version,
                required_embedding_status=required_embedding_status,
                required_index_version=required_index_version,
            )
            for rank, hit in enumerate(hits[:top_k], start=1)
        ]
        indexing_filtered_hit_count += sum(
            1
            for detail in hit_details
            if not detail["match_breakdown"].get("indexing_contract_match", True)
        )
        matched_rank: int | None = None
        location_rank: int | None = None
        location_ok = False
        hidden_leakage = _hidden_leakage(row, hits)
        hidden_leakage_count += 1 if hidden_leakage else 0

        for detail in hit_details:
            breakdown = detail["match_breakdown"]
            if breakdown["identity_match"]:
                if matched_rank is None:
                    matched_rank = detail["rank"]
                if breakdown["location_match"]:
                    location_rank = detail["rank"]
                    location_ok = True
                    break

        if not hits and search_error is None:
            result_empty_count += 1

        if is_negative:
            failure_reason = "unknown" if search_error else ("hidden_content_returned" if hidden_leakage else None)
            final_outcome = "search_error" if search_error else ("hidden_negative_failed" if hidden_leakage else "hidden_negative_pass")
            if hidden_leakage:
                bucket_failure_reason_counts[bucket]["hidden_content_returned"] += 1
                overall_failure_reason_counts["hidden_content_returned"] += 1
            elif search_error:
                bucket_failure_reason_counts[bucket]["unknown"] += 1
                overall_failure_reason_counts["unknown"] += 1
            else:
                hidden_negative_pass_count += 1
            query_results.append(
                _base_query_result(
                    row,
                    candidate_index_version=candidate_index_version,
                    baseline_index_version=baseline_index_version,
                    label_status=label_status,
                )
                | {
                    "top_k_results": hit_details,
                    "hit_rank": None,
                    "location_rank": None,
                    "final_match_outcome": final_outcome,
                    "failure_reason": failure_reason,
                    "hidden_leakage": hidden_leakage,
                    "search_error": search_error,
                }
            )
            per_query.append(_legacy_per_query(row, None, False, hidden_leakage, hits[0] if hits else None, failure_reason))
            continue

        ranks.append(matched_rank)
        bucket_ranks[bucket].append(matched_rank)
        location_hits.append(location_ok)
        bucket_stats[bucket]["citation_location_accuracy"].append(location_ok)

        expected_type = _cell(row, "expected_location_type")
        if expected_type == "xlsx":
            xlsx_location_hits.append(location_ok)
            xlsx_breakdown = _xlsx_query_breakdown(row, hit_details)
            xlsx_file_hits.append(xlsx_breakdown["file_hit"])
            xlsx_sheet_hits.append(xlsx_breakdown["sheet_hit"])
            if xlsx_breakdown["table_applicable"]:
                xlsx_table_hits.append(xlsx_breakdown["table_hit"])
            if xlsx_breakdown["range_applicable"]:
                xlsx_range_overlap_hits.append(xlsx_breakdown["range_overlap"])
                xlsx_range_contains_hits.append(xlsx_breakdown["range_contains"])
                xlsx_exact_range_hits.append(xlsx_breakdown["exact_range"])
            _extend_bucket_stats(bucket_stats[bucket], "xlsx", xlsx_breakdown)
        if expected_type in {"pdf", "ocr"}:
            pdf_location_hits.append(location_ok)
            pdf_breakdown = _pdf_query_breakdown(row, hit_details)
            pdf_file_hits.append(pdf_breakdown["file_hit"])
            if pdf_breakdown["page_applicable"]:
                pdf_page_hits.append(pdf_breakdown["page_hit"])
            if pdf_breakdown["bbox_applicable"]:
                pdf_bbox_overlap_hits.append(pdf_breakdown["bbox_overlap"])
                pdf_exact_bbox_hits.append(pdf_breakdown["exact_bbox"])
            _extend_bucket_stats(bucket_stats[bucket], "pdf", pdf_breakdown)

        candidate_mismatch = _candidate_index_mismatch(hit_details, candidate_index_version)
        if candidate_mismatch:
            candidate_index_mismatch_count += 1

        failure_reason = None
        final_outcome = "matched"
        if hidden_leakage:
            failure_reason = "hidden_content_returned"
            final_outcome = "hidden_content_returned"
        elif search_error:
            failure_reason = "unknown"
            final_outcome = "search_error"
        elif not location_ok:
            failure_reason = _classify_failure_reason(
                row,
                hit_details,
                candidate_index_version=candidate_index_version,
            )
            final_outcome = "search_result_empty" if failure_reason == "search_result_empty" else "not_matched"

        if failure_reason:
            bucket_failure_reason_counts[bucket][failure_reason] += 1
            overall_failure_reason_counts[failure_reason] += 1

        query_results.append(
            _base_query_result(
                row,
                candidate_index_version=candidate_index_version,
                baseline_index_version=baseline_index_version,
                label_status=label_status,
            )
            | {
                "top_k_results": hit_details,
                "hit_rank": matched_rank,
                "location_rank": location_rank,
                "hit_at_1": matched_rank == 1,
                "hit_at_3": matched_rank is not None and matched_rank <= 3,
                "hit_at_5": matched_rank is not None and matched_rank <= 5,
                "hit_at_10": matched_rank is not None and matched_rank <= 10,
                "location_match": location_ok,
                "hidden_leakage": hidden_leakage,
                "final_match_outcome": final_outcome,
                "failure_reason": failure_reason,
                "search_error": search_error,
            }
        )
        per_query.append(_legacy_per_query(row, matched_rank, location_ok, hidden_leakage, hits[0] if hits else None, failure_reason))

    metrics = {
        "Hit@1": _hit_at(ranks, 1),
        "Hit@3": _hit_at(ranks, 3),
        "Hit@5": _hit_at(ranks, 5),
        "Hit@10": _hit_at(ranks, 10),
        "MRR@10": _mrr_at(ranks, 10),
        "hit_at_10": _hit_at(ranks, 10),
        "mrr_at_10": _mrr_at(ranks, 10),
        "citation_accuracy": _mean_bool(location_hits),
        "citation_location_accuracy": _mean_bool(location_hits),
        "xlsx_citation_location_accuracy": _mean_bool(xlsx_location_hits),
        "pdf_citation_location_accuracy": _mean_bool(pdf_location_hits),
        "xlsx_file_hit@10": _mean_bool(xlsx_file_hits),
        "xlsx_sheet_hit@10": _mean_bool(xlsx_sheet_hits),
        "xlsx_table_hit@10": _mean_bool(xlsx_table_hits),
        "xlsx_range_overlap@10": _mean_bool(xlsx_range_overlap_hits),
        "xlsx_range_contains@10": _mean_bool(xlsx_range_contains_hits),
        "xlsx_exact_range@10": _mean_bool(xlsx_exact_range_hits),
        "pdf_file_hit@10": _mean_bool(pdf_file_hits),
        "pdf_page_hit@10": _mean_bool(pdf_page_hits),
        "pdf_bbox_overlap@10": _mean_bool(pdf_bbox_overlap_hits),
        "pdf_exact_bbox@10": _mean_bool(pdf_exact_bbox_hits),
        "result_empty_count": result_empty_count,
        "gold_label_invalid_count": gold_label_invalid_count,
        "candidate_index_mismatch_count": overall_failure_reason_counts.get("candidate_index_mismatch", 0),
        "embedding_status_mismatch_count": overall_failure_reason_counts.get("embedding_status_mismatch", 0),
        "required_index_version_mismatch_count": overall_failure_reason_counts.get(
            "required_index_version_mismatch",
            0,
        ),
        "indexing_filtered_hit_count": indexing_filtered_hit_count,
        "search_error_count": search_error_count,
        "hidden_content_leakage_count": hidden_leakage_count,
        "hidden_negative_pass_count": hidden_negative_pass_count,
        "embedding_filtered_eval": bool(required_embedding_status or required_index_version),
        "required_embedding_status": required_embedding_status,
        "required_index_version": required_index_version,
        "overall_failure_reason_counts": dict(sorted(overall_failure_reason_counts.items())),
        "bucket_failure_reason_counts": {
            bucket_name: dict(sorted(counter.items()))
            for bucket_name, counter in sorted(bucket_failure_reason_counts.items())
        },
    }
    bucket_metrics = {
        bucket: _bucket_metrics_payload(
            bucket_rank_list,
            bucket_stats.get(bucket, {}),
            bucket_failure_reason_counts.get(bucket, Counter()),
        )
        for bucket, bucket_rank_list in sorted(bucket_ranks.items())
    }
    return {
        "status": "COMPLETED" if validation.ok else "COMPLETED_WITH_INVALID_GOLD",
        "validation": {
            "ok": validation.ok,
            "errors": validation.errors,
            "row_errors": validation.row_errors,
            "row_count": validation.row_count,
            "bucket_counts": validation.bucket_counts,
        },
        "metrics": metrics,
        "bucket_metrics": bucket_metrics,
        "per_query": per_query,
        "query_results": query_results,
    }


def _bucket_metrics_payload(
    ranks: list[int | None],
    stats: dict[str, list[bool]],
    failure_reason_counts: Counter[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Hit@10": _hit_at(ranks, 10),
        "MRR@10": _mrr_at(ranks, 10),
        "count": len(ranks),
    }
    for key, values in sorted(stats.items()):
        payload[key] = _mean_bool(values)
    if failure_reason_counts:
        payload["bucket_failure_reason_counts"] = dict(sorted(failure_reason_counts.items()))
    return payload


def _base_query_result(
    row: dict[str, str],
    *,
    candidate_index_version: str | None,
    baseline_index_version: str | None,
    label_status: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query_id": _cell(row, "query_id"),
        "bucket": _cell(row, "bucket"),
        "query": _cell(row, "query"),
        "expected_file_name": _cell(row, "expected_file_name"),
        "expected_sheet_name": _cell(row, "expected_sheet_name"),
        "expected_cell_range": _cell(row, "expected_cell_range"),
        "expected_table_id": _cell(row, "expected_table_id"),
        "expected_physical_page_index": _cell(row, "expected_physical_page_index"),
        "expected_page_no": _cell(row, "expected_page_no"),
        "expected_page_label": _cell(row, "expected_page_label"),
        "expected_bbox": _cell(row, "expected_bbox"),
        "candidate_index_version": candidate_index_version,
        "label_status": label_status,
    }
    if baseline_index_version:
        payload["baseline_index_version"] = baseline_index_version
    return payload


def _legacy_per_query(
    row: dict[str, str],
    matched_rank: int | None,
    location_ok: bool,
    hidden_leakage: bool,
    top_hit: dict[str, Any] | None,
    failure_reason: str | None,
) -> dict[str, Any]:
    return {
        "query_id": _cell(row, "query_id"),
        "bucket": _cell(row, "bucket"),
        "query": _cell(row, "query"),
        "hit_rank": matched_rank,
        "hit_at_1": matched_rank == 1,
        "hit_at_3": matched_rank is not None and matched_rank <= 3,
        "hit_at_5": matched_rank is not None and matched_rank <= 5,
        "hit_at_10": matched_rank is not None and matched_rank <= 10,
        "location_match": location_ok,
        "hidden_leakage": hidden_leakage,
        "failure_reason": failure_reason,
        "top_hit": _summarize_hit(top_hit) if top_hit else None,
    }


def _summarize_ranked_hit(
    row: dict[str, str],
    hit: dict[str, Any],
    *,
    rank: int,
    candidate_index_version: str | None,
    required_embedding_status: str | None,
    required_index_version: str | None,
) -> dict[str, Any]:
    source = hit.get("sourceFile") or {}
    unit = hit.get("searchUnit") or {}
    location = _location(hit)
    breakdown = _match_breakdown(
        row,
        hit,
        candidate_index_version=candidate_index_version,
        required_embedding_status=required_embedding_status,
        required_index_version=required_index_version,
    )
    return {
        "rank": rank,
        "search_unit_id": unit.get("id") or unit.get("searchUnitId") or hit.get("searchUnitId"),
        "score": hit.get("score") or unit.get("score"),
        "source_file_name": source.get("originalFileName") or source.get("fileName") or hit.get("sourceFileName"),
        "source_file_type": unit.get("sourceFileType") or source.get("fileType") or hit.get("sourceFileType"),
        "chunk_type": unit.get("chunkType"),
        "citation_text": unit.get("citationText") or (unit.get("citation") or {}).get("citationText"),
        "location_json": location,
        "parser_name": unit.get("parserName") or location.get("parser_name"),
        "parser_version": unit.get("parserVersion") or location.get("parser_version"),
        "embedding_status": unit.get("embeddingStatus") or hit.get("embeddingStatus"),
        "index_version": unit.get("indexVersion") or location.get("index_version") or hit.get("indexVersion"),
        "location_json_present": bool(location),
        "match_breakdown": breakdown,
    }


def _match_breakdown(
    row: dict[str, str],
    hit: dict[str, Any],
    *,
    candidate_index_version: str | None,
    required_embedding_status: str | None,
    required_index_version: str | None,
) -> dict[str, Any]:
    source = hit.get("sourceFile") or {}
    unit = hit.get("searchUnit") or {}
    location = _location(hit)
    expected_type = _cell(row, "expected_location_type")
    source_name = source.get("originalFileName") or source.get("fileName") or hit.get("sourceFileName")
    expected_file = _cell(row, "expected_file_name")
    file_match = not expected_file or source_name == expected_file
    document_version_id = _document_version_id(hit, location)
    expected_docv = _cell(row, "expected_document_version_id")
    document_version_match = not expected_docv or document_version_id == expected_docv
    expected_chunk = _cell(row, "expected_chunk_type")
    chunk_type_match = not expected_chunk or unit.get("chunkType") == expected_chunk
    actual_location_type = unit.get("locationType") or location.get("type")
    location_type_match = not expected_type or actual_location_type == expected_type
    index_version = unit.get("indexVersion") or location.get("index_version") or hit.get("indexVersion")
    embedding_status = unit.get("embeddingStatus") or hit.get("embeddingStatus")
    candidate_index_match = (
        True
        if not candidate_index_version or index_version in {None, "", candidate_index_version}
        else False
    )
    embedding_status_match = (
        True
        if not required_embedding_status
        else str(embedding_status or "").upper() == required_embedding_status.upper()
    )
    required_index_version_match = (
        True
        if not required_index_version
        else index_version == required_index_version
    )
    indexing_contract_match = embedding_status_match and required_index_version_match

    xlsx = _xlsx_hit_breakdown(row, location) if expected_type == "xlsx" else {}
    pdf = _pdf_hit_breakdown(row, location) if expected_type in {"pdf", "ocr"} else {}
    location_match = _location_matches(row, hit)
    return {
        "file_match": file_match,
        "document_version_match": document_version_match,
        "chunk_type_match": chunk_type_match,
        "location_type_match": location_type_match,
        "identity_match": (
            file_match
            and document_version_match
            and chunk_type_match
            and location_type_match
            and indexing_contract_match
        ),
        "location_match": location_match,
        "candidate_index_match": candidate_index_match,
        "embedding_status_match": embedding_status_match,
        "required_index_version_match": required_index_version_match,
        "indexing_contract_match": indexing_contract_match,
        **xlsx,
        **pdf,
    }


def _document_version_id(hit: dict[str, Any], location: dict[str, Any]) -> Any:
    unit = hit.get("searchUnit") or {}
    return (
        location.get("document_version_id")
        or location.get("documentVersionId")
        or unit.get("documentVersionId")
        or unit.get("document_version_id")
        or hit.get("documentVersionId")
    )


def _xlsx_hit_breakdown(row: dict[str, str], location: dict[str, Any]) -> dict[str, bool]:
    expected_sheet = _cell(row, "expected_sheet_name")
    expected_table = _cell(row, "expected_table_id")
    expected_range = _cell(row, "expected_cell_range")
    actual_range = str(location.get("cell_range") or "")
    return {
        "xlsx_sheet_match": not expected_sheet or location.get("sheet_name") == expected_sheet,
        "xlsx_table_match": not expected_table or location.get("table_id") == expected_table,
        "xlsx_range_exact": _range_relation(expected_range, actual_range, "exact_match"),
        "xlsx_range_contains": _range_relation(expected_range, actual_range, "contains_expected"),
        "xlsx_range_overlap": _range_relation(expected_range, actual_range, "overlaps_expected"),
        "xlsx_range_policy_match": _range_matches(expected_range, actual_range, _cell(row, "range_match_policy") or "none"),
    }


def _pdf_hit_breakdown(row: dict[str, str], location: dict[str, Any]) -> dict[str, bool]:
    page_match = _pdf_page_matches(row, location)
    expected_bbox = _bbox(_cell(row, "expected_bbox"))
    bbox_overlap = bool(expected_bbox and _bbox_overlaps(expected_bbox, location.get("bbox")))
    exact_bbox = bool(expected_bbox and _bbox_exact(expected_bbox, location.get("bbox")))
    return {
        "pdf_page_match": page_match,
        "pdf_bbox_overlap": bbox_overlap,
        "pdf_exact_bbox": exact_bbox,
    }


def _xlsx_query_breakdown(row: dict[str, str], hit_details: list[dict[str, Any]]) -> dict[str, bool]:
    expected_table = bool(_cell(row, "expected_table_id"))
    expected_range = bool(_cell(row, "expected_cell_range"))
    file_hits = [hit for hit in hit_details if hit["match_breakdown"]["file_match"]]
    sheet_hits = [hit for hit in file_hits if hit["match_breakdown"].get("xlsx_sheet_match", False)]
    table_hits = [hit for hit in sheet_hits if hit["match_breakdown"].get("xlsx_table_match", False)]
    range_scope = table_hits if expected_table else sheet_hits
    return {
        "file_hit": bool(file_hits),
        "sheet_hit": bool(sheet_hits),
        "table_applicable": expected_table,
        "table_hit": bool(table_hits),
        "range_applicable": expected_range,
        "range_overlap": any(hit["match_breakdown"].get("xlsx_range_overlap", False) for hit in range_scope),
        "range_contains": any(hit["match_breakdown"].get("xlsx_range_contains", False) for hit in range_scope),
        "exact_range": any(hit["match_breakdown"].get("xlsx_range_exact", False) for hit in range_scope),
    }


def _pdf_query_breakdown(row: dict[str, str], hit_details: list[dict[str, Any]]) -> dict[str, bool]:
    expected_page = bool(_cell(row, "expected_physical_page_index") or _cell(row, "expected_page_no") or _cell(row, "expected_page_label"))
    expected_bbox = bool(_cell(row, "expected_bbox"))
    file_hits = [hit for hit in hit_details if hit["match_breakdown"]["file_match"]]
    page_hits = [hit for hit in file_hits if hit["match_breakdown"].get("pdf_page_match", False)]
    return {
        "file_hit": bool(file_hits),
        "page_applicable": expected_page,
        "page_hit": bool(page_hits),
        "bbox_applicable": expected_bbox,
        "bbox_overlap": any(hit["match_breakdown"].get("pdf_bbox_overlap", False) for hit in page_hits),
        "exact_bbox": any(hit["match_breakdown"].get("pdf_exact_bbox", False) for hit in page_hits),
    }


def _extend_bucket_stats(stats: dict[str, list[bool]], prefix: str, breakdown: dict[str, bool]) -> None:
    for key, value in breakdown.items():
        if key.endswith("_applicable"):
            continue
        if key in {"table_hit", "range_overlap", "range_contains", "exact_range"} and not breakdown.get(
            "table_applicable" if key == "table_hit" else "range_applicable",
            False,
        ):
            continue
        if key in {"page_hit"} and not breakdown.get("page_applicable", False):
            continue
        if key in {"bbox_overlap", "exact_bbox"} and not breakdown.get("bbox_applicable", False):
            continue
        stats[f"{prefix}_{key}@10"].append(value)


def _classify_failure_reason(
    row: dict[str, str],
    hit_details: list[dict[str, Any]],
    *,
    candidate_index_version: str | None,
) -> str:
    if not hit_details:
        return "search_result_empty"
    if _candidate_index_mismatch(hit_details, candidate_index_version):
        return "candidate_index_mismatch"
    if not any(hit["match_breakdown"].get("required_index_version_match", True) for hit in hit_details):
        return "required_index_version_mismatch"
    if not any(hit["match_breakdown"].get("embedding_status_match", True) for hit in hit_details):
        return "embedding_status_mismatch"
    if not any(hit["match_breakdown"]["file_match"] for hit in hit_details):
        return "expected_file_not_found"
    expected_type = _cell(row, "expected_location_type")
    if expected_type == "xlsx":
        file_hits = [hit for hit in hit_details if hit["match_breakdown"]["file_match"]]
        sheet_hits = [hit for hit in file_hits if hit["match_breakdown"].get("xlsx_sheet_match", False)]
        if _cell(row, "expected_sheet_name") and not sheet_hits:
            return "expected_sheet_not_found"
        table_hits = [hit for hit in sheet_hits if hit["match_breakdown"].get("xlsx_table_match", False)]
        if _cell(row, "expected_table_id") and not table_hits:
            return "expected_table_not_found"
        range_scope = table_hits if _cell(row, "expected_table_id") else sheet_hits
        if _cell(row, "expected_cell_range") and not any(
            hit["match_breakdown"].get("xlsx_range_policy_match", False) for hit in range_scope
        ):
            return "expected_range_not_found"
    if expected_type in {"pdf", "ocr"}:
        file_hits = [hit for hit in hit_details if hit["match_breakdown"]["file_match"]]
        page_hits = [hit for hit in file_hits if hit["match_breakdown"].get("pdf_page_match", False)]
        if (_cell(row, "expected_physical_page_index") or _cell(row, "expected_page_no")) and not page_hits:
            return "expected_page_not_found"
        if _cell(row, "expected_bbox") and not any(
            hit["match_breakdown"].get("pdf_bbox_overlap", False) for hit in page_hits
        ):
            return "bbox_mismatch"
    return "unknown"


def _candidate_index_mismatch(hit_details: list[dict[str, Any]], candidate_index_version: str | None) -> bool:
    if not candidate_index_version or not hit_details:
        return False
    indexed_hits = [hit for hit in hit_details if hit.get("index_version")]
    return bool(indexed_hits) and not any(hit["index_version"] == candidate_index_version for hit in indexed_hits)


def _invalid_failure_reason(errors: list[str]) -> str:
    if any("unsupported bucket" in error for error in errors):
        return "unsupported_bucket"
    if any("range_match_policy" in error or "A1-style" in error or "bbox" in error for error in errors):
        return "match_policy_error"
    return "gold_label_invalid"


def search_library(base_url: str, *, timeout_seconds: float = 60.0) -> Callable[[str, int], list[dict[str, Any]]]:
    client = httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def _search(query: str, top_k: int) -> list[dict[str, Any]]:
        response = client.get("/api/v1/library/search", params={"query": query, "limit": top_k})
        response.raise_for_status()
        return list(response.json().get("results") or [])

    return _search


def _hit_matches(row: dict[str, str], hit: dict[str, Any]) -> bool:
    source = hit.get("sourceFile") or {}
    unit = hit.get("searchUnit") or {}
    location = _location(hit)

    expected_file = _cell(row, "expected_file_name")
    source_name = source.get("originalFileName") or source.get("fileName") or hit.get("sourceFileName")
    if expected_file and source_name != expected_file:
        return False
    expected_docv = _cell(row, "expected_document_version_id")
    document_version_id = (
        location.get("document_version_id")
        or location.get("documentVersionId")
        or unit.get("documentVersionId")
        or unit.get("document_version_id")
        or hit.get("documentVersionId")
    )
    if expected_docv and document_version_id != expected_docv:
        return False
    expected_chunk = _cell(row, "expected_chunk_type")
    if expected_chunk and unit.get("chunkType") != expected_chunk:
        return False
    expected_type = _cell(row, "expected_location_type")
    if expected_type and (unit.get("locationType") or location.get("type")) != expected_type:
        return False
    return True


def _location_matches(row: dict[str, str], hit: dict[str, Any]) -> bool:
    location = _location(hit)
    expected_type = _cell(row, "expected_location_type")
    if expected_type == "xlsx":
        expected_sheet = _cell(row, "expected_sheet_name")
        if expected_sheet and location.get("sheet_name") != expected_sheet:
            return False
        expected_table = _cell(row, "expected_table_id")
        if expected_table and location.get("table_id") != expected_table:
            return False
        policy = _cell(row, "range_match_policy") or "none"
        return _range_matches(_cell(row, "expected_cell_range"), location.get("cell_range"), policy)
    if expected_type in {"pdf", "ocr"}:
        expected_physical = _int(_cell(row, "expected_physical_page_index"))
        expected_page = _int(_cell(row, "expected_page_no"))
        expected_label = _cell(row, "expected_page_label")
        if expected_physical is not None and location.get("physical_page_index") != expected_physical:
            return False
        if expected_page is not None and location.get("page_no") != expected_page:
            return False
        if expected_label and str(location.get("page_label")) != expected_label:
            return False
        expected_bbox = _bbox(_cell(row, "expected_bbox"))
        if expected_bbox:
            return _bbox_overlaps(expected_bbox, location.get("bbox"))
        return True
    return True


def _range_matches(expected: str, actual: str | None, policy: str) -> bool:
    if policy == "none" or not expected:
        return True
    if not actual:
        return False
    expected_range = _cell_range(expected)
    actual_range = _cell_range(str(actual))
    if expected_range is None or actual_range is None:
        return expected.strip().upper() == str(actual).strip().upper()
    if policy == "exact_match":
        return expected_range == actual_range
    if policy == "contains_expected":
        return _contains(actual_range, expected_range)
    if policy == "overlaps_expected":
        return _overlaps(actual_range, expected_range)
    return False


def _range_relation(expected: str, actual: str | None, policy: str) -> bool:
    if not expected or not actual:
        return False
    return _range_matches(expected, actual, policy)


def _pdf_page_matches(row: dict[str, str], location: dict[str, Any]) -> bool:
    expected_physical = _int(_cell(row, "expected_physical_page_index"))
    expected_page = _int(_cell(row, "expected_page_no"))
    expected_label = _cell(row, "expected_page_label")
    if expected_physical is not None and location.get("physical_page_index") != expected_physical:
        return False
    if expected_page is not None and location.get("page_no") != expected_page:
        return False
    if expected_label and str(location.get("page_label")) != expected_label:
        return False
    return True


def _hidden_leakage(row: dict[str, str], hits: list[dict[str, Any]]) -> bool:
    if _cell(row, "hidden_policy") not in {"exclude_hidden", "negative"}:
        return False
    blocked_terms = [term.lower() for term in _split_terms(_cell(row, "must_not_contain_terms"))]
    for hit in hits:
        text = json.dumps(hit, ensure_ascii=False, default=str).lower()
        if any(term and term in text for term in blocked_terms):
            return True
    return False


def _summarize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    source = hit.get("sourceFile") or {}
    unit = hit.get("searchUnit") or {}
    location = _location(hit)
    return {
        "search_unit_id": unit.get("id") or unit.get("searchUnitId") or hit.get("searchUnitId"),
        "source_file_name": source.get("originalFileName"),
        "chunk_type": unit.get("chunkType"),
        "location_type": unit.get("locationType"),
        "citation_text": unit.get("citationText"),
        "embedding_status": unit.get("embeddingStatus") or hit.get("embeddingStatus"),
        "index_version": unit.get("indexVersion") or location.get("index_version") or hit.get("indexVersion"),
        "location_json_present": bool(location),
    }


def _hit_at(ranks: Iterable[int | None], k: int) -> float:
    rank_list = list(ranks)
    if not rank_list:
        return 0.0
    return round(sum(1 for rank in rank_list if rank is not None and rank <= k) / len(rank_list), 4)


def _mrr_at(ranks: Iterable[int | None], k: int) -> float:
    rank_list = list(ranks)
    if not rank_list:
        return 0.0
    total = sum(1.0 / rank for rank in rank_list if rank is not None and rank <= k)
    return round(total / len(rank_list), 4)


def _mean_bool(values: list[bool]) -> float:
    if not values:
        return 0.0
    return round(sum(1 for value in values if value) / len(values), 4)


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _location(hit: dict[str, Any]) -> dict[str, Any]:
    unit = hit.get("searchUnit") or {}
    citation = unit.get("citation") or {}
    return _json(unit.get("locationJson")) or _json(citation.get("locationJson")) or {}


def _cell(row: dict[str, str], key: str) -> str:
    return str(row.get(key) or "").strip()


def _split_terms(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.replace("|", ";").split(";") if part.strip()]


def _int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cell_range(value: str) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    text = value.replace("$", "").strip().upper()
    parts = text.split(":", 1)
    start = _cell_ref(parts[0])
    end = _cell_ref(parts[1] if len(parts) == 2 else parts[0])
    if start is None or end is None:
        return None
    return (
        min(start[0], end[0]),
        min(start[1], end[1]),
        max(start[0], end[0]),
        max(start[1], end[1]),
    )


def _cell_ref(value: str) -> tuple[int, int] | None:
    letters = "".join(ch for ch in value if ch.isalpha())
    digits = "".join(ch for ch in value if ch.isdigit())
    if not letters or not digits:
        return None
    column = 0
    for char in letters:
        column = column * 26 + (ord(char) - ord("A") + 1)
    return (int(digits), column)


def _contains(actual: tuple[int, int, int, int], expected: tuple[int, int, int, int]) -> bool:
    return (
        actual[0] <= expected[0]
        and actual[1] <= expected[1]
        and actual[2] >= expected[2]
        and actual[3] >= expected[3]
    )


def _overlaps(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def _bbox(value: str) -> list[float] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in value.split(",")]
    if not isinstance(parsed, list) or len(parsed) != 4:
        return None
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None


def _bbox_overlaps(expected: list[float], actual: Any) -> bool:
    if not isinstance(actual, list) or len(actual) != 4:
        return False
    try:
        actual_box = [float(item) for item in actual]
    except (TypeError, ValueError):
        return False
    left = (expected[1], expected[0], expected[3], expected[2])
    right = (actual_box[1], actual_box[0], actual_box[3], actual_box[2])
    return _overlaps(left, right)


def _bbox_exact(expected: list[float], actual: Any, *, tolerance: float = 0.01) -> bool:
    if not isinstance(actual, list) or len(actual) != 4:
        return False
    try:
        actual_box = [float(item) for item in actual]
    except (TypeError, ValueError):
        return False
    return all(abs(left - right) <= tolerance for left, right in zip(expected, actual_box))


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_report(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        print(json.dumps(payload, ensure_ascii=True, indent=2))


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="eval/gold_queries_v0.csv")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", default="reports/rag_retrieval_eval_report.json")
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--candidate-index-version")
    parser.add_argument("--baseline-index-version")
    parser.add_argument(
        "--required-embedding-status",
        default="EMBEDDED",
        help="Only count hits with this SearchUnit embedding status. Use an empty value to disable.",
    )
    parser.add_argument(
        "--required-index-version",
        default=None,
        help="Only count hits from this index_version. Defaults to --candidate-index-version when provided.",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_gold_csv(Path(args.gold))
    validation = validate_gold_rows(rows, require_live_bound=False)
    validation_payload = {
        "ok": validation.ok,
        "errors": validation.errors,
        "row_errors": validation.row_errors,
        "row_count": validation.row_count,
        "bucket_counts": validation.bucket_counts,
    }
    if args.validate_only:
        payload = {
            "run_id": utc_run_id(),
            "status": "PASSED" if validation.ok else "FAILED",
            "validation": validation_payload,
        }
        write_report(Path(args.report), payload)
        print_report(payload)
        return 0 if validation.ok else 1
    if _has_fatal_validation_errors(validation):
        payload = {
            "run_id": utc_run_id(),
            "status": "VALIDATION_FAILED",
            "validation": validation_payload,
        }
        write_report(Path(args.report), payload)
        print_report(payload)
        return 1

    report = evaluate_gold_rows(
        rows,
        search_fn=search_library(args.base_url, timeout_seconds=args.request_timeout_seconds),
        top_k=args.top_k,
        candidate_index_version=args.candidate_index_version,
        baseline_index_version=args.baseline_index_version,
        required_embedding_status=args.required_embedding_status or None,
        required_index_version=args.required_index_version or args.candidate_index_version,
    )
    report["run_id"] = utc_run_id()
    report["gold"] = str(args.gold)
    report["top_k"] = args.top_k
    report["candidate_index_version"] = args.candidate_index_version
    if args.baseline_index_version:
        report["baseline_index_version"] = args.baseline_index_version
    report["required_embedding_status"] = args.required_embedding_status or None
    report["required_index_version"] = args.required_index_version or args.candidate_index_version
    write_report(Path(args.report), report)
    print_report(report)
    return 0 if report["validation"]["ok"] else 1


def _has_fatal_validation_errors(validation: GoldValidationResult) -> bool:
    return any(
        error.startswith("missing required columns") or error == "gold CSV must contain at least one row"
        for error in validation.errors
    )


if __name__ == "__main__":
    sys.exit(main())
