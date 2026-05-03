"""Rebind and extend RAG ingestion gold queries from the live catalog DB.

This helper keeps eval/gold_queries_v0.csv useful after repeated smoke runs.
It updates expected_document_version_id/page/bbox bindings against the latest
ingested document version for each file and can append conservative,
location-bound seed queries up to a target row count.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AI_WORKER = ROOT / "ai-worker"
if str(AI_WORKER) not in sys.path:
    sys.path.insert(0, str(AI_WORKER))

from eval.harness.rag_ingestion_retrieval_eval import REQUIRED_COLUMNS, validate_gold_rows  # noqa: E402


DEFAULT_DSN = "postgresql://aipipeline:aipipeline_pw@localhost:5433/aipipeline"
DEFAULT_GOLD = Path("eval/gold_queries_v0.csv")
DEFAULT_REPORT = Path("reports/rag_gold_query_rebind_report.json")


@dataclass(frozen=True)
class CatalogUnit:
    source_file_name: str
    document_version_id: str
    chunk_type: str
    location_type: str
    location: dict[str, Any]
    citation_text: str
    text: str


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_csv(Path(args.gold))
    original_count = len(rows)

    connection = connect(args.db_dsn)
    rebound_count = 0
    missing_bindings: list[str] = []
    for row in rows:
        rebound = rebind_row(connection, row)
        if rebound is None:
            missing_bindings.append(row.get("query_id") or "<missing query_id>")
            continue
        if rebound != row.get("expected_document_version_id"):
            rebound_count += 1
        row["expected_document_version_id"] = rebound

    appended_rows: list[dict[str, str]] = []
    if not args.no_append and len(rows) < args.target_count:
        appended_rows = build_candidate_rows(connection, rows, target_count=args.target_count)
        rows.extend(appended_rows)

    validation = validate_gold_rows(rows, require_live_bound=True)
    output = Path(args.output or args.gold)
    if args.dry_run:
        output = Path(args.output or args.gold).with_suffix(".dry-run.csv")
    write_csv(output, rows)

    report = {
        "run_id": utc_run_id(),
        "input": str(args.gold),
        "output": str(output),
        "original_count": original_count,
        "final_count": len(rows),
        "rebound_count": rebound_count,
        "appended_count": len(appended_rows),
        "missing_binding_count": len(missing_bindings),
        "missing_bindings": missing_bindings,
        "bucket_counts": dict(Counter(row["bucket"] for row in rows)),
        "validation": {
            "ok": validation.ok,
            "errors": validation.errors,
            "row_count": validation.row_count,
            "bucket_counts": validation.bucket_counts,
        },
    }
    write_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if validation.ok else 1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--output", help="CSV output path. Defaults to updating --gold.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--db-dsn", default=DEFAULT_DSN)
    parser.add_argument("--target-count", type=int, default=72)
    parser.add_argument("--no-append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def connect(dsn: str):
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("psycopg2 is required for live gold query rebinding") from exc
    return psycopg2.connect(dsn)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in REQUIRED_COLUMNS} for row in rows])


def rebind_row(connection: Any, row: dict[str, str]) -> str | None:
    query = """
        select document_version_id, location_json::text
        from search_unit
        where source_file_name = %s
          and location_type = %s
          and (%s = '' or chunk_type = %s)
          and (%s = '' or location_json->>'sheet_name' = %s)
          and (%s = '' or location_json->>'cell_range' = %s)
          and (%s = '' or location_json->>'physical_page_index' = %s)
          and (%s = '' or location_json->>'page_no' = %s)
          and (%s = '' or location_json->>'page_label' = %s)
        order by created_at desc
        limit 20
    """
    params = (
        row.get("expected_file_name", ""),
        row.get("expected_location_type", ""),
        row.get("expected_chunk_type", ""),
        row.get("expected_chunk_type", ""),
        row.get("expected_sheet_name", ""),
        row.get("expected_sheet_name", ""),
        row.get("expected_cell_range", ""),
        row.get("expected_cell_range", ""),
        row.get("expected_physical_page_index", ""),
        row.get("expected_physical_page_index", ""),
        row.get("expected_page_no", ""),
        row.get("expected_page_no", ""),
        row.get("expected_page_label", ""),
        row.get("expected_page_label", ""),
    )
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        candidates = cursor.fetchall()
    expected_bbox = parse_bbox(row.get("expected_bbox", ""))
    for docv, location_json in candidates:
        location = parse_location(location_json)
        if expected_bbox and not bbox_overlaps(expected_bbox, location.get("bbox")):
            continue
        if row.get("expected_location_type") in {"pdf", "ocr"} and location.get("bbox"):
            row["expected_bbox"] = json.dumps(location.get("bbox"), ensure_ascii=False, separators=(",", ":"))
        return str(docv)
    return None


def build_candidate_rows(connection: Any, existing_rows: list[dict[str, str]], *, target_count: int) -> list[dict[str, str]]:
    needed = max(0, target_count - len(existing_rows))
    if needed == 0:
        return []
    existing_keys = {
        (
            row.get("bucket", ""),
            row.get("query", ""),
            row.get("expected_file_name", ""),
            row.get("expected_cell_range", ""),
            row.get("expected_page_no", ""),
        )
        for row in existing_rows
    }
    next_index = next_query_index(existing_rows)
    candidate_rows: list[dict[str, str]] = []
    for unit in load_latest_units(connection):
        row = row_from_unit(unit, 0)
        if row is None:
            continue
        key = (
            row.get("bucket", ""),
            row.get("query", ""),
            row.get("expected_file_name", ""),
            row.get("expected_cell_range", ""),
            row.get("expected_page_no", ""),
        )
        if key in existing_keys:
            continue
        existing_keys.add(key)
        candidate_rows.append(row)
    return select_balanced_candidates(existing_rows, candidate_rows, needed=needed, first_index=next_index)


def select_balanced_candidates(
    existing_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    needed: int,
    first_index: int,
) -> list[dict[str, str]]:
    bucket_targets = {
        "xlsx_lookup": 14,
        "xlsx_header_ambiguous": 6,
        "xlsx_formula_value": 4,
        "xlsx_date_number_format": 8,
        "xlsx_hidden_policy": 4,
        "xlsx_aggregation": 8,
        "mixed_text_table": 4,
        "pdf_page_lookup": 8,
        "pdf_section_question": 8,
        "pdf_table_lookup": 6,
        "pdf_ocr_noise": 0,
    }
    counts = Counter(row.get("bucket", "") for row in existing_rows)
    selected: list[dict[str, str]] = []
    used: set[int] = set()
    priority = [
        "pdf_page_lookup",
        "pdf_section_question",
        "pdf_table_lookup",
        "xlsx_date_number_format",
        "xlsx_aggregation",
        "xlsx_lookup",
        "xlsx_header_ambiguous",
        "mixed_text_table",
    ]

    while len(selected) < needed:
        progress = False
        for bucket in priority:
            if len(selected) >= needed:
                break
            if counts[bucket] >= bucket_targets[bucket]:
                continue
            index = find_next_candidate(candidate_rows, used, bucket)
            if index is None:
                continue
            used.add(index)
            row = dict(candidate_rows[index])
            row["query_id"] = f"gq_auto_{first_index + len(selected):03d}"
            selected.append(row)
            counts[bucket] += 1
            progress = True
        if not progress:
            break

    for index, row in enumerate(candidate_rows):
        if len(selected) >= needed:
            break
        if index in used:
            continue
        row = dict(row)
        row["query_id"] = f"gq_auto_{first_index + len(selected):03d}"
        selected.append(row)
    return selected


def find_next_candidate(candidate_rows: list[dict[str, str]], used: set[int], bucket: str) -> int | None:
    for index, row in enumerate(candidate_rows):
        if index not in used and row.get("bucket") == bucket:
            return index
    return None


def load_latest_units(connection: Any) -> list[CatalogUnit]:
    sql = """
        with latest_doc as (
          select source_file_name, document_version_id
          from (
            select
              source_file_name,
              document_version_id,
              max(created_at) as latest_unit,
              row_number() over (
                partition by source_file_name
                order by max(created_at) desc
              ) as rn
            from search_unit
            where source_file_type in ('SPREADSHEET', 'PDF')
              and document_version_id is not null
            group by source_file_name, document_version_id
          ) ranked
          where rn = 1
        )
        select
          unit.source_file_name,
          unit.document_version_id,
          unit.chunk_type,
          unit.location_type,
          unit.location_json::text,
          coalesce(unit.citation_text, ''),
          coalesce(unit.display_text, unit.bm25_text, unit.text_content, '')
        from search_unit unit
        join latest_doc latest
          on latest.source_file_name = unit.source_file_name
         and latest.document_version_id = unit.document_version_id
        where unit.chunk_type in ('row_group', 'table', 'paragraph', 'page')
          and length(coalesce(unit.display_text, unit.bm25_text, unit.text_content, '')) >= 20
        order by
          case unit.location_type when 'xlsx' then 0 else 1 end,
          unit.source_file_name,
          case unit.chunk_type
            when 'row_group' then 0
            when 'table' then 1
            when 'paragraph' then 2
            when 'page' then 3
            else 4
          end,
          unit.created_at desc
    """
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
    return [
        CatalogUnit(
            source_file_name=str(row[0]),
            document_version_id=str(row[1]),
            chunk_type=str(row[2]),
            location_type=str(row[3]),
            location=parse_location(row[4]),
            citation_text=str(row[5] or ""),
            text=str(row[6] or ""),
        )
        for row in rows
    ]


def row_from_unit(unit: CatalogUnit, index: int) -> dict[str, str] | None:
    if unit.location_type == "xlsx":
        return xlsx_row_from_unit(unit, index)
    if unit.location_type == "pdf":
        return pdf_row_from_unit(unit, index)
    return None


def xlsx_row_from_unit(unit: CatalogUnit, index: int) -> dict[str, str] | None:
    line_name = first_match(unit.text, [r"노선명:\s*([^|\\n]+)"])
    query = first_match(unit.text, [
        r"장기요양기관이름:\s*([^|\\n]+)",
        r"시도 시군구 법정동명:\s*([^|\\n]+)",
        r"노선명:\s*([^|\\n]+)",
        r"지정일자:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
        r"승차총승객수:\s*([0-9,]+)",
    ])
    if not query:
        query = xlsx_header_query(unit.text)
    if not good_query(query):
        return None

    bucket = "xlsx_lookup"
    requires_formatted = "false"
    requires_aggregation = "false"
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", query) or "," in query:
        bucket = "xlsx_date_number_format"
        requires_formatted = "true"
    elif line_name and "승차총승객수" in unit.text:
        query = f"{line_name} 승차총승객수"
        bucket = "xlsx_aggregation"
        requires_aggregation = "true"
    elif query in {"대중교통구분", "승차총승객수", "장기요양기관이름", "지정일자", "설치신고일자"}:
        bucket = "xlsx_header_ambiguous"

    return base_row(index, bucket, query, unit) | {
        "expected_location_type": "xlsx",
        "expected_sheet_name": str(unit.location.get("sheet_name") or ""),
        "expected_cell_range": str(unit.location.get("cell_range") or ""),
        "expected_table_id": str(unit.location.get("table_id") or ""),
        "expected_answer_text": query,
        "must_contain_terms": query,
        "range_match_policy": "contains_expected",
        "hidden_policy": "exclude_hidden",
        "requires_formula_value": "false",
        "requires_formatted_value": requires_formatted,
        "requires_aggregation": requires_aggregation,
        "source_sample_id": "live_auto_p1_xlsx",
        "notes": "auto-bound seed from current normalized XLSX search_unit",
    }


def pdf_row_from_unit(unit: CatalogUnit, index: int) -> dict[str, str] | None:
    query = pdf_phrase(unit.text)
    if not good_query(query):
        return None
    bucket = "pdf_page_lookup"
    if unit.chunk_type == "page":
        bucket = "pdf_section_question"
    if re.search(r"[0-9][0-9.,%]*", query) and unit.chunk_type in {"page", "paragraph"}:
        bucket = "pdf_table_lookup"

    location = unit.location
    bbox = location.get("bbox")
    return base_row(index, bucket, query, unit) | {
        "expected_location_type": "pdf",
        "expected_physical_page_index": str(location.get("physical_page_index") or ""),
        "expected_page_no": str(location.get("page_no") or ""),
        "expected_page_label": str(location.get("page_label") or ""),
        "expected_bbox": json.dumps(bbox, ensure_ascii=False, separators=(",", ":")) if bbox else "",
        "expected_answer_text": query,
        "must_contain_terms": query,
        "range_match_policy": "none",
        "hidden_policy": "include_hidden",
        "requires_formula_value": "false",
        "requires_formatted_value": "false",
        "requires_aggregation": "false",
        "source_sample_id": "live_auto_p1_pdf",
        "notes": "auto-bound seed from current normalized PDF search_unit",
    }


def base_row(index: int, bucket: str, query: str, unit: CatalogUnit) -> dict[str, str]:
    return {column: "" for column in REQUIRED_COLUMNS} | {
        "query_id": f"gq_auto_{index:03d}",
        "bucket": bucket,
        "query": query,
        "expected_file_name": unit.source_file_name,
        "expected_document_version_id": unit.document_version_id,
        "expected_chunk_type": unit.chunk_type,
        "label_status": "bound",
    }


def first_match(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return clean_query(match.group(1))
    return ""


def xlsx_header_query(text: str) -> str:
    for label in ("대중교통구분", "승차총승객수", "장기요양기관이름", "지정일자", "설치신고일자"):
        if label in text:
            return label
    return ""


def pdf_phrase(text: str) -> str:
    for raw_line in text.splitlines():
        line = clean_query(raw_line)
        if not good_query(line):
            continue
        if len(line) > 36:
            line = clean_query(re.split(r"[·.]{3,}|\\s{2,}", line)[0])
        if good_query(line):
            return line
    return ""


def clean_query(value: str) -> str:
    value = re.sub(r"[\[\]{}()|]+", " ", value)
    value = re.sub(r"[-_]{2,}", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" :;,.")


def good_query(value: str) -> bool:
    if not value:
        return False
    if len(value) < 2 or len(value) > 40:
        return False
    if re.search(r"[A-Za-z]", value):
        return False
    if re.fullmatch(r"[0-9,.\s]+", value) and len(re.sub(r"\D", "", value)) < 4:
        return False
    if value.count("●") or value.count("·") > 3:
        return False
    has_hangul = re.search(r"[\uac00-\ud7a3]", value) is not None
    has_digit = re.search(r"\d", value) is not None
    return has_hangul or has_digit


def next_query_index(rows: list[dict[str, str]]) -> int:
    max_index = 0
    for row in rows:
        match = re.search(r"(\d+)$", row.get("query_id", ""))
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def parse_location(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_bbox(value: str) -> list[float] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in value.split(",")]
    if not isinstance(parsed, list) or len(parsed) != 4:
        return None
    try:
        return [float(part) for part in parsed]
    except (TypeError, ValueError):
        return None


def bbox_overlaps(left: list[float], right: Any) -> bool:
    if not isinstance(right, list) or len(right) != 4:
        return False
    try:
        right_box = [float(part) for part in right]
    except (TypeError, ValueError):
        return False
    return not (
        left[2] < right_box[0]
        or right_box[2] < left[0]
        or left[3] < right_box[1]
        or right_box[3] < left[1]
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


if __name__ == "__main__":
    sys.exit(main())
