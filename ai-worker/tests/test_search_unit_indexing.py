from __future__ import annotations

from app.capabilities.rag.retrieval_contract import citation_payload
from app.capabilities.rag.search_unit_indexing import (
    build_search_unit_embedding_text,
    document_from_claim,
    index_metadata,
    stable_index_id,
    to_chunk_row,
    to_document_row,
)
from app.capabilities.rag.generation import RetrievedChunk


def test_pdf_search_unit_embedding_text_includes_source_title_page_and_text():
    doc = document_from_claim({
        "searchUnitId": "unit-pdf-page-1",
        "claimToken": "claim-1",
        "indexId": "source_file:source-file-1:unit:PAGE:page:2",
        "sourceFileId": "source-file-1",
        "sourceFileName": "policy.pdf",
        "extractedArtifactId": "artifact-1",
        "artifactType": "OCR_RESULT_JSON",
        "unitType": "PAGE",
        "unitKey": "page:2",
        "title": "Refund Policy",
        "sectionPath": "Policy > Refunds",
        "pageStart": 2,
        "pageEnd": 2,
        "textContent": "Refunds are processed within seven business days.",
        "contentSha256": "hash-pdf-page",
    })

    built = build_search_unit_embedding_text(doc)

    assert built.variant == "retrieval_title_section_search_unit_v1"
    assert "Source: policy.pdf" in built.text
    assert "Title: Refund Policy" in built.text
    assert "Section: Policy > Refunds" in built.text
    assert "Page: 2" in built.text
    assert "Refunds are processed" in built.text


def test_xlsx_table_embedding_text_includes_sheet_range_header_and_table_text():
    doc = document_from_claim({
        "searchUnitId": "unit-xlsx-table-1",
        "claimToken": "claim-1",
        "indexId": "source_file:source-file-1:unit:TABLE:sheet:0:table:0:A1-F24",
        "sourceFileId": "source-file-1",
        "sourceFileName": "sales_report_2024.xlsx",
        "extractedArtifactId": "artifact-xlsx",
        "artifactType": "XLSX_WORKBOOK_JSON",
        "unitType": "TABLE",
        "unitKey": "sheet:0:table:0:A1-F24",
        "title": "SalesTable",
        "sectionPath": "workbook/Sales",
        "textContent": "| Region | Revenue |\n| --- | --- |\n| Seoul | 12000000 |",
        "contentSha256": "hash-xlsx-table",
        "metadataJson": {
            "fileType": "xlsx",
            "sheetName": "Sales",
            "sheetIndex": 0,
            "cellRange": "A1:F24",
            "headers": ["Region", "Revenue"],
            "tableId": "SalesTable",
            "rowStart": 1,
            "rowEnd": 24,
            "columnStart": 1,
            "columnEnd": 6,
        },
    })

    built = build_search_unit_embedding_text(doc)
    chunk = to_chunk_row(
        doc,
        faiss_row_id=3,
        index_version="idx-v1",
        embedding_model="hashing-test",
        embedding_text=built,
    )

    assert built.variant == "retrieval_title_section_spreadsheet_v1"
    assert "Source: sales_report_2024.xlsx" in built.text
    assert "Sheet: Sales" in built.text
    assert "Range: A1:F24" in built.text
    assert "Headers: Region | Revenue" in built.text
    assert "| Region | Revenue |" in built.text
    assert chunk.extra["sheetName"] == "Sales"
    assert chunk.extra["cellRange"] == "A1:F24"
    assert chunk.extra["rowStart"] == 1
    assert chunk.extra["columnEnd"] == 6
    assert chunk.extra["embeddingModel"] == "hashing-test"
    assert chunk.extra["embeddingTextVariant"] == "retrieval_title_section_spreadsheet_v1"


def test_xlsx_chunk_embedding_text_keeps_repeated_header_context():
    doc = document_from_claim({
        "searchUnitId": "unit-xlsx-chunk-1",
        "claimToken": "claim-1",
        "indexId": "source_file:source-file-1:unit:CHUNK:sheet:0:chunk:1:A52-B101",
        "sourceFileId": "source-file-1",
        "sourceFileName": "sales_report_2024.xlsx",
        "extractedArtifactId": "artifact-xlsx",
        "artifactType": "XLSX_WORKBOOK_JSON",
        "unitType": "CHUNK",
        "unitKey": "sheet:0:chunk:1:A52-B101",
        "title": "Sales A52:B101",
        "sectionPath": "workbook/Sales",
        "textContent": "Region: Seoul | Revenue: 12000000\nRegion: Busan | Revenue: 9000000",
        "contentSha256": "hash-xlsx-chunk",
        "metadataJson": {
            "fileType": "xlsx",
            "sheetName": "Sales",
            "cellRange": "A52:B101",
            "headers": ["Region", "Revenue"],
            "rowStart": 52,
            "rowEnd": 101,
            "columnStart": 1,
            "columnEnd": 2,
        },
    })

    built = build_search_unit_embedding_text(doc)

    assert "Headers: Region | Revenue" in built.text
    assert "Region: Seoul" in built.text
    assert "Revenue: 12000000" in built.text
    assert "rowStart" not in built.text
    assert "rowEnd" not in built.text


def test_xlsx_workbook_document_embedding_text_uses_sheet_list_not_full_text():
    doc = document_from_claim({
        "searchUnitId": "unit-xlsx-workbook",
        "claimToken": "claim-1",
        "indexId": "source_file:source-file-1:unit:DOCUMENT:workbook",
        "sourceFileId": "source-file-1",
        "sourceFileName": "sales_report_2024.xlsx",
        "extractedArtifactId": "artifact-xlsx",
        "artifactType": "XLSX_WORKBOOK_JSON",
        "unitType": "DOCUMENT",
        "unitKey": "workbook",
        "title": "sales_report_2024.xlsx",
        "textContent": "FULL WORKBOOK TEXT SHOULD NOT BE EMBEDDED",
        "contentSha256": "hash-xlsx-workbook",
        "metadataJson": {
            "fileType": "xlsx",
            "role": "workbook",
            "sheetNames": ["Sales", "Summary"],
        },
    })

    built = build_search_unit_embedding_text(doc)

    assert "Workbook: sales_report_2024.xlsx" in built.text
    assert "Sheets: Sales, Summary" in built.text
    assert "FULL WORKBOOK TEXT SHOULD NOT BE EMBEDDED" not in built.text


def test_search_unit_claim_maps_to_stable_chunk_metadata():
    payload = {
        "searchUnitId": "unit-1",
        "claimToken": "claim-1",
        "indexId": "source_file:source-file-1:unit:TABLE:page:2:table:1",
        "sourceFileId": "source-file-1",
        "sourceFileName": "receipt.pdf",
        "extractedArtifactId": "artifact-1",
        "artifactType": "OCR_RESULT_JSON",
        "unitType": "TABLE",
        "unitKey": "page:2:table:1",
        "title": "Totals",
        "sectionPath": "Invoice > Totals",
        "pageStart": 2,
        "pageEnd": 2,
        "textContent": "Item\tPrice\nTea\t3000",
        "contentSha256": "hash-1",
        "metadataJson": "{\"rowCount\":2}",
        "indexMetadata": {"content_hash": "hash-1"},
    }

    doc = document_from_claim(payload)
    document_row = to_document_row(doc)
    chunk_row = to_chunk_row(doc, faiss_row_id=7, index_version="idx-v1")

    assert stable_index_id("source-file-1", "TABLE", "page:2:table:1") == payload["indexId"]
    assert document_row.doc_id == "source-file-1"
    assert chunk_row.chunk_id == payload["indexId"]
    assert chunk_row.doc_id == "source-file-1"
    assert chunk_row.faiss_row_id == 7
    assert chunk_row.extra["searchUnitId"] == "unit-1"
    assert chunk_row.extra["indexId"] == payload["indexId"]
    assert chunk_row.extra["unitType"] == "TABLE"
    assert chunk_row.extra["unitKey"] == "page:2:table:1"
    assert chunk_row.extra["artifactType"] == "OCR_RESULT_JSON"
    assert chunk_row.extra["pageStart"] == 2
    assert chunk_row.extra["contentSha256"] == "hash-1"


def test_index_metadata_drops_none_values_and_keeps_search_unit_keys():
    doc = document_from_claim({
        "searchUnitId": "unit-image-1",
        "claimToken": "claim-1",
        "indexId": "source_file:source-file-1:unit:IMAGE:page:3:image:1",
        "sourceFileId": "source-file-1",
        "unitType": "IMAGE",
        "unitKey": "page:3:image:1",
        "pageStart": 3,
        "pageEnd": 3,
        "textContent": "architecture diagram",
        "contentSha256": "hash-image",
    })

    metadata = index_metadata(doc)

    assert metadata["search_unit_id"] == "unit-image-1"
    assert metadata["searchUnitId"] == "unit-image-1"
    assert metadata["unit_type"] == "IMAGE"
    assert metadata["unitKey"] == "page:3:image:1"
    assert metadata["content_hash"] == "hash-image"
    assert metadata["contentSha256"] == "hash-image"
    assert metadata["indexId"] == "source_file:source-file-1:unit:IMAGE:page:3:image:1"
    assert "source_file_name" not in metadata


def test_citation_extracts_table_and_image_ids_from_nested_unit_keys():
    table = RetrievedChunk(
        chunk_id="chunk-table",
        doc_id="source-file-1",
        section="tables",
        text="table text",
        score=0.8,
        search_unit_id="unit-table",
        source_file_id="source-file-1",
        unit_type="TABLE",
        unit_key="page:2:table:1",
        page_start=2,
        page_end=2,
    )
    image = RetrievedChunk(
        chunk_id="chunk-image",
        doc_id="source-file-1",
        section="figures",
        text="caption",
        score=0.7,
        search_unit_id="unit-image",
        source_file_id="source-file-1",
        unit_type="IMAGE",
        unit_key="page:3:image:fig-7",
        page_start=3,
        page_end=3,
    )

    assert citation_payload(table)["tableId"] == "1"
    assert citation_payload(image)["imageId"] == "fig-7"


def test_citation_exposes_xlsx_sheet_and_cell_range_directly():
    chunk = RetrievedChunk(
        chunk_id="chunk-xlsx",
        doc_id="source-file-1",
        section="workbook/Sales",
        text="Region: Seoul | Revenue: 12000000",
        score=0.8,
        search_unit_id="unit-xlsx",
        source_file_id="source-file-1",
        source_file_name="sales_report_2024.xlsx",
        extracted_artifact_id="artifact-xlsx",
        artifact_type="XLSX_WORKBOOK_JSON",
        unit_type="TABLE",
        unit_key="sheet:0:table:0:A1-F24",
        title="SalesTable",
        metadata_json={
            "sheetName": "Sales",
            "sheetIndex": 0,
            "cellRange": "A1:F24",
            "rowStart": 1,
            "rowEnd": 24,
            "columnStart": 1,
            "columnEnd": 6,
            "tableId": "SalesTable",
        },
    )

    citation = citation_payload(chunk)

    assert citation["sourceFileName"] == "sales_report_2024.xlsx"
    assert citation["sheetName"] == "Sales"
    assert citation["cellRange"] == "A1:F24"
    assert citation["rowStart"] == 1
    assert citation["columnEnd"] == 6
    assert citation["tableId"] == "SalesTable"
