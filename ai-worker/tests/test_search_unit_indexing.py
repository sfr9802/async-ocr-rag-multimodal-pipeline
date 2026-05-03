from __future__ import annotations

from app.capabilities.rag.retrieval_contract import citation_payload
from app.capabilities.rag.search_unit_indexing import (
    document_from_claim,
    index_metadata,
    stable_index_id,
    to_chunk_row,
    to_document_row,
)
from app.capabilities.rag.generation import RetrievedChunk


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
