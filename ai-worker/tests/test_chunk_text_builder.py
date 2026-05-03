from __future__ import annotations

from app.capabilities.rag.chunk_text_builder import (
    build_pdf_chunk_texts,
    build_xlsx_chunk_texts,
)


def test_xlsx_chunk_texts_separate_embedding_display_and_citation():
    texts = build_xlsx_chunk_texts(
        source_file_name="sales_report.xlsx",
        title="SalesTable",
        location={
            "type": "xlsx",
            "sheet_name": "Sales",
            "sheet_index": 0,
            "table_id": "tbl_001",
            "cell_range": "B12:F12",
            "header_path": ["Department", "Product", "Revenue", "Growth"],
        },
        display_text="| Department | Revenue |\n| --- | --- |\n| Online | 12000000000 |",
    )

    assert "Sheet: Sales" in texts.embedding_text
    assert "Range: B12:F12" in texts.embedding_text
    assert "Headers: Department | Product | Revenue | Growth" in texts.embedding_text
    assert texts.display_text.startswith("| Department |")
    assert texts.citation_text == "sales_report.xlsx > Sales > B12:F12"
    assert "sales_report.xlsx" in texts.bm25_text


def test_pdf_chunk_texts_include_page_bbox_and_ocr_confidence():
    texts = build_pdf_chunk_texts(
        source_file_name="contract.pdf",
        title="Termination",
        location={
            "type": "pdf",
            "physical_page_index": 4,
            "page_no": 5,
            "page_label": "5",
            "bbox": [72.0, 120.0, 510.0, 680.0],
            "section_path": ["3. Terms", "3.2 Termination"],
            "block_type": "paragraph",
            "ocr_used": True,
            "ocr_confidence": 0.83,
        },
        display_text="Either party may terminate the agreement with notice.",
    )

    assert "Page: 5" in texts.embedding_text
    assert "Section: 3. Terms | 3.2 Termination" in texts.embedding_text
    assert "OCR confidence: 0.83" in texts.embedding_text
    assert texts.citation_text == "contract.pdf > p.5 > bbox [72.0,120.0,510.0,680.0]"
    assert texts.display_text == "Either party may terminate the agreement with notice."
