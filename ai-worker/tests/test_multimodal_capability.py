"""MULTIMODAL capability tests.

All tests here are fully hermetic: they use fake OCR / vision /
retriever / generator stand-ins instead of Tesseract, PyMuPDF, a real
VLM, a FAISS index, or a running Postgres. The suite runs without any
external infra.

Scenarios covered:

  1. Happy path — image input, OCR + vision both succeed.
     Asserts the 4 required artifacts and that FINAL_RESPONSE is
     grounded in the fused OCR + vision content.

  2. Happy path — PDF input with a mock PDF rasterizer.
     Asserts page 1 is handed to the vision provider and the OCR
     pages roll up into the fused context.

  3. OCR succeeds, vision fails — pipeline still answers, VISION_RESULT
     records "available: false" with the failure warning, FINAL_RESPONSE
     is still grounded in the OCR text.

  4. Vision succeeds, OCR produces empty text — pipeline still answers,
     fused context falls back to the vision caption, FINAL_RESPONSE
     cites the vision-side grounding chunk.

  5. Unsupported file type (GIF) — UNSUPPORTED_INPUT_TYPE CapabilityError.

  6. Registry resilience — MULTIMODAL init failure leaves MOCK / RAG /
     OCR untouched; missing OCR parent causes MULTIMODAL to be skipped
     cleanly.

  7. Artifact count + shape assertions — every happy-path test verifies
     the exact artifact types, file names, and JSON schema fields.

  8. Both-fail hard error — OCR raises AND vision raises →
     MULTIMODAL_ALL_PROVIDERS_FAILED typed error.

  9. User question INPUT_TEXT is used when present; the fusion helper's
     retrieval query honours it; the generator query is the raw user
     question (not the fused blob).

A fusion-level unit test covers the determinism + source-tracking
contract of the build_fusion helper on its own, so the capability
tests don't need to re-verify every fusion edge case.
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, List, Optional

import pytest

from app.capabilities.base import (
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
)
from app.capabilities.multimodal.capability import (
    MultimodalCapability,
    MultimodalCapabilityConfig,
)
from app.capabilities.multimodal.fusion import build_fusion
from app.capabilities.multimodal.vision_provider import (
    VisionDescriptionProvider,
    VisionDescriptionResult,
    VisionError,
)
from app.capabilities.ocr.provider import (
    OcrDocumentResult,
    OcrError,
    OcrPageResult,
    OcrProvider,
)
from app.capabilities.rag.generation import GenerationProvider, RetrievedChunk
from app.capabilities.rag.retriever import RetrievalReport


# ---------------------------------------------------------------------------
# Fake providers + fixtures.
# ---------------------------------------------------------------------------


class FakeOcrProvider(OcrProvider):
    """Canned OcrProvider — identical shape to the one in test_ocr_capability."""

    def __init__(
        self,
        *,
        image_result: Optional[OcrPageResult] = None,
        pdf_result: Optional[OcrDocumentResult] = None,
        raise_on: Optional[str] = None,
        raise_error: Optional[OcrError] = None,
    ) -> None:
        self._image_result = image_result
        self._pdf_result = pdf_result
        self._raise_on = raise_on
        self._raise_error = raise_error
        self.image_calls: List[bytes] = []
        self.pdf_calls: List[bytes] = []

    @property
    def name(self) -> str:
        return "fake-ocr-1.0"

    def ocr_image(self, image_bytes: bytes, *, mime_type=None) -> OcrPageResult:
        self.image_calls.append(image_bytes)
        if self._raise_on == "image":
            raise self._raise_error or OcrError("FAKE_IMG_FAIL", "fake image failure")
        assert self._image_result is not None, "test did not provide image_result"
        return self._image_result

    def ocr_pdf(self, pdf_bytes: bytes) -> OcrDocumentResult:
        self.pdf_calls.append(pdf_bytes)
        if self._raise_on == "pdf":
            raise self._raise_error or OcrError("FAKE_PDF_FAIL", "fake pdf failure")
        assert self._pdf_result is not None, "test did not provide pdf_result"
        return self._pdf_result


class FakeVisionProvider(VisionDescriptionProvider):
    """Canned VisionDescriptionProvider that returns a scripted result.

    Scripting rules:
      - If `raise_error` is set, every call raises it.
      - Otherwise every call returns `result` (with incrementing
        `page_number` replicated from the call arg).
    """

    def __init__(
        self,
        *,
        result: Optional[VisionDescriptionResult] = None,
        raise_error: Optional[VisionError] = None,
    ) -> None:
        self._result = result
        self._raise_error = raise_error
        self.calls: List[dict] = []

    @property
    def name(self) -> str:
        return "fake-vision-1.0"

    def describe_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: Optional[str] = None,
        hint: Optional[str] = None,
        page_number: int = 1,
    ) -> VisionDescriptionResult:
        self.calls.append(
            {
                "bytes_len": len(image_bytes),
                "mime_type": mime_type,
                "hint": hint,
                "page_number": page_number,
            }
        )
        if self._raise_error is not None:
            raise self._raise_error
        assert self._result is not None, "test did not provide result"
        # Re-project the scripted result onto the requested page_number
        # so tests that need to assert page propagation can do so.
        return VisionDescriptionResult(
            provider_name=self._result.provider_name,
            caption=self._result.caption,
            details=list(self._result.details),
            warnings=list(self._result.warnings),
            latency_ms=self._result.latency_ms,
            page_number=page_number,
        )


class FakeRetriever:
    """Minimal retriever used by the multimodal capability tests.

    The real Retriever has an `ensure_ready()` method and wraps an
    EmbeddingProvider + FaissIndex + RagMetadataStore. The multimodal
    capability only ever calls `retrieve(query)`, so we stub just that.
    """

    def __init__(
        self,
        *,
        results: List[RetrievedChunk],
        index_version: str = "test-mm-v1",
        embedding_model: str = "fake-embedder",
        top_k: int = 5,
    ) -> None:
        self._results = list(results)
        self._index_version = index_version
        self._embedding_model = embedding_model
        self._top_k = top_k
        self.retrieve_calls: List[str] = []

    def retrieve(self, query: str) -> RetrievalReport:
        self.retrieve_calls.append(query)
        return RetrievalReport(
            query=query,
            top_k=self._top_k,
            index_version=self._index_version,
            embedding_model=self._embedding_model,
            results=list(self._results),
        )


class CapturingGenerator(GenerationProvider):
    """GenerationProvider that records its inputs and returns a
    predictable markdown blob mentioning every chunk's doc_id.

    Not a mock — it actually consumes the inputs so downstream
    assertions can verify the fused context flowed through.
    """

    def __init__(self) -> None:
        self.calls: List[dict] = []

    @property
    def name(self) -> str:
        return "capturing-gen-1"

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        self.calls.append({"query": query, "chunks": list(chunks)})
        lines: list[str] = [f"# Answer for {query}", ""]
        lines.append("**Grounded in:**")
        for c in chunks:
            snippet = c.text.replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            lines.append(f"- [{c.doc_id}#{c.section}] (score={c.score:.2f}) {snippet}")
        return "\n".join(lines)


def _png_bytes() -> bytes:
    """Minimal PNG header — passes the classifier's magic-byte sniff."""
    return b"\x89PNG\r\n\x1a\n"


def _pdf_bytes() -> bytes:
    """Minimal PDF header — passes magic-byte classification."""
    return b"%PDF-1.4\n% fake\n"


def _make_job_input(
    *,
    job_id: str = "job-mm-test",
    file_bytes: bytes,
    content_type: Optional[str] = None,
    filename: Optional[str] = None,
    question: Optional[str] = None,
) -> CapabilityInput:
    artifacts: List[CapabilityInputArtifact] = [
        CapabilityInputArtifact(
            artifact_id="art-mm-file-1",
            type="INPUT_FILE",
            content=file_bytes,
            content_type=content_type,
            filename=filename,
        )
    ]
    if question is not None:
        artifacts.append(
            CapabilityInputArtifact(
                artifact_id="art-mm-q-1",
                type="INPUT_TEXT",
                content=question.encode("utf-8"),
                content_type="text/plain",
            )
        )
    return CapabilityInput(
        job_id=job_id,
        capability="MULTIMODAL",
        attempt_no=1,
        inputs=artifacts,
    )


def _find_output(result: CapabilityOutput, artifact_type: str):
    for artifact in result.outputs:
        if artifact.type == artifact_type:
            return artifact
    raise AssertionError(
        f"no {artifact_type} in outputs: {[o.type for o in result.outputs]}"
    )


def _default_config(**overrides) -> MultimodalCapabilityConfig:
    base = {
        "pdf_vision_dpi": 150,
        "emit_trace": False,
    }
    base.update(overrides)
    return MultimodalCapabilityConfig(**base)


def _make_capability(
    *,
    ocr: FakeOcrProvider,
    vision: FakeVisionProvider,
    retriever_results: Optional[List[RetrievedChunk]] = None,
    generator: Optional[GenerationProvider] = None,
    config: Optional[MultimodalCapabilityConfig] = None,
    pdf_pages_rasterizer: Optional[
        Callable[[bytes, int, int], List[tuple]]
    ] = None,
) -> MultimodalCapability:
    retriever = FakeRetriever(
        results=retriever_results
        if retriever_results is not None
        else [
            RetrievedChunk(
                chunk_id="chunk-1",
                doc_id="doc-anime-cats",
                section="overview",
                text=(
                    "An elderly fisherman feeds the stray cats of a small harbor "
                    "every morning without fail."
                ),
                score=0.71,
            ),
            RetrievedChunk(
                chunk_id="chunk-2",
                doc_id="doc-book",
                section="plot",
                text="A retired translator runs a secondhand bookshop at the last station.",
                score=0.42,
            ),
        ],
        top_k=3,
    )
    return MultimodalCapability(
        ocr_provider=ocr,
        vision_provider=vision,
        retriever=retriever,
        generator=generator or CapturingGenerator(),
        config=config or _default_config(),
        pdf_pages_rasterizer=pdf_pages_rasterizer,
    )


def _sample_vision_result(caption: str, **overrides) -> VisionDescriptionResult:
    return VisionDescriptionResult(
        provider_name="fake-vision-1.0",
        caption=caption,
        details=overrides.get(
            "details",
            ["dimensions: 400x600 (portrait)", "mean brightness: 180.5/255 (light)"],
        ),
        warnings=overrides.get("warnings", []),
        latency_ms=overrides.get("latency_ms", 3.2),
        page_number=overrides.get("page_number", 1),
    )


# ---------------------------------------------------------------------------
# 1. Happy path — image input.
# ---------------------------------------------------------------------------


def test_image_happy_path_emits_four_artifacts_and_fuses_ocr_plus_vision():
    ocr = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1,
            text="Invoice total $129.95 due 2026-04-15",
            avg_confidence=91.0,
            warnings=[],
        )
    )
    vision = FakeVisionProvider(
        result=_sample_vision_result(
            "A portrait light-toned image dominated by white tones with moderate contrast."
        )
    )
    generator = CapturingGenerator()
    cap = _make_capability(ocr=ocr, vision=vision, generator=generator)

    result = cap.run(
        _make_job_input(
            file_bytes=_png_bytes(),
            content_type="image/png",
            filename="invoice.png",
            question="what is the total amount?",
        )
    )

    # 4 required outputs in the expected order.
    assert [a.type for a in result.outputs] == [
        "OCR_TEXT",
        "VISION_RESULT",
        "RETRIEVAL_RESULT",
        "FINAL_RESPONSE",
    ]

    # Filenames are capability-scoped so they don't collide with OCR / RAG
    # outputs in local-storage.
    filenames = {a.type: a.filename for a in result.outputs}
    assert filenames["OCR_TEXT"] == "multimodal-ocr.txt"
    assert filenames["VISION_RESULT"] == "multimodal-vision.json"
    assert filenames["RETRIEVAL_RESULT"] == "multimodal-retrieval.json"
    assert filenames["FINAL_RESPONSE"] == "multimodal-answer.md"

    # OCR_TEXT carries the raw OCR payload exactly as the provider returned it.
    ocr_text = _find_output(result, "OCR_TEXT").content.decode("utf-8")
    assert ocr_text == "Invoice total $129.95 due 2026-04-15"

    # VISION_RESULT envelope is populated (available=true, pages[] schema).
    vision_body = json.loads(_find_output(result, "VISION_RESULT").content)
    assert vision_body["available"] is True
    assert vision_body["kind"] == "image"
    assert vision_body["mimeType"] == "image/png"
    assert vision_body["pageCount"] == 1
    assert len(vision_body["pages"]) == 1
    page0 = vision_body["pages"][0]
    assert page0["provider"] == "fake-vision-1.0"
    assert "portrait light-toned" in page0["caption"]
    assert page0["pageNumber"] == 1

    # RETRIEVAL_RESULT matches the RAG schema — same field names, same
    # structure. Downstream consumers can treat it identically.
    retr_body = json.loads(_find_output(result, "RETRIEVAL_RESULT").content)
    assert retr_body["indexVersion"] == "test-mm-v1"
    assert retr_body["embeddingModel"] == "fake-embedder"
    assert retr_body["topK"] == 3
    assert retr_body["hitCount"] == 2
    assert retr_body["results"][0]["docId"] == "doc-anime-cats"

    # The retrieval query started from the user question. It must have
    # been the thing the retriever actually saw.
    assert len(vision.calls) == 1  # vision called exactly once for image input
    assert vision.calls[0]["hint"] == "what is the total amount?"
    assert vision.calls[0]["page_number"] == 1

    # The generator was called with the user's question (not the fused
    # context string) and a chunk list starting with the synthetic
    # "fused_context" chunk — that's how the OCR + vision signal ends
    # up grounding the extractive answer.
    assert len(generator.calls) == 1
    gen_call = generator.calls[0]
    assert gen_call["query"] == "what is the total amount?"
    assert gen_call["chunks"][0].doc_id == "input:multimodal"
    assert gen_call["chunks"][0].section == "fused_context"
    # Both the OCR text and the vision caption made it into the synthetic chunk.
    synthetic_text = gen_call["chunks"][0].text
    assert "Invoice total $129.95" in synthetic_text
    assert "portrait light-toned" in synthetic_text  # page-wise bullet
    assert "what is the total amount?" in synthetic_text
    # The retrieved chunks follow in order after the synthetic one.
    assert gen_call["chunks"][1].doc_id == "doc-anime-cats"
    assert gen_call["chunks"][2].doc_id == "doc-book"

    # FINAL_RESPONSE is the generator's output and cites the synthetic
    # grounding chunk (because it's rank 0 by construction).
    final = _find_output(result, "FINAL_RESPONSE").content.decode("utf-8")
    assert "input:multimodal" in final
    # It also cites at least one retrieved doc id, proving both signal
    # sources are visible in the final answer.
    assert "doc-anime-cats" in final or "doc-book" in final


# ---------------------------------------------------------------------------
# 2. Happy path — PDF input with a stub rasterizer.
# ---------------------------------------------------------------------------


def test_pdf_happy_path_rasterizes_page_one_and_aggregates_ocr_pages():
    ocr = FakeOcrProvider(
        pdf_result=OcrDocumentResult(
            pages=[
                OcrPageResult(page_number=1, text="Page one header text.", avg_confidence=90.0),
                OcrPageResult(page_number=2, text="Page two body text.", avg_confidence=82.0),
            ],
            engine_name="fake-ocr-1.0",
            warnings=[],
        )
    )
    vision = FakeVisionProvider(
        result=_sample_vision_result("A landscape document scan with heavy text density.")
    )
    stub_rasterizer_calls: List[dict] = []
    def _stub_rasterizer(
        pdf_bytes: bytes, dpi: int, max_pages: int
    ) -> List[tuple]:
        stub_rasterizer_calls.append(
            {"len": len(pdf_bytes), "dpi": dpi, "max_pages": max_pages}
        )
        return [(1, b"\x89PNG\r\n\x1a\nFAKE")]

    cap = _make_capability(
        ocr=ocr,
        vision=vision,
        pdf_pages_rasterizer=_stub_rasterizer,
    )

    result = cap.run(
        _make_job_input(
            file_bytes=_pdf_bytes(),
            content_type="application/pdf",
            filename="report.pdf",
            question="summarize the report",
        )
    )

    # OCR text concatenates both pages.
    ocr_text = _find_output(result, "OCR_TEXT").content.decode("utf-8")
    assert "Page one header text." in ocr_text
    assert "Page two body text." in ocr_text
    assert ocr_text.index("Page one") < ocr_text.index("Page two")

    # PDF rasterizer was called exactly once with the configured DPI and max_pages.
    assert len(stub_rasterizer_calls) == 1
    assert stub_rasterizer_calls[0]["dpi"] == 150
    assert stub_rasterizer_calls[0]["max_pages"] == 3

    # Vision stage was called with the rasterized PNG bytes, not the raw PDF.
    assert len(vision.calls) == 1
    assert vision.calls[0]["mime_type"] == "image/png"

    # RETRIEVAL_RESULT is still emitted — PDF flow reuses the same downstream path.
    retr_body = json.loads(_find_output(result, "RETRIEVAL_RESULT").content)
    assert retr_body["hitCount"] == 2

    # VISION_RESULT reports the pdf kind with pages[] schema.
    vision_body = json.loads(_find_output(result, "VISION_RESULT").content)
    assert vision_body["kind"] == "pdf"
    assert vision_body["available"] is True
    assert vision_body["pageCount"] == 1
    assert vision_body["pages"][0]["pageNumber"] == 1


# ---------------------------------------------------------------------------
# 2b. Happy path — multi-page PDF vision.
# ---------------------------------------------------------------------------


def test_pdf_multi_page_vision_describes_each_rasterized_page():
    """Multi-page rasterizer returns 3 pages; the vision provider is
    called once per page. VISION_RESULT contains all three page entries
    and the fused context carries page-wise bullets."""
    ocr = FakeOcrProvider(
        pdf_result=OcrDocumentResult(
            pages=[
                OcrPageResult(page_number=1, text="Page one.", avg_confidence=90.0),
                OcrPageResult(page_number=2, text="Page two.", avg_confidence=85.0),
                OcrPageResult(page_number=3, text="Page three.", avg_confidence=80.0),
            ],
            engine_name="fake-ocr-1.0",
            warnings=[],
        )
    )
    vision = FakeVisionProvider(
        result=_sample_vision_result("Dense text layout with tables.")
    )
    rasterizer_calls: List[dict] = []

    def _multi_page_rasterizer(
        pdf_bytes: bytes, dpi: int, max_pages: int
    ) -> List[tuple]:
        rasterizer_calls.append(
            {"len": len(pdf_bytes), "dpi": dpi, "max_pages": max_pages}
        )
        return [
            (1, b"\x89PNG\r\n\x1a\nP1"),
            (2, b"\x89PNG\r\n\x1a\nP2"),
            (3, b"\x89PNG\r\n\x1a\nP3"),
        ]

    generator = CapturingGenerator()
    cap = _make_capability(
        ocr=ocr,
        vision=vision,
        generator=generator,
        pdf_pages_rasterizer=_multi_page_rasterizer,
    )

    result = cap.run(
        _make_job_input(
            file_bytes=_pdf_bytes(),
            content_type="application/pdf",
            filename="multi.pdf",
            question="what tables are in this document?",
        )
    )

    # Rasterizer was called once.
    assert len(rasterizer_calls) == 1
    assert rasterizer_calls[0]["max_pages"] == 3

    # Vision provider was called 3 times, once per page.
    assert len(vision.calls) == 3
    assert [c["page_number"] for c in vision.calls] == [1, 2, 3]
    assert all(c["mime_type"] == "image/png" for c in vision.calls)

    # VISION_RESULT has 3 page entries.
    vision_body = json.loads(_find_output(result, "VISION_RESULT").content)
    assert vision_body["available"] is True
    assert vision_body["pageCount"] == 3
    assert len(vision_body["pages"]) == 3
    assert [p["pageNumber"] for p in vision_body["pages"]] == [1, 2, 3]

    # Fused context carries page-wise visual bullets.
    gen_call = generator.calls[0]
    synthetic = gen_call["chunks"][0].text
    assert "**Page 1:**" in synthetic
    assert "**Page 2:**" in synthetic
    assert "**Page 3:**" in synthetic
    assert "Dense text layout" in synthetic


def test_pdf_multi_page_partial_vision_failure_still_succeeds():
    """If one page fails vision but others succeed, the pipeline
    continues with the successful pages and records warnings."""
    ocr = FakeOcrProvider(
        pdf_result=OcrDocumentResult(
            pages=[
                OcrPageResult(page_number=1, text="Page one.", avg_confidence=90.0),
                OcrPageResult(page_number=2, text="Page two.", avg_confidence=85.0),
            ],
            engine_name="fake-ocr-1.0",
            warnings=[],
        )
    )

    call_count = 0

    class _FailOnPage2Vision(FakeVisionProvider):
        """Succeeds on page 1, fails on page 2."""

        def describe_image(self, image_bytes, *, mime_type=None, hint=None, page_number=1):
            nonlocal call_count
            call_count += 1
            if page_number == 2:
                raise VisionError("PAGE2_FAIL", "page 2 corrupt")
            return super().describe_image(
                image_bytes, mime_type=mime_type, hint=hint, page_number=page_number
            )

    vision = _FailOnPage2Vision(
        result=_sample_vision_result("Good caption for page 1.")
    )

    def _rasterizer(pdf_bytes, dpi, max_pages):
        return [(1, b"\x89PNG\r\n\x1a\nP1"), (2, b"\x89PNG\r\n\x1a\nP2")]

    cap = _make_capability(
        ocr=ocr,
        vision=vision,
        pdf_pages_rasterizer=_rasterizer,
    )

    result = cap.run(
        _make_job_input(
            file_bytes=_pdf_bytes(),
            content_type="application/pdf",
            filename="partial.pdf",
        )
    )

    # Pipeline still produces all 4 artifacts.
    assert [a.type for a in result.outputs] == [
        "OCR_TEXT", "VISION_RESULT", "RETRIEVAL_RESULT", "FINAL_RESPONSE",
    ]

    # Only page 1 made it through vision.
    vision_body = json.loads(_find_output(result, "VISION_RESULT").content)
    assert vision_body["available"] is True
    assert vision_body["pageCount"] == 1
    assert vision_body["pages"][0]["pageNumber"] == 1
    # Warning about page 2 failure is recorded.
    assert any("page 2" in w and "PAGE2_FAIL" in w for w in vision_body["warnings"])


# ---------------------------------------------------------------------------
# 3. OCR succeeds, vision fails.
# ---------------------------------------------------------------------------


def test_ocr_succeeds_vision_fails_pipeline_still_answers():
    ocr = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1,
            text="The quick brown fox jumps over the lazy dog.",
            avg_confidence=95.0,
        )
    )
    vision = FakeVisionProvider(
        raise_error=VisionError("FAKE_VISION_DOWN", "model unreachable")
    )
    generator = CapturingGenerator()
    cap = _make_capability(ocr=ocr, vision=vision, generator=generator)

    result = cap.run(
        _make_job_input(
            file_bytes=_png_bytes(),
            content_type="image/png",
            filename="doc.png",
        )
    )

    # Still produces the 4 artifacts — vision failure is non-fatal.
    assert [a.type for a in result.outputs] == [
        "OCR_TEXT",
        "VISION_RESULT",
        "RETRIEVAL_RESULT",
        "FINAL_RESPONSE",
    ]

    # VISION_RESULT marks available=false with empty pages and the failure warning.
    vision_body = json.loads(_find_output(result, "VISION_RESULT").content)
    assert vision_body["available"] is False
    assert vision_body["pageCount"] == 0
    assert vision_body["pages"] == []
    assert any("FAKE_VISION_DOWN" in w for w in vision_body["warnings"])

    # Fused synthetic chunk carries the OCR text but not a vision caption.
    gen_call = generator.calls[0]
    synthetic = gen_call["chunks"][0].text
    assert "quick brown fox" in synthetic
    assert "unavailable" in synthetic  # fusion marks vision section as unavailable

    # FINAL_RESPONSE is still produced and grounds on the OCR side.
    final = _find_output(result, "FINAL_RESPONSE").content.decode("utf-8")
    assert "input:multimodal" in final


# ---------------------------------------------------------------------------
# 4. Vision succeeds, OCR empty.
# ---------------------------------------------------------------------------


def test_ocr_empty_vision_succeeds_pipeline_still_answers():
    ocr = FakeOcrProvider(
        image_result=OcrPageResult(page_number=1, text="   ", avg_confidence=5.0),
    )
    vision = FakeVisionProvider(
        result=_sample_vision_result(
            "A square low-light image dominated by neutral tones with low contrast."
        )
    )
    generator = CapturingGenerator()
    cap = _make_capability(ocr=ocr, vision=vision, generator=generator)

    result = cap.run(
        _make_job_input(
            file_bytes=_png_bytes(),
            content_type="image/png",
            filename="dark.png",
        )
    )

    # Still 4 artifacts.
    assert [a.type for a in result.outputs] == [
        "OCR_TEXT",
        "VISION_RESULT",
        "RETRIEVAL_RESULT",
        "FINAL_RESPONSE",
    ]

    # OCR_TEXT content is the literal (empty/whitespace) result.
    ocr_text = _find_output(result, "OCR_TEXT").content.decode("utf-8")
    assert ocr_text.strip() == ""

    # VISION_RESULT is populated with pages[] schema.
    vision_body = json.loads(_find_output(result, "VISION_RESULT").content)
    assert vision_body["available"] is True
    assert vision_body["pageCount"] == 1
    assert "neutral" in vision_body["pages"][0]["caption"]

    # Retriever was called with a query derived from the vision caption
    # (since there's no OCR text and no user question).
    retr_body = json.loads(_find_output(result, "RETRIEVAL_RESULT").content)
    assert "neutral" in retr_body["query"] or "image" in retr_body["query"]

    # Generator's synthetic chunk includes the vision caption and the
    # "OCR empty" marker.
    gen_call = generator.calls[0]
    synthetic = gen_call["chunks"][0].text
    assert "neutral" in synthetic
    assert "empty" in synthetic  # fusion flags the empty OCR section


# ---------------------------------------------------------------------------
# 5. Unsupported file type.
# ---------------------------------------------------------------------------


def test_unsupported_file_type_raises_typed_error():
    ocr = FakeOcrProvider()
    vision = FakeVisionProvider(result=_sample_vision_result("ignored"))
    cap = _make_capability(ocr=ocr, vision=vision)

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(
            _make_job_input(
                file_bytes=b"GIF89a\x01\x00\x01\x00",
                content_type="image/gif",
                filename="cat.gif",
            )
        )

    assert exc_info.value.code == "UNSUPPORTED_INPUT_TYPE"
    assert "PNG" in exc_info.value.message
    assert "PDF" in exc_info.value.message
    # Neither provider should have been called.
    assert ocr.image_calls == []
    assert vision.calls == []


def test_job_without_input_file_is_rejected():
    ocr = FakeOcrProvider()
    vision = FakeVisionProvider(result=_sample_vision_result("ignored"))
    cap = _make_capability(ocr=ocr, vision=vision)

    input_obj = CapabilityInput(
        job_id="job-no-file",
        capability="MULTIMODAL",
        attempt_no=1,
        inputs=[
            CapabilityInputArtifact(
                artifact_id="art-text",
                type="INPUT_TEXT",
                content=b"just a question",
                content_type="text/plain",
            )
        ],
    )

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(input_obj)
    assert exc_info.value.code == "UNSUPPORTED_INPUT_TYPE"


# ---------------------------------------------------------------------------
# 6. Both-providers hard failure.
# ---------------------------------------------------------------------------


def test_both_providers_failing_raises_multimodal_all_providers_failed():
    ocr = FakeOcrProvider(
        raise_on="image",
        raise_error=OcrError("IMAGE_DECODE_FAILED", "corrupt png header"),
    )
    vision = FakeVisionProvider(
        raise_error=VisionError("VLM_TIMEOUT", "provider timed out")
    )
    cap = _make_capability(ocr=ocr, vision=vision)

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(
            _make_job_input(
                file_bytes=_png_bytes(),
                content_type="image/png",
                filename="broken.png",
            )
        )
    assert exc_info.value.code == "MULTIMODAL_ALL_PROVIDERS_FAILED"
    # Both upstream failures should be surfaced in the error message
    # so ops can diagnose without digging through worker logs.
    assert "IMAGE_DECODE_FAILED" in exc_info.value.message
    assert "VLM_TIMEOUT" in exc_info.value.message


# ---------------------------------------------------------------------------
# 7. MULTIMODAL_TRACE — normalized trace.v1 schema shape assertions.
# ---------------------------------------------------------------------------


def _find_stage(trace_body: dict, stage_name: str) -> dict:
    for rec in trace_body["stages"]:
        if rec["stage"] == stage_name:
            return rec
    raise AssertionError(
        f"no stage {stage_name!r} in trace; present: "
        f"{[r['stage'] for r in trace_body['stages']]}"
    )


def test_trace_artifact_emits_normalized_trace_v1_on_happy_path():
    """MULTIMODAL_TRACE should carry a schemaVersion=trace.v1 payload
    with every stage (classify, ocr, vision, fusion, retrieve, generate)
    recorded as status=ok. finalStatus=ok."""
    ocr = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1, text="short but real text", avg_confidence=80.0
        )
    )
    vision = FakeVisionProvider(result=_sample_vision_result("A landscape shot."))
    cap = _make_capability(
        ocr=ocr,
        vision=vision,
        config=_default_config(emit_trace=True),
    )

    result = cap.run(
        _make_job_input(
            file_bytes=_png_bytes(),
            content_type="image/png",
            filename="a.png",
        )
    )
    types = [a.type for a in result.outputs]
    assert "MULTIMODAL_TRACE" in types
    trace_body = json.loads(_find_output(result, "MULTIMODAL_TRACE").content)

    # schema envelope
    assert trace_body["schemaVersion"] == "trace.v1"
    assert trace_body["capability"] == "MULTIMODAL"
    assert trace_body["inputKind"] == "image"
    assert trace_body["finalStatus"] == "ok"

    # every canonical stage is recorded in order
    stage_names = [rec["stage"] for rec in trace_body["stages"]]
    assert stage_names == [
        "classify", "ocr", "vision", "fusion", "retrieve", "generate",
    ]

    # every stage status is ok on the happy path
    for rec in trace_body["stages"]:
        assert rec["status"] == "ok", rec
        assert rec["code"] is None
        assert rec["fallbackUsed"] is False
        # duration_ms should be a non-negative float — cheap to collect
        assert isinstance(rec["durationMs"], (int, float))
        assert rec["durationMs"] >= 0

    # selected provider fields made it into the details payload
    assert _find_stage(trace_body, "ocr")["provider"] == "fake-ocr-1.0"
    assert _find_stage(trace_body, "ocr")["details"]["textLength"] > 0
    assert _find_stage(trace_body, "vision")["provider"] == "fake-vision-1.0"
    assert _find_stage(trace_body, "vision")["details"]["pageCount"] == 1
    assert _find_stage(trace_body, "fusion")["details"]["sources"]
    assert (
        _find_stage(trace_body, "retrieve")["details"]["indexVersion"]
        == "test-mm-v1"
    )

    # summary line mentions every stage and carries timing
    assert "classify:ok" in trace_body["summary"]
    assert "ocr:ok" in trace_body["summary"]
    assert "generate:ok" in trace_body["summary"]


def test_trace_ocr_success_vision_fail_marks_vision_as_fail_with_fallback():
    """Partial fallback: OCR emits text but vision provider raises.

    The trace must show vision as fail(fallbackUsed=true), every
    downstream stage still as ok, and the run is finalStatus=partial
    (not ok) — so consumers can distinguish a clean run from a
    run-with-caveats at a glance."""
    ocr = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1,
            text="The quick brown fox jumps over the lazy dog.",
            avg_confidence=95.0,
        )
    )
    vision = FakeVisionProvider(
        raise_error=VisionError("FAKE_VISION_DOWN", "model unreachable")
    )
    cap = _make_capability(
        ocr=ocr, vision=vision, config=_default_config(emit_trace=True)
    )

    result = cap.run(
        _make_job_input(
            file_bytes=_png_bytes(),
            content_type="image/png",
            filename="doc.png",
        )
    )

    trace_body = json.loads(_find_output(result, "MULTIMODAL_TRACE").content)
    assert trace_body["finalStatus"] == "partial"

    ocr_stage = _find_stage(trace_body, "ocr")
    assert ocr_stage["status"] == "ok"
    assert ocr_stage["fallbackUsed"] is False

    vision_stage = _find_stage(trace_body, "vision")
    assert vision_stage["status"] == "fail"
    assert vision_stage["code"] == "VISION_FAKE_VISION_DOWN"
    assert vision_stage["fallbackUsed"] is True
    assert "model unreachable" in (vision_stage["message"] or "")
    # retryable=True for vision errors — transient provider outages
    # are the typical case, so the flag is helpful for ops.
    assert vision_stage["retryable"] is True

    # Downstream stages still completed.
    assert _find_stage(trace_body, "fusion")["status"] == "ok"
    assert _find_stage(trace_body, "retrieve")["status"] == "ok"
    assert _find_stage(trace_body, "generate")["status"] == "ok"

    # Summary surfaces the partial marker.
    assert "vision:fail" in trace_body["summary"]
    assert "fallback" in trace_body["summary"]


def test_trace_ocr_empty_vision_success_marks_ocr_as_warn_with_fallback():
    """Partial fallback: OCR returned whitespace-only text, vision
    produced a useful caption. The trace must show OCR as
    warn(OCR_EMPTY_TEXT, fallbackUsed=true) and everything downstream
    as ok. finalStatus=partial."""
    ocr = FakeOcrProvider(
        image_result=OcrPageResult(page_number=1, text="   ", avg_confidence=5.0),
    )
    vision = FakeVisionProvider(
        result=_sample_vision_result(
            "A square low-light image dominated by neutral tones."
        )
    )
    cap = _make_capability(
        ocr=ocr, vision=vision, config=_default_config(emit_trace=True)
    )

    result = cap.run(
        _make_job_input(
            file_bytes=_png_bytes(),
            content_type="image/png",
            filename="dark.png",
        )
    )

    trace_body = json.loads(_find_output(result, "MULTIMODAL_TRACE").content)

    ocr_stage = _find_stage(trace_body, "ocr")
    assert ocr_stage["status"] == "warn"
    assert ocr_stage["code"] == "OCR_EMPTY_TEXT"
    assert ocr_stage["fallbackUsed"] is True

    vision_stage = _find_stage(trace_body, "vision")
    assert vision_stage["status"] == "ok"

    # Downstream stages still succeeded.
    assert _find_stage(trace_body, "fusion")["status"] == "ok"
    assert _find_stage(trace_body, "retrieve")["status"] == "ok"
    assert _find_stage(trace_body, "generate")["status"] == "ok"


def test_trace_retrieve_empty_marks_retrieve_as_warn_with_fallback():
    """When retrieval returns zero hits the pipeline still answers
    using the synthetic rank-0 fused chunk. The trace flags this with
    a ``RETRIEVAL_EMPTY`` warn record so operators can distinguish a
    "no hits" run from a normal hit-bearing run; downstream behaviour
    is otherwise unchanged."""
    ocr = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1,
            text="The quick brown fox jumps over the lazy dog.",
            avg_confidence=95.0,
        )
    )
    vision = FakeVisionProvider(
        result=_sample_vision_result("A neutral document scan.")
    )
    cap = _make_capability(
        ocr=ocr,
        vision=vision,
        retriever_results=[],  # retrieve returns zero hits
        config=_default_config(emit_trace=True),
    )

    result = cap.run(
        _make_job_input(
            file_bytes=_png_bytes(),
            content_type="image/png",
            filename="empty.png",
        )
    )

    # Existing artifact set is unchanged: OCR_TEXT / VISION_RESULT /
    # RETRIEVAL_RESULT / FINAL_RESPONSE (+ MULTIMODAL_TRACE since
    # emit_trace=True). Retrieval-empty must not introduce a new type.
    types = {a.type for a in result.outputs}
    assert types == {
        "OCR_TEXT",
        "VISION_RESULT",
        "RETRIEVAL_RESULT",
        "FINAL_RESPONSE",
        "MULTIMODAL_TRACE",
    }

    trace_body = json.loads(_find_output(result, "MULTIMODAL_TRACE").content)
    retrieve_stage = _find_stage(trace_body, "retrieve")
    assert retrieve_stage["status"] == "warn"
    assert retrieve_stage["code"] == "RETRIEVAL_EMPTY"
    assert retrieve_stage["fallbackUsed"] is True
    assert retrieve_stage["details"]["hitCount"] == 0

    # Generate still ran on the synthetic rank-0 chunk.
    assert _find_stage(trace_body, "generate")["status"] == "ok"

    # FINAL_RESPONSE was still produced.
    final = _find_output(result, "FINAL_RESPONSE")
    assert final.content  # non-empty

    # Summary line carries the warn marker.
    assert "retrieve:warn(RETRIEVAL_EMPTY" in trace_body["summary"]


def test_both_fail_terminal_error_message_identifies_both_stages():
    """Terminal failure: both providers raise. The CapabilityError
    message must embed the stage-flow summary so operators see which
    stages failed from just the job record."""
    ocr = FakeOcrProvider(
        raise_on="image",
        raise_error=OcrError("IMAGE_DECODE_FAILED", "corrupt png"),
    )
    vision = FakeVisionProvider(
        raise_error=VisionError("VLM_TIMEOUT", "provider timed out")
    )
    cap = _make_capability(ocr=ocr, vision=vision)

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(
            _make_job_input(
                file_bytes=_png_bytes(),
                content_type="image/png",
                filename="broken.png",
            )
        )

    assert exc_info.value.code == "MULTIMODAL_ALL_PROVIDERS_FAILED"
    msg = exc_info.value.message
    # Both upstream codes appear verbatim.
    assert "IMAGE_DECODE_FAILED" in msg
    assert "VLM_TIMEOUT" in msg
    # Stage summary is folded in so operators can see progression.
    assert "trace:" in msg
    assert "ocr:fail" in msg
    assert "vision:fail" in msg
    # Downstream stages are explicitly marked skipped in the summary.
    assert "fusion:skipped" in msg
    assert "retrieve:skipped" in msg
    assert "generate:skipped" in msg


def test_unsupported_input_raises_stable_code_with_no_trace_embedded():
    """Unsupported file type path must return a single stable code
    with NO stage summary — the classifier runs before the trace
    builder exists, so there's nothing ambiguous to surface."""
    ocr = FakeOcrProvider()
    vision = FakeVisionProvider(result=_sample_vision_result("ignored"))
    cap = _make_capability(ocr=ocr, vision=vision)

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(
            _make_job_input(
                file_bytes=b"GIF89a\x01\x00\x01\x00",
                content_type="image/gif",
                filename="cat.gif",
            )
        )

    assert exc_info.value.code == "UNSUPPORTED_INPUT_TYPE"
    # The message is deliberately short and does NOT carry a trace
    # summary — the pipeline never reached a stage that could record one.
    assert "trace:" not in exc_info.value.message
    assert "PNG" in exc_info.value.message  # existing remediation text


def test_retrieval_failure_preserves_ocr_vision_success_in_trace():
    """Retrieval failure after OCR/vision succeeded: the CapabilityError
    must be MULTIMODAL_RETRIEVAL_FAILED and its message must preserve
    the earlier success context so operators can diagnose what broke."""

    class _FailingRetriever:
        def __init__(self) -> None:
            self._index_version = "test-mm-v1"

        def retrieve(self, query: str):
            raise RuntimeError("FAISS search blew up: index corrupt")

    ocr = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1,
            text="Invoice total: $99",
            avg_confidence=88.0,
        )
    )
    vision = FakeVisionProvider(
        result=_sample_vision_result("A landscape scan.")
    )
    cap = MultimodalCapability(
        ocr_provider=ocr,
        vision_provider=vision,
        retriever=_FailingRetriever(),
        generator=CapturingGenerator(),
        config=_default_config(),
    )

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(
            _make_job_input(
                file_bytes=_png_bytes(),
                content_type="image/png",
                filename="doc.png",
            )
        )

    err = exc_info.value
    assert err.code == "MULTIMODAL_RETRIEVAL_FAILED"
    # Earlier success context is preserved.
    assert "ocr:ok" in err.message
    assert "vision:ok" in err.message
    assert "fusion:ok" in err.message
    # The failure itself is recorded.
    assert "retrieve:fail" in err.message
    assert "FAISS search blew up" in err.message
    # Generation is marked skipped.
    assert "generate:skipped" in err.message


def test_generation_failure_preserves_retrieval_success_in_trace():
    """Generation failure after retrieval succeeded: same pattern,
    MULTIMODAL_GENERATION_FAILED, earlier success preserved."""

    class _FailingGenerator(GenerationProvider):
        @property
        def name(self) -> str:
            return "failing-generator-1"

        def generate(self, query, chunks):
            raise RuntimeError("LLM endpoint returned 500")

    ocr = FakeOcrProvider(
        image_result=OcrPageResult(
            page_number=1, text="body", avg_confidence=88.0
        )
    )
    vision = FakeVisionProvider(result=_sample_vision_result("caption."))
    cap = _make_capability(
        ocr=ocr, vision=vision, generator=_FailingGenerator()
    )

    with pytest.raises(CapabilityError) as exc_info:
        cap.run(
            _make_job_input(
                file_bytes=_png_bytes(),
                content_type="image/png",
                filename="doc.png",
            )
        )

    err = exc_info.value
    assert err.code == "MULTIMODAL_GENERATION_FAILED"
    assert "retrieve:ok" in err.message
    assert "generate:fail" in err.message
    assert "LLM endpoint returned 500" in err.message


# ---------------------------------------------------------------------------
# 8. Fusion helper — deterministic, source-tracking.
# ---------------------------------------------------------------------------


def test_fusion_builds_deterministic_query_and_context_with_all_signals():
    vision = _sample_vision_result("Landscape document scan with moderate contrast.")
    result_1 = build_fusion(
        user_question="what is the total?",
        ocr_text="INVOICE #42\nTotal: $99.00",
        vision_pages=[vision],
    )
    result_2 = build_fusion(
        user_question="what is the total?",
        ocr_text="INVOICE #42\nTotal: $99.00",
        vision_pages=[vision],
    )
    # Byte-for-byte deterministic.
    assert result_1.retrieval_query == result_2.retrieval_query
    assert result_1.fused_context == result_2.fused_context
    assert result_1.sources == result_2.sources

    # Short question gets enriched with OCR keywords.
    assert "what is the total?" in result_1.retrieval_query
    assert "OCR context" in result_1.retrieval_query
    assert "user_question" in result_1.sources
    assert "ocr_text" in result_1.sources
    assert "vision_description" in result_1.sources

    # Fused context has all three sections in order.
    assert "### User question" in result_1.fused_context
    assert "### Extracted text (OCR)" in result_1.fused_context
    assert "### Visual description" in result_1.fused_context
    # Section order is stable.
    assert (
        result_1.fused_context.index("### User question")
        < result_1.fused_context.index("### Extracted text (OCR)")
        < result_1.fused_context.index("### Visual description")
    )


def test_fusion_handles_missing_signals_with_defaults():
    # No user question, no OCR, no vision → default retrieval query + warning.
    result = build_fusion(user_question=None, ocr_text="", vision_pages=[])
    assert result.retrieval_query == "describe the submitted document"
    assert result.sources == []
    assert result.warnings
    assert "### User question" in result.fused_context
    assert "(none supplied)" in result.fused_context
    assert "(empty" in result.fused_context
    assert "(unavailable" in result.fused_context


def test_fusion_ocr_only_path_uses_ocr_head_as_query():
    long_ocr = "This is a long OCR paragraph " * 20
    result = build_fusion(user_question=None, ocr_text=long_ocr, vision_pages=[])
    assert result.retrieval_query.startswith("This is a long OCR paragraph")
    assert "ocr_text" in result.sources
    # Query is capped at the default max_query_chars.
    assert len(result.retrieval_query) <= 400


# ---------------------------------------------------------------------------
# 9. Registry resilience — MULTIMODAL failure doesn't break others, missing
#    parents cause MULTIMODAL to be skipped cleanly.
# ---------------------------------------------------------------------------


def test_multimodal_failure_does_not_affect_mock_rag_or_ocr(monkeypatch):
    """Even when the multimodal builder itself blows up, MOCK / RAG / OCR
    continue to register cleanly. Mirrors the OCR/RAG resilience tests."""
    from app.capabilities import registry as registry_module
    from app.capabilities.base import Capability, CapabilityInput, CapabilityOutput
    from app.core.config import WorkerSettings

    class _FakeRag(Capability):
        name = "RAG"
        def run(self, input: CapabilityInput) -> CapabilityOutput:  # pragma: no cover
            return CapabilityOutput(outputs=[])

    class _FakeOcr(Capability):
        name = "OCR"
        def run(self, input: CapabilityInput) -> CapabilityOutput:  # pragma: no cover
            return CapabilityOutput(outputs=[])

    def _failing_multimodal(_settings):
        raise RuntimeError(
            "simulated MULTIMODAL init failure (vision provider blew up)"
        )

    monkeypatch.setattr(
        registry_module, "_build_rag_capability", lambda _s: _FakeRag()
    )
    monkeypatch.setattr(
        registry_module, "_build_ocr_capability", lambda _s: _FakeOcr()
    )
    monkeypatch.setattr(
        registry_module, "_build_multimodal_capability", _failing_multimodal
    )

    settings = WorkerSettings(
        rag_enabled=True, ocr_enabled=True, multimodal_enabled=True
    )
    result = registry_module.build_default_registry(settings)

    assert "MOCK" in result.available()
    assert "RAG" in result.available()
    assert "OCR" in result.available()
    assert "MULTIMODAL" not in result.available()


def test_multimodal_skipped_when_ocr_parent_missing(monkeypatch):
    """With OCR unable to register, MULTIMODAL is skipped WITHOUT even
    calling its builder. The parent-dependency check is the gate — not
    the builder's own try/except."""
    from app.capabilities import registry as registry_module
    from app.capabilities.base import Capability, CapabilityInput, CapabilityOutput
    from app.core.config import WorkerSettings

    class _FakeRag(Capability):
        name = "RAG"
        def run(self, input: CapabilityInput) -> CapabilityOutput:  # pragma: no cover
            return CapabilityOutput(outputs=[])

    def _failing_ocr(_settings):
        raise RuntimeError("simulated OCR init failure")

    def _should_not_be_called(_settings):
        raise AssertionError(
            "_build_multimodal_capability must NOT be called when OCR is missing"
        )

    monkeypatch.setattr(
        registry_module, "_build_rag_capability", lambda _s: _FakeRag()
    )
    monkeypatch.setattr(registry_module, "_build_ocr_capability", _failing_ocr)
    monkeypatch.setattr(
        registry_module, "_build_multimodal_capability", _should_not_be_called
    )

    settings = WorkerSettings(
        rag_enabled=True, ocr_enabled=True, multimodal_enabled=True
    )
    result = registry_module.build_default_registry(settings)

    assert "MOCK" in result.available()
    assert "RAG" in result.available()
    assert "OCR" not in result.available()
    assert "MULTIMODAL" not in result.available()


def test_multimodal_skipped_when_rag_parent_missing(monkeypatch):
    """With RAG unable to register, MULTIMODAL is skipped cleanly — the
    multimodal builder is never called because it can't succeed
    without a retriever."""
    from app.capabilities import registry as registry_module
    from app.capabilities.base import Capability, CapabilityInput, CapabilityOutput
    from app.core.config import WorkerSettings

    class _FakeOcr(Capability):
        name = "OCR"
        def run(self, input: CapabilityInput) -> CapabilityOutput:  # pragma: no cover
            return CapabilityOutput(outputs=[])

    def _failing_rag(_settings):
        raise RuntimeError("simulated RAG init failure (index missing)")

    def _should_not_be_called(_settings):
        raise AssertionError(
            "_build_multimodal_capability must NOT be called when RAG is missing"
        )

    monkeypatch.setattr(registry_module, "_build_rag_capability", _failing_rag)
    monkeypatch.setattr(
        registry_module, "_build_ocr_capability", lambda _s: _FakeOcr()
    )
    monkeypatch.setattr(
        registry_module, "_build_multimodal_capability", _should_not_be_called
    )

    settings = WorkerSettings(
        rag_enabled=True, ocr_enabled=True, multimodal_enabled=True
    )
    result = registry_module.build_default_registry(settings)

    assert "MOCK" in result.available()
    assert "OCR" in result.available()
    assert "RAG" not in result.available()
    assert "MULTIMODAL" not in result.available()


# ---------------------------------------------------------------------------
# 10. HeuristicVisionProvider — exercised against a real Pillow image so
#     the v1 fallback is covered end-to-end. All the other tests use the
#     FakeVisionProvider to keep orchestration tests hermetic; this one
#     actually runs the default provider on live bytes.
# ---------------------------------------------------------------------------


def test_heuristic_vision_provider_describes_a_real_pillow_image():
    from io import BytesIO

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover — Pillow ships with sentence-transformers
        pytest.skip("Pillow not installed; heuristic provider cannot run")

    from app.capabilities.multimodal.heuristic_vision import HeuristicVisionProvider

    # Build a 120x180 red portrait image in memory.
    image = Image.new("RGB", (120, 180), color=(220, 30, 30))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    provider = HeuristicVisionProvider()
    result = provider.describe_image(
        png_bytes,
        mime_type="image/png",
        hint="what is in the image?",
        page_number=1,
    )

    assert result.provider_name == "heuristic-vision-v1"
    assert result.caption, "heuristic provider must emit a non-empty caption"
    # A portrait image with height > width should classify as "portrait".
    assert "portrait" in result.caption
    # A solid red image should classify "red" as dominant.
    assert "red" in result.caption
    # Hint is surfaced in the details for inspectability.
    assert any("hint" in d for d in result.details)
    assert result.page_number == 1


def test_multimodal_disabled_leaves_others_untouched(monkeypatch):
    """multimodal_enabled=False short-circuits the whole MM codepath."""
    from app.capabilities import registry as registry_module
    from app.capabilities.base import Capability, CapabilityInput, CapabilityOutput
    from app.core.config import WorkerSettings

    class _FakeRag(Capability):
        name = "RAG"
        def run(self, input: CapabilityInput) -> CapabilityOutput:  # pragma: no cover
            return CapabilityOutput(outputs=[])

    class _FakeOcr(Capability):
        name = "OCR"
        def run(self, input: CapabilityInput) -> CapabilityOutput:  # pragma: no cover
            return CapabilityOutput(outputs=[])

    def _should_not_be_called(_settings):
        raise AssertionError(
            "multimodal_enabled=False must skip MULTIMODAL construction"
        )

    monkeypatch.setattr(
        registry_module, "_build_rag_capability", lambda _s: _FakeRag()
    )
    monkeypatch.setattr(
        registry_module, "_build_ocr_capability", lambda _s: _FakeOcr()
    )
    monkeypatch.setattr(
        registry_module, "_build_multimodal_capability", _should_not_be_called
    )

    settings = WorkerSettings(
        rag_enabled=True,
        ocr_enabled=True,
        multimodal_enabled=False,
        ocr_extract_enabled=False,
        xlsx_extract_enabled=False,
    )
    result = registry_module.build_default_registry(settings)

    # AUTO + AGENT auto-register when at least one downstream sub is available.
    assert set(result.available()) == {"MOCK", "RAG", "OCR", "AUTO", "AGENT"}
