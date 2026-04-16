"""MULTIMODAL capability — v1.

Given a job's INPUT_FILE artifact (and optionally an INPUT_TEXT user
question), this capability runs a five-stage pipeline:

  stage A  OCR extraction:       file bytes  → OCR text
  stage B  vision description:   file bytes  → visual description
  stage C  fusion:               (question, OCR text, vision) → fused context
  stage D  retrieval:            fused context → top-k chunks (reuses text RAG)
  stage E  generation:           fused context + chunks → grounded answer

Output artifacts (always emitted when any stage produced signal):

  OCR_TEXT          plain UTF-8 text from stage A
  VISION_RESULT     JSON describing the vision provider's output
  RETRIEVAL_RESULT  JSON of the retrieval report (same schema the RAG
                    capability already emits — downstream consumers
                    can treat it identically)
  FINAL_RESPONSE    grounded markdown answer from stage E
  MULTIMODAL_TRACE  (optional) JSON trace recording which stages
                    contributed, warnings, fusion metadata. Gated
                    behind MultimodalCapabilityConfig.emit_trace so
                    ops can enable it without shipping a schema change.

Design choices for v1 that are NOT accidents:

* **OCR and vision are independent.** If either fails, the job still
  runs as long as the other produced something. Only the "both failed"
  case raises `MULTIMODAL_ALL_PROVIDERS_FAILED`.

* **Fusion builds retrieval input.** The fused context is both (a)
  the short retrieval query that hits FAISS and (b) a synthetic
  grounding chunk injected at rank 0 into the generator's chunk list.
  This is how the OCR + vision signal actually ends up in the
  FINAL_RESPONSE: the extractive generator picks its "short answer"
  sentence from the top-scoring chunk, and the top-scoring chunk is
  the fused context by construction.

* **Retrieval is the existing text-RAG retriever, unchanged.** We do
  not build a separate multimodal vector DB, we do not wire CLIP /
  VLM retrieval indexes, and we do not touch `RagCapability`. This
  is explicitly v1 scope — see `docs/architecture.md` "Multimodal v1
  limitations" for the deferred items.

* **PDF page handling is asymmetric.** OCR walks every page via
  `OcrProvider.ocr_pdf` (which already handles born-digital text
  layers + rasterization). Vision only sees the FIRST page rasterized
  via PyMuPDF. Page 1 is almost always representative of the visual
  layout; captioning every page doubles cost for negligible v1 gain.
  A later phase can add per-page captions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from app.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
    CapabilityOutput,
    CapabilityOutputArtifact,
)
from app.capabilities.multimodal.fusion import FusionResult, build_fusion
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
from app.capabilities.rag.retriever import RetrievalReport, Retriever
from app.capabilities.trace import (
    FINAL_FAILED,
    FINAL_OK,
    FINAL_PARTIAL,
    INPUT_KIND_IMAGE,
    INPUT_KIND_PDF,
    STAGE_CLASSIFY,
    STAGE_FUSION,
    STAGE_GENERATE,
    STAGE_OCR,
    STAGE_RETRIEVE,
    STAGE_VISION,
    PipelineTrace,
    TraceBuilder,
    elapsed_ms,
)

log = logging.getLogger(__name__)


# Mime + magic-byte classifications — intentionally a copy of the OCR
# capability's set so the two stays in lockstep without crossing a
# subclass boundary. If one ever drifts the tests in both files will
# flag it.
_IMAGE_MIME_PREFIXES = ("image/png", "image/jpeg", "image/jpg")
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
_PDF_MIME_TYPES = ("application/pdf", "application/x-pdf")
_PDF_EXTENSION = ".pdf"


@dataclass(frozen=True)
class MultimodalCapabilityConfig:
    """Runtime knobs for the multimodal capability.

    The retriever's top-k is NOT set here — it comes from the shared
    RAG `Retriever` instance, which is built once at startup with
    `settings.rag_top_k`. v1 deliberately reuses the RAG top-k so a
    MULTIMODAL job's retrieval shape is identical to a pure RAG job
    over the same index.

    Fields:
      pdf_vision_dpi:        DPI used when rasterizing PDF page 1 for
                             the vision provider. 150 is plenty for
                             captioning; 200+ wastes memory without
                             helping the heuristic provider.
      max_fused_chunk_chars: cap on how much of the fused context is
                             injected as the synthetic rank-0 chunk.
                             Prevents a huge OCR dump from blowing out
                             the extractive generator's excerpt logic.
      emit_trace:            when True, emit a MULTIMODAL_TRACE artifact
                             alongside the four main outputs. Off by
                             default to keep the artifact count at 4.
      default_user_question: fallback question used when the job didn't
                             supply INPUT_TEXT. Deliberately generic —
                             specific questions come from the client.
    """

    pdf_vision_dpi: int = 150
    max_fused_chunk_chars: int = 2000
    emit_trace: bool = False
    default_user_question: str = ""


class MultimodalCapability(Capability):
    """v1 multimodal capability: OCR + heuristic vision + text RAG.

    The dependencies are injected so tests can stub every seam and the
    registry can share a single OcrProvider / Retriever instance across
    RAG, OCR, and MULTIMODAL.
    """

    name = "MULTIMODAL"

    def __init__(
        self,
        *,
        ocr_provider: OcrProvider,
        vision_provider: VisionDescriptionProvider,
        retriever: Retriever,
        generator: GenerationProvider,
        config: MultimodalCapabilityConfig,
        pdf_rasterizer: Optional[Callable[[bytes, int], Optional[bytes]]] = None,
    ) -> None:
        self._ocr = ocr_provider
        self._vision = vision_provider
        self._retriever = retriever
        self._generator = generator
        self._config = config
        self._pdf_rasterizer = pdf_rasterizer or _default_rasterize_pdf_first_page

    # ------------------------------------------------------------------

    def run(self, input: CapabilityInput) -> CapabilityOutput:
        file_artifact = self._pick_input_file(input)
        question = self._extract_question(input)

        # -- stage 0: classify -----------------------------------------
        # Classification raises on unsupported input, so the TraceBuilder
        # doesn't exist yet — a `raise` here never leaks a trace into
        # the error message, and UNSUPPORTED_INPUT_TYPE stays a clean,
        # single-line typed error (no ambiguity — requirement #4).
        started = time.monotonic()
        mime_type, kind = self._classify(file_artifact)
        filename = self._derive_filename(file_artifact, kind)
        classify_ms = elapsed_ms(started)

        input_kind = INPUT_KIND_PDF if kind == "pdf" else INPUT_KIND_IMAGE
        builder = TraceBuilder(capability=self.name, input_kind=input_kind)
        builder.record_ok(
            STAGE_CLASSIFY,
            duration_ms=classify_ms,
            details={
                "mimeType": mime_type,
                "filename": filename,
                "kind": kind,
                "hasQuestion": bool(question),
                "sizeBytes": len(file_artifact.content),
            },
        )

        log.info(
            "MULTIMODAL start jobId=%s artifact=%s kind=%s mime=%s filename=%s "
            "size_bytes=%d has_question=%s",
            input.job_id,
            file_artifact.artifact_id,
            kind,
            mime_type,
            filename,
            len(file_artifact.content),
            bool(question),
        )

        # -- stage A: OCR ------------------------------------------------
        ocr_document, ocr_warnings = self._run_ocr_stage(
            file_artifact, kind=kind, mime_type=mime_type, builder=builder
        )
        ocr_text = ocr_document.full_text if ocr_document else ""

        # -- stage B: vision description --------------------------------
        vision_result, vision_warnings = self._run_vision_stage(
            file_artifact,
            kind=kind,
            mime_type=mime_type,
            hint=question,
            builder=builder,
        )

        # Hard failure: BOTH providers produced nothing usable.
        if (not ocr_text or not ocr_text.strip()) and vision_result is None:
            # Mark downstream stages as skipped so the trace shape is
            # complete even in the terminal path. Operators reading
            # just the summary can see the exact gap.
            builder.record_skipped(STAGE_FUSION, message="both providers failed")
            builder.record_skipped(STAGE_RETRIEVE, message="both providers failed")
            builder.record_skipped(STAGE_GENERATE, message="both providers failed")
            builder.finalize_failed()
            combined = "; ".join(ocr_warnings + vision_warnings) or "no details"
            raise CapabilityError(
                "MULTIMODAL_ALL_PROVIDERS_FAILED",
                "Multimodal pipeline could not extract any signal from the "
                "input — OCR returned no text AND the vision provider "
                f"returned no description. Upstream diagnostics: {combined} "
                f"| trace: {builder.summary()}",
            )

        # -- stage C: fusion --------------------------------------------
        started = time.monotonic()
        fusion = build_fusion(
            user_question=question,
            ocr_text=ocr_text,
            vision=vision_result,
        )
        fusion_ms = elapsed_ms(started)
        builder.record_ok(
            STAGE_FUSION,
            duration_ms=fusion_ms,
            details={
                "sources": list(fusion.sources),
                "retrievalQueryLength": len(fusion.retrieval_query),
                "fusedContextLength": len(fusion.fused_context),
                "fusionWarnings": len(fusion.warnings),
            },
        )
        for w in fusion.warnings:
            builder.add_warning(f"fusion: {w}")
        log.info(
            "MULTIMODAL fusion jobId=%s sources=%s query_len=%d "
            "fused_ctx_len=%d warnings=%d",
            input.job_id,
            fusion.sources,
            len(fusion.retrieval_query),
            len(fusion.fused_context),
            len(fusion.warnings),
        )

        # -- stage D: retrieval -----------------------------------------
        started = time.monotonic()
        try:
            retrieval_report = self._retriever.retrieve(fusion.retrieval_query)
        except Exception as ex:
            retrieve_ms = elapsed_ms(started)
            builder.record_fail(
                STAGE_RETRIEVE,
                provider=type(self._retriever).__name__,
                code="MULTIMODAL_RETRIEVAL_FAILED",
                message=f"{type(ex).__name__}: {ex}",
                duration_ms=retrieve_ms,
                retryable=True,
            )
            builder.record_skipped(
                STAGE_GENERATE, message="retrieval failed"
            )
            builder.finalize_failed()
            raise CapabilityError(
                "MULTIMODAL_RETRIEVAL_FAILED",
                "Retrieval stage failed after OCR / vision already produced "
                f"usable signal. Upstream error: {type(ex).__name__}: {ex} "
                f"| trace: {builder.summary()}",
            ) from ex
        retrieve_ms = elapsed_ms(started)
        builder.record_ok(
            STAGE_RETRIEVE,
            provider=type(self._retriever).__name__,
            duration_ms=retrieve_ms,
            details={
                "hitCount": len(retrieval_report.results),
                "indexVersion": retrieval_report.index_version,
                "embeddingModel": retrieval_report.embedding_model,
                "topK": retrieval_report.top_k,
            },
        )
        log.info(
            "MULTIMODAL retrieval jobId=%s hits=%d index_version=%s",
            input.job_id,
            len(retrieval_report.results),
            retrieval_report.index_version,
        )

        # -- stage E: generation ----------------------------------------
        started = time.monotonic()
        generator_query = question.strip() if question else fusion.retrieval_query
        grounding_chunks = self._build_grounding_chunks(
            fusion, retrieval_report.results
        )
        try:
            final_answer = self._generator.generate(generator_query, grounding_chunks)
        except Exception as ex:
            generate_ms = elapsed_ms(started)
            builder.record_fail(
                STAGE_GENERATE,
                provider=getattr(self._generator, "name", type(self._generator).__name__),
                code="MULTIMODAL_GENERATION_FAILED",
                message=f"{type(ex).__name__}: {ex}",
                duration_ms=generate_ms,
                retryable=True,
            )
            builder.finalize_failed()
            raise CapabilityError(
                "MULTIMODAL_GENERATION_FAILED",
                "Generation stage failed after OCR / vision / retrieval all "
                f"succeeded. Upstream error: {type(ex).__name__}: {ex} "
                f"| trace: {builder.summary()}",
            ) from ex
        generate_ms = elapsed_ms(started)
        builder.record_ok(
            STAGE_GENERATE,
            provider=getattr(self._generator, "name", type(self._generator).__name__),
            duration_ms=generate_ms,
            details={
                "answerLength": len(final_answer),
                "chunkCount": len(grounding_chunks),
            },
        )

        # Decide final status: any fail-with-fallback stage downgrades
        # the run to "partial" even though all outputs were produced.
        has_partial = any(
            rec.status == "fail" and rec.fallback_used
            for rec in builder.trace.stages
        )
        if has_partial:
            trace = builder.finalize_partial()
        else:
            trace = builder.finalize_ok()

        # -- artifact assembly ------------------------------------------
        outputs: List[CapabilityOutputArtifact] = []

        outputs.append(
            CapabilityOutputArtifact(
                type="OCR_TEXT",
                filename="multimodal-ocr.txt",
                content_type="text/plain; charset=utf-8",
                content=ocr_text.encode("utf-8"),
            )
        )
        outputs.append(
            CapabilityOutputArtifact(
                type="VISION_RESULT",
                filename="multimodal-vision.json",
                content_type="application/json",
                content=self._vision_result_json(
                    vision=vision_result,
                    filename=filename,
                    mime_type=mime_type,
                    kind=kind,
                    warnings=vision_warnings,
                ).encode("utf-8"),
            )
        )
        outputs.append(
            CapabilityOutputArtifact(
                type="RETRIEVAL_RESULT",
                filename="multimodal-retrieval.json",
                content_type="application/json",
                content=self._retrieval_payload(retrieval_report).encode("utf-8"),
            )
        )
        outputs.append(
            CapabilityOutputArtifact(
                type="FINAL_RESPONSE",
                filename="multimodal-answer.md",
                content_type="text/markdown; charset=utf-8",
                content=final_answer.encode("utf-8"),
            )
        )
        if self._config.emit_trace:
            outputs.append(
                CapabilityOutputArtifact(
                    type="MULTIMODAL_TRACE",
                    filename="multimodal-trace.json",
                    content_type="application/json",
                    content=json.dumps(
                        trace.to_dict(), ensure_ascii=False, indent=2
                    ).encode("utf-8"),
                )
            )

        log.info(
            "MULTIMODAL done jobId=%s outputs=%d final_status=%s summary=%s",
            input.job_id, len(outputs), trace.final_status, trace.summary(),
        )
        return CapabilityOutput(outputs=outputs)

    # ==================================================================
    # input handling
    # ==================================================================

    @staticmethod
    def _pick_input_file(input: CapabilityInput) -> CapabilityInputArtifact:
        """First INPUT_FILE wins. Absence is a clean typed error."""
        for candidate in input.inputs:
            if candidate.type == "INPUT_FILE":
                return candidate
        if not input.inputs:
            raise CapabilityError(
                "NO_INPUT",
                "MULTIMODAL job has no input artifacts.",
            )
        raise CapabilityError(
            "UNSUPPORTED_INPUT_TYPE",
            "MULTIMODAL requires an INPUT_FILE artifact; got "
            + ", ".join(sorted({i.type for i in input.inputs})),
        )

    def _extract_question(self, input: CapabilityInput) -> Optional[str]:
        """Pull the optional INPUT_TEXT question out of the job inputs.

        The multipart endpoint in core-api supplies this as a second
        staged artifact when the client sends a `text` form field on
        the multimodal submission. v1 is happy without it — the
        fusion helper has its own "no question" default.
        """
        for candidate in input.inputs:
            if candidate.type == "INPUT_TEXT":
                try:
                    text = candidate.content.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if text:
                    return text
        if self._config.default_user_question:
            return self._config.default_user_question
        return None

    def _classify(
        self, artifact: CapabilityInputArtifact
    ) -> tuple[Optional[str], str]:
        """Return (normalized_mime_type_or_None, kind).

        Detection order: explicit content_type → filename extension →
        magic-byte sniff. Mirrors the OCR capability's classifier so
        the two capabilities accept the same inputs.
        """
        mime = (artifact.content_type or "").split(";")[0].strip().lower() or None
        filename = (artifact.filename or "").lower()

        if mime:
            if mime in _PDF_MIME_TYPES:
                return mime, "pdf"
            if any(mime.startswith(prefix) for prefix in _IMAGE_MIME_PREFIXES):
                return mime, "image"

        if filename.endswith(_PDF_EXTENSION):
            return mime or "application/pdf", "pdf"
        if filename.endswith(_IMAGE_EXTENSIONS):
            ext = filename.rsplit(".", 1)[-1]
            guessed = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[ext]
            return mime or guessed, "image"

        head = artifact.content[:8]
        if head.startswith(b"%PDF-"):
            return mime or "application/pdf", "pdf"
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return mime or "image/png", "image"
        if head.startswith(b"\xff\xd8\xff"):
            return mime or "image/jpeg", "image"

        raise CapabilityError(
            "UNSUPPORTED_INPUT_TYPE",
            "MULTIMODAL supports PNG, JPEG, and PDF only. Received "
            f"content_type={artifact.content_type!r} filename={artifact.filename!r}.",
        )

    @staticmethod
    def _derive_filename(
        artifact: CapabilityInputArtifact, kind: str
    ) -> str:
        if artifact.filename:
            return artifact.filename
        ext = {"image": "img", "pdf": "pdf"}.get(kind, "bin")
        return f"{artifact.artifact_id}.{ext}"

    # ==================================================================
    # stage A — OCR
    # ==================================================================

    def _run_ocr_stage(
        self,
        artifact: CapabilityInputArtifact,
        *,
        kind: str,
        mime_type: Optional[str],
        builder: TraceBuilder,
    ) -> Tuple[Optional[OcrDocumentResult], List[str]]:
        """Run the OCR provider and convert failures into soft warnings.

        Returns a tuple `(OcrDocumentResult | None, warnings)`. A None
        document means OCR failed outright — the vision stage is the
        only remaining signal, and the caller will hard-fail only if
        vision also produces nothing.

        Records a StageRecord on the builder in either case:
          * ok with details on success
          * warn with `OCR_EMPTY_TEXT` when OCR completed but text was blank
          * fail with `fallback_used=True` on OcrError
        """
        warnings: List[str] = []
        started = time.monotonic()
        try:
            if kind == "image":
                page = self._ocr.ocr_image(
                    artifact.content, mime_type=mime_type
                )
                document = OcrDocumentResult(
                    pages=[page],
                    engine_name=self._ocr.name,
                    warnings=[],
                )
            elif kind == "pdf":
                document = self._ocr.ocr_pdf(artifact.content)
            else:  # pragma: no cover — _classify already guarded
                raise CapabilityError(
                    "UNSUPPORTED_INPUT_TYPE",
                    f"MULTIMODAL cannot process kind={kind!r}",
                )
        except OcrError as ex:
            ocr_ms = elapsed_ms(started)
            warnings.append(
                f"ocr stage failed ({ex.code}): {ex.message} — continuing "
                "with vision-only signal"
            )
            log.warning(
                "MULTIMODAL ocr stage failed: code=%s message=%s — "
                "pipeline will continue with vision signal",
                ex.code, ex.message,
            )
            builder.record_fail(
                STAGE_OCR,
                provider=self._ocr.name,
                code=f"OCR_{ex.code}",
                message=ex.message,
                duration_ms=ocr_ms,
                retryable=False,
                fallback_used=True,
            )
            return None, warnings
        ocr_ms = elapsed_ms(started)

        # Roll up provider warnings for the trace.
        for page in document.pages:
            for w in page.warnings:
                warnings.append(f"ocr page {page.page_number}: {w}")
        for w in document.warnings:
            warnings.append(f"ocr document: {w}")

        text_length = document.total_text_length
        avg_conf = document.avg_confidence
        if not document.full_text.strip():
            warnings.append(
                "ocr stage extracted zero characters — vision stage "
                "will be the primary grounding signal"
            )
            # Empty text is a partial-fallback condition: OCR itself
            # didn't crash, but the fused context will be vision-only.
            builder.record_warn(
                STAGE_OCR,
                provider=self._ocr.name,
                code="OCR_EMPTY_TEXT",
                message=(
                    "OCR produced zero characters — vision signal will "
                    "drive fusion"
                ),
                duration_ms=ocr_ms,
                fallback_used=True,
                details={
                    "pageCount": len(document.pages),
                    "textLength": text_length,
                    "avgConfidence": avg_conf,
                    "engineName": document.engine_name,
                },
            )
        else:
            builder.record_ok(
                STAGE_OCR,
                provider=self._ocr.name,
                duration_ms=ocr_ms,
                details={
                    "pageCount": len(document.pages),
                    "textLength": text_length,
                    "avgConfidence": avg_conf,
                    "engineName": document.engine_name,
                },
            )
        return document, warnings

    # ==================================================================
    # stage B — vision description
    # ==================================================================

    def _run_vision_stage(
        self,
        artifact: CapabilityInputArtifact,
        *,
        kind: str,
        mime_type: Optional[str],
        hint: Optional[str],
        builder: TraceBuilder,
    ) -> Tuple[Optional[VisionDescriptionResult], List[str]]:
        """Run the vision provider and convert failures into soft warnings.

        For image inputs we hand the raw artifact bytes directly to
        the provider. For PDF inputs we rasterize the first page via
        PyMuPDF and hand that PNG to the provider — see the capability
        docstring for the reasoning behind captioning only page 1.

        Records a StageRecord on the builder:
          * ok on success
          * fail with `fallback_used=True` on VisionError, PDF
            rasterization failure, or zero-byte rasterization output
        """
        warnings: List[str] = []
        started = time.monotonic()
        try:
            if kind == "image":
                result = self._vision.describe_image(
                    artifact.content,
                    mime_type=mime_type,
                    hint=hint,
                    page_number=1,
                )
                vision_ms = elapsed_ms(started)
                builder.record_ok(
                    STAGE_VISION,
                    provider=result.provider_name,
                    duration_ms=vision_ms,
                    details={
                        "pageNumber": result.page_number,
                        "captionPreview": result.caption[:200],
                        "latencyMs": result.latency_ms,
                        "detailCount": len(result.details),
                    },
                )
                return result, warnings

            # PDF path
            try:
                png_bytes = self._pdf_rasterizer(
                    artifact.content, self._config.pdf_vision_dpi
                )
            except Exception as ex:
                vision_ms = elapsed_ms(started)
                warnings.append(
                    "pdf page rasterization failed "
                    f"({type(ex).__name__}: {ex}) — vision stage skipped"
                )
                log.warning(
                    "MULTIMODAL pdf rasterization failed: %s: %s",
                    type(ex).__name__, ex,
                )
                builder.record_fail(
                    STAGE_VISION,
                    provider=self._vision.name,
                    code="VISION_PDF_RASTERIZATION_FAILED",
                    message=f"{type(ex).__name__}: {ex}",
                    duration_ms=vision_ms,
                    retryable=False,
                    fallback_used=True,
                )
                return None, warnings

            if png_bytes is None or len(png_bytes) == 0:
                vision_ms = elapsed_ms(started)
                warnings.append(
                    "pdf rasterization returned no bytes (zero-page PDF?) "
                    "— vision stage skipped"
                )
                builder.record_fail(
                    STAGE_VISION,
                    provider=self._vision.name,
                    code="VISION_PDF_EMPTY",
                    message="pdf rasterization returned zero bytes",
                    duration_ms=vision_ms,
                    retryable=False,
                    fallback_used=True,
                )
                return None, warnings

            result = self._vision.describe_image(
                png_bytes,
                mime_type="image/png",
                hint=hint,
                page_number=1,
            )
            vision_ms = elapsed_ms(started)
            builder.record_ok(
                STAGE_VISION,
                provider=result.provider_name,
                duration_ms=vision_ms,
                details={
                    "pageNumber": result.page_number,
                    "captionPreview": result.caption[:200],
                    "latencyMs": result.latency_ms,
                    "detailCount": len(result.details),
                },
            )
            return result, warnings

        except VisionError as ex:
            vision_ms = elapsed_ms(started)
            warnings.append(
                f"vision stage failed ({ex.code}): {ex.message} — "
                "continuing with OCR-only signal"
            )
            log.warning(
                "MULTIMODAL vision stage failed: code=%s message=%s — "
                "pipeline will continue with OCR signal",
                ex.code, ex.message,
            )
            builder.record_fail(
                STAGE_VISION,
                provider=self._vision.name,
                code=f"VISION_{ex.code}",
                message=ex.message,
                duration_ms=vision_ms,
                retryable=True,
                fallback_used=True,
            )
            return None, warnings

    # ==================================================================
    # stage E helpers — building grounded chunks + generator query
    # ==================================================================

    def _build_grounding_chunks(
        self,
        fusion: FusionResult,
        retrieval_results: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """Prepend the fused context as a synthetic rank-0 chunk.

        The extractive generator picks its "short answer" sentence
        from the highest-scoring chunk. By injecting the fused
        context (OCR + vision + question) at the top of the list
        with score=1.0, the short answer is grounded in the actual
        input document rather than in whatever the retriever
        happened to surface.

        Retrieval chunks from the existing text RAG index follow
        after it, so the final answer still cites them in the
        Supporting passages list.
        """
        fused_text = fusion.fused_context
        if len(fused_text) > self._config.max_fused_chunk_chars:
            fused_text = (
                fused_text[: self._config.max_fused_chunk_chars - 3] + "..."
            )

        synthetic = RetrievedChunk(
            chunk_id="multimodal-input#fused",
            doc_id="input:multimodal",
            section="fused_context",
            text=fused_text,
            score=1.0,
        )
        return [synthetic] + list(retrieval_results)

    # ==================================================================
    # artifact envelopes
    # ==================================================================

    @staticmethod
    def _vision_result_json(
        *,
        vision: Optional[VisionDescriptionResult],
        filename: str,
        mime_type: Optional[str],
        kind: str,
        warnings: List[str],
    ) -> str:
        if vision is None:
            body = {
                "filename": filename,
                "mimeType": mime_type,
                "kind": kind,
                "provider": None,
                "caption": None,
                "details": [],
                "pageNumber": None,
                "latencyMs": None,
                "warnings": warnings,
                "available": False,
            }
        else:
            body = {
                "filename": filename,
                "mimeType": mime_type,
                "kind": kind,
                "provider": vision.provider_name,
                "caption": vision.caption,
                "details": list(vision.details),
                "pageNumber": vision.page_number,
                "latencyMs": vision.latency_ms,
                "warnings": warnings + list(vision.warnings),
                "available": True,
            }
        return json.dumps(body, ensure_ascii=False, indent=2)

    @staticmethod
    def _retrieval_payload(report: RetrievalReport) -> str:
        """Match the RAG capability's RETRIEVAL_RESULT schema.

        Keeping the shape identical means downstream consumers that
        already handle the RAG output format work unchanged with
        MULTIMODAL jobs — the `capability` field on the job row is
        the only thing that distinguishes them from the artifact
        perspective.
        """
        body = {
            "query": report.query,
            "topK": report.top_k,
            "indexVersion": report.index_version,
            "embeddingModel": report.embedding_model,
            "hitCount": len(report.results),
            "results": [
                {
                    "rank": i + 1,
                    "chunkId": r.chunk_id,
                    "docId": r.doc_id,
                    "section": r.section,
                    "score": round(r.score, 6),
                    "text": r.text,
                }
                for i, r in enumerate(report.results)
            ],
        }
        return json.dumps(body, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# default PyMuPDF-based rasterizer (used in production; tests inject their
# own so they don't need PyMuPDF installed)
# ----------------------------------------------------------------------


def _default_rasterize_pdf_first_page(pdf_bytes: bytes, dpi: int) -> Optional[bytes]:
    """Rasterize the first page of a PDF to PNG bytes.

    Uses PyMuPDF (already a worker dep for the OCR stack). Kept as a
    module-level function so the capability constructor's default
    binds to something importable without lazy-loading state.

    Returns None for zero-page PDFs. Raises on decoder failures —
    the capability wraps the raise into a non-fatal warning.
    """
    import fitz  # type: ignore

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count == 0:
            return None
        page = doc.load_page(0)
        pixmap = page.get_pixmap(dpi=int(dpi))
        return pixmap.tobytes("png")
    finally:
        doc.close()
