package com.aipipeline.coreapi.job.application.service;

import com.aipipeline.coreapi.job.application.service.JobSubmissionValidator.ErrorCodes;
import com.aipipeline.coreapi.job.domain.JobCapability;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Pure unit tests for {@link JobSubmissionValidator}.
 *
 * <p>No Spring context, no controllers, no mocks — the validator is a
 * stateless static utility, so the tests can exercise every branch of the
 * capability/input matrix directly. The proof that rejection prevents
 * pipeline side effects is handled separately in
 * {@code JobControllerValidationTest}, which uses MockMvc to verify that
 * {@code createAndEnqueue} and {@code storage.store} are never called when
 * the validator throws.
 */
class JobSubmissionValidatorTest {

    // ===================================================================
    // parseCapability
    // ===================================================================

    @Nested
    class ParseCapability {

        @Test
        void accepts_known_capability() {
            assertThat(JobSubmissionValidator.parseCapability("MOCK"))
                    .isEqualTo(JobCapability.MOCK);
            assertThat(JobSubmissionValidator.parseCapability("RAG"))
                    .isEqualTo(JobCapability.RAG);
            assertThat(JobSubmissionValidator.parseCapability("OCR"))
                    .isEqualTo(JobCapability.OCR);
            assertThat(JobSubmissionValidator.parseCapability("OCR_EXTRACT"))
                    .isEqualTo(JobCapability.OCR_EXTRACT);
            assertThat(JobSubmissionValidator.parseCapability("XLSX_EXTRACT"))
                    .isEqualTo(JobCapability.XLSX_EXTRACT);
            assertThat(JobSubmissionValidator.parseCapability("MULTIMODAL"))
                    .isEqualTo(JobCapability.MULTIMODAL);
            assertThat(JobSubmissionValidator.parseCapability("AUTO"))
                    .isEqualTo(JobCapability.AUTO);
        }

        @Test
        void accepts_lowercase_capability_via_existing_from_string_semantics() {
            assertThat(JobSubmissionValidator.parseCapability("rag"))
                    .isEqualTo(JobCapability.RAG);
            assertThat(JobSubmissionValidator.parseCapability("Multimodal"))
                    .isEqualTo(JobCapability.MULTIMODAL);
        }

        @Test
        void null_capability_throws_CAPABILITY_REQUIRED() {
            assertThatThrownBy(() -> JobSubmissionValidator.parseCapability(null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.CAPABILITY_REQUIRED);
        }

        @Test
        void blank_capability_throws_CAPABILITY_REQUIRED() {
            assertThatThrownBy(() -> JobSubmissionValidator.parseCapability("   "))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.CAPABILITY_REQUIRED);
        }

        @Test
        void unknown_capability_throws_UNKNOWN_CAPABILITY() {
            assertThatThrownBy(() -> JobSubmissionValidator.parseCapability("SUMMARIZE"))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.UNKNOWN_CAPABILITY);
        }
    }

    // ===================================================================
    // validateTextSubmission — JSON endpoint
    // ===================================================================

    @Nested
    class TextSubmission {

        // ---- RAG: text required ----

        @Test
        void rag_with_non_blank_text_accepted() {
            JobSubmissionValidator.validateTextSubmission(JobCapability.RAG, "who feeds the harbor cats?");
            // No exception — accepted.
        }

        @Test
        void rag_without_text_rejected_TEXT_REQUIRED() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateTextSubmission(JobCapability.RAG, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.TEXT_REQUIRED);
        }

        @Test
        void rag_with_empty_text_rejected_TEXT_REQUIRED() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateTextSubmission(JobCapability.RAG, ""))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.TEXT_REQUIRED);
        }

        @Test
        void rag_with_whitespace_only_text_rejected_TEXT_REQUIRED() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateTextSubmission(JobCapability.RAG, "   \t  "))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.TEXT_REQUIRED);
        }

        // ---- MOCK: preserves phase-1 compatibility ----

        @Test
        void mock_with_text_accepted() {
            JobSubmissionValidator.validateTextSubmission(JobCapability.MOCK, "hello");
        }

        @Test
        void mock_with_empty_text_accepted_backward_compat() {
            // Phase-1 mock behavior: empty text string is allowed — the MOCK
            // capability just echoes whatever bytes it receives. Tightening
            // this would break existing smoke tests and is out of scope.
            JobSubmissionValidator.validateTextSubmission(JobCapability.MOCK, "");
        }

        @Test
        void mock_with_null_text_rejected_TEXT_REQUIRED() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateTextSubmission(JobCapability.MOCK, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.TEXT_REQUIRED);
        }

        // ---- OCR / MULTIMODAL: wrong endpoint ----

        @Test
        void ocr_on_text_endpoint_rejected_FILE_REQUIRED() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateTextSubmission(JobCapability.OCR, "extract this"))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.FILE_REQUIRED);
        }

        @Test
        void ocr_extract_on_text_endpoint_rejected_FILE_REQUIRED() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateTextSubmission(JobCapability.OCR_EXTRACT, "extract this"))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.FILE_REQUIRED);
        }

        @Test
        void multimodal_on_text_endpoint_rejected_FILE_REQUIRED() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateTextSubmission(JobCapability.MULTIMODAL, "caption it"))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.FILE_REQUIRED);
        }

        @Test
        void xlsx_extract_on_text_endpoint_rejected_FILE_REQUIRED() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateTextSubmission(JobCapability.XLSX_EXTRACT, "extract workbook"))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.FILE_REQUIRED);
        }

        // ---- AUTO: text optional, blank allowed, null rejected ----

        @Test
        void auto_with_text_accepted() {
            JobSubmissionValidator.validateTextSubmission(JobCapability.AUTO, "what is rag");
        }

        @Test
        void auto_with_blank_text_accepted() {
            // Router is allowed to emit clarify on blank text — the
            // validator must not reject blank text so the routing stage
            // gets a chance to run.
            JobSubmissionValidator.validateTextSubmission(JobCapability.AUTO, "   ");
            JobSubmissionValidator.validateTextSubmission(JobCapability.AUTO, "");
        }

        @Test
        void auto_with_null_text_rejected_TEXT_REQUIRED() {
            // Null would NPE in the controller's text-to-bytes path.
            // Match the MOCK-JSON contract: require the field to be
            // present (possibly empty) rather than absent.
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateTextSubmission(JobCapability.AUTO, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.TEXT_REQUIRED);
        }
    }

    // ===================================================================
    // validateFileSubmission — multipart endpoint
    // ===================================================================

    @Nested
    class FileSubmission {

        // ---- shared file-presence rules ----

        @Test
        void null_file_rejected_FILE_REQUIRED_for_mock() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.MOCK, null, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.FILE_REQUIRED);
        }

        @Test
        void null_file_rejected_FILE_REQUIRED_for_ocr() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.OCR, null, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.FILE_REQUIRED);
        }

        @Test
        void null_file_rejected_FILE_REQUIRED_for_ocr_extract() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.OCR_EXTRACT, null, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.FILE_REQUIRED);
        }

        @Test
        void null_file_rejected_FILE_REQUIRED_for_multimodal() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.MULTIMODAL, null, "q"))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.FILE_REQUIRED);
        }

        @Test
        void zero_byte_file_rejected_FILE_EMPTY() {
            MultipartFile empty = new MockMultipartFile(
                    "file", "blank.png", "image/png", new byte[0]);
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.OCR, empty, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.FILE_EMPTY);
        }

        @Test
        void zero_byte_file_rejected_FILE_EMPTY_for_mock_too() {
            // Zero-byte uploads produce no useful pipeline work regardless of
            // capability, so the rule fires for MOCK as well — the controller
            // would otherwise hand 0 bytes to storage.store and dispatch an
            // empty job.
            MultipartFile empty = new MockMultipartFile(
                    "file", "empty.txt", "text/plain", new byte[0]);
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.MOCK, empty, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.FILE_EMPTY);
        }

        // ---- OCR and MULTIMODAL file-type gate ----

        @Test
        void ocr_with_png_accepted() {
            MultipartFile png = pngFile("receipt.png");
            JobSubmissionValidator.validateFileSubmission(JobCapability.OCR, png, null);
        }

        @Test
        void ocr_extract_with_png_accepted() {
            MultipartFile png = pngFile("receipt.png");
            JobSubmissionValidator.validateFileSubmission(JobCapability.OCR_EXTRACT, png, null);
        }

        @Test
        void ocr_with_jpeg_accepted() {
            MultipartFile jpg = new MockMultipartFile(
                    "file", "scan.jpg", "image/jpeg", new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF, 0x00});
            JobSubmissionValidator.validateFileSubmission(JobCapability.OCR, jpg, null);
        }

        @Test
        void ocr_with_pdf_accepted() {
            MultipartFile pdf = new MockMultipartFile(
                    "file", "invoice.pdf", "application/pdf", "%PDF-1.4 fake".getBytes());
            JobSubmissionValidator.validateFileSubmission(JobCapability.OCR, pdf, null);
        }

        @Test
        void multimodal_with_png_and_optional_text_accepted() {
            MultipartFile png = pngFile("diagram.png");
            JobSubmissionValidator.validateFileSubmission(
                    JobCapability.MULTIMODAL, png, "what does this diagram show?");
        }

        @Test
        void multimodal_with_png_and_blank_text_accepted() {
            MultipartFile png = pngFile("diagram.png");
            JobSubmissionValidator.validateFileSubmission(JobCapability.MULTIMODAL, png, "   ");
        }

        @Test
        void multimodal_with_png_and_null_text_accepted() {
            MultipartFile png = pngFile("diagram.png");
            JobSubmissionValidator.validateFileSubmission(JobCapability.MULTIMODAL, png, null);
        }

        @Test
        void multimodal_with_pdf_accepted() {
            MultipartFile pdf = new MockMultipartFile(
                    "file", "report.pdf", "application/pdf", "%PDF-1.4 fake".getBytes());
            JobSubmissionValidator.validateFileSubmission(
                    JobCapability.MULTIMODAL, pdf, "summarize this report");
        }

        @Test
        void xlsx_extract_with_xlsx_accepted() {
            MultipartFile xlsx = new MockMultipartFile(
                    "file",
                    "sales.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "PK".getBytes());
            JobSubmissionValidator.validateFileSubmission(JobCapability.XLSX_EXTRACT, xlsx, null);
        }

        @Test
        void xlsx_extract_with_xlsm_accepted() {
            MultipartFile xlsm = new MockMultipartFile(
                    "file",
                    "macro.xlsm",
                    "application/vnd.ms-excel.sheet.macroenabled.12",
                    "PK".getBytes());
            JobSubmissionValidator.validateFileSubmission(JobCapability.XLSX_EXTRACT, xlsm, null);
        }

        @Test
        void xlsx_extract_accepts_xlsx_extension_with_octet_stream() {
            MultipartFile xlsx = new MockMultipartFile(
                    "file",
                    "sales.xlsx",
                    "application/octet-stream",
                    "PK".getBytes());
            JobSubmissionValidator.validateFileSubmission(JobCapability.XLSX_EXTRACT, xlsx, null);
        }

        @Test
        void xlsx_extract_with_legacy_xls_rejected_UNSUPPORTED_FILE_TYPE() {
            MultipartFile xls = new MockMultipartFile(
                    "file", "legacy.xls", "application/vnd.ms-excel", "BIFF".getBytes());
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.XLSX_EXTRACT, xls, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.UNSUPPORTED_FILE_TYPE);
        }

        @Test
        void xlsx_extract_with_legacy_xls_rejected_even_when_mime_is_xlsx() {
            MultipartFile xls = new MockMultipartFile(
                    "file",
                    "legacy.xls",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "BIFF".getBytes());
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.XLSX_EXTRACT, xls, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.UNSUPPORTED_FILE_TYPE);
        }

        @Test
        void ocr_with_gif_rejected_UNSUPPORTED_FILE_TYPE() {
            MultipartFile gif = new MockMultipartFile(
                    "file", "cat.gif", "image/gif", "GIF89a".getBytes());
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.OCR, gif, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.UNSUPPORTED_FILE_TYPE);
        }

        @Test
        void multimodal_with_webp_rejected_UNSUPPORTED_FILE_TYPE() {
            MultipartFile webp = new MockMultipartFile(
                    "file", "pic.webp", "image/webp", new byte[]{0x52, 0x49, 0x46, 0x46});
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.MULTIMODAL, webp, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.UNSUPPORTED_FILE_TYPE);
        }

        @Test
        void ocr_with_txt_extension_rejected_UNSUPPORTED_FILE_TYPE() {
            MultipartFile txt = new MockMultipartFile(
                    "file", "notes.txt", "text/plain", "hello".getBytes());
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.OCR, txt, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.UNSUPPORTED_FILE_TYPE);
        }

        @Test
        void ocr_accepts_when_extension_matches_even_if_content_type_is_octet_stream() {
            // Real-world clients (curl without --header) often ship
            // "application/octet-stream" regardless of the file payload.
            // The validator accepts when EITHER the mime OR the extension
            // matches, so this should pass.
            MultipartFile mixed = new MockMultipartFile(
                    "file", "Scan_1.PDF", "application/octet-stream",
                    "%PDF-1.4 fake".getBytes());
            JobSubmissionValidator.validateFileSubmission(JobCapability.OCR, mixed, null);
        }

        @Test
        void multimodal_accepts_when_content_type_matches_even_without_extension() {
            // And the inverse: some clients strip filenames entirely.
            MultipartFile mimeOnly = new MockMultipartFile(
                    "file", null, "image/png", new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47});
            JobSubmissionValidator.validateFileSubmission(
                    JobCapability.MULTIMODAL, mimeOnly, "describe");
        }

        // ---- RAG on multipart ----

        @Test
        void rag_via_multipart_without_text_rejected_TEXT_REQUIRED() {
            MultipartFile png = pngFile("reference.png");
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.RAG, png, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.TEXT_REQUIRED);
        }

        @Test
        void rag_via_multipart_with_blank_text_rejected_TEXT_REQUIRED() {
            MultipartFile png = pngFile("reference.png");
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.RAG, png, "   "))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.TEXT_REQUIRED);
        }

        @Test
        void rag_via_multipart_with_text_accepted() {
            // Multipart with RAG + text is allowed; the uploaded file is
            // ignored by the worker's RAG capability, but the contract is
            // explicit and the text is what drives retrieval.
            MultipartFile png = pngFile("reference.png");
            JobSubmissionValidator.validateFileSubmission(
                    JobCapability.RAG, png, "what does this reference say?");
        }

        // ---- MOCK on multipart (backward compat with any file type) ----

        @Test
        void mock_with_arbitrary_file_type_accepted_backward_compat() {
            MultipartFile arbitrary = new MockMultipartFile(
                    "file", "thing.bin", "application/octet-stream",
                    new byte[]{1, 2, 3, 4});
            JobSubmissionValidator.validateFileSubmission(JobCapability.MOCK, arbitrary, null);
        }

        // ---- AUTO on multipart: file OR text, PNG/JPEG/PDF when file present ----

        @Test
        void auto_with_text_only_accepted() {
            JobSubmissionValidator.validateFileSubmission(
                    JobCapability.AUTO, null, "route this question for me");
        }

        @Test
        void auto_with_file_only_accepted() {
            MultipartFile png = pngFile("receipt.png");
            JobSubmissionValidator.validateFileSubmission(
                    JobCapability.AUTO, png, null);
        }

        @Test
        void auto_with_file_and_text_accepted() {
            MultipartFile pdf = new MockMultipartFile(
                    "file", "report.pdf", "application/pdf", "%PDF-1.4 fake".getBytes());
            JobSubmissionValidator.validateFileSubmission(
                    JobCapability.AUTO, pdf, "what does this report say");
        }

        @Test
        void auto_with_neither_text_nor_file_rejected_AUTO_NO_INPUT() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.AUTO, null, null))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.AUTO_NO_INPUT);
        }

        @Test
        void auto_with_blank_text_and_no_file_rejected_AUTO_NO_INPUT() {
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.AUTO, null, "   "))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.AUTO_NO_INPUT);
        }

        @Test
        void auto_with_text_and_unsupported_file_type_rejected_UNSUPPORTED_FILE_TYPE() {
            // Fail fast at the boundary rather than letting the router
            // emit a clarify for a file it can never actually route to
            // OCR / MULTIMODAL.
            MultipartFile gif = new MockMultipartFile(
                    "file", "cat.gif", "image/gif", "GIF89a".getBytes());
            assertThatThrownBy(() ->
                    JobSubmissionValidator.validateFileSubmission(JobCapability.AUTO, gif, "what is this"))
                    .isInstanceOf(InvalidJobSubmissionException.class)
                    .extracting("errorCode").isEqualTo(ErrorCodes.UNSUPPORTED_FILE_TYPE);
        }

        @Test
        void auto_with_text_only_bypasses_file_type_gate() {
            // No file supplied at all — the unsupported-file check must
            // not fire because there's nothing to check.
            JobSubmissionValidator.validateFileSubmission(
                    JobCapability.AUTO, null, "describe the latest incident");
        }
    }

    // ===================================================================
    // helpers
    // ===================================================================

    private static MultipartFile pngFile(String filename) {
        // 8-byte PNG magic header is enough to pass both the validator's
        // mime/ext check AND any downstream magic-byte sniffing.
        return new MockMultipartFile(
                "file",
                filename,
                "image/png",
                new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A});
    }
}
