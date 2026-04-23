package com.aipipeline.coreapi.job.application.service;

import com.aipipeline.coreapi.job.domain.JobCapability;
import org.springframework.web.multipart.MultipartFile;

import java.util.Locale;
import java.util.Set;

/**
 * Capability-aware validation for incoming job submissions.
 *
 * <p>This is the first thing the controller calls after parsing the request,
 * BEFORE any artifact bytes are staged and before {@code createAndEnqueue} is
 * invoked. The invariant it enforces: <b>invalid submissions never reach the
 * async pipeline</b>. If the validator throws, no storage write, no Artifact
 * row, no Job row, and no Redis dispatch has happened yet.
 *
 * <h2>Capability / input matrix</h2>
 *
 * <table>
 *   <tr><th>Capability</th><th>Endpoint</th><th>Text</th><th>File</th><th>Allowed file types</th></tr>
 *   <tr><td>MOCK</td>       <td>JSON</td>     <td>optional (may be empty)</td> <td>-</td>            <td>-</td></tr>
 *   <tr><td>MOCK</td>       <td>multipart</td><td>optional</td>                <td>required, non-empty</td> <td>any</td></tr>
 *   <tr><td>RAG</td>        <td>JSON</td>     <td><b>required, non-blank</b></td> <td>-</td>         <td>-</td></tr>
 *   <tr><td>RAG</td>        <td>multipart</td><td><b>required, non-blank</b></td> <td>ignored</td>   <td>any</td></tr>
 *   <tr><td>OCR</td>        <td>JSON</td>     <td>-</td>                       <td><b>rejected: FILE_REQUIRED</b></td> <td>-</td></tr>
 *   <tr><td>OCR</td>        <td>multipart</td><td>optional (ignored by worker)</td> <td><b>required, non-empty</b></td> <td>PNG, JPEG, PDF</td></tr>
 *   <tr><td>MULTIMODAL</td> <td>JSON</td>     <td>-</td>                       <td><b>rejected: FILE_REQUIRED</b></td> <td>-</td></tr>
 *   <tr><td>MULTIMODAL</td> <td>multipart</td><td>optional</td>                <td><b>required, non-empty</b></td> <td>PNG, JPEG, PDF</td></tr>
 *   <tr><td>AUTO</td>       <td>JSON</td>     <td>optional (may be blank)</td> <td>-</td>            <td>-</td></tr>
 *   <tr><td>AUTO</td>       <td>multipart</td><td>optional</td>                <td>optional, non-empty (at least one of text/file required)</td> <td>PNG, JPEG, PDF</td></tr>
 *   <tr><td>AGENT</td>      <td>JSON</td>     <td>optional (may be blank)</td> <td>-</td>            <td>-</td></tr>
 *   <tr><td>AGENT</td>      <td>multipart</td><td>optional</td>                <td>optional, non-empty (at least one of text/file required)</td> <td>PNG, JPEG, PDF</td></tr>
 * </table>
 *
 * <h2>Error codes</h2>
 *
 * See {@link ErrorCodes}. Every code is stable enough to be matched by a
 * client without parsing the human-readable message.
 *
 * <p>Implemented as a final utility class with static methods because the
 * validator is pure, stateless logic and does not need Spring container
 * management. Tests instantiate nothing — they just call the methods.
 */
public final class JobSubmissionValidator {

    /**
     * Stable error-code constants. Anything thrown by this class carries one
     * of these values in {@link InvalidJobSubmissionException#getErrorCode()}.
     */
    public static final class ErrorCodes {
        /** Capability field was null, empty, or blank. */
        public static final String CAPABILITY_REQUIRED = "CAPABILITY_REQUIRED";
        /** Capability value did not match any {@link JobCapability} enum. */
        public static final String UNKNOWN_CAPABILITY = "UNKNOWN_CAPABILITY";
        /** Capability requires non-blank text but none was supplied. */
        public static final String TEXT_REQUIRED = "TEXT_REQUIRED";
        /** Capability requires a file upload but none was supplied (or wrong endpoint). */
        public static final String FILE_REQUIRED = "FILE_REQUIRED";
        /** A file field was present but carried zero bytes. */
        public static final String FILE_EMPTY = "FILE_EMPTY";
        /** File's content-type/extension is not allowed for this capability. */
        public static final String UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE";
        /** AUTO multipart submission with neither a non-blank text nor a file. */
        public static final String AUTO_NO_INPUT = "AUTO_NO_INPUT";

        private ErrorCodes() {}
    }

    // MIME prefixes / extensions accepted for OCR and MULTIMODAL file inputs.
    // Mirrors the worker-side OCR + multimodal classifier lists so the HTTP
    // contract matches the capability contract.
    private static final Set<String> SUPPORTED_FILE_MIME_TYPES = Set.of(
            "image/png",
            "image/jpeg",
            "image/jpg",
            "application/pdf",
            "application/x-pdf"
    );
    private static final Set<String> SUPPORTED_FILE_EXTENSIONS = Set.of(
            "png", "jpg", "jpeg", "pdf"
    );

    private JobSubmissionValidator() {}

    // ------------------------------------------------------------------
    // capability parsing
    // ------------------------------------------------------------------

    /**
     * Parse a raw capability string into a {@link JobCapability}.
     *
     * Unlike {@link JobCapability#fromString}, unknown / blank values
     * become {@link InvalidJobSubmissionException} with stable error codes
     * (CAPABILITY_REQUIRED / UNKNOWN_CAPABILITY) instead of the generic
     * {@code INVALID_ARGUMENT} bucket.
     */
    public static JobCapability parseCapability(String raw) {
        if (raw == null || raw.isBlank()) {
            throw new InvalidJobSubmissionException(
                    ErrorCodes.CAPABILITY_REQUIRED,
                    "capability field is required. Accepted values: MOCK, RAG, OCR, MULTIMODAL, AUTO, AGENT.");
        }
        try {
            return JobCapability.fromString(raw);
        } catch (IllegalArgumentException ex) {
            throw new InvalidJobSubmissionException(
                    ErrorCodes.UNKNOWN_CAPABILITY,
                    "Unknown capability: " + raw + ". Accepted values: MOCK, RAG, OCR, MULTIMODAL, AUTO, AGENT.");
        }
    }

    // ------------------------------------------------------------------
    // text-endpoint validation (JSON submission)
    // ------------------------------------------------------------------

    /**
     * Validate a JSON ("text") job submission.
     *
     * <p>Rules:
     * <ul>
     *   <li>OCR / MULTIMODAL on the text endpoint → FILE_REQUIRED. These
     *       capabilities demand a file upload and must use the multipart
     *       endpoint.</li>
     *   <li>RAG with null / blank text → TEXT_REQUIRED. The worker's RAG
     *       capability already rejects empty queries with {@code EMPTY_QUERY},
     *       but we surface that earlier as a typed boundary error so no
     *       pipeline side effects occur.</li>
     *   <li>MOCK with empty (but non-null) text → accepted, preserves phase-1
     *       compatibility where the MOCK capability simply echoes whatever
     *       bytes it receives.</li>
     *   <li>MOCK with null text → TEXT_REQUIRED. {@code text} is a required
     *       field on the JSON contract even for MOCK; the controller reads
     *       {@code text()} and passes bytes straight to storage, so null
     *       would NPE downstream.</li>
     * </ul>
     *
     * Throws {@link InvalidJobSubmissionException} on violation. All checks
     * run before any storage or enqueue work, so rejection leaves the
     * pipeline in a pristine state.
     */
    public static void validateTextSubmission(JobCapability capability, String text) {
        switch (capability) {
            case OCR -> throw new InvalidJobSubmissionException(
                    ErrorCodes.FILE_REQUIRED,
                    "OCR jobs require a file upload. Use the multipart endpoint "
                            + "(POST /api/v1/jobs with Content-Type: multipart/form-data, "
                            + "fields: capability=OCR, file=@path/to/input.(png|jpg|jpeg|pdf)).");
            case MULTIMODAL -> throw new InvalidJobSubmissionException(
                    ErrorCodes.FILE_REQUIRED,
                    "MULTIMODAL jobs require a file upload. Use the multipart endpoint "
                            + "(POST /api/v1/jobs with Content-Type: multipart/form-data, "
                            + "fields: capability=MULTIMODAL, file=@path/to/input.(png|jpg|jpeg|pdf), "
                            + "text=<optional user question>).");
            case RAG -> {
                if (text == null || text.isBlank()) {
                    throw new InvalidJobSubmissionException(
                            ErrorCodes.TEXT_REQUIRED,
                            "RAG jobs require a non-blank 'text' field in the JSON body.");
                }
            }
            case MOCK -> {
                if (text == null) {
                    throw new InvalidJobSubmissionException(
                            ErrorCodes.TEXT_REQUIRED,
                            "MOCK jobs on the JSON endpoint require a 'text' field "
                                    + "(empty string is acceptable for backward compatibility).");
                }
            }
            case AUTO, AGENT -> {
                // AUTO / AGENT on the JSON endpoint accept any non-null
                // text — blank / whitespace-only is fine because the
                // router emits a clarify FINAL_RESPONSE rather than
                // rejecting the job. Null is rejected so the controller's
                // text-to-bytes path does not NPE (matches the MOCK-JSON
                // contract). AGENT shares the rule matrix with AUTO; the
                // only Phase 6 difference is the loop behind the dispatch.
                if (text == null) {
                    throw new InvalidJobSubmissionException(
                            ErrorCodes.TEXT_REQUIRED,
                            capability + " jobs on the JSON endpoint require a 'text' "
                                    + "field (empty string is acceptable — use it "
                                    + "to force a clarify response). To include a "
                                    + "file, use the multipart endpoint.");
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // multipart-endpoint validation
    // ------------------------------------------------------------------

    /**
     * Validate a multipart job submission.
     *
     * <p>The multipart endpoint's implicit contract is "you are uploading a
     * file". This method enforces that for all capabilities, then layers
     * capability-specific rules on top:
     *
     * <ul>
     *   <li>File is required (FILE_REQUIRED) and non-empty (FILE_EMPTY) for
     *       every capability on this endpoint, including MOCK, so that the
     *       upload has semantics a worker can actually use.</li>
     *   <li>For OCR and MULTIMODAL, the file must be PNG, JPEG, or PDF —
     *       enforced by content-type prefix OR by filename extension.
     *       Requiring <b>both</b> to agree is too strict for real clients
     *       (curl sometimes omits content-type); requiring <b>at least one</b>
     *       to match the supported set is the v1 contract. Mismatches yield
     *       UNSUPPORTED_FILE_TYPE.</li>
     *   <li>For RAG on multipart, a non-blank text field is required
     *       (TEXT_REQUIRED). RAG only uses INPUT_TEXT at the worker side —
     *       the uploaded file is ignored by the capability but allowed so
     *       that clients can submit a single multipart payload with both a
     *       text query and a reference document.</li>
     *   <li>For MULTIMODAL, the text field is OPTIONAL — a blank or missing
     *       text field is accepted and the fusion layer falls back to a
     *       neutral default retrieval query.</li>
     *   <li>MOCK is left unopinionated: any file type is accepted, text is
     *       optional. Preserves phase-1 backward compatibility.</li>
     *   <li>AUTO accepts a text-only submission, a file-only submission, or
     *       both — the router inspects whichever arrived. At least one of
     *       (non-blank text, non-empty file) is required; the all-missing
     *       case fails fast with AUTO_NO_INPUT rather than creating a job
     *       that can only ever emit a clarify response. When a file IS
     *       supplied its type must be PNG/JPEG/PDF (same rule as
     *       MULTIMODAL — anything else fails the worker-side router
     *       anyway).</li>
     * </ul>
     */
    public static void validateFileSubmission(
            JobCapability capability,
            MultipartFile file,
            String text
    ) {
        // AUTO and AGENT are the capabilities where the file field is
        // OPTIONAL on the multipart endpoint — the decision is routed
        // from the (text, file) pair, so either side is enough. Every
        // other capability requires a real file; the shared "file must
        // exist + non-empty" guard below runs for them after the
        // AUTO/AGENT branch's own text/file check.
        if (capability == JobCapability.AUTO || capability == JobCapability.AGENT) {
            validateAutoMultipart(capability, file, text);
            return;
        }

        if (file == null) {
            throw new InvalidJobSubmissionException(
                    ErrorCodes.FILE_REQUIRED,
                    capability + " job requires a 'file' form field on the multipart endpoint.");
        }
        if (file.isEmpty() || file.getSize() <= 0) {
            throw new InvalidJobSubmissionException(
                    ErrorCodes.FILE_EMPTY,
                    "Uploaded file is empty (0 bytes). Check that the client actually attached "
                            + "content to the 'file' form field.");
        }

        switch (capability) {
            case OCR, MULTIMODAL -> requireSupportedFileType(capability, file);
            case RAG -> {
                if (text == null || text.isBlank()) {
                    throw new InvalidJobSubmissionException(
                            ErrorCodes.TEXT_REQUIRED,
                            "RAG jobs require a non-blank 'text' form field. The RAG capability "
                                    + "only uses INPUT_TEXT — the uploaded file is ignored.");
                }
            }
            case MOCK -> {
                // Phase-1 compatibility: MOCK accepts any non-empty file and
                // any (or no) text. No additional checks.
            }
            case AUTO, AGENT -> {
                // Handled above via the early-return branch.
            }
        }
    }

    private static void validateAutoMultipart(
            JobCapability capability,
            MultipartFile file,
            String text
    ) {
        boolean hasText = text != null && !text.isBlank();
        boolean hasFile = file != null && !file.isEmpty() && file.getSize() > 0;

        if (!hasText && !hasFile) {
            throw new InvalidJobSubmissionException(
                    ErrorCodes.AUTO_NO_INPUT,
                    capability + " jobs on the multipart endpoint require AT LEAST ONE of: "
                            + "a non-blank 'text' form field, or a non-empty 'file' form field. "
                            + "Neither was supplied — the router would have nothing to route on.");
        }

        // A file that is present but empty is always an error, even on
        // AUTO / AGENT — core-api would stage a zero-byte INPUT_FILE
        // that the worker's routing classifier cannot use.
        if (file != null && !file.isEmpty() && file.getSize() <= 0) {
            throw new InvalidJobSubmissionException(
                    ErrorCodes.FILE_EMPTY,
                    capability + " job supplied a 'file' form field but its size is 0 bytes. "
                            + "Omit the file entirely to submit a text-only "
                            + capability + " job.");
        }

        // When a file IS present, enforce the same PNG/JPEG/PDF rule as
        // MULTIMODAL. The worker-side router will otherwise return a
        // clarify response — we fail fast at the boundary instead so the
        // client gets a clean UNSUPPORTED_FILE_TYPE immediately.
        if (hasFile) {
            requireSupportedFileType(capability, file);
        }
    }

    // ------------------------------------------------------------------
    // internals
    // ------------------------------------------------------------------

    private static void requireSupportedFileType(JobCapability capability, MultipartFile file) {
        String mime = normalizeMime(file.getContentType());
        String ext = extractExtension(file.getOriginalFilename());

        boolean mimeMatches = mime != null && SUPPORTED_FILE_MIME_TYPES.contains(mime);
        boolean extMatches = ext != null && SUPPORTED_FILE_EXTENSIONS.contains(ext);

        if (!mimeMatches && !extMatches) {
            throw new InvalidJobSubmissionException(
                    ErrorCodes.UNSUPPORTED_FILE_TYPE,
                    "Unsupported file type for " + capability + ". "
                            + "Received contentType=" + describe(file.getContentType())
                            + " filename=" + describe(file.getOriginalFilename())
                            + ". Supported types: PNG, JPEG, PDF "
                            + "(image/png, image/jpeg, application/pdf).");
        }
    }

    /**
     * Normalize a raw HTTP content-type header to its lower-case primary type
     * ("image/png; charset=..." → "image/png"). Returns null for null input.
     */
    private static String normalizeMime(String raw) {
        if (raw == null) {
            return null;
        }
        String trimmed = raw.trim().toLowerCase(Locale.ROOT);
        if (trimmed.isEmpty()) {
            return null;
        }
        int semi = trimmed.indexOf(';');
        return semi >= 0 ? trimmed.substring(0, semi).trim() : trimmed;
    }

    /**
     * Extract the lower-case extension from a filename ("Receipt.PNG" → "png").
     * Returns null for null / extension-less names.
     */
    private static String extractExtension(String filename) {
        if (filename == null) {
            return null;
        }
        String lower = filename.toLowerCase(Locale.ROOT);
        int dot = lower.lastIndexOf('.');
        if (dot < 0 || dot == lower.length() - 1) {
            return null;
        }
        return lower.substring(dot + 1);
    }

    private static String describe(String value) {
        return value == null ? "<null>" : "'" + value + "'";
    }
}
