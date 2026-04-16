package com.aipipeline.coreapi.job.application.service;

/**
 * Thrown by {@link JobSubmissionValidator} when a client-facing job submission
 * violates the capability/input contract (see {@code docs/api-summary.md} and
 * the capability/input matrix).
 *
 * Unlike a plain {@link IllegalArgumentException}, this exception carries a
 * <b>stable error code</b> so the HTTP error body can be matched by clients
 * without string-parsing the message. Codes are defined in
 * {@link JobSubmissionValidator.ErrorCodes} and documented in
 * {@code docs/api-summary.md}.
 *
 * {@link com.aipipeline.coreapi.common.web.GlobalExceptionHandler} translates
 * this into a 400 Bad Request with {@code { "code": "...", "message": "..." }}.
 *
 * This is a pure boundary-level error: throwing it guarantees that no storage,
 * artifact, or job-row side effects have occurred yet — the validator is the
 * first thing the controller calls, before any {@code storage.store(...)} or
 * {@code jobManagement.createAndEnqueue(...)} work.
 */
public class InvalidJobSubmissionException extends RuntimeException {

    private final String errorCode;

    public InvalidJobSubmissionException(String errorCode, String message) {
        super(message);
        if (errorCode == null || errorCode.isBlank()) {
            throw new IllegalArgumentException("errorCode must not be blank");
        }
        this.errorCode = errorCode;
    }

    public String getErrorCode() {
        return errorCode;
    }
}
