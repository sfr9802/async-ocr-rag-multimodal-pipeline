package com.aipipeline.coreapi.common.web;

import com.aipipeline.coreapi.job.adapter.in.web.dto.JobResponses;
import com.aipipeline.coreapi.job.application.service.InvalidJobSubmissionException;
import com.aipipeline.coreapi.job.domain.JobStateTransitionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * Translates domain / validation / not-found exceptions into tidy HTTP
 * error bodies without leaking stack traces.
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    /**
     * Capability-specific contract violations thrown by
     * {@link com.aipipeline.coreapi.job.application.service.JobSubmissionValidator}.
     * Handled before the generic IllegalArgumentException branch so the
     * stable error code from the validator is preserved instead of being
     * flattened into {@code INVALID_ARGUMENT}.
     */
    @ExceptionHandler(InvalidJobSubmissionException.class)
    public ResponseEntity<JobResponses.ErrorBody> handleInvalidSubmission(InvalidJobSubmissionException ex) {
        return ResponseEntity.badRequest()
                .body(new JobResponses.ErrorBody(ex.getErrorCode(), ex.getMessage()));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<JobResponses.ErrorBody> handleBadInput(IllegalArgumentException ex) {
        return ResponseEntity.badRequest()
                .body(new JobResponses.ErrorBody("INVALID_ARGUMENT", ex.getMessage()));
    }

    @ExceptionHandler(JobStateTransitionException.class)
    public ResponseEntity<JobResponses.ErrorBody> handleStateTransition(JobStateTransitionException ex) {
        return ResponseEntity.status(HttpStatus.CONFLICT)
                .body(new JobResponses.ErrorBody("INVALID_STATE_TRANSITION", ex.getMessage()));
    }

    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<JobResponses.ErrorBody> handleIllegalState(IllegalStateException ex) {
        return ResponseEntity.status(HttpStatus.CONFLICT)
                .body(new JobResponses.ErrorBody("CONFLICT", ex.getMessage()));
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<JobResponses.ErrorBody> handleValidation(MethodArgumentNotValidException ex) {
        String msg = ex.getBindingResult().getAllErrors().stream()
                .findFirst()
                .map(e -> e.getDefaultMessage())
                .orElse("validation failed");
        return ResponseEntity.badRequest()
                .body(new JobResponses.ErrorBody("VALIDATION_ERROR", msg));
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<JobResponses.ErrorBody> handleMaxUploadSize(MaxUploadSizeExceededException ex) {
        return ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE)
                .body(new JobResponses.ErrorBody("UPLOAD_TOO_LARGE", "multipart upload exceeds configured limit"));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<JobResponses.ErrorBody> handleUnexpected(Exception ex) {
        log.error("Unhandled exception", ex);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new JobResponses.ErrorBody("INTERNAL_ERROR", ex.getClass().getSimpleName()));
    }
}
