package com.aipipeline.coreapi.common.web;

import com.aipipeline.coreapi.job.adapter.in.web.dto.JobResponses;
import com.aipipeline.coreapi.job.domain.JobStateTransitionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
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

    @ExceptionHandler(Exception.class)
    public ResponseEntity<JobResponses.ErrorBody> handleUnexpected(Exception ex) {
        log.error("Unhandled exception", ex);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new JobResponses.ErrorBody("INTERNAL_ERROR", ex.getClass().getSimpleName()));
    }
}
