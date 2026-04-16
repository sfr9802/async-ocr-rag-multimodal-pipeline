package com.aipipeline.coreapi.job.adapter.in.web.dto;

/**
 * Text-only job submission. File uploads use the multipart endpoint on the
 * same controller — this record handles the common "just a text prompt"
 * shape.
 *
 * <p>No Jakarta Validation annotations are present by design: capability and
 * text validity are handled uniformly by
 * {@link com.aipipeline.coreapi.job.application.service.JobSubmissionValidator}
 * so that both the JSON and multipart submission paths return the same stable
 * error codes (CAPABILITY_REQUIRED, UNKNOWN_CAPABILITY, TEXT_REQUIRED, ...).
 * Relying on {@code @NotBlank} here used to produce a generic
 * {@code VALIDATION_ERROR} that did not fit into the capability/input matrix.
 */
public record CreateJobRequest(
        String capability,
        String text
) {}
