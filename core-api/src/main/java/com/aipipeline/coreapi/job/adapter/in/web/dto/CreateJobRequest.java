package com.aipipeline.coreapi.job.adapter.in.web.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

/**
 * Text-only job submission. File uploads use the multipart endpoint on the
 * same controller — this record handles the common "just a text prompt"
 * shape.
 */
public record CreateJobRequest(
        @NotBlank String capability,
        @NotNull String text
) {}
