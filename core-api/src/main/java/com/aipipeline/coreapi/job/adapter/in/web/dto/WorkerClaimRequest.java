package com.aipipeline.coreapi.job.adapter.in.web.dto;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;

public record WorkerClaimRequest(
        @NotBlank String jobId,
        @NotBlank String workerClaimToken,
        @Min(1) int attemptNo
) {}
