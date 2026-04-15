package com.aipipeline.coreapi.job.adapter.in.web.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.util.List;

public record WorkerCallbackRequest(
        @NotBlank String jobId,
        @NotBlank String callbackId,
        @NotBlank String workerClaimToken,
        @NotNull Outcome outcome,
        String errorCode,
        String errorMessage,
        List<OutputArtifact> outputArtifacts
) {

    public enum Outcome { SUCCEEDED, FAILED }

    public record OutputArtifact(
            @NotBlank String type,
            @NotBlank String storageUri,
            String contentType,
            Long sizeBytes,
            String checksumSha256
    ) {}
}
