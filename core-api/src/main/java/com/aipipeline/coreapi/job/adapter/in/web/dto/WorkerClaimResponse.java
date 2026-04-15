package com.aipipeline.coreapi.job.adapter.in.web.dto;

import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase;

import java.util.List;

public record WorkerClaimResponse(
        boolean granted,
        String currentStatus,
        String reason,
        String capability,
        int attemptNo,
        List<ClaimedInput> inputs
) {

    public record ClaimedInput(
            String artifactId,
            String type,
            String storageUri,
            String contentType,
            Long sizeBytes
    ) {}

    public static WorkerClaimResponse from(JobExecutionUseCase.ClaimResult r) {
        List<ClaimedInput> mapped = r.inputs().stream()
                .map(i -> new ClaimedInput(
                        i.artifactId(),
                        i.type().name(),
                        i.storageUri(),
                        i.contentType(),
                        i.sizeBytes()))
                .toList();
        return new WorkerClaimResponse(
                r.granted(),
                r.currentStatus() == null ? null : r.currentStatus().name(),
                r.reason(),
                r.capability() == null ? null : r.capability().name(),
                r.attemptNo(),
                mapped);
    }
}
