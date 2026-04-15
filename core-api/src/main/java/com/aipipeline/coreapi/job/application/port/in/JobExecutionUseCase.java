package com.aipipeline.coreapi.job.application.port.in;

import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.aipipeline.coreapi.job.domain.JobId;
import com.aipipeline.coreapi.job.domain.JobStatus;

import java.util.List;

/**
 * Worker-facing operations. These are the ports the internal worker
 * controller depends on. They are deliberately split from the client-facing
 * port so that the two sides of the system have independent contracts.
 */
public interface JobExecutionUseCase {

    ClaimResult claim(ClaimCommand command);

    CallbackResult handleCallback(CallbackCommand command);

    // ---- value types ----

    record ClaimCommand(
            JobId jobId,
            String workerClaimToken,
            int attemptNo
    ) {}

    record ClaimResult(
            boolean granted,
            JobStatus currentStatus,
            String reason,
            JobCapability capability,
            int attemptNo,
            List<ClaimedInputArtifact> inputs
    ) {
        public static ClaimResult denied(JobStatus currentStatus, String reason) {
            return new ClaimResult(false, currentStatus, reason, null, 0, List.of());
        }
    }

    record ClaimedInputArtifact(
            String artifactId,
            ArtifactType type,
            String storageUri,
            String contentType,
            Long sizeBytes
    ) {}

    record CallbackCommand(
            JobId jobId,
            String callbackId,
            String workerClaimToken,
            CallbackOutcome outcome,
            String errorCode,
            String errorMessage,
            List<OutputArtifactRef> outputArtifacts
    ) {}

    enum CallbackOutcome { SUCCEEDED, FAILED }

    /**
     * Output artifact metadata supplied by the worker. The bytes are already
     * written into storage by the worker before the callback reaches us.
     */
    record OutputArtifactRef(
            ArtifactType type,
            String storageUri,
            String contentType,
            Long sizeBytes,
            String checksumSha256
    ) {}

    record CallbackResult(
            boolean applied,
            boolean duplicate,
            JobStatus currentStatus
    ) {}
}
