package com.aipipeline.coreapi.queue.adapter.out.redis;

import java.util.List;

/**
 * Wire shape of a dispatch message pushed onto Redis. This is the contract
 * the ai-worker queue consumer unmarshals — keep it stable and minimal.
 *
 * The queue is still a "wake up, come and claim" signal, not a full task
 * payload: workers fetch authoritative input bytes and sourceFileId via the
 * claim response. The optional inputArtifacts list carries non-secret metadata
 * so queue-level observers can confirm the intended INPUT_FILE contract.
 */
public record QueueMessage(
        String jobId,
        String capability,
        String taskKind,
        String pipelineVersion,
        int attemptNo,
        long enqueuedAtEpochMilli,
        String callbackBaseUrl,
        List<InputArtifactMessage> inputArtifacts
) {
    public QueueMessage(
            String jobId,
            String capability,
            String taskKind,
            String pipelineVersion,
            int attemptNo,
            long enqueuedAtEpochMilli,
            String callbackBaseUrl
    ) {
        this(
                jobId,
                capability,
                taskKind,
                pipelineVersion,
                attemptNo,
                enqueuedAtEpochMilli,
                callbackBaseUrl,
                List.of());
    }

    public record InputArtifactMessage(
            String artifactId,
            String role,
            String type,
            String storageUri,
            String contentType,
            Long sizeBytes,
            String checksumSha256
    ) {}
}
