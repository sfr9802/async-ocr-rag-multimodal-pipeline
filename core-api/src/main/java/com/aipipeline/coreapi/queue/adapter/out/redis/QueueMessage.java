package com.aipipeline.coreapi.queue.adapter.out.redis;

/**
 * Wire shape of a dispatch message pushed onto Redis. This is the contract
 * the ai-worker queue consumer unmarshals — keep it stable and minimal.
 *
 * Explicitly NOT included:
 *   - input artifact metadata (worker fetches it via the claim response)
 *   - any credentials
 *
 * The queue is a "wake up, come and claim" signal, not a full task payload.
 */
public record QueueMessage(
        String jobId,
        String capability,
        String taskKind,
        String pipelineVersion,
        int attemptNo,
        long enqueuedAtEpochMilli,
        String callbackBaseUrl
) {}
