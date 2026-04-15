package com.aipipeline.coreapi.job.domain;

import java.time.Duration;
import java.time.Instant;

/**
 * Job aggregate root.
 *
 * Intentionally a plain domain object with no persistence annotations. The
 * JPA entity lives in the outbound persistence adapter and is mapped to/from
 * this type, so the domain layer stays free of framework concerns.
 *
 * Mutation happens through explicit state-transition methods; fields are not
 * publicly settable. This keeps the lifecycle rules in one place.
 */
public class Job {

    private final JobId id;
    private final JobCapability capability;
    private final Instant createdAt;

    private JobStatus status;
    private int attemptNo;
    private String claimToken;
    private Instant claimedAt;
    private Instant claimExpiresAt;
    private String lastCallbackId;
    private String errorCode;
    private String errorMessage;
    private Instant updatedAt;

    // ---- constructors ----

    private Job(JobId id,
                JobCapability capability,
                JobStatus status,
                int attemptNo,
                String claimToken,
                Instant claimedAt,
                Instant claimExpiresAt,
                String lastCallbackId,
                String errorCode,
                String errorMessage,
                Instant createdAt,
                Instant updatedAt) {
        this.id = id;
        this.capability = capability;
        this.status = status;
        this.attemptNo = attemptNo;
        this.claimToken = claimToken;
        this.claimedAt = claimedAt;
        this.claimExpiresAt = claimExpiresAt;
        this.lastCallbackId = lastCallbackId;
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    /** Factory for brand-new jobs (status = PENDING). */
    public static Job createNew(JobCapability capability, Instant now) {
        return new Job(
                JobId.generate(),
                capability,
                JobStatus.PENDING,
                1,
                null, null, null,
                null,
                null, null,
                now, now);
    }

    /** Rehydration factory used by the persistence adapter only. */
    public static Job rehydrate(JobId id,
                                JobCapability capability,
                                JobStatus status,
                                int attemptNo,
                                String claimToken,
                                Instant claimedAt,
                                Instant claimExpiresAt,
                                String lastCallbackId,
                                String errorCode,
                                String errorMessage,
                                Instant createdAt,
                                Instant updatedAt) {
        return new Job(id, capability, status, attemptNo,
                claimToken, claimedAt, claimExpiresAt,
                lastCallbackId, errorCode, errorMessage,
                createdAt, updatedAt);
    }

    // ---- state transitions ----

    /** PENDING -> QUEUED (called right after the dispatch adapter enqueues). */
    public void markQueued(Instant now) {
        status.ensureCanTransitionTo(JobStatus.QUEUED);
        this.status = JobStatus.QUEUED;
        this.updatedAt = now;
    }

    /**
     * QUEUED -> RUNNING with a claim lease. Returns true if the claim was
     * granted. A claim is only granted when the current status is QUEUED
     * (or PENDING, as a convenience for the local-dev path) and no other
     * live lease is held.
     *
     * Atomic persistence-level claim guards against concurrent workers; this
     * in-memory method is the in-aggregate check and lease accounting.
     */
    public boolean tryClaim(String workerClaimToken, Duration leaseDuration, Instant now) {
        if (status != JobStatus.QUEUED && status != JobStatus.PENDING) {
            return false;
        }
        if (claimExpiresAt != null && claimExpiresAt.isAfter(now)
                && claimToken != null && !claimToken.equals(workerClaimToken)) {
            // a different worker currently holds a live lease
            return false;
        }
        this.status = JobStatus.RUNNING;
        this.claimToken = workerClaimToken;
        this.claimedAt = now;
        this.claimExpiresAt = now.plus(leaseDuration);
        this.updatedAt = now;
        return true;
    }

    /** RUNNING -> SUCCEEDED. */
    public void markSucceeded(String callbackId, Instant now) {
        status.ensureCanTransitionTo(JobStatus.SUCCEEDED);
        this.status = JobStatus.SUCCEEDED;
        this.lastCallbackId = callbackId;
        this.errorCode = null;
        this.errorMessage = null;
        this.updatedAt = now;
    }

    /** * -> FAILED. */
    public void markFailed(String callbackId, String errorCode, String errorMessage, Instant now) {
        status.ensureCanTransitionTo(JobStatus.FAILED);
        this.status = JobStatus.FAILED;
        this.lastCallbackId = callbackId;
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.updatedAt = now;
    }

    /** Returns true if this callback id has already been applied (idempotency). */
    public boolean isDuplicateCallback(String callbackId) {
        return callbackId != null && callbackId.equals(this.lastCallbackId);
    }

    // ---- accessors ----

    public JobId getId() { return id; }
    public JobCapability getCapability() { return capability; }
    public JobStatus getStatus() { return status; }
    public int getAttemptNo() { return attemptNo; }
    public String getClaimToken() { return claimToken; }
    public Instant getClaimedAt() { return claimedAt; }
    public Instant getClaimExpiresAt() { return claimExpiresAt; }
    public String getLastCallbackId() { return lastCallbackId; }
    public String getErrorCode() { return errorCode; }
    public String getErrorMessage() { return errorMessage; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
}
