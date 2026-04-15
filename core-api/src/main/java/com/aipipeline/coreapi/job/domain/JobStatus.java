package com.aipipeline.coreapi.job.domain;

import java.util.Set;

/**
 * Job lifecycle states.
 *
 * Transition rules are encoded here as a small state machine so that every
 * transition in the application layer can be validated against a single
 * authoritative source.
 *
 * <pre>
 *   PENDING ── enqueue ──▶ QUEUED
 *   QUEUED  ── claim   ──▶ RUNNING
 *   RUNNING ── success ──▶ SUCCEEDED
 *   RUNNING ── failure ──▶ FAILED
 * </pre>
 *
 * Retry support (RUNNING → QUEUED / FAILED → QUEUED) is intentionally left
 * for a later phase; the enum values are stable but transitions are
 * restricted to the happy path for now.
 */
public enum JobStatus {

    PENDING,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    FAILED;

    private static final Set<JobStatus> TERMINAL = Set.of(SUCCEEDED, FAILED);

    public boolean isTerminal() {
        return TERMINAL.contains(this);
    }

    public boolean canTransitionTo(JobStatus next) {
        return switch (this) {
            case PENDING    -> next == QUEUED || next == FAILED;
            case QUEUED     -> next == RUNNING || next == FAILED;
            case RUNNING    -> next == SUCCEEDED || next == FAILED;
            case SUCCEEDED, FAILED -> false;
        };
    }

    /**
     * Throws if the transition is not allowed. Kept as a helper so that
     * services can fail loudly at the domain boundary rather than silently
     * overwriting invalid states.
     */
    public void ensureCanTransitionTo(JobStatus next) {
        if (!canTransitionTo(next)) {
            throw new JobStateTransitionException(this, next);
        }
    }
}
