package com.aipipeline.coreapi.job.domain;

/**
 * Thrown when code attempts an illegal job status transition. This is a
 * domain-level exception; adapters translate it into HTTP 409 or similar.
 */
public class JobStateTransitionException extends RuntimeException {

    private final JobStatus from;
    private final JobStatus to;

    public JobStateTransitionException(JobStatus from, JobStatus to) {
        super("Illegal job state transition: " + from + " -> " + to);
        this.from = from;
        this.to = to;
    }

    public JobStatus getFrom() {
        return from;
    }

    public JobStatus getTo() {
        return to;
    }
}
