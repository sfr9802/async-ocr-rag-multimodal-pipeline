package com.aipipeline.coreapi.job.application.port.out;

import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobId;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

/**
 * Outbound port for job persistence. The application layer talks to this
 * interface; the JPA adapter lives in adapter/out/persistence and is the
 * only thing that knows about Hibernate.
 */
public interface JobRepository {

    Job save(Job job);

    Optional<Job> findById(JobId id);

    /**
     * Atomic claim at the persistence layer.
     *
     * This is separate from the in-aggregate {@code Job.tryClaim(...)} check
     * because multiple workers may race on the same job row. Implementations
     * must execute a conditional UPDATE that only succeeds when the job's
     * current status is QUEUED (or PENDING), or RUNNING with an expired
     * lease, and either no live claim exists or the live claim is already
     * owned by the same token.
     *
     * @return true if this call owns the claim after the operation
     */
    boolean tryAtomicClaim(JobId id,
                           String workerClaimToken,
                           Duration leaseDuration,
                           Instant now);

    /**
     * Jobs that should have a Redis delivery available but may not.
     *
     * Includes QUEUED rows and expired RUNNING leases so the dispatch
     * reconciler can repair lost after-commit dispatches and worker crashes.
     */
    List<Job> findRedispatchCandidates(Instant now, int limit);
}
