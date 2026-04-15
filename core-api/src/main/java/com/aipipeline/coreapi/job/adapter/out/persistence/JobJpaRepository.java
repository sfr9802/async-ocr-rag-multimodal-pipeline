package com.aipipeline.coreapi.job.adapter.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.Instant;

/**
 * Spring Data JPA repository. Only this interface knows the query shape —
 * the application layer depends on {@link com.aipipeline.coreapi.job.application.port.out.JobRepository}.
 */
public interface JobJpaRepository extends JpaRepository<JobJpaEntity, String> {

    /**
     * Atomic claim via conditional UPDATE.
     *
     * Succeeds (returns 1) only when the row's status is PENDING or QUEUED,
     * and either no claim is currently held, the current claim has expired,
     * or the current claim is already owned by the same token (idempotent
     * re-claim).
     */
    @Modifying
    @Query("""
           UPDATE JobJpaEntity j
              SET j.status = 'RUNNING',
                  j.claimToken = :token,
                  j.claimedAt = :now,
                  j.claimExpiresAt = :expiresAt,
                  j.updatedAt = :now
            WHERE j.id = :id
              AND j.status IN ('PENDING', 'QUEUED')
              AND (
                   j.claimToken IS NULL
                OR j.claimExpiresAt IS NULL
                OR j.claimExpiresAt < :now
                OR j.claimToken = :token
              )
           """)
    int claimAtomic(@Param("id") String id,
                    @Param("token") String workerClaimToken,
                    @Param("now") Instant now,
                    @Param("expiresAt") Instant expiresAt);
}
