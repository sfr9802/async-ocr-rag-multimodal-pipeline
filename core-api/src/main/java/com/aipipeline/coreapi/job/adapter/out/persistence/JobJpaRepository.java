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
     * Succeeds (returns 1) only when the row is dispatchable: PENDING,
     * QUEUED, or RUNNING with an expired/current-owner claim. RUNNING is
     * included so a worker crash can be recovered after the lease expires.
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
              AND j.status IN ('PENDING', 'QUEUED', 'RUNNING')
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

    /**
     * Jobs that should have a queue message available.
     *
     * QUEUED jobs may have lost their after-commit Redis dispatch. RUNNING
     * jobs with an expired lease may have lost their worker after BRPOP.
     */
    @Query("""
           SELECT j
             FROM JobJpaEntity j
            WHERE j.status = 'QUEUED'
               OR (
                    j.status = 'RUNNING'
                AND (
                    j.claimExpiresAt IS NULL
                 OR j.claimExpiresAt < :now
                )
               )
            ORDER BY j.updatedAt ASC
           """)
    java.util.List<JobJpaEntity> findRedispatchCandidates(
            @Param("now") Instant now,
            org.springframework.data.domain.Pageable pageable);
}
