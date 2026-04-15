package com.aipipeline.coreapi.job.adapter.out.persistence;

import com.aipipeline.coreapi.job.application.port.out.JobRepository;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.aipipeline.coreapi.job.domain.JobId;
import com.aipipeline.coreapi.job.domain.JobStatus;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.time.Instant;
import java.util.Optional;

/**
 * Outbound adapter implementing {@link JobRepository} on top of Spring Data
 * JPA. Handles bidirectional mapping between the JPA entity and the domain
 * aggregate.
 */
@Component
public class JobPersistenceAdapter implements JobRepository {

    private final JobJpaRepository jpa;

    public JobPersistenceAdapter(JobJpaRepository jpa) {
        this.jpa = jpa;
    }

    @Override
    public Job save(Job job) {
        JobJpaEntity entity = toEntity(job);
        JobJpaEntity saved = jpa.save(entity);
        return toDomain(saved);
    }

    @Override
    public Optional<Job> findById(JobId id) {
        return jpa.findById(id.value()).map(this::toDomain);
    }

    @Override
    public boolean tryAtomicClaim(JobId id, String workerClaimToken,
                                  Duration leaseDuration, Instant now) {
        Instant expiresAt = now.plus(leaseDuration);
        int updated = jpa.claimAtomic(id.value(), workerClaimToken, now, expiresAt);
        return updated > 0;
    }

    // ---- mapping ----

    private JobJpaEntity toEntity(Job job) {
        return new JobJpaEntity(
                job.getId().value(),
                job.getCapability().name(),
                job.getStatus().name(),
                job.getAttemptNo(),
                job.getClaimToken(),
                job.getClaimedAt(),
                job.getClaimExpiresAt(),
                job.getLastCallbackId(),
                job.getErrorCode(),
                job.getErrorMessage(),
                job.getCreatedAt(),
                job.getUpdatedAt());
    }

    private Job toDomain(JobJpaEntity e) {
        return Job.rehydrate(
                JobId.of(e.getId()),
                JobCapability.valueOf(e.getCapability()),
                JobStatus.valueOf(e.getStatus()),
                e.getAttemptNo(),
                e.getClaimToken(),
                e.getClaimedAt(),
                e.getClaimExpiresAt(),
                e.getLastCallbackId(),
                e.getErrorCode(),
                e.getErrorMessage(),
                e.getCreatedAt(),
                e.getUpdatedAt());
    }
}
