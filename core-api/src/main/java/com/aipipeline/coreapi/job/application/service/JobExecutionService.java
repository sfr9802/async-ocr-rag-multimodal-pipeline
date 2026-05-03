package com.aipipeline.coreapi.job.application.service;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactRepository;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactRole;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.catalog.application.service.DocumentCatalogService;
import com.aipipeline.coreapi.common.AipipelineProperties;
import com.aipipeline.coreapi.common.TimeProvider;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase;
import com.aipipeline.coreapi.job.application.port.out.JobRepository;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.aipipeline.coreapi.job.domain.JobStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Worker-facing application service — implements claim and callback.
 *
 * These two operations are the contract the worker depends on, and together
 * they carry the job from QUEUED → RUNNING → SUCCEEDED/FAILED. Everything
 * outside of this service should treat them as atomic, idempotent units.
 */
@Service
public class JobExecutionService implements JobExecutionUseCase {

    private static final Logger log = LoggerFactory.getLogger(JobExecutionService.class);

    private final JobRepository jobRepository;
    private final ArtifactRepository artifactRepository;
    private final DocumentCatalogService catalogService;
    private final TimeProvider timeProvider;
    private final Duration claimLease;

    public JobExecutionService(JobRepository jobRepository,
                               ArtifactRepository artifactRepository,
                               DocumentCatalogService catalogService,
                               TimeProvider timeProvider,
                               AipipelineProperties properties) {
        this.jobRepository = jobRepository;
        this.artifactRepository = artifactRepository;
        this.catalogService = catalogService;
        this.timeProvider = timeProvider;
        this.claimLease = Duration.ofSeconds(properties.claim().leaseSeconds());
    }

    JobExecutionService(JobRepository jobRepository,
                        ArtifactRepository artifactRepository,
                        TimeProvider timeProvider,
                        AipipelineProperties properties) {
        this(jobRepository, artifactRepository, null, timeProvider, properties);
    }

    // ------------------------------------------------------------------
    // Claim
    // ------------------------------------------------------------------

    @Override
    @Transactional
    public ClaimResult claim(ClaimCommand command) {
        Instant now = timeProvider.now();

        Optional<Job> existing = jobRepository.findById(command.jobId());
        if (existing.isEmpty()) {
            return ClaimResult.denied(null, "JOB_NOT_FOUND");
        }
        Job job = existing.get();

        if (job.getStatus().isTerminal()) {
            return ClaimResult.denied(job.getStatus(), "JOB_TERMINAL");
        }
        if (job.getStatus() == JobStatus.RUNNING
                && job.getClaimExpiresAt() != null
                && job.getClaimExpiresAt().isAfter(now)
                && !command.workerClaimToken().equals(job.getClaimToken())) {
            return ClaimResult.denied(job.getStatus(), "ALREADY_CLAIMED");
        }

        boolean granted = jobRepository.tryAtomicClaim(
                command.jobId(), command.workerClaimToken(), claimLease, now);
        if (!granted) {
            Job refreshed = jobRepository.findById(command.jobId()).orElse(job);
            return ClaimResult.denied(refreshed.getStatus(), "CLAIM_RACE");
        }

        Job after = jobRepository.findById(command.jobId()).orElseThrow();

        List<ClaimedInputArtifact> inputs = artifactRepository
                .findByJobIdAndRole(command.jobId(), ArtifactRole.INPUT)
                .stream()
                .map(a -> new ClaimedInputArtifact(
                        a.getId().value(),
                        sourceFileIdForClaim(job, a).orElse(null),
                        a.getType(),
                        a.getStorageUri(),
                        a.getContentType(),
                        a.getSizeBytes()))
                .toList();

        log.info("Claim granted jobId={} workerToken={}", command.jobId(), command.workerClaimToken());
        return new ClaimResult(
                true,
                after.getStatus(),
                null,
                after.getCapability(),
                after.getAttemptNo(),
                inputs);
    }

    // ------------------------------------------------------------------
    // Callback
    // ------------------------------------------------------------------

    @Override
    @Transactional
    public CallbackResult handleCallback(CallbackCommand command) {
        Instant now = timeProvider.now();

        Job job = jobRepository.findById(command.jobId())
                .orElseThrow(() -> new IllegalArgumentException("Unknown jobId: " + command.jobId()));

        // Idempotency: same callbackId on the same job is a no-op replay.
        if (job.isDuplicateCallback(command.callbackId())) {
            log.info("Duplicate callback ignored jobId={} callbackId={}", job.getId(), command.callbackId());
            return new CallbackResult(false, true, job.getStatus());
        }

        // Claim-token check: a stray worker can't complete a job it never owned.
        if (job.getClaimToken() != null
                && !job.getClaimToken().equals(command.workerClaimToken())) {
            throw new IllegalStateException("Callback claim token mismatch for job " + job.getId());
        }

        if (job.getStatus().isTerminal()) {
            log.info(
                    "Stale terminal callback ignored jobId={} callbackId={} status={}",
                    job.getId(), command.callbackId(), job.getStatus());
            return new CallbackResult(false, true, job.getStatus());
        }

        if (command.outcome() == CallbackOutcome.SUCCEEDED) {
            job.markSucceeded(command.callbackId(), now);
        } else {
            job.markFailed(command.callbackId(),
                    command.errorCode(),
                    command.errorMessage(),
                    now);
        }
        jobRepository.save(job);

        List<Artifact> savedOutputs = List.of();

        // Persist output artifacts produced by the worker.
        if (command.outputArtifacts() != null && !command.outputArtifacts().isEmpty()) {
            List<Artifact> toSave = new ArrayList<>();
            for (OutputArtifactRef ref : command.outputArtifacts()) {
                toSave.add(Artifact.createNew(
                        job.getId(),
                        ArtifactRole.OUTPUT,
                        ref.type(),
                        ref.storageUri(),
                        ref.contentType(),
                        ref.sizeBytes(),
                        ref.checksumSha256(),
                        now));
            }
            savedOutputs = artifactRepository.saveAll(toSave);
        }

        if (catalogService != null && job.getCapability() == JobCapability.OCR_EXTRACT) {
            List<Artifact> inputArtifacts = artifactRepository.findByJobIdAndRole(job.getId(), ArtifactRole.INPUT);
            if (command.outcome() == CallbackOutcome.SUCCEEDED) {
                catalogService.importOcrSucceeded(job, inputArtifacts, savedOutputs, now);
            } else {
                catalogService.markOcrFailed(job, inputArtifacts, now);
            }
        }

        log.info("Callback applied jobId={} outcome={} status={}",
                job.getId(), command.outcome(), job.getStatus());
        return new CallbackResult(true, false, job.getStatus());
    }

    private Optional<String> sourceFileIdForClaim(Job job, Artifact artifact) {
        if (catalogService == null
                || job.getCapability() != JobCapability.OCR_EXTRACT
                || artifact.getType() != ArtifactType.INPUT_FILE) {
            return Optional.empty();
        }
        return catalogService.findSourceFileIdByStorageUri(artifact.getStorageUri());
    }
}
