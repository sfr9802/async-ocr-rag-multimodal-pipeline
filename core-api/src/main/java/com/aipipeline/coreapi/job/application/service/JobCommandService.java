package com.aipipeline.coreapi.job.application.service;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactRepository;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactRole;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.catalog.application.service.DocumentCatalogService;
import com.aipipeline.coreapi.common.TimeProvider;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase;
import com.aipipeline.coreapi.job.application.port.out.JobDispatchPort;
import com.aipipeline.coreapi.job.application.port.out.JobRepository;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.aipipeline.coreapi.job.domain.JobId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Application service implementing the client-facing job operations.
 *
 * Responsibilities:
 *   - Create the job aggregate
 *   - Persist input artifacts that the web layer has already staged into storage
 *   - Mark the job as QUEUED and hand it to the dispatch port
 *   - Expose read operations for status and result
 *
 * Note on ordering: dispatch happens after commit so workers never consume a
 * database ghost. If Redis is down after the commit, the dispatch reconciler
 * re-enqueues the durable QUEUED row on a later scan.
 */
@Service
public class JobCommandService implements JobManagementUseCase {

    private static final Logger log = LoggerFactory.getLogger(JobCommandService.class);

    private final JobRepository jobRepository;
    private final ArtifactRepository artifactRepository;
    private final JobDispatchPort dispatchPort;
    private final TimeProvider timeProvider;
    private final DocumentCatalogService catalogService;

    public JobCommandService(JobRepository jobRepository,
                             ArtifactRepository artifactRepository,
                             JobDispatchPort dispatchPort,
                             TimeProvider timeProvider,
                             DocumentCatalogService catalogService) {
        this.jobRepository = jobRepository;
        this.artifactRepository = artifactRepository;
        this.dispatchPort = dispatchPort;
        this.timeProvider = timeProvider;
        this.catalogService = catalogService;
    }

    @Override
    @Transactional
    public JobCreationResult createAndEnqueue(CreateJobCommand command) {
        Instant now = timeProvider.now();

        Job job = Job.createNew(command.capability(), now);
        job.markQueued(now);
        Job saved = jobRepository.save(job);

        List<Artifact> inputs = new ArrayList<>();
        for (StagedInputArtifact staged : command.inputs()) {
            Artifact artifact = Artifact.createNew(
                    saved.getId(),
                    ArtifactRole.INPUT,
                    staged.type(),
                    staged.storageUri(),
                    staged.contentType(),
                    staged.sizeBytes(),
                    staged.checksumSha256(),
                    now);
            inputs.add(artifact);
            if (saved.getCapability() == JobCapability.OCR_EXTRACT
                    && staged.type() == ArtifactType.INPUT_FILE) {
                catalogService.registerProcessingSourceFile(
                        staged.originalFileName(),
                        staged.contentType(),
                        staged.storageUri(),
                        now);
            }
        }
        if (!inputs.isEmpty()) {
            artifactRepository.saveAll(inputs);
        }

        // dispatch fires after commit so we don't enqueue ghosts
        org.springframework.transaction.support.TransactionSynchronizationManager
                .registerSynchronization(new org.springframework.transaction.support.TransactionSynchronization() {
                    @Override
                    public void afterCommit() {
                        try {
                            dispatchPort.dispatch(saved);
                        } catch (JobDispatchPort.DispatchException ex) {
                            log.warn(
                                    "After-commit dispatch failed for job {}; "
                                    + "dispatch reconciler will retry",
                                    saved.getId(), ex);
                        }
                    }
                });

        return new JobCreationResult(saved, List.copyOf(inputs));
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<Job> findJob(JobId id) {
        return jobRepository.findById(id);
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<JobWithArtifacts> findJobWithArtifacts(JobId id) {
        return jobRepository.findById(id)
                .map(job -> new JobWithArtifacts(job, artifactRepository.findByJobId(id)));
    }
}
