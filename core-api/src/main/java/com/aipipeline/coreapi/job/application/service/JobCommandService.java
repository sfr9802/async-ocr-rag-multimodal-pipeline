package com.aipipeline.coreapi.job.application.service;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactRepository;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactRole;
import com.aipipeline.coreapi.common.TimeProvider;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase;
import com.aipipeline.coreapi.job.application.port.out.JobDispatchPort;
import com.aipipeline.coreapi.job.application.port.out.JobRepository;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobId;
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
 * Note on ordering: we persist and transition the job to QUEUED in the same
 * transaction, but dispatch happens after the transaction commits. This way
 * a dispatch failure cannot leave a QUEUED row that the worker can't see, and
 * a successful dispatch cannot lose the DB state.
 */
@Service
public class JobCommandService implements JobManagementUseCase {

    private final JobRepository jobRepository;
    private final ArtifactRepository artifactRepository;
    private final JobDispatchPort dispatchPort;
    private final TimeProvider timeProvider;

    public JobCommandService(JobRepository jobRepository,
                             ArtifactRepository artifactRepository,
                             JobDispatchPort dispatchPort,
                             TimeProvider timeProvider) {
        this.jobRepository = jobRepository;
        this.artifactRepository = artifactRepository;
        this.dispatchPort = dispatchPort;
        this.timeProvider = timeProvider;
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
        }
        if (!inputs.isEmpty()) {
            artifactRepository.saveAll(inputs);
        }

        // dispatch fires after commit so we don't enqueue ghosts
        org.springframework.transaction.support.TransactionSynchronizationManager
                .registerSynchronization(new org.springframework.transaction.support.TransactionSynchronization() {
                    @Override
                    public void afterCommit() {
                        dispatchPort.dispatch(saved);
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
