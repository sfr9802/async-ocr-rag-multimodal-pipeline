package com.aipipeline.coreapi.job.application.port.in;

import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.aipipeline.coreapi.job.domain.JobId;

import java.util.List;
import java.util.Optional;

/**
 * Client-facing (non-worker) operations on jobs. Controllers depend on this
 * port; they are not allowed to reach into the domain or persistence layers
 * directly.
 */
public interface JobManagementUseCase {

    /**
     * Create a new job plus any attached input artifacts, and enqueue it.
     * The enqueue happens transactionally via the outbound dispatch port.
     */
    JobCreationResult createAndEnqueue(CreateJobCommand command);

    Optional<Job> findJob(JobId id);

    /** Convenience: load the job together with all its artifacts. */
    Optional<JobWithArtifacts> findJobWithArtifacts(JobId id);

    // ---- value types carried across the port boundary ----

    /**
     * Command object for creating a new job. Input artifacts are passed as
     * already-staged records — the controller is expected to have persisted
     * any raw bytes through the storage port first.
     */
    record CreateJobCommand(
            JobCapability capability,
            List<StagedInputArtifact> inputs
    ) {}

    /**
     * An input artifact whose bytes are already stored somewhere the storage
     * adapter can reach. The core API owns the metadata side; the physical
     * bytes stay opaque behind {@code storageUri}.
     */
    record StagedInputArtifact(
            com.aipipeline.coreapi.artifact.domain.ArtifactType type,
            String storageUri,
            String contentType,
            Long sizeBytes,
            String checksumSha256,
            String originalFileName
    ) {}

    record JobCreationResult(Job job, List<Artifact> inputArtifacts) {}

    record JobWithArtifacts(Job job, List<Artifact> artifacts) {}
}
