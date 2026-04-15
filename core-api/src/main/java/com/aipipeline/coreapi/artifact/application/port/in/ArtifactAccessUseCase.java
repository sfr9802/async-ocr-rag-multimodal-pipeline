package com.aipipeline.coreapi.artifact.application.port.in;

import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.ArtifactId;

import java.io.InputStream;
import java.util.Optional;

/**
 * Client-facing artifact operations. The worker has its own HTTP-level
 * contract via {@link com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase};
 * this port is for end users and UI downloads.
 */
public interface ArtifactAccessUseCase {

    Optional<Artifact> findArtifact(ArtifactId id);

    /**
     * Open the content of an artifact for streaming to the caller. The
     * Artifact metadata is returned alongside so the adapter can set
     * Content-Type and similar headers without an extra lookup.
     */
    Optional<ArtifactContent> openContent(ArtifactId id);

    /** Generate an access URL for the artifact. */
    Optional<String> generateAccessUrl(ArtifactId id);

    record ArtifactContent(Artifact artifact, InputStream stream) {}
}
