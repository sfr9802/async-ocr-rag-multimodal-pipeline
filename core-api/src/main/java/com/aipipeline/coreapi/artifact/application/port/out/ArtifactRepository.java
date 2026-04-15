package com.aipipeline.coreapi.artifact.application.port.out;

import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactId;
import com.aipipeline.coreapi.artifact.domain.ArtifactRole;
import com.aipipeline.coreapi.job.domain.JobId;

import java.util.List;
import java.util.Optional;

public interface ArtifactRepository {

    Artifact save(Artifact artifact);

    List<Artifact> saveAll(List<Artifact> artifacts);

    Optional<Artifact> findById(ArtifactId id);

    List<Artifact> findByJobId(JobId jobId);

    List<Artifact> findByJobIdAndRole(JobId jobId, ArtifactRole role);
}
