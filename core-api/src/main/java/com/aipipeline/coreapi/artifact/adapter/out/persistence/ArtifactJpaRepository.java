package com.aipipeline.coreapi.artifact.adapter.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ArtifactJpaRepository extends JpaRepository<ArtifactJpaEntity, String> {

    List<ArtifactJpaEntity> findByJobId(String jobId);

    List<ArtifactJpaEntity> findByJobIdAndRole(String jobId, String role);
}
