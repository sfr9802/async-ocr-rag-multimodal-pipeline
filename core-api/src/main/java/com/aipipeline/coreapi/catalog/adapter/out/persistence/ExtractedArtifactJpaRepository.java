package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ExtractedArtifactJpaRepository extends JpaRepository<ExtractedArtifactJpaEntity, String> {

    Optional<ExtractedArtifactJpaEntity> findBySourceFileIdAndArtifactTypeAndArtifactKey(
            String sourceFileId,
            String artifactType,
            String artifactKey);
}
