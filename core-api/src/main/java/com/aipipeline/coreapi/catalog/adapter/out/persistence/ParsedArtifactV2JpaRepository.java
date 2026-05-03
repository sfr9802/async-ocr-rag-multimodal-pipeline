package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ParsedArtifactV2JpaRepository extends JpaRepository<ParsedArtifactV2JpaEntity, String> {
    Optional<ParsedArtifactV2JpaEntity> findFirstByExtractedArtifactId(String extractedArtifactId);
}
