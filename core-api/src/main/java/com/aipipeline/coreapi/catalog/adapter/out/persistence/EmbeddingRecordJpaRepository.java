package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface EmbeddingRecordJpaRepository extends JpaRepository<EmbeddingRecordJpaEntity, String> {
    Optional<EmbeddingRecordJpaEntity> findBySearchUnitIdAndIndexVersionAndEmbeddingModel(
            String searchUnitId,
            String indexVersion,
            String embeddingModel);
}
