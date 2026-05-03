package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface SourceFileJpaRepository extends JpaRepository<SourceFileJpaEntity, String> {

    Optional<SourceFileJpaEntity> findFirstByStorageUri(String storageUri);
}
