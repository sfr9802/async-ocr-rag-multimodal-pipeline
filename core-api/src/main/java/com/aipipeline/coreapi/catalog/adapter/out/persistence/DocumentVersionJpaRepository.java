package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface DocumentVersionJpaRepository extends JpaRepository<DocumentVersionJpaEntity, String> {
    Optional<DocumentVersionJpaEntity> findFirstBySourceFileIdOrderByVersionNoDesc(String sourceFileId);
}
