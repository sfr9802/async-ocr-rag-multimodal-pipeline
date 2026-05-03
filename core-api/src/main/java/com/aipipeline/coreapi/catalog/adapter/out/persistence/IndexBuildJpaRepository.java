package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface IndexBuildJpaRepository extends JpaRepository<IndexBuildJpaEntity, String> {

    boolean existsByIndexVersion(String indexVersion);

    Optional<IndexBuildJpaEntity> findByIndexVersion(String indexVersion);

    Optional<IndexBuildJpaEntity> findFirstByActiveTrue();
}
