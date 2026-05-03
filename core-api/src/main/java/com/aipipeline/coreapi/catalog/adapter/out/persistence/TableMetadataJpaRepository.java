package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

public interface TableMetadataJpaRepository extends JpaRepository<TableMetadataJpaEntity, String> {
}
