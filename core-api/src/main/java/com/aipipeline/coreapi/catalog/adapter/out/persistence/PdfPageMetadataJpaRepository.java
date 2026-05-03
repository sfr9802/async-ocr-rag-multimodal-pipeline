package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface PdfPageMetadataJpaRepository extends JpaRepository<PdfPageMetadataJpaEntity, String> {
    Optional<PdfPageMetadataJpaEntity> findByDocumentVersionIdAndPhysicalPageIndex(
            String documentVersionId,
            Integer physicalPageIndex);
}
