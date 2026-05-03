package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import jakarta.persistence.LockModeType;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.Set;

public interface SearchUnitJpaRepository extends JpaRepository<SearchUnitJpaEntity, String> {

    boolean existsByExtractedArtifactId(String extractedArtifactId);

    Optional<SearchUnitJpaEntity> findBySourceFileIdAndUnitTypeAndUnitKey(
            String sourceFileId,
            String unitType,
            String unitKey);

    @Query("""
            select unit
            from SearchUnitJpaEntity unit
            where lower(coalesce(unit.textContent, '')) like lower(concat('%', :query, '%'))
               or lower(coalesce(unit.bm25Text, '')) like lower(concat('%', :query, '%'))
               or lower(coalesce(unit.displayText, '')) like lower(concat('%', :query, '%'))
               or lower(coalesce(unit.citationText, '')) like lower(concat('%', :query, '%'))
               or lower(coalesce(unit.debugText, '')) like lower(concat('%', :query, '%'))
            order by
              case
                when lower(coalesce(unit.bm25Text, '')) like lower(concat(:query, '%')) then 0
                when lower(coalesce(unit.displayText, '')) like lower(concat(:query, '%')) then 1
                when lower(coalesce(unit.textContent, '')) like lower(concat(:query, '%')) then 2
                else 3
              end,
              case lower(coalesce(unit.chunkType, unit.unitType, ''))
                when 'row_group' then 0
                when 'paragraph' then 1
                when 'table' then 2
                when 'page' then 3
                when 'sheet_summary' then 4
                when 'document_summary' then 5
                when 'workbook_summary' then 6
                else 7
              end,
              unit.createdAt desc
            """)
    List<SearchUnitJpaEntity> searchByText(@Param("query") String query, Pageable pageable);

    Optional<SearchUnitJpaEntity> findByIdAndEmbeddingClaimToken(String id, String embeddingClaimToken);

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("""
            select unit
            from SearchUnitJpaEntity unit
            where unit.embeddingStatus = :embeddingStatus
              and exists (
                select source.id
                from SourceFileJpaEntity source
                where source.id = unit.sourceFileId
                  and source.status in :sourceStatuses
              )
            order by unit.updatedAt asc
            """)
    List<SearchUnitJpaEntity> findIndexingCandidates(
            @Param("embeddingStatus") String embeddingStatus,
            @Param("sourceStatuses") Set<String> sourceStatuses,
            Pageable pageable);

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("""
            select unit
            from SearchUnitJpaEntity unit
            where unit.embeddingStatus = :embeddingStatus
              and unit.embeddingClaimedAt is not null
              and unit.embeddingClaimedAt < :claimedBefore
              and exists (
                select source.id
                from SourceFileJpaEntity source
                where source.id = unit.sourceFileId
                  and source.status in :sourceStatuses
              )
            order by unit.embeddingClaimedAt asc
            """)
    List<SearchUnitJpaEntity> findStaleIndexingClaims(
            @Param("embeddingStatus") String embeddingStatus,
            @Param("claimedBefore") Instant claimedBefore,
            @Param("sourceStatuses") Set<String> sourceStatuses,
            Pageable pageable);
}
