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
            order by unit.createdAt desc
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
