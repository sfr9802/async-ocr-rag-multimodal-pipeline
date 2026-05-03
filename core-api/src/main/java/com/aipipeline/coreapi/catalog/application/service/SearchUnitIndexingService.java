package com.aipipeline.coreapi.catalog.application.service;

import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

@Service
public class SearchUnitIndexingService {

    public static final String EMBEDDING_STATUS_PENDING = "PENDING";
    public static final String EMBEDDING_STATUS_EMBEDDING = "EMBEDDING";
    public static final String EMBEDDING_STATUS_EMBEDDED = "EMBEDDED";
    public static final String EMBEDDING_STATUS_FAILED = "FAILED";
    public static final String EMBEDDING_STATUS_SKIPPED = "SKIPPED";

    private static final int MAX_BATCH_SIZE = 200;
    private static final Duration DEFAULT_STALE_AFTER = Duration.ofMinutes(15);
    private static final Set<String> INDEXABLE_SOURCE_STATUSES =
            Set.of(DocumentCatalogService.SOURCE_STATUS_READY);
    private static final Set<String> INDEXABLE_UNIT_TYPES = Set.of(
            DocumentCatalogService.SEARCH_UNIT_DOCUMENT,
            DocumentCatalogService.SEARCH_UNIT_PAGE,
            DocumentCatalogService.SEARCH_UNIT_SECTION,
            DocumentCatalogService.SEARCH_UNIT_TABLE,
            DocumentCatalogService.SEARCH_UNIT_IMAGE,
            DocumentCatalogService.SEARCH_UNIT_CHUNK);

    private static final Logger log = LoggerFactory.getLogger(SearchUnitIndexingService.class);

    private final SearchUnitJpaRepository searchUnits;
    private final SourceFileJpaRepository sourceFiles;
    private final ExtractedArtifactJpaRepository extractedArtifacts;
    private final ObjectMapper objectMapper;

    public SearchUnitIndexingService(SearchUnitJpaRepository searchUnits,
                                     SourceFileJpaRepository sourceFiles,
                                     ExtractedArtifactJpaRepository extractedArtifacts,
                                     ObjectMapper objectMapper) {
        this.searchUnits = searchUnits;
        this.sourceFiles = sourceFiles;
        this.extractedArtifacts = extractedArtifacts;
        this.objectMapper = objectMapper;
    }

    @Transactional
    public List<ClaimedSearchUnit> claimPending(String workerId,
                                                int batchSize,
                                                Duration staleAfter,
                                                Instant now) {
        int safeBatch = Math.max(1, Math.min(batchSize, MAX_BATCH_SIZE));
        Duration safeStaleAfter = staleAfter == null ? DEFAULT_STALE_AFTER : staleAfter;
        List<SearchUnitJpaEntity> candidates = new ArrayList<>();
        candidates.addAll(searchUnits.findIndexingCandidates(
                EMBEDDING_STATUS_PENDING,
                INDEXABLE_SOURCE_STATUSES,
                PageRequest.of(0, safeBatch * 2)));

        if (candidates.size() < safeBatch) {
            candidates.addAll(searchUnits.findStaleIndexingClaims(
                    EMBEDDING_STATUS_EMBEDDING,
                    now.minus(safeStaleAfter),
                    INDEXABLE_SOURCE_STATUSES,
                    PageRequest.of(0, safeBatch - candidates.size())));
        }

        LinkedHashMap<String, SearchUnitJpaEntity> distinct = new LinkedHashMap<>();
        for (SearchUnitJpaEntity candidate : candidates) {
            distinct.putIfAbsent(candidate.getId(), candidate);
        }

        List<SearchUnitJpaEntity> claimed = new ArrayList<>();
        for (SearchUnitJpaEntity unit : distinct.values()) {
            if (claimed.size() >= safeBatch) {
                break;
            }
            if (!EMBEDDING_STATUS_PENDING.equals(unit.getEmbeddingStatus())
                    && !EMBEDDING_STATUS_EMBEDDING.equals(unit.getEmbeddingStatus())) {
                continue;
            }
            Embeddability embeddability = embeddability(unit);
            if (!embeddability.indexable()) {
                unit.markEmbeddingSkipped(embeddability.reason(), now);
                searchUnits.save(unit);
                continue;
            }

            String token = claimToken(workerId, unit.getId());
            unit.claimEmbedding(token, now);
            claimed.add(searchUnits.save(unit));
        }

        Map<String, SourceFileJpaEntity> sourcesById = loadSources(claimed);
        Map<String, ExtractedArtifactJpaEntity> artifactsById = loadArtifacts(claimed);
        return claimed.stream()
                .map(unit -> toClaim(unit, sourcesById.get(unit.getSourceFileId()),
                        artifactsById.get(unit.getExtractedArtifactId())))
                .toList();
    }

    @Transactional
    public CompletionResult markEmbedded(String searchUnitId,
                                         String claimToken,
                                         String contentSha256,
                                         String indexId,
                                         Instant now) {
        Optional<SearchUnitJpaEntity> maybe = searchUnits.findByIdAndEmbeddingClaimToken(searchUnitId, claimToken);
        if (maybe.isEmpty()) {
            return CompletionResult.notApplied("claim token mismatch or SearchUnit not found");
        }
        SearchUnitJpaEntity unit = maybe.get();
        if (!Objects.equals(unit.getContentSha256(), contentSha256)) {
            unit.markEmbeddingPending("stale embedding result: content hash changed while indexing", now);
            searchUnits.save(unit);
            return CompletionResult.stale(unit.getEmbeddingStatusDetail());
        }

        String stableIndexId = stableIndexId(unit);
        if (indexId != null && !indexId.isBlank() && !stableIndexId.equals(indexId)) {
            log.warn(
                    "Ignoring mismatched SearchUnit index id searchUnitId={} requestedIndexId={} stableIndexId={}",
                    unit.getId(), indexId, stableIndexId);
        }
        unit.markEmbedded(stableIndexId, contentSha256, now);
        searchUnits.save(unit);
        return CompletionResult.applied(stableIndexId);
    }

    @Transactional
    public CompletionResult markFailed(String searchUnitId,
                                       String claimToken,
                                       String contentSha256,
                                       String detail,
                                       Instant now) {
        Optional<SearchUnitJpaEntity> maybe = searchUnits.findByIdAndEmbeddingClaimToken(searchUnitId, claimToken);
        if (maybe.isEmpty()) {
            return CompletionResult.notApplied("claim token mismatch or SearchUnit not found");
        }
        SearchUnitJpaEntity unit = maybe.get();
        if (!Objects.equals(unit.getContentSha256(), contentSha256)) {
            unit.markEmbeddingPending("stale failure result: content hash changed while indexing", now);
            searchUnits.save(unit);
            return CompletionResult.stale(unit.getEmbeddingStatusDetail());
        }
        unit.markEmbeddingFailed(limitDetail(detail), now);
        searchUnits.save(unit);
        return CompletionResult.applied(null);
    }

    public static String stableIndexId(SearchUnitJpaEntity unit) {
        return "source_file:" + unit.getSourceFileId()
                + ":unit:" + unit.getCanonicalUnitType()
                + ":" + unit.getUnitKey();
    }

    private ClaimedSearchUnit toClaim(SearchUnitJpaEntity unit,
                                      SourceFileJpaEntity source,
                                      ExtractedArtifactJpaEntity artifact) {
        String token = unit.getEmbeddingClaimToken();
        String artifactType = artifact == null ? null : artifact.getArtifactType();
        String sourceName = source == null ? null : source.getOriginalFileName();
        return new ClaimedSearchUnit(
                unit.getId(),
                token,
                stableIndexId(unit),
                unit.getSourceFileId(),
                sourceName,
                unit.getExtractedArtifactId(),
                artifactType,
                unit.getCanonicalUnitType(),
                unit.getUnitKey(),
                unit.getTitle(),
                unit.getSectionPath(),
                unit.getPageStart(),
                unit.getPageEnd(),
                unit.getTextContent(),
                unit.getContentSha256(),
                unit.getMetadataJson(),
                indexMetadata(unit, source, artifactType));
    }

    private Map<String, Object> indexMetadata(SearchUnitJpaEntity unit,
                                              SourceFileJpaEntity source,
                                              String artifactType) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        put(metadata, "search_unit_id", unit.getId());
        put(metadata, "source_file_id", unit.getSourceFileId());
        put(metadata, "extracted_artifact_id", unit.getExtractedArtifactId());
        put(metadata, "unit_type", unit.getCanonicalUnitType());
        put(metadata, "unit_key", unit.getUnitKey());
        put(metadata, "page_start", unit.getPageStart());
        put(metadata, "page_end", unit.getPageEnd());
        put(metadata, "section_path", unit.getSectionPath());
        put(metadata, "title", unit.getTitle());
        put(metadata, "content_hash", unit.getContentSha256());
        put(metadata, "artifact_type", artifactType);
        if (source != null) {
            put(metadata, "source_file_name", source.getOriginalFileName());
            put(metadata, "original_filename", source.getOriginalFileName());
        }
        return metadata;
    }

    private Embeddability embeddability(SearchUnitJpaEntity unit) {
        String type = unit.getCanonicalUnitType();
        if (!INDEXABLE_UNIT_TYPES.contains(type)) {
            return Embeddability.skip("unit_type is not indexable: " + type);
        }
        if (unit.getTextContent() == null || unit.getTextContent().trim().isEmpty()) {
            return Embeddability.skip("text_content is blank; SearchUnit is not embeddable");
        }
        if (unit.getContentSha256() == null || unit.getContentSha256().isBlank()) {
            return Embeddability.skip("content_sha256 is missing; SearchUnit is not embeddable");
        }
        if (!metadataAllowsIndexing(unit.getMetadataJson())) {
            return Embeddability.skip("metadata_json.indexable=false");
        }
        return Embeddability.yes();
    }

    private boolean metadataAllowsIndexing(String metadataJson) {
        if (metadataJson == null || metadataJson.isBlank()) {
            return true;
        }
        try {
            JsonNode root = objectMapper.readTree(metadataJson);
            JsonNode indexable = root.path("indexable");
            return !indexable.isBoolean() || indexable.asBoolean();
        } catch (IOException | RuntimeException ex) {
            log.warn("SearchUnit metadata_json could not be parsed for indexable flag: {}", ex.toString());
            return true;
        }
    }

    private Map<String, SourceFileJpaEntity> loadSources(List<SearchUnitJpaEntity> units) {
        LinkedHashSet<String> ids = new LinkedHashSet<>();
        for (SearchUnitJpaEntity unit : units) {
            ids.add(unit.getSourceFileId());
        }
        if (ids.isEmpty()) {
            return Map.of();
        }
        Map<String, SourceFileJpaEntity> byId = new LinkedHashMap<>();
        for (SourceFileJpaEntity source : sourceFiles.findAllById(ids)) {
            byId.put(source.getId(), source);
        }
        return byId;
    }

    private Map<String, ExtractedArtifactJpaEntity> loadArtifacts(List<SearchUnitJpaEntity> units) {
        LinkedHashSet<String> ids = new LinkedHashSet<>();
        for (SearchUnitJpaEntity unit : units) {
            if (unit.getExtractedArtifactId() != null) {
                ids.add(unit.getExtractedArtifactId());
            }
        }
        if (ids.isEmpty()) {
            return Map.of();
        }
        Map<String, ExtractedArtifactJpaEntity> byId = new LinkedHashMap<>();
        for (ExtractedArtifactJpaEntity artifact : extractedArtifacts.findAllById(ids)) {
            byId.put(artifact.getArtifactId(), artifact);
        }
        return byId;
    }

    private static void put(Map<String, Object> metadata, String key, Object value) {
        if (value != null) {
            metadata.put(key, value);
        }
    }

    private static String claimToken(String workerId, String searchUnitId) {
        String normalizedWorker = workerId == null || workerId.isBlank() ? "worker" : workerId.trim();
        return normalizedWorker + ":" + searchUnitId + ":" + UUID.randomUUID();
    }

    private static String limitDetail(String detail) {
        String normalized = detail == null || detail.isBlank() ? "SearchUnit indexing failed" : detail.trim();
        return normalized.length() <= 2000 ? normalized : normalized.substring(0, 2000);
    }

    private record Embeddability(boolean indexable, String reason) {
        static Embeddability yes() {
            return new Embeddability(true, null);
        }

        static Embeddability skip(String reason) {
            return new Embeddability(false, reason);
        }
    }

    public record ClaimedSearchUnit(
            String searchUnitId,
            String claimToken,
            String indexId,
            String sourceFileId,
            String sourceFileName,
            String extractedArtifactId,
            String artifactType,
            String unitType,
            String unitKey,
            String title,
            String sectionPath,
            Integer pageStart,
            Integer pageEnd,
            String textContent,
            String contentSha256,
            String metadataJson,
            Map<String, Object> indexMetadata
    ) {}

    public record CompletionResult(
            boolean applied,
            boolean stale,
            String indexId,
            String detail
    ) {
        static CompletionResult applied(String indexId) {
            return new CompletionResult(true, false, indexId, null);
        }

        static CompletionResult stale(String detail) {
            return new CompletionResult(false, true, null, detail);
        }

        static CompletionResult notApplied(String detail) {
            return new CompletionResult(false, false, null, detail);
        }
    }
}
