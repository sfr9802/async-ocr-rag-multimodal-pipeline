package com.aipipeline.coreapi.catalog.application.service;

import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.EmbeddingRecordJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.EmbeddingRecordJpaRepository;
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
            DocumentCatalogService.SEARCH_UNIT_CHUNK);

    private static final Logger log = LoggerFactory.getLogger(SearchUnitIndexingService.class);

    private final SearchUnitJpaRepository searchUnits;
    private final SourceFileJpaRepository sourceFiles;
    private final ExtractedArtifactJpaRepository extractedArtifacts;
    private final EmbeddingRecordJpaRepository embeddingRecords;
    private final ObjectMapper objectMapper;

    public SearchUnitIndexingService(SearchUnitJpaRepository searchUnits,
                                     SourceFileJpaRepository sourceFiles,
                                     ExtractedArtifactJpaRepository extractedArtifacts,
                                     EmbeddingRecordJpaRepository embeddingRecords,
                                     ObjectMapper objectMapper) {
        this.searchUnits = searchUnits;
        this.sourceFiles = sourceFiles;
        this.extractedArtifacts = extractedArtifacts;
        this.embeddingRecords = embeddingRecords;
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
                                         String indexVersion,
                                         String embeddingModel,
                                         String embeddingTextSha256,
                                         String vectorId,
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
        unit.markEmbedded(stableIndexId, indexVersion, contentSha256, now);
        searchUnits.save(unit);
        upsertEmbeddingRecord(unit, indexVersion, embeddingModel, embeddingTextSha256, vectorId, now);
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
                firstNonBlank(unit.getEmbeddingText(), unit.getTextContent()),
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
        put(metadata, "content_sha256", unit.getContentSha256());
        put(metadata, "artifact_type", artifactType);
        put(metadata, "index_id", stableIndexId(unit));
        put(metadata, "document_id", unit.getDocumentId());
        put(metadata, "documentId", unit.getDocumentId());
        put(metadata, "document_version_id", unit.getDocumentVersionId());
        put(metadata, "documentVersionId", unit.getDocumentVersionId());
        put(metadata, "parsed_artifact_id", unit.getParsedArtifactId());
        put(metadata, "parsedArtifactId", unit.getParsedArtifactId());
        put(metadata, "source_file_type", unit.getSourceFileType());
        put(metadata, "sourceFileType", unit.getSourceFileType());
        put(metadata, "chunk_type", unit.getChunkType());
        put(metadata, "chunkType", unit.getChunkType());
        put(metadata, "location_type", unit.getLocationType());
        put(metadata, "locationType", unit.getLocationType());
        putJsonIfPresent(metadata, "location_json", unit.getLocationJson());
        putJsonIfPresent(metadata, "locationJson", unit.getLocationJson());
        put(metadata, "citation_text", unit.getCitationText());
        put(metadata, "citationText", unit.getCitationText());
        put(metadata, "display_text", unit.getDisplayText());
        put(metadata, "displayText", unit.getDisplayText());
        put(metadata, "debug_text", unit.getDebugText());
        put(metadata, "debugText", unit.getDebugText());
        put(metadata, "parser_name", unit.getParserName());
        put(metadata, "parserName", unit.getParserName());
        put(metadata, "parser_version", unit.getParserVersion());
        put(metadata, "parserVersion", unit.getParserVersion());
        put(metadata, "quality_score", unit.getQualityScore());
        put(metadata, "qualityScore", unit.getQualityScore());
        put(metadata, "confidence_score", unit.getConfidenceScore());
        put(metadata, "confidenceScore", unit.getConfidenceScore());
        JsonNode unitMetadata = parseMetadata(unit.getMetadataJson());
        String unitFileType = textOrNull(unitMetadata, "fileType");
        put(metadata, "fileType", unitFileType == null && source != null ? source.getFileType() : unitFileType);
        put(metadata, "sheetName", textOrNull(unitMetadata, "sheetName"));
        put(metadata, "sheetIndex", intOrNull(unitMetadata, "sheetIndex"));
        put(metadata, "cellRange", firstText(unitMetadata, "cellRange", "range", "usedRange"));
        put(metadata, "range", firstText(unitMetadata, "range", "cellRange", "usedRange"));
        put(metadata, "tableId", firstText(unitMetadata, "tableId", "tableName"));
        put(metadata, "rowStart", intOrNull(unitMetadata, "rowStart"));
        put(metadata, "rowEnd", intOrNull(unitMetadata, "rowEnd"));
        put(metadata, "columnStart", intOrNull(unitMetadata, "columnStart"));
        put(metadata, "columnEnd", intOrNull(unitMetadata, "columnEnd"));
        if (source != null) {
            put(metadata, "source_file_name", source.getOriginalFileName());
            put(metadata, "sourceFileName", source.getOriginalFileName());
            put(metadata, "original_filename", source.getOriginalFileName());
        }
        return metadata;
    }

    private Embeddability embeddability(SearchUnitJpaEntity unit) {
        String type = unit.getCanonicalUnitType();
        String indexableText = firstNonBlank(unit.getEmbeddingText(), unit.getTextContent());
        if (indexableText == null || indexableText.trim().isEmpty()) {
            return Embeddability.skip("embedding_text/text_content is blank; SearchUnit is not embeddable");
        }
        JsonNode unitMetadata = parseMetadata(unit.getMetadataJson());
        if (!INDEXABLE_UNIT_TYPES.contains(type)) {
            return Embeddability.skip("unit_type is not indexable: " + type);
        }
        if (DocumentCatalogService.SEARCH_UNIT_DOCUMENT.equals(type) && !isSpreadsheetMetadata(unitMetadata)) {
            return Embeddability.skip("DOCUMENT SearchUnit is not embedded directly; use PAGE/SECTION/TABLE/CHUNK");
        }
        if (unit.getContentSha256() == null || unit.getContentSha256().isBlank()) {
            return Embeddability.skip("content_sha256 is missing; SearchUnit is not embeddable");
        }
        if (!metadataAllowsIndexing(unitMetadata)) {
            return Embeddability.skip("metadata_json.indexable=false");
        }
        if (requiresV2CitationGate(unit, unitMetadata)) {
            if (unit.getParserVersion() == null || unit.getParserVersion().isBlank()) {
                return Embeddability.skip("parser_version is required for v2 SearchUnit indexing");
            }
            if (unit.getLocationJson() == null || unit.getLocationJson().isBlank()) {
                return Embeddability.skip("location_json is required for v2 SearchUnit indexing");
            }
            if (unit.getCitationText() == null || unit.getCitationText().isBlank()) {
                return Embeddability.skip("citation_text is required for v2 SearchUnit indexing");
            }
            String invalidLocationReason = invalidV2LocationReason(unit, unitMetadata);
            if (invalidLocationReason != null) {
                return Embeddability.skip(invalidLocationReason);
            }
        }
        return Embeddability.yes();
    }

    private void upsertEmbeddingRecord(SearchUnitJpaEntity unit,
                                       String indexVersion,
                                       String embeddingModel,
                                       String embeddingTextSha256,
                                       String vectorId,
                                       Instant now) {
        if (indexVersion == null || indexVersion.isBlank()) {
            return;
        }
        String model = firstNonBlank(embeddingModel, "unknown");
        String textHash = firstNonBlank(embeddingTextSha256, unit.getContentSha256());
        if (textHash == null || textHash.isBlank()) {
            return;
        }
        EmbeddingRecordJpaEntity record = embeddingRecords
                .findBySearchUnitIdAndIndexVersionAndEmbeddingModel(unit.getId(), indexVersion, model)
                .orElseGet(() -> new EmbeddingRecordJpaEntity(UUID.randomUUID().toString()));
        record.refresh(
                unit.getId(),
                indexVersion,
                model,
                textHash,
                firstNonBlank(vectorId, stableIndexId(unit)),
                now);
        embeddingRecords.save(record);
    }

    private String invalidV2LocationReason(SearchUnitJpaEntity unit, JsonNode unitMetadata) {
        JsonNode location = parseMetadata(unit.getLocationJson());
        String locationType = firstNonBlank(
                textOrNull(location, "type"),
                firstNonBlank(unit.getLocationType(), textOrNull(unitMetadata, "fileType")));
        String normalizedType = locationType == null ? "" : locationType.trim().toLowerCase();
        String chunkType = unit.getChunkType() == null ? "" : unit.getChunkType().trim().toLowerCase();
        if ("xlsx".equals(normalizedType) || "spreadsheet".equals(normalizedType)) {
            if (!"workbook_summary".equals(chunkType) && isBlank(textOrNull(location, "sheet_name"))) {
                return "xlsx location_json.sheet_name is required for v2 SearchUnit indexing";
            }
            if (("table".equals(chunkType) || "row_group".equals(chunkType))
                    && isBlank(textOrNull(location, "cell_range"))) {
                return "xlsx location_json.cell_range is required for table/row_group indexing";
            }
        }
        if ("pdf".equals(normalizedType) || "ocr".equals(normalizedType)) {
            boolean hasPage = location.hasNonNull("page_no")
                    || location.hasNonNull("page_label")
                    || location.hasNonNull("physical_page_index");
            if (!hasPage) {
                return "pdf location_json page identifier is required for v2 SearchUnit indexing";
            }
            if ("paragraph".equals(chunkType) && !location.hasNonNull("bbox")) {
                return "pdf paragraph location_json.bbox is required for v2 SearchUnit indexing";
            }
            if (location.path("ocr_used").asBoolean(false) && !location.hasNonNull("ocr_confidence")) {
                return "ocr location_json.ocr_confidence is required for lower-trust indexing";
            }
        }
        return null;
    }

    private boolean requiresV2CitationGate(SearchUnitJpaEntity unit, JsonNode metadata) {
        if (unit.getParserVersion() != null && !unit.getParserVersion().isBlank()) {
            return true;
        }
        if (unit.getLocationJson() != null && !unit.getLocationJson().isBlank()) {
            return true;
        }
        String sourceFileType = unit.getSourceFileType();
        if (sourceFileType != null) {
            String normalized = sourceFileType.trim().toUpperCase();
            if ("SPREADSHEET".equals(normalized) || "PDF".equals(normalized)) {
                return true;
            }
        }
        String fileType = textOrNull(metadata, "fileType");
        if (fileType == null) {
            return false;
        }
        String normalized = fileType.trim().toLowerCase();
        return "xlsx".equals(normalized) || "xlsm".equals(normalized) || "pdf".equals(normalized);
    }

    private boolean metadataAllowsIndexing(JsonNode root) {
        if (root.isMissingNode()) {
            return true;
        }
        JsonNode indexable = root.path("indexable");
        return !indexable.isBoolean() || indexable.asBoolean();
    }

    private static boolean isSpreadsheetMetadata(JsonNode root) {
        String fileType = textOrNull(root, "fileType");
        if (fileType == null) {
            return false;
        }
        String normalized = fileType.trim().toLowerCase();
        return "xlsx".equals(normalized)
                || "xlsm".equals(normalized)
                || "spreadsheet".equals(normalized);
    }

    private JsonNode parseMetadata(String metadataJson) {
        if (metadataJson == null || metadataJson.isBlank()) {
            return objectMapper.getNodeFactory().missingNode();
        }
        try {
            return objectMapper.readTree(metadataJson);
        } catch (IOException | RuntimeException ex) {
            log.warn("SearchUnit metadata_json could not be parsed: {}", ex.toString());
            return objectMapper.getNodeFactory().missingNode();
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

    private void putJsonIfPresent(Map<String, Object> metadata, String key, String json) {
        if (json == null || json.isBlank()) {
            return;
        }
        JsonNode parsed = parseMetadata(json);
        if (!parsed.isMissingNode() && !parsed.isNull()) {
            metadata.put(key, parsed);
        }
    }

    private static String firstText(JsonNode node, String... fields) {
        for (String field : fields) {
            String value = textOrNull(node, field);
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }

    private static String textOrNull(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode value = node.path(field);
        return value.isMissingNode() || value.isNull() ? null : value.asText();
    }

    private static String firstNonBlank(String first, String second) {
        return first != null && !first.isBlank() ? first : second;
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private static Integer intOrNull(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode value = node.path(field);
        if (value.isIntegralNumber()) {
            return value.asInt();
        }
        if (value.isTextual()) {
            try {
                return Integer.parseInt(value.asText().trim());
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
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
