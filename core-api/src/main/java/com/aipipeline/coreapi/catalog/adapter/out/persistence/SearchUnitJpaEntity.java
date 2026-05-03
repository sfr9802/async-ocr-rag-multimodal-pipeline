package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

@Entity
@Table(name = "search_unit")
public class SearchUnitJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "source_file_id", nullable = false, length = 64)
    private String sourceFileId;

    @Column(name = "extracted_artifact_id", length = 64)
    private String extractedArtifactId;

    @Column(name = "unit_type", nullable = false, length = 64)
    private String unitType;

    @Column(name = "unit_key", nullable = false, length = 256)
    private String unitKey;

    @Column(name = "title", length = 512)
    private String title;

    @Column(name = "section_path", length = 1024)
    private String sectionPath;

    @Column(name = "page_start")
    private Integer pageStart;

    @Column(name = "page_end")
    private Integer pageEnd;

    @Column(name = "text_content", columnDefinition = "TEXT")
    private String textContent;

    @Column(name = "metadata_json", columnDefinition = "TEXT")
    private String metadataJson;

    @Column(name = "embedding_status", nullable = false, length = 32)
    private String embeddingStatus;

    @Column(name = "content_sha256", length = 128)
    private String contentSha256;

    @Column(name = "index_id", length = 512)
    private String indexId;

    @Column(name = "indexed_content_sha256", length = 128)
    private String indexedContentSha256;

    @Column(name = "embedding_claim_token", length = 128)
    private String embeddingClaimToken;

    @Column(name = "embedding_claimed_at")
    private Instant embeddingClaimedAt;

    @Column(name = "embedding_status_detail", columnDefinition = "TEXT")
    private String embeddingStatusDetail;

    @Column(name = "embedded_at")
    private Instant embeddedAt;

    @Column(name = "document_id", length = 64)
    private String documentId;

    @Column(name = "document_version_id", length = 64)
    private String documentVersionId;

    @Column(name = "parsed_artifact_id", length = 64)
    private String parsedArtifactId;

    @Column(name = "source_file_name", length = 512)
    private String sourceFileName;

    @Column(name = "source_file_type", length = 32)
    private String sourceFileType;

    @Column(name = "chunk_type", length = 64)
    private String chunkType;

    @Column(name = "location_type", length = 32)
    private String locationType;

    @Column(name = "location_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String locationJson;

    @Column(name = "embedding_text", columnDefinition = "TEXT")
    private String embeddingText;

    @Column(name = "bm25_text", columnDefinition = "TEXT")
    private String bm25Text;

    @Column(name = "display_text", columnDefinition = "TEXT")
    private String displayText;

    @Column(name = "citation_text", columnDefinition = "TEXT")
    private String citationText;

    @Column(name = "debug_text", columnDefinition = "TEXT")
    private String debugText;

    @Column(name = "parser_name", length = 128)
    private String parserName;

    @Column(name = "parser_version", length = 128)
    private String parserVersion;

    @Column(name = "index_version", length = 128)
    private String indexVersion;

    @Column(name = "quality_score")
    private Double qualityScore;

    @Column(name = "confidence_score")
    private Double confidenceScore;

    @Column(name = "acl_tags", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String aclTags = "[]";

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    protected SearchUnitJpaEntity() {}

    public SearchUnitJpaEntity(String id,
                               String sourceFileId,
                               String artifactId,
                               String unitType,
                               String text,
                               String metadataJson,
                               String embeddingStatus,
                               Instant createdAt) {
        this(
                id,
                sourceFileId,
                artifactId,
                unitType,
                unitType + ":" + Integer.toHexString((text == null ? "" : text).hashCode()),
                null,
                null,
                null,
                null,
                text,
                metadataJson,
                embeddingStatus,
                null,
                createdAt,
                createdAt);
    }

    public SearchUnitJpaEntity(String id,
                               String sourceFileId,
                               String extractedArtifactId,
                               String unitType,
                               String unitKey,
                               String title,
                               String sectionPath,
                               Integer pageStart,
                               Integer pageEnd,
                               String textContent,
                               String metadataJson,
                               String embeddingStatus,
                               String contentSha256,
                               Instant createdAt,
                               Instant updatedAt) {
        this.id = id;
        this.sourceFileId = sourceFileId;
        this.extractedArtifactId = extractedArtifactId;
        this.unitType = unitType;
        this.unitKey = unitKey;
        this.title = title;
        this.sectionPath = sectionPath;
        this.pageStart = pageStart;
        this.pageEnd = pageEnd;
        this.textContent = textContent;
        this.metadataJson = metadataJson;
        this.embeddingStatus = embeddingStatus;
        this.contentSha256 = contentSha256;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt == null ? createdAt : updatedAt;
    }

    public String getId() { return id; }
    public String getSourceFileId() { return sourceFileId; }
    public String getExtractedArtifactId() { return extractedArtifactId; }
    public String getArtifactId() { return extractedArtifactId; }
    public String getUnitType() { return unitType; }
    public String getCanonicalUnitType() {
        return "ocr_page".equalsIgnoreCase(unitType) ? "PAGE" : unitType;
    }
    public String getUnitKey() { return unitKey; }
    public String getTitle() { return title; }
    public String getSectionPath() { return sectionPath; }
    public Integer getPageStart() { return pageStart; }
    public Integer getPageEnd() { return pageEnd; }
    public String getTextContent() { return textContent; }
    public String getText() { return textContent; }
    public String getMetadataJson() { return metadataJson; }
    public String getEmbeddingStatus() { return embeddingStatus; }
    public String getContentSha256() { return contentSha256; }
    public String getIndexId() { return indexId; }
    public String getIndexedContentSha256() { return indexedContentSha256; }
    public String getEmbeddingClaimToken() { return embeddingClaimToken; }
    public Instant getEmbeddingClaimedAt() { return embeddingClaimedAt; }
    public String getEmbeddingStatusDetail() { return embeddingStatusDetail; }
    public Instant getEmbeddedAt() { return embeddedAt; }
    public String getDocumentId() { return documentId; }
    public String getDocumentVersionId() { return documentVersionId; }
    public String getParsedArtifactId() { return parsedArtifactId; }
    public String getSourceFileName() { return sourceFileName; }
    public String getSourceFileType() { return sourceFileType; }
    public String getChunkType() { return chunkType; }
    public String getLocationType() { return locationType; }
    public String getLocationJson() { return locationJson; }
    public String getEmbeddingText() { return embeddingText; }
    public String getBm25Text() { return bm25Text; }
    public String getDisplayText() { return displayText; }
    public String getCitationText() { return citationText; }
    public String getDebugText() { return debugText; }
    public String getParserName() { return parserName; }
    public String getParserVersion() { return parserVersion; }
    public String getIndexVersion() { return indexVersion; }
    public Double getQualityScore() { return qualityScore; }
    public Double getConfidenceScore() { return confidenceScore; }
    public String getAclTags() { return aclTags; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }

    public void updateCanonical(String extractedArtifactId,
                                String title,
                                String sectionPath,
                                Integer pageStart,
                                Integer pageEnd,
                                String textContent,
                                String metadataJson,
                                String contentSha256,
                                String pendingEmbeddingStatus,
                                Instant now) {
        this.extractedArtifactId = extractedArtifactId;
        this.title = title;
        this.sectionPath = sectionPath;
        this.pageStart = pageStart;
        this.pageEnd = pageEnd;
        this.metadataJson = metadataJson;
        if (!java.util.Objects.equals(this.contentSha256, contentSha256)) {
            this.textContent = textContent;
            this.contentSha256 = contentSha256;
            this.embeddingStatus = pendingEmbeddingStatus;
            this.indexedContentSha256 = null;
            this.embeddingClaimToken = null;
            this.embeddingClaimedAt = null;
            this.embeddingStatusDetail = null;
            this.embeddedAt = null;
        }
        this.updatedAt = now;
    }

    public void applyIngestionV2(String documentId,
                                 String documentVersionId,
                                 String parsedArtifactId,
                                 String sourceFileName,
                                 String sourceFileType,
                                 String chunkType,
                                 String locationType,
                                 String locationJson,
                                 String embeddingText,
                                 String bm25Text,
                                 String displayText,
                                 String citationText,
                                 String debugText,
                                 String parserName,
                                 String parserVersion,
                                 Double qualityScore,
                                 Double confidenceScore,
                                 String aclTags,
                                 Instant now) {
        this.documentId = documentId;
        this.documentVersionId = documentVersionId;
        this.parsedArtifactId = parsedArtifactId;
        this.sourceFileName = sourceFileName;
        this.sourceFileType = sourceFileType;
        this.chunkType = chunkType;
        this.locationType = locationType;
        this.locationJson = locationJson;
        this.embeddingText = embeddingText;
        this.bm25Text = bm25Text;
        this.displayText = displayText;
        this.citationText = citationText;
        this.debugText = debugText;
        this.parserName = parserName;
        this.parserVersion = parserVersion;
        this.qualityScore = qualityScore;
        this.confidenceScore = confidenceScore;
        this.aclTags = aclTags == null || aclTags.isBlank() ? "[]" : aclTags;
        this.updatedAt = now;
    }

    public void claimEmbedding(String claimToken, Instant now) {
        this.embeddingStatus = "EMBEDDING";
        this.embeddingClaimToken = claimToken;
        this.embeddingClaimedAt = now;
        this.embeddingStatusDetail = null;
        this.updatedAt = now;
    }

    public void markEmbedded(String indexId,
                             String indexedContentSha256,
                             Instant now) {
        markEmbedded(indexId, null, indexedContentSha256, now);
    }

    public void markEmbedded(String indexId,
                             String indexVersion,
                             String indexedContentSha256,
                             Instant now) {
        this.embeddingStatus = "EMBEDDED";
        this.indexId = indexId;
        this.indexVersion = indexVersion;
        this.indexedContentSha256 = indexedContentSha256;
        this.embeddingClaimToken = null;
        this.embeddingClaimedAt = null;
        this.embeddingStatusDetail = null;
        this.embeddedAt = now;
        this.updatedAt = now;
    }

    public void markEmbeddingFailed(String detail, Instant now) {
        this.embeddingStatus = "FAILED";
        this.embeddingClaimToken = null;
        this.embeddingClaimedAt = null;
        this.embeddingStatusDetail = detail;
        this.updatedAt = now;
    }

    public void markEmbeddingSkipped(String detail, Instant now) {
        this.embeddingStatus = "SKIPPED";
        this.embeddingClaimToken = null;
        this.embeddingClaimedAt = null;
        this.embeddingStatusDetail = detail;
        this.updatedAt = now;
    }

    public void markEmbeddingPending(String detail, Instant now) {
        this.embeddingStatus = "PENDING";
        this.embeddingClaimToken = null;
        this.embeddingClaimedAt = null;
        this.embeddingStatusDetail = detail;
        this.updatedAt = now;
    }
}
