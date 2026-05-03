package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

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
        this.embeddingStatus = "EMBEDDED";
        this.indexId = indexId;
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
