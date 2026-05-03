package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

@Entity
@Table(name = "document")
public class DocumentV2JpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "title", nullable = false, length = 512)
    private String title;

    @Column(name = "status", nullable = false, length = 32)
    private String status;

    @Column(name = "acl_tags", columnDefinition = "jsonb", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private String aclTags = "[]";

    @Column(name = "latest_version_id", length = 64)
    private String latestVersionId;

    @Column(name = "metadata_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String metadataJson;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    protected DocumentV2JpaEntity() {}

    public DocumentV2JpaEntity(String id,
                               String title,
                               String status,
                               String aclTags,
                               String latestVersionId,
                               String metadataJson,
                               Instant createdAt,
                               Instant updatedAt) {
        this.id = id;
        this.title = title;
        this.status = status;
        this.aclTags = aclTags == null || aclTags.isBlank() ? "[]" : aclTags;
        this.latestVersionId = latestVersionId;
        this.metadataJson = metadataJson;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt == null ? createdAt : updatedAt;
    }

    public String getId() { return id; }
    public String getTitle() { return title; }
    public String getStatus() { return status; }
    public String getAclTags() { return aclTags; }
    public String getLatestVersionId() { return latestVersionId; }
    public String getMetadataJson() { return metadataJson; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }

    public void refresh(String title,
                        String status,
                        String aclTags,
                        String latestVersionId,
                        String metadataJson,
                        Instant now) {
        this.title = title;
        this.status = status;
        this.aclTags = aclTags == null || aclTags.isBlank() ? "[]" : aclTags;
        this.latestVersionId = latestVersionId;
        this.metadataJson = metadataJson;
        this.updatedAt = now;
    }
}
