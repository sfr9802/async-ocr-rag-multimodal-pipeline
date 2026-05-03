package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "extracted_artifact")
public class ExtractedArtifactJpaEntity {

    @Id
    @Column(name = "artifact_id", nullable = false, length = 64)
    private String artifactId;

    @Column(name = "source_file_id", nullable = false, length = 64)
    private String sourceFileId;

    @Column(name = "artifact_type", nullable = false, length = 64)
    private String artifactType;

    @Column(name = "artifact_key", nullable = false, length = 128)
    private String artifactKey;

    @Column(name = "storage_uri", nullable = false, length = 1024)
    private String storageUri;

    @Column(name = "pipeline_version", nullable = false, length = 64)
    private String pipelineVersion;

    @Column(name = "checksum_sha256", length = 128)
    private String checksumSha256;

    @Column(name = "payload_json", columnDefinition = "TEXT")
    private String payloadJson;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    protected ExtractedArtifactJpaEntity() {}

    public ExtractedArtifactJpaEntity(String artifactId,
                                      String sourceFileId,
                                      String artifactType,
                                      String storageUri,
                                      String pipelineVersion,
                                      Instant createdAt) {
        this(
                artifactId,
                sourceFileId,
                artifactType,
                artifactId,
                storageUri,
                pipelineVersion,
                null,
                null,
                createdAt,
                createdAt);
    }

    public ExtractedArtifactJpaEntity(String artifactId,
                                      String sourceFileId,
                                      String artifactType,
                                      String artifactKey,
                                      String storageUri,
                                      String pipelineVersion,
                                      String checksumSha256,
                                      String payloadJson,
                                      Instant createdAt,
                                      Instant updatedAt) {
        this.artifactId = artifactId;
        this.sourceFileId = sourceFileId;
        this.artifactType = artifactType;
        this.artifactKey = artifactKey;
        this.storageUri = storageUri;
        this.pipelineVersion = pipelineVersion;
        this.checksumSha256 = checksumSha256;
        this.payloadJson = payloadJson;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt == null ? createdAt : updatedAt;
    }

    public String getArtifactId() { return artifactId; }
    public String getSourceFileId() { return sourceFileId; }
    public String getArtifactType() { return artifactType; }
    public String getKind() { return artifactType; }
    public String getArtifactKey() { return artifactKey; }
    public String getStorageUri() { return storageUri; }
    public String getPipelineVersion() { return pipelineVersion; }
    public String getChecksumSha256() { return checksumSha256; }
    public String getPayloadJson() { return payloadJson; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }

    public void updatePayload(String storageUri,
                              String pipelineVersion,
                              String checksumSha256,
                              String payloadJson,
                              Instant now) {
        this.storageUri = storageUri;
        this.pipelineVersion = pipelineVersion;
        this.checksumSha256 = checksumSha256;
        this.payloadJson = payloadJson;
        this.updatedAt = now;
    }
}
