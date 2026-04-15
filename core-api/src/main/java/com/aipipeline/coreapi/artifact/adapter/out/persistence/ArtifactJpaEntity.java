package com.aipipeline.coreapi.artifact.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "artifact")
public class ArtifactJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "job_id", nullable = false, length = 64)
    private String jobId;

    @Column(name = "role", nullable = false, length = 16)
    private String role;

    @Column(name = "type", nullable = false, length = 32)
    private String type;

    @Column(name = "storage_uri", nullable = false, length = 1024)
    private String storageUri;

    @Column(name = "content_type", length = 128)
    private String contentType;

    @Column(name = "size_bytes")
    private Long sizeBytes;

    @Column(name = "checksum_sha256", length = 128)
    private String checksumSha256;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected ArtifactJpaEntity() {}

    public ArtifactJpaEntity(String id, String jobId, String role, String type,
                             String storageUri, String contentType, Long sizeBytes,
                             String checksumSha256, Instant createdAt) {
        this.id = id;
        this.jobId = jobId;
        this.role = role;
        this.type = type;
        this.storageUri = storageUri;
        this.contentType = contentType;
        this.sizeBytes = sizeBytes;
        this.checksumSha256 = checksumSha256;
        this.createdAt = createdAt;
    }

    public String getId() { return id; }
    public String getJobId() { return jobId; }
    public String getRole() { return role; }
    public String getType() { return type; }
    public String getStorageUri() { return storageUri; }
    public String getContentType() { return contentType; }
    public Long getSizeBytes() { return sizeBytes; }
    public String getChecksumSha256() { return checksumSha256; }
    public Instant getCreatedAt() { return createdAt; }
}
