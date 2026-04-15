package com.aipipeline.coreapi.artifact.domain;

import com.aipipeline.coreapi.job.domain.JobId;

import java.time.Instant;

/**
 * Artifact domain object.
 *
 * An artifact is a content blob produced or consumed by a job. It knows how
 * to locate its content via {@link #getStorageUri()}, which is an opaque
 * string the storage adapter interprets (e.g. {@code local://...}, {@code
 * s3://bucket/key}, {@code gs://bucket/key}).
 */
public class Artifact {

    private final ArtifactId id;
    private final JobId jobId;
    private final ArtifactRole role;
    private final ArtifactType type;
    private final String storageUri;
    private final String contentType;
    private final Long sizeBytes;
    private final String checksumSha256;
    private final Instant createdAt;

    private Artifact(ArtifactId id,
                     JobId jobId,
                     ArtifactRole role,
                     ArtifactType type,
                     String storageUri,
                     String contentType,
                     Long sizeBytes,
                     String checksumSha256,
                     Instant createdAt) {
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

    public static Artifact createNew(JobId jobId,
                                     ArtifactRole role,
                                     ArtifactType type,
                                     String storageUri,
                                     String contentType,
                                     Long sizeBytes,
                                     String checksumSha256,
                                     Instant now) {
        return new Artifact(
                ArtifactId.generate(),
                jobId,
                role,
                type,
                storageUri,
                contentType,
                sizeBytes,
                checksumSha256,
                now);
    }

    public static Artifact rehydrate(ArtifactId id,
                                     JobId jobId,
                                     ArtifactRole role,
                                     ArtifactType type,
                                     String storageUri,
                                     String contentType,
                                     Long sizeBytes,
                                     String checksumSha256,
                                     Instant createdAt) {
        return new Artifact(id, jobId, role, type, storageUri, contentType,
                sizeBytes, checksumSha256, createdAt);
    }

    public ArtifactId getId() { return id; }
    public JobId getJobId() { return jobId; }
    public ArtifactRole getRole() { return role; }
    public ArtifactType getType() { return type; }
    public String getStorageUri() { return storageUri; }
    public String getContentType() { return contentType; }
    public Long getSizeBytes() { return sizeBytes; }
    public String getChecksumSha256() { return checksumSha256; }
    public Instant getCreatedAt() { return createdAt; }
}
