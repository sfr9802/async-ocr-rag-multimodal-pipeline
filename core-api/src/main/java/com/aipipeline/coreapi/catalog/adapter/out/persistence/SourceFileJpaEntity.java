package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "source_file")
public class SourceFileJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "original_file_name", nullable = false, length = 512)
    private String originalFileName;

    @Column(name = "mime_type", length = 128)
    private String mimeType;

    @Column(name = "file_type", nullable = false, length = 32)
    private String fileType;

    @Column(name = "storage_uri", nullable = false, length = 1024)
    private String storageUri;

    @Column(name = "status", nullable = false, length = 32)
    private String status;

    @Column(name = "status_detail", columnDefinition = "TEXT")
    private String statusDetail;

    @Column(name = "uploaded_at", nullable = false)
    private Instant uploadedAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    protected SourceFileJpaEntity() {}

    public SourceFileJpaEntity(String id,
                               String originalFileName,
                               String mimeType,
                               String fileType,
                               String storageUri,
                               String status,
                               Instant uploadedAt) {
        this(id, originalFileName, mimeType, fileType, storageUri, status, null, uploadedAt, uploadedAt);
    }

    public SourceFileJpaEntity(String id,
                               String originalFileName,
                               String mimeType,
                               String fileType,
                               String storageUri,
                               String status,
                               String statusDetail,
                               Instant uploadedAt,
                               Instant updatedAt) {
        this.id = id;
        this.originalFileName = originalFileName;
        this.mimeType = mimeType;
        this.fileType = fileType;
        this.storageUri = storageUri;
        this.status = status;
        this.statusDetail = statusDetail;
        this.uploadedAt = uploadedAt;
        this.updatedAt = updatedAt == null ? uploadedAt : updatedAt;
    }

    public String getId() { return id; }
    public String getOriginalFileName() { return originalFileName; }
    public String getMimeType() { return mimeType; }
    public String getFileType() { return fileType; }
    public String getStorageUri() { return storageUri; }
    public String getStatus() { return status; }
    public String getStatusDetail() { return statusDetail; }
    public Instant getUploadedAt() { return uploadedAt; }
    public Instant getUpdatedAt() { return updatedAt; }

    public void setOriginalFileName(String originalFileName) {
        this.originalFileName = originalFileName;
    }

    public void setMimeType(String mimeType) {
        this.mimeType = mimeType;
    }

    public void setFileType(String fileType) {
        this.fileType = fileType;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public void transitionTo(String status, String statusDetail, Instant now) {
        this.status = status;
        this.statusDetail = statusDetail;
        this.updatedAt = now == null ? this.updatedAt : now;
    }
}
