package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

@Entity
@Table(name = "document_version")
public class DocumentVersionJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "document_id", nullable = false, length = 64)
    private String documentId;

    @Column(name = "source_file_id", length = 64)
    private String sourceFileId;

    @Column(name = "version_no", nullable = false)
    private Integer versionNo;

    @Column(name = "source_file_name", nullable = false, length = 512)
    private String sourceFileName;

    @Column(name = "source_file_type", nullable = false, length = 32)
    private String sourceFileType;

    @Column(name = "mime_type", length = 128)
    private String mimeType;

    @Column(name = "storage_uri", nullable = false, length = 1024)
    private String storageUri;

    @Column(name = "checksum_sha256", length = 128)
    private String checksumSha256;

    @Column(name = "parse_status", nullable = false, length = 32)
    private String parseStatus;

    @Column(name = "parse_status_detail", columnDefinition = "TEXT")
    private String parseStatusDetail;

    @Column(name = "parser_policy_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String parserPolicyJson;

    @Column(name = "acl_tags", columnDefinition = "jsonb", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private String aclTags = "[]";

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    protected DocumentVersionJpaEntity() {}

    public DocumentVersionJpaEntity(String id,
                                    String documentId,
                                    String sourceFileId,
                                    Integer versionNo,
                                    String sourceFileName,
                                    String sourceFileType,
                                    String mimeType,
                                    String storageUri,
                                    String checksumSha256,
                                    String parseStatus,
                                    String parseStatusDetail,
                                    String parserPolicyJson,
                                    String aclTags,
                                    Instant createdAt,
                                    Instant updatedAt) {
        this.id = id;
        this.documentId = documentId;
        this.sourceFileId = sourceFileId;
        this.versionNo = versionNo;
        this.sourceFileName = sourceFileName;
        this.sourceFileType = sourceFileType;
        this.mimeType = mimeType;
        this.storageUri = storageUri;
        this.checksumSha256 = checksumSha256;
        this.parseStatus = parseStatus;
        this.parseStatusDetail = parseStatusDetail;
        this.parserPolicyJson = parserPolicyJson;
        this.aclTags = aclTags == null || aclTags.isBlank() ? "[]" : aclTags;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt == null ? createdAt : updatedAt;
    }

    public String getId() { return id; }
    public String getDocumentId() { return documentId; }
    public String getSourceFileId() { return sourceFileId; }
    public Integer getVersionNo() { return versionNo; }
    public String getSourceFileName() { return sourceFileName; }
    public String getSourceFileType() { return sourceFileType; }
    public String getMimeType() { return mimeType; }
    public String getStorageUri() { return storageUri; }
    public String getChecksumSha256() { return checksumSha256; }
    public String getParseStatus() { return parseStatus; }
    public String getParseStatusDetail() { return parseStatusDetail; }
    public String getParserPolicyJson() { return parserPolicyJson; }
    public String getAclTags() { return aclTags; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }

    public void refresh(String sourceFileName,
                        String sourceFileType,
                        String mimeType,
                        String storageUri,
                        String checksumSha256,
                        String parseStatus,
                        String parseStatusDetail,
                        String parserPolicyJson,
                        String aclTags,
                        Instant now) {
        this.sourceFileName = sourceFileName;
        this.sourceFileType = sourceFileType;
        this.mimeType = mimeType;
        this.storageUri = storageUri;
        this.checksumSha256 = checksumSha256;
        this.parseStatus = parseStatus;
        this.parseStatusDetail = parseStatusDetail;
        this.parserPolicyJson = parserPolicyJson;
        this.aclTags = aclTags == null || aclTags.isBlank() ? "[]" : aclTags;
        this.updatedAt = now;
    }
}
