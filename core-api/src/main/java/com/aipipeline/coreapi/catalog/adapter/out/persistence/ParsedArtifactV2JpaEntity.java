package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

@Entity
@Table(name = "parsed_artifact")
public class ParsedArtifactV2JpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "document_version_id", nullable = false, length = 64)
    private String documentVersionId;

    @Column(name = "source_file_id", length = 64)
    private String sourceFileId;

    @Column(name = "extracted_artifact_id", length = 64)
    private String extractedArtifactId;

    @Column(name = "artifact_type", nullable = false, length = 64)
    private String artifactType;

    @Column(name = "storage_uri", length = 1024)
    private String storageUri;

    @Column(name = "parser_name", nullable = false, length = 128)
    private String parserName;

    @Column(name = "parser_version", nullable = false, length = 128)
    private String parserVersion;

    @Column(name = "file_type", nullable = false, length = 32)
    private String fileType;

    @Column(name = "artifact_json", columnDefinition = "jsonb", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private String artifactJson;

    @Column(name = "warnings_json", columnDefinition = "jsonb", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private String warningsJson = "[]";

    @Column(name = "quality_score")
    private Double qualityScore;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected ParsedArtifactV2JpaEntity() {}

    public ParsedArtifactV2JpaEntity(String id,
                                     String documentVersionId,
                                     String sourceFileId,
                                     String extractedArtifactId,
                                     String artifactType,
                                     String storageUri,
                                     String parserName,
                                     String parserVersion,
                                     String fileType,
                                     String artifactJson,
                                     String warningsJson,
                                     Double qualityScore,
                                     Instant createdAt) {
        this.id = id;
        this.documentVersionId = documentVersionId;
        this.sourceFileId = sourceFileId;
        this.extractedArtifactId = extractedArtifactId;
        this.artifactType = artifactType;
        this.storageUri = storageUri;
        this.parserName = parserName;
        this.parserVersion = parserVersion;
        this.fileType = fileType;
        this.artifactJson = artifactJson == null || artifactJson.isBlank() ? "{}" : artifactJson;
        this.warningsJson = warningsJson == null || warningsJson.isBlank() ? "[]" : warningsJson;
        this.qualityScore = qualityScore;
        this.createdAt = createdAt;
    }

    public String getId() { return id; }
    public String getDocumentVersionId() { return documentVersionId; }
    public String getSourceFileId() { return sourceFileId; }
    public String getExtractedArtifactId() { return extractedArtifactId; }
    public String getArtifactType() { return artifactType; }
    public String getStorageUri() { return storageUri; }
    public String getParserName() { return parserName; }
    public String getParserVersion() { return parserVersion; }
    public String getFileType() { return fileType; }
    public String getArtifactJson() { return artifactJson; }
    public String getWarningsJson() { return warningsJson; }
    public Double getQualityScore() { return qualityScore; }
    public Instant getCreatedAt() { return createdAt; }

    public void refresh(String storageUri,
                        String parserName,
                        String parserVersion,
                        String fileType,
                        String artifactJson,
                        String warningsJson,
                        Double qualityScore) {
        this.storageUri = storageUri;
        this.parserName = parserName;
        this.parserVersion = parserVersion;
        this.fileType = fileType;
        this.artifactJson = artifactJson == null || artifactJson.isBlank() ? "{}" : artifactJson;
        this.warningsJson = warningsJson == null || warningsJson.isBlank() ? "[]" : warningsJson;
        this.qualityScore = qualityScore;
    }
}
