package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

@Entity
@Table(name = "pdf_page_metadata")
public class PdfPageMetadataJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "document_id", length = 64)
    private String documentId;

    @Column(name = "document_version_id", nullable = false, length = 64)
    private String documentVersionId;

    @Column(name = "parsed_artifact_id", length = 64)
    private String parsedArtifactId;

    @Column(name = "source_file_id", length = 64)
    private String sourceFileId;

    @Column(name = "physical_page_index", nullable = false)
    private Integer physicalPageIndex;

    @Column(name = "page_no", nullable = false)
    private Integer pageNo;

    @Column(name = "page_label", length = 64)
    private String pageLabel;

    @Column(name = "width")
    private Double width;

    @Column(name = "height")
    private Double height;

    @Column(name = "text_layer_present")
    private Boolean textLayerPresent;

    @Column(name = "ocr_used", nullable = false)
    private Boolean ocrUsed = false;

    @Column(name = "ocr_engine", length = 128)
    private String ocrEngine;

    @Column(name = "ocr_model", length = 256)
    private String ocrModel;

    @Column(name = "ocr_confidence")
    private Double ocrConfidence;

    @Column(name = "ocr_confidence_avg")
    private Double ocrConfidenceAvg;

    @Column(name = "block_count")
    private Integer blockCount;

    @Column(name = "table_count")
    private Integer tableCount;

    @Column(name = "char_count")
    private Integer charCount;

    @Column(name = "quality_score")
    private Double qualityScore;

    @Column(name = "warnings_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String warningsJson = "[]";

    @Column(name = "metadata_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String metadataJson;

    protected PdfPageMetadataJpaEntity() {}

    public PdfPageMetadataJpaEntity(String id,
                                    String documentId,
                                    String documentVersionId,
                                    String parsedArtifactId,
                                    String sourceFileId,
                                    Integer physicalPageIndex,
                                    Integer pageNo,
                                    String pageLabel) {
        this.id = id;
        this.documentId = documentId;
        this.documentVersionId = documentVersionId;
        this.parsedArtifactId = parsedArtifactId;
        this.sourceFileId = sourceFileId;
        this.physicalPageIndex = physicalPageIndex;
        this.pageNo = pageNo;
        this.pageLabel = pageLabel;
    }

    public String getId() { return id; }
    public String getDocumentId() { return documentId; }
    public String getDocumentVersionId() { return documentVersionId; }
    public String getParsedArtifactId() { return parsedArtifactId; }
    public String getSourceFileId() { return sourceFileId; }
    public Integer getPhysicalPageIndex() { return physicalPageIndex; }
    public Integer getPageNo() { return pageNo; }
    public String getPageLabel() { return pageLabel; }
    public Double getWidth() { return width; }
    public Double getHeight() { return height; }
    public Boolean getTextLayerPresent() { return textLayerPresent; }
    public Boolean getOcrUsed() { return ocrUsed; }
    public String getOcrEngine() { return ocrEngine; }
    public String getOcrModel() { return ocrModel; }
    public Double getOcrConfidence() { return ocrConfidence; }
    public Double getOcrConfidenceAvg() { return ocrConfidenceAvg; }
    public Integer getBlockCount() { return blockCount; }
    public Integer getTableCount() { return tableCount; }
    public Integer getCharCount() { return charCount; }
    public Double getQualityScore() { return qualityScore; }
    public String getWarningsJson() { return warningsJson; }
    public String getMetadataJson() { return metadataJson; }

    public void refresh(String documentId,
                        String parsedArtifactId,
                        String sourceFileId,
                        Integer pageNo,
                        String pageLabel,
                        Double width,
                        Double height,
                        Boolean textLayerPresent,
                        Boolean ocrUsed,
                        String ocrEngine,
                        String ocrModel,
                        Double ocrConfidenceAvg,
                        Integer blockCount,
                        Integer tableCount,
                        Integer charCount,
                        Double qualityScore,
                        String warningsJson,
                        String metadataJson) {
        this.documentId = documentId;
        this.parsedArtifactId = parsedArtifactId;
        this.sourceFileId = sourceFileId;
        this.pageNo = pageNo;
        this.pageLabel = pageLabel;
        this.width = width;
        this.height = height;
        this.textLayerPresent = textLayerPresent;
        this.ocrUsed = ocrUsed == null ? false : ocrUsed;
        this.ocrEngine = ocrEngine;
        this.ocrModel = ocrModel;
        this.ocrConfidence = ocrConfidenceAvg;
        this.ocrConfidenceAvg = ocrConfidenceAvg;
        this.blockCount = blockCount;
        this.tableCount = tableCount;
        this.charCount = charCount;
        this.qualityScore = qualityScore;
        this.warningsJson = warningsJson == null || warningsJson.isBlank() ? "[]" : warningsJson;
        this.metadataJson = metadataJson == null || metadataJson.isBlank() ? "{}" : metadataJson;
    }
}
