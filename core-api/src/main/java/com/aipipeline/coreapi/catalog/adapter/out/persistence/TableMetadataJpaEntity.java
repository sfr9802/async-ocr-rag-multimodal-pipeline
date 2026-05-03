package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

@Entity
@Table(name = "table_metadata")
public class TableMetadataJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "search_unit_id", length = 64)
    private String searchUnitId;

    @Column(name = "document_id", length = 64)
    private String documentId;

    @Column(name = "document_version_id", length = 64)
    private String documentVersionId;

    @Column(name = "parsed_artifact_id", length = 64)
    private String parsedArtifactId;

    @Column(name = "source_file_id", length = 64)
    private String sourceFileId;

    @Column(name = "sheet_index")
    private Integer sheetIndex;

    @Column(name = "sheet_name", length = 256)
    private String sheetName;

    @Column(name = "table_id", nullable = false, length = 128)
    private String tableId;

    @Column(name = "table_name", length = 256)
    private String tableName;

    @Column(name = "title", length = 512)
    private String title;

    @Column(name = "cell_range", length = 64)
    private String cellRange;

    @Column(name = "header_range", length = 64)
    private String headerRange;

    @Column(name = "data_range", length = 64)
    private String dataRange;

    @Column(name = "row_count")
    private Integer rowCount;

    @Column(name = "column_count")
    private Integer columnCount;

    @Column(name = "hidden_policy", length = 64)
    private String hiddenPolicy;

    @Column(name = "hidden_sheet", nullable = false)
    private Boolean hiddenSheet = false;

    @Column(name = "detected_table_type", length = 64)
    private String detectedTableType;

    @Column(name = "header_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String headerJson;

    @Column(name = "header_paths_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String headerPathsJson;

    @Column(name = "location_json", columnDefinition = "jsonb", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private String locationJson = "{}";

    @Column(name = "quality_score")
    private Double qualityScore;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected TableMetadataJpaEntity() {}

    public TableMetadataJpaEntity(String id, String tableId, Instant createdAt) {
        this.id = id;
        this.tableId = tableId;
        this.createdAt = createdAt;
    }

    public String getId() { return id; }
    public String getTableId() { return tableId; }
    public String getSheetName() { return sheetName; }
    public String getCellRange() { return cellRange; }
    public String getHeaderJson() { return headerJson; }
    public String getHeaderPathsJson() { return headerPathsJson; }
    public String getLocationJson() { return locationJson; }

    public void refresh(String searchUnitId,
                        String documentId,
                        String documentVersionId,
                        String parsedArtifactId,
                        String sourceFileId,
                        Integer sheetIndex,
                        String sheetName,
                        String tableId,
                        String tableName,
                        String title,
                        String cellRange,
                        String headerRange,
                        String dataRange,
                        Integer rowCount,
                        Integer columnCount,
                        String hiddenPolicy,
                        Boolean hiddenSheet,
                        String detectedTableType,
                        String headerJson,
                        String headerPathsJson,
                        String locationJson,
                        Double qualityScore) {
        this.searchUnitId = searchUnitId;
        this.documentId = documentId;
        this.documentVersionId = documentVersionId;
        this.parsedArtifactId = parsedArtifactId;
        this.sourceFileId = sourceFileId;
        this.sheetIndex = sheetIndex;
        this.sheetName = sheetName;
        this.tableId = tableId;
        this.tableName = tableName;
        this.title = title;
        this.cellRange = cellRange;
        this.headerRange = headerRange;
        this.dataRange = dataRange;
        this.rowCount = rowCount;
        this.columnCount = columnCount;
        this.hiddenPolicy = hiddenPolicy;
        this.hiddenSheet = hiddenSheet == null ? false : hiddenSheet;
        this.detectedTableType = detectedTableType;
        this.headerJson = headerJson;
        this.headerPathsJson = headerPathsJson;
        this.locationJson = locationJson == null || locationJson.isBlank() ? "{}" : locationJson;
        this.qualityScore = qualityScore;
    }
}
