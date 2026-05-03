package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

@Entity
@Table(name = "cell_metadata")
public class CellMetadataJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "document_id", length = 64)
    private String documentId;

    @Column(name = "document_version_id", length = 64)
    private String documentVersionId;

    @Column(name = "parsed_artifact_id", length = 64)
    private String parsedArtifactId;

    @Column(name = "source_file_id", length = 64)
    private String sourceFileId;

    @Column(name = "table_metadata_id", length = 64)
    private String tableMetadataId;

    @Column(name = "sheet_name", length = 256)
    private String sheetName;

    @Column(name = "sheet_index")
    private Integer sheetIndex;

    @Column(name = "cell_address", length = 32)
    private String cellAddress;

    @Column(name = "cell_ref", length = 32)
    private String cellRef;

    @Column(name = "row_index")
    private Integer rowIndex;

    @Column(name = "column_index")
    private Integer columnIndex;

    @Column(name = "column_letter", length = 16)
    private String columnLetter;

    @Column(name = "raw_value", columnDefinition = "TEXT")
    private String rawValue;

    @Column(name = "formatted_value", columnDefinition = "TEXT")
    private String formattedValue;

    @Column(name = "formula", columnDefinition = "TEXT")
    private String formula;

    @Column(name = "formula_text", columnDefinition = "TEXT")
    private String formulaText;

    @Column(name = "cached_value_text", columnDefinition = "TEXT")
    private String cachedValueText;

    @Column(name = "value_text", columnDefinition = "TEXT")
    private String valueText;

    @Column(name = "data_type", length = 64)
    private String dataType;

    @Column(name = "number_format", length = 128)
    private String numberFormat;

    @Column(name = "date_format", length = 128)
    private String dateFormat;

    @Column(name = "header_path", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String headerPath;

    @Column(name = "header_path_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String headerPathJson;

    @Column(name = "row_label_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String rowLabelJson;

    @Column(name = "column_label_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String columnLabelJson;

    @Column(name = "table_id", length = 128)
    private String tableId;

    @Column(name = "hidden", nullable = false)
    private Boolean hidden = false;

    @Column(name = "hidden_row", nullable = false)
    private Boolean hiddenRow = false;

    @Column(name = "hidden_column", nullable = false)
    private Boolean hiddenColumn = false;

    @Column(name = "merged_cell", nullable = false)
    private Boolean mergedCell = false;

    @Column(name = "merged_range", length = 64)
    private String mergedRange;

    @Column(name = "quality_score")
    private Double qualityScore;

    protected CellMetadataJpaEntity() {}

    public CellMetadataJpaEntity(String id) {
        this.id = id;
    }

    public String getId() { return id; }
    public String getCellRef() { return cellRef; }
    public String getFormattedValue() { return formattedValue; }
    public String getFormula() { return formula; }
    public String getHeaderPathJson() { return headerPathJson; }
    public String getColumnLabelJson() { return columnLabelJson; }

    public void refresh(String documentId,
                        String documentVersionId,
                        String parsedArtifactId,
                        String sourceFileId,
                        String tableMetadataId,
                        String sheetName,
                        Integer sheetIndex,
                        String cellRef,
                        Integer rowIndex,
                        Integer columnIndex,
                        String columnLetter,
                        String rawValue,
                        String formattedValue,
                        String formula,
                        String cachedValueText,
                        String dataType,
                        String numberFormat,
                        String headerPathJson,
                        String rowLabelJson,
                        String columnLabelJson,
                        String tableId,
                        Boolean hiddenRow,
                        Boolean hiddenColumn,
                        Boolean mergedCell,
                        String mergedRange,
                        Double qualityScore) {
        this.documentId = documentId;
        this.documentVersionId = documentVersionId;
        this.parsedArtifactId = parsedArtifactId;
        this.sourceFileId = sourceFileId;
        this.tableMetadataId = tableMetadataId;
        this.sheetName = sheetName;
        this.sheetIndex = sheetIndex;
        this.cellAddress = cellRef;
        this.cellRef = cellRef;
        this.rowIndex = rowIndex;
        this.columnIndex = columnIndex;
        this.columnLetter = columnLetter;
        this.rawValue = rawValue;
        this.formattedValue = formattedValue;
        this.formula = formula;
        this.formulaText = formula;
        this.cachedValueText = cachedValueText;
        this.valueText = formattedValue;
        this.dataType = dataType;
        this.numberFormat = numberFormat;
        this.dateFormat = numberFormat;
        this.headerPath = headerPathJson;
        this.headerPathJson = headerPathJson;
        this.rowLabelJson = rowLabelJson;
        this.columnLabelJson = columnLabelJson;
        this.tableId = tableId;
        this.hiddenRow = hiddenRow == null ? false : hiddenRow;
        this.hiddenColumn = hiddenColumn == null ? false : hiddenColumn;
        this.hidden = Boolean.TRUE.equals(hiddenRow) || Boolean.TRUE.equals(hiddenColumn);
        this.mergedCell = mergedCell == null ? false : mergedCell;
        this.mergedRange = mergedRange;
        this.qualityScore = qualityScore;
    }
}
