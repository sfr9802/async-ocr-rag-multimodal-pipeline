package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import com.aipipeline.coreapi.catalog.domain.EvalResultStatus;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

@Entity
@Table(name = "eval_result")
public class EvalResultJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "dataset_id", nullable = false, length = 64)
    private String datasetId;

    @Column(name = "eval_dataset_id", length = 64)
    private String evalDatasetId;

    @Column(name = "index_version", nullable = false, length = 128)
    private String indexVersion;

    @Column(name = "candidate_index_version", length = 128)
    private String candidateIndexVersion;

    @Column(name = "baseline_index_version", length = 128)
    private String baselineIndexVersion;

    @Column(name = "metrics_json", columnDefinition = "jsonb", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private String metricsJson = "{}";

    @Column(name = "passed", nullable = false)
    private boolean passed;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", length = 32)
    private EvalResultStatus status;

    @Column(name = "threshold_json", columnDefinition = "jsonb", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private String thresholdJson = "{}";

    @Column(name = "failure_reason_json", columnDefinition = "jsonb", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private String failureReasonJson = "[]";

    @Column(name = "report_uri", length = 1024)
    private String reportUri;

    @Column(name = "report_path", length = 1024)
    private String reportPath;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected EvalResultJpaEntity() {}

    public EvalResultJpaEntity(String id,
                               String datasetId,
                               String evalDatasetId,
                               String indexVersion,
                               String candidateIndexVersion,
                               String baselineIndexVersion,
                               String metricsJson,
                               boolean passed,
                               EvalResultStatus status,
                               String thresholdJson,
                               String failureReasonJson,
                               String reportUri,
                               String reportPath,
                               Instant createdAt) {
        this.id = id;
        this.datasetId = datasetId;
        this.evalDatasetId = evalDatasetId == null || evalDatasetId.isBlank() ? datasetId : evalDatasetId;
        this.indexVersion = indexVersion;
        this.candidateIndexVersion = candidateIndexVersion == null || candidateIndexVersion.isBlank()
                ? indexVersion
                : candidateIndexVersion;
        this.baselineIndexVersion = baselineIndexVersion;
        this.metricsJson = metricsJson == null || metricsJson.isBlank() ? "{}" : metricsJson;
        this.passed = passed;
        this.status = status;
        this.thresholdJson = thresholdJson == null || thresholdJson.isBlank() ? "{}" : thresholdJson;
        this.failureReasonJson = failureReasonJson == null || failureReasonJson.isBlank() ? "[]" : failureReasonJson;
        this.reportUri = reportUri;
        this.reportPath = reportPath;
        this.createdAt = createdAt;
    }

    public String getId() { return id; }
    public String getDatasetId() { return datasetId; }
    public String getEvalDatasetId() { return evalDatasetId; }
    public String getIndexVersion() { return indexVersion; }
    public String getCandidateIndexVersion() { return candidateIndexVersion; }
    public String getBaselineIndexVersion() { return baselineIndexVersion; }
    public String getMetricsJson() { return metricsJson; }
    public boolean isPassed() { return passed; }
    public EvalResultStatus getStatus() { return status; }
    public String getThresholdJson() { return thresholdJson; }
    public String getFailureReasonJson() { return failureReasonJson; }
    public String getReportUri() { return reportUri; }
    public String getReportPath() { return reportPath; }
    public Instant getCreatedAt() { return createdAt; }

    public EvalResultStatus effectiveStatus() {
        if (status != null) {
            return status;
        }
        return passed ? EvalResultStatus.PASSED : EvalResultStatus.FAILED;
    }
}
