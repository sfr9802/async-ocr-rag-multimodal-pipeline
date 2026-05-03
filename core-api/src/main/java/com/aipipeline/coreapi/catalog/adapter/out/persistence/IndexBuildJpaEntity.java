package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import com.aipipeline.coreapi.catalog.domain.IndexBuildStatus;
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
@Table(name = "index_build")
public class IndexBuildJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 128)
    private String id;

    @Column(name = "index_version", nullable = false, length = 128)
    private String indexVersion;

    @Column(name = "candidate_index_version", length = 128)
    private String candidateIndexVersion;

    @Column(name = "previous_index_version", length = 128)
    private String previousIndexVersion;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 32)
    private IndexBuildStatus status;

    @Column(name = "is_active", nullable = false)
    private boolean active;

    @Column(name = "parser_versions_json", columnDefinition = "jsonb", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private String parserVersionsJson = "{}";

    @Column(name = "chunk_count", nullable = false)
    private int chunkCount;

    @Column(name = "quality_gate_json", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String qualityGateJson;

    @Column(name = "eval_result_id", length = 64)
    private String evalResultId;

    @Column(name = "failure_reason_json", columnDefinition = "jsonb", nullable = false)
    @JdbcTypeCode(SqlTypes.JSON)
    private String failureReasonJson = "[]";

    @Column(name = "built_at")
    private Instant builtAt;

    @Column(name = "promoted_at")
    private Instant promotedAt;

    @Column(name = "rolled_back_at")
    private Instant rolledBackAt;

    @Column(name = "created_at")
    private Instant createdAt;

    @Column(name = "updated_at")
    private Instant updatedAt;

    protected IndexBuildJpaEntity() {}

    public IndexBuildJpaEntity(String id,
                               String indexVersion,
                               String candidateIndexVersion,
                               String previousIndexVersion,
                               IndexBuildStatus status,
                               boolean active,
                               String parserVersionsJson,
                               int chunkCount,
                               String qualityGateJson,
                               String evalResultId,
                               String failureReasonJson,
                               Instant builtAt,
                               Instant promotedAt,
                               Instant rolledBackAt,
                               Instant createdAt,
                               Instant updatedAt) {
        this.id = id;
        this.indexVersion = indexVersion;
        this.candidateIndexVersion = candidateIndexVersion == null || candidateIndexVersion.isBlank()
                ? indexVersion
                : candidateIndexVersion;
        this.previousIndexVersion = previousIndexVersion;
        this.status = status;
        this.active = active;
        this.parserVersionsJson = parserVersionsJson == null || parserVersionsJson.isBlank()
                ? "{}"
                : parserVersionsJson;
        this.chunkCount = Math.max(0, chunkCount);
        this.qualityGateJson = qualityGateJson;
        this.evalResultId = evalResultId;
        this.failureReasonJson = failureReasonJson == null || failureReasonJson.isBlank()
                ? "[]"
                : failureReasonJson;
        this.builtAt = builtAt;
        this.promotedAt = promotedAt;
        this.rolledBackAt = rolledBackAt;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt == null ? createdAt : updatedAt;
    }

    public String getId() { return id; }
    public String getIndexVersion() { return indexVersion; }
    public String getCandidateIndexVersion() { return candidateIndexVersion; }
    public String getPreviousIndexVersion() { return previousIndexVersion; }
    public IndexBuildStatus getStatus() { return status; }
    public boolean isActive() { return active; }
    public String getParserVersionsJson() { return parserVersionsJson; }
    public int getChunkCount() { return chunkCount; }
    public String getQualityGateJson() { return qualityGateJson; }
    public String getEvalResultId() { return evalResultId; }
    public String getFailureReasonJson() { return failureReasonJson; }
    public Instant getBuiltAt() { return builtAt; }
    public Instant getPromotedAt() { return promotedAt; }
    public Instant getRolledBackAt() { return rolledBackAt; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }

    public void attachEvalResult(String evalResultId, IndexBuildStatus evaluatedStatus, Instant now) {
        this.evalResultId = evalResultId;
        this.status = evaluatedStatus;
        this.updatedAt = now;
    }

    public void markPromoted(String previousIndexVersion, Instant now) {
        if (previousIndexVersion != null && !previousIndexVersion.isBlank()) {
            this.previousIndexVersion = previousIndexVersion;
        }
        this.status = IndexBuildStatus.PROMOTED;
        this.active = true;
        this.promotedAt = now;
        this.updatedAt = now;
    }

    public void deactivate(Instant now) {
        this.active = false;
        this.updatedAt = now;
    }

    public void markRolledBack(String currentIndexVersion, String previousIndexVersion, Instant now) {
        if (currentIndexVersion != null && !currentIndexVersion.isBlank()) {
            this.candidateIndexVersion = currentIndexVersion;
        }
        if (previousIndexVersion != null && !previousIndexVersion.isBlank()) {
            this.previousIndexVersion = previousIndexVersion;
        }
        this.status = IndexBuildStatus.ROLLED_BACK;
        this.active = false;
        this.rolledBackAt = now;
        this.updatedAt = now;
    }
}
