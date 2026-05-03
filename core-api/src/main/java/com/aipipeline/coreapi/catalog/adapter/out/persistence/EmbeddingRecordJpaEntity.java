package com.aipipeline.coreapi.catalog.adapter.out.persistence;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "embedding_record")
public class EmbeddingRecordJpaEntity {

    @Id
    @Column(name = "id", nullable = false, length = 64)
    private String id;

    @Column(name = "search_unit_id", nullable = false, length = 64)
    private String searchUnitId;

    @Column(name = "index_version", nullable = false, length = 128)
    private String indexVersion;

    @Column(name = "embedding_model", nullable = false, length = 256)
    private String embeddingModel;

    @Column(name = "embedding_text_sha256", nullable = false, length = 128)
    private String embeddingTextSha256;

    @Column(name = "vector_id", length = 512)
    private String vectorId;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected EmbeddingRecordJpaEntity() {}

    public EmbeddingRecordJpaEntity(String id) {
        this.id = id;
    }

    public String getId() { return id; }
    public String getSearchUnitId() { return searchUnitId; }
    public String getIndexVersion() { return indexVersion; }
    public String getEmbeddingModel() { return embeddingModel; }
    public String getEmbeddingTextSha256() { return embeddingTextSha256; }
    public String getVectorId() { return vectorId; }
    public Instant getCreatedAt() { return createdAt; }

    public void refresh(String searchUnitId,
                        String indexVersion,
                        String embeddingModel,
                        String embeddingTextSha256,
                        String vectorId,
                        Instant now) {
        this.searchUnitId = searchUnitId;
        this.indexVersion = indexVersion;
        this.embeddingModel = embeddingModel;
        this.embeddingTextSha256 = embeddingTextSha256;
        this.vectorId = vectorId;
        if (this.createdAt == null) {
            this.createdAt = now;
        }
    }
}
