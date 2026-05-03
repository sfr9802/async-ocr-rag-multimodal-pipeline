-- Minimal Document Catalog schema.
--
-- This intentionally stays separate from ragmeta.*. SearchUnit is the shared
-- future input shape for library search and RAG indexing, but this migration
-- does not build embeddings, image indexes, layout chunks, or multimodal RAG.

CREATE TABLE source_file (
    id                  VARCHAR(64)   PRIMARY KEY,
    original_file_name  VARCHAR(512)  NOT NULL,
    mime_type           VARCHAR(128),
    file_type           VARCHAR(32)   NOT NULL,
    storage_uri         VARCHAR(1024) NOT NULL,
    status              VARCHAR(32)   NOT NULL,
    status_detail       TEXT,
    uploaded_at         TIMESTAMP     NOT NULL,
    updated_at          TIMESTAMP     NOT NULL
);

CREATE UNIQUE INDEX idx_source_file_storage_uri ON source_file (storage_uri);
CREATE INDEX idx_source_file_status ON source_file (status);
CREATE INDEX idx_source_file_file_type ON source_file (file_type);

CREATE TABLE extracted_artifact (
    artifact_id         VARCHAR(64)   PRIMARY KEY,
    source_file_id      VARCHAR(64)   NOT NULL,
    artifact_type       VARCHAR(64)   NOT NULL,
    artifact_key        VARCHAR(128)  NOT NULL,
    storage_uri         VARCHAR(1024) NOT NULL,
    pipeline_version    VARCHAR(64)   NOT NULL,
    checksum_sha256     VARCHAR(128),
    payload_json        TEXT,
    created_at          TIMESTAMP     NOT NULL,
    updated_at          TIMESTAMP     NOT NULL,
    CONSTRAINT fk_extracted_artifact_source_file
        FOREIGN KEY (source_file_id) REFERENCES source_file (id),
    CONSTRAINT fk_extracted_artifact_artifact
        FOREIGN KEY (artifact_id) REFERENCES artifact (id)
);

CREATE INDEX idx_extracted_artifact_source_file ON extracted_artifact (source_file_id);
CREATE INDEX idx_extracted_artifact_type ON extracted_artifact (artifact_type);
CREATE UNIQUE INDEX idx_extracted_artifact_source_type_key
    ON extracted_artifact (source_file_id, artifact_type, artifact_key);

CREATE TABLE search_unit (
    id                  VARCHAR(64)   PRIMARY KEY,
    source_file_id      VARCHAR(64)   NOT NULL,
    extracted_artifact_id VARCHAR(64),
    unit_type           VARCHAR(64)   NOT NULL,
    unit_key            VARCHAR(256)  NOT NULL,
    title               VARCHAR(512),
    section_path        VARCHAR(1024),
    page_start          INTEGER,
    page_end            INTEGER,
    text_content        TEXT,
    metadata_json       TEXT,
    embedding_status    VARCHAR(32)   NOT NULL,
    content_sha256      VARCHAR(128),
    created_at          TIMESTAMP     NOT NULL,
    updated_at          TIMESTAMP     NOT NULL,
    CONSTRAINT fk_search_unit_source_file
        FOREIGN KEY (source_file_id) REFERENCES source_file (id),
    CONSTRAINT fk_search_unit_extracted_artifact
        FOREIGN KEY (extracted_artifact_id) REFERENCES extracted_artifact (artifact_id)
);

CREATE INDEX idx_search_unit_source_file ON search_unit (source_file_id);
CREATE INDEX idx_search_unit_artifact ON search_unit (extracted_artifact_id);
CREATE INDEX idx_search_unit_unit_type ON search_unit (unit_type);
CREATE INDEX idx_search_unit_embedding_status ON search_unit (embedding_status);
CREATE UNIQUE INDEX idx_search_unit_source_type_key
    ON search_unit (source_file_id, unit_type, unit_key);
