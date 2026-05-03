-- Document ingestion v2 for xlsx/pdf RAG corpus expansion.
--
-- Keep the existing source_file/extracted_artifact/search_unit path intact.
-- The new document/document_version/parsed_artifact tables provide stable
-- parser provenance, while search_unit gets v2 columns for location-aware
-- citation and separate retrieval/display texts.

CREATE TABLE IF NOT EXISTS document (
    id                  VARCHAR(64)   PRIMARY KEY,
    title               VARCHAR(512)  NOT NULL,
    status              VARCHAR(32)   NOT NULL DEFAULT 'ACTIVE',
    acl_tags            JSONB         NOT NULL DEFAULT '[]'::jsonb,
    latest_version_id   VARCHAR(64),
    metadata_json       JSONB,
    created_at          TIMESTAMP     NOT NULL,
    updated_at          TIMESTAMP     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_status ON document (status);

CREATE TABLE IF NOT EXISTS document_version (
    id                  VARCHAR(64)   PRIMARY KEY,
    document_id          VARCHAR(64)   NOT NULL,
    source_file_id       VARCHAR(64),
    version_no           INTEGER       NOT NULL,
    source_file_name     VARCHAR(512)  NOT NULL,
    source_file_type     VARCHAR(32)   NOT NULL,
    mime_type            VARCHAR(128),
    storage_uri          VARCHAR(1024) NOT NULL,
    checksum_sha256      VARCHAR(128),
    parse_status         VARCHAR(32)   NOT NULL DEFAULT 'UPLOADED',
    parse_status_detail  TEXT,
    parser_policy_json   JSONB,
    acl_tags             JSONB         NOT NULL DEFAULT '[]'::jsonb,
    created_at           TIMESTAMP     NOT NULL,
    updated_at           TIMESTAMP     NOT NULL,
    CONSTRAINT fk_document_version_document
        FOREIGN KEY (document_id) REFERENCES document (id),
    CONSTRAINT fk_document_version_source_file
        FOREIGN KEY (source_file_id) REFERENCES source_file (id),
    CONSTRAINT uq_document_version_no
        UNIQUE (document_id, version_no)
);

ALTER TABLE document
    ADD CONSTRAINT fk_document_latest_version
        FOREIGN KEY (latest_version_id) REFERENCES document_version (id);

CREATE INDEX IF NOT EXISTS idx_document_version_document
    ON document_version (document_id);
CREATE INDEX IF NOT EXISTS idx_document_version_source_file
    ON document_version (source_file_id);
CREATE INDEX IF NOT EXISTS idx_document_version_parse_status
    ON document_version (parse_status);

CREATE TABLE IF NOT EXISTS parsed_artifact (
    id                    VARCHAR(64)   PRIMARY KEY,
    document_version_id   VARCHAR(64)   NOT NULL,
    source_file_id        VARCHAR(64),
    extracted_artifact_id VARCHAR(64),
    artifact_type         VARCHAR(64)   NOT NULL,
    storage_uri           VARCHAR(1024),
    parser_name           VARCHAR(128)  NOT NULL,
    parser_version        VARCHAR(128)  NOT NULL,
    file_type             VARCHAR(32)   NOT NULL,
    artifact_json         JSONB         NOT NULL,
    warnings_json         JSONB         NOT NULL DEFAULT '[]'::jsonb,
    quality_score         DOUBLE PRECISION,
    created_at            TIMESTAMP     NOT NULL,
    CONSTRAINT fk_parsed_artifact_document_version
        FOREIGN KEY (document_version_id) REFERENCES document_version (id),
    CONSTRAINT fk_parsed_artifact_source_file
        FOREIGN KEY (source_file_id) REFERENCES source_file (id),
    CONSTRAINT fk_parsed_artifact_extracted_artifact
        FOREIGN KEY (extracted_artifact_id) REFERENCES extracted_artifact (artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_parsed_artifact_document_version
    ON parsed_artifact (document_version_id);
CREATE INDEX IF NOT EXISTS idx_parsed_artifact_parser
    ON parsed_artifact (parser_name, parser_version);

ALTER TABLE search_unit
    ADD COLUMN IF NOT EXISTS document_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS document_version_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS parsed_artifact_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_file_name VARCHAR(512),
    ADD COLUMN IF NOT EXISTS source_file_type VARCHAR(32),
    ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(64),
    ADD COLUMN IF NOT EXISTS location_type VARCHAR(32),
    ADD COLUMN IF NOT EXISTS location_json JSONB,
    ADD COLUMN IF NOT EXISTS embedding_text TEXT,
    ADD COLUMN IF NOT EXISTS bm25_text TEXT,
    ADD COLUMN IF NOT EXISTS display_text TEXT,
    ADD COLUMN IF NOT EXISTS citation_text TEXT,
    ADD COLUMN IF NOT EXISTS debug_text TEXT,
    ADD COLUMN IF NOT EXISTS parser_name VARCHAR(128),
    ADD COLUMN IF NOT EXISTS parser_version VARCHAR(128),
    ADD COLUMN IF NOT EXISTS index_version VARCHAR(128),
    ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS confidence_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS acl_tags JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE search_unit
    ADD CONSTRAINT fk_search_unit_document
        FOREIGN KEY (document_id) REFERENCES document (id);
ALTER TABLE search_unit
    ADD CONSTRAINT fk_search_unit_document_version
        FOREIGN KEY (document_version_id) REFERENCES document_version (id);
ALTER TABLE search_unit
    ADD CONSTRAINT fk_search_unit_parsed_artifact
        FOREIGN KEY (parsed_artifact_id) REFERENCES parsed_artifact (id);

CREATE INDEX IF NOT EXISTS idx_search_unit_document_version
    ON search_unit (document_version_id);
CREATE INDEX IF NOT EXISTS idx_search_unit_index_version
    ON search_unit (index_version);
CREATE INDEX IF NOT EXISTS idx_search_unit_parser_version
    ON search_unit (parser_version);
CREATE INDEX IF NOT EXISTS idx_search_unit_v2_ready
    ON search_unit (index_version, parser_version, chunk_type)
    WHERE citation_text IS NOT NULL AND location_json IS NOT NULL;

CREATE TABLE IF NOT EXISTS table_metadata (
    id                    VARCHAR(64)  PRIMARY KEY,
    search_unit_id         VARCHAR(64),
    document_version_id    VARCHAR(64),
    table_id              VARCHAR(128) NOT NULL,
    title                 VARCHAR(512),
    location_json         JSONB        NOT NULL,
    header_paths_json     JSONB,
    row_count             INTEGER,
    column_count          INTEGER,
    quality_score         DOUBLE PRECISION,
    created_at            TIMESTAMP    NOT NULL,
    CONSTRAINT fk_table_metadata_search_unit
        FOREIGN KEY (search_unit_id) REFERENCES search_unit (id),
    CONSTRAINT fk_table_metadata_document_version
        FOREIGN KEY (document_version_id) REFERENCES document_version (id)
);

CREATE INDEX IF NOT EXISTS idx_table_metadata_document_version
    ON table_metadata (document_version_id);
CREATE INDEX IF NOT EXISTS idx_table_metadata_table_id
    ON table_metadata (table_id);

CREATE TABLE IF NOT EXISTS cell_metadata (
    id                    VARCHAR(64) PRIMARY KEY,
    document_version_id    VARCHAR(64),
    table_metadata_id      VARCHAR(64),
    sheet_name            VARCHAR(256),
    sheet_index           INTEGER,
    cell_address          VARCHAR(32),
    row_index             INTEGER,
    column_index          INTEGER,
    header_path           JSONB,
    value_text            TEXT,
    formula_text          TEXT,
    cached_value_text     TEXT,
    number_format         VARCHAR(128),
    date_format           VARCHAR(128),
    merged_cell           BOOLEAN NOT NULL DEFAULT FALSE,
    hidden                BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT fk_cell_metadata_document_version
        FOREIGN KEY (document_version_id) REFERENCES document_version (id),
    CONSTRAINT fk_cell_metadata_table
        FOREIGN KEY (table_metadata_id) REFERENCES table_metadata (id)
);

CREATE INDEX IF NOT EXISTS idx_cell_metadata_document_version
    ON cell_metadata (document_version_id);
CREATE INDEX IF NOT EXISTS idx_cell_metadata_sheet_cell
    ON cell_metadata (document_version_id, sheet_index, cell_address);

CREATE TABLE IF NOT EXISTS pdf_page_metadata (
    id                    VARCHAR(64) PRIMARY KEY,
    document_version_id    VARCHAR(64) NOT NULL,
    physical_page_index    INTEGER     NOT NULL,
    page_no               INTEGER     NOT NULL,
    page_label            VARCHAR(64),
    width                 DOUBLE PRECISION,
    height                DOUBLE PRECISION,
    text_layer_present    BOOLEAN,
    ocr_used              BOOLEAN     NOT NULL DEFAULT FALSE,
    ocr_confidence        DOUBLE PRECISION,
    metadata_json         JSONB,
    CONSTRAINT fk_pdf_page_metadata_document_version
        FOREIGN KEY (document_version_id) REFERENCES document_version (id),
    CONSTRAINT uq_pdf_page_metadata_page
        UNIQUE (document_version_id, physical_page_index)
);

CREATE TABLE IF NOT EXISTS embedding_record (
    id                    VARCHAR(64)  PRIMARY KEY,
    search_unit_id         VARCHAR(64)  NOT NULL,
    index_version          VARCHAR(128) NOT NULL,
    embedding_model        VARCHAR(256) NOT NULL,
    embedding_text_sha256  VARCHAR(128) NOT NULL,
    vector_id              VARCHAR(512),
    created_at             TIMESTAMP    NOT NULL,
    CONSTRAINT fk_embedding_record_search_unit
        FOREIGN KEY (search_unit_id) REFERENCES search_unit (id)
);

CREATE INDEX IF NOT EXISTS idx_embedding_record_index_version
    ON embedding_record (index_version);
CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_record_unit_version
    ON embedding_record (search_unit_id, index_version, embedding_model);

CREATE TABLE IF NOT EXISTS index_build (
    index_version          VARCHAR(128) PRIMARY KEY,
    status                 VARCHAR(32)  NOT NULL,
    is_active              BOOLEAN      NOT NULL DEFAULT FALSE,
    parser_versions_json   JSONB        NOT NULL DEFAULT '{}'::jsonb,
    chunk_count            INTEGER      NOT NULL DEFAULT 0,
    quality_gate_json      JSONB,
    built_at               TIMESTAMP,
    promoted_at            TIMESTAMP,
    rolled_back_at         TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_index
    ON index_build (is_active)
    WHERE is_active = TRUE;

CREATE TABLE IF NOT EXISTS citation (
    id                    VARCHAR(64) PRIMARY KEY,
    retrieval_hit_id       VARCHAR(64),
    search_unit_id         VARCHAR(64),
    document_version_id    VARCHAR(64),
    citation_text          TEXT        NOT NULL,
    location_json          JSONB       NOT NULL,
    created_at             TIMESTAMP   NOT NULL,
    CONSTRAINT fk_citation_search_unit
        FOREIGN KEY (search_unit_id) REFERENCES search_unit (id),
    CONSTRAINT fk_citation_document_version
        FOREIGN KEY (document_version_id) REFERENCES document_version (id)
);

CREATE TABLE IF NOT EXISTS eval_dataset (
    id                  VARCHAR(64)  PRIMARY KEY,
    name                VARCHAR(256) NOT NULL,
    version             VARCHAR(64)  NOT NULL,
    created_at          TIMESTAMP    NOT NULL,
    CONSTRAINT uq_eval_dataset_name_version UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS eval_query (
    id                      VARCHAR(64) PRIMARY KEY,
    dataset_id              VARCHAR(64) NOT NULL,
    bucket                  VARCHAR(64) NOT NULL,
    query                   TEXT        NOT NULL,
    expected_location_json  JSONB       NOT NULL,
    created_at              TIMESTAMP   NOT NULL,
    CONSTRAINT fk_eval_query_dataset
        FOREIGN KEY (dataset_id) REFERENCES eval_dataset (id)
);

CREATE INDEX IF NOT EXISTS idx_eval_query_bucket
    ON eval_query (dataset_id, bucket);

CREATE TABLE IF NOT EXISTS eval_result (
    id              VARCHAR(64)  PRIMARY KEY,
    dataset_id      VARCHAR(64)  NOT NULL,
    index_version   VARCHAR(128) NOT NULL,
    metrics_json    JSONB        NOT NULL,
    passed          BOOLEAN      NOT NULL,
    created_at      TIMESTAMP    NOT NULL,
    CONSTRAINT fk_eval_result_dataset
        FOREIGN KEY (dataset_id) REFERENCES eval_dataset (id)
);

CREATE INDEX IF NOT EXISTS idx_eval_result_index_version
    ON eval_result (index_version);
