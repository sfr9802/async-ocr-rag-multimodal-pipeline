-- RAG metadata schema.
--
-- Design intent: the PIPELINE schema (public: job, artifact) and the RAG
-- METADATA schema (ragmeta: documents, chunks, index_builds) must be
-- visibly separated. They live in the same PostgreSQL database because we
-- explicitly decided against introducing MongoDB, but they are in
-- different schemas so that Spring's JPA layer never even sees the rag
-- tables, and the worker's psycopg2 paths never accidentally touch
-- pipeline state.
--
-- Spring's Flyway is the authoritative migration runner, but Spring does
-- NOT map these tables via JPA entities — the worker owns reads/writes.
-- Keeping the DDL in Spring's migrations just means there is one place
-- the DB schema is versioned, which is simpler than dual-running Flyway
-- from two services.

CREATE SCHEMA IF NOT EXISTS ragmeta;

-- ---------------------------------------------------------------------
-- Documents: one row per source document ingested into the RAG system.
-- ---------------------------------------------------------------------
CREATE TABLE ragmeta.documents (
    doc_id        VARCHAR(128) PRIMARY KEY,
    title         VARCHAR(512),
    source        VARCHAR(256),           -- dataset / file name
    category      VARCHAR(128),           -- optional coarse grouping
    metadata_json JSONB,                  -- flexible slot for extra fields
    created_at    TIMESTAMP NOT NULL,
    updated_at    TIMESTAMP NOT NULL
);

CREATE INDEX idx_ragmeta_documents_source   ON ragmeta.documents (source);
CREATE INDEX idx_ragmeta_documents_category ON ragmeta.documents (category);

-- ---------------------------------------------------------------------
-- Chunks: one row per FAISS-indexed text span. The `faiss_row_id` is
-- the row index inside the FAISS index file for this index_version; it
-- is how a retrieval result maps back to text + metadata.
-- ---------------------------------------------------------------------
CREATE TABLE ragmeta.chunks (
    chunk_id       VARCHAR(64)  PRIMARY KEY,
    doc_id         VARCHAR(128) NOT NULL,
    section        VARCHAR(256),
    chunk_order    INTEGER      NOT NULL,
    text           TEXT         NOT NULL,
    token_count    INTEGER,
    faiss_row_id   INTEGER      NOT NULL,
    index_version  VARCHAR(64)  NOT NULL,
    extra_json     JSONB,
    created_at     TIMESTAMP    NOT NULL,
    CONSTRAINT fk_chunk_document FOREIGN KEY (doc_id)
        REFERENCES ragmeta.documents (doc_id) ON DELETE CASCADE
);

CREATE INDEX idx_ragmeta_chunks_doc_id        ON ragmeta.chunks (doc_id);
CREATE INDEX idx_ragmeta_chunks_index_version ON ragmeta.chunks (index_version);
-- Lookup path used by the retriever after FAISS returns a (row_id, score):
CREATE UNIQUE INDEX idx_ragmeta_chunks_version_row
    ON ragmeta.chunks (index_version, faiss_row_id);

-- ---------------------------------------------------------------------
-- Index builds: one row per FAISS index build (versioned).
-- Lets you answer "which embedding model built the current index?"
-- without guessing from filenames.
-- ---------------------------------------------------------------------
CREATE TABLE ragmeta.index_builds (
    index_version    VARCHAR(64) PRIMARY KEY,
    embedding_model  VARCHAR(256) NOT NULL,
    embedding_dim    INTEGER      NOT NULL,
    chunk_count      INTEGER      NOT NULL,
    document_count   INTEGER      NOT NULL,
    faiss_index_path VARCHAR(1024),
    notes            TEXT,
    built_at         TIMESTAMP    NOT NULL
);
