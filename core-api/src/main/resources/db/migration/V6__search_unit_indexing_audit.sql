-- SearchUnit indexing state and retrieval audit traces.
--
-- V5 introduced the canonical SearchUnit shape. This migration keeps OCR
-- import separate from embedding/indexing by adding worker claim state on
-- search_unit, plus lightweight ragmeta audit tables owned by the worker.

ALTER TABLE search_unit ADD COLUMN IF NOT EXISTS index_id VARCHAR(512);
ALTER TABLE search_unit ADD COLUMN IF NOT EXISTS indexed_content_sha256 VARCHAR(128);
ALTER TABLE search_unit ADD COLUMN IF NOT EXISTS embedding_claim_token VARCHAR(128);
ALTER TABLE search_unit ADD COLUMN IF NOT EXISTS embedding_claimed_at TIMESTAMP;
ALTER TABLE search_unit ADD COLUMN IF NOT EXISTS embedding_status_detail TEXT;
ALTER TABLE search_unit ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_search_unit_index_id
    ON search_unit (index_id);
CREATE INDEX IF NOT EXISTS idx_search_unit_embedding_claim
    ON search_unit (embedding_status, embedding_claimed_at);
CREATE INDEX IF NOT EXISTS idx_search_unit_content_hash
    ON search_unit (content_sha256);

-- SearchUnit stable index ids can be longer than legacy chunk ids such as
-- `source_file:{uuid}:unit:PAGE:page:1`.
ALTER TABLE ragmeta.chunks ALTER COLUMN chunk_id TYPE VARCHAR(512);

CREATE TABLE IF NOT EXISTS ragmeta.retrieval_runs (
    id                VARCHAR(64) PRIMARY KEY,
    request_id        VARCHAR(128),
    user_id           VARCHAR(128),
    query             TEXT NOT NULL,
    normalized_query  TEXT,
    retriever_version VARCHAR(128),
    embedding_model   VARCHAR(256),
    reranker_model    VARCHAR(256),
    top_k             INTEGER,
    filters_json      JSONB,
    latency_ms        INTEGER,
    created_at        TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_request_id
    ON ragmeta.retrieval_runs (request_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_runs_created_at
    ON ragmeta.retrieval_runs (created_at);

CREATE TABLE IF NOT EXISTS ragmeta.retrieval_hits (
    id                    VARCHAR(64) PRIMARY KEY,
    retrieval_run_id      VARCHAR(64) NOT NULL,
    rank                  INTEGER NOT NULL,
    search_unit_id        VARCHAR(64),
    source_file_id        VARCHAR(64),
    unit_type             VARCHAR(64),
    unit_key              VARCHAR(256),
    dense_score           DOUBLE PRECISION,
    sparse_score          DOUBLE PRECISION,
    rrf_score             DOUBLE PRECISION,
    rerank_score          DOUBLE PRECISION,
    final_score           DOUBLE PRECISION,
    citation_json         JSONB,
    snippet               TEXT,
    selected_for_context  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at            TIMESTAMP NOT NULL,
    CONSTRAINT fk_retrieval_hit_run
        FOREIGN KEY (retrieval_run_id)
        REFERENCES ragmeta.retrieval_runs (id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_retrieval_hits_run_rank
    ON ragmeta.retrieval_hits (retrieval_run_id, rank);
CREATE INDEX IF NOT EXISTS idx_retrieval_hits_search_unit
    ON ragmeta.retrieval_hits (search_unit_id);
