-- Initial schema for AI processing platform (phase 1).
--
-- Design notes:
--   - job is the pipeline state source-of-truth. Redis is only the dispatch channel.
--   - claim_token + claim_expires_at implement RIA's lease-based claim pattern.
--   - last_callback_id provides (job_id, callback_id) idempotency for callbacks.
--   - artifact.storage_uri is opaque to the DB: local://..., s3://..., gs://...
--     The storage adapter decides how to resolve it.
--   - attempt_no is kept on job for now. A dedicated job_attempt table can be
--     introduced later if retries need per-attempt artifact isolation.

CREATE TABLE job (
    id                  VARCHAR(64)   PRIMARY KEY,
    capability          VARCHAR(64)   NOT NULL,
    status              VARCHAR(32)   NOT NULL,
    attempt_no          INTEGER       NOT NULL DEFAULT 1,
    claim_token         VARCHAR(128),
    claimed_at          TIMESTAMP,
    claim_expires_at    TIMESTAMP,
    last_callback_id    VARCHAR(128),
    error_code          VARCHAR(64),
    error_message       VARCHAR(2000),
    created_at          TIMESTAMP     NOT NULL,
    updated_at          TIMESTAMP     NOT NULL
);

CREATE INDEX idx_job_status       ON job (status);
CREATE INDEX idx_job_capability   ON job (capability);
CREATE INDEX idx_job_updated_at   ON job (updated_at);

CREATE TABLE artifact (
    id                  VARCHAR(64)   PRIMARY KEY,
    job_id              VARCHAR(64)   NOT NULL,
    role                VARCHAR(16)   NOT NULL,  -- INPUT | OUTPUT
    type                VARCHAR(32)   NOT NULL,  -- ArtifactType
    storage_uri         VARCHAR(1024) NOT NULL,
    content_type        VARCHAR(128),
    size_bytes          BIGINT,
    checksum_sha256     VARCHAR(128),
    created_at          TIMESTAMP     NOT NULL,
    CONSTRAINT fk_artifact_job FOREIGN KEY (job_id) REFERENCES job (id)
);

CREATE INDEX idx_artifact_job_id ON artifact (job_id);
CREATE INDEX idx_artifact_type   ON artifact (type);
