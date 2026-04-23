-- Cross-modal retrieval: image metadata for CLIP FAISS index.
--
-- Each row maps a faiss_row_id in the image index back to the source
-- image and its document. Captions come from the ingest manifest (human-
-- authored or pre-computed during dataset preparation), NOT from CLIP.
--
-- Follows the same ownership model as V2: schema owned by Flyway,
-- the ai-worker only reads/writes rows.

CREATE TABLE ragmeta.image_chunks (
    image_id        VARCHAR(128)    PRIMARY KEY,
    doc_id          VARCHAR(128)    NOT NULL
                        REFERENCES ragmeta.documents(doc_id) ON DELETE CASCADE,
    page_number     INTEGER,
    source_uri      VARCHAR(2048),
    sha256          VARCHAR(64),
    caption         TEXT,
    section_hint    VARCHAR(256),
    faiss_row_id    INTEGER         NOT NULL,
    index_version   VARCHAR(64)     NOT NULL,
    extra_json      JSONB,
    created_at      TIMESTAMPTZ     DEFAULT now()
);

CREATE UNIQUE INDEX idx_image_chunks_version_row
    ON ragmeta.image_chunks (index_version, faiss_row_id);

CREATE INDEX idx_image_chunks_doc
    ON ragmeta.image_chunks (doc_id);
