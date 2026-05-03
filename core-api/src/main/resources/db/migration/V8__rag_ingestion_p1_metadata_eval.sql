-- RAG ingestion v2 P1 normalized metadata and report-only eval support.
-- This migration is additive: it preserves the v7 production path and widens
-- the normalized side tables for xlsx/pdf provenance and evaluation.

ALTER TABLE pdf_page_metadata
    ADD COLUMN IF NOT EXISTS document_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS parsed_artifact_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_file_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS ocr_engine VARCHAR(128),
    ADD COLUMN IF NOT EXISTS ocr_model VARCHAR(256),
    ADD COLUMN IF NOT EXISTS ocr_confidence_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS block_count INTEGER,
    ADD COLUMN IF NOT EXISTS table_count INTEGER,
    ADD COLUMN IF NOT EXISTS char_count INTEGER,
    ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS warnings_json JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_pdf_page_metadata_source_file
    ON pdf_page_metadata (source_file_id);
CREATE INDEX IF NOT EXISTS idx_pdf_page_metadata_parsed_artifact
    ON pdf_page_metadata (parsed_artifact_id);

ALTER TABLE pdf_page_metadata
    ADD CONSTRAINT fk_pdf_page_metadata_document
        FOREIGN KEY (document_id) REFERENCES document (id);
ALTER TABLE pdf_page_metadata
    ADD CONSTRAINT fk_pdf_page_metadata_parsed_artifact
        FOREIGN KEY (parsed_artifact_id) REFERENCES parsed_artifact (id);
ALTER TABLE pdf_page_metadata
    ADD CONSTRAINT fk_pdf_page_metadata_source_file
        FOREIGN KEY (source_file_id) REFERENCES source_file (id);

ALTER TABLE table_metadata
    ADD COLUMN IF NOT EXISTS document_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS parsed_artifact_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_file_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS sheet_index INTEGER,
    ADD COLUMN IF NOT EXISTS sheet_name VARCHAR(256),
    ADD COLUMN IF NOT EXISTS table_name VARCHAR(256),
    ADD COLUMN IF NOT EXISTS cell_range VARCHAR(64),
    ADD COLUMN IF NOT EXISTS header_range VARCHAR(64),
    ADD COLUMN IF NOT EXISTS data_range VARCHAR(64),
    ADD COLUMN IF NOT EXISTS hidden_policy VARCHAR(64),
    ADD COLUMN IF NOT EXISTS hidden_sheet BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS detected_table_type VARCHAR(64),
    ADD COLUMN IF NOT EXISTS header_json JSONB;

CREATE INDEX IF NOT EXISTS idx_table_metadata_source_file
    ON table_metadata (source_file_id);
CREATE INDEX IF NOT EXISTS idx_table_metadata_parsed_artifact
    ON table_metadata (parsed_artifact_id);
CREATE INDEX IF NOT EXISTS idx_table_metadata_sheet
    ON table_metadata (document_version_id, sheet_index, sheet_name);

ALTER TABLE table_metadata
    ADD CONSTRAINT fk_table_metadata_document
        FOREIGN KEY (document_id) REFERENCES document (id);
ALTER TABLE table_metadata
    ADD CONSTRAINT fk_table_metadata_parsed_artifact
        FOREIGN KEY (parsed_artifact_id) REFERENCES parsed_artifact (id);
ALTER TABLE table_metadata
    ADD CONSTRAINT fk_table_metadata_source_file
        FOREIGN KEY (source_file_id) REFERENCES source_file (id);

ALTER TABLE cell_metadata
    ADD COLUMN IF NOT EXISTS document_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS parsed_artifact_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS source_file_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS cell_ref VARCHAR(32),
    ADD COLUMN IF NOT EXISTS column_letter VARCHAR(16),
    ADD COLUMN IF NOT EXISTS raw_value TEXT,
    ADD COLUMN IF NOT EXISTS formatted_value TEXT,
    ADD COLUMN IF NOT EXISTS formula TEXT,
    ADD COLUMN IF NOT EXISTS data_type VARCHAR(64),
    ADD COLUMN IF NOT EXISTS header_path_json JSONB,
    ADD COLUMN IF NOT EXISTS row_label_json JSONB,
    ADD COLUMN IF NOT EXISTS column_label_json JSONB,
    ADD COLUMN IF NOT EXISTS table_id VARCHAR(128),
    ADD COLUMN IF NOT EXISTS hidden_row BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS hidden_column BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS merged_range VARCHAR(64),
    ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_cell_metadata_source_file
    ON cell_metadata (source_file_id);
CREATE INDEX IF NOT EXISTS idx_cell_metadata_parsed_artifact
    ON cell_metadata (parsed_artifact_id);
CREATE INDEX IF NOT EXISTS idx_cell_metadata_table_id
    ON cell_metadata (document_version_id, table_id);

ALTER TABLE cell_metadata
    ADD CONSTRAINT fk_cell_metadata_document
        FOREIGN KEY (document_id) REFERENCES document (id);
ALTER TABLE cell_metadata
    ADD CONSTRAINT fk_cell_metadata_parsed_artifact
        FOREIGN KEY (parsed_artifact_id) REFERENCES parsed_artifact (id);
ALTER TABLE cell_metadata
    ADD CONSTRAINT fk_cell_metadata_source_file
        FOREIGN KEY (source_file_id) REFERENCES source_file (id);

ALTER TABLE eval_result
    ADD COLUMN IF NOT EXISTS eval_dataset_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS candidate_index_version VARCHAR(128),
    ADD COLUMN IF NOT EXISTS baseline_index_version VARCHAR(128),
    ADD COLUMN IF NOT EXISTS status VARCHAR(32),
    ADD COLUMN IF NOT EXISTS threshold_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS failure_reason_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS report_uri VARCHAR(1024),
    ADD COLUMN IF NOT EXISTS report_path VARCHAR(1024);

UPDATE eval_result
SET eval_dataset_id = COALESCE(eval_dataset_id, dataset_id),
    candidate_index_version = COALESCE(candidate_index_version, index_version),
    status = COALESCE(status, CASE WHEN passed THEN 'PASSED' ELSE 'FAILED' END)
WHERE eval_dataset_id IS NULL
   OR candidate_index_version IS NULL
   OR status IS NULL;

CREATE INDEX IF NOT EXISTS idx_eval_result_candidate_index
    ON eval_result (candidate_index_version);
CREATE INDEX IF NOT EXISTS idx_eval_result_status
    ON eval_result (status);

ALTER TABLE index_build
    ADD COLUMN IF NOT EXISTS id VARCHAR(128),
    ADD COLUMN IF NOT EXISTS candidate_index_version VARCHAR(128),
    ADD COLUMN IF NOT EXISTS previous_index_version VARCHAR(128),
    ADD COLUMN IF NOT EXISTS eval_result_id VARCHAR(64),
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP,
    ADD COLUMN IF NOT EXISTS failure_reason_json JSONB NOT NULL DEFAULT '[]'::jsonb;

UPDATE index_build
SET id = COALESCE(id, index_version),
    candidate_index_version = COALESCE(candidate_index_version, index_version),
    created_at = COALESCE(created_at, built_at, CURRENT_TIMESTAMP),
    updated_at = COALESCE(updated_at, promoted_at, rolled_back_at, built_at, CURRENT_TIMESTAMP)
WHERE id IS NULL
   OR candidate_index_version IS NULL
   OR created_at IS NULL
   OR updated_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_index_build_id
    ON index_build (id);
CREATE INDEX IF NOT EXISTS idx_index_build_eval_result
    ON index_build (eval_result_id);

ALTER TABLE index_build
    ADD CONSTRAINT fk_index_build_eval_result
        FOREIGN KEY (eval_result_id) REFERENCES eval_result (id);
