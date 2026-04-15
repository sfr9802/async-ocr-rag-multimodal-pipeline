package com.aipipeline.coreapi.artifact.domain;

/**
 * Artifact content classifications that the platform will carry over time.
 *
 * Phase 1 actively uses:
 *   - INPUT_TEXT
 *   - INPUT_FILE
 *   - FINAL_RESPONSE
 *   - ERROR_REPORT
 *
 * Phase 2 adds:
 *   - OCR_TEXT     : plain extracted text from an OCR run
 *   - OCR_RESULT   : JSON envelope describing the OCR run (filename,
 *                    mime type, page count, text length, confidence,
 *                    engine name, warnings)
 *   - RETRIEVAL_RESULT : JSON report produced by the RAG capability
 *
 * Core-api only validates the enum name — artifact schemas are owned by
 * the producing capability.
 */
public enum ArtifactType {
    INPUT_TEXT,
    INPUT_FILE,
    OCR_TEXT,
    OCR_RESULT,
    RETRIEVAL_RESULT,
    FINAL_RESPONSE,
    ERROR_REPORT;

    public static ArtifactType fromString(String raw) {
        if (raw == null) {
            throw new IllegalArgumentException("artifact type must not be null");
        }
        try {
            return ArtifactType.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException ex) {
            throw new IllegalArgumentException("Unknown artifact type: " + raw, ex);
        }
    }
}
