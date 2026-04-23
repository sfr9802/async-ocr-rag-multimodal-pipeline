package com.aipipeline.coreapi.job.domain;

/**
 * What kind of work a job represents.
 *
 * The core API does not execute these — it just routes them via the queue
 * and lets the worker pick the matching capability implementation. New
 * capability values can be added freely; the core API does not need to know
 * their internals.
 *
 * Phase 1 only uses MOCK. The other values exist to keep the domain shape
 * stable as OCR / RAG / multimodal are added in later phases.
 */
public enum JobCapability {
    MOCK,
    OCR,
    RAG,
    MULTIMODAL,
    AUTO,
    AGENT;

    public static JobCapability fromString(String raw) {
        if (raw == null) {
            throw new IllegalArgumentException("capability must not be null");
        }
        try {
            return JobCapability.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException ex) {
            throw new IllegalArgumentException("Unknown capability: " + raw, ex);
        }
    }
}
