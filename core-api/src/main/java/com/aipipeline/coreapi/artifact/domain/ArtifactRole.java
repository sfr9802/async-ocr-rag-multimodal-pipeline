package com.aipipeline.coreapi.artifact.domain;

/**
 * Whether an artifact was attached by the client at job creation (INPUT) or
 * produced by the worker during processing (OUTPUT). Kept as a separate
 * concept from {@link ArtifactType} so that a given type (e.g. INPUT_TEXT)
 * always has a single canonical direction and the worker cannot accidentally
 * overwrite client inputs.
 */
public enum ArtifactRole {
    INPUT,
    OUTPUT
}
