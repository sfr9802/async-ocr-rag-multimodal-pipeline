package com.aipipeline.coreapi.artifact.domain;

import java.util.Objects;
import java.util.UUID;

public record ArtifactId(String value) {

    public ArtifactId {
        Objects.requireNonNull(value, "ArtifactId value must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException("ArtifactId value must not be blank");
        }
    }

    public static ArtifactId generate() {
        return new ArtifactId(UUID.randomUUID().toString());
    }

    public static ArtifactId of(String value) {
        return new ArtifactId(value);
    }

    @Override
    public String toString() {
        return value;
    }
}
