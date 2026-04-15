package com.aipipeline.coreapi.job.domain;

import java.util.Objects;
import java.util.UUID;

/**
 * Opaque job identifier. UUID under the hood but typed so the domain never
 * passes raw strings around.
 */
public record JobId(String value) {

    public JobId {
        Objects.requireNonNull(value, "JobId value must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException("JobId value must not be blank");
        }
    }

    public static JobId generate() {
        return new JobId(UUID.randomUUID().toString());
    }

    public static JobId of(String value) {
        return new JobId(value);
    }

    @Override
    public String toString() {
        return value;
    }
}
