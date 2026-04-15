package com.aipipeline.coreapi.common;

import org.springframework.stereotype.Component;

import java.time.Clock;
import java.time.Instant;

/**
 * Thin wrapper around {@link Clock} so that services can obtain the current
 * time via injection instead of calling {@code Instant.now()} directly. This
 * keeps tests deterministic.
 */
@Component
public class TimeProvider {

    private final Clock clock;

    public TimeProvider() {
        this(Clock.systemUTC());
    }

    public TimeProvider(Clock clock) {
        this.clock = clock;
    }

    public Instant now() {
        return Instant.now(clock);
    }
}
