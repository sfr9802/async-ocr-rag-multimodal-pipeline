package com.aipipeline.coreapi.common;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Shared secret used to authenticate requests on {@code /api/internal/**}.
 *
 * <p>When {@code secret} is null or blank the interceptor runs in dev
 * pass-through mode (a WARN is logged once at startup). In production the
 * same value must be injected into the worker as
 * {@code AIPIPELINE_WORKER_INTERNAL_SECRET}.
 */
@ConfigurationProperties(prefix = "aipipeline.internal")
public record InternalSecretProperties(String secret) {}
