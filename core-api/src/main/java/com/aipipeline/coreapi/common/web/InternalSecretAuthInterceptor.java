package com.aipipeline.coreapi.common.web;

import com.aipipeline.coreapi.common.InternalSecretProperties;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.servlet.HandlerInterceptor;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Validates the {@code X-Internal-Secret} header on every
 * {@code /api/internal/**} request.
 *
 * <ul>
 *   <li>If no secret is configured (null / blank) the interceptor logs a
 *       one-time WARN and passes all requests through — this is the
 *       local-dev / CI default.</li>
 *   <li>If a secret IS configured but the header is absent or wrong, a
 *       {@code 401} is returned with
 *       {@code {"error":"internal_auth_failed"}}.</li>
 * </ul>
 *
 * Comparison uses {@link MessageDigest#isEqual} for constant-time
 * evaluation.
 */
public class InternalSecretAuthInterceptor implements HandlerInterceptor {

    private static final Logger log = LoggerFactory.getLogger(InternalSecretAuthInterceptor.class);
    private static final String HEADER = "X-Internal-Secret";

    private final byte[] expectedBytes;
    private final boolean enforcing;
    private final AtomicBoolean warnedOnce = new AtomicBoolean(false);

    public InternalSecretAuthInterceptor(InternalSecretProperties props) {
        String secret = props.secret();
        if (secret == null || secret.isBlank()) {
            this.expectedBytes = null;
            this.enforcing = false;
        } else {
            this.expectedBytes = secret.getBytes(StandardCharsets.UTF_8);
            this.enforcing = true;
        }
    }

    @Override
    public boolean preHandle(HttpServletRequest request,
                             HttpServletResponse response,
                             Object handler) throws Exception {
        if (!enforcing) {
            if (warnedOnce.compareAndSet(false, true)) {
                log.warn("aipipeline.internal.secret is not configured — "
                        + "/api/internal/** requests are unauthenticated (dev mode)");
            }
            return true;
        }

        String header = request.getHeader(HEADER);
        if (header == null || !constantTimeEquals(header)) {
            log.warn("Internal auth failed: {} {} from {}",
                    request.getMethod(), request.getRequestURI(), request.getRemoteAddr());
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.setContentType(MediaType.APPLICATION_JSON_VALUE);
            response.getWriter().write("{\"error\":\"internal_auth_failed\"}");
            return false;
        }
        return true;
    }

    private boolean constantTimeEquals(String provided) {
        byte[] providedBytes = provided.getBytes(StandardCharsets.UTF_8);
        return MessageDigest.isEqual(expectedBytes, providedBytes);
    }
}
