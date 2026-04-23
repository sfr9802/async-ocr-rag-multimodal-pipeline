package com.aipipeline.coreapi.common.web;

import com.aipipeline.coreapi.common.InternalSecretProperties;
import com.aipipeline.coreapi.job.adapter.in.web.InternalWorkerController;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase.ClaimResult;
import com.aipipeline.coreapi.job.domain.JobId;
import com.aipipeline.coreapi.job.domain.JobStatus;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Tests for {@link InternalSecretAuthInterceptor}.
 *
 * <p>Uses standalone MockMvc (no Spring context) following the same pattern
 * as {@code JobControllerValidationTest}. The interceptor is registered
 * manually via {@code addInterceptors}.
 */
class InternalSecretAuthInterceptorTest {

    private static final String CLAIM_URL = "/api/internal/jobs/claim";
    private static final String CLAIM_BODY = """
            {"jobId":"job-1","workerClaimToken":"w1","attemptNo":1}""";

    private MockMvc buildMvc(String configuredSecret) {
        JobExecutionUseCase execution = Mockito.mock(JobExecutionUseCase.class);
        when(execution.claim(any())).thenReturn(new ClaimResult(
                false, JobStatus.PENDING, "not granted", null, 1, List.of()));

        InternalWorkerController controller = new InternalWorkerController(execution);
        InternalSecretProperties props = new InternalSecretProperties(configuredSecret);

        return MockMvcBuilders.standaloneSetup(controller)
                .addInterceptors(new InternalSecretAuthInterceptor(props))
                .build();
    }

    // ==========================================================
    // Secret NOT configured → dev pass-through
    // ==========================================================

    @Nested
    class DevMode {
        @Test
        void no_configured_secret_passes_without_header() throws Exception {
            MockMvc mvc = buildMvc(null);
            mvc.perform(post(CLAIM_URL)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(CLAIM_BODY))
                    .andExpect(status().isOk());
        }

        @Test
        void blank_configured_secret_passes_without_header() throws Exception {
            MockMvc mvc = buildMvc("  ");
            mvc.perform(post(CLAIM_URL)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(CLAIM_BODY))
                    .andExpect(status().isOk());
        }
    }

    // ==========================================================
    // Secret configured → enforcing mode
    // ==========================================================

    @Nested
    class Enforcing {
        private final MockMvc mvc = buildMvc("s3cret-value");

        @Test
        void missing_header_returns_401() throws Exception {
            mvc.perform(post(CLAIM_URL)
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(CLAIM_BODY))
                    .andExpect(status().isUnauthorized())
                    .andExpect(jsonPath("$.error").value("internal_auth_failed"));
        }

        @Test
        void wrong_header_returns_401() throws Exception {
            mvc.perform(post(CLAIM_URL)
                            .contentType(MediaType.APPLICATION_JSON)
                            .header("X-Internal-Secret", "wrong-value")
                            .content(CLAIM_BODY))
                    .andExpect(status().isUnauthorized())
                    .andExpect(jsonPath("$.error").value("internal_auth_failed"));
        }

        @Test
        void correct_header_passes() throws Exception {
            mvc.perform(post(CLAIM_URL)
                            .contentType(MediaType.APPLICATION_JSON)
                            .header("X-Internal-Secret", "s3cret-value")
                            .content(CLAIM_BODY))
                    .andExpect(status().isOk());
        }
    }
}
