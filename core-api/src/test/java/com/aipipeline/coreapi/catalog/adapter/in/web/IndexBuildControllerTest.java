package com.aipipeline.coreapi.catalog.adapter.in.web;

import com.aipipeline.coreapi.catalog.adapter.out.persistence.IndexBuildJpaEntity;
import com.aipipeline.coreapi.catalog.application.service.RagIndexBuildService;
import com.aipipeline.coreapi.catalog.application.service.RagIndexBuildService.CreateIndexBuildCommand;
import com.aipipeline.coreapi.catalog.application.service.RagIndexBuildService.RollbackIndexBuildCommand;
import com.aipipeline.coreapi.catalog.domain.IndexBuildStatus;
import com.aipipeline.coreapi.common.TimeProvider;
import com.aipipeline.coreapi.common.web.GlobalExceptionHandler;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.time.Instant;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class IndexBuildControllerTest {

    private static final Instant NOW = Instant.parse("2026-05-03T00:00:00Z");

    private RagIndexBuildService indexBuilds;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        indexBuilds = Mockito.mock(RagIndexBuildService.class);
        TimeProvider timeProvider = Mockito.mock(TimeProvider.class);
        when(timeProvider.now()).thenReturn(NOW);
        IndexBuildController controller = new IndexBuildController(indexBuilds, timeProvider);
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
    }

    @Test
    void create_endpoint_persists_index_build_contract() throws Exception {
        when(indexBuilds.createIndexBuild(any(CreateIndexBuildCommand.class), eq(NOW)))
                .thenReturn(build("build-1", IndexBuildStatus.CREATED, null, false));

        mockMvc.perform(post("/api/v1/rag/index-builds")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "id":"build-1",
                                  "indexVersion":"idx-candidate",
                                  "candidateIndexVersion":"idx-candidate",
                                  "previousIndexVersion":"idx-active",
                                  "parserVersionsJson":"{\\"pdf\\":\\"pdf-extract-v1\\"}",
                                  "chunkCount":7,
                                  "qualityGateJson":"{\\"gate\\":\\"seed\\"}"
                                }
                                """))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value("build-1"))
                .andExpect(jsonPath("$.indexVersion").value("idx-candidate"))
                .andExpect(jsonPath("$.candidateIndexVersion").value("idx-candidate"))
                .andExpect(jsonPath("$.previousIndexVersion").value("idx-active"))
                .andExpect(jsonPath("$.status").value("CREATED"))
                .andExpect(jsonPath("$.active").value(false))
                .andExpect(jsonPath("$.chunkCount").value(7));

        ArgumentCaptor<CreateIndexBuildCommand> captor = ArgumentCaptor.forClass(CreateIndexBuildCommand.class);
        verify(indexBuilds).createIndexBuild(captor.capture(), eq(NOW));
        assertThat(captor.getValue().id()).isEqualTo("build-1");
        assertThat(captor.getValue().candidateIndexVersion()).isEqualTo("idx-candidate");
        assertThat(captor.getValue().chunkCount()).isEqualTo(7);
    }

    @Test
    void get_endpoint_returns_index_build() throws Exception {
        when(indexBuilds.findIndexBuild("build-1"))
                .thenReturn(Optional.of(build("build-1", IndexBuildStatus.EVAL_PASSED, "eval-1", false)));

        mockMvc.perform(get("/api/v1/rag/index-builds/build-1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value("build-1"))
                .andExpect(jsonPath("$.evalResultId").value("eval-1"))
                .andExpect(jsonPath("$.status").value("EVAL_PASSED"));
    }

    @Test
    void eval_endpoint_attaches_eval_result() throws Exception {
        when(indexBuilds.attachEvalResult("build-1", "eval-1", NOW))
                .thenReturn(build("build-1", IndexBuildStatus.EVAL_PASSED, "eval-1", false));

        mockMvc.perform(post("/api/v1/rag/index-builds/build-1/eval")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"evalResultId":"eval-1"}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.evalResultId").value("eval-1"))
                .andExpect(jsonPath("$.status").value("EVAL_PASSED"));
    }

    @Test
    void promote_endpoint_maps_gate_failure_to_conflict() throws Exception {
        when(indexBuilds.promote("build-1", NOW))
                .thenThrow(new IllegalStateException("index build must be EVAL_PASSED before promotion"));

        mockMvc.perform(post("/api/v1/rag/index-builds/build-1/promote"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code").value("CONFLICT"));
    }

    @Test
    void rollback_endpoint_records_current_and_previous_versions() throws Exception {
        when(indexBuilds.rollback(eq("build-1"), any(RollbackIndexBuildCommand.class), eq(NOW)))
                .thenReturn(rolledBackBuild());

        mockMvc.perform(post("/api/v1/rag/index-builds/build-1/rollback")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"currentIndexVersion":"idx-candidate","previousIndexVersion":"idx-active"}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentIndexVersion").value("idx-candidate"))
                .andExpect(jsonPath("$.previousIndexVersion").value("idx-active"))
                .andExpect(jsonPath("$.status").value("ROLLED_BACK"));

        ArgumentCaptor<RollbackIndexBuildCommand> captor = ArgumentCaptor.forClass(RollbackIndexBuildCommand.class);
        verify(indexBuilds).rollback(eq("build-1"), captor.capture(), eq(NOW));
        assertThat(captor.getValue().currentIndexVersion()).isEqualTo("idx-candidate");
        assertThat(captor.getValue().previousIndexVersion()).isEqualTo("idx-active");
    }

    private static IndexBuildJpaEntity build(String id,
                                             IndexBuildStatus status,
                                             String evalResultId,
                                             boolean active) {
        return new IndexBuildJpaEntity(
                id,
                "idx-candidate",
                "idx-candidate",
                "idx-active",
                status,
                active,
                "{\"pdf\":\"pdf-extract-v1\"}",
                7,
                "{\"gate\":\"seed\"}",
                evalResultId,
                "[]",
                null,
                active ? NOW : null,
                null,
                NOW,
                NOW);
    }

    private static IndexBuildJpaEntity rolledBackBuild() {
        return new IndexBuildJpaEntity(
                "build-1",
                "idx-candidate",
                "idx-candidate",
                "idx-active",
                IndexBuildStatus.ROLLED_BACK,
                false,
                "{}",
                7,
                null,
                "eval-1",
                "[]",
                null,
                NOW.minusSeconds(60),
                NOW,
                NOW.minusSeconds(120),
                NOW);
    }
}
