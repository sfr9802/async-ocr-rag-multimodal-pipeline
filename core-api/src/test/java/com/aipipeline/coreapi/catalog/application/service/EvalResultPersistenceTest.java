package com.aipipeline.coreapi.catalog.application.service;

import com.aipipeline.coreapi.catalog.adapter.out.persistence.EvalResultJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.EvalResultJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.IndexBuildJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.IndexBuildJpaRepository;
import com.aipipeline.coreapi.catalog.domain.EvalResultStatus;
import com.aipipeline.coreapi.catalog.domain.IndexBuildStatus;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class EvalResultPersistenceTest {

    private static final Instant NOW = Instant.parse("2026-05-03T00:00:00Z");

    private final IndexBuildJpaRepository indexBuilds = mock(IndexBuildJpaRepository.class);
    private final EvalResultJpaRepository evalResults = mock(EvalResultJpaRepository.class);

    @Test
    void create_index_build_fills_v7_v8_required_fields() {
        when(indexBuilds.existsById("idx-candidate")).thenReturn(false);
        when(indexBuilds.existsByIndexVersion("idx-candidate")).thenReturn(false);
        when(indexBuilds.save(any(IndexBuildJpaEntity.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        IndexBuildJpaEntity build = service().createIndexBuild(
                new RagIndexBuildService.CreateIndexBuildCommand(
                        null,
                        "idx-candidate",
                        null,
                        "idx-active",
                        null,
                        null,
                        null),
                NOW);

        assertThat(build.getId()).isEqualTo("idx-candidate");
        assertThat(build.getIndexVersion()).isEqualTo("idx-candidate");
        assertThat(build.getCandidateIndexVersion()).isEqualTo("idx-candidate");
        assertThat(build.getPreviousIndexVersion()).isEqualTo("idx-active");
        assertThat(build.getStatus()).isEqualTo(IndexBuildStatus.CREATED);
        assertThat(build.getParserVersionsJson()).isEqualTo("{}");
        assertThat(build.getFailureReasonJson()).isEqualTo("[]");
        assertThat(build.getCreatedAt()).isEqualTo(NOW);
        assertThat(build.getUpdatedAt()).isEqualTo(NOW);
    }

    @Test
    void eval_result_passed_status_marks_build_eval_passed() {
        IndexBuildJpaEntity build = build();
        when(indexBuilds.findById("build-1")).thenReturn(Optional.of(build));
        when(evalResults.findById("eval-1")).thenReturn(Optional.of(eval("eval-1", EvalResultStatus.PASSED, true)));
        when(indexBuilds.save(any(IndexBuildJpaEntity.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        IndexBuildJpaEntity evaluated = service().attachEvalResult("build-1", "eval-1", NOW);

        assertThat(evaluated.getEvalResultId()).isEqualTo("eval-1");
        assertThat(evaluated.getStatus()).isEqualTo(IndexBuildStatus.EVAL_PASSED);
        assertThat(evaluated.getUpdatedAt()).isEqualTo(NOW);
    }

    @Test
    void eval_result_failed_status_marks_build_eval_failed() {
        IndexBuildJpaEntity build = build();
        when(indexBuilds.findById("build-1")).thenReturn(Optional.of(build));
        when(evalResults.findById("eval-2")).thenReturn(Optional.of(eval("eval-2", EvalResultStatus.FAILED, false)));
        when(indexBuilds.save(any(IndexBuildJpaEntity.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        IndexBuildJpaEntity evaluated = service().attachEvalResult("build-1", "eval-2", NOW);

        assertThat(evaluated.getEvalResultId()).isEqualTo("eval-2");
        assertThat(evaluated.getStatus()).isEqualTo(IndexBuildStatus.EVAL_FAILED);
    }

    private RagIndexBuildService service() {
        return new RagIndexBuildService(indexBuilds, evalResults);
    }

    private static IndexBuildJpaEntity build() {
        return new IndexBuildJpaEntity(
                "build-1",
                "idx-candidate",
                "idx-candidate",
                "idx-active",
                IndexBuildStatus.CREATED,
                false,
                "{}",
                0,
                null,
                null,
                "[]",
                null,
                null,
                null,
                NOW.minusSeconds(60),
                NOW.minusSeconds(60));
    }

    private static EvalResultJpaEntity eval(String id, EvalResultStatus status, boolean passed) {
        return new EvalResultJpaEntity(
                id,
                "dataset-1",
                "dataset-1",
                "idx-candidate",
                "idx-candidate",
                "idx-active",
                "{\"hit_at_10\":1.0}",
                passed,
                status,
                "{}",
                "[]",
                null,
                null,
                NOW);
    }
}
