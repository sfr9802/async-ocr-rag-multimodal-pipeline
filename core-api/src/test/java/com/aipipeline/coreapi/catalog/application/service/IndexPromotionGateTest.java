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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class IndexPromotionGateTest {

    private static final Instant NOW = Instant.parse("2026-05-03T00:00:00Z");

    private final IndexBuildJpaRepository indexBuilds = mock(IndexBuildJpaRepository.class);
    private final EvalResultJpaRepository evalResults = mock(EvalResultJpaRepository.class);

    @Test
    void create_index_build_rejects_distinct_index_and_candidate_versions() {
        assertThatThrownBy(() -> service().createIndexBuild(
                new RagIndexBuildService.CreateIndexBuildCommand(
                        null,
                        "idx-index",
                        "idx-candidate",
                        null,
                        null,
                        null,
                        null),
                NOW))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must match");
    }

    @Test
    void promote_requires_build_to_be_eval_passed() {
        IndexBuildJpaEntity build = build("build-1", "idx-candidate", IndexBuildStatus.EVAL_FAILED, "eval-1", false);
        when(indexBuilds.findById("build-1")).thenReturn(Optional.of(build));

        assertThatThrownBy(() -> service().promote("build-1", NOW))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("EVAL_PASSED");
    }

    @Test
    void promote_requires_attached_eval_result_to_be_passed() {
        IndexBuildJpaEntity build = build("build-1", "idx-candidate", IndexBuildStatus.EVAL_PASSED, "eval-1", false);
        when(indexBuilds.findById("build-1")).thenReturn(Optional.of(build));
        when(evalResults.findById("eval-1"))
                .thenReturn(Optional.of(eval("eval-1", EvalResultStatus.FAILED, false)));

        assertThatThrownBy(() -> service().promote("build-1", NOW))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("PASSED");
    }

    @Test
    void attach_eval_result_requires_matching_candidate_index_version() {
        IndexBuildJpaEntity build = build("build-1", "idx-candidate", IndexBuildStatus.CREATED, null, false);
        when(indexBuilds.findById("build-1")).thenReturn(Optional.of(build));
        when(evalResults.findById("eval-other"))
                .thenReturn(Optional.of(eval(
                        "eval-other",
                        EvalResultStatus.PASSED,
                        true,
                        "idx-other")));

        assertThatThrownBy(() -> service().attachEvalResult("build-1", "eval-other", NOW))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("index version");
    }

    @Test
    void promote_requires_attached_eval_result_index_version_to_still_match() {
        IndexBuildJpaEntity build = build("build-1", "idx-candidate", IndexBuildStatus.EVAL_PASSED, "eval-other", false);
        when(indexBuilds.findById("build-1")).thenReturn(Optional.of(build));
        when(evalResults.findById("eval-other"))
                .thenReturn(Optional.of(eval(
                        "eval-other",
                        EvalResultStatus.PASSED,
                        true,
                        "idx-other")));

        assertThatThrownBy(() -> service().promote("build-1", NOW))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("index version");
    }

    @Test
    void promote_activates_candidate_and_records_previous_active_index() {
        IndexBuildJpaEntity active = build("build-active", "idx-active", IndexBuildStatus.PROMOTED, "eval-active", true);
        IndexBuildJpaEntity candidate = build("build-1", "idx-candidate", IndexBuildStatus.EVAL_PASSED, "eval-1", false);
        when(indexBuilds.findById("build-1")).thenReturn(Optional.of(candidate));
        when(evalResults.findById("eval-1"))
                .thenReturn(Optional.of(eval("eval-1", EvalResultStatus.PASSED, true)));
        when(indexBuilds.findFirstByActiveTrue()).thenReturn(Optional.of(active));
        when(indexBuilds.save(any(IndexBuildJpaEntity.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        IndexBuildJpaEntity promoted = service().promote("build-1", NOW);

        assertThat(active.isActive()).isFalse();
        assertThat(promoted.isActive()).isTrue();
        assertThat(promoted.getStatus()).isEqualTo(IndexBuildStatus.PROMOTED);
        assertThat(promoted.getPreviousIndexVersion()).isEqualTo("idx-active");
        assertThat(promoted.getPromotedAt()).isEqualTo(NOW);
    }

    @Test
    void rollback_records_current_and_previous_index_versions() {
        IndexBuildJpaEntity build = build("build-1", "idx-candidate", IndexBuildStatus.PROMOTED, "eval-1", true);
        when(indexBuilds.findById("build-1")).thenReturn(Optional.of(build));
        when(indexBuilds.save(any(IndexBuildJpaEntity.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        IndexBuildJpaEntity rolledBack = service().rollback(
                "build-1",
                new RagIndexBuildService.RollbackIndexBuildCommand("idx-candidate", "idx-active"),
                NOW);

        assertThat(rolledBack.getCandidateIndexVersion()).isEqualTo("idx-candidate");
        assertThat(rolledBack.getPreviousIndexVersion()).isEqualTo("idx-active");
        assertThat(rolledBack.getStatus()).isEqualTo(IndexBuildStatus.ROLLED_BACK);
        assertThat(rolledBack.isActive()).isFalse();
        assertThat(rolledBack.getRolledBackAt()).isEqualTo(NOW);
    }

    private RagIndexBuildService service() {
        return new RagIndexBuildService(indexBuilds, evalResults);
    }

    private static IndexBuildJpaEntity build(String id,
                                             String indexVersion,
                                             IndexBuildStatus status,
                                             String evalResultId,
                                             boolean active) {
        return new IndexBuildJpaEntity(
                id,
                indexVersion,
                indexVersion,
                null,
                status,
                active,
                "{}",
                10,
                null,
                evalResultId,
                "[]",
                null,
                active ? NOW.minusSeconds(60) : null,
                null,
                NOW.minusSeconds(120),
                NOW.minusSeconds(120));
    }

    private static EvalResultJpaEntity eval(String id, EvalResultStatus status, boolean passed) {
        return eval(id, status, passed, "idx-candidate");
    }

    private static EvalResultJpaEntity eval(String id, EvalResultStatus status, boolean passed, String indexVersion) {
        return new EvalResultJpaEntity(
                id,
                "dataset-1",
                "dataset-1",
                indexVersion,
                indexVersion,
                "idx-active",
                "{}",
                passed,
                status,
                "{}",
                "[]",
                null,
                null,
                NOW.minusSeconds(30));
    }
}
