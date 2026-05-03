package com.aipipeline.coreapi.catalog.application.service;

import com.aipipeline.coreapi.catalog.adapter.out.persistence.EvalResultJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.EvalResultJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.IndexBuildJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.IndexBuildJpaRepository;
import com.aipipeline.coreapi.catalog.domain.EvalResultStatus;
import com.aipipeline.coreapi.catalog.domain.IndexBuildStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.Optional;
import java.util.UUID;

@Service
public class RagIndexBuildService {

    private final IndexBuildJpaRepository indexBuilds;
    private final EvalResultJpaRepository evalResults;

    public RagIndexBuildService(IndexBuildJpaRepository indexBuilds,
                                EvalResultJpaRepository evalResults) {
        this.indexBuilds = indexBuilds;
        this.evalResults = evalResults;
    }

    @Transactional
    public IndexBuildJpaEntity createIndexBuild(CreateIndexBuildCommand command, Instant now) {
        CreateIndexBuildCommand safeCommand = command == null ? CreateIndexBuildCommand.empty() : command;
        if (safeCommand.indexVersion() != null
                && !safeCommand.indexVersion().isBlank()
                && safeCommand.candidateIndexVersion() != null
                && !safeCommand.candidateIndexVersion().isBlank()
                && !safeCommand.indexVersion().equals(safeCommand.candidateIndexVersion())) {
            throw new IllegalArgumentException("indexVersion and candidateIndexVersion must match");
        }
        String generated = "candidate-" + UUID.randomUUID();
        String candidateIndexVersion = firstNonBlank(
                safeCommand.candidateIndexVersion(),
                safeCommand.indexVersion(),
                generated);
        String indexVersion = firstNonBlank(safeCommand.indexVersion(), candidateIndexVersion);
        String id = firstNonBlank(safeCommand.id(), indexVersion);
        if (indexBuilds.existsById(id)) {
            throw new IllegalArgumentException("index build already exists: " + id);
        }
        if (indexBuilds.existsByIndexVersion(indexVersion)) {
            throw new IllegalArgumentException("index version already exists: " + indexVersion);
        }

        return indexBuilds.save(new IndexBuildJpaEntity(
                id,
                indexVersion,
                candidateIndexVersion,
                blankToNull(safeCommand.previousIndexVersion()),
                IndexBuildStatus.CREATED,
                false,
                defaultJsonObject(safeCommand.parserVersionsJson()),
                safeCommand.chunkCount() == null ? 0 : safeCommand.chunkCount(),
                blankToNull(safeCommand.qualityGateJson()),
                null,
                "[]",
                null,
                null,
                null,
                now,
                now));
    }

    @Transactional(readOnly = true)
    public Optional<IndexBuildJpaEntity> findIndexBuild(String id) {
        return indexBuilds.findById(id);
    }

    @Transactional
    public IndexBuildJpaEntity attachEvalResult(String id, String evalResultId, Instant now) {
        if (evalResultId == null || evalResultId.isBlank()) {
            throw new IllegalArgumentException("evalResultId must not be blank");
        }
        IndexBuildJpaEntity build = findRequiredIndexBuild(id);
        EvalResultJpaEntity evalResult = evalResults.findById(evalResultId)
                .orElseThrow(() -> new IllegalArgumentException("Unknown evalResultId: " + evalResultId));
        requireEvalResultMatchesBuild(build, evalResult);
        IndexBuildStatus nextStatus = evalResult.effectiveStatus() == EvalResultStatus.PASSED
                ? IndexBuildStatus.EVAL_PASSED
                : IndexBuildStatus.EVAL_FAILED;
        build.attachEvalResult(evalResult.getId(), nextStatus, now);
        return indexBuilds.save(build);
    }

    @Transactional
    public IndexBuildJpaEntity promote(String id, Instant now) {
        IndexBuildJpaEntity build = findRequiredIndexBuild(id);
        if (build.getStatus() != IndexBuildStatus.EVAL_PASSED) {
            throw new IllegalStateException("index build must be EVAL_PASSED before promotion");
        }
        String evalResultId = build.getEvalResultId();
        if (evalResultId == null || evalResultId.isBlank()) {
            throw new IllegalStateException("index build has no eval result");
        }
        EvalResultJpaEntity evalResult = evalResults.findById(evalResultId)
                .orElseThrow(() -> new IllegalStateException("index build eval result is missing"));
        requireEvalResultMatchesBuild(build, evalResult);
        if (evalResult.effectiveStatus() != EvalResultStatus.PASSED) {
            throw new IllegalStateException("index build eval result must be PASSED before promotion");
        }

        Optional<IndexBuildJpaEntity> activeBuild = indexBuilds.findFirstByActiveTrue()
                .filter(active -> !active.getId().equals(build.getId()));
        String previousIndexVersion = activeBuild
                .map(IndexBuildJpaEntity::getIndexVersion)
                .orElse(build.getPreviousIndexVersion());
        activeBuild.ifPresent(active -> {
            active.deactivate(now);
            indexBuilds.save(active);
        });

        build.markPromoted(previousIndexVersion, now);
        return indexBuilds.save(build);
    }

    @Transactional
    public IndexBuildJpaEntity rollback(String id, RollbackIndexBuildCommand command, Instant now) {
        IndexBuildJpaEntity build = findRequiredIndexBuild(id);
        RollbackIndexBuildCommand safeCommand = command == null ? RollbackIndexBuildCommand.empty() : command;
        String currentIndexVersion = firstNonBlank(
                safeCommand.currentIndexVersion(),
                build.getCandidateIndexVersion(),
                build.getIndexVersion());
        String previousIndexVersion = firstNonBlank(
                safeCommand.previousIndexVersion(),
                build.getPreviousIndexVersion());
        build.markRolledBack(currentIndexVersion, previousIndexVersion, now);
        return indexBuilds.save(build);
    }

    private IndexBuildJpaEntity findRequiredIndexBuild(String id) {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("index build id must not be blank");
        }
        return indexBuilds.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Unknown index build id: " + id));
    }

    private static String firstNonBlank(String... values) {
        if (values == null) {
            return null;
        }
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }

    private static String blankToNull(String value) {
        return value == null || value.isBlank() ? null : value;
    }

    private static String defaultJsonObject(String value) {
        return value == null || value.isBlank() ? "{}" : value;
    }

    private static void requireEvalResultMatchesBuild(IndexBuildJpaEntity build, EvalResultJpaEntity evalResult) {
        if (!equalsNonBlank(evalResult.getIndexVersion(), build.getIndexVersion())) {
            throw new IllegalStateException("eval result index version must match index build");
        }
        if (!equalsNonBlank(evalResult.getCandidateIndexVersion(), build.getCandidateIndexVersion())) {
            throw new IllegalStateException("eval result candidate index version must match index build");
        }
    }

    private static boolean equalsNonBlank(String left, String right) {
        return left != null && !left.isBlank() && left.equals(right);
    }

    public record CreateIndexBuildCommand(
            String id,
            String indexVersion,
            String candidateIndexVersion,
            String previousIndexVersion,
            String parserVersionsJson,
            Integer chunkCount,
            String qualityGateJson
    ) {
        static CreateIndexBuildCommand empty() {
            return new CreateIndexBuildCommand(null, null, null, null, null, null, null);
        }
    }

    public record RollbackIndexBuildCommand(
            String currentIndexVersion,
            String previousIndexVersion
    ) {
        static RollbackIndexBuildCommand empty() {
            return new RollbackIndexBuildCommand(null, null);
        }
    }
}
