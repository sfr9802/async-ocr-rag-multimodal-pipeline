package com.aipipeline.coreapi.catalog.adapter.in.web;

import com.aipipeline.coreapi.catalog.adapter.out.persistence.IndexBuildJpaEntity;
import com.aipipeline.coreapi.catalog.application.service.RagIndexBuildService;
import com.aipipeline.coreapi.catalog.application.service.RagIndexBuildService.CreateIndexBuildCommand;
import com.aipipeline.coreapi.catalog.application.service.RagIndexBuildService.RollbackIndexBuildCommand;
import com.aipipeline.coreapi.common.TimeProvider;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.time.Instant;

@RestController
@RequestMapping("/api/v1/rag/index-builds")
public class IndexBuildController {

    private final RagIndexBuildService indexBuilds;
    private final TimeProvider timeProvider;

    public IndexBuildController(RagIndexBuildService indexBuilds,
                                TimeProvider timeProvider) {
        this.indexBuilds = indexBuilds;
        this.timeProvider = timeProvider;
    }

    @PostMapping
    public ResponseEntity<IndexBuildResponse> create(
            @RequestBody(required = false) CreateIndexBuildRequest body
    ) {
        CreateIndexBuildRequest safeBody = body == null ? CreateIndexBuildRequest.empty() : body;
        IndexBuildJpaEntity build = indexBuilds.createIndexBuild(safeBody.toCommand(), timeProvider.now());
        return ResponseEntity.status(HttpStatus.CREATED).body(IndexBuildResponse.from(build));
    }

    @GetMapping("/{id}")
    public ResponseEntity<IndexBuildResponse> get(@PathVariable String id) {
        IndexBuildJpaEntity build = indexBuilds.findIndexBuild(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "index build not found"));
        return ResponseEntity.ok(IndexBuildResponse.from(build));
    }

    @PostMapping("/{id}/eval")
    public ResponseEntity<IndexBuildResponse> eval(
            @PathVariable String id,
            @Valid @RequestBody AttachEvalResultRequest body
    ) {
        IndexBuildJpaEntity build = indexBuilds.attachEvalResult(id, body.evalResultId(), timeProvider.now());
        return ResponseEntity.ok(IndexBuildResponse.from(build));
    }

    @PostMapping("/{id}/promote")
    public ResponseEntity<IndexBuildResponse> promote(@PathVariable String id) {
        IndexBuildJpaEntity build = indexBuilds.promote(id, timeProvider.now());
        return ResponseEntity.ok(IndexBuildResponse.from(build));
    }

    @PostMapping("/{id}/rollback")
    public ResponseEntity<IndexBuildResponse> rollback(
            @PathVariable String id,
            @RequestBody(required = false) RollbackIndexBuildRequest body
    ) {
        RollbackIndexBuildRequest safeBody = body == null ? RollbackIndexBuildRequest.empty() : body;
        IndexBuildJpaEntity build = indexBuilds.rollback(id, safeBody.toCommand(), timeProvider.now());
        return ResponseEntity.ok(IndexBuildResponse.from(build));
    }

    public record CreateIndexBuildRequest(
            String id,
            String indexVersion,
            String candidateIndexVersion,
            String previousIndexVersion,
            String parserVersionsJson,
            Integer chunkCount,
            String qualityGateJson
    ) {
        static CreateIndexBuildRequest empty() {
            return new CreateIndexBuildRequest(null, null, null, null, null, null, null);
        }

        CreateIndexBuildCommand toCommand() {
            return new CreateIndexBuildCommand(
                    id,
                    indexVersion,
                    candidateIndexVersion,
                    previousIndexVersion,
                    parserVersionsJson,
                    chunkCount,
                    qualityGateJson);
        }
    }

    public record AttachEvalResultRequest(
            @NotBlank String evalResultId
    ) {}

    public record RollbackIndexBuildRequest(
            String currentIndexVersion,
            String previousIndexVersion
    ) {
        static RollbackIndexBuildRequest empty() {
            return new RollbackIndexBuildRequest(null, null);
        }

        RollbackIndexBuildCommand toCommand() {
            return new RollbackIndexBuildCommand(currentIndexVersion, previousIndexVersion);
        }
    }

    public record IndexBuildResponse(
            String id,
            String indexVersion,
            String candidateIndexVersion,
            String currentIndexVersion,
            String previousIndexVersion,
            String evalResultId,
            String status,
            boolean active,
            int chunkCount,
            String parserVersionsJson,
            String qualityGateJson,
            String failureReasonJson,
            Instant builtAt,
            Instant promotedAt,
            Instant rolledBackAt,
            Instant createdAt,
            Instant updatedAt
    ) {
        static IndexBuildResponse from(IndexBuildJpaEntity build) {
            return new IndexBuildResponse(
                    build.getId(),
                    build.getIndexVersion(),
                    build.getCandidateIndexVersion(),
                    build.getCandidateIndexVersion(),
                    build.getPreviousIndexVersion(),
                    build.getEvalResultId(),
                    build.getStatus().name(),
                    build.isActive(),
                    build.getChunkCount(),
                    build.getParserVersionsJson(),
                    build.getQualityGateJson(),
                    build.getFailureReasonJson(),
                    build.getBuiltAt(),
                    build.getPromotedAt(),
                    build.getRolledBackAt(),
                    build.getCreatedAt(),
                    build.getUpdatedAt());
        }
    }
}
