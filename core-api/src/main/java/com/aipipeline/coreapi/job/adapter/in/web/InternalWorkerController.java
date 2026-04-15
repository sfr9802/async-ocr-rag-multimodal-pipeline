package com.aipipeline.coreapi.job.adapter.in.web;

import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.job.adapter.in.web.dto.WorkerCallbackRequest;
import com.aipipeline.coreapi.job.adapter.in.web.dto.WorkerCallbackResponse;
import com.aipipeline.coreapi.job.adapter.in.web.dto.WorkerClaimRequest;
import com.aipipeline.coreapi.job.adapter.in.web.dto.WorkerClaimResponse;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase.CallbackCommand;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase.CallbackOutcome;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase.ClaimCommand;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase.OutputArtifactRef;
import com.aipipeline.coreapi.job.domain.JobId;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * Worker-facing endpoints. These are the mirror of the worker's
 * core-api client and are deliberately placed under /api/internal to make
 * it obvious they are NOT for public clients.
 *
 * Phase 1 has no auth on this path — a real deployment would gate it
 * behind a shared secret or mTLS. Keep that assumption documented.
 */
@RestController
@RequestMapping("/api/internal/jobs")
public class InternalWorkerController {

    private final JobExecutionUseCase execution;

    public InternalWorkerController(JobExecutionUseCase execution) {
        this.execution = execution;
    }

    @PostMapping("/claim")
    public ResponseEntity<WorkerClaimResponse> claim(@Valid @RequestBody WorkerClaimRequest body) {
        var result = execution.claim(new ClaimCommand(
                JobId.of(body.jobId()),
                body.workerClaimToken(),
                body.attemptNo()));
        return ResponseEntity.ok(WorkerClaimResponse.from(result));
    }

    @PostMapping("/callback")
    public ResponseEntity<WorkerCallbackResponse> callback(@Valid @RequestBody WorkerCallbackRequest body) {
        List<OutputArtifactRef> outputs = body.outputArtifacts() == null
                ? List.of()
                : body.outputArtifacts().stream()
                .map(o -> new OutputArtifactRef(
                        ArtifactType.fromString(o.type()),
                        o.storageUri(),
                        o.contentType(),
                        o.sizeBytes(),
                        o.checksumSha256()))
                .toList();

        var result = execution.handleCallback(new CallbackCommand(
                JobId.of(body.jobId()),
                body.callbackId(),
                body.workerClaimToken(),
                body.outcome() == WorkerCallbackRequest.Outcome.SUCCEEDED
                        ? CallbackOutcome.SUCCEEDED
                        : CallbackOutcome.FAILED,
                body.errorCode(),
                body.errorMessage(),
                outputs));
        return ResponseEntity.ok(WorkerCallbackResponse.from(result));
    }
}
