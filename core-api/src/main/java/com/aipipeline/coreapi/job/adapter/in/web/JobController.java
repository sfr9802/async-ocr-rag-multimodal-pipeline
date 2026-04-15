package com.aipipeline.coreapi.job.adapter.in.web;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.job.adapter.in.web.dto.CreateJobRequest;
import com.aipipeline.coreapi.job.adapter.in.web.dto.JobResponses;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase.CreateJobCommand;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase.StagedInputArtifact;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.aipipeline.coreapi.job.domain.JobId;
import jakarta.validation.Valid;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Client-facing job endpoints.
 *
 * Design notes:
 *   - Two create endpoints: text (JSON body) and file (multipart). Both
 *     stage the bytes through the storage port, then delegate to the
 *     application use case to persist and enqueue.
 *   - The controller is intentionally thin — it only maps HTTP to/from the
 *     use-case boundary.
 *   - Authentication is deferred to a later phase; the user explicitly
 *     scoped this out.
 */
@RestController
@RequestMapping("/api/v1/jobs")
public class JobController {

    private final JobManagementUseCase jobManagement;
    private final ArtifactStoragePort storage;

    public JobController(JobManagementUseCase jobManagement, ArtifactStoragePort storage) {
        this.jobManagement = jobManagement;
        this.storage = storage;
    }

    /**
     * Text-based submission. Stages the prompt text as an INPUT_TEXT artifact.
     */
    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<JobResponses.JobCreated> createTextJob(@Valid @RequestBody CreateJobRequest body) {
        JobCapability capability = JobCapability.fromString(body.capability());
        byte[] bytes = body.text().getBytes(StandardCharsets.UTF_8);
        var stored = storage.store(
                com.aipipeline.coreapi.job.domain.JobId.generate(),  // provisional prefix
                ArtifactType.INPUT_TEXT,
                "prompt.txt",
                "text/plain; charset=utf-8",
                new ByteArrayInputStream(bytes),
                bytes.length);

        StagedInputArtifact staged = new StagedInputArtifact(
                ArtifactType.INPUT_TEXT,
                stored.storageUri(),
                "text/plain; charset=utf-8",
                stored.sizeBytes(),
                stored.checksumSha256());

        var result = jobManagement.createAndEnqueue(new CreateJobCommand(capability, List.of(staged)));
        return ResponseEntity.accepted().body(JobResponses.JobCreated.from(result.job(), result.inputArtifacts()));
    }

    /**
     * File-based submission via multipart/form-data. Fields:
     *   - capability : form field
     *   - file       : the binary upload
     */
    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<JobResponses.JobCreated> createFileJob(
            @RequestParam("capability") String capabilityRaw,
            @RequestParam("file") MultipartFile file
    ) throws IOException {
        JobCapability capability = JobCapability.fromString(capabilityRaw);
        var stored = storage.store(
                com.aipipeline.coreapi.job.domain.JobId.generate(),
                ArtifactType.INPUT_FILE,
                file.getOriginalFilename(),
                file.getContentType(),
                file.getInputStream(),
                file.getSize());

        StagedInputArtifact staged = new StagedInputArtifact(
                ArtifactType.INPUT_FILE,
                stored.storageUri(),
                file.getContentType(),
                stored.sizeBytes(),
                stored.checksumSha256());

        var result = jobManagement.createAndEnqueue(new CreateJobCommand(capability, List.of(staged)));
        return ResponseEntity.accepted().body(JobResponses.JobCreated.from(result.job(), result.inputArtifacts()));
    }

    @GetMapping("/{jobId}")
    public ResponseEntity<JobResponses.JobView> getJob(@PathVariable String jobId) {
        return jobManagement.findJob(JobId.of(jobId))
                .map(JobResponses.JobView::from)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    @GetMapping("/{jobId}/result")
    public ResponseEntity<JobResponses.JobResult> getJobResult(@PathVariable String jobId) {
        return jobManagement.findJobWithArtifacts(JobId.of(jobId))
                .map(view -> JobResponses.JobResult.from(view.job(), view.artifacts()))
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }
}
