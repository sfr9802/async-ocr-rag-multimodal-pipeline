package com.aipipeline.coreapi.job.adapter.in.web;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.job.adapter.in.web.dto.CreateJobRequest;
import com.aipipeline.coreapi.job.adapter.in.web.dto.JobResponses;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase.CreateJobCommand;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase.StagedInputArtifact;
import com.aipipeline.coreapi.job.application.service.JobSubmissionValidator;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.aipipeline.coreapi.job.domain.JobId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
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
 *   - Capability/input validation is delegated to
 *     {@link JobSubmissionValidator}, which runs BEFORE any storage.store()
 *     or createAndEnqueue() work. If it throws, the pipeline is left in a
 *     pristine state (no Artifact row, no Job row, no Redis dispatch).
 *   - Authentication is deferred to a later phase; the user explicitly
 *     scoped this out.
 */
@RestController
@RequestMapping("/api/v1/jobs")
public class JobController {

    private static final Logger log = LoggerFactory.getLogger(JobController.class);

    private final JobManagementUseCase jobManagement;
    private final ArtifactStoragePort storage;

    public JobController(JobManagementUseCase jobManagement, ArtifactStoragePort storage) {
        this.jobManagement = jobManagement;
        this.storage = storage;
    }

    /**
     * Text-based submission. Stages the prompt text as an INPUT_TEXT artifact.
     *
     * Contract:
     *   - capability: required (CAPABILITY_REQUIRED / UNKNOWN_CAPABILITY)
     *   - text:       required, content rules depend on capability
     *                 (TEXT_REQUIRED for RAG when blank/null)
     *   - OCR / MULTIMODAL on this endpoint → FILE_REQUIRED (wrong endpoint)
     */
    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<JobResponses.JobCreated> createTextJob(@RequestBody CreateJobRequest body) {
        // Parse + validate BEFORE any side effects. If this throws, nothing
        // has been persisted or enqueued yet.
        JobCapability capability = JobSubmissionValidator.parseCapability(body.capability());
        JobSubmissionValidator.validateTextSubmission(capability, body.text());

        List<ArtifactStoragePort.StoredObject> storedObjects = new ArrayList<>();
        try {
            byte[] bytes = body.text().getBytes(StandardCharsets.UTF_8);
            var stored = storage.store(
                    JobId.generate(),  // provisional prefix
                    ArtifactType.INPUT_TEXT,
                    "prompt.txt",
                    "text/plain; charset=utf-8",
                    new ByteArrayInputStream(bytes),
                    bytes.length);
            storedObjects.add(stored);

            StagedInputArtifact staged = new StagedInputArtifact(
                    ArtifactType.INPUT_TEXT,
                    stored.storageUri(),
                    "text/plain; charset=utf-8",
                    stored.sizeBytes(),
                    stored.checksumSha256());

            var result = jobManagement.createAndEnqueue(new CreateJobCommand(capability, List.of(staged)));
            return ResponseEntity.accepted().body(JobResponses.JobCreated.from(result.job(), result.inputArtifacts()));
        } catch (RuntimeException ex) {
            cleanupStoredObjects(storedObjects);
            throw ex;
        }
    }

    /**
     * File-based submission via multipart/form-data. Fields:
     *   - capability : form field (required)
     *   - file       : the binary upload (required, non-empty; FILE_REQUIRED /
     *                  FILE_EMPTY otherwise). The {@code required} flag is
     *                  left at false on the annotation so missing files reach
     *                  {@link JobSubmissionValidator} and produce the typed
     *                  FILE_REQUIRED error code instead of Spring's generic
     *                  {@code MissingServletRequestParameter}.
     *   - text       : optional accompanying user question/prompt. When
     *                  present, it is staged as a second INPUT_TEXT artifact
     *                  alongside the INPUT_FILE. The MULTIMODAL capability
     *                  uses this as the user question for its fusion step;
     *                  OCR / MOCK capabilities ignore it (they only ever
     *                  pick up INPUT_FILE / INPUT_TEXT respectively). RAG
     *                  on this endpoint requires the text field to be
     *                  non-blank — TEXT_REQUIRED otherwise.
     *
     * Per-capability file-type rules (OCR + MULTIMODAL): PNG, JPEG, PDF only.
     * Anything else → UNSUPPORTED_FILE_TYPE.
     */
    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<JobResponses.JobCreated> createFileJob(
            @RequestParam(value = "capability", required = false) String capabilityRaw,
            @RequestParam(value = "file", required = false) MultipartFile file,
            @RequestParam(value = "text", required = false) String text
    ) throws IOException {
        // Parse + validate BEFORE staging any bytes. If either call throws,
        // nothing has been stored or enqueued — the pipeline is left in a
        // pristine state.
        JobCapability capability = JobSubmissionValidator.parseCapability(capabilityRaw);
        JobSubmissionValidator.validateFileSubmission(capability, file, text);

        List<StagedInputArtifact> stagedInputs = new ArrayList<>();
        List<ArtifactStoragePort.StoredObject> storedObjects = new ArrayList<>();
        try {
            var storedFile = storage.store(
                    JobId.generate(),
                    ArtifactType.INPUT_FILE,
                    file.getOriginalFilename(),
                    file.getContentType(),
                    file.getInputStream(),
                    file.getSize());
            storedObjects.add(storedFile);
            stagedInputs.add(new StagedInputArtifact(
                    ArtifactType.INPUT_FILE,
                    storedFile.storageUri(),
                    file.getContentType(),
                    storedFile.sizeBytes(),
                    storedFile.checksumSha256()));

            if (text != null && !text.isBlank()) {
                byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);
                var storedText = storage.store(
                        JobId.generate(),
                        ArtifactType.INPUT_TEXT,
                        "prompt.txt",
                        "text/plain; charset=utf-8",
                        new ByteArrayInputStream(textBytes),
                        textBytes.length);
                storedObjects.add(storedText);

                stagedInputs.add(new StagedInputArtifact(
                        ArtifactType.INPUT_TEXT,
                        storedText.storageUri(),
                        "text/plain; charset=utf-8",
                        storedText.sizeBytes(),
                        storedText.checksumSha256()));
            }

            var result = jobManagement.createAndEnqueue(new CreateJobCommand(capability, stagedInputs));
            return ResponseEntity.accepted().body(JobResponses.JobCreated.from(result.job(), result.inputArtifacts()));
        } catch (IOException | RuntimeException ex) {
            cleanupStoredObjects(storedObjects);
            throw ex;
        }
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

    private void cleanupStoredObjects(List<ArtifactStoragePort.StoredObject> storedObjects) {
        for (ArtifactStoragePort.StoredObject stored : storedObjects) {
            try {
                storage.delete(stored.storageUri());
            } catch (RuntimeException cleanupEx) {
                log.warn(
                        "Failed to cleanup staged input artifact {} after job creation failure",
                        stored.storageUri(), cleanupEx);
            }
        }
    }
}
