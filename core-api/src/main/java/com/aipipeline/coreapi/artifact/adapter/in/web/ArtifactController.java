package com.aipipeline.coreapi.artifact.adapter.in.web;

import com.aipipeline.coreapi.artifact.application.port.in.ArtifactAccessUseCase;
import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.ArtifactId;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.job.domain.JobId;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

/**
 * Controller covering both sides of artifact access:
 *
 *   - {@code GET /api/v1/artifacts/{id}/content}
 *        public download for end users
 *   - {@code POST /api/internal/artifacts}
 *        internal upload endpoint the worker uses to push result bytes
 *        before calling back. This is the "write" side of the phase-1
 *        storage port — treat it as analogous to a presigned PUT URL in
 *        production.
 *
 * Important: the upload endpoint writes BYTES ONLY. It does not create an
 * {@code Artifact} row on its own. The row is created by the subsequent
 * callback, which references the returned {@code storageUri}. This matches
 * the semantics of a real presigned-upload flow, where the upload lands
 * directly on object storage and only the callback teaches the database
 * that the object now exists. It also eliminates the double-write that
 * would otherwise happen when the callback's {@code outputArtifacts} list
 * echoed back the same bytes.
 *
 * In phase 1 the internal endpoint is unauthenticated; a real deployment
 * would restrict it to the worker network or require a shared secret.
 */
@RestController
public class ArtifactController {

    private final ArtifactAccessUseCase accessUseCase;
    private final ArtifactStoragePort storage;

    public ArtifactController(ArtifactAccessUseCase accessUseCase,
                              ArtifactStoragePort storage) {
        this.accessUseCase = accessUseCase;
        this.storage = storage;
    }

    @GetMapping("/api/v1/artifacts/{id}/content")
    public ResponseEntity<InputStreamResource> download(@PathVariable String id) {
        return accessUseCase.openContent(ArtifactId.of(id))
                .map(content -> {
                    HttpHeaders headers = new HttpHeaders();
                    if (content.artifact().getContentType() != null) {
                        headers.setContentType(MediaType.parseMediaType(content.artifact().getContentType()));
                    }
                    if (content.artifact().getSizeBytes() != null) {
                        headers.setContentLength(content.artifact().getSizeBytes());
                    }
                    return ResponseEntity.ok()
                            .headers(headers)
                            .body(new InputStreamResource(content.stream()));
                })
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    @PostMapping(value = "/api/internal/artifacts",
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<UploadResponse> uploadFromWorker(
            @RequestParam("jobId") String jobIdRaw,
            @RequestParam("type") String typeRaw,
            @RequestParam("file") MultipartFile file
    ) throws IOException {
        JobId jobId = JobId.of(jobIdRaw);
        ArtifactType type = ArtifactType.fromString(typeRaw);

        var stored = storage.store(
                jobId,
                type,
                file.getOriginalFilename(),
                file.getContentType(),
                file.getInputStream(),
                file.getSize());

        return ResponseEntity.ok(new UploadResponse(
                stored.storageUri(),
                stored.sizeBytes(),
                stored.checksumSha256()));
    }

    /** Upload response: bytes-only write, no artifact row yet. */
    public record UploadResponse(
            String storageUri,
            Long sizeBytes,
            String checksumSha256
    ) {}
}
