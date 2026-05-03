package com.aipipeline.coreapi.catalog.adapter.in.web;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaEntity;
import com.aipipeline.coreapi.catalog.application.service.DocumentCatalogService;
import com.aipipeline.coreapi.common.TimeProvider;
import com.aipipeline.coreapi.job.adapter.in.web.dto.JobResponses;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase.CreateJobCommand;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase.StagedInputArtifact;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.aipipeline.coreapi.job.domain.JobId;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.time.Instant;
import java.util.List;

@RestController
@RequestMapping("/api/v1/library")
public class DocumentCatalogController {

    private final DocumentCatalogService catalog;
    private final ArtifactStoragePort storage;
    private final JobManagementUseCase jobManagement;
    private final TimeProvider timeProvider;

    public DocumentCatalogController(DocumentCatalogService catalog,
                                     ArtifactStoragePort storage,
                                     JobManagementUseCase jobManagement,
                                     TimeProvider timeProvider) {
        this.catalog = catalog;
        this.storage = storage;
        this.jobManagement = jobManagement;
        this.timeProvider = timeProvider;
    }

    @PostMapping(value = "/source-files", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<SourceFileResponse> uploadSourceFile(
            @RequestParam("file") MultipartFile file
    ) throws IOException {
        if (file == null || file.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "file must be non-empty");
        }

        var stored = storage.store(
                JobId.generate(),
                ArtifactType.INPUT_FILE,
                file.getOriginalFilename(),
                file.getContentType(),
                file.getInputStream(),
                file.getSize());
        try {
            SourceFileJpaEntity source = catalog.createUploadedSourceFile(
                    file.getOriginalFilename(),
                    file.getContentType(),
                    stored.storageUri(),
                    timeProvider.now());
            return ResponseEntity.status(HttpStatus.CREATED).body(SourceFileResponse.from(source));
        } catch (RuntimeException ex) {
            storage.delete(stored.storageUri());
            throw ex;
        }
    }

    @PostMapping("/source-files/{sourceFileId}/ocr-extract")
    public ResponseEntity<JobResponses.JobCreated> startOcrExtract(
            @PathVariable String sourceFileId
    ) {
        SourceFileJpaEntity source = catalog.findSourceFile(sourceFileId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "source file not found"));
        if (!catalog.canStartOcrExtract(source)) {
            throw new ResponseStatusException(
                    HttpStatus.CONFLICT,
                    "OCR_EXTRACT can start only when source_file.status is UPLOADED, FAILED, or EXTRACTION_FAILED.");
        }

        var result = jobManagement.createAndEnqueue(new CreateJobCommand(
                JobCapability.OCR_EXTRACT,
                List.of(new StagedInputArtifact(
                        ArtifactType.INPUT_FILE,
                        source.getStorageUri(),
                        source.getMimeType(),
                        null,
                        null,
                        source.getOriginalFileName()))));
        return ResponseEntity.accepted()
                .body(JobResponses.JobCreated.from(result.job(), result.inputArtifacts()));
    }

    @GetMapping("/search")
    public ResponseEntity<LibrarySearchResponse> search(
            @RequestParam("query") String query,
            @RequestParam(value = "limit", defaultValue = "20") int limit
    ) {
        List<LibrarySearchResult> results = catalog.search(query, limit).stream()
                .map(LibrarySearchResult::from)
                .toList();
        return ResponseEntity.ok(new LibrarySearchResponse(query, results));
    }

    public record SourceFileResponse(
            String sourceFileId,
            String originalFileName,
            String mimeType,
            String fileType,
            String storageUri,
            String status,
            Instant uploadedAt
    ) {
        static SourceFileResponse from(SourceFileJpaEntity source) {
            return new SourceFileResponse(
                    source.getId(),
                    source.getOriginalFileName(),
                    source.getMimeType(),
                    source.getFileType(),
                    source.getStorageUri(),
                    source.getStatus(),
                    source.getUploadedAt());
        }
    }

    public record SearchUnitResponse(
            String searchUnitId,
            String sourceFileId,
            String extractedArtifactId,
            String unitType,
            String unitKey,
            String title,
            String sectionPath,
            Integer pageStart,
            Integer pageEnd,
            String text,
            String textPreview,
            String metadataJson,
            String embeddingStatus,
            Citation citation
    ) {
        static SearchUnitResponse from(SourceFileJpaEntity source, SearchUnitJpaEntity unit) {
            return new SearchUnitResponse(
                    unit.getId(),
                    unit.getSourceFileId(),
                    unit.getExtractedArtifactId(),
                    unit.getCanonicalUnitType(),
                    unit.getUnitKey(),
                    unit.getTitle(),
                    unit.getSectionPath(),
                    unit.getPageStart(),
                    unit.getPageEnd(),
                    unit.getTextContent(),
                    preview(unit.getTextContent()),
                    unit.getMetadataJson(),
                    unit.getEmbeddingStatus(),
                    Citation.from(source, unit));
        }
    }

    public record LibrarySearchResult(
            SourceFileResponse sourceFile,
            SearchUnitResponse searchUnit
    ) {
        static LibrarySearchResult from(DocumentCatalogService.SearchResult result) {
            return new LibrarySearchResult(
                    SourceFileResponse.from(result.sourceFile()),
                    SearchUnitResponse.from(result.sourceFile(), result.searchUnit()));
        }
    }

    public record LibrarySearchResponse(
            String query,
            List<LibrarySearchResult> results
    ) {}

    public record Citation(
            String sourceFileId,
            String sourceFileName,
            String unitId,
            String searchUnitId,
            String unitType,
            String unitKey,
            String title,
            Integer pageStart,
            Integer pageEnd,
            String sectionPath,
            String tableId,
            String imageId,
            Object bbox,
            String artifactId,
            String artifactType
    ) {
        static Citation from(SourceFileJpaEntity source, SearchUnitJpaEntity unit) {
            return new Citation(
                    source.getId(),
                    source.getOriginalFileName(),
                    unit.getId(),
                    unit.getId(),
                    unit.getCanonicalUnitType(),
                    unit.getUnitKey(),
                    unit.getTitle(),
                    unit.getPageStart(),
                    unit.getPageEnd(),
                    unit.getSectionPath(),
                    idSuffix(unit.getCanonicalUnitType(), unit.getUnitKey(), "table:"),
                    idSuffix(unit.getCanonicalUnitType(), unit.getUnitKey(), "image:"),
                    null,
                    unit.getExtractedArtifactId(),
                    null);
        }
    }

    private static String preview(String text) {
        if (text == null) {
            return null;
        }
        String normalized = text.strip().replaceAll("\\s+", " ");
        return normalized.length() <= 240 ? normalized : normalized.substring(0, 237) + "...";
    }

    private static String idSuffix(String unitType, String unitKey, String prefix) {
        if (unitKey == null) {
            return null;
        }
        if ("table:".equals(prefix) && !"TABLE".equals(unitType)) {
            return null;
        }
        if ("image:".equals(prefix) && !"IMAGE".equals(unitType)) {
            return null;
        }
        if (unitKey.startsWith(prefix)) {
            return unitKey.substring(prefix.length());
        }
        String infix = ":" + prefix;
        int index = unitKey.indexOf(infix);
        if (index < 0) {
            return null;
        }
        return unitKey.substring(index + infix.length());
    }
}
