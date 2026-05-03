package com.aipipeline.coreapi.catalog.application.service;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactId;
import com.aipipeline.coreapi.artifact.domain.ArtifactRole;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaRepository;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DocumentCatalogServiceTest {

    private static final Instant NOW = Instant.parse("2026-05-02T00:00:00Z");

    private final SourceFileJpaRepository sourceFiles = mock(SourceFileJpaRepository.class);
    private final ExtractedArtifactJpaRepository extractedArtifacts = mock(ExtractedArtifactJpaRepository.class);
    private final SearchUnitJpaRepository searchUnits = mock(SearchUnitJpaRepository.class);
    private final ArtifactStoragePort storage = mock(ArtifactStoragePort.class);

    @Test
    void register_processing_source_file_creates_catalog_record() {
        when(sourceFiles.findFirstByStorageUri("local://source.png"))
                .thenReturn(Optional.empty());
        when(sourceFiles.save(any()))
                .thenAnswer(invocation -> invocation.getArgument(0));

        SourceFileJpaEntity source = service().registerProcessingSourceFile(
                "receipt.png",
                "image/png",
                "local://source.png",
                NOW);

        assertThat(source.getOriginalFileName()).isEqualTo("receipt.png");
        assertThat(source.getMimeType()).isEqualTo("image/png");
        assertThat(source.getFileType()).isEqualTo("IMAGE");
        assertThat(source.getStorageUri()).isEqualTo("local://source.png");
        assertThat(source.getStatus()).isEqualTo(DocumentCatalogService.SOURCE_STATUS_PROCESSING);
        assertThat(source.getUpdatedAt()).isEqualTo(NOW);
    }

    @Test
    void import_ocr_success_creates_extracted_artifacts_document_and_page_search_units() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        SourceFileJpaEntity source = processingSource();
        stubSourceAndUpserts(source);
        when(storage.openForRead("local://ocr-result.json"))
                .thenReturn(stream(validOcrJson()));

        Artifact input = artifact("input-artifact-1", job, ArtifactRole.INPUT, ArtifactType.INPUT_FILE,
                "local://source.png", "image/png");
        Artifact result = artifact("ocr-json-1", job, ArtifactRole.OUTPUT, ArtifactType.OCR_RESULT_JSON,
                "local://ocr-result.json", "application/json");
        Artifact markdown = artifact("ocr-md-1", job, ArtifactRole.OUTPUT, ArtifactType.OCR_TEXT_MARKDOWN,
                "local://ocr-text.md", "text/markdown; charset=utf-8");

        service().importOcrSucceeded(job, List.of(input), List.of(result, markdown), NOW);

        ArgumentCaptor<ExtractedArtifactJpaEntity> extractedCaptor =
                ArgumentCaptor.forClass(ExtractedArtifactJpaEntity.class);
        verify(extractedArtifacts, times(2)).save(extractedCaptor.capture());
        assertThat(extractedCaptor.getAllValues())
                .extracting(ExtractedArtifactJpaEntity::getArtifactType)
                .containsExactly("OCR_RESULT_JSON", "OCR_TEXT_MARKDOWN");
        assertThat(extractedCaptor.getAllValues())
                .extracting(ExtractedArtifactJpaEntity::getArtifactKey)
                .containsOnly(job.getId().value());

        ArgumentCaptor<SearchUnitJpaEntity> unitCaptor = ArgumentCaptor.forClass(SearchUnitJpaEntity.class);
        verify(searchUnits, times(2)).save(unitCaptor.capture());
        assertThat(unitCaptor.getAllValues())
                .extracting(SearchUnitJpaEntity::getUnitType)
                .containsExactly(
                        DocumentCatalogService.SEARCH_UNIT_DOCUMENT,
                        DocumentCatalogService.SEARCH_UNIT_PAGE);

        SearchUnitJpaEntity document = unitCaptor.getAllValues().get(0);
        assertThat(document.getUnitKey()).isEqualTo("document");
        assertThat(document.getPageStart()).isEqualTo(1);
        assertThat(document.getPageEnd()).isEqualTo(1);
        assertThat(document.getTextContent()).isEqualTo("first line\nsecond line");

        SearchUnitJpaEntity page = unitCaptor.getAllValues().get(1);
        assertThat(page.getExtractedArtifactId()).isEqualTo("ocr-json-1");
        assertThat(page.getUnitKey()).isEqualTo("page:1");
        assertThat(page.getTextContent()).isEqualTo("first line\nsecond line");
        assertThat(page.getMetadataJson()).contains("\"rawPage\"");
        assertThat(page.getEmbeddingStatus()).isEqualTo(DocumentCatalogService.EMBEDDING_STATUS_PENDING);

        assertThat(source.getStatus()).isEqualTo(DocumentCatalogService.SOURCE_STATUS_READY);
        verify(sourceFiles).save(source);
    }

    @Test
    void duplicate_ocr_import_does_not_duplicate_extracted_artifacts_or_search_units() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        SourceFileJpaEntity source = processingSource();
        stubSourceAndUpserts(source);
        when(storage.openForRead("local://ocr-result.json"))
                .thenReturn(stream(validOcrJson()));

        Artifact input = artifact("input-artifact-1", job, ArtifactRole.INPUT, ArtifactType.INPUT_FILE,
                "local://source.png", "image/png");
        List<Artifact> outputs = List.of(
                artifact("ocr-json-1", job, ArtifactRole.OUTPUT, ArtifactType.OCR_RESULT_JSON,
                        "local://ocr-result.json", "application/json"),
                artifact("ocr-md-1", job, ArtifactRole.OUTPUT, ArtifactType.OCR_TEXT_MARKDOWN,
                        "local://ocr-text.md", "text/markdown; charset=utf-8"));

        service().importOcrSucceeded(job, List.of(input), outputs, NOW);
        service().importOcrSucceeded(job, List.of(input), outputs, NOW);

        verify(extractedArtifacts, times(2)).save(any(ExtractedArtifactJpaEntity.class));
        verify(searchUnits, times(2)).save(any(SearchUnitJpaEntity.class));
        assertThat(source.getStatus()).isEqualTo(DocumentCatalogService.SOURCE_STATUS_READY);
    }

    @Test
    void empty_pages_payload_saves_artifact_document_unit_and_marks_ready_with_warning() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        SourceFileJpaEntity source = processingSource();
        stubSourceAndUpserts(source);
        when(storage.openForRead("local://ocr-result.json"))
                .thenReturn(stream("""
                        {
                          "sourceRecordId": "source-file-1",
                          "pipelineVersion": "ocr-lite-v1",
                          "engine": "fixture",
                          "pages": [],
                          "plainText": ""
                        }
                        """));

        service().importOcrSucceeded(
                job,
                List.of(artifact("input-artifact-1", job, ArtifactRole.INPUT, ArtifactType.INPUT_FILE,
                        "local://source.png", "image/png")),
                List.of(artifact("ocr-json-1", job, ArtifactRole.OUTPUT, ArtifactType.OCR_RESULT_JSON,
                        "local://ocr-result.json", "application/json")),
                NOW);

        verify(extractedArtifacts).save(any(ExtractedArtifactJpaEntity.class));
        ArgumentCaptor<SearchUnitJpaEntity> unitCaptor = ArgumentCaptor.forClass(SearchUnitJpaEntity.class);
        verify(searchUnits).save(unitCaptor.capture());
        assertThat(unitCaptor.getValue().getUnitType()).isEqualTo(DocumentCatalogService.SEARCH_UNIT_DOCUMENT);
        assertThat(unitCaptor.getValue().getTextContent()).isNull();
        assertThat(source.getStatus()).isEqualTo(DocumentCatalogService.SOURCE_STATUS_READY);
        assertThat(source.getStatusDetail()).contains("no page SearchUnits");
    }

    @Test
    void malformed_ocr_result_json_marks_source_file_failed_without_importing_units() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        SourceFileJpaEntity source = processingSource();
        when(sourceFiles.findFirstByStorageUri("local://source.png"))
                .thenReturn(Optional.of(source));
        when(storage.openForRead("local://ocr-result.json"))
                .thenReturn(stream("{not-json"));

        service().importOcrSucceeded(
                job,
                List.of(artifact("input-artifact-1", job, ArtifactRole.INPUT, ArtifactType.INPUT_FILE,
                        "local://source.png", "image/png")),
                List.of(artifact("ocr-json-1", job, ArtifactRole.OUTPUT, ArtifactType.OCR_RESULT_JSON,
                        "local://ocr-result.json", "application/json")),
                NOW);

        assertThat(source.getStatus()).isEqualTo(DocumentCatalogService.SOURCE_STATUS_FAILED);
        assertThat(source.getStatusDetail()).contains("Failed to parse OCR_RESULT_JSON");
        verify(extractedArtifacts, never()).save(any());
        verify(searchUnits, never()).save(any());
        verify(sourceFiles).save(source);
    }

    @Test
    void table_payload_creates_table_search_unit_with_stable_key_and_metadata() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        SourceFileJpaEntity source = processingSource();
        stubSourceAndUpserts(source);
        when(storage.openForRead("local://ocr-result.json"))
                .thenReturn(stream("""
                        {
                          "sourceRecordId": "source-file-1",
                          "pipelineVersion": "ocr-lite-v1",
                          "engine": "fixture",
                          "pages": [
                            {
                              "pageNo": 2,
                              "blocks": [{"text": "table page", "confidence": 0.9}],
                              "tables": [
                                {
                                  "rowCount": 2,
                                  "columnCount": 2,
                                  "rows": [["Item", "Price"], ["Tea", "3000"]]
                                }
                              ]
                            }
                          ],
                          "plainText": "table page"
                        }
                        """));

        service().importOcrSucceeded(
                job,
                List.of(artifact("input-artifact-1", job, ArtifactRole.INPUT, ArtifactType.INPUT_FILE,
                        "local://source.png", "image/png")),
                List.of(artifact("ocr-json-1", job, ArtifactRole.OUTPUT, ArtifactType.OCR_RESULT_JSON,
                        "local://ocr-result.json", "application/json")),
                NOW);

        ArgumentCaptor<SearchUnitJpaEntity> unitCaptor = ArgumentCaptor.forClass(SearchUnitJpaEntity.class);
        verify(searchUnits, times(3)).save(unitCaptor.capture());
        SearchUnitJpaEntity table = unitCaptor.getAllValues().stream()
                .filter(unit -> DocumentCatalogService.SEARCH_UNIT_TABLE.equals(unit.getUnitType()))
                .findFirst()
                .orElseThrow();
        assertThat(table.getUnitKey()).isEqualTo("page:2:table:1");
        assertThat(table.getTextContent()).contains("Item\tPrice");
        assertThat(table.getMetadataJson()).contains("\"rowCount\":\"2\"");
    }

    @Test
    void image_payload_creates_image_search_unit_with_caption_metadata() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        SourceFileJpaEntity source = processingSource();
        stubSourceAndUpserts(source);
        when(storage.openForRead("local://ocr-result.json"))
                .thenReturn(stream("""
                        {
                          "sourceRecordId": "source-file-1",
                          "pipelineVersion": "ocr-lite-v1",
                          "engine": "fixture",
                          "pages": [
                            {
                              "pageNo": 3,
                              "blocks": [{"text": "figure page", "confidence": 0.9}],
                              "images": [
                                {
                                  "imageId": "fig-7",
                                  "caption": "architecture diagram",
                                  "captionSource": "ocr-caption"
                                }
                              ]
                            }
                          ],
                          "plainText": "figure page"
                        }
                        """));

        service().importOcrSucceeded(
                job,
                List.of(artifact("input-artifact-1", job, ArtifactRole.INPUT, ArtifactType.INPUT_FILE,
                        "local://source.png", "image/png")),
                List.of(artifact("ocr-json-1", job, ArtifactRole.OUTPUT, ArtifactType.OCR_RESULT_JSON,
                        "local://ocr-result.json", "application/json")),
                NOW);

        ArgumentCaptor<SearchUnitJpaEntity> unitCaptor = ArgumentCaptor.forClass(SearchUnitJpaEntity.class);
        verify(searchUnits, times(3)).save(unitCaptor.capture());
        SearchUnitJpaEntity image = unitCaptor.getAllValues().stream()
                .filter(unit -> DocumentCatalogService.SEARCH_UNIT_IMAGE.equals(unit.getUnitType()))
                .findFirst()
                .orElseThrow();
        assertThat(image.getUnitKey()).isEqualTo("image:fig-7");
        assertThat(image.getTextContent()).isEqualTo("architecture diagram");
        assertThat(image.getMetadataJson()).contains("\"captionSource\":\"ocr-caption\"");
    }

    @Test
    void mark_ocr_failed_moves_source_file_to_failed() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        job.markQueued(NOW);
        job.tryClaim("worker-1", java.time.Duration.ofSeconds(60), NOW);
        job.markFailed("callback-1", "OCR_FIXTURE_FAILED", "fixture provider failed", NOW);
        SourceFileJpaEntity source = processingSource();
        when(sourceFiles.findFirstByStorageUri("local://source.png"))
                .thenReturn(Optional.of(source));

        service().markOcrFailed(job, List.of(artifact(
                "input-artifact-1",
                job,
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source.png",
                "image/png")), NOW);

        assertThat(source.getStatus()).isEqualTo(DocumentCatalogService.SOURCE_STATUS_FAILED);
        assertThat(source.getStatusDetail()).isEqualTo("fixture provider failed");
        verify(sourceFiles).save(source);
    }

    @Test
    void mark_ocr_failed_keeps_ready_source_file_ready_for_late_failure_callback() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        SourceFileJpaEntity source = new SourceFileJpaEntity(
                "source-file-1",
                "receipt.png",
                "image/png",
                "IMAGE",
                "local://source.png",
                DocumentCatalogService.SOURCE_STATUS_READY,
                NOW);
        when(sourceFiles.findFirstByStorageUri("local://source.png"))
                .thenReturn(Optional.of(source));

        service().markOcrFailed(job, List.of(artifact(
                "input-artifact-1",
                job,
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source.png",
                "image/png")), NOW);

        assertThat(source.getStatus()).isEqualTo(DocumentCatalogService.SOURCE_STATUS_READY);
        verify(sourceFiles, never()).save(source);
    }

    @Test
    void legacy_callback_without_resolvable_source_file_does_not_throw_or_save() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);

        service().markOcrFailed(job, List.of(), NOW);

        verify(sourceFiles, never()).save(any());
    }

    private void stubSourceAndUpserts(SourceFileJpaEntity source) {
        when(sourceFiles.findFirstByStorageUri("local://source.png"))
                .thenReturn(Optional.of(source));
        when(extractedArtifacts.findBySourceFileIdAndArtifactTypeAndArtifactKey(anyString(), anyString(), anyString()))
                .thenReturn(Optional.empty());
        when(extractedArtifacts.save(any()))
                .thenAnswer(invocation -> invocation.getArgument(0));
        when(searchUnits.findBySourceFileIdAndUnitTypeAndUnitKey(anyString(), anyString(), anyString()))
                .thenReturn(Optional.empty());
        when(searchUnits.save(any()))
                .thenAnswer(invocation -> invocation.getArgument(0));
    }

    private DocumentCatalogService service() {
        return new DocumentCatalogService(
                sourceFiles,
                extractedArtifacts,
                searchUnits,
                storage,
                new ObjectMapper());
    }

    private static SourceFileJpaEntity processingSource() {
        return new SourceFileJpaEntity(
                "source-file-1",
                "receipt.png",
                "image/png",
                "IMAGE",
                "local://source.png",
                DocumentCatalogService.SOURCE_STATUS_PROCESSING,
                NOW);
    }

    private static String validOcrJson() {
        return """
                {
                  "sourceRecordId": "source-file-1",
                  "pipelineVersion": "ocr-lite-v1",
                  "engine": "fixture",
                  "pages": [
                    {
                      "pageNo": 1,
                      "blocks": [
                        {"text": "first line", "confidence": 0.95, "bbox": [0, 0, 100, 20]},
                        {"text": "second line", "confidence": 0.91, "bbox": [0, 30, 100, 50]}
                      ]
                    }
                  ],
                  "plainText": "first line\\nsecond line"
                }
                """;
    }

    private static ByteArrayInputStream stream(String value) {
        return new ByteArrayInputStream(value.getBytes(StandardCharsets.UTF_8));
    }

    private static Artifact artifact(String artifactId,
                                     Job job,
                                     ArtifactRole role,
                                     ArtifactType type,
                                     String storageUri,
                                     String contentType) {
        return Artifact.rehydrate(
                ArtifactId.of(artifactId),
                job.getId(),
                role,
                type,
                storageUri,
                contentType,
                10L,
                "sha-" + artifactId,
                NOW);
    }
}
