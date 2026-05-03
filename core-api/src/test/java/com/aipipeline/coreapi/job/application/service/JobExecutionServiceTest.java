package com.aipipeline.coreapi.job.application.service;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactRepository;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactId;
import com.aipipeline.coreapi.artifact.domain.ArtifactRole;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.catalog.application.service.DocumentCatalogService;
import com.aipipeline.coreapi.common.AipipelineProperties;
import com.aipipeline.coreapi.common.TimeProvider;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase.ClaimCommand;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase.CallbackCommand;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase.CallbackOutcome;
import com.aipipeline.coreapi.job.application.port.in.JobExecutionUseCase.OutputArtifactRef;
import com.aipipeline.coreapi.job.application.port.out.JobRepository;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.aipipeline.coreapi.job.domain.JobStatus;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class JobExecutionServiceTest {

    private static final Instant NOW = Instant.parse("2026-05-02T00:00:00Z");

    @Test
    void callback_imports_ocr_extract_output_artifacts_and_marks_succeeded() {
        JobRepository jobs = mock(JobRepository.class);
        ArtifactRepository artifacts = mock(ArtifactRepository.class);
        Job job = runningOcrExtractJob();
        when(jobs.findById(job.getId())).thenReturn(Optional.of(job));

        var service = new JobExecutionService(
                jobs,
                artifacts,
                fixedTimeProvider(),
                properties());

        var result = service.handleCallback(new CallbackCommand(
                job.getId(),
                "callback-1",
                "worker-1",
                CallbackOutcome.SUCCEEDED,
                null,
                null,
                List.of(
                        new OutputArtifactRef(
                                ArtifactType.OCR_RESULT_JSON,
                                "local://job/ocr-result.json",
                                "application/json",
                                128L,
                                "sha-json"),
                        new OutputArtifactRef(
                                ArtifactType.OCR_TEXT_MARKDOWN,
                                "local://job/ocr-text.md",
                                "text/markdown; charset=utf-8",
                                64L,
                                "sha-md"))));

        ArgumentCaptor<List<Artifact>> savedArtifacts = ArgumentCaptor.forClass(List.class);
        verify(artifacts).saveAll(savedArtifacts.capture());
        verify(jobs).save(job);

        assertThat(result.currentStatus()).isEqualTo(JobStatus.SUCCEEDED);
        assertThat(job.getStatus()).isEqualTo(JobStatus.SUCCEEDED);
        assertThat(savedArtifacts.getValue())
                .extracting(Artifact::getType)
                .containsExactly(ArtifactType.OCR_RESULT_JSON, ArtifactType.OCR_TEXT_MARKDOWN);
        assertThat(savedArtifacts.getValue())
                .extracting(Artifact::getRole)
                .containsOnly(ArtifactRole.OUTPUT);
    }

    @Test
    void failed_ocr_extract_callback_marks_failed_without_artifact_import() {
        JobRepository jobs = mock(JobRepository.class);
        ArtifactRepository artifacts = mock(ArtifactRepository.class);
        Job job = runningOcrExtractJob();
        when(jobs.findById(job.getId())).thenReturn(Optional.of(job));

        var service = new JobExecutionService(
                jobs,
                artifacts,
                fixedTimeProvider(),
                properties());

        var result = service.handleCallback(new CallbackCommand(
                job.getId(),
                "callback-2",
                "worker-1",
                CallbackOutcome.FAILED,
                "OCR_FIXTURE_FAILED",
                "fixture provider failed",
                List.of()));

        verify(jobs).save(job);
        org.mockito.Mockito.verifyNoInteractions(artifacts);

        assertThat(result.currentStatus()).isEqualTo(JobStatus.FAILED);
        assertThat(job.getStatus()).isEqualTo(JobStatus.FAILED);
        assertThat(job.getErrorCode()).isEqualTo("OCR_FIXTURE_FAILED");
        assertThat(job.getErrorMessage()).isEqualTo("fixture provider failed");
    }

    @Test
    void artifact_type_accepts_ocr_lite_kinds() {
        assertThat(ArtifactType.fromString("OCR_RESULT_JSON"))
                .isEqualTo(ArtifactType.OCR_RESULT_JSON);
        assertThat(ArtifactType.fromString("ocr_text_markdown"))
                .isEqualTo(ArtifactType.OCR_TEXT_MARKDOWN);
        assertThat(ArtifactType.fromString("XLSX_WORKBOOK_JSON"))
                .isEqualTo(ArtifactType.XLSX_WORKBOOK_JSON);
        assertThat(ArtifactType.fromString("xlsx_markdown"))
                .isEqualTo(ArtifactType.XLSX_MARKDOWN);
        assertThat(ArtifactType.fromString("XLSX_TABLE_JSON"))
                .isEqualTo(ArtifactType.XLSX_TABLE_JSON);
    }

    @Test
    void claim_includes_source_file_id_for_ocr_extract_inputs() {
        JobRepository jobs = mock(JobRepository.class);
        ArtifactRepository artifacts = mock(ArtifactRepository.class);
        DocumentCatalogService catalog = mock(DocumentCatalogService.class);
        Job job = queuedOcrExtractJob();
        when(jobs.findById(job.getId())).thenReturn(Optional.of(job));
        when(jobs.tryAtomicClaim(job.getId(), "worker-1", Duration.ofSeconds(60), NOW))
                .thenReturn(true);

        Artifact inputArtifact = Artifact.rehydrate(
                ArtifactId.of("input-artifact-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source-file.png",
                "image/png",
                10L,
                "sha-input",
                NOW);
        when(artifacts.findByJobIdAndRole(job.getId(), ArtifactRole.INPUT))
                .thenReturn(List.of(inputArtifact));
        when(catalog.findSourceFileIdByStorageUri("local://source-file.png"))
                .thenReturn(Optional.of("source-file-1"));

        var service = new JobExecutionService(
                jobs,
                artifacts,
                catalog,
                fixedTimeProvider(),
                properties());

        var result = service.claim(new ClaimCommand(job.getId(), "worker-1", 1));

        assertThat(result.granted()).isTrue();
        assertThat(result.inputs()).hasSize(1);
        assertThat(result.inputs().get(0).sourceFileId()).isEqualTo("source-file-1");
    }

    @Test
    void claim_includes_source_file_id_for_xlsx_extract_inputs() {
        JobRepository jobs = mock(JobRepository.class);
        ArtifactRepository artifacts = mock(ArtifactRepository.class);
        DocumentCatalogService catalog = mock(DocumentCatalogService.class);
        Job job = queuedXlsxExtractJob();
        when(jobs.findById(job.getId())).thenReturn(Optional.of(job));
        when(jobs.tryAtomicClaim(job.getId(), "worker-1", Duration.ofSeconds(60), NOW))
                .thenReturn(true);

        Artifact inputArtifact = Artifact.rehydrate(
                ArtifactId.of("input-artifact-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source-file.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                10L,
                "sha-input",
                NOW);
        when(artifacts.findByJobIdAndRole(job.getId(), ArtifactRole.INPUT))
                .thenReturn(List.of(inputArtifact));
        when(catalog.findSourceFileIdByStorageUri("local://source-file.xlsx"))
                .thenReturn(Optional.of("source-file-1"));

        var service = new JobExecutionService(
                jobs,
                artifacts,
                catalog,
                fixedTimeProvider(),
                properties());

        var result = service.claim(new ClaimCommand(job.getId(), "worker-1", 1));

        assertThat(result.granted()).isTrue();
        assertThat(result.inputs()).hasSize(1);
        assertThat(result.inputs().get(0).sourceFileId()).isEqualTo("source-file-1");
    }

    @Test
    void successful_ocr_extract_callback_imports_catalog_artifacts_and_search_units() {
        JobRepository jobs = mock(JobRepository.class);
        ArtifactRepository artifacts = mock(ArtifactRepository.class);
        DocumentCatalogService catalog = mock(DocumentCatalogService.class);
        Job job = runningOcrExtractJob();
        when(jobs.findById(job.getId())).thenReturn(Optional.of(job));

        Artifact inputArtifact = Artifact.rehydrate(
                ArtifactId.of("input-artifact-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source-file.png",
                "image/png",
                10L,
                "sha-input",
                NOW);
        when(artifacts.findByJobIdAndRole(job.getId(), ArtifactRole.INPUT))
                .thenReturn(List.of(inputArtifact));
        when(artifacts.saveAll(org.mockito.ArgumentMatchers.anyList()))
                .thenAnswer(invocation -> invocation.getArgument(0));

        var service = new JobExecutionService(
                jobs,
                artifacts,
                catalog,
                fixedTimeProvider(),
                properties());

        service.handleCallback(new CallbackCommand(
                job.getId(),
                "callback-3",
                "worker-1",
                CallbackOutcome.SUCCEEDED,
                null,
                null,
                List.of(
                        new OutputArtifactRef(
                                ArtifactType.OCR_RESULT_JSON,
                                "local://job/ocr-result.json",
                                "application/json",
                                128L,
                                "sha-json"),
                        new OutputArtifactRef(
                                ArtifactType.OCR_TEXT_MARKDOWN,
                                "local://job/ocr-text.md",
                                "text/markdown; charset=utf-8",
                                64L,
                                "sha-md"))));

        ArgumentCaptor<List<Artifact>> savedOutputs = ArgumentCaptor.forClass(List.class);
        verify(catalog).importOcrSucceeded(
                org.mockito.ArgumentMatchers.eq(job),
                org.mockito.ArgumentMatchers.eq(List.of(inputArtifact)),
                savedOutputs.capture(),
                org.mockito.ArgumentMatchers.eq(NOW));
        assertThat(savedOutputs.getValue())
                .extracting(Artifact::getType)
                .containsExactly(ArtifactType.OCR_RESULT_JSON, ArtifactType.OCR_TEXT_MARKDOWN);
    }

    @Test
    void failed_ocr_extract_callback_marks_catalog_source_failed() {
        JobRepository jobs = mock(JobRepository.class);
        ArtifactRepository artifacts = mock(ArtifactRepository.class);
        DocumentCatalogService catalog = mock(DocumentCatalogService.class);
        Job job = runningOcrExtractJob();
        when(jobs.findById(job.getId())).thenReturn(Optional.of(job));

        Artifact inputArtifact = Artifact.rehydrate(
                ArtifactId.of("input-artifact-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source-file.png",
                "image/png",
                10L,
                "sha-input",
                NOW);
        when(artifacts.findByJobIdAndRole(job.getId(), ArtifactRole.INPUT))
                .thenReturn(List.of(inputArtifact));

        var service = new JobExecutionService(
                jobs,
                artifacts,
                catalog,
                fixedTimeProvider(),
                properties());

        service.handleCallback(new CallbackCommand(
                job.getId(),
                "callback-4",
                "worker-1",
                CallbackOutcome.FAILED,
                "OCR_PADDLE_FAILED",
                "provider failed",
                List.of()));

        verify(catalog).markOcrFailed(job, List.of(inputArtifact), NOW);
    }

    @Test
    void successful_xlsx_extract_callback_imports_catalog_artifacts_and_search_units() {
        JobRepository jobs = mock(JobRepository.class);
        ArtifactRepository artifacts = mock(ArtifactRepository.class);
        DocumentCatalogService catalog = mock(DocumentCatalogService.class);
        Job job = runningXlsxExtractJob();
        when(jobs.findById(job.getId())).thenReturn(Optional.of(job));

        Artifact inputArtifact = Artifact.rehydrate(
                ArtifactId.of("input-artifact-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source-file.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                10L,
                "sha-input",
                NOW);
        when(artifacts.findByJobIdAndRole(job.getId(), ArtifactRole.INPUT))
                .thenReturn(List.of(inputArtifact));
        when(artifacts.saveAll(org.mockito.ArgumentMatchers.anyList()))
                .thenAnswer(invocation -> invocation.getArgument(0));

        var service = new JobExecutionService(
                jobs,
                artifacts,
                catalog,
                fixedTimeProvider(),
                properties());

        service.handleCallback(new CallbackCommand(
                job.getId(),
                "callback-xlsx-1",
                "worker-1",
                CallbackOutcome.SUCCEEDED,
                null,
                null,
                List.of(
                        new OutputArtifactRef(
                                ArtifactType.XLSX_WORKBOOK_JSON,
                                "local://job/xlsx-workbook.json",
                                "application/json",
                                128L,
                                "sha-json"),
                        new OutputArtifactRef(
                                ArtifactType.XLSX_MARKDOWN,
                                "local://job/xlsx.md",
                                "text/markdown; charset=utf-8",
                                64L,
                                "sha-md"))));

        ArgumentCaptor<List<Artifact>> savedOutputs = ArgumentCaptor.forClass(List.class);
        verify(catalog).importXlsxSucceeded(
                org.mockito.ArgumentMatchers.eq(job),
                org.mockito.ArgumentMatchers.eq(List.of(inputArtifact)),
                savedOutputs.capture(),
                org.mockito.ArgumentMatchers.eq(NOW));
        assertThat(savedOutputs.getValue())
                .extracting(Artifact::getType)
                .containsExactly(ArtifactType.XLSX_WORKBOOK_JSON, ArtifactType.XLSX_MARKDOWN);
    }

    @Test
    void failed_xlsx_extract_callback_marks_catalog_source_failed() {
        JobRepository jobs = mock(JobRepository.class);
        ArtifactRepository artifacts = mock(ArtifactRepository.class);
        DocumentCatalogService catalog = mock(DocumentCatalogService.class);
        Job job = runningXlsxExtractJob();
        when(jobs.findById(job.getId())).thenReturn(Optional.of(job));

        Artifact inputArtifact = Artifact.rehydrate(
                ArtifactId.of("input-artifact-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source-file.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                10L,
                "sha-input",
                NOW);
        when(artifacts.findByJobIdAndRole(job.getId(), ArtifactRole.INPUT))
                .thenReturn(List.of(inputArtifact));

        var service = new JobExecutionService(
                jobs,
                artifacts,
                catalog,
                fixedTimeProvider(),
                properties());

        service.handleCallback(new CallbackCommand(
                job.getId(),
                "callback-xlsx-2",
                "worker-1",
                CallbackOutcome.FAILED,
                "UNSUPPORTED_INPUT_TYPE",
                "Legacy .xls is not supported",
                List.of()));

        verify(catalog).markXlsxFailed(job, List.of(inputArtifact), NOW);
    }

    @Test
    void stale_terminal_ocr_callback_is_ignored_without_duplicate_artifact_import() {
        JobRepository jobs = mock(JobRepository.class);
        ArtifactRepository artifacts = mock(ArtifactRepository.class);
        DocumentCatalogService catalog = mock(DocumentCatalogService.class);
        Job job = runningOcrExtractJob();
        when(jobs.findById(job.getId())).thenReturn(Optional.of(job));
        when(artifacts.saveAll(org.mockito.ArgumentMatchers.anyList()))
                .thenAnswer(invocation -> invocation.getArgument(0));
        Artifact inputArtifact = Artifact.rehydrate(
                ArtifactId.of("input-artifact-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source-file.png",
                "image/png",
                10L,
                "sha-input",
                NOW);
        when(artifacts.findByJobIdAndRole(job.getId(), ArtifactRole.INPUT))
                .thenReturn(List.of(inputArtifact));

        var service = new JobExecutionService(
                jobs,
                artifacts,
                catalog,
                fixedTimeProvider(),
                properties());
        List<OutputArtifactRef> outputs = List.of(new OutputArtifactRef(
                ArtifactType.OCR_RESULT_JSON,
                "local://job/ocr-result.json",
                "application/json",
                128L,
                "sha-json"));

        service.handleCallback(new CallbackCommand(
                job.getId(),
                "callback-first",
                "worker-1",
                CallbackOutcome.SUCCEEDED,
                null,
                null,
                outputs));
        var replay = service.handleCallback(new CallbackCommand(
                job.getId(),
                "callback-second",
                "worker-1",
                CallbackOutcome.SUCCEEDED,
                null,
                null,
                outputs));

        assertThat(replay.applied()).isFalse();
        verify(artifacts, times(1)).saveAll(org.mockito.ArgumentMatchers.anyList());
        verify(catalog, times(1)).importOcrSucceeded(
                org.mockito.ArgumentMatchers.eq(job),
                org.mockito.ArgumentMatchers.eq(List.of(inputArtifact)),
                org.mockito.ArgumentMatchers.anyList(),
                org.mockito.ArgumentMatchers.eq(NOW));
    }

    @Test
    void stale_terminal_xlsx_callback_is_ignored_without_duplicate_artifact_import() {
        JobRepository jobs = mock(JobRepository.class);
        ArtifactRepository artifacts = mock(ArtifactRepository.class);
        DocumentCatalogService catalog = mock(DocumentCatalogService.class);
        Job job = runningXlsxExtractJob();
        when(jobs.findById(job.getId())).thenReturn(Optional.of(job));
        when(artifacts.saveAll(org.mockito.ArgumentMatchers.anyList()))
                .thenAnswer(invocation -> invocation.getArgument(0));
        Artifact inputArtifact = Artifact.rehydrate(
                ArtifactId.of("input-artifact-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source-file.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                10L,
                "sha-input",
                NOW);
        when(artifacts.findByJobIdAndRole(job.getId(), ArtifactRole.INPUT))
                .thenReturn(List.of(inputArtifact));

        var service = new JobExecutionService(
                jobs,
                artifacts,
                catalog,
                fixedTimeProvider(),
                properties());
        List<OutputArtifactRef> outputs = List.of(new OutputArtifactRef(
                ArtifactType.XLSX_WORKBOOK_JSON,
                "local://job/xlsx-workbook.json",
                "application/json",
                128L,
                "sha-json"));

        service.handleCallback(new CallbackCommand(
                job.getId(),
                "callback-first",
                "worker-1",
                CallbackOutcome.SUCCEEDED,
                null,
                null,
                outputs));
        var replay = service.handleCallback(new CallbackCommand(
                job.getId(),
                "callback-second",
                "worker-1",
                CallbackOutcome.SUCCEEDED,
                null,
                null,
                outputs));

        assertThat(replay.applied()).isFalse();
        verify(artifacts, times(1)).saveAll(org.mockito.ArgumentMatchers.anyList());
        verify(catalog, times(1)).importXlsxSucceeded(
                org.mockito.ArgumentMatchers.eq(job),
                org.mockito.ArgumentMatchers.eq(List.of(inputArtifact)),
                org.mockito.ArgumentMatchers.anyList(),
                org.mockito.ArgumentMatchers.eq(NOW));
    }

    private static Job runningOcrExtractJob() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        job.markQueued(NOW);
        boolean claimed = job.tryClaim("worker-1", Duration.ofSeconds(60), NOW);
        assertThat(claimed).isTrue();
        return job;
    }

    private static Job queuedOcrExtractJob() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        job.markQueued(NOW);
        return job;
    }

    private static Job runningXlsxExtractJob() {
        Job job = Job.createNew(JobCapability.XLSX_EXTRACT, NOW);
        job.markQueued(NOW);
        boolean claimed = job.tryClaim("worker-1", Duration.ofSeconds(60), NOW);
        assertThat(claimed).isTrue();
        return job;
    }

    private static Job queuedXlsxExtractJob() {
        Job job = Job.createNew(JobCapability.XLSX_EXTRACT, NOW);
        job.markQueued(NOW);
        return job;
    }

    private static TimeProvider fixedTimeProvider() {
        return new TimeProvider(Clock.fixed(NOW, ZoneOffset.UTC));
    }

    private static AipipelineProperties properties() {
        return new AipipelineProperties(
                new AipipelineProperties.Storage(
                        "local",
                        new AipipelineProperties.Storage.Local("../local-storage"),
                        new AipipelineProperties.Storage.S3(null, "us-east-1", "artifacts", null, null),
                        new AipipelineProperties.Storage.SignedUrl(300)),
                new AipipelineProperties.Queue(
                        "redis",
                        new AipipelineProperties.Queue.RedisQueue(
                                "aipipeline:jobs:pending",
                                "aipipeline:jobs:inflight")),
                new AipipelineProperties.Worker("http://localhost:8080"),
                new AipipelineProperties.Claim(60));
    }
}
