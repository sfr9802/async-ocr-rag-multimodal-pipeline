package com.aipipeline.coreapi.job.application.service;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactRepository;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactRole;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.common.AipipelineProperties;
import com.aipipeline.coreapi.common.TimeProvider;
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
    }

    private static Job runningOcrExtractJob() {
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        job.markQueued(NOW);
        boolean claimed = job.tryClaim("worker-1", Duration.ofSeconds(60), NOW);
        assertThat(claimed).isTrue();
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
