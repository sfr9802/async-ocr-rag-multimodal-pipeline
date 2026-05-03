package com.aipipeline.coreapi.queue.adapter.out.redis;

import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactId;
import com.aipipeline.coreapi.artifact.domain.ArtifactRole;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.common.AipipelineProperties;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.ListOperations;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.time.Instant;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RedisJobDispatchAdapterTest {

    @Test
    void ocr_extract_dispatch_payload_includes_task_kind_and_pipeline_version() throws Exception {
        StringRedisTemplate redis = mock(StringRedisTemplate.class);
        @SuppressWarnings("unchecked")
        ListOperations<String, String> ops = mock(ListOperations.class);
        when(redis.opsForList()).thenReturn(ops);

        var adapter = new RedisJobDispatchAdapter(
                redis,
                new ObjectMapper(),
                properties());
        Job job = Job.createNew(JobCapability.OCR_EXTRACT, Instant.parse("2026-05-02T00:00:00Z"));
        job.markQueued(Instant.parse("2026-05-02T00:00:00Z"));

        adapter.dispatch(job);

        org.mockito.ArgumentCaptor<String> payload = org.mockito.ArgumentCaptor.forClass(String.class);
        verify(ops).leftPush(eq("aipipeline:jobs:pending"), payload.capture());

        var json = new ObjectMapper().readTree(payload.getValue());
        assertThat(json.get("jobId").asText()).isEqualTo(job.getId().value());
        assertThat(json.get("capability").asText()).isEqualTo("OCR_EXTRACT");
        assertThat(json.get("taskKind").asText()).isEqualTo("OCR_EXTRACT");
        assertThat(json.get("pipelineVersion").asText()).isEqualTo("ocr-lite-v1");
        assertThat(json.get("callbackBaseUrl").asText()).isEqualTo("http://localhost:8080");
    }

    @Test
    void xlsx_extract_dispatch_payload_includes_task_kind_pipeline_version_and_input_file() throws Exception {
        StringRedisTemplate redis = mock(StringRedisTemplate.class);
        @SuppressWarnings("unchecked")
        ListOperations<String, String> ops = mock(ListOperations.class);
        when(redis.opsForList()).thenReturn(ops);

        var adapter = new RedisJobDispatchAdapter(
                redis,
                new ObjectMapper(),
                properties());
        Job job = Job.createNew(JobCapability.XLSX_EXTRACT, Instant.parse("2026-05-02T00:00:00Z"));
        job.markQueued(Instant.parse("2026-05-02T00:00:00Z"));
        Artifact input = Artifact.rehydrate(
                ArtifactId.of("input-artifact-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source/sales.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                1024L,
                "sha-input",
                Instant.parse("2026-05-02T00:00:00Z"));

        adapter.dispatch(job, List.of(input));

        org.mockito.ArgumentCaptor<String> payload = org.mockito.ArgumentCaptor.forClass(String.class);
        verify(ops).leftPush(eq("aipipeline:jobs:pending"), payload.capture());

        var json = new ObjectMapper().readTree(payload.getValue());
        assertThat(json.get("jobId").asText()).isEqualTo(job.getId().value());
        assertThat(json.get("capability").asText()).isEqualTo("XLSX_EXTRACT");
        assertThat(json.get("taskKind").asText()).isEqualTo("XLSX_EXTRACT");
        assertThat(json.get("pipelineVersion").asText()).isEqualTo("xlsx-extract-v1");
        assertThat(json.get("callbackBaseUrl").asText()).isEqualTo("http://localhost:8080");
        assertThat(json.get("inputArtifacts").isArray()).isTrue();
        assertThat(json.get("inputArtifacts").size()).isEqualTo(1);
        assertThat(json.get("inputArtifacts").get(0).get("artifactId").asText()).isEqualTo("input-artifact-1");
        assertThat(json.get("inputArtifacts").get(0).get("role").asText()).isEqualTo("INPUT");
        assertThat(json.get("inputArtifacts").get(0).get("type").asText()).isEqualTo("INPUT_FILE");
        assertThat(json.get("inputArtifacts").get(0).get("storageUri").asText()).isEqualTo("local://source/sales.xlsx");
    }

    @Test
    void pdf_extract_dispatch_payload_includes_task_kind_pipeline_version_and_input_file() throws Exception {
        StringRedisTemplate redis = mock(StringRedisTemplate.class);
        @SuppressWarnings("unchecked")
        ListOperations<String, String> ops = mock(ListOperations.class);
        when(redis.opsForList()).thenReturn(ops);

        var adapter = new RedisJobDispatchAdapter(
                redis,
                new ObjectMapper(),
                properties());
        Job job = Job.createNew(JobCapability.PDF_EXTRACT, Instant.parse("2026-05-02T00:00:00Z"));
        job.markQueued(Instant.parse("2026-05-02T00:00:00Z"));
        Artifact input = Artifact.rehydrate(
                ArtifactId.of("input-artifact-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://source/contract.pdf",
                "application/pdf",
                1024L,
                "sha-input",
                Instant.parse("2026-05-02T00:00:00Z"));

        adapter.dispatch(job, List.of(input));

        org.mockito.ArgumentCaptor<String> payload = org.mockito.ArgumentCaptor.forClass(String.class);
        verify(ops).leftPush(eq("aipipeline:jobs:pending"), payload.capture());

        var json = new ObjectMapper().readTree(payload.getValue());
        assertThat(json.get("jobId").asText()).isEqualTo(job.getId().value());
        assertThat(json.get("capability").asText()).isEqualTo("PDF_EXTRACT");
        assertThat(json.get("taskKind").asText()).isEqualTo("PDF_EXTRACT");
        assertThat(json.get("pipelineVersion").asText()).isEqualTo("pdf-extract-v1");
        assertThat(json.get("inputArtifacts").isArray()).isTrue();
        assertThat(json.get("inputArtifacts").size()).isEqualTo(1);
        assertThat(json.get("inputArtifacts").get(0).get("type").asText()).isEqualTo("INPUT_FILE");
        assertThat(json.get("inputArtifacts").get(0).get("storageUri").asText()).isEqualTo("local://source/contract.pdf");
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
