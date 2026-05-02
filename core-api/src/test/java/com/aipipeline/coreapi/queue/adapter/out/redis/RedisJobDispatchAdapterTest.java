package com.aipipeline.coreapi.queue.adapter.out.redis;

import com.aipipeline.coreapi.common.AipipelineProperties;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.ListOperations;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.time.Instant;

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
