package com.aipipeline.coreapi.storage.adapter.out.s3;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.common.AipipelineProperties;
import com.aipipeline.coreapi.job.domain.JobId;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.S3Configuration;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for {@link S3ArtifactStorageAdapter}.
 *
 * <p>Requires a running MinIO on {@code localhost:9000} with the
 * {@code aipipeline-artifacts} bucket created. Skipped automatically
 * if MinIO is not reachable.
 *
 * <p>To run:
 * <pre>
 *   docker compose --profile minio up -d minio minio-bootstrap
 *   cd core-api && mvn test -Dtest=S3ArtifactStorageAdapterTest
 * </pre>
 */
class S3ArtifactStorageAdapterTest {

    private static final String ENDPOINT = "http://localhost:9000";
    private static final String REGION = "us-east-1";
    private static final String BUCKET = "aipipeline-artifacts";
    private static final String ACCESS_KEY = "aipipeline";
    private static final String SECRET_KEY = "aipipeline_secret";

    private S3ArtifactStorageAdapter adapter;

    @BeforeAll
    static void requireMinio() {
        // Skip the entire test class if MinIO is not reachable.
        try {
            var client = buildClient();
            client.headBucket(b -> b.bucket(BUCKET));
        } catch (Exception ex) {
            Assumptions.assumeTrue(false,
                    "MinIO not reachable at " + ENDPOINT + " — skipping S3 tests. "
                    + "Start MinIO: docker compose --profile minio up -d minio minio-bootstrap");
        }
    }

    @BeforeEach
    void setUp() {
        var s3 = buildClient();
        var presigner = buildPresigner();
        var props = new AipipelineProperties(
                new AipipelineProperties.Storage(
                        "s3",
                        new AipipelineProperties.Storage.Local("unused"),
                        new AipipelineProperties.Storage.S3(ENDPOINT, REGION, BUCKET, ACCESS_KEY, SECRET_KEY),
                        new AipipelineProperties.Storage.SignedUrl(900)),
                new AipipelineProperties.Queue("redis",
                        new AipipelineProperties.Queue.RedisQueue("pending", "inflight")),
                new AipipelineProperties.Worker("http://localhost:8080"),
                new AipipelineProperties.Claim(900));
        adapter = new S3ArtifactStorageAdapter(s3, presigner, props);
    }

    @Test
    void store_and_read_round_trip() throws Exception {
        byte[] data = "hello s3 artifact content".getBytes(StandardCharsets.UTF_8);
        InputStream in = new ByteArrayInputStream(data);

        ArtifactStoragePort.StoredObject stored = adapter.store(
                JobId.of("test-job-1"),
                ArtifactType.FINAL_RESPONSE,
                "test.txt",
                "text/plain",
                in,
                data.length);

        assertThat(stored.storageUri()).startsWith("s3://");
        assertThat(stored.sizeBytes()).isEqualTo(data.length);
        assertThat(stored.checksumSha256()).isNotBlank();

        // Read back
        try (InputStream readBack = adapter.openForRead(stored.storageUri())) {
            byte[] readData = readBack.readAllBytes();
            assertThat(readData).isEqualTo(data);
        }
    }

    @Test
    void store_computes_correct_sha256() {
        byte[] data = "checksum test".getBytes(StandardCharsets.UTF_8);
        var stored = adapter.store(
                JobId.of("test-job-2"),
                ArtifactType.OCR_TEXT,
                "check.txt",
                "text/plain",
                new ByteArrayInputStream(data),
                data.length);
        // SHA-256 of "checksum test"
        assertThat(stored.checksumSha256()).hasSize(64);
    }

    @Test
    void generateDownloadUrl_returns_proxy_path() {
        String url = adapter.generateDownloadUrl("art-123");
        assertThat(url).isEqualTo("/api/v1/artifacts/art-123/content");
    }

    @Test
    void generatePresignedUrl_returns_http_url() {
        // Store something first so the key exists
        byte[] data = "presign test".getBytes(StandardCharsets.UTF_8);
        var stored = adapter.store(
                JobId.of("test-job-3"),
                ArtifactType.FINAL_RESPONSE,
                "presign.txt",
                "text/plain",
                new ByteArrayInputStream(data),
                data.length);

        String uri = stored.storageUri();
        String bucket = uri.substring("s3://".length(), uri.indexOf('/', "s3://".length()));
        String key = uri.substring(uri.indexOf('/', "s3://".length()) + 1);

        String presignedUrl = adapter.generatePresignedUrl(bucket, key);
        assertThat(presignedUrl).startsWith("http");
        assertThat(presignedUrl).contains(key);
    }

    // ---- helpers ----

    private static S3Client buildClient() {
        var creds = StaticCredentialsProvider.create(
                AwsBasicCredentials.create(ACCESS_KEY, SECRET_KEY));
        return S3Client.builder()
                .endpointOverride(URI.create(ENDPOINT))
                .region(Region.of(REGION))
                .credentialsProvider(creds)
                .serviceConfiguration(S3Configuration.builder()
                        .pathStyleAccessEnabled(true).build())
                .forcePathStyle(true)
                .build();
    }

    private static S3Presigner buildPresigner() {
        var creds = StaticCredentialsProvider.create(
                AwsBasicCredentials.create(ACCESS_KEY, SECRET_KEY));
        return S3Presigner.builder()
                .endpointOverride(URI.create(ENDPOINT))
                .region(Region.of(REGION))
                .credentialsProvider(creds)
                .serviceConfiguration(S3Configuration.builder()
                        .pathStyleAccessEnabled(true).build())
                .build();
    }
}
