package com.aipipeline.coreapi.storage.adapter.out.s3;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.common.AipipelineProperties;
import com.aipipeline.coreapi.job.domain.JobId;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.HeadBucketRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.UUID;

/**
 * S3/MinIO storage backend. Activated when
 * {@code aipipeline.storage.backend=s3}.
 *
 * <p>Storage URIs produced by this adapter look like
 * {@code s3://{bucket}/{key}}. The same format is understood by the
 * ai-worker's {@code StorageResolver} (boto3 path) and by this adapter's
 * {@link #openForRead(String)} when streaming back through core-api.
 *
 * <p>MinIO compatibility is ensured via path-style access, configured
 * in {@link S3StorageConfiguration}.
 */
@Component
@ConditionalOnProperty(prefix = "aipipeline.storage", name = "backend", havingValue = "s3")
public class S3ArtifactStorageAdapter implements ArtifactStoragePort {

    private static final Logger log = LoggerFactory.getLogger(S3ArtifactStorageAdapter.class);
    private static final String SCHEME = "s3://";

    private final S3Client s3;
    private final S3Presigner presigner;
    private final String bucket;
    private final Duration presignTtl;

    public S3ArtifactStorageAdapter(
            S3Client s3,
            S3Presigner presigner,
            AipipelineProperties properties) {
        this.s3 = s3;
        this.presigner = presigner;
        this.bucket = properties.storage().s3().bucket();
        this.presignTtl = Duration.ofSeconds(properties.storage().signedUrl().ttlSeconds());
    }

    @PostConstruct
    void verifyBucket() {
        s3.headBucket(HeadBucketRequest.builder().bucket(bucket).build());
        log.info("S3 storage ready: bucket={}, presign-ttl={}s", bucket, presignTtl.toSeconds());
    }

    @Override
    public StoredObject store(JobId jobId,
                              ArtifactType type,
                              String originalFilename,
                              String contentType,
                              InputStream content,
                              long contentLength) {
        String safeName = sanitize(originalFilename == null ? "blob" : originalFilename);
        String key = String.format("%s/%s/%s-%s",
                jobId.value(),
                type.name().toLowerCase(),
                UUID.randomUUID(),
                safeName);

        try {
            // Write to temp file while computing checksum, then upload.
            // This avoids buffering the entire stream in memory and gives
            // us an accurate Content-Length for the S3 PutObject call.
            Path tmp = Files.createTempFile("s3-upload-", ".tmp");
            try {
                MessageDigest digest = MessageDigest.getInstance("SHA-256");
                long size = 0;
                try (var out = Files.newOutputStream(tmp)) {
                    byte[] buf = new byte[8192];
                    int n;
                    while ((n = content.read(buf)) >= 0) {
                        out.write(buf, 0, n);
                        digest.update(buf, 0, n);
                        size += n;
                    }
                }
                String checksum = HexFormat.of().formatHex(digest.digest());

                PutObjectRequest put = PutObjectRequest.builder()
                        .bucket(bucket)
                        .key(key)
                        .contentType(contentType)
                        .contentLength(size)
                        .build();
                s3.putObject(put, RequestBody.fromFile(tmp));

                String uri = SCHEME + bucket + "/" + key;
                log.debug("Stored artifact: {} ({} bytes)", uri, size);
                return new StoredObject(uri, size, checksum);
            } finally {
                Files.deleteIfExists(tmp);
            }
        } catch (IOException ex) {
            throw new RuntimeException("Failed to store artifact to S3: " + key, ex);
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 not available", ex);
        }
    }

    @Override
    public InputStream openForRead(String storageUri) {
        var parsed = parseUri(storageUri);
        GetObjectRequest get = GetObjectRequest.builder()
                .bucket(parsed.bucket)
                .key(parsed.key)
                .build();
        return s3.getObject(get);
    }

    @Override
    public void delete(String storageUri) {
        var parsed = parseUri(storageUri);
        s3.deleteObject(DeleteObjectRequest.builder()
                .bucket(parsed.bucket)
                .key(parsed.key)
                .build());
        log.debug("Deleted S3 artifact: {}", storageUri);
    }

    @Override
    public String generateDownloadUrl(String artifactId) {
        // For S3 backend, we still return the core-api proxy path by default
        // so clients work identically to local mode. The presigned URL is
        // available for direct-download integrations that bypass core-api.
        return "/api/v1/artifacts/" + artifactId + "/content";
    }

    /**
     * Generate a presigned S3 GET URL for direct download. This is NOT
     * called by the default {@link #generateDownloadUrl} path (which
     * proxies through core-api), but is available for future use.
     */
    public String generatePresignedUrl(String bucket, String key) {
        var presignReq = GetObjectPresignRequest.builder()
                .signatureDuration(presignTtl)
                .getObjectRequest(GetObjectRequest.builder()
                        .bucket(bucket)
                        .key(key)
                        .build())
                .build();
        return presigner.presignGetObject(presignReq).url().toString();
    }

    // ---- helpers ----

    private record S3Ref(String bucket, String key) {}

    private static S3Ref parseUri(String uri) {
        if (uri == null || !uri.startsWith(SCHEME)) {
            throw new IllegalArgumentException("Not an s3:// URI: " + uri);
        }
        String path = uri.substring(SCHEME.length());
        int slash = path.indexOf('/');
        if (slash < 0) {
            throw new IllegalArgumentException("s3:// URI has no key: " + uri);
        }
        return new S3Ref(path.substring(0, slash), path.substring(slash + 1));
    }

    private static String sanitize(String name) {
        return name.replaceAll("[^a-zA-Z0-9._-]", "_");
    }
}
