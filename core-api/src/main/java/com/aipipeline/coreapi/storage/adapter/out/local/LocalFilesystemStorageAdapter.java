package com.aipipeline.coreapi.storage.adapter.out.local;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.common.AipipelineProperties;
import com.aipipeline.coreapi.job.domain.JobId;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.UUID;

/**
 * Phase-1 storage backend: plain local filesystem rooted under {@code
 * aipipeline.storage.local.root-dir}.
 *
 * Storage URIs produced by this adapter look like
 * {@code local://{relative/path/under/root}}. The same root directory is
 * assumed to be reachable from both core-api and ai-worker in phase 1
 * (they run on the same host). When MinIO or S3 is introduced, the URI
 * scheme changes but the port contract stays the same.
 *
 * Activated when {@code aipipeline.storage.backend=local} (the default).
 */
@Component
@ConditionalOnProperty(prefix = "aipipeline.storage", name = "backend",
        havingValue = "local", matchIfMissing = true)
public class LocalFilesystemStorageAdapter implements ArtifactStoragePort {

    private static final Logger log = LoggerFactory.getLogger(LocalFilesystemStorageAdapter.class);
    private static final String SCHEME = "local://";

    private final Path rootDir;

    public LocalFilesystemStorageAdapter(AipipelineProperties properties) {
        this.rootDir = Paths.get(properties.storage().local().rootDir()).toAbsolutePath().normalize();
    }

    @PostConstruct
    void ensureRootExists() throws IOException {
        Files.createDirectories(rootDir);
        log.info("Local storage root: {}", rootDir);
    }

    @Override
    public StoredObject store(JobId jobId,
                              ArtifactType type,
                              String originalFilename,
                              String contentType,
                              InputStream content,
                              long contentLength) {
        String safeName = sanitize(originalFilename == null ? "blob" : originalFilename);
        String objectKey = String.format("%s/%s/%s-%s",
                jobId.value(),
                type.name().toLowerCase(),
                UUID.randomUUID(),
                safeName);
        Path target = rootDir.resolve(objectKey).normalize();
        if (!target.startsWith(rootDir)) {
            throw new IllegalStateException("Computed object key escapes root: " + objectKey);
        }
        try {
            Files.createDirectories(target.getParent());
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            long size = 0;
            try (var out = Files.newOutputStream(target)) {
                byte[] buf = new byte[8192];
                int n;
                while ((n = content.read(buf)) >= 0) {
                    out.write(buf, 0, n);
                    digest.update(buf, 0, n);
                    size += n;
                }
            }
            String checksum = HexFormat.of().formatHex(digest.digest());
            return new StoredObject(SCHEME + objectKey, size, checksum);
        } catch (IOException ex) {
            throw new RuntimeException("Failed to store artifact: " + objectKey, ex);
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 not available", ex);
        }
    }

    @Override
    public InputStream openForRead(String storageUri) {
        Path path = resolve(storageUri);
        try {
            return new FileInputStream(path.toFile());
        } catch (IOException ex) {
            throw new RuntimeException("Failed to open artifact: " + storageUri, ex);
        }
    }

    @Override
    public void delete(String storageUri) {
        Path path = resolve(storageUri);
        try {
            Files.deleteIfExists(path);
            log.debug("Deleted local artifact: {}", storageUri);
        } catch (IOException ex) {
            throw new RuntimeException("Failed to delete artifact: " + storageUri, ex);
        }
    }

    @Override
    public String generateDownloadUrl(String artifactId) {
        // Phase 1: advise callers to use the artifact controller. We return
        // a relative URL so the inbound adapter decides on hostname/auth.
        return "/api/v1/artifacts/" + artifactId + "/content";
    }

    // ---- helpers ----

    Path resolve(String storageUri) {
        if (storageUri == null || !storageUri.startsWith(SCHEME)) {
            throw new IllegalArgumentException("Not a local:// storage URI: " + storageUri);
        }
        String key = storageUri.substring(SCHEME.length());
        Path path = rootDir.resolve(key).normalize();
        if (!path.startsWith(rootDir)) {
            throw new IllegalArgumentException("URI escapes root: " + storageUri);
        }
        return path;
    }

    Path rootDir() {
        return rootDir;
    }

    private static String sanitize(String name) {
        return name.replaceAll("[^a-zA-Z0-9._-]", "_");
    }
}
