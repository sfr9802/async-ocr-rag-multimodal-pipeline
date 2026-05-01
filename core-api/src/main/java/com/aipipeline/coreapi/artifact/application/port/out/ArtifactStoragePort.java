package com.aipipeline.coreapi.artifact.application.port.out;

import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.job.domain.JobId;

import java.io.InputStream;

/**
 * Outbound port abstracting the physical artifact store. Phase 1 ships a
 * local-filesystem adapter; a MinIO / S3 adapter can slot in later without
 * application-layer changes.
 *
 * "Signed URL" in this codebase is an intentionally generic concept: the
 * returned URL is whatever an external caller can use to download the
 * content. For the local adapter it is a core-api endpoint; for S3 it would
 * be a real presigned URL.
 */
public interface ArtifactStoragePort {

    /**
     * Persist bytes and return an opaque storage URI usable by
     * {@link #openForRead(String)}. The URI is what gets saved on the
     * Artifact domain object.
     */
    StoredObject store(JobId jobId, ArtifactType type, String originalFilename,
                       String contentType, InputStream content, long contentLength);

    /**
     * Open the content behind a storage URI for reading. Used by the
     * download controller to stream artifacts back to clients.
     */
    InputStream openForRead(String storageUri);

    /**
     * Delete a stored object by URI. Used for compensation when bytes have
     * been staged but the database operation that should reference them fails.
     */
    void delete(String storageUri);

    /**
     * Generate an access URL the caller can hand to an external client so
     * that the client can fetch (or upload) content without a persistent
     * credential. Phase 1 returns a core-api download URL for outputs.
     */
    String generateDownloadUrl(String artifactId);

    /**
     * Descriptor for a freshly stored object. All fields feed directly into
     * the Artifact domain object.
     */
    record StoredObject(
            String storageUri,
            long sizeBytes,
            String checksumSha256
    ) {}
}
