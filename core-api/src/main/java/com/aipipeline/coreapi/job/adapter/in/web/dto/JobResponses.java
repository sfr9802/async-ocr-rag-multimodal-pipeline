package com.aipipeline.coreapi.job.adapter.in.web.dto;

import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.job.domain.Job;

import java.time.Instant;
import java.util.List;
import java.util.Objects;

/**
 * Response DTOs used by the client-facing job controller. Grouped in one
 * file because they're all plain records describing the same resource and
 * splitting each into its own file is unnecessary ceremony.
 */
public final class JobResponses {

    private JobResponses() {}

    public record JobCreated(
            String jobId,
            String status,
            String capability,
            List<ArtifactView> inputs
    ) {
        public static JobCreated from(Job job, List<Artifact> inputs) {
            return new JobCreated(
                    job.getId().value(),
                    job.getStatus().name(),
                    job.getCapability().name(),
                    inputs.stream().map(ArtifactView::from).toList());
        }
    }

    public record JobView(
            String jobId,
            String capability,
            String status,
            int attemptNo,
            String errorCode,
            String errorMessage,
            Instant createdAt,
            Instant updatedAt
    ) {
        public static JobView from(Job job) {
            return new JobView(
                    job.getId().value(),
                    job.getCapability().name(),
                    job.getStatus().name(),
                    job.getAttemptNo(),
                    job.getErrorCode(),
                    job.getErrorMessage(),
                    job.getCreatedAt(),
                    job.getUpdatedAt());
        }
    }

    public record JobResult(
            String jobId,
            String status,
            List<ArtifactView> inputs,
            List<ArtifactView> outputs,
            String errorCode,
            String errorMessage
    ) {
        public static JobResult from(Job job, List<Artifact> all) {
            List<ArtifactView> inputs = all.stream()
                    .filter(a -> a.getRole().name().equals("INPUT"))
                    .map(ArtifactView::from)
                    .toList();
            List<ArtifactView> outputs = all.stream()
                    .filter(a -> a.getRole().name().equals("OUTPUT"))
                    .map(ArtifactView::from)
                    .toList();
            return new JobResult(
                    job.getId().value(),
                    job.getStatus().name(),
                    inputs,
                    outputs,
                    job.getErrorCode(),
                    job.getErrorMessage());
        }
    }

    public record ArtifactView(
            String id,
            String role,
            String type,
            String contentType,
            Long sizeBytes,
            String checksumSha256,
            String accessUrl
    ) {
        public static ArtifactView from(Artifact a) {
            return new ArtifactView(
                    a.getId().value(),
                    a.getRole().name(),
                    a.getType().name(),
                    a.getContentType(),
                    a.getSizeBytes(),
                    a.getChecksumSha256(),
                    "/api/v1/artifacts/" + a.getId().value() + "/content");
        }
    }

    public record ErrorBody(String code, String message) {
        public ErrorBody {
            Objects.requireNonNull(code, "code");
        }
    }
}
