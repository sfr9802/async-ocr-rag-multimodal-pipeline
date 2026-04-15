package com.aipipeline.coreapi.artifact.adapter.out.persistence;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactRepository;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactId;
import com.aipipeline.coreapi.artifact.domain.ArtifactRole;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.job.domain.JobId;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Optional;

@Component
public class ArtifactPersistenceAdapter implements ArtifactRepository {

    private final ArtifactJpaRepository jpa;

    public ArtifactPersistenceAdapter(ArtifactJpaRepository jpa) {
        this.jpa = jpa;
    }

    @Override
    public Artifact save(Artifact artifact) {
        return toDomain(jpa.save(toEntity(artifact)));
    }

    @Override
    public List<Artifact> saveAll(List<Artifact> artifacts) {
        List<ArtifactJpaEntity> entities = artifacts.stream().map(this::toEntity).toList();
        return jpa.saveAll(entities).stream().map(this::toDomain).toList();
    }

    @Override
    public Optional<Artifact> findById(ArtifactId id) {
        return jpa.findById(id.value()).map(this::toDomain);
    }

    @Override
    public List<Artifact> findByJobId(JobId jobId) {
        return jpa.findByJobId(jobId.value()).stream().map(this::toDomain).toList();
    }

    @Override
    public List<Artifact> findByJobIdAndRole(JobId jobId, ArtifactRole role) {
        return jpa.findByJobIdAndRole(jobId.value(), role.name()).stream()
                .map(this::toDomain).toList();
    }

    private ArtifactJpaEntity toEntity(Artifact a) {
        return new ArtifactJpaEntity(
                a.getId().value(),
                a.getJobId().value(),
                a.getRole().name(),
                a.getType().name(),
                a.getStorageUri(),
                a.getContentType(),
                a.getSizeBytes(),
                a.getChecksumSha256(),
                a.getCreatedAt());
    }

    private Artifact toDomain(ArtifactJpaEntity e) {
        return Artifact.rehydrate(
                ArtifactId.of(e.getId()),
                JobId.of(e.getJobId()),
                ArtifactRole.valueOf(e.getRole()),
                ArtifactType.valueOf(e.getType()),
                e.getStorageUri(),
                e.getContentType(),
                e.getSizeBytes(),
                e.getChecksumSha256(),
                e.getCreatedAt());
    }
}
