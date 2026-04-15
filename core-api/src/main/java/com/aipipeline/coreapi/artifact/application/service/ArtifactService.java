package com.aipipeline.coreapi.artifact.application.service;

import com.aipipeline.coreapi.artifact.application.port.in.ArtifactAccessUseCase;
import com.aipipeline.coreapi.artifact.application.port.out.ArtifactRepository;
import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactId;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;

@Service
public class ArtifactService implements ArtifactAccessUseCase {

    private final ArtifactRepository repository;
    private final ArtifactStoragePort storage;

    public ArtifactService(ArtifactRepository repository, ArtifactStoragePort storage) {
        this.repository = repository;
        this.storage = storage;
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<Artifact> findArtifact(ArtifactId id) {
        return repository.findById(id);
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<ArtifactContent> openContent(ArtifactId id) {
        return repository.findById(id)
                .map(a -> new ArtifactContent(a, storage.openForRead(a.getStorageUri())));
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<String> generateAccessUrl(ArtifactId id) {
        return repository.findById(id)
                .map(a -> storage.generateDownloadUrl(a.getId().value()));
    }
}
