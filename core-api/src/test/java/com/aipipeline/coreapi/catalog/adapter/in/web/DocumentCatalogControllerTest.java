package com.aipipeline.coreapi.catalog.adapter.in.web;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaEntity;
import com.aipipeline.coreapi.catalog.application.service.DocumentCatalogService;
import com.aipipeline.coreapi.common.TimeProvider;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase.CreateJobCommand;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase.JobCreationResult;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class DocumentCatalogControllerTest {

    private static final Instant NOW = Instant.parse("2026-05-02T00:00:00Z");

    private DocumentCatalogService catalog;
    private JobManagementUseCase jobManagement;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        catalog = Mockito.mock(DocumentCatalogService.class);
        ArtifactStoragePort storage = Mockito.mock(ArtifactStoragePort.class);
        jobManagement = Mockito.mock(JobManagementUseCase.class);
        TimeProvider timeProvider = Mockito.mock(TimeProvider.class);

        DocumentCatalogController controller = new DocumentCatalogController(
                catalog,
                storage,
                jobManagement,
                timeProvider);
        mockMvc = MockMvcBuilders.standaloneSetup(controller).build();

        Job job = Job.createNew(JobCapability.OCR_EXTRACT, NOW);
        job.markQueued(NOW);
        when(jobManagement.createAndEnqueue(any(CreateJobCommand.class)))
                .thenReturn(new JobCreationResult(job, List.of()));
    }

    @Test
    void ready_source_file_ocr_rerequest_is_rejected() throws Exception {
        SourceFileJpaEntity source = sourceWithStatus(DocumentCatalogService.SOURCE_STATUS_READY);
        when(catalog.findSourceFile("source-file-1")).thenReturn(Optional.of(source));
        when(catalog.canStartOcrExtract(source)).thenReturn(false);

        mockMvc.perform(post("/api/v1/library/source-files/source-file-1/ocr-extract"))
                .andExpect(status().isConflict());

        verify(jobManagement, never()).createAndEnqueue(any(CreateJobCommand.class));
    }

    @Test
    void processing_source_file_ocr_rerequest_is_rejected() throws Exception {
        SourceFileJpaEntity source = sourceWithStatus(DocumentCatalogService.SOURCE_STATUS_PROCESSING);
        when(catalog.findSourceFile("source-file-1")).thenReturn(Optional.of(source));
        when(catalog.canStartOcrExtract(source)).thenReturn(false);

        mockMvc.perform(post("/api/v1/library/source-files/source-file-1/ocr-extract"))
                .andExpect(status().isConflict());

        verify(jobManagement, never()).createAndEnqueue(any(CreateJobCommand.class));
    }

    @Test
    void failed_source_file_ocr_retry_is_allowed() throws Exception {
        SourceFileJpaEntity source = sourceWithStatus(DocumentCatalogService.SOURCE_STATUS_FAILED);
        when(catalog.findSourceFile("source-file-1")).thenReturn(Optional.of(source));
        when(catalog.canStartOcrExtract(source)).thenReturn(true);

        mockMvc.perform(post("/api/v1/library/source-files/source-file-1/ocr-extract"))
                .andExpect(status().isAccepted());

        ArgumentCaptor<CreateJobCommand> command = ArgumentCaptor.forClass(CreateJobCommand.class);
        verify(jobManagement).createAndEnqueue(command.capture());
        assertThat(command.getValue().capability()).isEqualTo(JobCapability.OCR_EXTRACT);
        assertThat(command.getValue().inputs()).hasSize(1);
    }

    private static SourceFileJpaEntity sourceWithStatus(String status) {
        return new SourceFileJpaEntity(
                "source-file-1",
                "receipt.png",
                "image/png",
                "IMAGE",
                "local://source.png",
                status,
                NOW);
    }
}
