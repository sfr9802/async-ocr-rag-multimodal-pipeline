package com.aipipeline.coreapi.catalog.adapter.in.web;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaEntity;
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

    @Test
    void failed_source_file_xlsx_retry_is_allowed() throws Exception {
        SourceFileJpaEntity source = new SourceFileJpaEntity(
                "source-file-1",
                "sales.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "SPREADSHEET",
                "local://source.xlsx",
                DocumentCatalogService.SOURCE_STATUS_FAILED,
                NOW);
        when(catalog.findSourceFile("source-file-1")).thenReturn(Optional.of(source));
        when(catalog.supportsXlsxExtract(source)).thenReturn(true);
        when(catalog.canStartXlsxExtract(source)).thenReturn(true);

        mockMvc.perform(post("/api/v1/library/source-files/source-file-1/xlsx-extract"))
                .andExpect(status().isAccepted());

        ArgumentCaptor<CreateJobCommand> command = ArgumentCaptor.forClass(CreateJobCommand.class);
        verify(jobManagement).createAndEnqueue(command.capture());
        assertThat(command.getValue().capability()).isEqualTo(JobCapability.XLSX_EXTRACT);
        assertThat(command.getValue().inputs()).hasSize(1);
    }

    @Test
    void xlsx_extract_rejects_unsupported_source_type() throws Exception {
        SourceFileJpaEntity source = sourceWithStatus(DocumentCatalogService.SOURCE_STATUS_FAILED);
        when(catalog.findSourceFile("source-file-1")).thenReturn(Optional.of(source));
        when(catalog.supportsXlsxExtract(source)).thenReturn(false);

        mockMvc.perform(post("/api/v1/library/source-files/source-file-1/xlsx-extract"))
                .andExpect(status().isUnsupportedMediaType());

        verify(jobManagement, never()).createAndEnqueue(any(CreateJobCommand.class));
    }

    @Test
    void xlsx_table_citation_includes_sheet_and_cell_range_fields() {
        SourceFileJpaEntity source = new SourceFileJpaEntity(
                "source-file-1",
                "sales.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "SPREADSHEET",
                "local://source.xlsx",
                DocumentCatalogService.SOURCE_STATUS_READY,
                NOW);
        SearchUnitJpaEntity unit = new SearchUnitJpaEntity(
                "unit-1",
                "source-file-1",
                "xlsx-json-1",
                DocumentCatalogService.SEARCH_UNIT_TABLE,
                "sheet:0:매출:table:SalesTable",
                "SalesTable",
                "workbook/매출",
                null,
                null,
                "직원명: 홍길동",
                """
                        {
                          "fileType": "xlsx",
                          "sheetName": "매출",
                          "sheetIndex": 0,
                          "cellRange": "A1:D30",
                          "rowStart": 1,
                          "rowEnd": 30,
                          "columnStart": 1,
                          "columnEnd": 4,
                          "tableId": "SalesTable"
                        }
                        """,
                DocumentCatalogService.EMBEDDING_STATUS_PENDING,
                "sha",
                NOW,
                NOW);

        DocumentCatalogController.SearchUnitResponse response =
                DocumentCatalogController.SearchUnitResponse.from(source, unit);

        assertThat(response.citation().sheetName()).isEqualTo("매출");
        assertThat(response.citation().sheetIndex()).isEqualTo(0);
        assertThat(response.citation().cellRange()).isEqualTo("A1:D30");
        assertThat(response.citation().tableId()).isEqualTo("SalesTable");
        assertThat(response.citation().rowStart()).isEqualTo(1);
        assertThat(response.citation().columnEnd()).isEqualTo(4);
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
