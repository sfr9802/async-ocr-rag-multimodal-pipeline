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
    void failed_source_file_pdf_retry_is_allowed() throws Exception {
        SourceFileJpaEntity source = new SourceFileJpaEntity(
                "source-file-1",
                "contract.pdf",
                "application/pdf",
                "PDF",
                "local://source.pdf",
                DocumentCatalogService.SOURCE_STATUS_FAILED,
                NOW);
        when(catalog.findSourceFile("source-file-1")).thenReturn(Optional.of(source));
        when(catalog.supportsPdfExtract(source)).thenReturn(true);
        when(catalog.canStartPdfExtract(source)).thenReturn(true);

        mockMvc.perform(post("/api/v1/library/source-files/source-file-1/pdf-extract"))
                .andExpect(status().isAccepted());

        ArgumentCaptor<CreateJobCommand> command = ArgumentCaptor.forClass(CreateJobCommand.class);
        verify(jobManagement).createAndEnqueue(command.capture());
        assertThat(command.getValue().capability()).isEqualTo(JobCapability.PDF_EXTRACT);
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
    void pdf_extract_rejects_unsupported_source_type() throws Exception {
        SourceFileJpaEntity source = sourceWithStatus(DocumentCatalogService.SOURCE_STATUS_FAILED);
        when(catalog.findSourceFile("source-file-1")).thenReturn(Optional.of(source));
        when(catalog.supportsPdfExtract(source)).thenReturn(false);

        mockMvc.perform(post("/api/v1/library/source-files/source-file-1/pdf-extract"))
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

    @Test
    void v2_xlsx_citation_prefers_location_json_and_exposes_parser_fields() {
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
                "sheet:0:table:0:A1:D30",
                "SalesTable",
                "workbook/legacy",
                null,
                null,
                "legacy text",
                """
                        {
                          "fileType": "xlsx",
                          "sheetName": "legacy",
                          "cellRange": "Z1:Z9"
                        }
                        """,
                DocumentCatalogService.EMBEDDING_STATUS_PENDING,
                "sha",
                NOW,
                NOW);
        unit.applyIngestionV2(
                "doc-1",
                "docv-1",
                "pa-1",
                "sales.xlsx",
                "SPREADSHEET",
                "table",
                "xlsx",
                """
                        {
                          "type": "xlsx",
                          "sheet_name": "매출",
                          "sheet_index": 0,
                          "table_id": "SalesTable",
                          "cell_range": "A1:D30",
                          "row_range": "1:30",
                          "hidden_policy": "exclude_hidden"
                        }
                        """,
                "embedding text",
                "bm25 text",
                "| 직원명 | 매출 |",
                "sales.xlsx > 매출 > A1:D30",
                "{}",
                "xlsx-openpyxl",
                "xlsx-extract-v1",
                0.92,
                null,
                "[]",
                NOW);
        unit.markEmbedded("source_file:source-file-1:unit:TABLE:sheet:0:table:0:A1:D30",
                "idx-v1",
                "sha",
                NOW);

        DocumentCatalogController.SearchUnitResponse response =
                DocumentCatalogController.SearchUnitResponse.from(source, unit);

        assertThat(response.chunkType()).isEqualTo("table");
        assertThat(response.locationType()).isEqualTo("xlsx");
        assertThat(response.displayText()).isEqualTo("| 직원명 | 매출 |");
        assertThat(response.citationText()).isEqualTo("sales.xlsx > 매출 > A1:D30");
        assertThat(response.parserVersion()).isEqualTo("xlsx-extract-v1");
        assertThat(response.indexVersion()).isEqualTo("idx-v1");
        assertThat(response.citation().sheetName()).isEqualTo("매출");
        assertThat(response.citation().cellRange()).isEqualTo("A1:D30");
        assertThat(response.citation().tableId()).isEqualTo("SalesTable");
        assertThat(response.citation().citationText()).isEqualTo("sales.xlsx > 매출 > A1:D30");
        assertThat(response.citation().locationJson()).contains("\"cell_range\": \"A1:D30\"");
    }

    @Test
    void v2_pdf_citation_exposes_physical_page_page_label_and_bbox() {
        SourceFileJpaEntity source = new SourceFileJpaEntity(
                "source-file-1",
                "contract.pdf",
                "application/pdf",
                "PDF",
                "local://contract.pdf",
                DocumentCatalogService.SOURCE_STATUS_READY,
                NOW);
        SearchUnitJpaEntity unit = new SearchUnitJpaEntity(
                "unit-pdf-1",
                "source-file-1",
                "ocr-json-1",
                DocumentCatalogService.SEARCH_UNIT_PAGE,
                "page:5",
                null,
                "3. 계약 조건",
                5,
                5,
                "해지 조건 본문",
                "{}",
                DocumentCatalogService.EMBEDDING_STATUS_PENDING,
                "sha",
                NOW,
                NOW);
        unit.applyIngestionV2(
                "doc-1",
                "docv-1",
                "pa-1",
                "contract.pdf",
                "PDF",
                "page",
                "pdf",
                """
                        {
                          "type": "pdf",
                          "physical_page_index": 4,
                          "page_no": 5,
                          "page_label": "v",
                          "bbox": [72.0, 120.0, 510.0, 680.0],
                          "section_path": ["3. 계약 조건"],
                          "block_type": "paragraph",
                          "ocr_used": false,
                          "ocr_confidence": null
                        }
                        """,
                "embedding text",
                "bm25 text",
                "해지 조건 본문",
                "contract.pdf > p.5 > bbox [72.0,120.0,510.0,680.0]",
                "{}",
                "pymupdf",
                "pdf-extract-v1",
                0.88,
                null,
                "[]",
                NOW);

        DocumentCatalogController.SearchUnitResponse response =
                DocumentCatalogController.SearchUnitResponse.from(source, unit);

        assertThat(response.citation().citationText())
                .isEqualTo("contract.pdf > p.5 > bbox [72.0,120.0,510.0,680.0]");
        assertThat(response.citation().physicalPageIndex()).isEqualTo(4);
        assertThat(response.citation().pageNo()).isEqualTo(5);
        assertThat(response.citation().pageLabel()).isEqualTo("v");
        assertThat(response.citation().bbox().toString()).contains("72.0");
        assertThat(response.citation().parserVersion()).isEqualTo("pdf-extract-v1");
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
