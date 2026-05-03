package com.aipipeline.coreapi.catalog.application.service;

import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.data.domain.Pageable;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anySet;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class SearchUnitIndexingServiceTest {

    private static final Instant NOW = Instant.parse("2026-05-03T00:00:00Z");

    private final SearchUnitJpaRepository searchUnits = mock(SearchUnitJpaRepository.class);
    private final SourceFileJpaRepository sourceFiles = mock(SourceFileJpaRepository.class);
    private final ExtractedArtifactJpaRepository extractedArtifacts = mock(ExtractedArtifactJpaRepository.class);

    @Test
    void pending_nonblank_search_unit_is_claimed_with_stable_index_id_and_metadata() {
        SearchUnitJpaEntity unit = unit(
                "unit-1",
                DocumentCatalogService.SEARCH_UNIT_PAGE,
                "page:1",
                "page text",
                SearchUnitIndexingService.EMBEDDING_STATUS_PENDING,
                "hash-1");
        SourceFileJpaEntity source = source();
        ExtractedArtifactJpaEntity artifact = artifact();
        stubCandidates(List.of(unit), List.of());
        when(searchUnits.save(any())).thenAnswer(invocation -> invocation.getArgument(0));
        when(sourceFiles.findAllById(any())).thenReturn(List.of(source));
        when(extractedArtifacts.findAllById(any())).thenReturn(List.of(artifact));

        List<SearchUnitIndexingService.ClaimedSearchUnit> claimed = service()
                .claimPending("worker-1", 10, Duration.ofMinutes(10), NOW);

        assertThat(claimed).hasSize(1);
        SearchUnitIndexingService.ClaimedSearchUnit item = claimed.getFirst();
        assertThat(item.searchUnitId()).isEqualTo("unit-1");
        assertThat(item.claimToken()).startsWith("worker-1:unit-1:");
        assertThat(item.indexId()).isEqualTo("source_file:source-file-1:unit:PAGE:page:1");
        assertThat(item.sourceFileName()).isEqualTo("receipt.png");
        assertThat(item.artifactType()).isEqualTo("OCR_RESULT_JSON");
        assertThat(item.indexMetadata())
                .containsEntry("search_unit_id", "unit-1")
                .containsEntry("source_file_id", "source-file-1")
                .containsEntry("unit_type", "PAGE")
                .containsEntry("unit_key", "page:1")
                .containsEntry("content_hash", "hash-1")
                .containsEntry("artifact_type", "OCR_RESULT_JSON")
                .containsEntry("source_file_name", "receipt.png");
        assertThat(unit.getEmbeddingStatus()).isEqualTo(SearchUnitIndexingService.EMBEDDING_STATUS_EMBEDDING);
    }

    @Test
    void xlsx_search_unit_claim_includes_sheet_range_and_table_index_metadata() {
        SearchUnitJpaEntity unit = new SearchUnitJpaEntity(
                "unit-xlsx-table",
                "source-file-1",
                "artifact-1",
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
                SearchUnitIndexingService.EMBEDDING_STATUS_PENDING,
                "hash-xlsx",
                NOW,
                NOW);
        SourceFileJpaEntity source = new SourceFileJpaEntity(
                "source-file-1",
                "sales.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "SPREADSHEET",
                "local://source.xlsx",
                DocumentCatalogService.SOURCE_STATUS_READY,
                NOW);
        ExtractedArtifactJpaEntity artifact = new ExtractedArtifactJpaEntity(
                "artifact-1",
                "source-file-1",
                "XLSX_WORKBOOK_JSON",
                "job-1",
                "local://xlsx-workbook.json",
                DocumentCatalogService.XLSX_PIPELINE_VERSION,
                "checksum",
                "{}",
                NOW,
                NOW);
        stubCandidates(List.of(unit), List.of());
        when(searchUnits.save(any())).thenAnswer(invocation -> invocation.getArgument(0));
        when(sourceFiles.findAllById(any())).thenReturn(List.of(source));
        when(extractedArtifacts.findAllById(any())).thenReturn(List.of(artifact));

        List<SearchUnitIndexingService.ClaimedSearchUnit> claimed = service()
                .claimPending("worker-1", 10, Duration.ofMinutes(10), NOW);

        assertThat(claimed).hasSize(1);
        assertThat(claimed.getFirst().indexMetadata())
                .containsEntry("fileType", "xlsx")
                .containsEntry("sheetName", "매출")
                .containsEntry("sheetIndex", 0)
                .containsEntry("cellRange", "A1:D30")
                .containsEntry("tableId", "SalesTable")
                .containsEntry("rowStart", 1)
                .containsEntry("columnEnd", 4)
                .containsEntry("content_sha256", "hash-xlsx")
                .containsEntry("index_id", "source_file:source-file-1:unit:TABLE:sheet:0:매출:table:SalesTable");
    }

    @Test
    void pending_blank_text_is_marked_skipped_instead_of_retried_forever() {
        SearchUnitJpaEntity unit = unit(
                "unit-blank",
                DocumentCatalogService.SEARCH_UNIT_DOCUMENT,
                "document",
                "   ",
                SearchUnitIndexingService.EMBEDDING_STATUS_PENDING,
                "hash-blank");
        stubCandidates(List.of(unit), List.of());
        when(searchUnits.save(any())).thenAnswer(invocation -> invocation.getArgument(0));

        List<SearchUnitIndexingService.ClaimedSearchUnit> claimed = service()
                .claimPending("worker-1", 10, Duration.ofMinutes(10), NOW);

        assertThat(claimed).isEmpty();
        assertThat(unit.getEmbeddingStatus()).isEqualTo(SearchUnitIndexingService.EMBEDDING_STATUS_SKIPPED);
        assertThat(unit.getEmbeddingStatusDetail()).contains("text_content is blank");
    }

    @Test
    void embedded_and_failed_units_are_not_claimed() {
        SearchUnitJpaEntity embedded = unit(
                "unit-embedded",
                DocumentCatalogService.SEARCH_UNIT_PAGE,
                "page:1",
                "embedded text",
                SearchUnitIndexingService.EMBEDDING_STATUS_EMBEDDED,
                "hash-embedded");
        SearchUnitJpaEntity failed = unit(
                "unit-failed",
                DocumentCatalogService.SEARCH_UNIT_PAGE,
                "page:2",
                "failed text",
                SearchUnitIndexingService.EMBEDDING_STATUS_FAILED,
                "hash-failed");
        stubCandidates(List.of(embedded, failed), List.of());

        List<SearchUnitIndexingService.ClaimedSearchUnit> claimed = service()
                .claimPending("worker-1", 10, Duration.ofMinutes(10), NOW);

        assertThat(claimed).isEmpty();
        assertThat(embedded.getEmbeddingStatus()).isEqualTo(SearchUnitIndexingService.EMBEDDING_STATUS_EMBEDDED);
        assertThat(failed.getEmbeddingStatus()).isEqualTo(SearchUnitIndexingService.EMBEDDING_STATUS_FAILED);
        verify(searchUnits, never()).save(embedded);
        verify(searchUnits, never()).save(failed);
    }

    @Test
    void metadata_indexable_false_is_skipped() {
        SearchUnitJpaEntity unit = new SearchUnitJpaEntity(
                "unit-hidden",
                "source-file-1",
                "artifact-1",
                DocumentCatalogService.SEARCH_UNIT_PAGE,
                "page:1",
                null,
                null,
                1,
                1,
                "hidden text",
                "{\"indexable\":false}",
                SearchUnitIndexingService.EMBEDDING_STATUS_PENDING,
                "hash-hidden",
                NOW,
                NOW);
        stubCandidates(List.of(unit), List.of());
        when(searchUnits.save(any())).thenAnswer(invocation -> invocation.getArgument(0));

        List<SearchUnitIndexingService.ClaimedSearchUnit> claimed = service()
                .claimPending("worker-1", 10, Duration.ofMinutes(10), NOW);

        assertThat(claimed).isEmpty();
        assertThat(unit.getEmbeddingStatus()).isEqualTo(SearchUnitIndexingService.EMBEDDING_STATUS_SKIPPED);
        assertThat(unit.getEmbeddingStatusDetail()).isEqualTo("metadata_json.indexable=false");
    }

    @Test
    void stale_completion_requeues_when_content_hash_changed() {
        SearchUnitJpaEntity unit = unit(
                "unit-1",
                DocumentCatalogService.SEARCH_UNIT_PAGE,
                "page:1",
                "new text",
                SearchUnitIndexingService.EMBEDDING_STATUS_EMBEDDING,
                "hash-new");
        unit.claimEmbedding("claim-1", NOW.minusSeconds(1));
        when(searchUnits.findByIdAndEmbeddingClaimToken("unit-1", "claim-1"))
                .thenReturn(Optional.of(unit));
        when(searchUnits.save(any())).thenAnswer(invocation -> invocation.getArgument(0));

        SearchUnitIndexingService.CompletionResult result = service()
                .markEmbedded("unit-1", "claim-1", "hash-old", "ignored", NOW);

        assertThat(result.applied()).isFalse();
        assertThat(result.stale()).isTrue();
        assertThat(unit.getEmbeddingStatus()).isEqualTo(SearchUnitIndexingService.EMBEDDING_STATUS_PENDING);
        assertThat(unit.getEmbeddedAt()).isNull();
    }

    @Test
    void embedded_completion_records_stable_index_id_and_indexed_hash() {
        SearchUnitJpaEntity unit = unit(
                "unit-1",
                DocumentCatalogService.SEARCH_UNIT_TABLE,
                "page:2:table:1",
                "table text",
                SearchUnitIndexingService.EMBEDDING_STATUS_EMBEDDING,
                "hash-table");
        unit.claimEmbedding("claim-1", NOW.minusSeconds(1));
        when(searchUnits.findByIdAndEmbeddingClaimToken("unit-1", "claim-1"))
                .thenReturn(Optional.of(unit));
        when(searchUnits.save(any())).thenAnswer(invocation -> invocation.getArgument(0));

        SearchUnitIndexingService.CompletionResult result = service()
                .markEmbedded("unit-1", "claim-1", "hash-table", "wrong-id", NOW);

        assertThat(result.applied()).isTrue();
        assertThat(result.indexId()).isEqualTo("source_file:source-file-1:unit:TABLE:page:2:table:1");
        assertThat(unit.getEmbeddingStatus()).isEqualTo(SearchUnitIndexingService.EMBEDDING_STATUS_EMBEDDED);
        assertThat(unit.getIndexId()).isEqualTo("source_file:source-file-1:unit:TABLE:page:2:table:1");
        assertThat(unit.getIndexedContentSha256()).isEqualTo("hash-table");
    }

    @Test
    void canonical_update_marks_previously_embedded_unit_pending_only_when_content_changes() {
        SearchUnitJpaEntity unit = unit(
                "unit-1",
                DocumentCatalogService.SEARCH_UNIT_PAGE,
                "page:1",
                "old text",
                SearchUnitIndexingService.EMBEDDING_STATUS_PENDING,
                "hash-old");
        unit.markEmbedded("source_file:source-file-1:unit:PAGE:page:1", "hash-old", NOW);

        unit.updateCanonical(
                "artifact-1",
                null,
                null,
                1,
                1,
                "old text",
                "{}",
                "hash-old",
                SearchUnitIndexingService.EMBEDDING_STATUS_PENDING,
                NOW.plusSeconds(1));

        assertThat(unit.getEmbeddingStatus()).isEqualTo(SearchUnitIndexingService.EMBEDDING_STATUS_EMBEDDED);
        assertThat(unit.getIndexedContentSha256()).isEqualTo("hash-old");

        unit.updateCanonical(
                "artifact-1",
                null,
                null,
                1,
                1,
                "new text",
                "{}",
                "hash-new",
                SearchUnitIndexingService.EMBEDDING_STATUS_PENDING,
                NOW.plusSeconds(2));

        assertThat(unit.getEmbeddingStatus()).isEqualTo(SearchUnitIndexingService.EMBEDDING_STATUS_PENDING);
        assertThat(unit.getIndexedContentSha256()).isNull();
    }

    private void stubCandidates(List<SearchUnitJpaEntity> pending, List<SearchUnitJpaEntity> stale) {
        when(searchUnits.findIndexingCandidates(
                eq(SearchUnitIndexingService.EMBEDDING_STATUS_PENDING),
                anySet(),
                any(Pageable.class)))
                .thenReturn(pending);
        when(searchUnits.findStaleIndexingClaims(
                eq(SearchUnitIndexingService.EMBEDDING_STATUS_EMBEDDING),
                any(Instant.class),
                any(Set.class),
                any(Pageable.class)))
                .thenReturn(stale);
    }

    private SearchUnitIndexingService service() {
        return new SearchUnitIndexingService(
                searchUnits,
                sourceFiles,
                extractedArtifacts,
                new ObjectMapper());
    }

    private static SearchUnitJpaEntity unit(String id,
                                            String unitType,
                                            String unitKey,
                                            String text,
                                            String embeddingStatus,
                                            String contentSha256) {
        return new SearchUnitJpaEntity(
                id,
                "source-file-1",
                "artifact-1",
                unitType,
                unitKey,
                null,
                null,
                1,
                1,
                text,
                "{}",
                embeddingStatus,
                contentSha256,
                NOW,
                NOW);
    }

    private static SourceFileJpaEntity source() {
        return new SourceFileJpaEntity(
                "source-file-1",
                "receipt.png",
                "image/png",
                "IMAGE",
                "local://source.png",
                DocumentCatalogService.SOURCE_STATUS_READY,
                NOW);
    }

    private static ExtractedArtifactJpaEntity artifact() {
        return new ExtractedArtifactJpaEntity(
                "artifact-1",
                "source-file-1",
                "OCR_RESULT_JSON",
                "job-1",
                "local://ocr-result.json",
                DocumentCatalogService.OCR_LITE_PIPELINE_VERSION,
                "checksum",
                "{}",
                NOW,
                NOW);
    }
}
