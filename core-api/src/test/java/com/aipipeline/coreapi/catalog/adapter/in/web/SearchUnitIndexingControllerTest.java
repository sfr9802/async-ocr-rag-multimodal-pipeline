package com.aipipeline.coreapi.catalog.adapter.in.web;

import com.aipipeline.coreapi.catalog.application.service.SearchUnitIndexingService;
import com.aipipeline.coreapi.common.TimeProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class SearchUnitIndexingControllerTest {

    private static final Instant NOW = Instant.parse("2026-05-03T00:00:00Z");

    private SearchUnitIndexingService indexing;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        indexing = Mockito.mock(SearchUnitIndexingService.class);
        TimeProvider timeProvider = Mockito.mock(TimeProvider.class);
        when(timeProvider.now()).thenReturn(NOW);
        SearchUnitIndexingController controller = new SearchUnitIndexingController(indexing, timeProvider);
        mockMvc = MockMvcBuilders.standaloneSetup(controller).build();
    }

    @Test
    void claim_returns_search_unit_indexing_contract() throws Exception {
        when(indexing.claimPending(eq("worker-1"), eq(2), isNull(), eq(NOW)))
                .thenReturn(List.of(new SearchUnitIndexingService.ClaimedSearchUnit(
                        "unit-1",
                        "claim-1",
                        "source_file:source-file-1:unit:PAGE:page:1",
                        "source-file-1",
                        "receipt.png",
                        "artifact-1",
                        "OCR_RESULT_JSON",
                        "PAGE",
                        "page:1",
                        "Receipt",
                        null,
                        1,
                        1,
                        "page text",
                        "hash-1",
                        "{}",
                        Map.of("search_unit_id", "unit-1", "unit_type", "PAGE"))));

        mockMvc.perform(post("/api/internal/search-units/indexing/claim")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"workerId":"worker-1","batchSize":2}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.units[0].searchUnitId").value("unit-1"))
                .andExpect(jsonPath("$.units[0].indexId").value("source_file:source-file-1:unit:PAGE:page:1"))
                .andExpect(jsonPath("$.units[0].unitType").value("PAGE"))
                .andExpect(jsonPath("$.units[0].indexMetadata.search_unit_id").value("unit-1"));
    }

    @Test
    void embedded_endpoint_passes_completion_request_to_service() throws Exception {
        when(indexing.markEmbedded("unit-1", "claim-1", "hash-1", "index-1", NOW))
                .thenReturn(new SearchUnitIndexingService.CompletionResult(true, false, "index-1", null));

        mockMvc.perform(post("/api/internal/search-units/indexing/unit-1/embedded")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"claimToken":"claim-1","contentSha256":"hash-1","indexId":"index-1"}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.applied").value(true))
                .andExpect(jsonPath("$.indexId").value("index-1"));
    }

    @Test
    void failed_endpoint_uses_stale_after_and_failure_detail() throws Exception {
        when(indexing.markFailed(eq("unit-1"), eq("claim-1"), eq("hash-1"), eq("boom"), eq(NOW)))
                .thenReturn(new SearchUnitIndexingService.CompletionResult(true, false, null, null));

        mockMvc.perform(post("/api/internal/search-units/indexing/unit-1/failed")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"claimToken":"claim-1","contentSha256":"hash-1","detail":"boom"}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.applied").value(true));

        verify(indexing).markFailed("unit-1", "claim-1", "hash-1", "boom", NOW);
    }
}
