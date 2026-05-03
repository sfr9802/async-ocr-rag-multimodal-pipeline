package com.aipipeline.coreapi.catalog.adapter.in.web;

import com.aipipeline.coreapi.catalog.application.service.SearchUnitIndexingService;
import com.aipipeline.coreapi.catalog.application.service.SearchUnitIndexingService.ClaimedSearchUnit;
import com.aipipeline.coreapi.catalog.application.service.SearchUnitIndexingService.CompletionResult;
import com.aipipeline.coreapi.common.TimeProvider;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.Duration;
import java.util.List;

@RestController
@RequestMapping("/api/internal/search-units/indexing")
public class SearchUnitIndexingController {

    private final SearchUnitIndexingService indexing;
    private final TimeProvider timeProvider;

    public SearchUnitIndexingController(SearchUnitIndexingService indexing,
                                        TimeProvider timeProvider) {
        this.indexing = indexing;
        this.timeProvider = timeProvider;
    }

    @PostMapping("/claim")
    public ResponseEntity<ClaimResponse> claim(@Valid @RequestBody ClaimRequest body) {
        int batchSize = body.batchSize() == null ? 50 : body.batchSize();
        Duration staleAfter = body.staleAfterSeconds() == null
                ? null
                : Duration.ofSeconds(Math.max(1, body.staleAfterSeconds()));
        List<ClaimedSearchUnit> units = indexing.claimPending(
                body.workerId(),
                batchSize,
                staleAfter,
                timeProvider.now());
        return ResponseEntity.ok(new ClaimResponse(units));
    }

    @PostMapping("/{searchUnitId}/embedded")
    public ResponseEntity<CompletionResponse> embedded(
            @PathVariable String searchUnitId,
            @Valid @RequestBody EmbeddedRequest body
    ) {
        CompletionResult result = indexing.markEmbedded(
                searchUnitId,
                body.claimToken(),
                body.contentSha256(),
                body.indexId(),
                timeProvider.now());
        return ResponseEntity.ok(CompletionResponse.from(result));
    }

    @PostMapping("/{searchUnitId}/failed")
    public ResponseEntity<CompletionResponse> failed(
            @PathVariable String searchUnitId,
            @Valid @RequestBody FailedRequest body
    ) {
        CompletionResult result = indexing.markFailed(
                searchUnitId,
                body.claimToken(),
                body.contentSha256(),
                body.detail(),
                timeProvider.now());
        return ResponseEntity.ok(CompletionResponse.from(result));
    }

    public record ClaimRequest(
            @NotBlank String workerId,
            Integer batchSize,
            Long staleAfterSeconds
    ) {}

    public record ClaimResponse(List<ClaimedSearchUnit> units) {}

    public record EmbeddedRequest(
            @NotBlank String claimToken,
            @NotBlank String contentSha256,
            String indexId
    ) {}

    public record FailedRequest(
            @NotBlank String claimToken,
            @NotBlank String contentSha256,
            String detail
    ) {}

    public record CompletionResponse(
            boolean applied,
            boolean stale,
            String indexId,
            String detail
    ) {
        static CompletionResponse from(CompletionResult result) {
            return new CompletionResponse(
                    result.applied(),
                    result.stale(),
                    result.indexId(),
                    result.detail());
        }
    }
}
