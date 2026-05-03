package com.aipipeline.coreapi.catalog.application.service;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.DocumentV2JpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.DocumentV2JpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.DocumentVersionJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.DocumentVersionJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ParsedArtifactV2JpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ParsedArtifactV2JpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.PdfPageMetadataJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.PdfPageMetadataJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.TableMetadataJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.TableMetadataJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.CellMetadataJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.CellMetadataJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaRepository;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

@Service
public class DocumentCatalogService {

    public static final String OCR_LITE_PIPELINE_VERSION = "ocr-lite-v1";
    public static final String XLSX_PIPELINE_VERSION = "xlsx-extract-v1";
    public static final String PDF_PIPELINE_VERSION = "pdf-extract-v1";
    public static final String SOURCE_STATUS_UPLOADED = "UPLOADED";
    public static final String SOURCE_STATUS_PROCESSING = "PROCESSING";
    public static final String SOURCE_STATUS_READY = "READY";
    public static final String SOURCE_STATUS_FAILED = "FAILED";
    public static final String SOURCE_STATUS_EXTRACTION_FAILED = "EXTRACTION_FAILED";

    public static final String SEARCH_UNIT_DOCUMENT = "DOCUMENT";
    public static final String SEARCH_UNIT_PAGE = "PAGE";
    public static final String SEARCH_UNIT_SECTION = "SECTION";
    public static final String SEARCH_UNIT_TABLE = "TABLE";
    public static final String SEARCH_UNIT_IMAGE = "IMAGE";
    public static final String SEARCH_UNIT_CHUNK = "CHUNK";
    public static final String SEARCH_UNIT_OCR_PAGE = "ocr_page";

    public static final String EMBEDDING_STATUS_PENDING = "PENDING";

    private static final int DOCUMENT_TEXT_MAX_CHARS = 200_000;
    private static final int MAX_NORMALIZED_CELLS_PER_SHEET = 500;
    private static final int MAX_NORMALIZED_CELLS_PER_WORKBOOK = 4_000;
    private static final Logger log = LoggerFactory.getLogger(DocumentCatalogService.class);

    private final SourceFileJpaRepository sourceFiles;
    private final ExtractedArtifactJpaRepository extractedArtifacts;
    private final SearchUnitJpaRepository searchUnits;
    private final DocumentV2JpaRepository documents;
    private final DocumentVersionJpaRepository documentVersions;
    private final ParsedArtifactV2JpaRepository parsedArtifacts;
    private final PdfPageMetadataJpaRepository pdfPageMetadata;
    private final TableMetadataJpaRepository tableMetadata;
    private final CellMetadataJpaRepository cellMetadata;
    private final ArtifactStoragePort storage;
    private final ObjectMapper objectMapper;

    public DocumentCatalogService(SourceFileJpaRepository sourceFiles,
                                  ExtractedArtifactJpaRepository extractedArtifacts,
                                  SearchUnitJpaRepository searchUnits,
                                  DocumentV2JpaRepository documents,
                                  DocumentVersionJpaRepository documentVersions,
                                  ParsedArtifactV2JpaRepository parsedArtifacts,
                                  PdfPageMetadataJpaRepository pdfPageMetadata,
                                  TableMetadataJpaRepository tableMetadata,
                                  CellMetadataJpaRepository cellMetadata,
                                  ArtifactStoragePort storage,
                                  ObjectMapper objectMapper) {
        this.sourceFiles = sourceFiles;
        this.extractedArtifacts = extractedArtifacts;
        this.searchUnits = searchUnits;
        this.documents = documents;
        this.documentVersions = documentVersions;
        this.parsedArtifacts = parsedArtifacts;
        this.pdfPageMetadata = pdfPageMetadata;
        this.tableMetadata = tableMetadata;
        this.cellMetadata = cellMetadata;
        this.storage = storage;
        this.objectMapper = objectMapper;
    }

    @Transactional
    public SourceFileJpaEntity createUploadedSourceFile(String originalFileName,
                                                        String mimeType,
                                                        String storageUri,
                                                        Instant now) {
        SourceFileJpaEntity entity = new SourceFileJpaEntity(
                newId(),
                normalizeFileName(originalFileName, storageUri),
                normalizeMime(mimeType),
                classifyFileType(mimeType, originalFileName),
                storageUri,
                SOURCE_STATUS_UPLOADED,
                null,
                now,
                now);
        return sourceFiles.save(entity);
    }

    @Transactional
    public SourceFileJpaEntity registerProcessingSourceFile(String originalFileName,
                                                            String mimeType,
                                                            String storageUri,
                                                            Instant now) {
        Optional<SourceFileJpaEntity> existing = sourceFiles.findFirstByStorageUri(storageUri);
        if (existing.isPresent()) {
            SourceFileJpaEntity source = existing.get();
            source.setOriginalFileName(normalizeFileName(originalFileName, storageUri));
            source.setMimeType(normalizeMime(mimeType));
            source.setFileType(classifyFileType(mimeType, originalFileName));
            transitionSource(source, SOURCE_STATUS_PROCESSING, null, now);
            return sourceFiles.save(source);
        }

        SourceFileJpaEntity entity = new SourceFileJpaEntity(
                newId(),
                normalizeFileName(originalFileName, storageUri),
                normalizeMime(mimeType),
                classifyFileType(mimeType, originalFileName),
                storageUri,
                SOURCE_STATUS_PROCESSING,
                null,
                now,
                now);
        return sourceFiles.save(entity);
    }

    @Transactional(readOnly = true)
    public Optional<SourceFileJpaEntity> findSourceFile(String sourceFileId) {
        return sourceFiles.findById(sourceFileId);
    }

    public boolean canStartOcrExtract(SourceFileJpaEntity sourceFile) {
        String status = sourceFile.getStatus();
        return SOURCE_STATUS_UPLOADED.equals(status)
                || SOURCE_STATUS_FAILED.equals(status)
                || SOURCE_STATUS_EXTRACTION_FAILED.equals(status);
    }

    public boolean canStartXlsxExtract(SourceFileJpaEntity sourceFile) {
        return canStartOcrExtract(sourceFile);
    }

    public boolean canStartPdfExtract(SourceFileJpaEntity sourceFile) {
        return canStartOcrExtract(sourceFile);
    }

    public boolean supportsXlsxExtract(SourceFileJpaEntity sourceFile) {
        String fileName = sourceFile.getOriginalFileName() == null
                ? ""
                : sourceFile.getOriginalFileName().toLowerCase(Locale.ROOT);
        String mimeType = normalizeMime(sourceFile.getMimeType());
        if (fileName.endsWith(".xls")) {
            return false;
        }
        return fileName.endsWith(".xlsx")
                || fileName.endsWith(".xlsm")
                || "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".equals(mimeType)
                || "application/vnd.ms-excel.sheet.macroenabled.12".equals(mimeType);
    }

    public boolean supportsPdfExtract(SourceFileJpaEntity sourceFile) {
        String fileName = sourceFile.getOriginalFileName() == null
                ? ""
                : sourceFile.getOriginalFileName().toLowerCase(Locale.ROOT);
        String mimeType = normalizeMime(sourceFile.getMimeType());
        return fileName.endsWith(".pdf")
                || "application/pdf".equals(mimeType)
                || "application/x-pdf".equals(mimeType);
    }

    @Transactional(readOnly = true)
    public Optional<String> findSourceFileIdByStorageUri(String storageUri) {
        return sourceFiles.findFirstByStorageUri(storageUri).map(SourceFileJpaEntity::getId);
    }

    @Transactional
    public void importOcrSucceeded(Job job,
                                   List<Artifact> inputArtifacts,
                                   List<Artifact> outputArtifacts,
                                   Instant now) {
        if (job.getCapability() != JobCapability.OCR_EXTRACT) {
            return;
        }

        SourceFileJpaEntity source = resolveOrCreateSourceFile(inputArtifacts, now)
                .orElseGet(() -> {
                    log.warn("Skipping OCR catalog import: no source file for jobId={}", job.getId());
                    return null;
                });
        if (source == null) {
            return;
        }
        if (SOURCE_STATUS_READY.equals(source.getStatus())) {
            log.info("Ignoring OCR catalog import for already READY sourceFileId={}", source.getId());
            return;
        }

        ImportPlan plan;
        try {
            plan = buildImportPlan(job, source, outputArtifacts, now);
        } catch (OcrCatalogImportException ex) {
            log.warn(
                    "OCR catalog import failed sourceFileId={} jobId={} reason={}",
                    source.getId(), job.getId(), ex.getMessage());
            transitionSource(source, SOURCE_STATUS_FAILED, ex.getMessage(), now);
            sourceFiles.save(source);
            return;
        }

        if (!plan.valid()) {
            log.warn(
                    "OCR catalog import failed sourceFileId={} jobId={} reason={}",
                    source.getId(), job.getId(), plan.errorMessage());
            transitionSource(source, SOURCE_STATUS_FAILED, plan.errorMessage(), now);
            sourceFiles.save(source);
            return;
        }

        try {
            Map<String, ExtractedArtifactJpaEntity> extractedByArtifactId = upsertExtractedArtifacts(plan, now);
            DocumentIngestionContext documentContext =
                    upsertDocumentContext(source, plan, extractedByArtifactId, now);
            for (SearchUnitDraft draft : plan.searchUnits()) {
                String resolvedArtifactId = draft.extractedArtifactId();
                if (resolvedArtifactId != null && !extractedByArtifactId.containsKey(resolvedArtifactId)) {
                    log.warn(
                            "Skipping search unit import with missing extracted artifact sourceFileId={} unitType={} unitKey={}",
                            source.getId(), draft.unitType(), draft.unitKey());
                    continue;
                }
                upsertSearchUnit(source, documentContext, draft, now);
            }
        } catch (RuntimeException ex) {
            log.warn(
                    "OCR catalog import write failed sourceFileId={} jobId={}: {}",
                    source.getId(), job.getId(), ex.toString());
            transitionSource(source, SOURCE_STATUS_FAILED, "OCR catalog write failed: " + ex.getMessage(), now);
            sourceFiles.save(source);
            return;
        }

        String detail = plan.warningMessage();
        transitionSource(source, SOURCE_STATUS_READY, detail, now);
        sourceFiles.save(source);
    }

    @Transactional
    public void markOcrFailed(Job job,
                              List<Artifact> inputArtifacts,
                              Instant now) {
        if (job.getCapability() != JobCapability.OCR_EXTRACT) {
            return;
        }
        Optional<SourceFileJpaEntity> maybeSource = resolveOrCreateSourceFile(inputArtifacts, now);
        if (maybeSource.isEmpty()) {
            log.warn("OCR failure callback could not resolve source file jobId={}", job.getId());
            return;
        }
        SourceFileJpaEntity source = maybeSource.get();
        if (SOURCE_STATUS_READY.equals(source.getStatus())) {
            log.info("Ignoring late OCR failure for READY sourceFileId={}", source.getId());
            return;
        }
        String detail = job.getErrorMessage() == null || job.getErrorMessage().isBlank()
                ? job.getErrorCode()
                : job.getErrorMessage();
        transitionSource(source, SOURCE_STATUS_FAILED, detail, now);
        sourceFiles.save(source);
    }

    @Transactional
    public void importXlsxSucceeded(Job job,
                                    List<Artifact> inputArtifacts,
                                    List<Artifact> outputArtifacts,
                                    Instant now) {
        if (job.getCapability() != JobCapability.XLSX_EXTRACT) {
            return;
        }

        SourceFileJpaEntity source = resolveOrCreateSourceFile(inputArtifacts, now)
                .orElseGet(() -> {
                    log.warn("Skipping XLSX catalog import: no source file for jobId={}", job.getId());
                    return null;
                });
        if (source == null) {
            return;
        }
        if (SOURCE_STATUS_READY.equals(source.getStatus())) {
            log.info("Ignoring XLSX catalog import for already READY sourceFileId={}", source.getId());
            return;
        }

        ImportPlan plan;
        try {
            plan = buildXlsxImportPlan(job, source, outputArtifacts);
        } catch (XlsxCatalogImportException ex) {
            log.warn(
                    "XLSX catalog import failed sourceFileId={} jobId={} reason={}",
                    source.getId(), job.getId(), ex.getMessage());
            transitionSource(source, SOURCE_STATUS_FAILED, ex.getMessage(), now);
            sourceFiles.save(source);
            return;
        }

        if (!plan.valid()) {
            log.warn(
                    "XLSX catalog import failed sourceFileId={} jobId={} reason={}",
                    source.getId(), job.getId(), plan.errorMessage());
            transitionSource(source, SOURCE_STATUS_FAILED, plan.errorMessage(), now);
            sourceFiles.save(source);
            return;
        }

        try {
            Map<String, ExtractedArtifactJpaEntity> extractedByArtifactId = upsertExtractedArtifacts(plan, now);
            DocumentIngestionContext documentContext =
                    upsertDocumentContext(source, plan, extractedByArtifactId, now);
            for (SearchUnitDraft draft : plan.searchUnits()) {
                String resolvedArtifactId = draft.extractedArtifactId();
                if (resolvedArtifactId != null && !extractedByArtifactId.containsKey(resolvedArtifactId)) {
                    log.warn(
                            "Skipping XLSX search unit import with missing extracted artifact sourceFileId={} unitType={} unitKey={}",
                            source.getId(), draft.unitType(), draft.unitKey());
                    continue;
                }
                upsertSearchUnit(source, documentContext, draft, now);
            }
            populateXlsxNormalizedMetadata(source, documentContext, plan, now);
        } catch (RuntimeException ex) {
            log.warn(
                    "XLSX catalog import write failed sourceFileId={} jobId={}: {}",
                    source.getId(), job.getId(), ex.toString());
            transitionSource(source, SOURCE_STATUS_FAILED, "XLSX catalog write failed: " + ex.getMessage(), now);
            sourceFiles.save(source);
            return;
        }

        transitionSource(source, SOURCE_STATUS_READY, plan.warningMessage(), now);
        sourceFiles.save(source);
    }

    @Transactional
    public void markXlsxFailed(Job job,
                               List<Artifact> inputArtifacts,
                               Instant now) {
        if (job.getCapability() != JobCapability.XLSX_EXTRACT) {
            return;
        }
        Optional<SourceFileJpaEntity> maybeSource = resolveOrCreateSourceFile(inputArtifacts, now);
        if (maybeSource.isEmpty()) {
            log.warn("XLSX failure callback could not resolve source file jobId={}", job.getId());
            return;
        }
        SourceFileJpaEntity source = maybeSource.get();
        if (SOURCE_STATUS_READY.equals(source.getStatus())) {
            log.info("Ignoring late XLSX failure for READY sourceFileId={}", source.getId());
            return;
        }
        String detail = job.getErrorMessage() == null || job.getErrorMessage().isBlank()
                ? job.getErrorCode()
                : job.getErrorMessage();
        transitionSource(source, SOURCE_STATUS_FAILED, detail, now);
        sourceFiles.save(source);
    }

    @Transactional
    public void importPdfSucceeded(Job job,
                                   List<Artifact> inputArtifacts,
                                   List<Artifact> outputArtifacts,
                                   Instant now) {
        if (job.getCapability() != JobCapability.PDF_EXTRACT) {
            return;
        }

        SourceFileJpaEntity source = resolveOrCreateSourceFile(inputArtifacts, now)
                .orElseGet(() -> {
                    log.warn("Skipping PDF catalog import: no source file for jobId={}", job.getId());
                    return null;
                });
        if (source == null) {
            return;
        }
        if (SOURCE_STATUS_READY.equals(source.getStatus())) {
            log.info("Ignoring PDF catalog import for already READY sourceFileId={}", source.getId());
            return;
        }

        ImportPlan plan;
        try {
            plan = buildPdfImportPlan(job, source, outputArtifacts);
        } catch (PdfCatalogImportException ex) {
            log.warn(
                    "PDF catalog import failed sourceFileId={} jobId={} reason={}",
                    source.getId(), job.getId(), ex.getMessage());
            transitionSource(source, SOURCE_STATUS_FAILED, ex.getMessage(), now);
            sourceFiles.save(source);
            return;
        }

        if (!plan.valid()) {
            log.warn(
                    "PDF catalog import failed sourceFileId={} jobId={} reason={}",
                    source.getId(), job.getId(), plan.errorMessage());
            transitionSource(source, SOURCE_STATUS_FAILED, plan.errorMessage(), now);
            sourceFiles.save(source);
            return;
        }

        try {
            Map<String, ExtractedArtifactJpaEntity> extractedByArtifactId = upsertExtractedArtifacts(plan, now);
            DocumentIngestionContext documentContext =
                    upsertDocumentContext(source, plan, extractedByArtifactId, now);
            for (SearchUnitDraft draft : plan.searchUnits()) {
                String resolvedArtifactId = draft.extractedArtifactId();
                if (resolvedArtifactId != null && !extractedByArtifactId.containsKey(resolvedArtifactId)) {
                    log.warn(
                            "Skipping PDF search unit import with missing extracted artifact sourceFileId={} unitType={} unitKey={}",
                            source.getId(), draft.unitType(), draft.unitKey());
                    continue;
                }
                upsertSearchUnit(source, documentContext, draft, now);
            }
            populatePdfPageMetadata(source, documentContext, plan);
        } catch (RuntimeException ex) {
            log.warn(
                    "PDF catalog import write failed sourceFileId={} jobId={}: {}",
                    source.getId(), job.getId(), ex.toString());
            transitionSource(source, SOURCE_STATUS_FAILED, "PDF catalog write failed: " + ex.getMessage(), now);
            sourceFiles.save(source);
            return;
        }

        transitionSource(source, SOURCE_STATUS_READY, plan.warningMessage(), now);
        sourceFiles.save(source);
    }

    @Transactional
    public void markPdfFailed(Job job,
                              List<Artifact> inputArtifacts,
                              Instant now) {
        if (job.getCapability() != JobCapability.PDF_EXTRACT) {
            return;
        }
        Optional<SourceFileJpaEntity> maybeSource = resolveOrCreateSourceFile(inputArtifacts, now);
        if (maybeSource.isEmpty()) {
            log.warn("PDF failure callback could not resolve source file jobId={}", job.getId());
            return;
        }
        SourceFileJpaEntity source = maybeSource.get();
        if (SOURCE_STATUS_READY.equals(source.getStatus())) {
            log.info("Ignoring late PDF failure for READY sourceFileId={}", source.getId());
            return;
        }
        String detail = job.getErrorMessage() == null || job.getErrorMessage().isBlank()
                ? job.getErrorCode()
                : job.getErrorMessage();
        transitionSource(source, SOURCE_STATUS_FAILED, detail, now);
        sourceFiles.save(source);
    }

    @Transactional(readOnly = true)
    public List<SearchResult> search(String query, int limit) {
        String normalized = query == null ? "" : query.trim();
        if (normalized.isEmpty()) {
            return List.of();
        }
        int safeLimit = Math.max(1, Math.min(limit, 50));
        List<SearchUnitJpaEntity> units = searchUnits.searchByText(
                normalized,
                PageRequest.of(0, safeLimit));
        List<String> terms = queryTerms(normalized);
        if (terms.size() > 1) {
            units = mergeAllTermMatches(units, terms, safeLimit);
        }
        Map<String, SourceFileJpaEntity> sourcesById = new LinkedHashMap<>();
        for (SourceFileJpaEntity source : sourceFiles.findAllById(
                units.stream().map(SearchUnitJpaEntity::getSourceFileId).distinct().toList())) {
            sourcesById.put(source.getId(), source);
        }
        return units.stream()
                .map(unit -> new SearchResult(sourcesById.get(unit.getSourceFileId()), unit))
                .filter(result -> result.sourceFile() != null)
                .toList();
    }

    private List<SearchUnitJpaEntity> mergeAllTermMatches(List<SearchUnitJpaEntity> exactPhraseUnits,
                                                          List<String> terms,
                                                          int safeLimit) {
        Map<String, SearchUnitJpaEntity> merged = new LinkedHashMap<>();
        for (String term : terms) {
            List<SearchUnitJpaEntity> termUnits = searchUnits.searchByText(term, PageRequest.of(0, Math.max(100, safeLimit * 10)));
            for (SearchUnitJpaEntity unit : termUnits) {
                if (containsAllTerms(unit, terms)) {
                    merged.putIfAbsent(unit.getId(), unit);
                }
            }
        }
        for (SearchUnitJpaEntity unit : exactPhraseUnits) {
            merged.putIfAbsent(unit.getId(), unit);
        }
        return merged.values().stream()
                .limit(safeLimit)
                .toList();
    }

    private List<String> queryTerms(String query) {
        return Arrays.stream(query.split("\\s+"))
                .map(String::trim)
                .filter(term -> term.length() >= 2)
                .distinct()
                .toList();
    }

    private boolean containsAllTerms(SearchUnitJpaEntity unit, List<String> terms) {
        String haystack = String.join("\n",
                nullToEmpty(unit.getTextContent()),
                nullToEmpty(unit.getBm25Text()),
                nullToEmpty(unit.getDisplayText()),
                nullToEmpty(unit.getCitationText()),
                nullToEmpty(unit.getDebugText()),
                nullToEmpty(unit.getLocationJson()),
                nullToEmpty(unit.getMetadataJson())).toLowerCase(Locale.ROOT);
        return terms.stream()
                .map(term -> term.toLowerCase(Locale.ROOT))
                .allMatch(haystack::contains);
    }

    private Optional<SourceFileJpaEntity> resolveOrCreateSourceFile(List<Artifact> inputArtifacts,
                                                                    Instant now) {
        for (Artifact input : inputArtifacts) {
            Optional<SourceFileJpaEntity> found = sourceFiles.findFirstByStorageUri(input.getStorageUri());
            if (found.isPresent()) {
                return found;
            }
        }
        return inputArtifacts.stream()
                .filter(input -> input.getType() == ArtifactType.INPUT_FILE)
                .findFirst()
                .map(input -> registerProcessingSourceFile(
                        fileNameFromStorageUri(input.getStorageUri()),
                        input.getContentType(),
                        input.getStorageUri(),
                        now));
    }

    private ImportPlan buildImportPlan(Job job,
                                       SourceFileJpaEntity source,
                                       List<Artifact> outputArtifacts,
                                       Instant now) {
        List<Artifact> ocrArtifacts = outputArtifacts.stream()
                .filter(this::isOcrExtractArtifact)
                .toList();
        Optional<Artifact> resultJson = ocrArtifacts.stream()
                .filter(artifact -> artifact.getType() == ArtifactType.OCR_RESULT_JSON)
                .findFirst();
        if (resultJson.isEmpty()) {
            return ImportPlan.invalid("OCR_RESULT_JSON output artifact is required.");
        }

        ParsedOcrJson parsed = readOcrResultJson(resultJson.get());
        String pipelineVersion = textOrDefault(parsed.root().path("pipelineVersion"), OCR_LITE_PIPELINE_VERSION);
        String artifactKey = job.getId().value();
        List<ArtifactPayload> extracted = ocrArtifacts.stream()
                .map(artifact -> new ArtifactPayload(
                        artifact,
                        artifactKey,
                        artifact.getType() == ArtifactType.OCR_RESULT_JSON ? parsed.rawJson() : null,
                        pipelineVersion))
                .toList();

        List<SearchUnitDraft> units = buildOcrSearchUnitDrafts(
                source,
                resultJson.get().getId().value(),
                parsed.root(),
                now);
        String warning = units.stream().noneMatch(unit -> SEARCH_UNIT_PAGE.equals(unit.unitType()))
                ? "OCR_RESULT_JSON contained no page SearchUnits; saved artifacts only."
                : null;
        return ImportPlan.valid(source.getId(), extracted, units, warning);
    }

    private ImportPlan buildXlsxImportPlan(Job job,
                                           SourceFileJpaEntity source,
                                           List<Artifact> outputArtifacts) {
        List<Artifact> xlsxArtifacts = outputArtifacts.stream()
                .filter(this::isXlsxExtractArtifact)
                .toList();
        Optional<Artifact> workbookJson = xlsxArtifacts.stream()
                .filter(artifact -> artifact.getType() == ArtifactType.XLSX_WORKBOOK_JSON)
                .findFirst();
        if (workbookJson.isEmpty()) {
            return ImportPlan.invalid("XLSX_WORKBOOK_JSON output artifact is required.");
        }

        ParsedXlsxJson parsed = readXlsxWorkbookJson(workbookJson.get());
        String pipelineVersion = textOrDefault(parsed.root().path("pipelineVersion"), XLSX_PIPELINE_VERSION);
        String artifactKey = job.getId().value();
        List<ArtifactPayload> extracted = xlsxArtifacts.stream()
                .map(artifact -> new ArtifactPayload(
                        artifact,
                        artifactKey,
                        artifact.getType() == ArtifactType.XLSX_WORKBOOK_JSON ? parsed.rawJson() : null,
                        pipelineVersion))
                .toList();

        List<SearchUnitDraft> units = buildXlsxSearchUnitDrafts(
                source,
                workbookJson.get().getId().value(),
                parsed.root());
        String warning = units.stream().noneMatch(unit -> SEARCH_UNIT_SECTION.equals(unit.unitType()))
                ? "XLSX_WORKBOOK_JSON contained no visible sheet SearchUnits; saved artifacts only."
                : null;
        return ImportPlan.valid(source.getId(), extracted, units, warning);
    }

    private ImportPlan buildPdfImportPlan(Job job,
                                          SourceFileJpaEntity source,
                                          List<Artifact> outputArtifacts) {
        List<Artifact> pdfArtifacts = outputArtifacts.stream()
                .filter(this::isPdfExtractArtifact)
                .toList();
        Optional<Artifact> parsedJson = pdfArtifacts.stream()
                .filter(artifact -> artifact.getType() == ArtifactType.PDF_PARSED_JSON)
                .findFirst();
        if (parsedJson.isEmpty()) {
            return ImportPlan.invalid("PDF_PARSED_JSON output artifact is required.");
        }

        ParsedPdfJson parsed = readPdfParsedJson(parsedJson.get());
        String pipelineVersion = firstNonBlank(
                firstText(parsed.root(), "pipelineVersion", "parserVersion", "parser_version"),
                PDF_PIPELINE_VERSION);
        String artifactKey = job.getId().value();
        List<ArtifactPayload> extracted = pdfArtifacts.stream()
                .map(artifact -> new ArtifactPayload(
                        artifact,
                        artifactKey,
                        artifact.getType() == ArtifactType.PDF_PARSED_JSON ? parsed.rawJson() : null,
                        pipelineVersion))
                .toList();

        List<SearchUnitDraft> units = buildPdfSearchUnitDrafts(
                source,
                parsedJson.get().getId().value(),
                parsed.root());
        String warning = units.stream().noneMatch(unit -> SEARCH_UNIT_PAGE.equals(unit.unitType()))
                ? "PDF_PARSED_JSON contained no page SearchUnits; saved artifacts only."
                : warningFromParsedPdf(parsed.root());
        return ImportPlan.valid(source.getId(), extracted, units, warning);
    }

    private ParsedOcrJson readOcrResultJson(Artifact resultJsonArtifact) {
        try (InputStream stream = storage.openForRead(resultJsonArtifact.getStorageUri())) {
            String rawJson = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
            return new ParsedOcrJson(objectMapper.readTree(rawJson), rawJson);
        } catch (IOException | RuntimeException ex) {
            throw new OcrCatalogImportException(
                    "Failed to parse OCR_RESULT_JSON for artifact "
                            + resultJsonArtifact.getId().value());
        }
    }

    private ParsedXlsxJson readXlsxWorkbookJson(Artifact workbookJsonArtifact) {
        try (InputStream stream = storage.openForRead(workbookJsonArtifact.getStorageUri())) {
            String rawJson = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
            return new ParsedXlsxJson(objectMapper.readTree(rawJson), rawJson);
        } catch (IOException | RuntimeException ex) {
            throw new XlsxCatalogImportException(
                    "Failed to parse XLSX_WORKBOOK_JSON for artifact "
                            + workbookJsonArtifact.getId().value());
        }
    }

    private ParsedPdfJson readPdfParsedJson(Artifact parsedJsonArtifact) {
        try (InputStream stream = storage.openForRead(parsedJsonArtifact.getStorageUri())) {
            String rawJson = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
            return new ParsedPdfJson(objectMapper.readTree(rawJson), rawJson);
        } catch (IOException | RuntimeException ex) {
            throw new PdfCatalogImportException(
                    "Failed to parse PDF_PARSED_JSON for artifact "
                            + parsedJsonArtifact.getId().value());
        }
    }

    private Map<String, ExtractedArtifactJpaEntity> upsertExtractedArtifacts(ImportPlan plan, Instant now) {
        Map<String, ExtractedArtifactJpaEntity> byArtifactId = new HashMap<>();
        for (ArtifactPayload payload : plan.extractedArtifacts()) {
            Artifact artifact = payload.artifact();
            String artifactType = artifact.getType().name();
            ExtractedArtifactJpaEntity entity = extractedArtifacts
                    .findBySourceFileIdAndArtifactTypeAndArtifactKey(
                            plan.sourceFileId(),
                            artifactType,
                            payload.artifactKey())
                    .map(existing -> {
                        existing.updatePayload(
                                artifact.getStorageUri(),
                                payload.pipelineVersion(),
                                artifact.getChecksumSha256(),
                                payload.payloadJson(),
                                now);
                        return existing;
                    })
                    .orElseGet(() -> new ExtractedArtifactJpaEntity(
                            artifact.getId().value(),
                            plan.sourceFileId(),
                            artifactType,
                            payload.artifactKey(),
                            artifact.getStorageUri(),
                            payload.pipelineVersion(),
                            artifact.getChecksumSha256(),
                            payload.payloadJson(),
                            artifact.getCreatedAt(),
                            now));
            ExtractedArtifactJpaEntity saved = extractedArtifacts.save(entity);
            byArtifactId.put(saved.getArtifactId(), saved);
        }
        return byArtifactId;
    }

    private DocumentIngestionContext upsertDocumentContext(SourceFileJpaEntity source,
                                                           ImportPlan plan,
                                                           Map<String, ExtractedArtifactJpaEntity> extractedByArtifactId,
                                                           Instant now) {
        String documentId = documentIdForSource(source.getId());
        String versionId = documentVersionIdForSource(source.getId());
        String sourceFileType = source.getFileType();
        String fileType = fileTypeForParsedArtifact(source, plan);
        String parserName = parserNameFor(source, fileType, plan);
        String parserVersion = parserVersionFor(plan);
        String aclTags = "[]";

        ObjectNode documentMetadata = objectMapper.createObjectNode();
        documentMetadata.put("sourceFileId", source.getId());
        documentMetadata.put("storageUri", source.getStorageUri());
        documentMetadata.put("sourceFileType", sourceFileType);

        DocumentV2JpaEntity document = documents.findById(documentId)
                .map(existing -> {
                    existing.refresh(
                            source.getOriginalFileName(),
                            "ACTIVE",
                            aclTags,
                            existing.getLatestVersionId(),
                            documentMetadata.toString(),
                            now);
                    return existing;
                })
                .orElseGet(() -> new DocumentV2JpaEntity(
                        documentId,
                        source.getOriginalFileName(),
                        "ACTIVE",
                        aclTags,
                        null,
                        documentMetadata.toString(),
                        source.getUploadedAt(),
                        now));
        documents.save(document);

        ObjectNode parserPolicy = objectMapper.createObjectNode();
        parserPolicy.put("hiddenPolicy", "exclude_hidden");
        parserPolicy.put("parserName", parserName);
        parserPolicy.put("parserVersion", parserVersion);

        DocumentVersionJpaEntity version = documentVersions.findById(versionId)
                .map(existing -> {
                    existing.refresh(
                            source.getOriginalFileName(),
                            sourceFileType,
                            source.getMimeType(),
                            source.getStorageUri(),
                            checksumForPlan(plan),
                            SOURCE_STATUS_READY,
                            plan.warningMessage(),
                            parserPolicy.toString(),
                            aclTags,
                            now);
                    return existing;
                })
                .orElseGet(() -> new DocumentVersionJpaEntity(
                        versionId,
                        documentId,
                        source.getId(),
                        1,
                        source.getOriginalFileName(),
                        sourceFileType,
                        source.getMimeType(),
                        source.getStorageUri(),
                        checksumForPlan(plan),
                        SOURCE_STATUS_READY,
                        plan.warningMessage(),
                        parserPolicy.toString(),
                        aclTags,
                        source.getUploadedAt(),
                        now));
        documentVersions.save(version);
        document.refresh(
                source.getOriginalFileName(),
                "ACTIVE",
                aclTags,
                versionId,
                documentMetadata.toString(),
                now);
        documents.save(document);

        Map<String, String> parsedArtifactIds = new HashMap<>();
        for (ExtractedArtifactJpaEntity extracted : extractedByArtifactId.values()) {
            if (!isPrimaryParsedArtifact(extracted.getArtifactType())) {
                continue;
            }
            String parsedArtifactId = parsedArtifactIdForExtractedArtifact(extracted.getArtifactId());
            String artifactJson = validJsonOrDefault(extracted.getPayloadJson(), "{}");
            String warningsJson = warningsJson(artifactJson);
            Double qualityScore = doubleOrNull(metadataOrEmpty(artifactJson), "qualityScore", "quality_score");
            ParsedArtifactV2JpaEntity parsed = parsedArtifacts.findById(parsedArtifactId)
                    .map(existing -> {
                        existing.refresh(
                                extracted.getStorageUri(),
                                parserName,
                                extracted.getPipelineVersion(),
                                fileType,
                                artifactJson,
                                warningsJson,
                                qualityScore);
                        return existing;
                    })
                    .orElseGet(() -> new ParsedArtifactV2JpaEntity(
                            parsedArtifactId,
                            versionId,
                            source.getId(),
                            extracted.getArtifactId(),
                            extracted.getArtifactType(),
                            extracted.getStorageUri(),
                            parserName,
                            extracted.getPipelineVersion(),
                            fileType,
                            artifactJson,
                            warningsJson,
                            qualityScore,
                            now));
            ParsedArtifactV2JpaEntity saved = parsedArtifacts.save(parsed);
            parsedArtifactIds.put(extracted.getArtifactId(), saved.getId());
        }
        return new DocumentIngestionContext(
                documentId,
                versionId,
                parsedArtifactIds,
                parserName,
                parserVersion,
                fileType);
    }

    private void upsertSearchUnit(SourceFileJpaEntity source,
                                  DocumentIngestionContext documentContext,
                                  SearchUnitDraft draft,
                                  Instant now) {
        String canonicalType = canonicalUnitType(draft.unitType());
        SearchUnitV2Payload v2Payload = buildSearchUnitV2Payload(source, documentContext, draft, canonicalType);
        String contentHash = sha256OrNull(firstNonBlank(v2Payload.embeddingText(), draft.textContent()));
        SearchUnitJpaEntity entity = searchUnits
                .findBySourceFileIdAndUnitTypeAndUnitKey(
                        draft.sourceFileId(),
                        canonicalType,
                        draft.unitKey())
                .map(existing -> {
                    existing.updateCanonical(
                            draft.extractedArtifactId(),
                            draft.title(),
                            draft.sectionPath(),
                            draft.pageStart(),
                            draft.pageEnd(),
                            draft.textContent(),
                            draft.metadataJson(),
                            contentHash,
                            EMBEDDING_STATUS_PENDING,
                            now);
                    return existing;
                })
                .orElseGet(() -> new SearchUnitJpaEntity(
                        newId(),
                        draft.sourceFileId(),
                        draft.extractedArtifactId(),
                        canonicalType,
                        draft.unitKey(),
                        draft.title(),
                        draft.sectionPath(),
                        draft.pageStart(),
                        draft.pageEnd(),
                        draft.textContent(),
                        draft.metadataJson(),
                        EMBEDDING_STATUS_PENDING,
                        contentHash,
                        now,
                        now));
        entity.applyIngestionV2(
                documentContext.documentId(),
                documentContext.documentVersionId(),
                documentContext.parsedArtifactIds().get(draft.extractedArtifactId()),
                source.getOriginalFileName(),
                source.getFileType(),
                v2Payload.chunkType(),
                v2Payload.locationType(),
                v2Payload.locationJson(),
                v2Payload.embeddingText(),
                v2Payload.bm25Text(),
                v2Payload.displayText(),
                v2Payload.citationText(),
                v2Payload.debugText(),
                v2Payload.parserName(),
                v2Payload.parserVersion(),
                v2Payload.qualityScore(),
                v2Payload.confidenceScore(),
                "[]",
                now);
        searchUnits.save(entity);
    }

    private void populatePdfPageMetadata(SourceFileJpaEntity source,
                                         DocumentIngestionContext documentContext,
                                         ImportPlan plan) {
        Optional<ArtifactPayload> maybePayload = primaryPayload(plan, "PDF_PARSED_JSON");
        if (maybePayload.isEmpty()) {
            return;
        }
        ArtifactPayload payload = maybePayload.get();
        String parsedArtifactId = documentContext.parsedArtifactIds().get(payload.artifact().getId().value());
        JsonNode root = metadataOrEmpty(payload.payloadJson());
        JsonNode pages = root.path("pages");
        if (!pages.isArray()) {
            return;
        }

        Double documentQuality = doubleOrNull(root, "qualityScore", "quality_score");
        int fallbackIndex = 0;
        for (JsonNode page : pages) {
            Integer physicalPageIndex = intOrNull(page, "physicalPageIndex", "physical_page_index");
            if (physicalPageIndex == null) {
                physicalPageIndex = fallbackIndex;
            }
            Integer pageNo = intOrNull(page, "pageNo", "page_no", "pageNumber");
            if (pageNo == null) {
                pageNo = physicalPageIndex + 1;
            }
            String pageLabel = firstNonBlank(
                    firstText(page, "pageLabel", "page_label"),
                    String.valueOf(pageNo));
            JsonNode blocks = page.path("blocks");
            JsonNode tables = page.path("tables");
            String text = pageText(page);
            Boolean textLayerPresent = page.has("textLayerPresent") || page.has("text_layer_present")
                    ? page.path("textLayerPresent").asBoolean(page.path("text_layer_present").asBoolean(false))
                    : !text.isBlank();
            Boolean ocrUsed = page.path("ocrUsed").asBoolean(page.path("ocr_used").asBoolean(false));
            Double ocrConfidenceAvg = firstNonNull(
                    doubleOrNull(page, "ocrConfidenceAvg", "ocr_confidence_avg", "ocrConfidence", "ocr_confidence"),
                    averageBlockConfidence(blocks));

            ObjectNode metadata = objectMapper.createObjectNode();
            metadata.set("raw_page", page.deepCopy());
            metadata.put("parser_name", documentContext.parserName());
            metadata.put("parser_version", documentContext.parserVersion());

            Integer savedPhysicalPageIndex = physicalPageIndex;
            Integer savedPageNo = pageNo;
            PdfPageMetadataJpaEntity entity = pdfPageMetadata
                    .findByDocumentVersionIdAndPhysicalPageIndex(documentContext.documentVersionId(), savedPhysicalPageIndex)
                    .orElseGet(() -> new PdfPageMetadataJpaEntity(
                            pdfPageMetadataId(documentContext.documentVersionId(), savedPhysicalPageIndex),
                            documentContext.documentId(),
                            documentContext.documentVersionId(),
                            parsedArtifactId,
                            source.getId(),
                            savedPhysicalPageIndex,
                            savedPageNo,
                            pageLabel));
            entity.refresh(
                    documentContext.documentId(),
                    parsedArtifactId,
                    source.getId(),
                    pageNo,
                    pageLabel,
                    doubleOrNull(page, "width"),
                    doubleOrNull(page, "height"),
                    textLayerPresent,
                    ocrUsed,
                    firstText(page, "ocrEngine", "ocr_engine"),
                    firstText(page, "ocrModel", "ocr_model"),
                    ocrConfidenceAvg,
                    blocks.isArray() ? blocks.size() : 0,
                    tables.isArray() ? tables.size() : 0,
                    text.length(),
                    firstNonNull(doubleOrNull(page, "qualityScore", "quality_score"), documentQuality),
                    pageWarningsJson(root, physicalPageIndex, pageNo),
                    metadata.toString());
            pdfPageMetadata.save(entity);
            fallbackIndex++;
        }
    }

    private void populateXlsxNormalizedMetadata(SourceFileJpaEntity source,
                                                DocumentIngestionContext documentContext,
                                                ImportPlan plan,
                                                Instant now) {
        Optional<ArtifactPayload> maybePayload = primaryPayload(plan, "XLSX_WORKBOOK_JSON");
        if (maybePayload.isEmpty()) {
            return;
        }
        ArtifactPayload payload = maybePayload.get();
        String parsedArtifactId = documentContext.parsedArtifactIds().get(payload.artifact().getId().value());
        JsonNode root = metadataOrEmpty(payload.payloadJson());
        JsonNode sheets = root.path("workbook").path("sheets");
        if (!sheets.isArray()) {
            return;
        }

        Double documentQuality = doubleOrNull(root, "qualityScore", "quality_score");
        int remainingCellBudget = MAX_NORMALIZED_CELLS_PER_WORKBOOK;
        for (JsonNode sheet : sheets) {
            boolean hiddenSheet = sheet.path("hidden").asBoolean(false);
            boolean indexable = !sheet.path("indexable").isBoolean() || sheet.path("indexable").asBoolean();
            if (hiddenSheet && !sheet.path("indexable").isBoolean()) {
                indexable = false;
            }
            if (!indexable) {
                continue;
            }
            int sheetIndex = intOrNull(sheet, "index", "sheetIndex") == null
                    ? 0
                    : intOrNull(sheet, "index", "sheetIndex");
            String sheetName = textOrDefault(sheet.path("name"), "Sheet" + (sheetIndex + 1));
            Map<String, String> tableIdsByRange = new HashMap<>();
            JsonNode tables = sheet.path("tables");
            if (tables.isArray() && tables.size() > 0) {
                for (JsonNode table : tables) {
                    String metadataId = upsertXlsxTableMetadata(
                            source,
                            documentContext,
                            parsedArtifactId,
                            sheet,
                            table,
                            sheetIndex,
                            sheetName,
                            hiddenSheet,
                            documentQuality,
                            now);
                    String cellRange = firstText(table, "cellRange", "range");
                    if (cellRange != null) {
                        tableIdsByRange.put(normalizeRangeForKey(cellRange), metadataId);
                    }
                }
            }
            JsonNode chunks = sheet.path("chunks");
            if (chunks.isArray() && chunks.size() > 0) {
                for (JsonNode chunk : chunks) {
                    String metadataId = upsertXlsxDetectedRegionMetadata(
                            source,
                            documentContext,
                            parsedArtifactId,
                            sheet,
                            chunk,
                            sheetIndex,
                            sheetName,
                            hiddenSheet,
                            documentQuality,
                            now);
                    String cellRange = firstText(chunk, "cellRange", "range");
                    if (cellRange != null) {
                        tableIdsByRange.put(normalizeRangeForKey(cellRange), metadataId);
                    }
                }
            }
            if (remainingCellBudget > 0) {
                remainingCellBudget -= populateXlsxCellMetadata(
                        source,
                        documentContext,
                        parsedArtifactId,
                        sheet,
                        sheetIndex,
                        sheetName,
                        tableIdsByRange,
                        documentQuality,
                        remainingCellBudget);
            }
        }
    }

    private String upsertXlsxTableMetadata(SourceFileJpaEntity source,
                                           DocumentIngestionContext documentContext,
                                           String parsedArtifactId,
                                           JsonNode sheet,
                                           JsonNode table,
                                           int sheetIndex,
                                           String sheetName,
                                           boolean hiddenSheet,
                                           Double documentQuality,
                                           Instant now) {
        String tableName = firstText(table, "name", "tableName");
        String tableId = firstNonBlank(firstText(table, "tableId", "id"), tableName);
        if (tableId == null || tableId.isBlank()) {
            tableId = "table-" + sheetIndex + "-" + firstNonBlank(firstText(table, "tableIndex"), "0");
        }
        String cellRange = firstText(table, "cellRange", "range");
        String normalizedTableId = normalizeUnitKeyPart(tableId);
        String metadataId = tableMetadataId(documentContext.documentVersionId(), sheetIndex, normalizedTableId, cellRange);
        TableMetadataJpaEntity entity = tableMetadata.findById(metadataId)
                .orElseGet(() -> new TableMetadataJpaEntity(metadataId, normalizedTableId, now));
        entity.refresh(
                null,
                documentContext.documentId(),
                documentContext.documentVersionId(),
                parsedArtifactId,
                source.getId(),
                sheetIndex,
                sheetName,
                normalizedTableId,
                tableName,
                firstNonBlank(tableName, normalizedTableId),
                cellRange,
                headerRange(cellRange),
                dataRange(cellRange),
                intOrNull(table, "rowCount"),
                intOrNull(table, "columnCount"),
                "exclude_hidden",
                hiddenSheet,
                firstNonBlank(firstText(table, "type", "detectedTableType"), "named"),
                headerJson(table, sheet).toString(),
                headerJson(table, sheet).toString(),
                xlsxTableLocationJson(documentContext, sheetIndex, sheetName, normalizedTableId, cellRange, "exclude_hidden"),
                firstNonNull(doubleOrNull(table, "qualityScore", "quality_score"), documentQuality));
        tableMetadata.save(entity);
        return metadataId;
    }

    private String upsertXlsxDetectedRegionMetadata(SourceFileJpaEntity source,
                                                    DocumentIngestionContext documentContext,
                                                    String parsedArtifactId,
                                                    JsonNode sheet,
                                                    JsonNode chunk,
                                                    int sheetIndex,
                                                    String sheetName,
                                                    boolean hiddenSheet,
                                                    Double documentQuality,
                                                    Instant now) {
        String cellRange = firstText(chunk, "cellRange", "range");
        String chunkIndex = firstNonBlank(firstText(chunk, "chunkIndex"), "0");
        ObjectNode detected = objectMapper.createObjectNode();
        detected.put("tableId", "detected-" + sheetIndex + "-" + chunkIndex);
        detected.put("type", "row_group");
        copyIfPresent(detected, "rowStart", chunk.path("rowStart"));
        copyIfPresent(detected, "rowEnd", chunk.path("rowEnd"));
        copyIfPresent(detected, "columnStart", chunk.path("columnStart"));
        copyIfPresent(detected, "columnEnd", chunk.path("columnEnd"));
        putIfPresent(detected, "cellRange", cellRange);
        return upsertXlsxTableMetadata(
                source,
                documentContext,
                parsedArtifactId,
                sheet,
                detected,
                sheetIndex,
                sheetName,
                hiddenSheet,
                documentQuality,
                now);
    }

    private int populateXlsxCellMetadata(SourceFileJpaEntity source,
                                          DocumentIngestionContext documentContext,
                                          String parsedArtifactId,
                                          JsonNode sheet,
                                          int sheetIndex,
                                          String sheetName,
                                          Map<String, String> tableIdsByRange,
                                          Double documentQuality,
                                          int remainingWorkbookBudget) {
        Map<String, JsonNode> formulasByCell = formulasByCell(visibleFormulas(sheet));
        Map<String, String> mergedRangesByCell = mergedRangesByCell(sheet.path("mergedCells"));
        Map<Integer, String> headersByColumn = headersByColumn(sheet, sheet.path("tables"));
        Set<String> hiddenCells = hiddenCellRefs(sheet);
        Set<Integer> hiddenRows = hiddenRowIndexes(sheet);
        Set<Integer> hiddenColumns = hiddenColumnIndexes(sheet);
        int sheetBudget = Math.max(0, Math.min(MAX_NORMALIZED_CELLS_PER_SHEET, remainingWorkbookBudget));
        int savedCount = 0;

        JsonNode cells = sheet.path("cells");
        if (cells.isArray()) {
            for (JsonNode cell : cells) {
                if (savedCount >= sheetBudget) {
                    break;
                }
                if (isHiddenXlsxCell(cell, hiddenRows, hiddenColumns, hiddenCells)) {
                    continue;
                }
                String cellRef = firstText(cell, "cell", "cellRef", "cell_ref", "address");
                if (cellRef == null || cellRef.isBlank()) {
                    continue;
                }
                String normalizedCellRef = normalizeRangeForKey(cellRef);
                if (hiddenCells.contains(normalizedCellRef)) {
                    continue;
                }
                JsonNode formula = formulasByCell.get(normalizedCellRef);
                boolean merged = mergedRangesByCell.containsKey(normalizedCellRef);
                boolean header = isHeaderCell(cell, sheet.path("tables"));
                if (!header && formula == null && !merged) {
                    CellRangeParts parts = parseCellRange(firstText(cell, "range", "cellRange"));
                    if (parts == null && savedCount > 2_000) {
                        continue;
                    }
                }
                upsertXlsxCellMetadata(
                        source,
                        documentContext,
                        parsedArtifactId,
                        sheetIndex,
                        sheetName,
                        cell,
                        formula,
                        mergedRangesByCell.get(normalizedCellRef),
                        sheet.path("tables"),
                        tableIdsByRange,
                        headersByColumn,
                        documentQuality);
                savedCount++;
            }
        }

        for (Map.Entry<String, JsonNode> entry : formulasByCell.entrySet()) {
            if (savedCount >= sheetBudget) {
                break;
            }
            if (cells.isArray() && hasCell(cells, entry.getKey())) {
                continue;
            }
            if (hiddenCells.contains(normalizeRangeForKey(entry.getKey()))) {
                continue;
            }
            ObjectNode synthetic = objectMapper.createObjectNode();
            synthetic.put("cell", entry.getKey());
            CellRef ref = parseCellRef(entry.getKey());
            if (ref != null) {
                synthetic.put("row", ref.row());
                synthetic.put("column", ref.column());
            }
            putIfPresent(synthetic, "value", firstText(entry.getValue(), "cachedValue", "value"));
            upsertXlsxCellMetadata(
                    source,
                    documentContext,
                    parsedArtifactId,
                    sheetIndex,
                    sheetName,
                    synthetic,
                    entry.getValue(),
                    mergedRangesByCell.get(normalizeRangeForKey(entry.getKey())),
                    sheet.path("tables"),
                    tableIdsByRange,
                    headersByColumn,
                    documentQuality);
            savedCount++;
        }
        return savedCount;
    }

    private void upsertXlsxCellMetadata(SourceFileJpaEntity source,
                                        DocumentIngestionContext documentContext,
                                        String parsedArtifactId,
                                        int sheetIndex,
                                        String sheetName,
                                        JsonNode cell,
                                        JsonNode formulaNode,
                                        String mergedRange,
                                        JsonNode tables,
                                        Map<String, String> tableIdsByRange,
                                        Map<Integer, String> headersByColumn,
                                        Double documentQuality) {
        String cellRef = firstText(cell, "cell", "cellRef", "cell_ref", "address");
        if (cellRef == null || cellRef.isBlank()) {
            return;
        }
        CellRef parsedRef = parseCellRef(cellRef);
        Integer rowIndex = firstNonNull(intOrNull(cell, "row", "rowIndex", "row_index"), parsedRef == null ? null : parsedRef.row());
        Integer columnIndex = firstNonNull(intOrNull(cell, "column", "columnIndex", "column_index"), parsedRef == null ? null : parsedRef.column());
        String columnLetter = firstNonBlank(firstText(cell, "columnLetter", "column_letter"), columnIndex == null ? null : columnLetter(columnIndex));
        String formula = formulaNode == null ? firstText(cell, "formula") : firstText(formulaNode, "formula");
        String cachedValue = formulaNode == null ? firstText(cell, "cachedValue", "cached_value") : firstText(formulaNode, "cachedValue", "cached_value", "value");
        String formattedValue = firstNonBlank(firstText(cell, "formattedValue", "formatted_value", "value"), cachedValue);
        String rawValue = firstNonBlank(firstText(cell, "rawValue", "raw_value"), formattedValue);
        String dataType = firstText(cell, "dataType", "data_type");
        String numberFormat = firstText(cell, "numberFormat", "number_format");
        String tableId = tableIdForCell(cellRef, tables);
        String tableMetadataId = tableMetadataIdForCell(cellRef, tableIdsByRange, tables);
        ArrayNode headerPath = objectMapper.createArrayNode();
        String header = columnIndex == null ? null : headersByColumn.get(columnIndex);
        if (header != null && !header.isBlank()) {
            headerPath.add(header);
        }
        ObjectNode columnLabel = objectMapper.createObjectNode();
        putIfPresent(columnLabel, "column_letter", columnLetter);
        putIfPresent(columnLabel, "header", header);

        String metadataId = cellMetadataId(documentContext.documentVersionId(), sheetIndex, tableId, cellRef);
        CellMetadataJpaEntity entity = cellMetadata.findById(metadataId)
                .orElseGet(() -> new CellMetadataJpaEntity(metadataId));
        entity.refresh(
                documentContext.documentId(),
                documentContext.documentVersionId(),
                parsedArtifactId,
                source.getId(),
                tableMetadataId,
                sheetName,
                sheetIndex,
                cellRef,
                rowIndex,
                columnIndex,
                columnLetter,
                rawValue,
                formattedValue,
                formula,
                cachedValue,
                dataType,
                numberFormat,
                headerPath.toString(),
                "[]",
                columnLabel.toString(),
                tableId,
                cell.path("hiddenRow").asBoolean(cell.path("hidden_row").asBoolean(false)),
                cell.path("hiddenColumn").asBoolean(cell.path("hidden_column").asBoolean(false)),
                mergedRange != null,
                mergedRange,
                firstNonNull(doubleOrNull(cell, "qualityScore", "quality_score"), documentQuality));
        cellMetadata.save(entity);
    }

    private Optional<ArtifactPayload> primaryPayload(ImportPlan plan, String artifactType) {
        return plan.extractedArtifacts().stream()
                .filter(payload -> artifactType.equals(payload.artifact().getType().name()))
                .findFirst();
    }

    private List<SearchUnitDraft> buildOcrSearchUnitDrafts(SourceFileJpaEntity source,
                                                           String resultJsonArtifactId,
                                                           JsonNode root,
                                                           Instant now) {
        String engine = firstText(root, "engine", "engineName");
        String pipelineVersion = textOrDefault(root.path("pipelineVersion"), OCR_LITE_PIPELINE_VERSION);
        String sourceRecordId = textOrNull(root.path("sourceRecordId"));
        JsonNode pages = root.path("pages");

        List<SearchUnitDraft> units = new ArrayList<>();
        List<String> pageTexts = new ArrayList<>();
        Integer firstPage = null;
        Integer lastPage = null;

        if (pages.isArray()) {
            int fallbackPageNo = 1;
            Map<Integer, Integer> tableCounters = new HashMap<>();
            Map<Integer, Integer> imageCounters = new HashMap<>();
            for (JsonNode page : pages) {
                int pageNo = pageNumber(page, fallbackPageNo);
                firstPage = firstPage == null ? pageNo : Math.min(firstPage, pageNo);
                lastPage = lastPage == null ? pageNo : Math.max(lastPage, pageNo);

                JsonNode blocks = page.path("blocks");
                String pageText = pageText(page);
                if (!pageText.isBlank()) {
                    pageTexts.add(pageText);
                }

                ObjectNode metadata = baseMetadata(engine, pipelineVersion, sourceRecordId);
                metadata.put("pageNo", pageNo);
                metadata.put("blockCount", blocks.isArray() ? blocks.size() : 0);
                putIfPresent(metadata, "confidence", firstText(page, "confidence", "avgConfidence"));
                metadata.set("rawPage", page.deepCopy());

                units.add(new SearchUnitDraft(
                        source.getId(),
                        resultJsonArtifactId,
                        SEARCH_UNIT_PAGE,
                        "page:" + pageNo,
                        null,
                        null,
                        pageNo,
                        pageNo,
                        blankToNull(pageText),
                        metadata.toString()));

                units.addAll(sectionUnits(source, resultJsonArtifactId, page.path("sections"), pageNo, engine, pipelineVersion, sourceRecordId));
                units.addAll(tableUnits(source, resultJsonArtifactId, page.path("tables"), pageNo, tableCounters, engine, pipelineVersion, sourceRecordId));
                units.addAll(imageUnits(source, resultJsonArtifactId, firstPresent(page.path("images"), page.path("figures")), pageNo, imageCounters, engine, pipelineVersion, sourceRecordId));
                fallbackPageNo++;
            }
        }

        units.addAll(sectionUnits(source, resultJsonArtifactId, root.path("sections"), null, engine, pipelineVersion, sourceRecordId));
        units.addAll(tableUnits(source, resultJsonArtifactId, root.path("tables"), null, new HashMap<>(), engine, pipelineVersion, sourceRecordId));
        units.addAll(imageUnits(source, resultJsonArtifactId, firstPresent(root.path("images"), root.path("figures")), null, new HashMap<>(), engine, pipelineVersion, sourceRecordId));

        String documentText = firstText(root, "plainText", "text", "documentText");
        if (documentText == null || documentText.isBlank()) {
            documentText = String.join("\n\n", pageTexts).trim();
        }
        boolean truncated = false;
        if (documentText != null && documentText.length() > DOCUMENT_TEXT_MAX_CHARS) {
            documentText = documentText.substring(0, DOCUMENT_TEXT_MAX_CHARS);
            truncated = true;
        }

        ObjectNode documentMetadata = baseMetadata(engine, pipelineVersion, sourceRecordId);
        documentMetadata.put("pageCount", pageCount(root, pages, pageTexts.size()));
        documentMetadata.put("resultArtifactId", resultJsonArtifactId);
        documentMetadata.put("textTruncated", truncated);
        if (pages.isArray() && pages.size() == 0) {
            documentMetadata.put("warning", "OCR_RESULT_JSON pages array is empty.");
        } else if (!pages.isArray()) {
            documentMetadata.put("warning", "OCR_RESULT_JSON has no pages array.");
        }

        units.add(0, new SearchUnitDraft(
                source.getId(),
                resultJsonArtifactId,
                SEARCH_UNIT_DOCUMENT,
                "document",
                source.getOriginalFileName(),
                null,
                firstPage,
                lastPage,
                blankToNull(documentText),
                documentMetadata.toString()));
        return units;
    }

    private List<SearchUnitDraft> sectionUnits(SourceFileJpaEntity source,
                                               String resultJsonArtifactId,
                                               JsonNode sections,
                                               Integer pageNo,
                                               String engine,
                                               String pipelineVersion,
                                               String sourceRecordId) {
        if (!sections.isArray()) {
            return List.of();
        }
        List<SearchUnitDraft> units = new ArrayList<>();
        int index = 1;
        for (JsonNode section : sections) {
            String title = firstText(section, "title", "heading", "name");
            String sectionPath = sectionPath(section, title, pageNo, index);
            String text = firstText(section, "text", "content");
            if ((title == null || title.isBlank()) && (text == null || text.isBlank())) {
                index++;
                continue;
            }
            Integer start = intOrNull(section, "pageStart", "startPage", "pageNo", "pageNumber");
            Integer end = intOrNull(section, "pageEnd", "endPage");
            if (start == null) {
                start = pageNo;
            }
            if (end == null) {
                end = start;
            }
            String explicitId = firstText(section, "id", "sectionId");
            String unitKey = explicitId == null || explicitId.isBlank()
                    ? "section:" + stableHash(sectionPath)
                    : "section:" + explicitId;
            ObjectNode metadata = baseMetadata(engine, pipelineVersion, sourceRecordId);
            metadata.put("sectionIndex", index);
            if (pageNo != null) {
                metadata.put("pageNo", pageNo);
            }
            metadata.set("rawSection", section.deepCopy());
            units.add(new SearchUnitDraft(
                    source.getId(),
                    resultJsonArtifactId,
                    SEARCH_UNIT_SECTION,
                    unitKey,
                    title,
                    sectionPath,
                    start,
                    end,
                    blankToNull(text),
                    metadata.toString()));
            index++;
        }
        return units;
    }

    private List<SearchUnitDraft> tableUnits(SourceFileJpaEntity source,
                                             String resultJsonArtifactId,
                                             JsonNode tables,
                                             Integer pageNo,
                                             Map<Integer, Integer> counters,
                                             String engine,
                                             String pipelineVersion,
                                             String sourceRecordId) {
        if (!tables.isArray()) {
            return List.of();
        }
        List<SearchUnitDraft> units = new ArrayList<>();
        for (JsonNode table : tables) {
            int resolvedPage = pageNo == null ? pageNumber(table, 1) : pageNo;
            int tableIndex = counters.merge(resolvedPage, 1, Integer::sum);
            String tableId = firstText(table, "id", "tableId");
            String unitKey = tableId == null || tableId.isBlank()
                    ? "page:" + resolvedPage + ":table:" + tableIndex
                    : "table:" + tableId;
            ObjectNode metadata = baseMetadata(engine, pipelineVersion, sourceRecordId);
            metadata.put("pageNo", resolvedPage);
            metadata.put("tableIndex", tableIndex);
            putIfPresent(metadata, "rowCount", firstText(table, "rowCount", "rowsCount"));
            putIfPresent(metadata, "columnCount", firstText(table, "columnCount", "columnsCount"));
            metadata.set("rawTable", table.deepCopy());
            units.add(new SearchUnitDraft(
                    source.getId(),
                    resultJsonArtifactId,
                    SEARCH_UNIT_TABLE,
                    unitKey,
                    firstText(table, "title", "caption"),
                    null,
                    resolvedPage,
                    resolvedPage,
                    blankToNull(tableText(table)),
                    metadata.toString()));
        }
        return units;
    }

    private List<SearchUnitDraft> imageUnits(SourceFileJpaEntity source,
                                             String resultJsonArtifactId,
                                             JsonNode images,
                                             Integer pageNo,
                                             Map<Integer, Integer> counters,
                                             String engine,
                                             String pipelineVersion,
                                             String sourceRecordId) {
        if (!images.isArray()) {
            return List.of();
        }
        List<SearchUnitDraft> units = new ArrayList<>();
        for (JsonNode image : images) {
            int resolvedPage = pageNo == null ? pageNumber(image, 1) : pageNo;
            int imageIndex = counters.merge(resolvedPage, 1, Integer::sum);
            String imageId = firstText(image, "id", "imageId", "figureId");
            String unitKey = imageId == null || imageId.isBlank()
                    ? "page:" + resolvedPage + ":image:" + imageIndex
                    : "image:" + imageId;
            String caption = firstText(image, "caption", "altText", "surroundingText", "text");
            ObjectNode metadata = baseMetadata(engine, pipelineVersion, sourceRecordId);
            metadata.put("pageNo", resolvedPage);
            metadata.put("imageIndex", imageIndex);
            putIfPresent(metadata, "captionSource", firstText(image, "captionSource", "source"));
            metadata.set("rawImage", image.deepCopy());
            units.add(new SearchUnitDraft(
                    source.getId(),
                    resultJsonArtifactId,
                    SEARCH_UNIT_IMAGE,
                    unitKey,
                    firstText(image, "title", "caption"),
                    null,
                    resolvedPage,
                    resolvedPage,
                    blankToNull(caption),
                    metadata.toString()));
        }
        return units;
    }

    private List<SearchUnitDraft> buildXlsxSearchUnitDrafts(SourceFileJpaEntity source,
                                                            String workbookJsonArtifactId,
                                                            JsonNode root) {
        String fileType = textOrDefault(root.path("fileType"), "xlsx");
        String pipelineVersion = textOrDefault(root.path("pipelineVersion"), XLSX_PIPELINE_VERSION);
        String sourceRecordId = textOrNull(root.path("sourceRecordId"));
        JsonNode workbook = root.path("workbook");
        JsonNode sheets = workbook.path("sheets");

        List<SearchUnitDraft> units = new ArrayList<>();
        List<String> visibleSheetTexts = new ArrayList<>();
        List<String> visibleSheetNames = new ArrayList<>();
        int sheetCount = intOrNull(workbook, "sheetCount") == null
                ? (sheets.isArray() ? sheets.size() : 0)
                : intOrNull(workbook, "sheetCount");

        if (sheets.isArray()) {
            int fallbackIndex = 0;
            for (JsonNode sheet : sheets) {
                int sheetIndex = intOrNull(sheet, "index") == null
                        ? fallbackIndex
                        : intOrNull(sheet, "index");
                fallbackIndex++;
                boolean hidden = sheet.path("hidden").asBoolean(false);
                boolean indexable = !sheet.path("indexable").isBoolean() || sheet.path("indexable").asBoolean();
                if (hidden && !sheet.path("indexable").isBoolean()) {
                    indexable = false;
                }
                String sheetName = textOrDefault(sheet.path("name"), "Sheet" + (sheetIndex + 1));
                if (!indexable) {
                    continue;
                }

                visibleSheetNames.add(sheetName);
                String sheetKey = xlsxSheetKey(sheetIndex, sheetName);
                String usedRange = firstText(sheet, "usedRange", "range");
                String sheetText = firstText(sheet, "compactText", "text");
                if (sheetText != null && !sheetText.isBlank()) {
                    visibleSheetTexts.add(sheetText.trim());
                }

                ObjectNode sheetMetadata = xlsxBaseMetadata(fileType, pipelineVersion, sourceRecordId);
                sheetMetadata.put("role", "sheet");
                sheetMetadata.put("sheetName", sheetName);
                sheetMetadata.put("sheetIndex", sheetIndex);
                sheetMetadata.put("hidden", hidden);
                putIfPresent(sheetMetadata, "usedRange", usedRange);
                putIfPresent(sheetMetadata, "cellRange", usedRange);
                putIntIfPresent(sheetMetadata, "rowStart", firstText(sheet, "rowStart"));
                putIntIfPresent(sheetMetadata, "rowEnd", firstText(sheet, "rowEnd"));
                putIntIfPresent(sheetMetadata, "columnStart", firstText(sheet, "columnStart"));
                putIntIfPresent(sheetMetadata, "columnEnd", firstText(sheet, "columnEnd"));
                putIntIfPresent(sheetMetadata, "rowCount", firstText(sheet, "rowCount", "maxRow"));
                putIntIfPresent(sheetMetadata, "columnCount", firstText(sheet, "columnCount", "maxColumn"));
                putCellRangeMetadata(sheetMetadata, usedRange);
                copyIfPresent(sheetMetadata, "mergedCells", sheet.path("mergedCells"));
                copyIfPresent(sheetMetadata, "hiddenRows", sheet.path("hiddenRows"));
                copyIfPresent(sheetMetadata, "hiddenColumns", sheet.path("hiddenColumns"));
                copyIfPresent(sheetMetadata, "hiddenCells", sheet.path("hiddenCells"));
                copyIfPresent(sheetMetadata, "hidden_cells", sheet.path("hidden_cells"));
                ArrayNode visibleFormulas = visibleFormulas(sheet);
                if (visibleFormulas.size() > 0) {
                    sheetMetadata.set("formulas", visibleFormulas);
                }
                units.add(new SearchUnitDraft(
                        source.getId(),
                        workbookJsonArtifactId,
                        SEARCH_UNIT_SECTION,
                        "sheet:" + sheetKey,
                        sheetName,
                        "workbook/" + sheetName,
                        null,
                        null,
                        blankToNull(sheetText),
                        sheetMetadata.toString()));

                units.addAll(xlsxTableUnits(
                        source,
                        workbookJsonArtifactId,
                        sheet,
                        sheetName,
                        sheetIndex,
                        sheetKey,
                        fileType,
                        pipelineVersion,
                        sourceRecordId));
                units.addAll(xlsxChunkUnits(
                        source,
                        workbookJsonArtifactId,
                        sheet,
                        sheetName,
                        sheetIndex,
                        sheetKey,
                        fileType,
                        pipelineVersion,
                        sourceRecordId));
            }
        }

        String documentText = firstText(root, "plainText", "text", "summary");
        if (documentText == null || documentText.isBlank()) {
            documentText = String.join("\n\n", visibleSheetTexts).trim();
        }
        ObjectNode documentMetadata = xlsxBaseMetadata(fileType, pipelineVersion, sourceRecordId);
        documentMetadata.put("role", "workbook");
        documentMetadata.put("sheetCount", sheetCount);
        documentMetadata.put("visibleSheetCount", workbook.path("visibleSheetCount").asInt(visibleSheetTexts.size()));
        documentMetadata.put("resultArtifactId", workbookJsonArtifactId);
        documentMetadata.set("sheetNames", objectMapper.valueToTree(visibleSheetNames));
        units.add(0, new SearchUnitDraft(
                source.getId(),
                workbookJsonArtifactId,
                SEARCH_UNIT_DOCUMENT,
                "workbook",
                source.getOriginalFileName(),
                null,
                null,
                null,
                blankToNull(documentText),
                documentMetadata.toString()));
        return units;
    }

    private List<SearchUnitDraft> xlsxTableUnits(SourceFileJpaEntity source,
                                                 String workbookJsonArtifactId,
                                                 JsonNode sheet,
                                                 String sheetName,
                                                 int sheetIndex,
                                                 String sheetKey,
                                                 String fileType,
                                                 String pipelineVersion,
                                                 String sourceRecordId) {
        JsonNode tables = sheet.path("tables");
        if (!tables.isArray()) {
            return List.of();
        }
        List<SearchUnitDraft> units = new ArrayList<>();
        int tableIndex = 1;
        for (JsonNode table : tables) {
            String tableName = firstText(table, "name", "tableId", "id");
            String range = firstText(table, "range", "cellRange");
            int currentTableIndex = intOrNull(table, "tableIndex") == null
                    ? tableIndex - 1
                    : intOrNull(table, "tableIndex");
            String tableId = tableName == null || tableName.isBlank()
                    ? "table-" + tableIndex
                    : normalizeUnitKeyPart(tableName);
            ObjectNode metadata = xlsxBaseMetadata(fileType, pipelineVersion, sourceRecordId);
            metadata.put("role", "table");
            metadata.put("sheetName", sheetName);
            metadata.put("sheetIndex", sheetIndex);
            metadata.put("tableIndex", currentTableIndex);
            metadata.put("tableId", tableId);
            putIfPresent(metadata, "tableName", tableName);
            putIfPresent(metadata, "range", range);
            putIfPresent(metadata, "cellRange", range);
            putIntIfPresent(metadata, "rowStart", firstText(table, "rowStart"));
            putIntIfPresent(metadata, "rowEnd", firstText(table, "rowEnd"));
            putIntIfPresent(metadata, "columnStart", firstText(table, "columnStart"));
            putIntIfPresent(metadata, "columnEnd", firstText(table, "columnEnd"));
            putIntIfPresent(metadata, "rowCount", firstText(table, "rowCount"));
            putIntIfPresent(metadata, "columnCount", firstText(table, "columnCount"));
            putIfPresent(metadata, "tableType", firstText(table, "type"));
            putCellRangeMetadata(metadata, range);
            ArrayNode safeHeaders = safeHeaderLabels(sheet, table);
            if (safeHeaders.size() > 0) {
                metadata.set("headers", safeHeaders);
            }
            putIfPresent(metadata, "headerRange", firstText(table, "headerRange"));
            putIfPresent(metadata, "dataRange", firstText(table, "dataRange"));
            units.add(new SearchUnitDraft(
                    source.getId(),
                    workbookJsonArtifactId,
                    SEARCH_UNIT_TABLE,
                    "sheet:" + sheetIndex + ":table:" + currentTableIndex + ":" + normalizeRangeForKey(range),
                    tableName == null || tableName.isBlank() ? tableId : tableName,
                    "workbook/" + sheetName,
                    null,
                    null,
                    blankToNull(firstText(table, "markdown", "text", "compactText")),
                    metadata.toString()));
            tableIndex++;
        }
        return units;
    }

    private List<SearchUnitDraft> xlsxChunkUnits(SourceFileJpaEntity source,
                                                 String workbookJsonArtifactId,
                                                 JsonNode sheet,
                                                 String sheetName,
                                                 int sheetIndex,
                                                 String sheetKey,
                                                 String fileType,
                                                 String pipelineVersion,
                                                 String sourceRecordId) {
        JsonNode chunks = sheet.path("chunks");
        if (!chunks.isArray()) {
            return List.of();
        }
        List<SearchUnitDraft> units = new ArrayList<>();
        int index = 1;
        for (JsonNode chunk : chunks) {
            String range = firstText(chunk, "range", "cellRange");
            if (range == null || range.isBlank()) {
                range = "chunk:" + index;
            }
            int chunkIndex = intOrNull(chunk, "chunkIndex") == null
                    ? index - 1
                    : intOrNull(chunk, "chunkIndex");
            ObjectNode metadata = xlsxBaseMetadata(fileType, pipelineVersion, sourceRecordId);
            metadata.put("role", "chunk");
            metadata.put("sheetName", sheetName);
            metadata.put("sheetIndex", sheetIndex);
            metadata.put("chunkIndex", chunkIndex);
            metadata.put("range", range);
            metadata.put("cellRange", range);
            putIntIfPresent(metadata, "rowStart", firstText(chunk, "rowStart"));
            putIntIfPresent(metadata, "rowEnd", firstText(chunk, "rowEnd"));
            putIntIfPresent(metadata, "columnStart", firstText(chunk, "columnStart"));
            putIntIfPresent(metadata, "columnEnd", firstText(chunk, "columnEnd"));
            putCellRangeMetadata(metadata, range);
            ArrayNode safeHeaders = safeHeaderLabels(sheet, chunk);
            if (safeHeaders.size() > 0) {
                metadata.set("headers", safeHeaders);
            }
            units.add(new SearchUnitDraft(
                    source.getId(),
                    workbookJsonArtifactId,
                    SEARCH_UNIT_CHUNK,
                    "sheet:" + sheetIndex + ":chunk:" + chunkIndex + ":" + normalizeRangeForKey(range),
                    sheetName + " " + range,
                    "workbook/" + sheetName,
                    null,
                    null,
                    blankToNull(firstText(chunk, "text", "compactText")),
                    metadata.toString()));
            index++;
        }
        return units;
    }

    private List<SearchUnitDraft> buildPdfSearchUnitDrafts(SourceFileJpaEntity source,
                                                           String parsedJsonArtifactId,
                                                           JsonNode root) {
        String fileType = firstNonBlank(firstText(root, "fileType", "file_type"), "pdf");
        String pipelineVersion = firstNonBlank(
                firstText(root, "pipelineVersion", "parserVersion", "parser_version"),
                PDF_PIPELINE_VERSION);
        String parserName = firstNonBlank(firstText(root, "parserName", "parser_name"), "pymupdf");
        String sourceRecordId = textOrNull(root.path("sourceRecordId"));
        JsonNode pages = root.path("pages");

        List<SearchUnitDraft> units = new ArrayList<>();
        List<String> pageTexts = new ArrayList<>();
        Integer firstPage = null;
        Integer lastPage = null;

        if (pages.isArray()) {
            int fallbackIndex = 0;
            for (JsonNode page : pages) {
                int physicalPageIndex = intOrNull(page, "physicalPageIndex", "physical_page_index") == null
                        ? fallbackIndex
                        : intOrNull(page, "physicalPageIndex", "physical_page_index");
                int pageNo = intOrNull(page, "pageNo", "page_no", "pageNumber") == null
                        ? physicalPageIndex + 1
                        : intOrNull(page, "pageNo", "page_no", "pageNumber");
                String pageLabel = firstNonBlank(firstText(page, "pageLabel", "page_label"), String.valueOf(pageNo));
                firstPage = firstPage == null ? pageNo : Math.min(firstPage, pageNo);
                lastPage = lastPage == null ? pageNo : Math.max(lastPage, pageNo);

                JsonNode blocks = page.path("blocks");
                String pageText = pageText(page);
                if (!pageText.isBlank()) {
                    pageTexts.add(pageText);
                }

                ObjectNode pageMetadata = pdfBaseMetadata(fileType, pipelineVersion, parserName, sourceRecordId);
                pageMetadata.put("role", "page");
                pageMetadata.put("physicalPageIndex", physicalPageIndex);
                pageMetadata.put("physical_page_index", physicalPageIndex);
                pageMetadata.put("pageNo", pageNo);
                pageMetadata.put("pageLabel", pageLabel);
                putIfPresent(pageMetadata, "page_label", pageLabel);
                if (page.has("width") && page.path("width").isNumber()) {
                    pageMetadata.put("width", page.path("width").asDouble());
                }
                if (page.has("height") && page.path("height").isNumber()) {
                    pageMetadata.put("height", page.path("height").asDouble());
                }
                pageMetadata.put("textLayerPresent", page.path("textLayerPresent").asBoolean(page.path("text_layer_present").asBoolean(!pageText.isBlank())));
                pageMetadata.put("ocrUsed", page.path("ocrUsed").asBoolean(page.path("ocr_used").asBoolean(false)));
                putIfPresent(pageMetadata, "ocrEngine", firstText(page, "ocrEngine", "ocr_engine"));
                putIfPresent(pageMetadata, "ocrModel", firstText(page, "ocrModel", "ocr_model"));
                putIfPresent(pageMetadata, "ocrLanguage", firstText(page, "ocrLanguage", "ocr_language"));
                putIfPresent(pageMetadata, "ocrConfidence", firstText(page, "ocrConfidence", "ocr_confidence", "ocrConfidenceAvg", "ocr_confidence_avg"));
                pageMetadata.put("blockCount", blocks.isArray() ? blocks.size() : 0);
                units.add(new SearchUnitDraft(
                        source.getId(),
                        parsedJsonArtifactId,
                        SEARCH_UNIT_PAGE,
                        "page:" + pageNo,
                        null,
                        null,
                        pageNo,
                        pageNo,
                        blankToNull(pageText),
                        pageMetadata.toString()));

                if (blocks.isArray()) {
                    int blockIndex = 0;
                    for (JsonNode block : blocks) {
                        String text = firstText(block, "text", "content");
                        if (text == null || text.isBlank()) {
                            blockIndex++;
                            continue;
                        }
                        String blockId = firstText(block, "blockId", "block_id", "id");
                        String unitKey = blockId == null || blockId.isBlank()
                                ? "page:" + pageNo + ":block:" + blockIndex
                                : "block:" + blockId;
                        ObjectNode blockMetadata = pdfBaseMetadata(fileType, pipelineVersion, parserName, sourceRecordId);
                        blockMetadata.put("role", "block");
                        blockMetadata.put("physicalPageIndex", physicalPageIndex);
                        blockMetadata.put("physical_page_index", physicalPageIndex);
                        blockMetadata.put("pageNo", pageNo);
                        blockMetadata.put("pageLabel", pageLabel);
                        putIfPresent(blockMetadata, "page_label", pageLabel);
                        putIfPresent(blockMetadata, "blockId", blockId);
                        putIfPresent(blockMetadata, "blockType", firstNonBlank(firstText(block, "blockType", "block_type"), "paragraph"));
                        putIntIfPresent(blockMetadata, "readingOrder", firstText(block, "readingOrder", "reading_order"));
                        copyIfPresent(blockMetadata, "bbox", firstPresent(block.path("bbox"), block.path("boundingBox")));
                        copyIfPresent(blockMetadata, "sectionPath", firstPresent(block.path("sectionPath"), block.path("section_path")));
                        blockMetadata.put("ocrUsed", block.path("ocrUsed").asBoolean(block.path("ocr_used").asBoolean(pageMetadata.path("ocrUsed").asBoolean(false))));
                        putIfPresent(blockMetadata, "ocrEngine", firstNonBlank(firstText(block, "ocrEngine", "ocr_engine"), firstText(page, "ocrEngine", "ocr_engine")));
                        putIfPresent(blockMetadata, "ocrModel", firstNonBlank(firstText(block, "ocrModel", "ocr_model"), firstText(page, "ocrModel", "ocr_model")));
                        putIfPresent(blockMetadata, "ocrLanguage", firstNonBlank(firstText(block, "ocrLanguage", "ocr_language"), firstText(page, "ocrLanguage", "ocr_language")));
                        putIfPresent(blockMetadata, "ocrConfidence", firstText(block, "ocrConfidence", "ocr_confidence", "confidence"));
                        Double blockQuality = doubleOrNull(block, "qualityScore", "quality_score");
                        if (blockQuality != null) {
                            blockMetadata.put("qualityScore", blockQuality);
                        }
                        units.add(new SearchUnitDraft(
                                source.getId(),
                                parsedJsonArtifactId,
                                SEARCH_UNIT_CHUNK,
                                unitKey,
                                null,
                                sectionPathText(firstPresent(block.path("sectionPath"), block.path("section_path"))),
                                pageNo,
                                pageNo,
                                text.trim(),
                                blockMetadata.toString()));
                        blockIndex++;
                    }
                }
                fallbackIndex++;
            }
        }

        String documentText = firstText(root, "plainText", "text", "documentText");
        if (documentText == null || documentText.isBlank()) {
            documentText = String.join("\n\n", pageTexts).trim();
        }
        boolean truncated = false;
        if (documentText != null && documentText.length() > DOCUMENT_TEXT_MAX_CHARS) {
            documentText = documentText.substring(0, DOCUMENT_TEXT_MAX_CHARS);
            truncated = true;
        }

        ObjectNode documentMetadata = pdfBaseMetadata(fileType, pipelineVersion, parserName, sourceRecordId);
        documentMetadata.put("role", "document");
        documentMetadata.put("pageCount", pageCount(root, pages, pageTexts.size()));
        documentMetadata.put("resultArtifactId", parsedJsonArtifactId);
        documentMetadata.put("textTruncated", truncated);
        units.add(0, new SearchUnitDraft(
                source.getId(),
                parsedJsonArtifactId,
                SEARCH_UNIT_DOCUMENT,
                "document",
                source.getOriginalFileName(),
                null,
                firstPage,
                lastPage,
                blankToNull(documentText),
                documentMetadata.toString()));
        return units;
    }

    private SearchUnitV2Payload buildSearchUnitV2Payload(SourceFileJpaEntity source,
                                                         DocumentIngestionContext context,
                                                         SearchUnitDraft draft,
                                                         String canonicalType) {
        JsonNode metadata = metadataOrEmpty(draft.metadataJson());
        String fileType = firstNonBlank(textOrNull(metadata.path("fileType")), context.fileType());
        String parserName = context.parserName();
        String parserVersion = firstNonBlank(textOrNull(metadata.path("pipelineVersion")), context.parserVersion());
        String locationType = locationType(source, metadata);
        String chunkType = chunkType(canonicalType, metadata, locationType);
        String locationJson = locationJson(locationType, source, context, draft, metadata, canonicalType);
        String citationText = citationText(locationType, source, metadata, draft);
        String displayText = firstNonBlank(draft.textContent(), citationText);
        String embeddingText = embeddingText(source, metadata, draft, locationType, chunkType, citationText);
        String bm25Text = bm25Text(source, citationText, draft.title(), draft.sectionPath(), draft.textContent());
        String debugText = debugText(fileType, parserName, parserVersion, chunkType, citationText);
        Double qualityScore = doubleOrNull(metadata, "qualityScore", "quality_score");
        Double confidenceScore = doubleOrNull(metadata, "confidence", "avgConfidence", "ocrConfidence", "ocr_confidence");
        return new SearchUnitV2Payload(
                chunkType,
                locationType,
                locationJson,
                embeddingText,
                bm25Text,
                displayText,
                citationText,
                debugText,
                parserName,
                parserVersion,
                qualityScore,
                confidenceScore);
    }

    private String locationJson(String locationType,
                                SourceFileJpaEntity source,
                                DocumentIngestionContext context,
                                SearchUnitDraft draft,
                                JsonNode metadata,
                                String canonicalType) {
        ObjectNode location = objectMapper.createObjectNode();
        location.put("type", locationType);
        location.put("document_version_id", context.documentVersionId());
        if ("xlsx".equals(locationType)) {
            putIfPresent(location, "sheet_name", firstText(metadata, "sheetName"));
            putIntIfPresent(location, "sheet_index", firstText(metadata, "sheetIndex"));
            putIfPresent(location, "table_id", firstText(metadata, "tableId", "tableName"));
            putIfPresent(location, "cell_range", firstText(metadata, "cellRange", "range", "usedRange"));
            putIfPresent(location, "row_range", rowRange(metadata));
            copyHeaderPath(location, metadata);
            location.put("contains_formula", metadata.has("formulas") || metadata.has("formulaCount"));
            location.put("contains_merged_cell", metadata.has("mergedCells") || metadata.has("mergedCellCount"));
            location.put("hidden_policy", "exclude_hidden");
            location.put("chunk_type", chunkType(canonicalType, metadata, locationType));
            return location.toString();
        }

        Integer pageNo = draft.pageStart() == null
                ? intOrNull(metadata, "pageNo", "pageNumber", "page")
                : draft.pageStart();
        Integer physicalPageIndex = intOrNull(metadata, "physicalPageIndex", "physical_page_index");
        if (physicalPageIndex == null && pageNo != null) {
            physicalPageIndex = Math.max(0, pageNo - 1);
        }
        putIntIfPresent(location, "physical_page_index", physicalPageIndex == null ? null : physicalPageIndex.toString());
        putIntIfPresent(location, "page_no", pageNo == null ? null : pageNo.toString());
        putIfPresent(location, "page_label", firstNonBlank(firstText(metadata, "pageLabel", "page_label"), pageNo == null ? null : String.valueOf(pageNo)));
        copyIfPresent(location, "bbox", firstPresent(metadata.path("bbox"), metadata.path("boundingBox")));
        copySectionPath(location, draft.sectionPath());
        location.put("block_type", blockType(canonicalType));
        boolean metadataOcrUsed = metadata.path("ocrUsed").asBoolean(metadata.path("ocr_used").asBoolean(false));
        location.put("ocr_used", metadataOcrUsed
                || !"pdf".equals(locationType)
                || source.getFileType() == null
                || !"PDF".equalsIgnoreCase(source.getFileType())
                || "ocr-lite".equals(context.parserName()));
        putIfPresent(location, "ocr_engine", firstText(metadata, "ocrEngine", "ocr_engine"));
        putIfPresent(location, "ocr_model", firstText(metadata, "ocrModel", "ocr_model"));
        putIfPresent(location, "ocr_language", firstText(metadata, "ocrLanguage", "ocr_language"));
        Double confidence = doubleOrNull(metadata, "confidence", "avgConfidence", "ocrConfidence", "ocr_confidence");
        if (confidence == null) {
            location.putNull("ocr_confidence");
        } else {
            location.put("ocr_confidence", confidence);
        }
        return location.toString();
    }

    private String citationText(String locationType,
                                SourceFileJpaEntity source,
                                JsonNode metadata,
                                SearchUnitDraft draft) {
        String fileName = source.getOriginalFileName();
        if ("xlsx".equals(locationType)) {
            String sheetName = firstText(metadata, "sheetName");
            String range = firstText(metadata, "cellRange", "range", "usedRange");
            List<String> parts = new ArrayList<>();
            parts.add(fileName);
            if (sheetName != null && !sheetName.isBlank()) {
                parts.add(sheetName);
            }
            if (range != null && !range.isBlank()) {
                parts.add(range);
            } else if (draft.unitKey() != null && draft.unitKey().equals("workbook")) {
                parts.add("workbook");
            }
            return String.join(" > ", parts);
        }
        Integer pageNo = draft.pageStart() == null
                ? intOrNull(metadata, "pageNo", "pageNumber", "page")
                : draft.pageStart();
        List<String> parts = new ArrayList<>();
        parts.add(fileName);
        if (pageNo != null) {
            parts.add("p." + pageNo);
        }
        String bbox = bboxText(firstPresent(metadata.path("bbox"), metadata.path("boundingBox")));
        if (bbox != null) {
            parts.add("bbox " + bbox);
        }
        return String.join(" > ", parts);
    }

    private String embeddingText(SourceFileJpaEntity source,
                                 JsonNode metadata,
                                 SearchUnitDraft draft,
                                 String locationType,
                                 String chunkType,
                                 String citationText) {
        List<String> parts = new ArrayList<>();
        appendLabeled(parts, "Source", source.getOriginalFileName());
        appendLabeled(parts, "Citation", citationText);
        appendLabeled(parts, "Chunk", chunkType);
        if ("xlsx".equals(locationType)) {
            appendLabeled(parts, "Sheet", firstText(metadata, "sheetName"));
            appendLabeled(parts, "Table", firstText(metadata, "tableName", "tableId"));
            appendLabeled(parts, "Range", firstText(metadata, "cellRange", "range", "usedRange"));
            appendLabeled(parts, "Headers", headerText(metadata.path("headers")));
        } else {
            appendLabeled(parts, "Page", firstNonBlank(
                    firstText(metadata, "pageLabel", "page_label"),
                    firstText(metadata, "pageNo", "pageNumber", "page")));
            appendLabeled(parts, "Section", draft.sectionPath());
            appendLabeled(parts, "Block", firstText(metadata, "blockType", "block_type", "role"));
            if (metadata.path("ocrUsed").asBoolean(metadata.path("ocr_used").asBoolean(false))) {
                appendLabeled(parts, "OCR", "used");
                appendLabeled(parts, "OCR confidence", firstText(metadata, "ocrConfidence", "ocr_confidence", "confidence"));
            }
        }
        String body = firstNonBlank(draft.textContent(), draft.title());
        if (body != null && !body.isBlank()) {
            parts.add("Content:\n" + body.trim());
        }
        return String.join("\n", parts);
    }

    private String bm25Text(SourceFileJpaEntity source,
                            String citationText,
                            String title,
                            String sectionPath,
                            String textContent) {
        List<String> parts = new ArrayList<>();
        putText(parts, source.getOriginalFileName());
        putText(parts, citationText);
        putText(parts, title);
        putText(parts, sectionPath);
        putText(parts, textContent);
        return String.join("\n", parts);
    }

    private static void appendLabeled(List<String> parts, String label, String value) {
        if (value != null && !value.isBlank()) {
            parts.add(label + ": " + value.trim());
        }
    }

    private static String headerText(JsonNode headers) {
        if (headers == null || headers.isMissingNode() || headers.isNull()) {
            return null;
        }
        if (headers.isArray()) {
            List<String> parts = new ArrayList<>();
            for (JsonNode header : headers) {
                String value = header.isTextual() ? header.asText() : header.toString();
                if (value != null && !value.isBlank()) {
                    parts.add(value.trim());
                }
            }
            return parts.isEmpty() ? null : String.join(" | ", parts);
        }
        if (headers.isObject()) {
            List<String> parts = new ArrayList<>();
            headers.fields().forEachRemaining(entry -> {
                String value = entry.getValue().isTextual() ? entry.getValue().asText() : entry.getValue().toString();
                if (value != null && !value.isBlank()) {
                    parts.add(value.trim());
                }
            });
            return parts.isEmpty() ? null : String.join(" | ", parts);
        }
        return headers.asText(null);
    }

    private String debugText(String fileType,
                             String parserName,
                             String parserVersion,
                             String chunkType,
                             String citationText) {
        ObjectNode debug = objectMapper.createObjectNode();
        putIfPresent(debug, "fileType", fileType);
        putIfPresent(debug, "parserName", parserName);
        putIfPresent(debug, "parserVersion", parserVersion);
        putIfPresent(debug, "chunkType", chunkType);
        putIfPresent(debug, "citationText", citationText);
        return debug.toString();
    }

    private boolean isOcrExtractArtifact(Artifact artifact) {
        return artifact.getType() == ArtifactType.OCR_RESULT_JSON
                || artifact.getType() == ArtifactType.OCR_TEXT_MARKDOWN;
    }

    private boolean isXlsxExtractArtifact(Artifact artifact) {
        return artifact.getType() == ArtifactType.XLSX_WORKBOOK_JSON
                || artifact.getType() == ArtifactType.XLSX_MARKDOWN
                || artifact.getType() == ArtifactType.XLSX_TABLE_JSON;
    }

    private boolean isPdfExtractArtifact(Artifact artifact) {
        return artifact.getType() == ArtifactType.PDF_PARSED_JSON
                || artifact.getType() == ArtifactType.PDF_PLAINTEXT
                || artifact.getType() == ArtifactType.PDF_TEXT_MARKDOWN;
    }

    private boolean isPrimaryParsedArtifact(String artifactType) {
        return "OCR_RESULT_JSON".equals(artifactType)
                || "XLSX_WORKBOOK_JSON".equals(artifactType)
                || "PDF_PARSED_JSON".equals(artifactType);
    }

    private String parserNameFor(SourceFileJpaEntity source, String fileType, ImportPlan plan) {
        Optional<String> artifactParserName = plan.extractedArtifacts().stream()
                .map(ArtifactPayload::payloadJson)
                .map(this::metadataOrEmpty)
                .map(metadata -> firstText(metadata, "parserName", "parser_name", "extractor", "engine"))
                .filter(value -> value != null && !value.isBlank())
                .findFirst();
        if (artifactParserName.isPresent()) {
            return artifactParserName.get();
        }
        String normalized = fileType == null ? "" : fileType.trim().toLowerCase(Locale.ROOT);
        if ("xlsx".equals(normalized) || "xlsm".equals(normalized)
                || "SPREADSHEET".equalsIgnoreCase(source.getFileType())) {
            return "xlsx-openpyxl";
        }
        if ("pdf".equals(normalized) && "PDF".equalsIgnoreCase(source.getFileType())) {
            return "pymupdf";
        }
        return "ocr-lite";
    }

    private String parserVersionFor(ImportPlan plan) {
        return plan.extractedArtifacts().stream()
                .map(ArtifactPayload::pipelineVersion)
                .filter(value -> value != null && !value.isBlank())
                .findFirst()
                .orElse("unknown-parser-version");
    }

    private String fileTypeForParsedArtifact(SourceFileJpaEntity source, ImportPlan plan) {
        Optional<String> artifactFileType = plan.searchUnits().stream()
                .map(SearchUnitDraft::metadataJson)
                .map(this::metadataOrEmpty)
                .map(metadata -> textOrNull(metadata.path("fileType")))
                .filter(value -> value != null && !value.isBlank())
                .findFirst();
        if (artifactFileType.isPresent()) {
            return artifactFileType.get();
        }
        String sourceType = source.getFileType() == null ? "" : source.getFileType().trim().toUpperCase(Locale.ROOT);
        return switch (sourceType) {
            case "SPREADSHEET" -> "xlsx";
            case "PDF" -> "pdf";
            case "IMAGE" -> "image";
            case "TEXT" -> "text";
            default -> "unknown";
        };
    }

    private String locationType(SourceFileJpaEntity source, JsonNode metadata) {
        String fileType = textOrNull(metadata.path("fileType"));
        if (fileType != null) {
            String normalized = fileType.trim().toLowerCase(Locale.ROOT);
            if ("xlsx".equals(normalized) || "xlsm".equals(normalized) || "spreadsheet".equals(normalized)) {
                return "xlsx";
            }
            if ("pdf".equals(normalized)) {
                return "pdf";
            }
        }
        String sourceType = source.getFileType() == null ? "" : source.getFileType().trim().toUpperCase(Locale.ROOT);
        if ("SPREADSHEET".equals(sourceType)) {
            return "xlsx";
        }
        if ("PDF".equals(sourceType)) {
            return "pdf";
        }
        return "ocr";
    }

    private String chunkType(String canonicalType, JsonNode metadata, String locationType) {
        String role = textOrNull(metadata.path("role"));
        if ("xlsx".equals(locationType)) {
            if ("workbook".equals(role)) {
                return "workbook_summary";
            }
            if ("sheet".equals(role)) {
                return "sheet_summary";
            }
            if ("table".equals(role)) {
                return "table";
            }
            if ("chunk".equals(role)) {
                return "row_group";
            }
        }
        if ("pdf".equals(locationType)) {
            if ("document".equals(role)) {
                return "document_summary";
            }
            if ("page".equals(role)) {
                return "page";
            }
            if ("block".equals(role)) {
                return firstNonBlank(firstText(metadata, "blockType", "block_type"), "paragraph");
            }
        }
        return switch (canonicalType) {
            case SEARCH_UNIT_DOCUMENT -> "document_summary";
            case SEARCH_UNIT_PAGE -> "page";
            case SEARCH_UNIT_SECTION -> "section";
            case SEARCH_UNIT_TABLE -> "table";
            case SEARCH_UNIT_IMAGE -> "figure_caption";
            case SEARCH_UNIT_CHUNK -> "ocr_block";
            default -> canonicalType == null ? "chunk" : canonicalType.toLowerCase(Locale.ROOT);
        };
    }

    private String blockType(String canonicalType) {
        return switch (canonicalType) {
            case SEARCH_UNIT_PAGE -> "page";
            case SEARCH_UNIT_SECTION -> "paragraph";
            case SEARCH_UNIT_TABLE -> "table";
            case SEARCH_UNIT_IMAGE -> "figure_caption";
            default -> "paragraph";
        };
    }

    private void copyHeaderPath(ObjectNode location, JsonNode metadata) {
        JsonNode headers = firstPresent(metadata.path("headerPath"), metadata.path("headers"));
        if (headers == null || headers.isMissingNode() || headers.isNull()) {
            headers = metadata.path("header_paths");
        }
        if (headers != null && !headers.isMissingNode() && !headers.isNull()) {
            location.set("header_path", headers.deepCopy());
        }
    }

    private void copySectionPath(ObjectNode location, String sectionPath) {
        if (sectionPath == null || sectionPath.isBlank()) {
            return;
        }
        location.set("section_path", objectMapper.valueToTree(List.of(sectionPath.split("\\s*>\\s*"))));
    }

    private String sectionPathText(JsonNode sectionPath) {
        if (sectionPath == null || sectionPath.isMissingNode() || sectionPath.isNull()) {
            return null;
        }
        if (sectionPath.isArray()) {
            List<String> parts = new ArrayList<>();
            for (JsonNode part : sectionPath) {
                String text = textOrNull(part);
                if (text != null && !text.isBlank()) {
                    parts.add(text.trim());
                }
            }
            return parts.isEmpty() ? null : String.join(" > ", parts);
        }
        return textOrNull(sectionPath);
    }

    private String rowRange(JsonNode metadata) {
        Integer rowStart = intOrNull(metadata, "rowStart");
        Integer rowEnd = intOrNull(metadata, "rowEnd");
        if (rowStart == null && rowEnd == null) {
            return null;
        }
        if (rowStart == null) {
            return String.valueOf(rowEnd);
        }
        if (rowEnd == null || rowEnd.equals(rowStart)) {
            return String.valueOf(rowStart);
        }
        return rowStart + ":" + rowEnd;
    }

    private String bboxText(JsonNode bbox) {
        if (bbox == null || bbox.isMissingNode() || bbox.isNull()) {
            return null;
        }
        if (!bbox.isArray() || bbox.size() == 0) {
            return null;
        }
        List<String> parts = new ArrayList<>();
        for (JsonNode value : bbox) {
            if (value.isNumber()) {
                parts.add(String.valueOf(value.asDouble()));
            } else if (value.isTextual() && !value.asText().isBlank()) {
                parts.add(value.asText().trim());
            }
        }
        return parts.isEmpty() ? null : "[" + String.join(",", parts) + "]";
    }

    private String checksumForPlan(ImportPlan plan) {
        return plan.extractedArtifacts().stream()
                .map(ArtifactPayload::artifact)
                .map(Artifact::getChecksumSha256)
                .filter(value -> value != null && !value.isBlank())
                .findFirst()
                .orElse(null);
    }

    private String validJsonOrDefault(String value, String fallback) {
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            objectMapper.readTree(value);
            return value;
        } catch (IOException | RuntimeException ex) {
            return fallback;
        }
    }

    private String warningsJson(String artifactJson) {
        JsonNode root = metadataOrEmpty(artifactJson);
        JsonNode warnings = root.path("warnings");
        return warnings.isArray() ? warnings.toString() : "[]";
    }

    private JsonNode metadataOrEmpty(String metadataJson) {
        if (metadataJson == null || metadataJson.isBlank()) {
            return objectMapper.getNodeFactory().objectNode();
        }
        try {
            return objectMapper.readTree(metadataJson);
        } catch (IOException | RuntimeException ex) {
            return objectMapper.getNodeFactory().objectNode();
        }
    }

    private static Double doubleOrNull(JsonNode node, String... names) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        for (String name : names) {
            JsonNode child = node.path(name);
            if (child.isNumber()) {
                return child.asDouble();
            }
            if (child.isTextual()) {
                try {
                    return Double.parseDouble(child.asText().trim());
                } catch (NumberFormatException ignored) {
                    // Try the next alias.
                }
            }
        }
        return null;
    }

    private static String documentIdForSource(String sourceFileId) {
        return "doc_" + stableHash(sourceFileId);
    }

    private static String documentVersionIdForSource(String sourceFileId) {
        return "docv_" + stableHash(sourceFileId);
    }

    private static String parsedArtifactIdForExtractedArtifact(String extractedArtifactId) {
        return "pa_" + stableHash(extractedArtifactId);
    }

    private static String firstNonBlank(String first, String second) {
        return first != null && !first.isBlank() ? first : second;
    }

    private static <T> T firstNonNull(T first, T second) {
        return first != null ? first : second;
    }

    private static void putText(List<String> parts, String value) {
        if (value != null && !value.isBlank()) {
            parts.add(value.trim());
        }
    }

    private ObjectNode baseMetadata(String engine, String pipelineVersion, String sourceRecordId) {
        ObjectNode metadata = objectMapper.createObjectNode();
        putIfPresent(metadata, "engine", engine);
        putIfPresent(metadata, "pipelineVersion", pipelineVersion);
        putIfPresent(metadata, "sourceRecordId", sourceRecordId);
        return metadata;
    }

    private ObjectNode xlsxBaseMetadata(String fileType, String pipelineVersion, String sourceRecordId) {
        ObjectNode metadata = objectMapper.createObjectNode();
        putIfPresent(metadata, "fileType", fileType);
        putIfPresent(metadata, "pipelineVersion", pipelineVersion);
        putIfPresent(metadata, "sourceRecordId", sourceRecordId);
        return metadata;
    }

    private ObjectNode pdfBaseMetadata(String fileType,
                                       String pipelineVersion,
                                       String parserName,
                                       String sourceRecordId) {
        ObjectNode metadata = objectMapper.createObjectNode();
        putIfPresent(metadata, "fileType", fileType);
        putIfPresent(metadata, "pipelineVersion", pipelineVersion);
        putIfPresent(metadata, "parserName", parserName);
        putIfPresent(metadata, "sourceRecordId", sourceRecordId);
        return metadata;
    }

    private String pageWarningsJson(JsonNode root, int physicalPageIndex, int pageNo) {
        JsonNode warnings = root.path("warnings");
        if (!warnings.isArray()) {
            return "[]";
        }
        ArrayNode filtered = objectMapper.createArrayNode();
        for (JsonNode warning : warnings) {
            Integer warningPhysical = intOrNull(warning, "physicalPageIndex", "physical_page_index");
            Integer warningPageNo = intOrNull(warning, "pageNo", "page_no", "pageNumber");
            if ((warningPhysical != null && warningPhysical == physicalPageIndex)
                    || (warningPageNo != null && warningPageNo == pageNo)) {
                filtered.add(warning.deepCopy());
            }
        }
        return filtered.toString();
    }

    private Double averageBlockConfidence(JsonNode blocks) {
        if (!blocks.isArray()) {
            return null;
        }
        double sum = 0.0;
        int count = 0;
        for (JsonNode block : blocks) {
            Double confidence = doubleOrNull(block, "confidence", "avgConfidence", "ocrConfidence", "ocr_confidence");
            if (confidence != null) {
                sum += confidence;
                count++;
            }
        }
        return count == 0 ? null : Math.round((sum / count) * 10_000.0) / 10_000.0;
    }

    private String xlsxTableLocationJson(DocumentIngestionContext documentContext,
                                         int sheetIndex,
                                         String sheetName,
                                         String tableId,
                                         String cellRange,
                                         String hiddenPolicy) {
        ObjectNode location = objectMapper.createObjectNode();
        location.put("type", "xlsx");
        location.put("document_version_id", documentContext.documentVersionId());
        location.put("sheet_index", sheetIndex);
        putIfPresent(location, "sheet_name", sheetName);
        putIfPresent(location, "table_id", tableId);
        putIfPresent(location, "cell_range", cellRange);
        putIfPresent(location, "hidden_policy", hiddenPolicy);
        return location.toString();
    }

    private ObjectNode headerJson(JsonNode table, JsonNode sheet) {
        ObjectNode node = objectMapper.createObjectNode();
        String range = firstText(table, "cellRange", "range");
        putIfPresent(node, "header_range", headerRange(range));
        ArrayNode headers = objectMapper.createArrayNode();
        Map<Integer, String> headersByColumn = headersByColumn(sheet, objectMapper.valueToTree(List.of(table)));
        for (Map.Entry<Integer, String> entry : headersByColumn.entrySet()) {
            ObjectNode header = objectMapper.createObjectNode();
            header.put("column_index", entry.getKey());
            header.put("column_letter", columnLetter(entry.getKey()));
            header.put("label", entry.getValue());
            headers.add(header);
        }
        node.set("headers", headers);
        return node;
    }

    private ArrayNode safeHeaderLabels(JsonNode sheet, JsonNode rangeNode) {
        ArrayNode headers = objectMapper.createArrayNode();
        Map<Integer, String> headersByColumn = headersByColumn(
                sheet,
                objectMapper.valueToTree(List.of(rangeNode)));
        for (String label : headersByColumn.values()) {
            headers.add(label);
        }
        return headers;
    }

    private Map<String, JsonNode> formulasByCell(JsonNode formulas) {
        Map<String, JsonNode> byCell = new HashMap<>();
        if (!formulas.isArray()) {
            return byCell;
        }
        for (JsonNode formula : formulas) {
            String cellRef = firstText(formula, "cell", "cellRef", "cell_ref", "address");
            if (cellRef != null && !cellRef.isBlank()) {
                byCell.put(normalizeRangeForKey(cellRef), formula);
            }
        }
        return byCell;
    }

    private ArrayNode visibleFormulas(JsonNode sheet) {
        ArrayNode visible = objectMapper.createArrayNode();
        JsonNode formulas = sheet.path("formulas");
        if (!formulas.isArray()) {
            return visible;
        }
        Set<Integer> hiddenRows = hiddenRowIndexes(sheet);
        Set<Integer> hiddenColumns = hiddenColumnIndexes(sheet);
        Set<String> hiddenCells = hiddenCellRefs(sheet);
        for (JsonNode formula : formulas) {
            String cellRef = firstText(formula, "cell", "cellRef", "cell_ref", "address");
            CellRef ref = parseCellRef(cellRef);
            boolean hiddenCell = cellRef != null && hiddenCells.contains(normalizeRangeForKey(cellRef));
            boolean hiddenRow = formula.path("hiddenRow").asBoolean(formula.path("hidden_row").asBoolean(false))
                    || (ref != null && hiddenRows.contains(ref.row()));
            boolean hiddenColumn = formula.path("hiddenColumn").asBoolean(formula.path("hidden_column").asBoolean(false))
                    || (ref != null && hiddenColumns.contains(ref.column()));
            if (!hiddenCell && !hiddenRow && !hiddenColumn) {
                visible.add(formula);
            }
        }
        return visible;
    }

    private Set<Integer> hiddenRowIndexes(JsonNode sheet) {
        java.util.HashSet<Integer> rows = new java.util.HashSet<>();
        addIntegerValues(rows, sheet.path("hiddenRows"));
        addIntegerValues(rows, sheet.path("hidden_rows"));
        return rows;
    }

    private Set<Integer> hiddenColumnIndexes(JsonNode sheet) {
        java.util.HashSet<Integer> columns = new java.util.HashSet<>();
        addIntegerValues(columns, sheet.path("hiddenColumnIndexes"));
        addIntegerValues(columns, sheet.path("hidden_column_indexes"));
        addColumnValues(columns, sheet.path("hiddenColumns"));
        addColumnValues(columns, sheet.path("hidden_columns"));
        return columns;
    }

    private Set<String> hiddenCellRefs(JsonNode sheet) {
        java.util.HashSet<String> cells = new java.util.HashSet<>();
        addCellRefs(cells, sheet.path("hiddenCells"));
        addCellRefs(cells, sheet.path("hidden_cells"));
        return cells;
    }

    private static void addCellRefs(Set<String> target, JsonNode node) {
        if (!node.isArray()) {
            return;
        }
        for (JsonNode item : node) {
            String value = item.asText(null);
            if (value != null && !value.isBlank()) {
                target.add(normalizeRangeForKey(value));
            }
        }
    }

    private static void addIntegerValues(Set<Integer> target, JsonNode node) {
        if (!node.isArray()) {
            return;
        }
        for (JsonNode item : node) {
            Integer value = intValue(item);
            if (value != null) {
                target.add(value);
            }
        }
    }

    private static void addColumnValues(Set<Integer> target, JsonNode node) {
        if (!node.isArray()) {
            return;
        }
        for (JsonNode item : node) {
            Integer value = intValue(item);
            if (value == null) {
                value = columnIndex(item.asText(null));
            }
            if (value != null) {
                target.add(value);
            }
        }
    }

    private Map<String, String> mergedRangesByCell(JsonNode mergedCells) {
        Map<String, String> result = new HashMap<>();
        if (!mergedCells.isArray()) {
            return result;
        }
        for (JsonNode merged : mergedCells) {
            String range = merged.asText("").trim();
            CellRangeParts parts = parseCellRange(range);
            if (parts == null) {
                continue;
            }
            for (int row = parts.rowStart(); row <= parts.rowEnd(); row++) {
                for (int column = parts.columnStart(); column <= parts.columnEnd(); column++) {
                    result.put(columnLetter(column) + row, range);
                }
            }
        }
        return result;
    }

    private Map<Integer, String> headersByColumn(JsonNode sheet, JsonNode tables) {
        Map<Integer, String> headers = new HashMap<>();
        JsonNode cells = sheet.path("cells");
        if (!cells.isArray()) {
            return headers;
        }
        Set<Integer> hiddenRows = hiddenRowIndexes(sheet);
        Set<Integer> hiddenColumns = hiddenColumnIndexes(sheet);
        Set<String> hiddenCells = hiddenCellRefs(sheet);
        Integer headerRow = null;
        if (tables.isArray() && tables.size() > 0) {
            String firstRange = firstText(tables.get(0), "cellRange", "range");
            CellRangeParts parts = parseCellRange(firstRange);
            if (parts != null) {
                headerRow = parts.rowStart();
            }
        }
        if (headerRow == null) {
            for (JsonNode cell : cells) {
                Integer row = intOrNull(cell, "row", "rowIndex", "row_index");
                if (row != null) {
                    headerRow = headerRow == null ? row : Math.min(headerRow, row);
                }
            }
        }
        if (headerRow == null) {
            return headers;
        }
        for (JsonNode cell : cells) {
            if (isHiddenXlsxCell(cell, hiddenRows, hiddenColumns, hiddenCells)) {
                continue;
            }
            Integer row = intOrNull(cell, "row", "rowIndex", "row_index");
            Integer column = intOrNull(cell, "column", "columnIndex", "column_index");
            String value = firstText(cell, "formattedValue", "formatted_value", "value", "rawValue", "raw_value");
            if (row != null && row.equals(headerRow) && column != null && value != null && !value.isBlank()) {
                headers.put(column, value);
            }
        }
        return headers;
    }

    private static boolean isHiddenXlsxCell(JsonNode cell,
                                            Set<Integer> hiddenRows,
                                            Set<Integer> hiddenColumns,
                                            Set<String> hiddenCells) {
        String cellRef = firstText(cell, "cell", "cellRef", "cell_ref", "address");
        CellRef ref = parseCellRef(cellRef);
        return cell.path("hiddenRow").asBoolean(cell.path("hidden_row").asBoolean(false))
                || cell.path("hiddenColumn").asBoolean(cell.path("hidden_column").asBoolean(false))
                || (ref != null && hiddenRows.contains(ref.row()))
                || (ref != null && hiddenColumns.contains(ref.column()))
                || (cellRef != null && hiddenCells.contains(normalizeRangeForKey(cellRef)));
    }

    private static boolean isHeaderCell(JsonNode cell, JsonNode tables) {
        String cellRef = firstText(cell, "cell", "cellRef", "cell_ref", "address");
        if (cellRef == null || !tables.isArray()) {
            return false;
        }
        CellRef ref = parseCellRef(cellRef);
        if (ref == null) {
            return false;
        }
        for (JsonNode table : tables) {
            CellRangeParts parts = parseCellRange(firstText(table, "cellRange", "range"));
            if (parts != null
                    && ref.row() == parts.rowStart()
                    && ref.column() >= parts.columnStart()
                    && ref.column() <= parts.columnEnd()) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasCell(JsonNode cells, String cellRef) {
        String normalized = normalizeRangeForKey(cellRef);
        for (JsonNode cell : cells) {
            String current = firstText(cell, "cell", "cellRef", "cell_ref", "address");
            if (normalized.equals(normalizeRangeForKey(current))) {
                return true;
            }
        }
        return false;
    }

    private static String tableIdForCell(String cellRef, JsonNode tables) {
        if (!tables.isArray()) {
            return null;
        }
        for (JsonNode table : tables) {
            String range = firstText(table, "cellRange", "range");
            if (cellInRange(cellRef, range)) {
                String tableName = firstText(table, "tableId", "id", "name", "tableName");
                return tableName == null ? null : normalizeUnitKeyPart(tableName);
            }
        }
        return null;
    }

    private static String tableMetadataIdForCell(String cellRef,
                                                 Map<String, String> tableIdsByRange,
                                                 JsonNode tables) {
        for (Map.Entry<String, String> entry : tableIdsByRange.entrySet()) {
            if (cellInRange(cellRef, entry.getKey())) {
                return entry.getValue();
            }
        }
        if (!tables.isArray()) {
            return null;
        }
        for (JsonNode table : tables) {
            String range = firstText(table, "cellRange", "range");
            if (cellInRange(cellRef, range)) {
                return tableIdsByRange.get(normalizeRangeForKey(range));
            }
        }
        return null;
    }

    private static boolean cellInRange(String cellRef, String range) {
        CellRef ref = parseCellRef(cellRef);
        CellRangeParts parts = parseCellRange(range);
        return ref != null
                && parts != null
                && ref.row() >= parts.rowStart()
                && ref.row() <= parts.rowEnd()
                && ref.column() >= parts.columnStart()
                && ref.column() <= parts.columnEnd();
    }

    private static String headerRange(String range) {
        CellRangeParts parts = parseCellRange(range);
        if (parts == null) {
            return null;
        }
        return columnLetter(parts.columnStart()) + parts.rowStart()
                + ":" + columnLetter(parts.columnEnd()) + parts.rowStart();
    }

    private static String dataRange(String range) {
        CellRangeParts parts = parseCellRange(range);
        if (parts == null) {
            return null;
        }
        int dataStart = Math.min(parts.rowStart() + 1, parts.rowEnd());
        return columnLetter(parts.columnStart()) + dataStart
                + ":" + columnLetter(parts.columnEnd()) + parts.rowEnd();
    }

    private static String pdfPageMetadataId(String documentVersionId, int physicalPageIndex) {
        return "pdfpg_" + stableHash(documentVersionId + ":" + physicalPageIndex);
    }

    private static String tableMetadataId(String documentVersionId,
                                          int sheetIndex,
                                          String tableId,
                                          String cellRange) {
        return "tbl_" + stableHash(documentVersionId + ":" + sheetIndex + ":" + tableId + ":" + cellRange);
    }

    private static String cellMetadataId(String documentVersionId,
                                         int sheetIndex,
                                         String tableId,
                                         String cellRef) {
        return "cell_" + stableHash(documentVersionId + ":" + sheetIndex + ":" + tableId + ":" + cellRef);
    }

    private static String columnLetter(int columnIndex) {
        if (columnIndex <= 0) {
            return null;
        }
        StringBuilder builder = new StringBuilder();
        int value = columnIndex;
        while (value > 0) {
            value--;
            builder.insert(0, (char) ('A' + (value % 26)));
            value /= 26;
        }
        return builder.toString();
    }

    private static Integer columnIndex(String columnLetter) {
        if (columnLetter == null || columnLetter.isBlank()) {
            return null;
        }
        String letters = columnLetter.trim().replaceAll("[^A-Za-z]", "").toUpperCase(Locale.ROOT);
        if (letters.isBlank()) {
            return null;
        }
        int column = 0;
        for (int i = 0; i < letters.length(); i++) {
            column = column * 26 + (letters.charAt(i) - 'A' + 1);
        }
        return column;
    }

    private static Integer intValue(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        if (node.isIntegralNumber()) {
            return node.asInt();
        }
        if (node.isObject()) {
            Integer value = intValue(node.path("index"));
            if (value != null) {
                return value;
            }
            value = intValue(node.path("row"));
            if (value != null) {
                return value;
            }
            value = intValue(node.path("column"));
            if (value != null) {
                return value;
            }
        }
        if (node.isTextual()) {
            try {
                return Integer.parseInt(node.asText().trim());
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    private String warningFromParsedPdf(JsonNode root) {
        JsonNode warnings = root.path("warnings");
        if (!warnings.isArray() || warnings.size() == 0) {
            return null;
        }
        List<String> parts = new ArrayList<>();
        for (JsonNode warning : warnings) {
            String text = warning.isTextual() ? warning.asText() : warning.toString();
            if (text != null && !text.isBlank()) {
                parts.add(text.trim());
            }
        }
        return parts.isEmpty() ? null : String.join("; ", parts);
    }

    private static JsonNode firstPresent(JsonNode first, JsonNode second) {
        return first != null && !first.isMissingNode() && !first.isNull() ? first : second;
    }

    private static String pageText(JsonNode page) {
        String direct = firstText(page, "text", "plainText");
        if (direct != null && !direct.isBlank()) {
            return direct.trim();
        }
        JsonNode blocks = page.path("blocks");
        if (!blocks.isArray()) {
            return "";
        }
        List<String> lines = new ArrayList<>();
        for (JsonNode block : blocks) {
            String text = firstText(block, "text", "content");
            if (text != null && !text.isBlank()) {
                lines.add(text.trim());
            }
        }
        return String.join("\n", lines);
    }

    private static String tableText(JsonNode table) {
        String direct = firstText(table, "markdown", "text", "plainText", "content");
        if (direct != null && !direct.isBlank()) {
            return direct.trim();
        }
        JsonNode rows = table.path("rows");
        if (!rows.isArray()) {
            rows = table.path("cells");
        }
        if (!rows.isArray()) {
            return "";
        }
        List<String> lines = new ArrayList<>();
        for (JsonNode row : rows) {
            if (row.isArray()) {
                List<String> cells = new ArrayList<>();
                for (JsonNode cell : row) {
                    cells.add(cell.isValueNode() ? cell.asText("") : firstText(cell, "text", "value"));
                }
                lines.add(String.join("\t", cells).trim());
            } else {
                String rowText = firstText(row, "text", "value");
                if (rowText != null && !rowText.isBlank()) {
                    lines.add(rowText.trim());
                }
            }
        }
        return String.join("\n", lines).trim();
    }

    private static int pageNumber(JsonNode node, int fallback) {
        Integer value = intOrNull(node, "pageNo", "pageNumber", "page", "pageStart");
        return value == null ? fallback : value;
    }

    private static int pageCount(JsonNode root, JsonNode pages, int fallback) {
        Integer explicit = intOrNull(root, "pageCount", "pagesCount");
        if (explicit != null) {
            return explicit;
        }
        return pages.isArray() ? pages.size() : fallback;
    }

    private static String sectionPath(JsonNode section, String title, Integer pageNo, int index) {
        JsonNode path = section.path("sectionPath");
        if (path.isMissingNode()) {
            path = section.path("path");
        }
        if (path.isArray()) {
            List<String> parts = new ArrayList<>();
            for (JsonNode part : path) {
                String text = part.asText("").trim();
                if (!text.isEmpty()) {
                    parts.add(text);
                }
            }
            if (!parts.isEmpty()) {
                return String.join(" > ", parts);
            }
        }
        String pathText = path.asText("").trim();
        if (!pathText.isEmpty()) {
            return pathText;
        }
        String label = title == null || title.isBlank() ? "section-" + index : title.trim();
        return pageNo == null ? label : "page:" + pageNo + " > " + label;
    }

    private static Integer intOrNull(JsonNode node, String... names) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        for (String name : names) {
            JsonNode child = node.path(name);
            if (child.isIntegralNumber()) {
                return child.asInt();
            }
            if (child.isTextual()) {
                try {
                    return Integer.parseInt(child.asText().trim());
                } catch (NumberFormatException ignored) {
                    // Try the next alias.
                }
            }
        }
        return null;
    }

    private static void putIfPresent(ObjectNode node, String field, String value) {
        if (value != null && !value.isBlank()) {
            node.put(field, value);
        }
    }

    private static void putIntIfPresent(ObjectNode node, String field, String value) {
        if (value == null || value.isBlank()) {
            return;
        }
        try {
            node.put(field, Integer.parseInt(value.trim()));
        } catch (NumberFormatException ex) {
            node.put(field, value);
        }
    }

    private static void copyIfPresent(ObjectNode node, String field, JsonNode value) {
        if (value != null && !value.isMissingNode() && !value.isNull()) {
            node.set(field, value.deepCopy());
        }
    }

    private static void putCellRangeMetadata(ObjectNode metadata, String range) {
        CellRangeParts parts = parseCellRange(range);
        if (parts == null) {
            return;
        }
        if (!metadata.has("rowStart")) {
            metadata.put("rowStart", parts.rowStart());
        }
        if (!metadata.has("rowEnd")) {
            metadata.put("rowEnd", parts.rowEnd());
        }
        if (!metadata.has("columnStart")) {
            metadata.put("columnStart", parts.columnStart());
        }
        if (!metadata.has("columnEnd")) {
            metadata.put("columnEnd", parts.columnEnd());
        }
    }

    private static String firstText(JsonNode node, String... names) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        for (String name : names) {
            String value = textOrNull(node.path(name));
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }

    private static String canonicalUnitType(String raw) {
        if (raw == null || raw.isBlank()) {
            return SEARCH_UNIT_CHUNK;
        }
        String normalized = raw.trim().toUpperCase(Locale.ROOT);
        if (SEARCH_UNIT_OCR_PAGE.equalsIgnoreCase(raw)) {
            return SEARCH_UNIT_PAGE;
        }
        return switch (normalized) {
            case SEARCH_UNIT_DOCUMENT,
                 SEARCH_UNIT_PAGE,
                 SEARCH_UNIT_SECTION,
                 SEARCH_UNIT_TABLE,
                 SEARCH_UNIT_IMAGE,
                 SEARCH_UNIT_CHUNK -> normalized;
            default -> normalized;
        };
    }

    private static String xlsxSheetKey(int sheetIndex, String sheetName) {
        return sheetIndex + ":" + normalizeUnitKeyPart(sheetName);
    }

    private static String normalizeUnitKeyPart(String raw) {
        if (raw == null || raw.isBlank()) {
            return "unnamed";
        }
        String normalized = raw.trim()
                .replace('/', '_')
                .replace('\\', '_')
                .replace(':', '_')
                .replaceAll("\\s+", " ");
        if (normalized.length() <= 96) {
            return normalized;
        }
        return normalized.substring(0, 80) + "-" + stableHash(normalized);
    }

    private static String normalizeRangeForKey(String range) {
        if (range == null || range.isBlank()) {
            return "unknown";
        }
        return range.trim().replace("$", "").replaceAll("\\s+", "").toUpperCase(Locale.ROOT);
    }

    private static CellRangeParts parseCellRange(String range) {
        if (range == null || range.isBlank()) {
            return null;
        }
        String normalized = range.replace("$", "").trim().toUpperCase(Locale.ROOT);
        String[] parts = normalized.split(":", 2);
        CellRef start = parseCellRef(parts[0]);
        CellRef end = parseCellRef(parts.length == 2 ? parts[1] : parts[0]);
        if (start == null || end == null) {
            return null;
        }
        return new CellRangeParts(
                Math.min(start.row(), end.row()),
                Math.max(start.row(), end.row()),
                Math.min(start.column(), end.column()),
                Math.max(start.column(), end.column()));
    }

    private static CellRef parseCellRef(String ref) {
        if (ref == null || ref.isBlank()) {
            return null;
        }
        String letters = ref.replaceAll("[^A-Z]", "");
        String digits = ref.replaceAll("[^0-9]", "");
        if (letters.isBlank() || digits.isBlank()) {
            return null;
        }
        int column = 0;
        for (int i = 0; i < letters.length(); i++) {
            column = column * 26 + (letters.charAt(i) - 'A' + 1);
        }
        try {
            return new CellRef(Integer.parseInt(digits), column);
        } catch (NumberFormatException ex) {
            return null;
        }
    }

    private static void transitionSource(SourceFileJpaEntity source,
                                         String status,
                                         String detail,
                                         Instant now) {
        source.transitionTo(status, blankToNull(detail), now);
    }

    private static String classifyFileType(String mimeType, String fileName) {
        String mime = normalizeMime(mimeType);
        String lowerName = fileName == null ? "" : fileName.toLowerCase(Locale.ROOT);
        if ("application/pdf".equals(mime) || lowerName.endsWith(".pdf")) {
            return "PDF";
        }
        if (mime != null && mime.startsWith("image/")) {
            return "IMAGE";
        }
        if (lowerName.endsWith(".png") || lowerName.endsWith(".jpg") || lowerName.endsWith(".jpeg")) {
            return "IMAGE";
        }
        if (mime != null && (mime.contains("spreadsheet") || mime.contains("excel") || mime.contains("csv"))) {
            return "SPREADSHEET";
        }
        if (lowerName.endsWith(".xlsx") || lowerName.endsWith(".xlsm") || lowerName.endsWith(".xls") || lowerName.endsWith(".csv")) {
            return "SPREADSHEET";
        }
        if (mime != null && mime.startsWith("text/")) {
            return "TEXT";
        }
        return "UNKNOWN";
    }

    private static String normalizeMime(String raw) {
        if (raw == null || raw.isBlank()) {
            return null;
        }
        return raw.split(";", 2)[0].trim().toLowerCase(Locale.ROOT);
    }

    private static String normalizeFileName(String originalFileName, String storageUri) {
        if (originalFileName != null && !originalFileName.isBlank()) {
            return originalFileName;
        }
        return fileNameFromStorageUri(storageUri);
    }

    private static String fileNameFromStorageUri(String storageUri) {
        if (storageUri == null || storageUri.isBlank()) {
            return "unknown";
        }
        String lastSegment = storageUri.replace('\\', '/').replaceAll("/+$", "");
        int slash = lastSegment.lastIndexOf('/');
        if (slash >= 0) {
            lastSegment = lastSegment.substring(slash + 1);
        }
        if (lastSegment.length() > 37 && lastSegment.charAt(36) == '-') {
            return lastSegment.substring(37);
        }
        return lastSegment.isBlank() ? "unknown" : lastSegment;
    }

    private static String textOrNull(JsonNode node) {
        return node == null || node.isMissingNode() || node.isNull() ? null : node.asText();
    }

    private static String textOrDefault(JsonNode node, String fallback) {
        String value = textOrNull(node);
        return value == null || value.isBlank() ? fallback : value;
    }

    private static String blankToNull(String value) {
        return value == null || value.isBlank() ? null : value;
    }

    private static String nullToEmpty(String value) {
        return value == null ? "" : value;
    }

    private static String stableHash(String value) {
        String hash = sha256OrNull(value == null ? "" : value);
        return hash.substring(0, 16);
    }

    private static String sha256OrNull(String value) {
        if (value == null) {
            return null;
        }
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 digest unavailable", ex);
        }
    }

    private static String newId() {
        return UUID.randomUUID().toString();
    }

    private record ParsedOcrJson(JsonNode root, String rawJson) {}

    private record ParsedXlsxJson(JsonNode root, String rawJson) {}

    private record ParsedPdfJson(JsonNode root, String rawJson) {}

    private record DocumentIngestionContext(
            String documentId,
            String documentVersionId,
            Map<String, String> parsedArtifactIds,
            String parserName,
            String parserVersion,
            String fileType
    ) {}

    private record SearchUnitV2Payload(
            String chunkType,
            String locationType,
            String locationJson,
            String embeddingText,
            String bm25Text,
            String displayText,
            String citationText,
            String debugText,
            String parserName,
            String parserVersion,
            Double qualityScore,
            Double confidenceScore
    ) {}

    private record ArtifactPayload(
            Artifact artifact,
            String artifactKey,
            String payloadJson,
            String pipelineVersion
    ) {}

    private record SearchUnitDraft(
            String sourceFileId,
            String extractedArtifactId,
            String unitType,
            String unitKey,
            String title,
            String sectionPath,
            Integer pageStart,
            Integer pageEnd,
            String textContent,
            String metadataJson
    ) {}

    private record ImportPlan(
            boolean valid,
            String sourceFileId,
            List<ArtifactPayload> extractedArtifacts,
            List<SearchUnitDraft> searchUnits,
            String warningMessage,
            String errorMessage
    ) {
        static ImportPlan valid(String sourceFileId,
                                List<ArtifactPayload> extractedArtifacts,
                                List<SearchUnitDraft> searchUnits,
                                String warningMessage) {
            return new ImportPlan(true, sourceFileId, extractedArtifacts, searchUnits, warningMessage, null);
        }

        static ImportPlan invalid(String errorMessage) {
            return new ImportPlan(false, null, List.of(), List.of(), null, errorMessage);
        }
    }

    private static class OcrCatalogImportException extends RuntimeException {
        OcrCatalogImportException(String message) {
            super(message);
        }
    }

    private static class XlsxCatalogImportException extends RuntimeException {
        XlsxCatalogImportException(String message) {
            super(message);
        }
    }

    private static class PdfCatalogImportException extends RuntimeException {
        PdfCatalogImportException(String message) {
            super(message);
        }
    }

    private record CellRef(int row, int column) {}

    private record CellRangeParts(int rowStart, int rowEnd, int columnStart, int columnEnd) {}

    public record SearchResult(
            SourceFileJpaEntity sourceFile,
            SearchUnitJpaEntity searchUnit
    ) {}
}
