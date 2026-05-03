package com.aipipeline.coreapi.catalog.application.service;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.ExtractedArtifactJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SearchUnitJpaRepository;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaEntity;
import com.aipipeline.coreapi.catalog.adapter.out.persistence.SourceFileJpaRepository;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

@Service
public class DocumentCatalogService {

    public static final String OCR_LITE_PIPELINE_VERSION = "ocr-lite-v1";
    public static final String XLSX_PIPELINE_VERSION = "xlsx-extract-v1";
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
    private static final Logger log = LoggerFactory.getLogger(DocumentCatalogService.class);

    private final SourceFileJpaRepository sourceFiles;
    private final ExtractedArtifactJpaRepository extractedArtifacts;
    private final SearchUnitJpaRepository searchUnits;
    private final ArtifactStoragePort storage;
    private final ObjectMapper objectMapper;

    public DocumentCatalogService(SourceFileJpaRepository sourceFiles,
                                  ExtractedArtifactJpaRepository extractedArtifacts,
                                  SearchUnitJpaRepository searchUnits,
                                  ArtifactStoragePort storage,
                                  ObjectMapper objectMapper) {
        this.sourceFiles = sourceFiles;
        this.extractedArtifacts = extractedArtifacts;
        this.searchUnits = searchUnits;
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
            for (SearchUnitDraft draft : plan.searchUnits()) {
                String resolvedArtifactId = draft.extractedArtifactId();
                if (resolvedArtifactId != null && !extractedByArtifactId.containsKey(resolvedArtifactId)) {
                    log.warn(
                            "Skipping search unit import with missing extracted artifact sourceFileId={} unitType={} unitKey={}",
                            source.getId(), draft.unitType(), draft.unitKey());
                    continue;
                }
                upsertSearchUnit(draft, now);
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
            for (SearchUnitDraft draft : plan.searchUnits()) {
                String resolvedArtifactId = draft.extractedArtifactId();
                if (resolvedArtifactId != null && !extractedByArtifactId.containsKey(resolvedArtifactId)) {
                    log.warn(
                            "Skipping XLSX search unit import with missing extracted artifact sourceFileId={} unitType={} unitKey={}",
                            source.getId(), draft.unitType(), draft.unitKey());
                    continue;
                }
                upsertSearchUnit(draft, now);
            }
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

    private void upsertSearchUnit(SearchUnitDraft draft, Instant now) {
        String canonicalType = canonicalUnitType(draft.unitType());
        String contentHash = sha256OrNull(draft.textContent());
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
        searchUnits.save(entity);
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
                if (!indexable) {
                    continue;
                }

                String sheetName = textOrDefault(sheet.path("name"), "Sheet" + (sheetIndex + 1));
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
                copyIfPresent(sheetMetadata, "formulas", sheet.path("formulas"));
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

    private boolean isOcrExtractArtifact(Artifact artifact) {
        return artifact.getType() == ArtifactType.OCR_RESULT_JSON
                || artifact.getType() == ArtifactType.OCR_TEXT_MARKDOWN;
    }

    private boolean isXlsxExtractArtifact(Artifact artifact) {
        return artifact.getType() == ArtifactType.XLSX_WORKBOOK_JSON
                || artifact.getType() == ArtifactType.XLSX_MARKDOWN
                || artifact.getType() == ArtifactType.XLSX_TABLE_JSON;
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

    private record CellRef(int row, int column) {}

    private record CellRangeParts(int rowStart, int rowEnd, int columnStart, int columnEnd) {}

    public record SearchResult(
            SourceFileJpaEntity sourceFile,
            SearchUnitJpaEntity searchUnit
    ) {}
}
