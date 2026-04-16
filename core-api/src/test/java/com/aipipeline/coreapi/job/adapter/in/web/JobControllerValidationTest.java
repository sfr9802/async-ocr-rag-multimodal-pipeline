package com.aipipeline.coreapi.job.adapter.in.web;

import com.aipipeline.coreapi.artifact.application.port.out.ArtifactStoragePort;
import com.aipipeline.coreapi.artifact.domain.Artifact;
import com.aipipeline.coreapi.artifact.domain.ArtifactId;
import com.aipipeline.coreapi.artifact.domain.ArtifactRole;
import com.aipipeline.coreapi.artifact.domain.ArtifactType;
import com.aipipeline.coreapi.common.web.GlobalExceptionHandler;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase.CreateJobCommand;
import com.aipipeline.coreapi.job.application.port.in.JobManagementUseCase.JobCreationResult;
import com.aipipeline.coreapi.job.domain.Job;
import com.aipipeline.coreapi.job.domain.JobCapability;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.time.Instant;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * {@link JobController} contract tests for the capability/input validation
 * layer.
 *
 * <p>The primary invariant this test class pins is:
 *
 * <blockquote>Invalid submissions do NOT reach the async pipeline.</blockquote>
 *
 * <p>Concretely: when the validator rejects a request, neither
 * {@link ArtifactStoragePort#store} nor
 * {@link JobManagementUseCase#createAndEnqueue} may be called — if they were,
 * the rejected request would have half-written rows and potentially fired a
 * Redis dispatch. The tests verify both of these are never invoked on the
 * 400 path using Mockito's {@code verify(..., never())}.
 *
 * <p><b>No Spring context is loaded.</b> Spring Boot 4.0.3 moved the
 * {@code @WebMvcTest} slice out of the default {@code spring-boot-starter-test}
 * dependencies. Rather than pull in an extra module just for a slice that
 * exists to do exactly what {@link MockMvcBuilders#standaloneSetup} already
 * does, this class instantiates the controller directly with Mockito mocks
 * and wires it into MockMvc with the {@link GlobalExceptionHandler} as a
 * controller advice. This gives the same end-to-end coverage with zero
 * context loading, no datasource, no Flyway, and no Redis.
 *
 * <p>Happy-path tests prove the validator does not over-reject: a valid
 * MULTIMODAL / OCR / RAG submission hits both the storage port and the
 * enqueue use case exactly as many times as the controller intends.
 */
class JobControllerValidationTest {

    private JobManagementUseCase jobManagement;
    private ArtifactStoragePort storage;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        jobManagement = Mockito.mock(JobManagementUseCase.class);
        storage = Mockito.mock(ArtifactStoragePort.class);

        JobController controller = new JobController(jobManagement, storage);
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();

        // Happy-path stubs. Rejection tests never reach these code paths, so
        // Mockito still records zero invocations when nothing calls them.
        when(storage.store(
                any(),
                any(),
                any(),
                any(),
                any(),
                org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(new ArtifactStoragePort.StoredObject("local://test/uri", 42L, "sha256"));

        Job job = Job.createNew(JobCapability.MULTIMODAL, Instant.parse("2026-04-15T10:00:00Z"));
        job.markQueued(Instant.parse("2026-04-15T10:00:00Z"));
        Artifact fakeArtifact = Artifact.rehydrate(
                ArtifactId.of("art-test-1"),
                job.getId(),
                ArtifactRole.INPUT,
                ArtifactType.INPUT_FILE,
                "local://test/uri",
                "image/png",
                42L,
                "sha256",
                Instant.parse("2026-04-15T10:00:00Z"));
        when(jobManagement.createAndEnqueue(any(CreateJobCommand.class)))
                .thenReturn(new JobCreationResult(job, List.of(fakeArtifact)));
    }

    // ==================================================================
    // REJECTION: RAG without text
    // ==================================================================

    @Test
    void rag_without_text_rejected_TEXT_REQUIRED_and_no_enqueue() throws Exception {
        mockMvc.perform(post("/api/v1/jobs")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"capability\":\"RAG\",\"text\":\"\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("TEXT_REQUIRED"));

        assertPipelineNotInvoked();
    }

    @Test
    void rag_with_whitespace_only_text_rejected_TEXT_REQUIRED_and_no_enqueue() throws Exception {
        mockMvc.perform(post("/api/v1/jobs")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"capability\":\"RAG\",\"text\":\"   \"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("TEXT_REQUIRED"));

        assertPipelineNotInvoked();
    }

    // ==================================================================
    // REJECTION: OCR without file (wrong endpoint)
    // ==================================================================

    @Test
    void ocr_via_json_endpoint_rejected_FILE_REQUIRED_and_no_enqueue() throws Exception {
        mockMvc.perform(post("/api/v1/jobs")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"capability\":\"OCR\",\"text\":\"extract this\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("FILE_REQUIRED"));

        assertPipelineNotInvoked();
    }

    @Test
    void ocr_multipart_without_file_rejected_FILE_REQUIRED_and_no_enqueue() throws Exception {
        // Multipart submission with no 'file' part at all → validator fires,
        // returns FILE_REQUIRED, no side effects.
        mockMvc.perform(multipart("/api/v1/jobs")
                        .param("capability", "OCR"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("FILE_REQUIRED"));

        assertPipelineNotInvoked();
    }

    // ==================================================================
    // REJECTION: MULTIMODAL without file
    // ==================================================================

    @Test
    void multimodal_via_json_endpoint_rejected_FILE_REQUIRED_and_no_enqueue() throws Exception {
        mockMvc.perform(post("/api/v1/jobs")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"capability\":\"MULTIMODAL\",\"text\":\"caption this\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("FILE_REQUIRED"));

        assertPipelineNotInvoked();
    }

    @Test
    void multimodal_multipart_without_file_rejected_FILE_REQUIRED_and_no_enqueue() throws Exception {
        mockMvc.perform(multipart("/api/v1/jobs")
                        .param("capability", "MULTIMODAL")
                        .param("text", "describe the image"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("FILE_REQUIRED"));

        assertPipelineNotInvoked();
    }

    // ==================================================================
    // REJECTION: zero-byte file
    // ==================================================================

    @Test
    void ocr_multipart_with_zero_byte_file_rejected_FILE_EMPTY_and_no_enqueue() throws Exception {
        MockMultipartFile empty = new MockMultipartFile(
                "file", "blank.png", "image/png", new byte[0]);

        mockMvc.perform(multipart("/api/v1/jobs")
                        .file(empty)
                        .param("capability", "OCR"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("FILE_EMPTY"));

        assertPipelineNotInvoked();
    }

    // ==================================================================
    // REJECTION: unsupported file type
    // ==================================================================

    @Test
    void ocr_multipart_with_gif_rejected_UNSUPPORTED_FILE_TYPE_and_no_enqueue() throws Exception {
        MockMultipartFile gif = new MockMultipartFile(
                "file", "cat.gif", "image/gif", "GIF89a...".getBytes());

        mockMvc.perform(multipart("/api/v1/jobs")
                        .file(gif)
                        .param("capability", "OCR"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("UNSUPPORTED_FILE_TYPE"));

        assertPipelineNotInvoked();
    }

    @Test
    void multimodal_multipart_with_webp_rejected_UNSUPPORTED_FILE_TYPE_and_no_enqueue() throws Exception {
        MockMultipartFile webp = new MockMultipartFile(
                "file", "pic.webp", "image/webp",
                new byte[]{0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00});

        mockMvc.perform(multipart("/api/v1/jobs")
                        .file(webp)
                        .param("capability", "MULTIMODAL")
                        .param("text", "what is this?"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("UNSUPPORTED_FILE_TYPE"));

        assertPipelineNotInvoked();
    }

    // ==================================================================
    // REJECTION: unknown / missing capability
    // ==================================================================

    @Test
    void unknown_capability_rejected_UNKNOWN_CAPABILITY_and_no_enqueue() throws Exception {
        mockMvc.perform(post("/api/v1/jobs")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"capability\":\"SUMMARIZE\",\"text\":\"hi\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("UNKNOWN_CAPABILITY"));

        assertPipelineNotInvoked();
    }

    @Test
    void missing_capability_rejected_CAPABILITY_REQUIRED_and_no_enqueue() throws Exception {
        mockMvc.perform(post("/api/v1/jobs")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"capability\":\"\",\"text\":\"hi\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("CAPABILITY_REQUIRED"));

        assertPipelineNotInvoked();
    }

    // ==================================================================
    // ACCEPTANCE: valid submissions produce exactly one enqueue
    // ==================================================================

    @Test
    void valid_multimodal_with_file_and_text_accepted_and_enqueued() throws Exception {
        MockMultipartFile png = new MockMultipartFile(
                "file", "invoice.png", "image/png",
                new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A});

        mockMvc.perform(multipart("/api/v1/jobs")
                        .file(png)
                        .param("capability", "MULTIMODAL")
                        .param("text", "what is the total on this invoice?"))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.capability").value("MULTIMODAL"))
                .andExpect(jsonPath("$.status").value("QUEUED"));

        // Exactly one enqueue happened for the valid request.
        verify(jobManagement).createAndEnqueue(any(CreateJobCommand.class));
        // Two storage writes: INPUT_FILE + INPUT_TEXT.
        verify(storage, Mockito.times(2))
                .store(any(), any(), any(), any(), any(), org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void valid_multimodal_with_file_but_no_text_accepted_and_enqueued() throws Exception {
        MockMultipartFile png = new MockMultipartFile(
                "file", "diagram.png", "image/png",
                new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A});

        mockMvc.perform(multipart("/api/v1/jobs")
                        .file(png)
                        .param("capability", "MULTIMODAL"))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.capability").value("MULTIMODAL"))
                .andExpect(jsonPath("$.status").value("QUEUED"));

        verify(jobManagement).createAndEnqueue(any(CreateJobCommand.class));
        // Only one storage write: INPUT_FILE. No INPUT_TEXT since text was absent.
        verify(storage, Mockito.times(1))
                .store(any(), any(), any(), any(), any(), org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void valid_rag_with_text_accepted_and_enqueued() throws Exception {
        mockMvc.perform(post("/api/v1/jobs")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"capability\":\"RAG\",\"text\":\"who feeds the harbor cats?\"}"))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.status").value("QUEUED"));

        verify(jobManagement).createAndEnqueue(any(CreateJobCommand.class));
        verify(storage).store(any(), any(), any(), any(), any(), org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void valid_ocr_with_png_accepted_and_enqueued() throws Exception {
        MockMultipartFile png = new MockMultipartFile(
                "file", "receipt.png", "image/png",
                new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A});

        mockMvc.perform(multipart("/api/v1/jobs")
                        .file(png)
                        .param("capability", "OCR"))
                .andExpect(status().isAccepted());

        verify(jobManagement).createAndEnqueue(any(CreateJobCommand.class));
        verify(storage).store(any(), any(), any(), any(), any(), org.mockito.ArgumentMatchers.anyLong());
    }

    // ==================================================================
    // PIPELINE-STATE INVARIANT: no enqueue on rejection
    // ==================================================================
    //
    // This is the concrete proof that a rejected request cannot create
    // ambiguous downstream pipeline state. If either of these interactions
    // happened, the pipeline would have:
    //   - a half-written storage object (storage.store), and/or
    //   - a QUEUED Job row + Redis dispatch (jobManagement.createAndEnqueue)
    // neither of which matches the 4xx response the client received.
    //
    // Every rejection test above calls this helper.

    private void assertPipelineNotInvoked() {
        verify(storage, never())
                .store(any(), any(), any(), any(), any(), org.mockito.ArgumentMatchers.anyLong());
        verify(jobManagement, never()).createAndEnqueue(any(CreateJobCommand.class));
    }
}
