"""Tests for the OCR_EXTRACT OCR-lite capability."""

from __future__ import annotations

import json

import pytest

from app.capabilities.base import (
    CapabilityError,
    CapabilityInput,
    CapabilityInputArtifact,
)
from app.capabilities.ocr.artifact_builder import (
    OCR_LITE_PIPELINE_VERSION,
    OCR_RESULT_JSON,
    OCR_TEXT_MARKDOWN,
    build_output_artifacts,
)
from app.capabilities.ocr.fixture_provider import FixtureOcrProvider
from app.capabilities.ocr.service import OcrExtractCapability, OcrExtractService


def test_fixture_provider_output_normalization():
    provider = FixtureOcrProvider("fixture invoice total")

    document = provider.extract(
        b"ignored",
        source_record_id="source-1",
        pipeline_version=OCR_LITE_PIPELINE_VERSION,
        content_type="image/png",
        filename="invoice.png",
    )

    assert document.source_record_id == "source-1"
    assert document.pipeline_version == OCR_LITE_PIPELINE_VERSION
    assert document.engine == "fixture"
    assert document.plain_text == "fixture invoice total"
    assert document.pages[0].blocks[0].bbox == [0, 0, 100, 30]


def test_ocr_artifact_builder_emits_expected_shapes():
    document = FixtureOcrProvider("fixture invoice total").extract(
        b"ignored",
        source_record_id="source-1",
        pipeline_version=OCR_LITE_PIPELINE_VERSION,
        content_type="image/png",
        filename="invoice.png",
    )

    outputs = build_output_artifacts(document)

    assert [artifact.type for artifact in outputs] == [
        OCR_RESULT_JSON,
        OCR_TEXT_MARKDOWN,
    ]
    body = json.loads(outputs[0].content)
    assert body["sourceRecordId"] == "source-1"
    assert body["pipelineVersion"] == OCR_LITE_PIPELINE_VERSION
    assert body["engine"] == "fixture"
    assert body["plainText"] == "fixture invoice total"
    assert body["pages"][0]["blocks"][0] == {
        "text": "fixture invoice total",
        "confidence": 0.95,
        "bbox": [0, 0, 100, 30],
    }
    markdown = outputs[1].content.decode("utf-8")
    assert "Engine: fixture" in markdown


def test_paddle_output_normalization_uses_text_confidence_and_bbox():
    from app.capabilities.ocr.paddle_provider import _normalize_blocks

    blocks = _normalize_blocks([
        [
            [
                [[10, 20], [110, 20], [110, 50], [10, 50]],
                ("paddle text", 0.91),
            ]
        ]
    ])

    assert len(blocks) == 1
    assert blocks[0].text == "paddle text"
    assert blocks[0].confidence == 0.91
    assert blocks[0].bbox == [10, 20, 110, 50]


def test_paddle_v3_result_normalization_uses_res_wrapper_and_rec_boxes():
    from app.capabilities.ocr.paddle_provider import _normalize_blocks

    blocks = _normalize_blocks([
        {
            "res": {
                "rec_texts": ["v3 paddle text"],
                "rec_scores": [0.87],
                "rec_boxes": [[12, 24, 140, 58]],
            }
        }
    ])

    assert len(blocks) == 1
    assert blocks[0].text == "v3 paddle text"
    assert blocks[0].confidence == 0.87
    assert blocks[0].bbox == [12, 24, 140, 58]


def test_ocr_extract_emits_json_and_markdown_artifacts():
    capability = OcrExtractCapability(
        service=OcrExtractService(
            provider=FixtureOcrProvider("fixture invoice total"),
        ),
    )

    result = capability.run(
        CapabilityInput(
            job_id="job-1",
            capability="OCR_EXTRACT",
            attempt_no=1,
            inputs=[
                CapabilityInputArtifact(
                    artifact_id="source-1",
                    source_file_id="source-1",
                    type="INPUT_FILE",
                    content=b"\x89PNG\r\n\x1a\nfixture",
                    content_type="image/png",
                    filename="invoice.png",
                )
            ],
        )
    )

    assert [artifact.type for artifact in result.outputs] == [
        OCR_RESULT_JSON,
        OCR_TEXT_MARKDOWN,
    ]

    body = json.loads(result.outputs[0].content)
    assert body == {
        "sourceRecordId": "source-1",
        "pipelineVersion": OCR_LITE_PIPELINE_VERSION,
        "engine": "fixture",
        "pages": [
            {
                "pageNo": 1,
                "blocks": [
                    {
                        "text": "fixture invoice total",
                        "confidence": 0.95,
                        "bbox": [0, 0, 100, 30],
                    }
                ],
            }
        ],
        "plainText": "fixture invoice total",
    }

    markdown = result.outputs[1].content.decode("utf-8")
    assert "# OCR Text" in markdown
    assert "Pipeline: ocr-lite-v1" in markdown
    assert "fixture invoice total" in markdown


def test_ocr_extract_uses_source_file_id_when_claim_provides_it():
    capability = OcrExtractCapability(
        service=OcrExtractService(
            provider=FixtureOcrProvider("catalog text"),
        ),
    )

    result = capability.run(
        CapabilityInput(
            job_id="job-1",
            capability="OCR_EXTRACT",
            attempt_no=1,
            inputs=[
                CapabilityInputArtifact(
                    artifact_id="input-artifact-1",
                    source_file_id="source-file-1",
                    type="INPUT_FILE",
                    content=b"\x89PNG\r\n\x1a\nfixture",
                    content_type="image/png",
                    filename="invoice.png",
                )
            ],
        )
    )

    body = json.loads(result.outputs[0].content)
    assert body["sourceRecordId"] == "source-file-1"


def test_ocr_extract_uses_legacy_source_record_id_fallback_when_claim_omits_source_file_id():
    capability = OcrExtractCapability(
        service=OcrExtractService(provider=FixtureOcrProvider()),
    )

    result = capability.run(
        CapabilityInput(
            job_id="job-1",
            capability="OCR_EXTRACT",
            attempt_no=1,
            inputs=[
                CapabilityInputArtifact(
                    artifact_id="input-artifact-1",
                    type="INPUT_FILE",
                    content=b"\x89PNG\r\n\x1a\nfixture",
                    content_type="image/png",
                    filename="invoice.png",
                )
            ],
        )
    )

    body = json.loads(result.outputs[0].content)
    assert body["sourceRecordId"] == "input-artifact:input-artifact-1"


def test_ocr_extract_rejects_text_only_input():
    capability = OcrExtractCapability(
        service=OcrExtractService(provider=FixtureOcrProvider()),
    )

    with pytest.raises(CapabilityError) as raised:
        capability.run(
            CapabilityInput(
                job_id="job-1",
                capability="OCR_EXTRACT",
                attempt_no=1,
                inputs=[
                    CapabilityInputArtifact(
                        artifact_id="text-1",
                        type="INPUT_TEXT",
                        content=b"not a file",
                        content_type="text/plain",
                        filename="prompt.txt",
                    )
                ],
            )
        )

    assert raised.value.code == "UNSUPPORTED_INPUT_TYPE"


def test_registry_registers_ocr_extract_without_heavy_ocr_provider():
    from app.capabilities import registry as registry_module
    from app.core.config import WorkerSettings

    settings = WorkerSettings(
        rag_enabled=False,
        ocr_enabled=False,
        multimodal_enabled=False,
        ocr_extract_enabled=True,
        ocr_extract_provider="fixture",
        xlsx_extract_enabled=False,
        pdf_extract_enabled=False,
    )

    result = registry_module.build_default_registry(settings)

    assert result.available() == ["MOCK", "OCR_EXTRACT"]
