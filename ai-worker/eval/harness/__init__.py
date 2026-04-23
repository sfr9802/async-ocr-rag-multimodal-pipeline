"""Eval harness building blocks.

Public re-exports for easy importing from the CLI and tests:

    from eval.harness import (
        cer, wer, hit_at_k, reciprocal_rank, keyword_coverage,
        load_jsonl, write_json_report, write_csv_report,
        run_rag_eval, run_ocr_eval,
    )
"""

from eval.harness.io_utils import (
    load_jsonl,
    write_csv_report,
    write_json_report,
)
from eval.harness.metrics import (
    cer,
    edit_distance,
    hit_at_k,
    keyword_coverage,
    reciprocal_rank,
    wer,
)
from eval.harness.multimodal_eval import (
    MultimodalEvalRow,
    MultimodalEvalSummary,
    run_multimodal_eval,
)
from eval.harness.ocr_eval import OcrEvalRow, OcrEvalSummary, run_ocr_eval
from eval.harness.rag_eval import RagEvalRow, RagEvalSummary, run_rag_eval

__all__ = [
    "cer",
    "edit_distance",
    "hit_at_k",
    "keyword_coverage",
    "reciprocal_rank",
    "wer",
    "load_jsonl",
    "write_csv_report",
    "write_json_report",
    "MultimodalEvalRow",
    "MultimodalEvalSummary",
    "run_multimodal_eval",
    "OcrEvalRow",
    "OcrEvalSummary",
    "run_ocr_eval",
    "RagEvalRow",
    "RagEvalSummary",
    "run_rag_eval",
]
