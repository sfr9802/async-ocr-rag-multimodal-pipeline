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
    answer_recall_delta,
    avg_cost_multiplier,
    cer,
    dup_rate,
    edit_distance,
    hit_at_k,
    iter_count_mean,
    keyword_coverage,
    loop_recovery_rate,
    p_percentile,
    recall_at_k,
    reciprocal_rank,
    topk_gap,
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
    "answer_recall_delta",
    "avg_cost_multiplier",
    "cer",
    "dup_rate",
    "edit_distance",
    "hit_at_k",
    "iter_count_mean",
    "keyword_coverage",
    "loop_recovery_rate",
    "p_percentile",
    "recall_at_k",
    "reciprocal_rank",
    "topk_gap",
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
