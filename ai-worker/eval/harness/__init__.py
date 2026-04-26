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
    count_whitespace_tokens,
    dup_rate,
    edit_distance,
    expected_keyword_match_rate,
    hit_at_k,
    iter_count_mean,
    keyword_coverage,
    loop_recovery_rate,
    ndcg_at_k,
    normalized_text_hash,
    p_percentile,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_at_k,
    top1_score_margin,
    topk_gap,
    unique_doc_coverage,
    wer,
)
from eval.harness.multimodal_eval import (
    MultimodalEvalRow,
    MultimodalEvalSummary,
    run_multimodal_eval,
)
from eval.harness.ocr_eval import OcrEvalRow, OcrEvalSummary, run_ocr_eval
from eval.harness.rag_eval import RagEvalRow, RagEvalSummary, run_rag_eval
from eval.harness.retrieval_eval import (
    DuplicateAnalysis,
    RetrievalEvalRow,
    RetrievalEvalSummary,
    TopKDumpRow,
    render_markdown_report,
    run_retrieval_eval,
)
from eval.harness.miss_analysis import (
    MissAnalysis,
    MissBucketSample,
    MissBucketStats,
    classify_rows as classify_miss_buckets,
    miss_analysis_to_dict,
    render_miss_analysis_markdown,
)
from eval.harness.baseline_comparison import (
    BaselineComparison,
    BaselineSlice,
    compute_baseline_slice,
    comparison_to_dict,
    render_comparison_markdown,
    run_comparison,
)

__all__ = [
    "answer_recall_delta",
    "avg_cost_multiplier",
    "cer",
    "count_whitespace_tokens",
    "dup_rate",
    "edit_distance",
    "expected_keyword_match_rate",
    "hit_at_k",
    "iter_count_mean",
    "keyword_coverage",
    "loop_recovery_rate",
    "ndcg_at_k",
    "normalized_text_hash",
    "p_percentile",
    "recall_at_k",
    "reciprocal_rank",
    "reciprocal_rank_at_k",
    "top1_score_margin",
    "topk_gap",
    "unique_doc_coverage",
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
    "DuplicateAnalysis",
    "RetrievalEvalRow",
    "RetrievalEvalSummary",
    "TopKDumpRow",
    "render_markdown_report",
    "run_retrieval_eval",
    "MissAnalysis",
    "MissBucketSample",
    "MissBucketStats",
    "classify_miss_buckets",
    "miss_analysis_to_dict",
    "render_miss_analysis_markdown",
    "BaselineComparison",
    "BaselineSlice",
    "compute_baseline_slice",
    "comparison_to_dict",
    "render_comparison_markdown",
    "run_comparison",
]
