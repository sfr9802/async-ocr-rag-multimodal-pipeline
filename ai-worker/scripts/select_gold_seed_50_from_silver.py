"""Select balanced human-review gold seed candidates from a silver query set.

Example:
    python -m scripts.select_gold_seed_50_from_silver \
      --silver-path eval/reports/phase7/7.12_silver_manual_curated/queries_v4_silver_manual_curated_500.jsonl \
      --out-dir eval/reports/phase7/seeds/gold_seed_50_manual_curated \
      --target-count 50 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from eval.harness.gold_seed_sampler import (
    DEFAULT_TARGET_QUERY_TYPE_COUNTS,
    distribution,
    issue_distribution,
    load_jsonl,
    risk_review_rows,
    select_gold_seed_candidates,
    write_sampling_bundle,
)


log = logging.getLogger("scripts.select_gold_seed_50_from_silver")

DEFAULT_SILVER_PATH = Path(
    "eval/reports/phase7/7.12_silver_manual_curated/queries_v4_silver_manual_curated_500.jsonl"
)
DEFAULT_OUT_DIR = Path("eval/reports/phase7/seeds/gold_seed_50_manual_curated")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select balanced gold_seed_50 review candidates from the "
            "manual-curated Phase 7 v4 silver JSONL."
        )
    )
    parser.add_argument("--silver-path", type=Path, default=DEFAULT_SILVER_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    silver_path = _resolve(args.silver_path)
    out_dir = _resolve(args.out_dir)
    rows = load_jsonl(silver_path)
    result = select_gold_seed_candidates(
        rows,
        input_path=str(silver_path),
        target_count=args.target_count,
        seed=args.seed,
        query_type_targets=DEFAULT_TARGET_QUERY_TYPE_COUNTS,
    )
    paths = write_sampling_bundle(result, out_dir)
    payload = {
        "silver_path": str(silver_path),
        "out_dir": str(out_dir),
        "paths": {key: str(value) for key, value in paths.items()},
        "total_silver_count": result.total_silver_count,
        "schema_valid_count": len(result.schema_valid_rows),
        "valid_candidate_count": len(result.eligible_rows),
        "invalid_rejected_candidate_count": len(result.invalid_issues)
        + len(result.rejected_issues),
        "selected_count": len(result.selected_gold_rows),
        "selected_distributions": {
            "query_type": distribution(result.selected_gold_rows, "query_type"),
            "difficulty": distribution(result.selected_gold_rows, "difficulty"),
            "title_mention_level": distribution(
                result.selected_gold_rows, "title_mention_level"
            ),
            "answerability": distribution(result.selected_gold_rows, "answerability"),
        },
        "invalid_rejected_reason_counts": issue_distribution(
            [*result.invalid_issues, *result.rejected_issues]
        ),
        "risk_review_samples": risk_review_rows(result.selected_gold_rows, limit=5),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.info("wrote gold seed candidate bundle to %s", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
