"""Export a Phase 7.x human audit seed bundle.

Reads Phase 7.3's ``per_query_confidence.jsonl`` and (optionally) Phase
7.4's ``recovery_attempts.jsonl`` + the chunks_jsonl used by both
phases. Produces a small, stratified, human-readable audit bundle so a
reviewer can decide whether the Phase 7.0–7.4 silver labels actually
match the corpus.

Production code is NOT touched. The exporter is post-hoc analysis over
JSONL artefacts; nothing in ``app/capabilities/`` is invoked.

Usage::

    python -m scripts.export_phase7_human_audit_seed \\
        --confidence-jsonl       eval/reports/.../per_query_confidence.jsonl \\
        --recovery-attempts-jsonl eval/reports/.../recovery_attempts.jsonl \\
        --chunks                  eval/reports/.../rag_chunks_*.jsonl \\
        --out-dir                 eval/reports/.../human_audit/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from eval.harness.human_audit_export import (
    AuditExportConfig,
    DEFAULT_BUCKET_QUOTA,
    DEFAULT_EDGE_CASE_QUOTA_PER_TAG,
    DEFAULT_SNIPPET_MAX_CHARS,
    _default_edge_case_quotas,
    export_audit_bundle,
)


log = logging.getLogger("scripts.export_phase7_human_audit_seed")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export a Phase 7.x human audit seed bundle. Stratified by "
            "bucket and confidence/failure edge cases. Output is "
            "deterministic: same inputs + config → byte-identical "
            "JSONL/CSV/MD."
        ),
    )
    p.add_argument(
        "--confidence-jsonl", type=Path, required=True,
        help=(
            "Phase 7.3 per_query_confidence.jsonl. Source of the verdict, "
            "input block (query / silver target / top_candidates_preview), "
            "and bucket."
        ),
    )
    p.add_argument(
        "--recovery-attempts-jsonl", type=Path, default=None,
        help=(
            "Optional Phase 7.4 recovery_attempts.jsonl. When provided, "
            "audit rows whose qid had a recovery attempt gain the "
            "Phase 7.4 recovery block (action, rewrite_mode, "
            "rewritten_query, before/after rank) and are tagged "
            "with the synthetic 'query_rewrite_candidate' edge case."
        ),
    )
    p.add_argument(
        "--chunks", type=Path, default=None,
        help=(
            "Optional rag_chunks_*.jsonl. When provided, the top-5 "
            "chunk evidence rows in each audit record gain a real "
            "text snippet excerpted from chunk_text."
        ),
    )
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory the audit bundle (jsonl/csv/md/summary) is written to.",
    )
    p.add_argument(
        "--bundle-basename", default="phase7_human_audit_seed",
        help="Stem of the output filenames. Default 'phase7_human_audit_seed'.",
    )
    # Quota overrides (for narrower / wider audits).
    p.add_argument(
        "--main-work-quota", type=int,
        default=DEFAULT_BUCKET_QUOTA["main_work"],
        help="Main_work bucket quota.",
    )
    p.add_argument(
        "--subpage-generic-quota", type=int,
        default=DEFAULT_BUCKET_QUOTA["subpage_generic"],
        help="Subpage_generic bucket quota.",
    )
    p.add_argument(
        "--subpage-named-quota", type=int,
        default=DEFAULT_BUCKET_QUOTA["subpage_named"],
        help="Subpage_named bucket quota.",
    )
    p.add_argument(
        "--edge-case-quota-per-tag", type=int,
        default=DEFAULT_EDGE_CASE_QUOTA_PER_TAG,
        help=(
            "Per-tag quota for confidence-label, failure-reason, and "
            "synthetic edge cases. Default 5."
        ),
    )
    p.add_argument(
        "--snippet-max-chars", type=int, default=DEFAULT_SNIPPET_MAX_CHARS,
        help="Maximum length of the chunk-text snippet in each top-5 row.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bucket_quota = {
        "main_work": int(args.main_work_quota),
        "subpage_generic": int(args.subpage_generic_quota),
        "subpage_named": int(args.subpage_named_quota),
    }
    # Apply the per-tag override on top of the default tag list.
    edge_case_quota = _default_edge_case_quotas()
    if args.edge_case_quota_per_tag != DEFAULT_EDGE_CASE_QUOTA_PER_TAG:
        edge_case_quota = {
            tag: int(args.edge_case_quota_per_tag)
            for tag in edge_case_quota
        }

    cfg = AuditExportConfig(
        bucket_quota=bucket_quota,
        edge_case_quota=edge_case_quota,
        snippet_max_chars=int(args.snippet_max_chars),
    )

    log.info(
        "exporting audit bundle: confidence=%s recovery=%s chunks=%s "
        "out_dir=%s",
        args.confidence_jsonl,
        args.recovery_attempts_jsonl,
        args.chunks,
        args.out_dir,
    )

    paths = export_audit_bundle(
        confidence_jsonl=args.confidence_jsonl,
        out_dir=args.out_dir,
        recovery_attempts_jsonl=args.recovery_attempts_jsonl,
        chunks_jsonl=args.chunks,
        config=cfg,
        bundle_basename=args.bundle_basename,
    )
    for role, path in paths.items():
        log.info("  %s -> %s", role, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
