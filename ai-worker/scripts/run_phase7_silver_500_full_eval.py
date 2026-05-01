"""Phase 7 silver-500 full evaluation orchestrator (eval-only).

Wraps the five-step silver-500 evaluation stack into a single CLI:

  A. Phase 7.0 retrieval eval over the silver-500 queries
     (retrieval_title_section variant — the production default).
     Output dir: ``phase7/silver500/retrieval/``.

  B. Phase 7.3 confidence detector over (A)'s per_query_comparison.
     Output dir: ``phase7/silver500/confidence/``.

  C. Phase 7.4 controlled recovery over (B)'s confidence verdicts +
     (A)'s frozen dense top-N.
     Output dir: ``phase7/silver500/recovery/``.

  D. Phase 7 human-gold-seed audit export drawn from the silver-500
     query set + the (A)/(B)/(C) artefacts.
     Output dir: ``phase7/seeds/human_seed_50/``.

  E. ``PHASE7_SILVER500_FULL_EVAL_REPORT.md`` — orchestration report
     summarising every stage's distribution counts and stamping the
     silver-vs-gold disclaimer.

The orchestrator does NOT touch production code. Every output is
silver, not human-verified gold — the final report carries the
disclaimer and refuses to claim precision/recall/accuracy.

Usage::

    # Document-only: print the planned commands without running them.
    python -m scripts.run_phase7_silver_500_full_eval \\
        --pages-v4   PATH/pages_v4.jsonl \\
        --rag-chunks PATH/rag_chunks.jsonl \\
        --reports-root eval/reports/ \\
        --index-root   eval/indexes/ \\
        --print-commands

    # Run end-to-end (assumes Phase 7.0 indexes are pre-built):
    python -m scripts.run_phase7_silver_500_full_eval \\
        --pages-v4   PATH/pages_v4.jsonl \\
        --rag-chunks PATH/rag_chunks.jsonl \\
        --reports-root eval/reports/ \\
        --index-root   eval/indexes/ \\
        --run

The ``--run`` and ``--print-commands`` flags compose: ``--run`` does
the work, ``--print-commands`` always lists what would be done so the
log is self-documenting. ``--report-only`` is a third mode that skips
all stages and just rebuilds the orchestration report off existing
artefacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


log = logging.getLogger("scripts.run_phase7_silver_500_full_eval")


# Stage IDs — opaque identifiers used as dispatch keys in
# :func:`run_stage` and :class:`StageCommand`. The values are kept
# stable for backward compatibility with existing snapshot artefacts
# even though they no longer drive directory naming.
STAGE_SILVER = "phase7_silver500_queries"
STAGE_PHASE7_0 = "phase7_0_silver500_retrieval"
STAGE_PHASE7_3 = "phase7_3_silver500_confidence"
STAGE_PHASE7_4 = "phase7_4_silver500_recovery"
STAGE_HUMAN_SEED = "phase7_human_seed_50"

# Stage directories — relative to ``reports_root``. Produced by the
# 2026-05 reports/ reorg that grouped artefacts by phase rather than
# by corpus-version prefix.
_SILVER_DIR_PARTS = ("phase7", "silver500", "queries")
_PHASE7_0_DIR_PARTS = ("phase7", "silver500", "retrieval")
_PHASE7_3_DIR_PARTS = ("phase7", "silver500", "confidence")
_PHASE7_4_DIR_PARTS = ("phase7", "silver500", "recovery")
_HUMAN_SEED_DIR_PARTS = ("phase7", "seeds", "human_seed_50")
_PHASE7_0_SOURCE_DIR_PARTS = ("phase7", "7.0_retrieval_title_ab")

ALL_STAGES: Sequence[str] = (
    STAGE_SILVER,
    STAGE_PHASE7_0,
    STAGE_PHASE7_3,
    STAGE_PHASE7_4,
    STAGE_HUMAN_SEED,
)


# Filename constants — pinned by tests to keep the orchestration report
# stable when individual stage writers shift artefact names.
SILVER_JSONL_NAME = "queries_v4_silver_500.jsonl"
SILVER_SUMMARY_JSON_NAME = "queries_v4_silver_500.summary.json"
SILVER_SUMMARY_MD_NAME = "queries_v4_silver_500.summary.md"
PHASE7_0_PER_QUERY_NAME = "per_query_comparison.jsonl"
PHASE7_0_CHUNKS_NAME = "rag_chunks_retrieval_title_section.jsonl"
PHASE7_0_AB_SUMMARY_NAME = "ab_summary.json"
PHASE7_3_CONFIDENCE_NAME = "per_query_confidence.jsonl"
PHASE7_3_SUMMARY_NAME = "confidence_summary.json"
PHASE7_4_ATTEMPTS_NAME = "recovery_attempts.jsonl"
PHASE7_4_SUMMARY_NAME = "recovery_summary.json"
HUMAN_SEED_BASE_NAME = "phase7_human_gold_seed_50"
FINAL_REPORT_NAME = "PHASE7_SILVER500_FULL_EVAL_REPORT.md"


# Disclaimer lines pinned by the test suite.
SILVER_NOT_GOLD_DISCLAIMER = (
    "This silver-500 evaluation uses a *silver* query set "
    "(synthetic / deterministic), **not** human-verified gold."
)
NO_PR_CLAIM_DISCLAIMER = (
    "No precision / recall / accuracy claim should be made until "
    "human labels are filled in via the audit-seed export."
)
HUMAN_LABELS_REQUIRED_DISCLAIMER = (
    "Human-verified labels are REQUIRED before publishing any "
    "precision / recall / accuracy number."
)


# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase7SilverPaths:
    """Resolved output directories + artefact paths for the silver500 stack.

    Constructed once from ``reports_root`` and used by both command
    builders (so the printed commands match the report reader) and the
    orchestration-report renderer.
    """

    reports_root: Path

    # ------------------------------------------------------------------
    # Stage directories
    # ------------------------------------------------------------------

    @property
    def silver_dir(self) -> Path:
        return Path(self.reports_root, *_SILVER_DIR_PARTS)

    @property
    def phase7_0_dir(self) -> Path:
        return Path(self.reports_root, *_PHASE7_0_DIR_PARTS)

    @property
    def phase7_3_dir(self) -> Path:
        return Path(self.reports_root, *_PHASE7_3_DIR_PARTS)

    @property
    def phase7_4_dir(self) -> Path:
        return Path(self.reports_root, *_PHASE7_4_DIR_PARTS)

    @property
    def human_seed_dir(self) -> Path:
        return Path(self.reports_root, *_HUMAN_SEED_DIR_PARTS)

    # ------------------------------------------------------------------
    # Artefact paths
    # ------------------------------------------------------------------

    @property
    def silver_queries_jsonl(self) -> Path:
        return self.silver_dir / SILVER_JSONL_NAME

    @property
    def silver_summary_json(self) -> Path:
        return self.silver_dir / SILVER_SUMMARY_JSON_NAME

    @property
    def silver_summary_md(self) -> Path:
        return self.silver_dir / SILVER_SUMMARY_MD_NAME

    @property
    def phase7_0_per_query(self) -> Path:
        return self.phase7_0_dir / PHASE7_0_PER_QUERY_NAME

    @property
    def phase7_0_chunks(self) -> Path:
        return self.phase7_0_dir / PHASE7_0_CHUNKS_NAME

    @property
    def phase7_0_ab_summary(self) -> Path:
        return self.phase7_0_dir / PHASE7_0_AB_SUMMARY_NAME

    @property
    def phase7_3_confidence(self) -> Path:
        return self.phase7_3_dir / PHASE7_3_CONFIDENCE_NAME

    @property
    def phase7_3_summary(self) -> Path:
        return self.phase7_3_dir / PHASE7_3_SUMMARY_NAME

    @property
    def phase7_4_attempts(self) -> Path:
        return self.phase7_4_dir / PHASE7_4_ATTEMPTS_NAME

    @property
    def phase7_4_summary(self) -> Path:
        return self.phase7_4_dir / PHASE7_4_SUMMARY_NAME

    @property
    def human_seed_jsonl(self) -> Path:
        return self.human_seed_dir / f"{HUMAN_SEED_BASE_NAME}.jsonl"

    @property
    def human_seed_csv(self) -> Path:
        return self.human_seed_dir / f"{HUMAN_SEED_BASE_NAME}.csv"

    @property
    def human_seed_md(self) -> Path:
        return self.human_seed_dir / f"{HUMAN_SEED_BASE_NAME}.md"

    @property
    def final_report(self) -> Path:
        return self.silver_dir / FINAL_REPORT_NAME

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def all_stage_dirs(self) -> Dict[str, Path]:
        """Return ``{stage_slug: dir_path}`` for every stage."""
        return {
            STAGE_SILVER: self.silver_dir,
            STAGE_PHASE7_0: self.phase7_0_dir,
            STAGE_PHASE7_3: self.phase7_3_dir,
            STAGE_PHASE7_4: self.phase7_4_dir,
            STAGE_HUMAN_SEED: self.human_seed_dir,
        }


# ---------------------------------------------------------------------------
# Command builders — every stage has a deterministic argv layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageCommand:
    """One stage's invocation: module + argv plus a documentation note."""

    stage: str
    module: str
    argv: List[str]
    description: str

    def shell_form(self) -> str:
        """Return the equivalent shell line ``python -m <module> ...``."""
        return "python -m " + self.module + " " + " ".join(
            shlex.quote(a) for a in self.argv
        )


def build_silver_command(
    paths: Phase7SilverPaths,
    *,
    pages_v4: Path,
    seed: int = 42,
    main_work_target: int = 150,
    subpage_generic_target: int = 200,
    subpage_named_target: int = 150,
) -> StageCommand:
    """Build the silver-500 generation invocation."""
    argv = [
        "--pages-v4", str(pages_v4),
        "--out-dir", str(paths.silver_dir),
        "--seed", str(int(seed)),
        "--main-work-target", str(int(main_work_target)),
        "--subpage-generic-target", str(int(subpage_generic_target)),
        "--subpage-named-target", str(int(subpage_named_target)),
    ]
    return StageCommand(
        stage=STAGE_SILVER,
        module="scripts.run_phase7_silver_500",
        argv=argv,
        description=(
            "Generate the silver-500 query set + summary report. "
            "Output is silver, NOT human-verified gold."
        ),
    )


def build_phase7_0_command(
    paths: Phase7SilverPaths,
    *,
    rag_chunks: Path,
    pages_v4: Path,
    index_root: Path,
    embedding_model: str = "BAAI/bge-m3",
    max_seq_length: int = 512,
    top_k: int = 10,
) -> StageCommand:
    """Build the Phase 7.0 retrieval-eval invocation against silver500.

    Reuses the existing Phase 7.0 orchestrator with ``--queries`` set to
    the silver-500 jsonl. ``--skip-export / --skip-diff / --skip-index-
    build`` keep us off the heavy chunk-export + index-build path; the
    Phase 7.0 export already lives in the canonical location.
    """
    argv = [
        "--rag-chunks", str(rag_chunks),
        "--pages-v4", str(pages_v4),
        "--report-dir", str(paths.phase7_0_dir),
        "--index-root", str(index_root),
        "--embedding-model", embedding_model,
        "--max-seq-length", str(int(max_seq_length)),
        "--top-k", str(int(top_k)),
        "--queries", str(paths.silver_queries_jsonl),
        "--skip-export",
        "--skip-diff",
        "--skip-index-build",
    ]
    return StageCommand(
        stage=STAGE_PHASE7_0,
        module="scripts.run_phase7_0_retrieval_title_ab",
        argv=argv,
        description=(
            "Phase 7.0 retrieval A/B against the silver-500 set "
            "(retrieval_title_section is the production default). Skips "
            "chunk export / variant diff / index build — assumes the "
            "Phase 7.0 export is the source of truth."
        ),
    )


def build_phase7_3_command(
    paths: Phase7SilverPaths,
    *,
    side: str = "candidate",
) -> StageCommand:
    """Build the Phase 7.3 confidence-eval invocation against silver500."""
    argv = [
        "--per-query", str(paths.phase7_0_per_query),
        "--chunks", str(paths.phase7_0_chunks),
        "--silver-queries", str(paths.silver_queries_jsonl),
        "--side", side,
        "--report-dir", str(paths.phase7_3_dir),
    ]
    return StageCommand(
        stage=STAGE_PHASE7_3,
        module="scripts.run_phase7_3_confidence_eval",
        argv=argv,
        description=(
            "Phase 7.3 retrieval confidence detector + failure "
            "classifier over the silver-500 Phase 7.0 outputs."
        ),
    )


def build_phase7_4_command(
    paths: Phase7SilverPaths,
    *,
    rewrite_mode: str = "both",
    final_k: int = 10,
    hybrid_top_k: int = 10,
    bm25_pool_size: int = 100,
    side: str = "candidate",
    no_strict_label_leakage: bool = False,
) -> StageCommand:
    """Build the Phase 7.4 controlled-recovery invocation against silver500.

    ``rewrite_mode='both'`` runs oracle and production-like rewrites in
    parallel — Phase 7.4's recommended setting because it lets the
    report compare oracle upper-bound vs production-like recall.
    ``no_strict_label_leakage`` is left off by default; flipping it on
    relaxes the LabelLeakageError raise that the report needs to count.
    """
    argv = [
        "--confidence-jsonl", str(paths.phase7_3_confidence),
        "--per-query", str(paths.phase7_0_per_query),
        "--chunks", str(paths.phase7_0_chunks),
        "--silver-queries", str(paths.silver_queries_jsonl),
        "--report-dir", str(paths.phase7_4_dir),
        "--rewrite-mode", rewrite_mode,
        "--final-k", str(int(final_k)),
        "--hybrid-top-k", str(int(hybrid_top_k)),
        "--bm25-pool-size", str(int(bm25_pool_size)),
        "--side", side,
    ]
    if no_strict_label_leakage:
        argv.append("--no-strict-label-leakage")
    return StageCommand(
        stage=STAGE_PHASE7_4,
        module="scripts.run_phase7_4_controlled_recovery",
        argv=argv,
        description=(
            "Phase 7.4 controlled recovery loop (hybrid + query "
            "rewrite) over the silver-500 Phase 7.3 verdicts. Default "
            "rewrite_mode=both so oracle and production-like rewrites "
            "are both reported (the production-like leakage refusal "
            "count surfaces in the orchestration report)."
        ),
    )


def build_human_seed_command(
    paths: Phase7SilverPaths,
    *,
    target_total: int = 50,
    main_work_target: int = 10,
    subpage_generic_target: int = 20,
    subpage_named_target: int = 20,
    side: str = "candidate",
    base_name: str = HUMAN_SEED_BASE_NAME,
    seed: int = 42,
) -> StageCommand:
    """Build the human-gold-seed audit-export invocation against silver500."""
    argv = [
        "--silver-queries", str(paths.silver_queries_jsonl),
        "--per-query", str(paths.phase7_0_per_query),
        "--confidence", str(paths.phase7_3_confidence),
        "--recovery", str(paths.phase7_4_attempts),
        "--chunks", str(paths.phase7_0_chunks),
        "--out-dir", str(paths.human_seed_dir),
        "--target-total", str(int(target_total)),
        "--main-work-target", str(int(main_work_target)),
        "--subpage-generic-target", str(int(subpage_generic_target)),
        "--subpage-named-target", str(int(subpage_named_target)),
        "--side", side,
        "--seed", str(int(seed)),
        "--base-name", base_name,
    ]
    return StageCommand(
        stage=STAGE_HUMAN_SEED,
        module="scripts.export_phase7_human_gold_seed",
        argv=argv,
        description=(
            "Phase 7 human-gold-seed audit export drawn from the new "
            "silver-500 outputs. Every row's `human_label` is BLANK by "
            "design — the seed is silver-derived, NOT human-verified "
            "gold."
        ),
    )


@dataclass(frozen=True)
class OrchestrationPlan:
    """The five stage commands plus a stable ordered key list."""

    paths: Phase7SilverPaths
    silver: StageCommand
    phase7_0: StageCommand
    phase7_3: StageCommand
    phase7_4: StageCommand
    human_seed: StageCommand

    def as_ordered_list(self) -> List[StageCommand]:
        return [
            self.silver,
            self.phase7_0,
            self.phase7_3,
            self.phase7_4,
            self.human_seed,
        ]


def build_plan(
    *,
    paths: Phase7SilverPaths,
    pages_v4: Path,
    rag_chunks: Path,
    index_root: Path,
    embedding_model: str = "BAAI/bge-m3",
    max_seq_length: int = 512,
    top_k: int = 10,
    seed: int = 42,
    rewrite_mode: str = "both",
    final_k: int = 10,
    hybrid_top_k: int = 10,
    bm25_pool_size: int = 100,
    side: str = "candidate",
    main_work_target: int = 150,
    subpage_generic_target: int = 200,
    subpage_named_target: int = 150,
    seed_target_total: int = 50,
    seed_main_work_target: int = 10,
    seed_subpage_generic_target: int = 20,
    seed_subpage_named_target: int = 20,
    no_strict_label_leakage: bool = False,
) -> OrchestrationPlan:
    """Build the full plan in one call so tests can assert on it."""
    return OrchestrationPlan(
        paths=paths,
        silver=build_silver_command(
            paths,
            pages_v4=pages_v4,
            seed=seed,
            main_work_target=main_work_target,
            subpage_generic_target=subpage_generic_target,
            subpage_named_target=subpage_named_target,
        ),
        phase7_0=build_phase7_0_command(
            paths,
            rag_chunks=rag_chunks,
            pages_v4=pages_v4,
            index_root=index_root,
            embedding_model=embedding_model,
            max_seq_length=max_seq_length,
            top_k=top_k,
        ),
        phase7_3=build_phase7_3_command(paths, side=side),
        phase7_4=build_phase7_4_command(
            paths,
            rewrite_mode=rewrite_mode,
            final_k=final_k,
            hybrid_top_k=hybrid_top_k,
            bm25_pool_size=bm25_pool_size,
            side=side,
            no_strict_label_leakage=no_strict_label_leakage,
        ),
        human_seed=build_human_seed_command(
            paths,
            target_total=seed_target_total,
            main_work_target=seed_main_work_target,
            subpage_generic_target=seed_subpage_generic_target,
            subpage_named_target=seed_subpage_named_target,
            side=side,
            seed=seed,
        ),
    )


# ---------------------------------------------------------------------------
# In-process executor — calls each stage's ``main(argv)`` directly
# ---------------------------------------------------------------------------


def _stage_one_chunks_file(src: Path, dst: Path) -> Path:
    """Hardlink, falling back to copy, ``src`` -> ``dst``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import os as _os
        _os.link(str(src), str(dst))
        log.info("hardlinked %s -> %s", src, dst)
    except (OSError, NotImplementedError):
        import shutil as _shutil
        _shutil.copy2(str(src), str(dst))
        log.info("copied (no hardlink) %s -> %s", src, dst)
    return dst


def stage_existing_chunks(
    paths: Phase7SilverPaths,
    *,
    source_phase7_0_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Hardlink (or copy) the Phase 7.0 chunks file into the silver500 dir.

    The Phase 7.0 orchestrator's ``--skip-export`` validation requires
    ``rag_chunks_<variant>.jsonl`` to exist inside ``--report-dir`` —
    for *both* variants in ``V4_EXPORT_VARIANTS`` (``title_section`` +
    ``retrieval_title_section``). When we run Phase 7.0 against
    silver-500 we re-use the existing Phase 7.0 export rather than
    re-running the heavy chunk export, so the orchestrator pre-stages
    both chunk files by hardlinking (instant, zero copy) — falling
    back to a file copy when hardlinks aren't supported on the
    filesystem.

    ``source_phase7_0_dir`` defaults to the canonical Phase 7.0 report
    directory (``<reports_root>/phase7/7.0_retrieval_title_ab``).
    Returns the *retrieval_title_section* destination path (the one
    the silver500 stages downstream actually read). Returns ``None``
    when the candidate source file can't be located.
    """
    candidate_dst = paths.phase7_0_chunks
    baseline_dst = paths.phase7_0_dir / "rag_chunks_title_section.jsonl"
    candidate_present = candidate_dst.exists()
    baseline_present = baseline_dst.exists()
    if candidate_present and baseline_present:
        # No-op when the silver500 phase7_0 dir already has both files.
        return candidate_dst

    if source_phase7_0_dir is None:
        source_phase7_0_dir = Path(
            paths.reports_root, *_PHASE7_0_SOURCE_DIR_PARTS,
        )

    candidate_src = source_phase7_0_dir / PHASE7_0_CHUNKS_NAME
    baseline_src = source_phase7_0_dir / "rag_chunks_title_section.jsonl"

    if not candidate_present:
        if not candidate_src.exists():
            log.warning(
                "stage_existing_chunks: source missing - %s. The Phase "
                "7.0 orchestrator's validation will fail unless this "
                "file is produced by an earlier export pass.",
                candidate_src,
            )
            return None
        _stage_one_chunks_file(candidate_src, candidate_dst)

    # The baseline file is required by the Phase 7.0 validation step
    # but not used downstream by Phase 7.3/7.4. Stage it best-effort.
    if not baseline_present:
        if baseline_src.exists():
            _stage_one_chunks_file(baseline_src, baseline_dst)
        else:
            log.info(
                "stage_existing_chunks: baseline %s missing; Phase 7.0 "
                "validation will produce its own error message.",
                baseline_src,
            )

    return candidate_dst


def run_stage(
    cmd: StageCommand,
    *,
    paths: Optional[Phase7SilverPaths] = None,
    source_phase7_0_dir: Optional[Path] = None,
) -> int:
    """Invoke ``cmd``'s module ``main(argv)`` in the current process.

    Each Phase 7.x orchestrator already exposes a ``main(argv)`` that
    returns an int — running them in-process keeps the orchestration
    deterministic (no subprocess slowdown, no PYTHONPATH surprises) and
    lets the reporter inspect any exception raised by the stage.

    When ``cmd.stage == STAGE_PHASE7_0`` and ``paths`` is provided, the
    runner pre-stages the existing Phase 7.0 chunks file into the
    silver500 phase7_0 dir so the Phase 7.0 ``--skip-export`` validation
    passes.
    """
    log.info("[%s] %s", cmd.stage, cmd.shell_form())
    if cmd.stage == STAGE_PHASE7_0 and paths is not None:
        stage_existing_chunks(paths, source_phase7_0_dir=source_phase7_0_dir)
    if cmd.stage == STAGE_SILVER:
        from scripts.run_phase7_silver_500 import main as _m
    elif cmd.stage == STAGE_PHASE7_0:
        from scripts.run_phase7_0_retrieval_title_ab import main as _m
    elif cmd.stage == STAGE_PHASE7_3:
        from scripts.run_phase7_3_confidence_eval import main as _m
    elif cmd.stage == STAGE_PHASE7_4:
        from scripts.run_phase7_4_controlled_recovery import main as _m
    elif cmd.stage == STAGE_HUMAN_SEED:
        from scripts.export_phase7_human_gold_seed import main as _m
    else:
        raise ValueError(f"unknown stage: {cmd.stage!r}")
    return int(_m(cmd.argv))


def run_plan(
    plan: OrchestrationPlan,
    *,
    stop_on_error: bool = True,
    source_phase7_0_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """Run every stage; return ``{stage: return_code}``.

    When ``stop_on_error=True`` (default) the first non-zero rc aborts
    the run; remaining stages are reported as ``-1``.

    ``source_phase7_0_dir`` is forwarded to :func:`run_stage` for the
    Phase 7.0 stage's chunks-file pre-staging step.
    """
    rcs: Dict[str, int] = {}
    for cmd in plan.as_ordered_list():
        try:
            rc = run_stage(
                cmd, paths=plan.paths,
                source_phase7_0_dir=source_phase7_0_dir,
            )
        except SystemExit as e:
            rc = int(e.code) if e.code is not None else 0
        except Exception:  # noqa: BLE001
            log.exception("[%s] FAILED with unhandled exception", cmd.stage)
            rcs[cmd.stage] = 1
            if stop_on_error:
                for c in plan.as_ordered_list():
                    if c.stage not in rcs:
                        rcs[c.stage] = -1
                return rcs
            continue
        rcs[cmd.stage] = rc
        if rc != 0 and stop_on_error:
            for c in plan.as_ordered_list():
                if c.stage not in rcs:
                    rcs[c.stage] = -1
            return rcs
    return rcs


# ---------------------------------------------------------------------------
# Final orchestration report
# ---------------------------------------------------------------------------


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON, returning ``None`` on missing-file / parse errors.

    The orchestration report has to render even when an upstream stage
    failed — a half-empty report is more useful than a crash.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        log.warning("could not parse %s as json: %s", path, e)
        return None


def _count_jsonl_lines(path: Path) -> Optional[int]:
    """Count non-blank lines in a JSONL file. ``None`` when missing."""
    if not Path(path).exists():
        return None
    n = 0
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                n += 1
    return n


@dataclass
class ReportSnapshot:
    """Container holding everything the report renderer needs.

    Built by :func:`load_report_snapshot` from the artefacts on disk so
    the rendering function is pure and easy to test.
    """

    paths: Phase7SilverPaths
    silver_summary: Optional[Dict[str, Any]] = None
    phase7_0_ab_summary: Optional[Dict[str, Any]] = None
    phase7_0_query_count: Optional[int] = None
    phase7_3_summary: Optional[Dict[str, Any]] = None
    phase7_4_summary: Optional[Dict[str, Any]] = None
    human_seed_md_text: Optional[str] = None
    human_seed_jsonl_count: Optional[int] = None


def load_report_snapshot(paths: Phase7SilverPaths) -> ReportSnapshot:
    """Read every artefact the orchestration report can use."""
    snap = ReportSnapshot(paths=paths)
    snap.silver_summary = _safe_load_json(paths.silver_summary_json)
    snap.phase7_0_ab_summary = _safe_load_json(paths.phase7_0_ab_summary)
    snap.phase7_0_query_count = _count_jsonl_lines(paths.phase7_0_per_query)
    snap.phase7_3_summary = _safe_load_json(paths.phase7_3_summary)
    snap.phase7_4_summary = _safe_load_json(paths.phase7_4_summary)
    snap.human_seed_jsonl_count = _count_jsonl_lines(paths.human_seed_jsonl)
    if paths.human_seed_md.exists():
        try:
            snap.human_seed_md_text = paths.human_seed_md.read_text(
                encoding="utf-8",
            )
        except OSError:  # pragma: no cover - defensive
            snap.human_seed_md_text = None
    return snap


def _bucket_table(
    targets: Mapping[str, Any], actual: Mapping[str, Any],
    deficits: Mapping[str, Any], *, bucket_order: Sequence[str],
) -> List[str]:
    """Render a 4-column bucket table (target / actual / deficit)."""
    out: List[str] = []
    out.append("| bucket | target | actual | deficit |")
    out.append("|---|---:|---:|---:|")
    for b in bucket_order:
        out.append(
            f"| {b} | {int(targets.get(b, 0))} | "
            f"{int(actual.get(b, 0))} | {int(deficits.get(b, 0))} |"
        )
    return out


def _format_int_dict_table(
    title_pair: tuple[str, str], counts: Mapping[str, Any],
) -> List[str]:
    """Render a generic 2-column key/count table."""
    out: List[str] = []
    out.append(f"| {title_pair[0]} | {title_pair[1]} |")
    out.append("|---|---:|")
    for k in counts:
        out.append(f"| {k} | {int(counts.get(k, 0) or 0)} |")
    return out


def render_final_report(snap: ReportSnapshot) -> str:
    """Render ``PHASE7_SILVER500_FULL_EVAL_REPORT.md`` from a snapshot.

    The report intentionally does NOT include precision / recall /
    accuracy figures. It surfaces the silver-vs-gold disclaimer in two
    places (header banner + footer reminder) and lists every artefact
    location so a reviewer can navigate from the report to the raw data.
    """
    paths = snap.paths
    lines: List[str] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines.append("# Phase 7 silver-500 full evaluation — orchestration report")
    lines.append("")
    lines.append(f"> {SILVER_NOT_GOLD_DISCLAIMER}")
    lines.append(">")
    lines.append(f"> {NO_PR_CLAIM_DISCLAIMER}")
    lines.append(">")
    lines.append(f"> {HUMAN_LABELS_REQUIRED_DISCLAIMER}")
    lines.append("")
    lines.append(
        "Every metric below is a **silver-agreement metric** — agreement "
        "with the silver `expected_doc_id` / `expected_title` targets "
        "produced by the silver-500 generator. Silver labels are template-"
        "driven; they have NOT been verified by a human reviewer."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Stage 1 — silver-500 generation
    # ------------------------------------------------------------------
    lines.append("## 1. Silver-500 query set")
    lines.append("")
    lines.append(f"- Queries: `{paths.silver_queries_jsonl}`")
    lines.append(f"- Summary (json): `{paths.silver_summary_json}`")
    lines.append(f"- Summary (md):   `{paths.silver_summary_md}`")
    lines.append("")
    if snap.silver_summary is not None:
        s = snap.silver_summary
        lines.append(
            f"- requested_total: **{int(s.get('requested_total', 0))}**"
        )
        lines.append(
            f"- actual_total: **{int(s.get('actual_total', 0))}**"
        )
        lines.append(
            f"- seed: **{s.get('seed')}**"
        )
        lines.append("")
        lines.append("### Bucket distribution (silver)")
        lines.append("")
        lines += _bucket_table(
            s.get("bucket_targets") or {},
            s.get("bucket_actual_counts") or {},
            s.get("bucket_deficits") or {},
            bucket_order=("main_work", "subpage_generic", "subpage_named"),
        )
        lines.append("")
        lines.append("### Silver-label confidence distribution")
        lines.append("")
        conf = s.get("label_confidence_counts") or {}
        lines += _format_int_dict_table(
            ("confidence", "count"),
            {k: conf.get(k, 0) for k in ("high", "medium", "low")},
        )
        lines.append("")
        lines.append("### Template-kind distribution")
        lines.append("")
        tk = s.get("template_kind_counts") or {}
        if tk:
            lines += _format_int_dict_table(
                ("template_kind", "count"),
                {k: tk[k] for k in tk},
            )
            lines.append("")
    else:
        lines.append("> Silver summary not available (stage A did not run).")
        lines.append("")

    # ------------------------------------------------------------------
    # Stage 2 — Phase 7.0 retrieval over silver-500
    # ------------------------------------------------------------------
    lines.append("## 2. Phase 7.0 retrieval over silver-500")
    lines.append("")
    lines.append(f"- Per-query comparison: `{paths.phase7_0_per_query}`")
    lines.append(f"- A/B summary (json):   `{paths.phase7_0_ab_summary}`")
    lines.append(f"- Chunks (variant):     `{paths.phase7_0_chunks}`")
    lines.append("")
    if snap.phase7_0_query_count is not None:
        lines.append(
            f"- silver queries scored: **{snap.phase7_0_query_count}**"
        )
    if snap.phase7_0_ab_summary is not None:
        ab = snap.phase7_0_ab_summary
        cand = ab.get("candidate") or {}
        base = ab.get("baseline") or {}
        # The Phase 7.0 ab_summary stores metrics directly on the side
        # block (``hit_at_1`` / ``mrr_at_10`` / ...). The orchestrator
        # report uses the canonical ``hit@k`` / ``mrr@k`` names so the
        # silver-vs-gold disclaimer language stays consistent.
        _METRIC_KEY_MAP = (
            ("hit@1", "hit_at_1"),
            ("hit@3", "hit_at_3"),
            ("hit@5", "hit_at_5"),
            ("hit@10", "hit_at_10"),
            ("mrr@10", "mrr_at_10"),
            ("ndcg@10", "ndcg_at_10"),
        )
        if isinstance(cand, dict) and any(
            k_src in cand for _, k_src in _METRIC_KEY_MAP
        ):
            lines.append("")
            lines.append(
                "### Candidate (`retrieval_title_section`) silver-agreement metrics"
            )
            lines.append("")
            lines.append("| metric | candidate | baseline | delta |")
            lines.append("|---|---:|---:|---:|")
            for k_dst, k_src in _METRIC_KEY_MAP:
                if k_src not in cand:
                    continue
                cv = float(cand.get(k_src) or 0.0)
                bv = float(base.get(k_src) or 0.0) if isinstance(base, dict) else 0.0
                lines.append(
                    f"| {k_dst} | {cv:.4f} | {bv:.4f} | {cv - bv:+.4f} |"
                )
            lines.append("")
        status_counts = ab.get("status_counts")
        if isinstance(status_counts, dict):
            lines.append("### Status counts (improved / regressed / both_hit / both_missed)")
            lines.append("")
            lines += _format_int_dict_table(
                ("status", "count"), status_counts,
            )
            lines.append("")
    else:
        lines.append("")
        lines.append("> Phase 7.0 ab_summary not available (stage B did not run).")
        lines.append("")

    # ------------------------------------------------------------------
    # Stage 3 — Phase 7.3 confidence
    # ------------------------------------------------------------------
    lines.append("## 3. Phase 7.3 confidence detector over silver-500")
    lines.append("")
    lines.append(f"- Per-query confidence: `{paths.phase7_3_confidence}`")
    lines.append(f"- Summary (json):       `{paths.phase7_3_summary}`")
    lines.append("")
    if snap.phase7_3_summary is not None:
        c = snap.phase7_3_summary
        labels = c.get("labels") or {}
        actions = c.get("actions") or {}
        lines.append("### Confidence label distribution")
        lines.append("")
        lines += _format_int_dict_table(
            ("confidence_label", "count"),
            {k: labels.get(k, 0) for k in (
                "CONFIDENT", "AMBIGUOUS", "LOW_CONFIDENCE", "FAILED",
            )},
        )
        lines.append("")
        lines.append("### Recommended-action distribution")
        lines.append("")
        lines += _format_int_dict_table(
            ("recommended_action", "count"),
            {k: actions.get(k, 0) for k in (
                "ANSWER", "ANSWER_WITH_CAUTION",
                "HYBRID_RECOVERY", "QUERY_REWRITE",
                "ASK_CLARIFICATION", "INSUFFICIENT_EVIDENCE",
            )},
        )
        lines.append("")
    else:
        lines.append("> Phase 7.3 summary not available (stage C did not run).")
        lines.append("")

    # ------------------------------------------------------------------
    # Stage 4 — Phase 7.4 controlled recovery
    # ------------------------------------------------------------------
    lines.append("## 4. Phase 7.4 controlled recovery over silver-500")
    lines.append("")
    lines.append(f"- Recovery attempts: `{paths.phase7_4_attempts}`")
    lines.append(f"- Summary (json):    `{paths.phase7_4_summary}`")
    lines.append("")
    if snap.phase7_4_summary is not None:
        r = snap.phase7_4_summary
        totals = r.get("totals") or {}
        invariants = r.get("invariants") or {}
        by_action = r.get("by_action") or {}
        rewrite_mode = (r.get("config") or {}).get("rewrite_mode")
        lines.append(f"- rewrite_mode: **{rewrite_mode!s}**")
        lines.append(f"- n_decisions: **{int(r.get('n_queries', 0))}**")
        lines.append(f"- attempted: **{int(totals.get('attempted', 0))}**")
        lines.append(f"- recovered: **{int(totals.get('recovered', 0))}**")
        lines.append(f"- regressed: **{int(totals.get('regressed', 0))}**")
        lines.append(
            f"- gold_newly_entered_candidates: "
            f"**{int(totals.get('gold_newly_entered_candidates', 0))}**"
        )
        lines.append("")
        # Per-action counts
        lines.append("### Recovery action counts")
        lines.append("")
        lines.append("| action | count | attempted | recovered | regressed |")
        lines.append("|---|---:|---:|---:|---:|")
        for action_name in (
            "NOOP", "SKIP_REFUSE", "SKIP_CAUTION", "SKIP_DEFER",
            "ATTEMPT_HYBRID", "ATTEMPT_REWRITE",
        ):
            block = by_action.get(action_name) or {}
            if not block:
                continue
            lines.append(
                f"| {action_name} | {int(block.get('count', 0))} "
                f"| {int(block.get('attempted', 0))} "
                f"| {int(block.get('recovered', 0))} "
                f"| {int(block.get('regressed', 0))} |"
            )
        lines.append("")
        # Production-like rewrite leakage
        leakage = int(invariants.get("label_leakage_refused_count", 0))
        attempt_rewrite_block = by_action.get("ATTEMPT_REWRITE") or {}
        rewrite_attempted = int(attempt_rewrite_block.get("attempted", 0))
        # Oracle vs production-like
        by_rewrite = r.get("by_rewrite_mode") or {}
        oracle_attempted = int((by_rewrite.get("oracle") or {}).get("attempted", 0))
        prod_attempted = int(
            (by_rewrite.get("production_like") or {}).get("attempted", 0)
        )
        rewrite_action_total = int(attempt_rewrite_block.get("count", 0))
        # Phase 7.3 → QUERY_REWRITE recommended count is the count
        # Phase 7.4 saw at the entry (== ATTEMPT_REWRITE.count when no
        # row was filtered).
        query_rewrite_recommended = rewrite_action_total
        lines.append("### Production-like rewrite behavior (silver-500)")
        lines.append("")
        lines.append(
            f"- QUERY_REWRITE recommended (Phase 7.3 → Phase 7.4 input): "
            f"**{int(query_rewrite_recommended)}**"
        )
        lines.append(
            f"- ATTEMPT_REWRITE total attempts (oracle + production_like): "
            f"**{int(rewrite_attempted)}**"
        )
        lines.append(
            f"- ATTEMPT_REWRITE — oracle attempts: **{oracle_attempted}**"
        )
        lines.append(
            f"- ATTEMPT_REWRITE — production_like attempts: **{prod_attempted}**"
        )
        lines.append(
            f"- production_like LabelLeakageError refused: **{leakage}**"
        )
        if leakage > 0:
            lines.append("")
            lines.append(
                "> The `expected_title` presence on the QUERY_REWRITE rows "
                "caused the production-like rewriter to refuse — this is "
                "the strict-leakage guard firing. Rows are recorded in "
                "`recovery_attempts.jsonl` with `recovery_action="
                "SKIP_DEFER` and `skip_reason=LABEL_LEAKAGE_REFUSED`. "
                "**The leakage guard is NOT fixed in this orchestration "
                "phase — it is reported as-is.**"
            )
        else:
            lines.append("")
            lines.append(
                "> No production_like rewrite was refused for label "
                "leakage on this run. (When `expected_title` is absent "
                "from the verdict row the guard is a no-op.)"
            )
        lines.append("")
        lines.append("### Invariants")
        lines.append("")
        lines.append(
            f"- ANSWER_WITH_CAUTION skip count: "
            f"**{int(invariants.get('answer_with_caution_skip_count', 0))}** "
            f"(recovered: "
            f"{invariants.get('answer_with_caution_recovered', False)!s})"
        )
        lines.append(
            f"- INSUFFICIENT_EVIDENCE refused count: "
            f"**{int(invariants.get('insufficient_evidence_refused_count', 0))}** "
            f"(recovered: "
            f"{invariants.get('insufficient_evidence_recovered', False)!s})"
        )
        lines.append(
            f"- ASK_CLARIFICATION deferred count: "
            f"**{int(invariants.get('ask_clarification_deferred_count', 0))}**"
        )
        lines.append("")
    else:
        lines.append("> Phase 7.4 summary not available (stage D did not run).")
        lines.append("")

    # ------------------------------------------------------------------
    # Stage 5 — human-gold-seed audit export
    # ------------------------------------------------------------------
    lines.append("## 5. Human-gold-seed audit export from silver-500")
    lines.append("")
    lines.append(f"- JSONL: `{paths.human_seed_jsonl}`")
    lines.append(f"- CSV:   `{paths.human_seed_csv}`")
    lines.append(f"- MD:    `{paths.human_seed_md}`")
    lines.append("")
    if snap.human_seed_jsonl_count is not None:
        lines.append(
            f"- audit rows emitted: **{snap.human_seed_jsonl_count}**"
        )
    if snap.human_seed_md_text:
        # Carry forward the bucket coverage table by inlining the first
        # bucket-distribution table from the seed's md (best effort).
        lines.append("")
        lines.append("> The seed's per-row table lives in the human-seed "
                     "MD next to this report. Every row's `human_label` "
                     "column is BLANK by design.")
        lines.append("")
    lines.append(
        "All `human_label` / `human_correct_*` / `human_notes` columns "
        "are **blank** by design. The seed is **silver-derived**, NOT "
        "human-verified gold."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Footer disclaimer
    # ------------------------------------------------------------------
    lines.append("## 6. Reminders")
    lines.append("")
    lines.append(f"- {SILVER_NOT_GOLD_DISCLAIMER}")
    lines.append(f"- {NO_PR_CLAIM_DISCLAIMER}")
    lines.append(f"- {HUMAN_LABELS_REQUIRED_DISCLAIMER}")
    lines.append("")
    lines.append(
        "- This phase did **not** modify any production code. Every "
        "stage runs as an eval-only post-hoc analysis."
    )
    lines.append(
        "- The silver-500 generator output is deterministic under a "
        "fixed seed; rerunning the orchestration with the same seed "
        "produces byte-identical query JSONL."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def write_final_report(snap: ReportSnapshot) -> Path:
    """Render and persist the orchestration report under ``silver_dir``."""
    body = render_final_report(snap)
    out_path = snap.paths.final_report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 7 silver-500 full evaluation orchestrator. Documents "
            "and (optionally) executes the five-step pipeline. Output "
            "is silver, NOT human-verified gold."
        ),
    )
    p.add_argument(
        "--reports-root", type=Path, required=True,
        help="Eval reports root (e.g. eval/reports/).",
    )
    p.add_argument(
        "--pages-v4", type=Path, default=None,
        help=(
            "Phase 6.3 pages_v4.jsonl input (required for stage A and "
            "stage B). Not needed when --report-only is set."
        ),
    )
    p.add_argument(
        "--rag-chunks", type=Path, default=None,
        help=(
            "Phase 6.3 rag_chunks.jsonl input (required for stage B "
            "when re-exporting; ignored when --skip-export is implicit "
            "via reuse of an existing Phase 7.0 export)."
        ),
    )
    p.add_argument(
        "--index-root", type=Path, default=None,
        help=(
            "Cache root under which the per-variant FAISS indexes live "
            "(required for stage B). Indexes are assumed pre-built."
        ),
    )
    p.add_argument(
        "--embedding-model", default="BAAI/bge-m3",
        help="Sentence-transformers model name (default BAAI/bge-m3).",
    )
    p.add_argument(
        "--max-seq-length", type=int, default=512,
        help="Embedder max_seq_length (default 512).",
    )
    p.add_argument(
        "--top-k", type=int, default=10,
        help="Top-k for the Phase 7.0 retrieval (default 10).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--main-work-target", type=int, default=150,
        help="Silver-500 bucket target — main_work.",
    )
    p.add_argument(
        "--subpage-generic-target", type=int, default=200,
        help="Silver-500 bucket target — subpage_generic.",
    )
    p.add_argument(
        "--subpage-named-target", type=int, default=150,
        help="Silver-500 bucket target — subpage_named.",
    )
    p.add_argument(
        "--rewrite-mode", default="both",
        choices=("oracle", "production_like", "both"),
        help="Phase 7.4 rewrite mode (default both).",
    )
    p.add_argument("--final-k", type=int, default=10)
    p.add_argument("--hybrid-top-k", type=int, default=10)
    p.add_argument("--bm25-pool-size", type=int, default=100)
    p.add_argument(
        "--side", default="candidate", choices=("baseline", "candidate"),
        help="Which side of the Phase 7.0 A/B feeds Phase 7.3 / 7.4.",
    )
    p.add_argument(
        "--no-strict-label-leakage", action="store_true",
        help=(
            "Pass through to Phase 7.4 — disables the LabelLeakageError "
            "raise on production_like. Diagnostic only."
        ),
    )
    p.add_argument("--seed-target-total", type=int, default=50)
    p.add_argument("--seed-main-work-target", type=int, default=10)
    p.add_argument("--seed-subpage-generic-target", type=int, default=20)
    p.add_argument("--seed-subpage-named-target", type=int, default=20)

    p.add_argument(
        "--print-commands", action="store_true",
        help="Print the planned per-stage shell commands.",
    )
    p.add_argument(
        "--run", action="store_true",
        help="Execute every stage in-process (calls each stage's main).",
    )
    p.add_argument(
        "--report-only", action="store_true",
        help=(
            "Skip every stage; just rebuild "
            "PHASE7_SILVER500_FULL_EVAL_REPORT.md off existing artefacts."
        ),
    )
    p.add_argument(
        "--no-stop-on-error", action="store_true",
        help=(
            "When --run is set, keep running stages even if an earlier "
            "one returned non-zero. The orchestration report is still "
            "written off whatever artefacts exist."
        ),
    )
    p.add_argument(
        "--source-phase7-0-dir", type=Path, default=None,
        help=(
            "Existing Phase 7.0 report directory whose chunks JSONL is "
            "hardlinked into the silver500 phase7_0 dir before the "
            "silver500 retrieval stage runs. Defaults to "
            "<reports-root>/phase7/7.0_retrieval_title_ab/ — set this "
            "only when you want to point at a different Phase 7.0 export."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _build_paths(args: argparse.Namespace) -> Phase7SilverPaths:
    return Phase7SilverPaths(reports_root=Path(args.reports_root))


def _build_plan(args: argparse.Namespace) -> OrchestrationPlan:
    paths = _build_paths(args)
    if args.pages_v4 is None:
        raise SystemExit(
            "--pages-v4 is required when building the plan. Use "
            "--report-only to skip plan construction."
        )
    if args.rag_chunks is None:
        raise SystemExit("--rag-chunks is required when building the plan.")
    if args.index_root is None:
        raise SystemExit("--index-root is required when building the plan.")
    return build_plan(
        paths=paths,
        pages_v4=Path(args.pages_v4),
        rag_chunks=Path(args.rag_chunks),
        index_root=Path(args.index_root),
        embedding_model=args.embedding_model,
        max_seq_length=int(args.max_seq_length),
        top_k=int(args.top_k),
        seed=int(args.seed),
        rewrite_mode=str(args.rewrite_mode),
        final_k=int(args.final_k),
        hybrid_top_k=int(args.hybrid_top_k),
        bm25_pool_size=int(args.bm25_pool_size),
        side=str(args.side),
        main_work_target=int(args.main_work_target),
        subpage_generic_target=int(args.subpage_generic_target),
        subpage_named_target=int(args.subpage_named_target),
        seed_target_total=int(args.seed_target_total),
        seed_main_work_target=int(args.seed_main_work_target),
        seed_subpage_generic_target=int(args.seed_subpage_generic_target),
        seed_subpage_named_target=int(args.seed_subpage_named_target),
        no_strict_label_leakage=bool(args.no_strict_label_leakage),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    paths = _build_paths(args)

    if args.report_only:
        snap = load_report_snapshot(paths)
        out_path = write_final_report(snap)
        log.info("orchestration report written to %s", out_path)
        return 0

    plan = _build_plan(args)

    if args.print_commands:
        log.info("Phase 7 silver-500 full eval - planned commands")
        for cmd in plan.as_ordered_list():
            log.info("[%s] %s", cmd.stage, cmd.description)
            log.info("CMD: %s", cmd.shell_form())

    if args.run:
        rcs = run_plan(
            plan,
            stop_on_error=not args.no_stop_on_error,
            source_phase7_0_dir=args.source_phase7_0_dir,
        )
        log.info("stage return codes: %s", rcs)
        # Always write the orchestration report off whatever artefacts
        # exist — even when an upstream stage failed, the report is
        # still useful and surfaces the gap.
        snap = load_report_snapshot(paths)
        out_path = write_final_report(snap)
        log.info("orchestration report written to %s", out_path)
        # Exit non-zero if any stage failed.
        if any(rc != 0 for rc in rcs.values()):
            return max(rc for rc in rcs.values() if rc is not None)
        return 0

    # Default: just log the plan - no work done.
    if not args.print_commands:
        for cmd in plan.as_ordered_list():
            log.info("[%s] %s", cmd.stage, cmd.shell_form())
    return 0


if __name__ == "__main__":
    sys.exit(main())
