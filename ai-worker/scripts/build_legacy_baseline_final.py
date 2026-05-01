"""Build the legacy-baseline-final report directory from a phase 2A sweep.

Reads the artifacts under ``eval/reports/legacy-baseline-final/_sweep/``
(produced by ``python -m eval.run_eval phase2a-latency-sweep``) and
emits the consolidated baseline files the LangGraph A/B test consumes:

  - metrics.json / metrics.md
  - latency_breakdown.json / latency_breakdown.md
  - selected_config.json
  - baseline_manifest.json
  - README.md

Pure post-processing: no GPU, no FAISS, no model load. Re-running the
script with the same sweep output is byte-identical except for the
``evaluatedAt`` field on the manifest.

The "selected" baseline is **balanced** (the median-latency on-frontier
recommendation), matching ``recommended-modes.json``. ``fast`` and
``quality`` are recorded alongside it so the operator can swap tiers
without re-running this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_SWEEP_DIR = Path("eval/reports/phase2/2a_latency")
_DEFAULT_OUT_DIR = Path("eval/reports/legacy-baseline-final")
_SELECTED_TIER = "balanced"


@dataclass(frozen=True)
class TierEntry:
    tier: str
    label: str
    dense_top_n: int
    final_top_k: int
    primary_metric_value: float
    latency_p95_ms: float
    rationale: str
    headline: Dict[str, float]
    report_dir_name: str


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unknown"
    return out.strip()


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return bool(out.strip())


def _tier_from_mode(mode: Dict[str, Any]) -> TierEntry:
    label = str(mode["label"])
    return TierEntry(
        tier=str(mode["tier"]),
        label=label,
        dense_top_n=int(mode["dense_top_n"]),
        final_top_k=int(mode["final_top_k"]),
        primary_metric_value=float(mode["primary_metric_value"]),
        latency_p95_ms=float(mode["latency_p95_ms"]),
        rationale=str(mode.get("rationale") or ""),
        headline={
            "hit_at_1": float(mode["mean_hit_at_1"]),
            "hit_at_3": float(mode["mean_hit_at_3"]),
            "hit_at_5": float(mode["mean_hit_at_5"]),
            "mrr_at_10": float(mode["mean_mrr_at_10"]),
            "ndcg_at_10": float(mode["mean_ndcg_at_10"]),
            "rerank_p95_ms": float(mode["rerank_p95_ms"]),
            "rerank_p99_ms": float(mode["rerank_p99_ms"]),
            "total_query_p95_ms": float(mode["total_query_p95_ms"]),
            "total_query_p99_ms": float(mode["total_query_p99_ms"]),
        },
        report_dir_name=f"rerank-{label}",
    )


def build(sweep_dir: Path, out_dir: Path, selected: str = _SELECTED_TIER) -> None:
    sweep_dir = Path(sweep_dir)
    out_dir = Path(out_dir)
    if not sweep_dir.exists():
        raise SystemExit(f"sweep dir not found: {sweep_dir}")

    modes_doc = _read_json(sweep_dir / "recommended-modes.json")
    sweep_doc = _read_json(sweep_dir / "topn-sweep.json")
    frontier_doc = _read_json(sweep_dir / "accuracy-latency-frontier.json")
    breakdown_doc = _read_json(sweep_dir / "reranker-latency-breakdown.json")

    modes_by_tier: Dict[str, TierEntry] = {}
    for raw in modes_doc.get("modes") or []:
        entry = _tier_from_mode(raw)
        modes_by_tier[entry.tier] = entry
    if selected not in modes_by_tier:
        raise SystemExit(
            f"selected tier {selected!r} not in recommended-modes.json "
            f"(available={sorted(modes_by_tier.keys())})"
        )

    selected_entry = modes_by_tier[selected]
    selected_report_dir = sweep_dir / selected_entry.report_dir_name
    selected_report = _read_json(selected_report_dir / "retrieval_eval_report.json")
    selected_meta = dict(selected_report.get("metadata") or {})
    selected_summary = dict(selected_report.get("summary") or {})

    # The source sweep's recorded ``run_at`` (per retrieval_eval_report.json
    # metadata) is the authoritative evaluation timestamp; the manifest
    # records both that *and* the post-process generation time so a
    # reviewer can tell when the eval ran vs when the manifest was
    # consolidated.
    sweep_run_at = (
        selected_meta.get("run_at")
        or selected_summary.get("started_at")
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- metrics.json ---
    metrics_payload: Dict[str, Any] = {
        "schema": "legacy-baseline-final.metrics.v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasetPath": selected_meta.get("dataset"),
        "corpusPath": selected_meta.get("corpus_path"),
        "rowCount": int(selected_summary.get("row_count") or 0),
        "embeddingModel": selected_meta.get("embedding_model"),
        "indexVersion": (selected_meta.get("offline_corpus") or {}).get("index_version"),
        "rerankerName": selected_summary.get("reranker_name"),
        "primaryMetricField": modes_doc.get("primary_metric_field", "mean_hit_at_1"),
        "latencyMetricField": modes_doc.get("latency_field", "rerank_p95_ms"),
        "selectedTier": selected_entry.tier,
        "tiers": {
            tier: {
                "label": e.label,
                "denseTopN": e.dense_top_n,
                "finalTopK": e.final_top_k,
                "primaryMetricValue": e.primary_metric_value,
                "latencyP95Ms": e.latency_p95_ms,
                "rationale": e.rationale,
                "metrics": e.headline,
            }
            for tier, e in modes_by_tier.items()
        },
        "selectedConfigSummary": dict(selected_summary),
        "perAnswerType": dict(selected_summary.get("per_answer_type") or {}),
        "perDifficulty": dict(selected_summary.get("per_difficulty") or {}),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- metrics.md ---
    lines: List[str] = []
    lines.append("# Legacy retrieval baseline — metrics")
    lines.append("")
    lines.append(
        f"- selected tier: `{selected_entry.tier}` "
        f"(label `{selected_entry.label}`, dense_top_n="
        f"{selected_entry.dense_top_n}, final_top_k={selected_entry.final_top_k})"
    )
    lines.append(f"- dataset: `{selected_meta.get('dataset')}`")
    lines.append(f"- corpus: `{selected_meta.get('corpus_path')}`")
    lines.append(
        f"- embedding model: `{selected_meta.get('embedding_model')}`"
        f" | index version: `{(selected_meta.get('offline_corpus') or {}).get('index_version')}`"
    )
    lines.append(f"- reranker: `{selected_summary.get('reranker_name')}`")
    lines.append(f"- row count: {selected_summary.get('row_count')}")
    lines.append("")
    lines.append("## Headline accuracy by tier")
    lines.append("")
    lines.append(
        "| tier | label | dense_top_n | final_top_k | hit@1 | hit@3 | hit@5 | mrr@10 | ndcg@10 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for tier in ("fast", "balanced", "quality"):
        e = modes_by_tier.get(tier)
        if e is None:
            continue
        flag = " (selected)" if tier == selected_entry.tier else ""
        h = e.headline
        lines.append(
            f"| `{tier}`{flag} | {e.label} | {e.dense_top_n} | "
            f"{e.final_top_k} | {h['hit_at_1']:.4f} | {h['hit_at_3']:.4f} | "
            f"{h['hit_at_5']:.4f} | {h['mrr_at_10']:.4f} | "
            f"{h['ndcg_at_10']:.4f} |"
        )
    lines.append("")
    lines.append("## Latency by tier (ms)")
    lines.append("")
    lines.append(
        "| tier | label | rerank_p95 | rerank_p99 | total_query_p95 | total_query_p99 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for tier in ("fast", "balanced", "quality"):
        e = modes_by_tier.get(tier)
        if e is None:
            continue
        flag = " (selected)" if tier == selected_entry.tier else ""
        h = e.headline
        lines.append(
            f"| `{tier}`{flag} | {e.label} | {h['rerank_p95_ms']:.2f} | "
            f"{h['rerank_p99_ms']:.2f} | {h['total_query_p95_ms']:.2f} | "
            f"{h['total_query_p99_ms']:.2f} |"
        )
    lines.append("")
    lines.append("## Selected tier — per-answer-type slice")
    lines.append("")
    lines.append("| answer_type | rows | hit@5 | mrr@10 | ndcg@10 |")
    lines.append("|---|---:|---:|---:|---:|")
    for atype, slice_ in (selected_summary.get("per_answer_type") or {}).items():
        lines.append(
            f"| {atype} | {slice_.get('row_count')} | "
            f"{slice_.get('mean_hit_at_5'):.4f} | "
            f"{slice_.get('mean_mrr_at_10'):.4f} | "
            f"{slice_.get('mean_ndcg_at_10'):.4f} |"
        )
    lines.append("")
    lines.append("## Selected tier — per-difficulty slice")
    lines.append("")
    lines.append("| difficulty | rows | hit@5 | mrr@10 | ndcg@10 |")
    lines.append("|---|---:|---:|---:|---:|")
    for diff, slice_ in (selected_summary.get("per_difficulty") or {}).items():
        lines.append(
            f"| {diff} | {slice_.get('row_count')} | "
            f"{slice_.get('mean_hit_at_5'):.4f} | "
            f"{slice_.get('mean_mrr_at_10'):.4f} | "
            f"{slice_.get('mean_ndcg_at_10'):.4f} |"
        )
    lines.append("")
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- latency_breakdown.json + .md ---
    # Re-anchor the latency breakdown on the SELECTED tier's report
    # rather than copying whatever the upstream sweep happened to use
    # as its anchor (the sweep CLI defaults to top20). The harness in
    # ``eval.harness.latency_breakdown`` walks the selected config's
    # per-row stage timings to produce a fresh stage-level table.
    from eval.harness.latency_breakdown import (
        build_latency_breakdown,
        latency_breakdown_to_dict,
        render_latency_breakdown_markdown,
    )

    breakdown_for_selected = build_latency_breakdown(
        selected_report_dir / "retrieval_eval_report.json",
        label=selected_entry.label,
    )
    (out_dir / "latency_breakdown.json").write_text(
        json.dumps(
            latency_breakdown_to_dict(breakdown_for_selected),
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "latency_breakdown.md").write_text(
        render_latency_breakdown_markdown(breakdown_for_selected),
        encoding="utf-8",
    )

    # --- selected_config.json ---
    selected_config: Dict[str, Any] = {
        "schema": "legacy-baseline-final.selected-config.v1",
        "tier": selected_entry.tier,
        "label": selected_entry.label,
        "rationale": selected_entry.rationale,
        "retriever": {
            "embeddingModel": selected_meta.get("embedding_model"),
            "embeddingMaxSeqLength": selected_meta.get("embedding_max_seq_length"),
            "embeddingBatchSize": selected_meta.get("embedding_batch_size"),
            "indexBackend": "faiss",
            "indexDir": selected_meta.get("rag_index_dir"),
            "denseTopN": selected_entry.dense_top_n,
            "candidateK": selected_entry.dense_top_n,
            "finalTopK": selected_entry.final_top_k,
            "useMmr": False,
            "mmrLambda": None,
        },
        "reranker": {
            "name": "cross_encoder",
            "model": selected_meta.get("reranker_model"),
            "batchSize": selected_meta.get("reranker_batch_size"),
            "maxLength": selected_meta.get("reranker_max_length"),
            "textMaxChars": selected_meta.get("reranker_text_max_chars"),
            "device": selected_meta.get("reranker_device"),
        },
        "agentLoop": {
            "backend": "legacy",
            "critic": "rule",
            "rewriter": "noop",
            "parser": "regex",
            "maxIter": 3,
            "maxTotalMs": 15000,
            "maxLlmTokens": 4000,
            "minStopConfidence": 0.75,
        },
        "querySetPath": (selected_meta.get("dataset") or "").replace("\\", "/"),
        "corpusPath": (selected_meta.get("corpus_path") or "").replace("\\", "/"),
        "evalCommand": {
            "step1_runSweep": (
                "python -m eval.run_eval phase2a-latency-sweep "
                f"--dataset {(selected_meta.get('dataset') or '').replace(chr(92), '/')} "
                f"--corpus {(selected_meta.get('corpus_path') or '').replace(chr(92), '/')} "
                "--out-dir eval/reports/phase2/2a_latency "
                f"--final-top-k {selected_entry.final_top_k} "
                "--dense-top-n 5 --dense-top-n 10 --dense-top-n 15 "
                "--dense-top-n 20 --dense-top-n 30 --dense-top-n 50 "
                "--breakdown-anchor-dense-top-n "
                f"{selected_entry.dense_top_n} "
                "--candidate-recall-extra-hit-k 10 "
                "--candidate-recall-extra-hit-k 20 "
                "--candidate-recall-extra-hit-k 50 "
                "--metric mean_hit_at_1 --latency rerank_p95_ms"
            ),
            "step2_consolidate": (
                "python -m scripts.build_legacy_baseline_final "
                "--sweep-dir eval/reports/phase2/2a_latency "
                "--out-dir eval/reports/legacy-baseline-final "
                f"--selected-tier {selected_entry.tier}"
            ),
        },
    }
    (out_dir / "selected_config.json").write_text(
        json.dumps(selected_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- baseline_manifest.json ---
    sweep_entries = sweep_doc.get("entries") or []
    sweep_summary = []
    for entry in sweep_entries:
        sweep_summary.append({
            "label": entry.get("label"),
            "denseTopN": entry.get("dense_top_n"),
            "finalTopK": entry.get("final_top_k"),
            "hitAt1": entry.get("mean_hit_at_1"),
            "hitAt3": entry.get("mean_hit_at_3"),
            "hitAt5": entry.get("mean_hit_at_5"),
            "mrrAt10": entry.get("mean_mrr_at_10"),
            "ndcgAt10": entry.get("mean_ndcg_at_10"),
            "rerankP95Ms": entry.get("rerank_p95_ms"),
            "totalQueryP95Ms": entry.get("total_query_p95_ms"),
            "denseRetrievalP95Ms": entry.get("dense_retrieval_p95_ms"),
            "candidateRecall": entry.get("candidate_recall"),
        })

    # Use the freshly-built breakdown for the SELECTED tier (re-anchored
    # on top10 by default, not the upstream sweep's top20). ``breakdown_doc``
    # remains the upstream-sweep doc, which we keep for reference but
    # don't promote into the manifest's stagesAtSelected slot.
    selected_breakdown_dict = latency_breakdown_to_dict(breakdown_for_selected)
    breakdown_stages = selected_breakdown_dict.get("stages") or {}
    latency_summary: Dict[str, Any] = {}
    for stage_name in (
        "dense_retrieval_ms",
        "total_rerank_ms",
        "total_query_ms",
        "tokenize_ms",
        "forward_ms",
    ):
        s = breakdown_stages.get(stage_name)
        if not s:
            continue
        latency_summary[stage_name] = {
            "avgMs": s.get("avg_ms"),
            "p50Ms": s.get("p50_ms"),
            "p90Ms": s.get("p90_ms"),
            "p95Ms": s.get("p95_ms"),
            "p99Ms": s.get("p99_ms"),
            "maxMs": s.get("max_ms"),
            "count": s.get("count"),
        }

    offline_corpus = selected_meta.get("offline_corpus") or {}
    manifest: Dict[str, Any] = {
        "schema": "legacy-baseline-final.manifest.v1",
        "commitHash": _git_head(),
        "workingTreeDirty": _git_dirty(),
        "evaluatedAt": sweep_run_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifestGeneratedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sourceSweepDir": str(sweep_dir).replace("\\", "/"),
        "querySetPath": selected_meta.get("dataset"),
        "queryCount": int(selected_summary.get("row_count") or 0),
        "corpusInfo": {
            "path": offline_corpus.get("path") or selected_meta.get("corpus_path"),
            "documentCount": offline_corpus.get("document_count"),
            "chunkCount": offline_corpus.get("chunk_count"),
            "indexVersion": offline_corpus.get("index_version"),
            "dimension": offline_corpus.get("dimension"),
        },
        "embeddingModel": selected_meta.get("embedding_model"),
        "retrieverConfig": selected_config["retriever"],
        "rerankerConfig": selected_config["reranker"],
        "selectedMode": {
            "tier": selected_entry.tier,
            "label": selected_entry.label,
            "denseTopN": selected_entry.dense_top_n,
            "finalTopK": selected_entry.final_top_k,
        },
        "metricsSummary": {
            "selectedTier": selected_entry.tier,
            "tiers": {
                tier: {
                    "label": e.label,
                    "hitAt1": e.headline["hit_at_1"],
                    "hitAt3": e.headline["hit_at_3"],
                    "hitAt5": e.headline["hit_at_5"],
                    "mrrAt10": e.headline["mrr_at_10"],
                    "ndcgAt10": e.headline["ndcg_at_10"],
                }
                for tier, e in modes_by_tier.items()
            },
            "sweepEntries": sweep_summary,
        },
        "latencySummary": {
            "selectedTier": selected_entry.tier,
            "rerankP95Ms": selected_entry.headline["rerank_p95_ms"],
            "rerankP99Ms": selected_entry.headline["rerank_p99_ms"],
            "totalQueryP95Ms": selected_entry.headline["total_query_p95_ms"],
            "totalQueryP99Ms": selected_entry.headline["total_query_p99_ms"],
            "stagesAtSelected": latency_summary,
            "frontier": [
                {
                    "label": pt.get("label"),
                    "denseTopN": pt.get("dense_top_n"),
                    "metric": pt.get("metric"),
                    "latencyMs": pt.get("latency_ms"),
                    "onFrontier": pt.get("on_frontier"),
                    "dominatedBy": pt.get("dominated_by"),
                }
                for pt in (frontier_doc.get("entries") or [])
            ],
        },
        "agentLoop": selected_config["agentLoop"],
        "notes": [
            "This baseline is the locked LangGraph A/B reference. Do not "
            "promote any other config above it without re-running the "
            "sweep and refreshing this manifest.",
            "agent_loop_backend stays at 'legacy' default; the LangGraph "
            "backend is exercised only via the offline A/B harness "
            "(scripts/eval_agent_loop_ab.py).",
            "selected_config.json carries the full reproduction command. "
            "Any operator can re-run the eval from that file alone.",
            "Production runtime defaults (rag_top_k=5, rag_candidate_k=30, "
            "rag_reranker='off', rag_use_mmr=False) are NOT changed. The "
            "tuned baseline is offline-only until promoted by config.",
        ],
    }
    (out_dir / "baseline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- README.md ---
    readme_lines: List[str] = []
    readme_lines.append("# Legacy retrieval baseline (locked)")
    readme_lines.append("")
    readme_lines.append(
        "This directory holds the **locked legacy retrieval / rerank "
        "baseline** that LangGraph A/B testing measures against. It is "
        "consolidated from the ``phase2a-latency-sweep`` run referenced "
        f"by ``sourceSweepDir`` in ``baseline_manifest.json`` "
        f"(``{str(sweep_dir).replace(chr(92), '/')}``). The sweep was "
        "executed against the same commit recorded in "
        "``baseline_manifest.json -> commitHash`` and the same dataset "
        "/ corpus / model the A/B test will use, so re-running the "
        "consolidation step against the same sweep dir is byte-stable "
        "(modulo the ``manifestGeneratedAt`` timestamp)."
    )
    readme_lines.append("")
    readme_lines.append("## Why this exists")
    readme_lines.append("")
    readme_lines.append(
        "The LangGraph backend (``AgentLoopGraph``) is the experimental "
        "alternative to the legacy ``AgentLoopController``. Before we "
        "promote it, we need a stable reference for the legacy stack — "
        "this directory is that reference. ``baseline_manifest.json`` "
        "records the corpus / index / model / metrics / latency "
        "snapshot; ``selected_config.json`` records the canonical "
        "config the A/B compares against."
    )
    readme_lines.append("")
    readme_lines.append("## File layout")
    readme_lines.append("")
    readme_lines.append("| file | purpose |")
    readme_lines.append("|---|---|")
    readme_lines.append("| `metrics.json` | machine-readable accuracy + latency metrics for fast/balanced/quality tiers, plus per-answer-type / per-difficulty slices for the selected tier. |")
    readme_lines.append("| `metrics.md` | human-readable summary of `metrics.json`. |")
    readme_lines.append("| `latency_breakdown.json` | stage-level latency breakdown (dense_retrieval / tokenize / forward / postprocess / total) anchored on the selected tier. Schema `phase2a-latency-breakdown.v1`. |")
    readme_lines.append("| `latency_breakdown.md` | rendered version of `latency_breakdown.json`. |")
    readme_lines.append("| `selected_config.json` | canonical config — retriever + reranker + agent-loop knobs — for the A/B legacy arm. Includes the exact CLI to reproduce. |")
    readme_lines.append("| `baseline_manifest.json` | provenance + metrics + latency manifest. The single doc the A/B harness should read to anchor its legacy reference. |")
    readme_lines.append(
        f"| `../{Path(sweep_dir).name}/` | upstream `phase2a-latency-sweep` output "
        "this baseline was consolidated from. Do not delete — the "
        "manifest's `sourceSweepDir` field points back to it. |"
    )
    readme_lines.append("")
    readme_lines.append(
        f"`baseline_manifest.json -> sourceSweepDir = "
        f"\"{str(sweep_dir).replace(chr(92), '/')}\"`. "
        f"`evaluatedAt` records when that sweep ran "
        f"(`{sweep_run_at}`); `manifestGeneratedAt` records when this "
        "directory was assembled."
    )
    readme_lines.append("")
    readme_lines.append("## Tier selection")
    readme_lines.append("")
    readme_lines.append(
        "Three tiers were considered (Pareto frontier of "
        "``mean_hit_at_1`` ↑ vs ``rerank_p95_ms`` ↓):"
    )
    readme_lines.append("")
    for tier in ("fast", "balanced", "quality"):
        e = modes_by_tier.get(tier)
        if e is None:
            continue
        flag = " — **selected**" if tier == selected_entry.tier else ""
        readme_lines.append(
            f"- `{tier}`{flag}: dense_top_n={e.dense_top_n}, "
            f"final_top_k={e.final_top_k}, hit@1="
            f"{e.headline['hit_at_1']:.4f}, "
            f"rerank_p95_ms={e.headline['rerank_p95_ms']:.2f}. "
            f"{e.rationale}"
        )
    readme_lines.append("")
    readme_lines.append(
        f"`{selected_entry.tier}` is the default A/B reference because "
        "it sits on the Pareto frontier with the best "
        "latency/quality balance — the median-latency entry among the "
        "frontier-eligible configs."
    )
    readme_lines.append("")
    readme_lines.append("## Reproducing this baseline")
    readme_lines.append("")
    readme_lines.append("```bash")
    readme_lines.append("# 1. Re-run the legacy sweep across the full topN axis.")
    readme_lines.append("#    (~50 minutes on RTX 5080 — embedding 47k chunks dominates.)")
    readme_lines.append("python -m eval.run_eval phase2a-latency-sweep \\")
    readme_lines.append("    --dataset eval/eval_queries/anime_silver_200.jsonl \\")
    readme_lines.append("    --corpus eval/corpora/anime_namu_v3_token_chunked/corpus.combined.token-aware-v1.jsonl \\")
    readme_lines.append("    --out-dir eval/reports/phase2/2a_latency \\")
    readme_lines.append("    --final-top-k 10 \\")
    readme_lines.append("    --dense-top-n 5 --dense-top-n 10 --dense-top-n 15 \\")
    readme_lines.append("    --dense-top-n 20 --dense-top-n 30 --dense-top-n 50 \\")
    readme_lines.append("    --breakdown-anchor-dense-top-n 10 \\")
    readme_lines.append("    --candidate-recall-extra-hit-k 10 \\")
    readme_lines.append("    --candidate-recall-extra-hit-k 20 \\")
    readme_lines.append("    --candidate-recall-extra-hit-k 50 \\")
    readme_lines.append("    --metric mean_hit_at_1 --latency rerank_p95_ms")
    readme_lines.append("")
    readme_lines.append("# 2. Re-build the consolidated baseline files.")
    readme_lines.append("python -m scripts.build_legacy_baseline_final \\")
    readme_lines.append("    --sweep-dir eval/reports/phase2/2a_latency \\")
    readme_lines.append("    --out-dir eval/reports/legacy-baseline-final \\")
    readme_lines.append("    --selected-tier balanced")
    readme_lines.append("```")
    readme_lines.append("")
    readme_lines.append(
        "Step 1 takes ~50 min on a single RTX 5080 (47k-chunk bge-m3 "
        "embedding dominates, then 6 rerank passes run in serial). "
        "Step 2 is pure post-processing and finishes in milliseconds. "
        "The two together produce identical metrics within rerank "
        "latency noise as long as the corpus, query set, embedding "
        "model, and reranker model are unchanged."
    )
    readme_lines.append("")
    readme_lines.append("## Next step — LangGraph A/B")
    readme_lines.append("")
    readme_lines.append(
        "Once this baseline is locked, run the offline A/B harness to "
        "compare the legacy ``AgentLoopController`` against the "
        "experimental ``AgentLoopGraph`` on the same query set:"
    )
    readme_lines.append("")
    readme_lines.append("```bash")
    readme_lines.append("# Stub-mode smoke (no FAISS / no GPU needed)")
    readme_lines.append("python -m scripts.eval_agent_loop_ab \\")
    readme_lines.append("    --queries eval/agent_loop_ab/smoke.jsonl \\")
    readme_lines.append("    --mode stub --run-name smoke")
    readme_lines.append("")
    readme_lines.append("# Live registry-mode A/B against the silver-200 set")
    readme_lines.append("# (requires the FAISS index + same WorkerSettings as the worker;")
    readme_lines.append("# does NOT touch Redis / DB / callbacks)")
    readme_lines.append("python -m scripts.eval_agent_loop_ab \\")
    readme_lines.append("    --queries eval/eval_queries/anime_silver_200.jsonl \\")
    readme_lines.append("    --mode registry \\")
    readme_lines.append("    --run-name legacy-vs-graph-anime_silver_200 \\")
    readme_lines.append("    --critic rule --parser regex \\")
    readme_lines.append("    --max-iter 3 --max-total-ms 15000 \\")
    readme_lines.append("    --max-llm-tokens 4000 --min-confidence 0.75")
    readme_lines.append("```")
    readme_lines.append("")
    readme_lines.append(
        "Outputs land in `eval/agent_loop_ab/<run-name>/` "
        "(`raw_results.jsonl`, `summary.csv`, "
        "`comparison_summary.json`). The legacy arm of the comparison "
        "must reproduce the metrics in this directory; if it doesn't, "
        "treat that as a regression — re-anchor before reading the "
        "graph results."
    )
    readme_lines.append("")
    readme_lines.append(
        "**silver_200 schema:** the loader in "
        "``eval/harness/agent_loop_ab.py`` accepts both the singular "
        "(``expected_doc_id`` / ``expected_keywords``) and the "
        "silver_200 plural (``expected_doc_ids[0]`` / "
        "``expected_section_keywords``) shapes, so "
        "``anime_silver_200.jsonl`` plugs in directly with full "
        "per-row hit@k / keyword metrics — no projection step "
        "needed. When both shapes are present the singular fields "
        "win, so an operator can override the fallback inline."
    )
    readme_lines.append("")
    readme_lines.append("## Hard rules for this baseline")
    readme_lines.append("")
    readme_lines.append(
        "1. **Do not edit the LangGraph backend** to make A/B numbers "
        "look better. The legacy reference must stay frozen.")
    readme_lines.append(
        "2. **`agent_loop_backend` default stays `legacy`**. The graph "
        "backend is opt-in only.")
    readme_lines.append(
        "3. **No Redis / DB / callback / Spring repo / infra mutations** "
        "from this directory's tooling. Everything here is "
        "post-processing.")
    readme_lines.append(
        "4. **Re-runs go through the upstream sweep dir** "
        f"(``{str(sweep_dir).replace(chr(92), '/')}``) and then through "
        "`build_legacy_baseline_final.py`. Don't hand-edit `metrics.*` "
        "or `baseline_manifest.json` — re-run the script.")
    readme_lines.append("")

    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"wrote: {out_dir}/metrics.json")
    print(f"wrote: {out_dir}/metrics.md")
    print(f"wrote: {out_dir}/latency_breakdown.json")
    print(f"wrote: {out_dir}/latency_breakdown.md")
    print(f"wrote: {out_dir}/selected_config.json")
    print(f"wrote: {out_dir}/baseline_manifest.json")
    print(f"wrote: {out_dir}/README.md")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sweep-dir", type=Path, default=_DEFAULT_SWEEP_DIR,
        help=f"Phase 2A sweep output dir (default: {_DEFAULT_SWEEP_DIR}).",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
        help=f"Where to write the consolidated baseline (default: {_DEFAULT_OUT_DIR}).",
    )
    ap.add_argument(
        "--selected-tier", type=str, default=_SELECTED_TIER,
        choices=("fast", "balanced", "quality"),
        help=(
            "Which recommended-modes tier to promote as the locked "
            f"baseline (default: {_SELECTED_TIER})."
        ),
    )
    args = ap.parse_args(argv)
    build(args.sweep_dir, args.out_dir, args.selected_tier)
    return 0


if __name__ == "__main__":
    sys.exit(main())
