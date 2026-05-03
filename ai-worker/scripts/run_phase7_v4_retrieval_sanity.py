"""Tiny Phase 7 v4 active-config retrieval sanity run.

This wrapper deliberately does not build indexes, run Optuna, sweep MMR, or
score answerability. It loads the already-built v4 retrieval-title-section
cache, runs a small sample from the active config dataset, and verifies that
returned chunk ids join against canonical rag_chunks.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
from eval.harness.embedding_text_reindex import load_variant_dense_stack


log = logging.getLogger("scripts.run_phase7_v4_retrieval_sanity")

FORBIDDEN_ACTIVE_STRINGS = (
    "v3",
    "anime_namu_v3",
    "rag-cheap-sweep-v3",
    "bge-m3-anime-namu-v3",
)
DEFAULT_INDEX_CACHE = "namu-v4-2008-2026-04-retrieval-title-section-mseq512"
DEFAULT_REPORT_DIR = Path("eval/reports/phase7/7.8_retrieval_sanity")


@dataclass(frozen=True)
class QuerySample:
    query_id: str
    query: str
    expected_page_id: str
    expected_title: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 7 v4 active config retrieval sanity run."
    )
    p.add_argument(
        "--active-yaml",
        type=Path,
        default=Path("eval/experiments/active.yaml"),
        help="Active config to validate and sample from.",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for sanity artifacts.",
    )
    p.add_argument(
        "--index-root",
        type=Path,
        default=Path("eval/indexes"),
        help="Existing v4 FAISS cache root. Indexes are loaded, not built.",
    )
    p.add_argument(
        "--index-cache",
        default=DEFAULT_INDEX_CACHE,
        help="Existing v4 cache directory name under --index-root.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of active dataset queries to run. Must be between 1 and 20.",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--embedding-model", default="BAAI/bge-m3")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_active_config(path: Path) -> tuple[dict[str, Any], str]:
    text = _read_text(path)
    lowered = text.lower()
    hits = [token for token in FORBIDDEN_ACTIVE_STRINGS if token in lowered]
    if hits:
        raise SystemExit(
            "active.yaml contains a forbidden legacy string; aborting sanity run."
        )
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise SystemExit("active.yaml must be a mapping.")
    return payload, text


def _resolve_ai_worker_path(relative_or_abs: str) -> Path:
    path = Path(relative_or_abs)
    return path if path.is_absolute() else Path.cwd() / path


def _validate_active_config(payload: Mapping[str, Any]) -> dict[str, Path]:
    experiment_id = str(payload.get("experiment_id") or "")
    if not experiment_id.startswith("phase7-v4-"):
        raise SystemExit(f"unexpected experiment_id for v4 sanity: {experiment_id!r}")

    meta = payload.get("_meta") or {}
    if not isinstance(meta, dict):
        raise SystemExit("_meta must be a mapping.")
    if meta.get("fail_closed") is True:
        raise SystemExit("active config is fail-closed; aborting sanity run.")
    if str(meta.get("status") or "").lower() in {
        "fail_closed",
        "fail-closed",
        "disabled",
    }:
        raise SystemExit("active config status blocks execution.")

    policy = meta.get("execution_policy") or {}
    if any(bool(value) for value in policy.values()):
        raise SystemExit("active config execution_policy allows a broad run.")

    objective = payload.get("objective") or {}
    if objective.get("mode") != "rag":
        raise SystemExit("Phase 7 v4 retrieval sanity requires objective.mode=rag.")
    dataset = _resolve_ai_worker_path(str(objective.get("dataset") or ""))
    if not dataset.exists():
        raise SystemExit(f"active dataset is missing: {dataset}")

    artifacts = meta.get("canonical_v4_artifacts") or {}
    audit = meta.get("answerability_audit") or {}
    rag_chunks = _resolve_ai_worker_path(str(audit.get("production_join_chunks") or ""))
    if rag_chunks.name != "rag_chunks.jsonl":
        raise SystemExit("production join chunks must be rag_chunks.jsonl.")
    forbidden_join = _resolve_ai_worker_path(
        str(audit.get("forbidden_production_join_chunks") or "")
    )
    if rag_chunks == forbidden_join or rag_chunks.name == "chunks_v4.jsonl":
        raise SystemExit("chunks_v4.jsonl must not be used as the join source.")
    if str(artifacts.get("rag_chunks") or "") != str(
        audit.get("production_join_chunks") or ""
    ):
        raise SystemExit("canonical rag_chunks and answerability join path diverge.")

    pages_v4 = _resolve_ai_worker_path(str(artifacts.get("pages_v4") or ""))
    if not pages_v4.exists():
        raise SystemExit(f"pages_v4 artifact is missing: {pages_v4}")
    if not rag_chunks.exists():
        raise SystemExit(f"rag_chunks artifact is missing: {rag_chunks}")

    return {
        "dataset": dataset,
        "rag_chunks": rag_chunks,
        "pages_v4": pages_v4,
    }


def _load_query_samples(dataset: Path, limit: int) -> list[QuerySample]:
    if limit < 1 or limit > 20:
        raise SystemExit("--limit must be between 1 and 20 for sanity runs.")

    rows: list[QuerySample] = []
    with dataset.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("expected_not_in_corpus") is True:
                continue
            qid = str(rec.get("query_id") or rec.get("id") or "")
            query = str(rec.get("query") or "")
            expected_page_id = str(
                rec.get("silver_expected_page_id")
                or rec.get("expected_page_id")
                or rec.get("page_id")
                or ""
            )
            expected_title = str(
                rec.get("silver_expected_title")
                or rec.get("expected_title")
                or ""
            )
            if not qid or not query:
                continue
            rows.append(QuerySample(qid, query, expected_page_id, expected_title))
            if len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"no runnable query samples found in {dataset}")
    return rows


def _scan_rag_chunks(
    rag_chunks: Path, chunk_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    wanted = {str(x) for x in chunk_ids if str(x)}
    found: dict[str, dict[str, Any]] = {}
    if not wanted:
        return found
    with rag_chunks.open("r", encoding="utf-8") as fp:
        for line in fp:
            if len(found) == len(wanted):
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_id = str(rec.get("chunk_id") or "")
            if chunk_id in wanted:
                found[chunk_id] = rec
    return found


def _scan_pages(pages_v4: Path, page_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    wanted = {str(x) for x in page_ids if str(x)}
    found: dict[str, dict[str, Any]] = {}
    if not wanted:
        return found
    with pages_v4.open("r", encoding="utf-8") as fp:
        for line in fp:
            if len(found) == len(wanted):
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            page_id = str(rec.get("page_id") or "")
            if page_id in wanted:
                found[page_id] = rec
    return found


def _contains_forbidden_text(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(token in text for token in FORBIDDEN_ACTIVE_STRINGS)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    active, _ = _load_active_config(args.active_yaml)
    paths = _validate_active_config(active)
    queries = _load_query_samples(paths["dataset"], int(args.limit))

    cache_dir = Path(args.index_root) / str(args.index_cache)
    required_cache_files = ["faiss.index", "build.json", "chunks.jsonl"]
    missing = [name for name in required_cache_files if not (cache_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"existing index cache is incomplete: {cache_dir}; missing={missing}"
        )

    log.info("loading existing v4 cache: %s", cache_dir)
    embedder = SentenceTransformerEmbedder(
        model_name=str(args.embedding_model),
        max_seq_length=int(args.max_seq_length),
    )
    retriever, info, manifest = load_variant_dense_stack(
        cache_dir,
        embedder=embedder,
        top_k=int(args.top_k),
    )

    rows: list[dict[str, Any]] = []
    all_chunk_ids: set[str] = set()
    all_doc_ids: set[str] = set()
    reciprocal_ranks: list[float] = []
    hit_count = 0
    scored_count = 0

    for sample in queries:
        started = time.perf_counter()
        report = retriever.retrieve(sample.query)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        docs = []
        expected_rank: Optional[int] = None
        for rank, chunk in enumerate(report.results[: int(args.top_k)], start=1):
            all_chunk_ids.add(chunk.chunk_id)
            all_doc_ids.add(chunk.doc_id)
            if sample.expected_page_id and chunk.doc_id == sample.expected_page_id:
                expected_rank = expected_rank or rank
            docs.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "score": float(chunk.score) if chunk.score is not None else None,
                }
            )
        if sample.expected_page_id:
            scored_count += 1
            if expected_rank is not None:
                hit_count += 1
                reciprocal_ranks.append(1.0 / float(expected_rank))
            else:
                reciprocal_ranks.append(0.0)
        rows.append(
            {
                "query_id": sample.query_id,
                "query": sample.query,
                "expected_page_id": sample.expected_page_id,
                "expected_title": sample.expected_title,
                "elapsed_ms": elapsed_ms,
                "docs": docs,
            }
        )

    rag_lookup = _scan_rag_chunks(paths["rag_chunks"], all_chunk_ids)
    page_lookup = _scan_pages(paths["pages_v4"], all_doc_ids)

    missing_chunks: list[str] = []
    missing_pages: list[str] = []
    mismatched_chunk_docs: list[str] = []
    for row in rows:
        for doc in row["docs"]:
            chunk_id = doc["chunk_id"]
            doc_id = doc["doc_id"]
            chunk_rec = rag_lookup.get(chunk_id)
            if chunk_rec is None:
                missing_chunks.append(chunk_id)
                doc["rag_chunks_joined"] = False
                continue
            doc["rag_chunks_joined"] = True
            doc["rag_chunks_doc_id"] = str(chunk_rec.get("doc_id") or "")
            doc["title"] = str(chunk_rec.get("title") or "")
            section_path = chunk_rec.get("section_path") or []
            doc["section_path"] = (
                section_path if isinstance(section_path, list) else [section_path]
            )
            if doc["rag_chunks_doc_id"] != doc_id:
                mismatched_chunk_docs.append(chunk_id)
            if doc_id not in page_lookup:
                missing_pages.append(doc_id)
            doc["pages_v4_joined"] = doc_id in page_lookup
            if doc_id in page_lookup:
                page = page_lookup[doc_id]
                doc["page_title"] = str(page.get("page_title") or "")
                doc["retrieval_title"] = str(page.get("retrieval_title") or "")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = report_dir / "retrieval_results.jsonl"
    with retrieval_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    hit_at_k = (hit_count / scored_count) if scored_count else None
    mrr = (sum(reciprocal_ranks) / scored_count) if scored_count else None
    manifest_payload = {
        "experiment_id": active["experiment_id"],
        "sanity_only": True,
        "query_count": len(rows),
        "top_k": int(args.top_k),
        "dataset": str(paths["dataset"]),
        "join_source_kind": "rag_chunks",
        "rag_chunks_path": str(paths["rag_chunks"]),
        "pages_v4_path": str(paths["pages_v4"]),
        "index_cache_dir": str(cache_dir),
        "index_version": info.index_version,
        "embedding_model": info.embedding_model,
        "index_chunk_count": info.chunk_count,
        "index_document_count": info.document_count,
        "variant": manifest.variant if manifest else "unknown",
        "artifacts": {
            "retrieval_results_jsonl": str(retrieval_path),
            "manifest_json": str(report_dir / "manifest.json"),
        },
        "namespace_validation": {
            "retrieved_chunk_count": len(all_chunk_ids),
            "retrieved_doc_count": len(all_doc_ids),
            "missing_rag_chunks_chunk_ids": sorted(set(missing_chunks)),
            "missing_pages_v4_doc_ids": sorted(set(missing_pages)),
            "mismatched_rag_chunk_doc_ids": sorted(set(mismatched_chunk_docs)),
            "all_chunk_ids_joined_rag_chunks": not missing_chunks,
            "all_doc_ids_joined_pages_v4": not missing_pages,
            "chunks_v4_used_as_join_source": False,
        },
        "metrics_reference_only": {
            "scored_query_count": scored_count,
            "hit_at_k": hit_at_k,
            "mrr": mrr,
            "mean_elapsed_ms": (
                sum(float(row["elapsed_ms"]) for row in rows) / len(rows)
                if rows else None
            ),
        },
    }
    manifest_path = report_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    contaminated = any(
        _contains_forbidden_text(path) for path in (retrieval_path, manifest_path)
    )
    if contaminated:
        raise SystemExit("sanity artifacts contain forbidden legacy strings.")
    if missing_chunks or missing_pages or mismatched_chunk_docs:
        raise SystemExit(
            "retrieval results failed v4 namespace validation; see manifest.json."
        )

    log.info("wrote retrieval sanity results: %s", retrieval_path)
    log.info("wrote retrieval sanity manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
