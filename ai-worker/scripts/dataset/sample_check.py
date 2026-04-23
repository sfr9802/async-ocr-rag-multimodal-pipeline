"""Render a sample-check HTML review report for a validated eval dataset.

For each sampled query, emits:
    query / question_type / difficulty / expected_doc_ids / expected_keywords
    top-3 retrieved chunks (full text, with score + doc_id + section)
    annotation form: radio (good / bad / fix) + free-text note

The HTML exports all annotations as a JSON blob the reviewer can download;
feeding that JSON back via ``--merge`` updates the source JSONL rows with
``human_reviewed`` + ``human_note`` fields.

Usage
-----

Render::

    python -m scripts.dataset.sample_check \\
        --dataset eval/datasets/rag_anime_kr.jsonl \\
        --corpus  fixtures/anime_corpus_kr.jsonl \\
        --sample-rate 0.1 \\
        --out     review/anime_sample.html

Merge annotations back::

    python -m scripts.dataset.sample_check \\
        --dataset eval/datasets/rag_anime_kr.jsonl \\
        --merge   review/anime_sample.annotations.json
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.dataset._common import (
    configure_logging,
    read_jsonl,
    stable_seed,
)

log = logging.getLogger("scripts.dataset.sample_check")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_rows(
    rows: List[Dict[str, Any]],
    *,
    sample_rate: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if sample_rate >= 1.0:
        return list(rows)
    k = max(1, int(round(len(rows) * sample_rate)))
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(rows)), k=min(k, len(rows))))
    return [rows[i] for i in idxs]


# ---------------------------------------------------------------------------
# Retrieval for top-3 chunks
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    doc_id: str
    section: str
    text: str
    score: float


def _retrieve_top_n(retriever, query: str, n: int) -> List[RetrievedChunk]:
    report = retriever.retrieve(query)
    out: List[RetrievedChunk] = []
    for r in report.results[:n]:
        # RetrievedChunk in app.capabilities.rag.generation has .doc_id,
        # .section, .text, .score
        out.append(RetrievedChunk(
            doc_id=getattr(r, "doc_id", ""),
            section=getattr(r, "section", "") or "",
            text=getattr(r, "text", "") or "",
            score=float(getattr(r, "score", 0.0)),
        ))
    return out


def _build_retriever(corpus_path: Path, top_k: int = 3):
    """Lazy-import so the merge path doesn't load ML deps."""
    import tempfile
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
    from app.core.config import get_settings
    from eval.harness.offline_corpus import build_offline_rag_stack

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
    )
    # Stash a long-lived temp dir inside the caller's process; we only
    # need the FAISS files while this script runs.
    tmp = Path(tempfile.mkdtemp(prefix="sample_check_"))
    retriever, _, _ = build_offline_rag_stack(
        corpus_path,
        embedder=embedder,
        index_dir=tmp,
        top_k=top_k,
    )
    return retriever


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
       max-width: 960px; margin: 24px auto; padding: 0 16px; color: #1f2328; }
h1 { font-size: 20px; border-bottom: 1px solid #d0d7de; padding-bottom: 8px; }
.row { border: 1px solid #d0d7de; border-radius: 6px; padding: 16px;
       margin-bottom: 20px; background: #ffffff; }
.header { display: flex; justify-content: space-between; flex-wrap: wrap;
          gap: 8px; margin-bottom: 8px; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 10px;
        font-size: 12px; background: #eaeef2; color: #24292f; }
.pill.diff-easy { background: #dafbe1; color: #1a7f37; }
.pill.diff-medium { background: #fff8c5; color: #9a6700; }
.pill.diff-hard { background: #ffebe9; color: #cf222e; }
.pill.diff-impossible { background: #eaeef2; color: #6e7781; }
.query { font-size: 16px; font-weight: 600; margin: 8px 0; }
.meta { font-size: 13px; color: #57606a; }
.chunks { margin-top: 12px; }
.chunk { background: #f6f8fa; border: 1px solid #eaeef2; border-radius: 4px;
         padding: 10px; margin-bottom: 8px; font-size: 13px; }
.chunk .chead { color: #57606a; font-size: 12px; margin-bottom: 4px;
               font-family: ui-monospace, SFMono-Regular, monospace; }
.chunk.expected { border-color: #1a7f37; background: #dafbe120; }
.form { margin-top: 12px; padding: 10px; background: #fafbfc; border-radius: 4px; }
.form label { margin-right: 12px; cursor: pointer; }
.form textarea { width: 100%; min-height: 40px; margin-top: 6px;
                 font-family: inherit; font-size: 13px; padding: 6px;
                 border: 1px solid #d0d7de; border-radius: 4px; }
.toolbar { position: sticky; top: 0; background: #ffffff; padding: 10px 0;
           border-bottom: 1px solid #d0d7de; margin-bottom: 16px; z-index: 10; }
.toolbar button { font-size: 14px; padding: 6px 12px; cursor: pointer;
                  background: #1a7f37; color: #ffffff; border: none;
                  border-radius: 4px; margin-right: 8px; }
.toolbar button.secondary { background: #eaeef2; color: #24292f; }
"""


_EXPORT_JS = """
function exportAnnotations() {
  const rows = document.querySelectorAll('.row');
  const out = [];
  rows.forEach(row => {
    const key = row.dataset.key;
    const verdict = (row.querySelector('input[type=radio]:checked') || {}).value || null;
    const note = (row.querySelector('textarea').value || '').trim();
    if (verdict || note) {
      out.push({ key: key, verdict: verdict, note: note });
    }
  });
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'anime_sample.annotations.json';
  a.click();
  URL.revokeObjectURL(a.href);
}
function loadAnnotations(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const arr = JSON.parse(reader.result);
      arr.forEach(entry => {
        const row = document.querySelector('.row[data-key="' + entry.key + '"]');
        if (!row) return;
        if (entry.verdict) {
          const radio = row.querySelector('input[value="' + entry.verdict + '"]');
          if (radio) radio.checked = true;
        }
        if (entry.note) {
          row.querySelector('textarea').value = entry.note;
        }
      });
      alert('Loaded ' + arr.length + ' annotations.');
    } catch (e) {
      alert('Failed to parse: ' + e.message);
    }
  };
  reader.readAsText(file);
}
"""


def _annotation_key(row: Dict[str, Any]) -> str:
    """Stable key for matching HTML form rows back to JSONL rows."""
    return f"{row.get('source_title','')}|{row.get('query','')[:60]}"


def _render_html(
    *,
    rows: List[Dict[str, Any]],
    top3_by_key: Dict[str, List[RetrievedChunk]],
    corpus_path: Path,
    dataset_path: Path,
) -> str:
    parts: List[str] = []
    parts.append("<!DOCTYPE html><html lang='ko'><head>")
    parts.append("<meta charset='utf-8'><title>Anime RAG sample check</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append(f"<script>{_EXPORT_JS}</script>")
    parts.append("</head><body>")
    parts.append("<h1>Anime RAG sample check</h1>")
    parts.append(
        f"<div class='meta'>Dataset: {html.escape(str(dataset_path))} "
        f"| Corpus: {html.escape(str(corpus_path))} "
        f"| Rows: {len(rows)}</div>"
    )
    parts.append("<div class='toolbar'>")
    parts.append("<button onclick='exportAnnotations()'>Download annotations JSON</button>")
    parts.append("<label class='secondary' style='margin-left:8px'>")
    parts.append("<input type='file' accept='application/json' onchange='loadAnnotations(event)' "
                 "style='display:none' id='loadfile'>")
    parts.append("<button class='secondary' type='button' "
                 "onclick='document.getElementById(\"loadfile\").click()'>"
                 "Load existing annotations</button>")
    parts.append("</label></div>")

    for row in rows:
        key = _annotation_key(row)
        difficulty = row.get("difficulty", "") or ""
        expected_doc_ids = row.get("expected_doc_ids", []) or []
        expected_set = {str(d) for d in expected_doc_ids}
        expected_keywords = row.get("expected_keywords", []) or []

        parts.append(f"<div class='row' data-key='{html.escape(key)}'>")
        parts.append("<div class='header'>")
        parts.append(
            f"<span class='pill'>{html.escape(str(row.get('question_type','-')))}</span>"
        )
        parts.append(
            f"<span class='pill diff-{html.escape(difficulty)}'>difficulty: "
            f"{html.escape(difficulty)}</span>"
        )
        if row.get("top1_score") is not None:
            parts.append(
                f"<span class='pill'>top1={row['top1_score']:.2f}</span>"
            )
        if row.get("rank_of_expected") is not None:
            parts.append(
                f"<span class='pill'>rank={row['rank_of_expected']}</span>"
            )
        parts.append("</div>")

        parts.append(f"<div class='query'>{html.escape(str(row.get('query','')))}</div>")
        parts.append("<div class='meta'>")
        parts.append(
            "expected_doc_ids: " +
            ", ".join(html.escape(d) for d in expected_doc_ids)
        )
        if expected_keywords:
            parts.append(
                " &nbsp;|&nbsp; keywords: " +
                ", ".join(html.escape(k) for k in expected_keywords)
            )
        parts.append(
            f" &nbsp;|&nbsp; source_title: {html.escape(str(row.get('source_title','')))}"
        )
        parts.append("</div>")

        parts.append("<div class='chunks'>")
        chunks = top3_by_key.get(key, [])
        if not chunks:
            parts.append("<div class='meta'><em>No retrieved chunks.</em></div>")
        for i, ch in enumerate(chunks, 1):
            cls = "chunk expected" if ch.doc_id in expected_set else "chunk"
            parts.append(f"<div class='{cls}'>")
            parts.append(
                f"<div class='chead'>#{i} &nbsp; doc_id={html.escape(ch.doc_id)} "
                f"&nbsp; section={html.escape(ch.section)} "
                f"&nbsp; score={ch.score:.3f}</div>"
            )
            parts.append(f"<div>{html.escape(ch.text)}</div>")
            parts.append("</div>")
        parts.append("</div>")

        parts.append("<div class='form'>")
        parts.append(f"<label><input type='radio' name='verdict-{html.escape(key)}' "
                     f"value='good'>good</label>")
        parts.append(f"<label><input type='radio' name='verdict-{html.escape(key)}' "
                     f"value='bad'>bad</label>")
        parts.append(f"<label><input type='radio' name='verdict-{html.escape(key)}' "
                     f"value='fix'>fix</label>")
        parts.append("<textarea placeholder='note (optional)'></textarea>")
        parts.append("</div>")

        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Merge mode
# ---------------------------------------------------------------------------


def _merge_annotations(
    dataset_path: Path,
    annotations_path: Path,
) -> int:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        log.error("Expected a JSON array in %s", annotations_path)
        return 2
    by_key: Dict[str, Dict[str, Any]] = {}
    for entry in annotations:
        if isinstance(entry, dict) and "key" in entry:
            by_key[entry["key"]] = entry
    if not by_key:
        log.warning("No annotation entries found in %s", annotations_path)
        return 0

    rows = read_jsonl(dataset_path)
    updated = 0
    for row in rows:
        key = _annotation_key(row)
        ann = by_key.get(key)
        if not ann:
            continue
        row["human_reviewed"] = True
        verdict = ann.get("verdict")
        note = ann.get("note") or ""
        if verdict:
            row["human_verdict"] = str(verdict)
        if note:
            row["human_note"] = str(note)
        updated += 1

    with dataset_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")
    log.info("Merged %d annotations into %s", updated, dataset_path)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Validated eval JSONL (rag_anime_kr.jsonl).")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="Corpus JSONL. Required for render mode.")
    parser.add_argument("--sample-rate", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path (default: review/anime_sample.html).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge", type=Path, default=None,
                        help="Merge mode: JSON of annotations to merge back into --dataset.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)

    if args.merge is not None:
        return _merge_annotations(args.dataset, args.merge)

    if args.corpus is None:
        parser.error("--corpus is required in render mode.")

    out_path = args.out or Path("review/anime_sample.html")
    rows = read_jsonl(args.dataset)
    if not rows:
        log.error("Dataset is empty: %s", args.dataset)
        return 2
    sampled = _sample_rows(rows, sample_rate=args.sample_rate, seed=args.seed)
    log.info("Sampled %d / %d rows (rate=%.2f)", len(sampled), len(rows), args.sample_rate)

    log.info("Building retriever from %s (top_k=3) ...", args.corpus)
    retriever = _build_retriever(args.corpus, top_k=3)

    top3_by_key: Dict[str, List[RetrievedChunk]] = {}
    for row in sampled:
        key = _annotation_key(row)
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        try:
            top3_by_key[key] = _retrieve_top_n(retriever, query, 3)
        except Exception as ex:  # noqa: BLE001
            log.warning("retrieval failed for key=%s: %s", key, ex)
            top3_by_key[key] = []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = _render_html(
        rows=sampled,
        top3_by_key=top3_by_key,
        corpus_path=args.corpus,
        dataset_path=args.dataset,
    )
    out_path.write_text(html_text, encoding="utf-8")
    log.info("Wrote HTML report to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
