"""Phase 7.0 — diff report between two v4 chunk exports.

Reads the two ``rag_chunks_<variant>.jsonl`` files produced by
:mod:`eval.harness.v4_chunk_export` and computes structured diff
statistics. Joins each chunk with its page-level metadata (loaded
from ``pages_v4.jsonl``) so we can break the change-rate down by
``page_type``, ``section_type``, ``title_source``, ``alias_source`` —
the Phase 6.3 schema fields that drive the retrieval_title behaviour.

The two chunk files are expected to differ *only* in
``embedding_text``; everything else (``chunk_id``, ``doc_id``,
``chunk_text``, metadata) must be byte-identical for a meaningful A/B.
That invariant is also asserted as we go: a divergence on any other
field is a bug in the export pipeline and the report flags it loudly
rather than silently averaging it away.

Outputs:
  - ``variant_diff_report.json`` — full structured payload
  - ``variant_diff_report.md``   — human-readable summary
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

log = logging.getLogger(__name__)


_TOP_BREAKDOWN_KEYS = 12
_TOP_EXAMPLES = 15
_PREVIEW_CHARS = 220


@dataclass(frozen=True)
class PageMeta:
    """Page-level fields needed for breakdowns. All optional."""

    page_id: str
    page_type: str
    page_title: str
    canonical_url: str
    title_source: str
    alias_source: str


def load_page_meta(pages_v4_path: Path) -> Dict[str, PageMeta]:
    """Stream pages_v4.jsonl into a ``page_id → PageMeta`` map.

    Only the fields needed for the diff report breakdowns are kept; the
    full ~366MB page payload (sections, summaries, etc.) is discarded
    during the streaming pass so the resulting dict stays compact
    (~4,314 entries ≈ a few MB).
    """
    out: Dict[str, PageMeta] = {}
    with Path(pages_v4_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            page_id = str(rec.get("page_id") or "")
            if not page_id:
                continue
            out[page_id] = PageMeta(
                page_id=page_id,
                page_type=str(rec.get("page_type") or ""),
                page_title=str(rec.get("page_title") or ""),
                canonical_url=str(rec.get("canonical_url") or ""),
                title_source=str(rec.get("title_source") or ""),
                alias_source=str(rec.get("alias_source") or ""),
            )
    return out


def _iter_chunk_pairs(
    baseline_path: Path,
    candidate_path: Path,
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield aligned (baseline_record, candidate_record) pairs.

    Assumes both files share the same chunk order — they were produced
    by the same streaming export from the same source rag_chunks.jsonl.
    Raises if they fall out of sync (different chunk_ids at the same
    line index) since that means the comparison is meaningless.
    """
    with Path(baseline_path).open("r", encoding="utf-8") as bfp, \
         Path(candidate_path).open("r", encoding="utf-8") as cfp:
        for lineno, (b_line, c_line) in enumerate(zip(bfp, cfp), start=1):
            b_line = b_line.strip()
            c_line = c_line.strip()
            if not b_line and not c_line:
                continue
            if not b_line or not c_line:
                raise RuntimeError(
                    f"Chunk export length mismatch at line {lineno}: "
                    f"baseline={'<empty>' if not b_line else 'present'}, "
                    f"candidate={'<empty>' if not c_line else 'present'}"
                )
            b_rec = json.loads(b_line)
            c_rec = json.loads(c_line)
            if b_rec.get("chunk_id") != c_rec.get("chunk_id"):
                raise RuntimeError(
                    f"Chunk_id desync at line {lineno}: "
                    f"baseline={b_rec.get('chunk_id')!r} "
                    f"candidate={c_rec.get('chunk_id')!r}"
                )
            yield b_rec, c_rec


def _short_section_path(section_path: Any) -> str:
    if isinstance(section_path, list):
        return " > ".join(str(s) for s in section_path if s)
    return str(section_path or "")


def _take_preview(s: str) -> str:
    if not s:
        return ""
    return s if len(s) <= _PREVIEW_CHARS else s[:_PREVIEW_CHARS] + "…"


def compute_variant_diff(
    baseline_chunks_path: Path,
    candidate_chunks_path: Path,
    *,
    pages_v4_path: Path,
    baseline_variant: str,
    candidate_variant: str,
) -> Dict[str, Any]:
    """Compute a structured diff between two chunk exports.

    Returns a dict ready for json.dumps. Side-by-side stats:

      - total_chunks
      - changed_embedding_text_count + ratio
      - changed_by_page_type / section_type / title_source / alias_source
      - top changed examples (up to :data:`_TOP_EXAMPLES`)
      - integrity counters (non-embedding-text divergence, missing page meta)
    """
    pages_meta = load_page_meta(pages_v4_path)

    total = 0
    changed = 0
    changed_by_page_type: Counter = Counter()
    changed_by_section_type: Counter = Counter()
    changed_by_title_source: Counter = Counter()
    changed_by_alias_source: Counter = Counter()
    total_by_page_type: Counter = Counter()
    total_by_section_type: Counter = Counter()
    total_by_title_source: Counter = Counter()
    total_by_alias_source: Counter = Counter()
    integrity_non_embed_diffs = 0
    missing_page_meta = 0
    examples: List[Dict[str, Any]] = []

    for b_rec, c_rec in _iter_chunk_pairs(
        baseline_chunks_path, candidate_chunks_path,
    ):
        total += 1
        # Integrity check: every non-embedding_text field must match
        # byte-for-byte. ``aliases`` is a list-of-string and ``metadata``
        # is a dict; both should compare with == directly.
        all_keys = set(b_rec.keys()) | set(c_rec.keys())
        for key in all_keys:
            if key == "embedding_text":
                continue
            if b_rec.get(key) != c_rec.get(key):
                integrity_non_embed_diffs += 1
                break

        page_id = str(b_rec.get("doc_id") or "")
        meta = pages_meta.get(page_id)
        page_type = meta.page_type if meta else ""
        title_source = meta.title_source if meta else ""
        alias_source = meta.alias_source if meta else ""
        if not meta:
            missing_page_meta += 1
        section_type = str(b_rec.get("section_type") or "")

        total_by_page_type[page_type or "<unknown>"] += 1
        total_by_section_type[section_type or "<unknown>"] += 1
        total_by_title_source[title_source or "<unknown>"] += 1
        total_by_alias_source[alias_source or "<unknown>"] += 1

        if b_rec.get("embedding_text") != c_rec.get("embedding_text"):
            changed += 1
            changed_by_page_type[page_type or "<unknown>"] += 1
            changed_by_section_type[section_type or "<unknown>"] += 1
            changed_by_title_source[title_source or "<unknown>"] += 1
            changed_by_alias_source[alias_source or "<unknown>"] += 1
            if len(examples) < _TOP_EXAMPLES:
                examples.append({
                    "chunk_id": b_rec.get("chunk_id"),
                    "page_id": page_id,
                    "canonical_url": meta.canonical_url if meta else "",
                    "page_title": b_rec.get("title"),
                    "retrieval_title": b_rec.get("retrieval_title"),
                    "section_path": _short_section_path(
                        b_rec.get("section_path"),
                    ),
                    "section_type": section_type,
                    "page_type": page_type,
                    "title_source": title_source,
                    "alias_source": alias_source,
                    "old_embedding_text_preview": _take_preview(
                        b_rec.get("embedding_text") or ""
                    ),
                    "new_embedding_text_preview": _take_preview(
                        c_rec.get("embedding_text") or ""
                    ),
                })

    ratio = (changed / total) if total else 0.0
    return {
        "schema": "v4-variant-diff-report.v1",
        "baseline_variant": baseline_variant,
        "candidate_variant": candidate_variant,
        "baseline_path": str(Path(baseline_chunks_path).resolve()),
        "candidate_path": str(Path(candidate_chunks_path).resolve()),
        "pages_v4_path": str(Path(pages_v4_path).resolve()),
        "total_chunks": total,
        "changed_embedding_text_count": changed,
        "changed_embedding_text_ratio": ratio,
        "changed_by_page_type": _topn_dict(
            changed_by_page_type, total_by_page_type,
        ),
        "changed_by_section_type": _topn_dict(
            changed_by_section_type, total_by_section_type,
        ),
        "changed_by_title_source": _topn_dict(
            changed_by_title_source, total_by_title_source,
        ),
        "changed_by_alias_source": _topn_dict(
            changed_by_alias_source, total_by_alias_source,
        ),
        "integrity_non_embedding_text_diffs": integrity_non_embed_diffs,
        "missing_page_meta": missing_page_meta,
        "top_examples": examples,
    }


def _topn_dict(
    changed: Counter, total: Counter, *, limit: int = _TOP_BREAKDOWN_KEYS,
) -> List[Dict[str, Any]]:
    """Return a ranked list of the top breakdown buckets.

    Ranks by changed-count descending, ties broken by total-count
    descending then key. ``limit`` caps the list so the JSON stays
    scannable for pages with hundreds of section_types.
    """
    keys = sorted(
        set(changed.keys()) | set(total.keys()),
        key=lambda k: (-changed.get(k, 0), -total.get(k, 0), k),
    )
    out: List[Dict[str, Any]] = []
    for k in keys[:limit]:
        c = int(changed.get(k, 0))
        t = int(total.get(k, 0))
        ratio = (c / t) if t else 0.0
        out.append({
            "key": k,
            "changed": c,
            "total": t,
            "ratio": ratio,
        })
    return out


def render_variant_diff_md(report: Dict[str, Any]) -> str:
    """Format ``compute_variant_diff`` output as Markdown for humans."""
    lines: List[str] = []
    lines.append(
        f"# v4 variant diff — "
        f"{report['baseline_variant']} → {report['candidate_variant']}"
    )
    lines.append("")
    lines.append(f"- baseline_path: `{report['baseline_path']}`")
    lines.append(f"- candidate_path: `{report['candidate_path']}`")
    lines.append(f"- pages_v4_path: `{report['pages_v4_path']}`")
    lines.append(f"- total_chunks: **{report['total_chunks']}**")
    ratio_pct = report["changed_embedding_text_ratio"] * 100.0
    lines.append(
        f"- changed_embedding_text_count: "
        f"**{report['changed_embedding_text_count']} "
        f"({ratio_pct:.2f}%)**"
    )
    lines.append(
        f"- integrity_non_embedding_text_diffs: "
        f"{report['integrity_non_embedding_text_diffs']}"
    )
    lines.append(f"- missing_page_meta: {report['missing_page_meta']}")
    lines.append("")

    for label, key in (
        ("Changed by page_type", "changed_by_page_type"),
        ("Changed by section_type", "changed_by_section_type"),
        ("Changed by title_source", "changed_by_title_source"),
        ("Changed by alias_source", "changed_by_alias_source"),
    ):
        rows = report.get(key) or []
        lines.append(f"## {label}")
        lines.append("")
        if not rows:
            lines.append("(empty)")
            lines.append("")
            continue
        lines.append("| key | changed | total | ratio |")
        lines.append("|---|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| `{r['key']}` | {r['changed']} | "
                f"{r['total']} | {r['ratio']*100:.1f}% |"
            )
        lines.append("")

    examples = report.get("top_examples") or []
    if examples:
        lines.append("## Top changed examples")
        lines.append("")
        for i, ex in enumerate(examples, 1):
            lines.append(
                f"### {i}. page_title=`{ex['page_title']}` "
                f"→ retrieval_title=`{ex['retrieval_title']}`"
            )
            lines.append("")
            lines.append(f"- chunk_id: `{ex['chunk_id']}`")
            lines.append(f"- page_id: `{ex['page_id']}`")
            lines.append(f"- canonical_url: `{ex['canonical_url']}`")
            lines.append(f"- section_path: `{ex['section_path']}`")
            lines.append(
                f"- section_type=`{ex['section_type']}`, "
                f"page_type=`{ex['page_type']}`, "
                f"title_source=`{ex['title_source']}`, "
                f"alias_source=`{ex['alias_source']}`"
            )
            lines.append("")
            lines.append("**old embedding_text:**")
            lines.append("")
            lines.append("```")
            lines.append(ex["old_embedding_text_preview"])
            lines.append("```")
            lines.append("")
            lines.append("**new embedding_text:**")
            lines.append("")
            lines.append("```")
            lines.append(ex["new_embedding_text_preview"])
            lines.append("```")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_variant_diff_report(
    report: Dict[str, Any],
    *,
    out_dir: Path,
    json_name: str = "variant_diff_report.json",
    md_name: str = "variant_diff_report.md",
) -> Tuple[Path, Path]:
    """Persist the diff report as ``.json`` + ``.md`` under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / json_name
    md_path = out_dir / md_name
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_variant_diff_md(report), encoding="utf-8")
    return json_path, md_path
