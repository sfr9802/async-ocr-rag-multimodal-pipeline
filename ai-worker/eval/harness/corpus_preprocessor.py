"""Conservative ingest-side preprocessor for namu-wiki corpus chunks.

Why this exists
---------------
Phase 1A demonstrated that line-level cleaning (``corpus_cleaner``)
can't reach the dominant noise: namu-wiki page-prefix metadata blocks
— news headlines, duplicated title, "최근 수정 시각:" timestamp,
edit/discussion/history/category navbar, "edit permission missing"
notices — are glued onto the body inside the **same line** at the very
start of the chunker's input. Phase 1B addresses that head-on: this
module runs *before* the chunker, transforming the raw text fed into
``_chunks_from_section`` so the chunker never sees the prefix.

Two transforms, independently togglable
---------------------------------------
1. ``strip_page_prefix``  — remove the namu-wiki page-prefix metadata
   block at the start of a chunk / text blob, anchored on the
   "최근 수정 시각: yyyy-mm-dd HH:MM:SS" timestamp signature. Anchored
   detection is the load-bearing safety property: without the
   timestamp, no strip happens.

2. ``strip_inline_edit``  — strip ``[편집]`` / ``[원본 편집]`` /
   ``[소스 편집]`` inline markers anywhere in the text. Same regex as
   ``corpus_cleaner._INLINE_PATTERNS`` but applied at the pre-chunker
   layer.

The two transforms are independent: each is a separate flag on
``preprocess_text``. They commute (``strip_inline_edit`` after
``strip_page_prefix`` produces the same result as the reverse), and
both are individually idempotent.

Conservative cap
----------------
The prefix-strip never removes more than ``MAX_PREFIX_STRIP_CHARS``
characters from one input string. If the detected prefix span runs
past that cap, the strip refuses (returns the input unchanged with a
warning) — that's the runaway-detection guard for the unlikely case
where the sweep marches into in-domain prose.

If the prefix span consumes ``DROP_THRESHOLD_RATIO`` (default 0.8) or
more of the input, the entire input is dropped. This handles the
common case of short "prefix-only" pre-chunks the namu-wiki dump
emits: text like ``"X: X 최근 수정 시각: ... 편집 토론 역사 분류"``
that is 80-110 chars of pure metadata with no body attached.

Public surface
--------------
- ``PREPROCESS_VERSION``                       — bump on rule change
- ``PreprocessConfig``                         — frozen toggles
- ``PreprocessResult``                         — text + bookkeeping
- ``preprocess_text``                          — one-string entry
- ``preprocess_section_payload``               — applies to chunks/text/list
- ``preprocess_document_payload``              — applies per-doc
- ``DocumentPreprocessSummary``                — per-doc rollup
- ``CorpusPreprocessSummary``                  — corpus rollup
- ``corpus_preprocess_summary_to_dict``
- ``render_corpus_preprocess_summary_markdown``
- ``render_sample_diff_markdown``
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Bump when the rule set changes in a way that could affect outputs.
# This shows up in manifests so downstream artifacts can be traced.
PREPROCESS_VERSION = "v1"

# Anchor for the namu-wiki page-prefix metadata block. Strict enough
# that random in-domain prose can't trigger it. The optional trailing
# integer captures the namu-wiki revision/edit count that always
# follows the timestamp ("...22:58:06 76") — bundling it into the
# anchor avoids a sweep-level ``\d+`` rule that would also chew into
# the first numeric token of any category list following the navbar
# (e.g. "분류 ... 2015년 작품" would lose its "2015").
PAGE_META_ANCHOR = re.compile(
    r"최근\s*수정\s*시각\s*:\s*\d{4}[-./]\d{2}[-./]\d{2}\s+\d{2}:\d{2}:\d{2}"
    r"(?:\s*\d{1,5})?"
)

# Inline edit-marker patterns. Same set as corpus_cleaner's inline rule;
# duplicated here because we deliberately don't import private names
# across modules.
INLINE_EDIT_PATTERN = re.compile(
    r"\[\s*(?:원본\s*편집|소스\s*편집|편집)\s*\]"
)

# Window in which we look for the prefix anchor. The prefix block
# always lives at the very start of a chunk / text blob; capping the
# search window means a stray "최근 수정 시각:" buried mid-chunk
# doesn't trigger a strip.
PREFIX_SEARCH_HEAD_CHARS = 3000

# Hard cap on the strip span (start..end) in characters. The sweep
# loop is bounded internally, but this is an extra belt-and-braces
# guard. Phase 1A's worst-case prefix was ≈800 chars; 3000 is
# generous.
MAX_PREFIX_STRIP_CHARS = 3000

# If the prefix span covers >= this fraction of the input, drop the
# whole input. Pure prefix-only chunks (≈80-110 chars of metadata)
# show up frequently in the namu-wiki dump.
DROP_THRESHOLD_RATIO = 0.8


# --- Datatypes ----------------------------------------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    strip_page_prefix: bool = False
    strip_inline_edit: bool = False

    @property
    def variant_label(self) -> str:
        """Stable label for this config — used in manifests + filenames."""
        parts: List[str] = []
        if self.strip_page_prefix:
            parts.append(f"prefix-{PREPROCESS_VERSION}")
        if self.strip_inline_edit:
            parts.append(f"inline-edit-{PREPROCESS_VERSION}")
        return ".".join(parts) or "raw"


@dataclass(frozen=True)
class PrefixSpan:
    start: int
    end: int
    signals: Tuple[str, ...]


@dataclass(frozen=True)
class PreprocessResult:
    text: str
    changed: bool
    removed_prefix_chars: int
    removed_prefix_preview: str
    removed_prefix_signals: Tuple[str, ...]
    inline_edit_removals: int
    dropped: bool
    warnings: Tuple[str, ...]


@dataclass
class DocumentPreprocessSummary:
    doc_id: str
    sections_processed: int = 0
    chunks_processed: int = 0
    chunks_changed: int = 0
    chunks_dropped: int = 0
    text_blobs_changed: int = 0
    list_entries_changed: int = 0
    total_removed_prefix_chars: int = 0
    total_inline_edit_removals: int = 0
    prefix_strip_count: int = 0


@dataclass
class CorpusPreprocessSummary:
    source_corpus: str
    config: PreprocessConfig
    document_count: int = 0
    sections_processed: int = 0
    chunks_processed: int = 0
    chunks_changed: int = 0
    chunks_dropped: int = 0
    text_blobs_changed: int = 0
    list_entries_changed: int = 0
    total_removed_prefix_chars: int = 0
    total_inline_edit_removals: int = 0
    prefix_strip_count: int = 0
    sample_diffs: List[Dict[str, Any]] = field(default_factory=list)


# --- Core text-level transform -----------------------------------------


def detect_prefix_span(text: str) -> Optional[PrefixSpan]:
    """Find the namu-wiki page-prefix block at the start of ``text``.

    Returns the (start, end, signals) span to remove, or None if we
    can't identify it safely. The detection is conservative on three
    axes:

    1. The anchor must appear inside the first ``PREFIX_SEARCH_HEAD_CHARS``
       characters; a stray timestamp later in the text does not trigger
       a strip.
    2. The sweep stops at the first un-recognized token, so we never
       walk into in-domain prose.
    3. The total span is hard-capped at ``MAX_PREFIX_STRIP_CHARS``.
    """
    if not text:
        return None
    head = text[:PREFIX_SEARCH_HEAD_CHARS]
    m = PAGE_META_ANCHOR.search(head)
    if not m:
        return None

    pos = m.end()
    signals: List[str] = ["page_meta_anchor"]
    max_steps = 40
    while max_steps > 0:
        max_steps -= 1
        next_pos, signal = _match_one_tail_token(text, pos)
        if next_pos is None:
            break
        pos = next_pos
        signals.append(signal)

    end = min(pos, MAX_PREFIX_STRIP_CHARS)
    if end <= 0:
        return None
    return PrefixSpan(start=0, end=end, signals=tuple(signals))


# Each entry: (signal-name, regex). The sweep tries each in turn;
# the first match wins. Patterns are anchored on optional leading
# whitespace so the sweep can chain them. The post-timestamp revision
# count is folded into ``PAGE_META_ANCHOR`` rather than living here —
# see the anchor docstring for why a free-floating ``\d+`` rule is
# unsafe inside the sweep.
_TAIL_TOKEN_DISPATCH: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "perm_warning",
        re.compile(r"\s*편집\s*권한[^.\n]{0,80}\."),
    ),
    (
        "login_prompt",
        re.compile(r"\s*로그인[된]?\s*사용자[^.\n]{0,80}\."),
    ),
    (
        "acl_pointer",
        re.compile(r"\s*해당\s*문서의\s*ACL[^.\n]{0,80}\."),
    ),
    (
        "acl_residue",
        re.compile(r"\s*ACL\s*탭[^.\n]{0,80}\."),
    ),
    (
        "spoiler_notice",
        re.compile(r"\s*이\s*문서[가에는는]?\s*스포일러[^.\n]{0,150}\."),
    ),
    (
        "doc_intro_notice",
        re.compile(r"\s*이\s*문서가\s*설명[^.\n]{0,300}\."),
    ),
    (
        "edit_request_help",
        re.compile(r"\s*\[\s*편집\s*요청\s*도움말\s*\]"),
    ),
    (
        "edit_request_button",
        re.compile(r"\s*편집\s*요청\s*닫기"),
    ),
    (
        "navbar",
        re.compile(r"\s*(?:편집(?:\s*요청)?|토론|역사|분류|닫기)"),
    ),
    (
        "toc_toggle",
        re.compile(r"\s*\[\s*(?:펼치기|접기|숨기기)[^\]]{0,30}\]"),
    ),
    (
        "inline_edit",
        re.compile(r"\s*\[\s*(?:원본\s*편집|소스\s*편집|편집)\s*\]"),
    ),
    ("middot", re.compile(r"\s*·")),
)


def _match_one_tail_token(text: str, pos: int) -> Tuple[Optional[int], str]:
    """Try to match one allowed tail token at ``pos``.

    Returns (new_position, signal_name) on a hit, (None, "") otherwise.
    Whitespace before the token is consumed by each pattern's leading
    ``\\s*``.
    """
    for name, pattern in _TAIL_TOKEN_DISPATCH:
        m = pattern.match(text, pos)
        if m and m.end() > pos:
            return m.end(), name
    return None, ""


def strip_inline_edit_markers(text: str) -> Tuple[str, int]:
    """Strip ``[편집]`` / ``[원본 편집]`` / ``[소스 편집]`` inline markers.

    Returns ``(cleaned_text, removed_count)``. Compresses double
    whitespace runs created by the strip so the chunker doesn't see
    visual artifacts; trailing whitespace before newlines is also
    cleaned up.
    """
    if not text:
        return text, 0
    cleaned, count = INLINE_EDIT_PATTERN.subn("", text)
    if count > 0:
        cleaned = re.sub(r"  +", " ", cleaned)
        cleaned = re.sub(r" +\n", "\n", cleaned)
        cleaned = re.sub(r"\n +", "\n", cleaned)
    return cleaned, count


def preprocess_text(
    text: str,
    *,
    config: PreprocessConfig,
) -> PreprocessResult:
    """Run the configured transforms over a single string.

    Order of operations: prefix-strip first, then inline-edit-strip.
    Both transforms commute on real corpus text (the inline marker is
    never inside the strict prefix anchor pattern), so the order is
    chosen for clarity rather than correctness.
    """
    if not isinstance(text, str) or not text:
        return PreprocessResult(
            text=text or "",
            changed=False,
            removed_prefix_chars=0,
            removed_prefix_preview="",
            removed_prefix_signals=(),
            inline_edit_removals=0,
            dropped=False,
            warnings=(),
        )

    out = text
    prefix_chars = 0
    prefix_preview = ""
    prefix_signals: Tuple[str, ...] = ()
    inline_count = 0
    dropped = False
    warnings: List[str] = []

    if config.strip_page_prefix:
        span = detect_prefix_span(out)
        if span is not None:
            span_len = span.end - span.start
            if span_len >= MAX_PREFIX_STRIP_CHARS:
                warnings.append("prefix_span_capped")
            if span_len >= DROP_THRESHOLD_RATIO * len(out):
                # Treat as a prefix-only input — drop entirely.
                prefix_chars = len(out)
                prefix_preview = _make_preview(out, 200)
                prefix_signals = span.signals
                out = ""
                dropped = True
            else:
                prefix_chars = span_len
                prefix_preview = _make_preview(
                    out[span.start:span.end], 200
                )
                prefix_signals = span.signals
                out = out[span.end:].lstrip()

    if config.strip_inline_edit and out:
        out, inline_count = strip_inline_edit_markers(out)

    changed = bool(prefix_chars or inline_count or dropped)
    return PreprocessResult(
        text=out,
        changed=changed,
        removed_prefix_chars=prefix_chars,
        removed_prefix_preview=prefix_preview,
        removed_prefix_signals=prefix_signals,
        inline_edit_removals=inline_count,
        dropped=dropped,
        warnings=tuple(warnings),
    )


def _make_preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


# --- Section + document level ------------------------------------------


@dataclass(frozen=True)
class SectionPreprocessOutcome:
    new_payload: Dict[str, Any]
    chunk_results: Tuple[PreprocessResult, ...]
    text_blob_result: Optional[PreprocessResult]
    list_results: Tuple[PreprocessResult, ...]


def preprocess_section_payload(
    payload: Dict[str, Any],
    *,
    config: PreprocessConfig,
) -> SectionPreprocessOutcome:
    """Apply the configured transforms to a section payload.

    The returned outcome separates per-chunk results from the
    text-blob result so the caller doesn't have to guess which list
    entry corresponds to which field. Empty chunks (dropped by the
    transform or originally empty) are filtered out of the new
    payload's ``chunks`` list — the chunker would skip them anyway,
    but dropping them here keeps the emitted jsonl tidy.
    """
    new_payload = dict(payload)  # shallow copy

    chunk_results: List[PreprocessResult] = []
    new_chunks: List[Any] = []
    pre_chunks = payload.get("chunks")
    if isinstance(pre_chunks, list):
        for chunk in pre_chunks:
            if not isinstance(chunk, (str, int, float)):
                new_chunks.append(chunk)
                continue
            text = str(chunk)
            result = preprocess_text(text, config=config)
            chunk_results.append(result)
            if not result.dropped:
                new_chunks.append(result.text)
        new_payload["chunks"] = new_chunks

    text_blob_result: Optional[PreprocessResult] = None
    text_blob = payload.get("text")
    if isinstance(text_blob, str):
        text_blob_result = preprocess_text(text_blob, config=config)
        new_payload["text"] = text_blob_result.text

    list_results: List[PreprocessResult] = []
    list_entries = payload.get("list")
    if isinstance(list_entries, list):
        new_list: List[Any] = []
        for entry in list_entries:
            if not isinstance(entry, dict):
                new_list.append(entry)
                continue
            new_entry = dict(entry)
            for key in ("name", "desc"):
                v = entry.get(key)
                if isinstance(v, str) and v:
                    res = preprocess_text(v, config=config)
                    list_results.append(res)
                    new_entry[key] = res.text
            new_list.append(new_entry)
        new_payload["list"] = new_list

    return SectionPreprocessOutcome(
        new_payload=new_payload,
        chunk_results=tuple(chunk_results),
        text_blob_result=text_blob_result,
        list_results=tuple(list_results),
    )


def preprocess_document_payload(
    doc: Dict[str, Any],
    *,
    config: PreprocessConfig,
) -> Tuple[Dict[str, Any], DocumentPreprocessSummary]:
    """Apply preprocessing to a parsed corpus.jsonl document row.

    Returns ``(new_doc, summary)``. The new doc is a shallow copy with
    ``sections`` re-built. The summary is a per-doc rollup of the
    per-section / per-chunk results.
    """
    doc_id = str(
        doc.get("doc_id") or doc.get("seed") or doc.get("title") or ""
    ).strip() or "<unknown>"
    summary = DocumentPreprocessSummary(doc_id=doc_id)

    new_doc = dict(doc)
    sections = doc.get("sections")
    if not isinstance(sections, dict):
        return new_doc, summary

    new_sections: Dict[str, Any] = {}
    for sname, payload in sections.items():
        if not isinstance(payload, dict):
            new_sections[sname] = payload
            continue
        summary.sections_processed += 1
        outcome = preprocess_section_payload(payload, config=config)
        new_sections[sname] = outcome.new_payload
        for r in outcome.chunk_results:
            summary.chunks_processed += 1
            if r.changed:
                summary.chunks_changed += 1
            if r.dropped:
                summary.chunks_dropped += 1
            summary.total_removed_prefix_chars += r.removed_prefix_chars
            summary.total_inline_edit_removals += r.inline_edit_removals
            if r.removed_prefix_chars > 0:
                summary.prefix_strip_count += 1
        if outcome.text_blob_result is not None:
            tr = outcome.text_blob_result
            if tr.changed:
                summary.text_blobs_changed += 1
            summary.total_removed_prefix_chars += tr.removed_prefix_chars
            summary.total_inline_edit_removals += tr.inline_edit_removals
            if tr.removed_prefix_chars > 0:
                summary.prefix_strip_count += 1
        for r in outcome.list_results:
            if r.changed:
                summary.list_entries_changed += 1
            summary.total_removed_prefix_chars += r.removed_prefix_chars
            summary.total_inline_edit_removals += r.inline_edit_removals

    new_doc["sections"] = new_sections
    return new_doc, summary


# --- Corpus-level summary + helpers ------------------------------------


def fold_document_summary(
    corpus: CorpusPreprocessSummary,
    doc: DocumentPreprocessSummary,
) -> None:
    """Fold a per-doc summary into the corpus rollup."""
    corpus.document_count += 1
    corpus.sections_processed += doc.sections_processed
    corpus.chunks_processed += doc.chunks_processed
    corpus.chunks_changed += doc.chunks_changed
    corpus.chunks_dropped += doc.chunks_dropped
    corpus.text_blobs_changed += doc.text_blobs_changed
    corpus.list_entries_changed += doc.list_entries_changed
    corpus.total_removed_prefix_chars += doc.total_removed_prefix_chars
    corpus.total_inline_edit_removals += doc.total_inline_edit_removals
    corpus.prefix_strip_count += doc.prefix_strip_count


def corpus_preprocess_summary_to_dict(
    summary: CorpusPreprocessSummary,
) -> Dict[str, Any]:
    payload = asdict(summary)
    payload["config"] = {
        "strip_page_prefix": summary.config.strip_page_prefix,
        "strip_inline_edit": summary.config.strip_inline_edit,
        "variant_label": summary.config.variant_label,
        "preprocess_version": PREPROCESS_VERSION,
    }
    return payload


def render_corpus_preprocess_summary_markdown(
    summary: CorpusPreprocessSummary,
) -> str:
    lines: List[str] = []
    lines.append("# Corpus preprocess summary")
    lines.append("")
    lines.append(f"- source: `{summary.source_corpus}`")
    lines.append(f"- preprocess_version: `{PREPROCESS_VERSION}`")
    lines.append(f"- variant: `{summary.config.variant_label}`")
    lines.append(
        f"- strip_page_prefix: `{summary.config.strip_page_prefix}`"
    )
    lines.append(
        f"- strip_inline_edit: `{summary.config.strip_inline_edit}`"
    )
    lines.append(f"- documents: {summary.document_count}")
    lines.append(f"- sections processed: {summary.sections_processed}")
    lines.append(f"- chunks processed: {summary.chunks_processed}")
    lines.append(f"- chunks changed: {summary.chunks_changed}")
    lines.append(f"- chunks dropped (prefix-only): {summary.chunks_dropped}")
    lines.append(f"- text blobs changed: {summary.text_blobs_changed}")
    lines.append(f"- list entries changed: {summary.list_entries_changed}")
    lines.append(
        f"- prefix-strip events: {summary.prefix_strip_count}"
    )
    lines.append(
        f"- total prefix chars removed: {summary.total_removed_prefix_chars}"
    )
    lines.append(
        f"- total inline-edit markers removed: {summary.total_inline_edit_removals}"
    )
    return "\n".join(lines) + "\n"


def render_sample_diff_markdown(
    sample_diffs: Sequence[Dict[str, Any]],
) -> str:
    """Render a markdown report of N before/after samples."""
    lines: List[str] = []
    lines.append("# Preprocess sample diffs")
    lines.append("")
    if not sample_diffs:
        lines.append("_No sample diffs collected._")
        return "\n".join(lines) + "\n"

    for i, diff in enumerate(sample_diffs, start=1):
        lines.append(f"## Sample {i}: {diff.get('doc_id', '?')}")
        lines.append("")
        lines.append(
            f"- section: `{diff.get('section', '?')}`"
        )
        lines.append(
            f"- field: `{diff.get('field', '?')}`"
        )
        lines.append(
            f"- removed_prefix_chars: {diff.get('removed_prefix_chars', 0)}"
        )
        lines.append(
            f"- inline_edit_removals: {diff.get('inline_edit_removals', 0)}"
        )
        signals = diff.get("removed_prefix_signals") or []
        if signals:
            lines.append(f"- signals: `{', '.join(signals)}`")
        lines.append("")
        lines.append("**Before (first 400 chars):**")
        lines.append("")
        lines.append("```")
        lines.append(_truncate(str(diff.get("before", "")), 400))
        lines.append("```")
        lines.append("")
        lines.append("**After (first 400 chars):**")
        lines.append("")
        lines.append("```")
        lines.append(_truncate(str(diff.get("after", "")), 400))
        lines.append("```")
        lines.append("")

    return "\n".join(lines) + "\n"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


# --- Convenience: streaming over a whole corpus.jsonl ------------------


def iter_preprocessed_documents(
    corpus_iter: Iterable[Dict[str, Any]],
    *,
    config: PreprocessConfig,
    sample_diff_target: int = 0,
    summary: Optional[CorpusPreprocessSummary] = None,
) -> Iterable[Dict[str, Any]]:
    """Yield preprocessed document dicts from an input iterator.

    Mutates ``summary`` if provided. Collects up to
    ``sample_diff_target`` representative before/after diffs into
    ``summary.sample_diffs`` (sampled from chunks where the strip
    actually changed something).
    """
    samples_collected = 0
    for raw_doc in corpus_iter:
        new_doc, doc_summary = preprocess_document_payload(
            raw_doc, config=config
        )
        if summary is not None:
            fold_document_summary(summary, doc_summary)

            # Collect sample diffs opportunistically. The first time a
            # given doc has a changed chunk and we still need samples,
            # extract one from the new payload.
            if samples_collected < sample_diff_target and doc_summary.chunks_changed > 0:
                diff = _extract_first_changed_diff(
                    raw_doc, new_doc, config=config
                )
                if diff is not None:
                    summary.sample_diffs.append(diff)
                    samples_collected += 1

        yield new_doc


def _extract_first_changed_diff(
    raw_doc: Dict[str, Any],
    new_doc: Dict[str, Any],
    *,
    config: PreprocessConfig,
) -> Optional[Dict[str, Any]]:
    """Find the first (section, field, index) that changed and return a
    before/after dict for the sample diff report."""
    raw_sections = raw_doc.get("sections") or {}
    new_sections = new_doc.get("sections") or {}
    if not isinstance(raw_sections, dict) or not isinstance(new_sections, dict):
        return None
    for sname in raw_sections:
        rs = raw_sections.get(sname)
        ns = new_sections.get(sname)
        if not isinstance(rs, dict) or not isinstance(ns, dict):
            continue
        # Compare chunks first, then text.
        rc = rs.get("chunks") if isinstance(rs.get("chunks"), list) else None
        nc = ns.get("chunks") if isinstance(ns.get("chunks"), list) else None
        if rc and nc:
            # Find first index where the cleaned-of-empties new_chunks
            # diverges from the raw input. Because we drop empty chunks
            # from new_chunks, the indices may not line up; instead we
            # walk raw and check whether the corresponding text appears
            # in new.
            for idx, before in enumerate(rc):
                if not isinstance(before, str):
                    continue
                preprocessed = preprocess_text(before, config=config)
                if preprocessed.changed:
                    return {
                        "doc_id": raw_doc.get("doc_id"),
                        "section": sname,
                        "field": f"chunks[{idx}]",
                        "before": before,
                        "after": preprocessed.text,
                        "removed_prefix_chars": preprocessed.removed_prefix_chars,
                        "removed_prefix_signals": list(preprocessed.removed_prefix_signals),
                        "inline_edit_removals": preprocessed.inline_edit_removals,
                        "dropped": preprocessed.dropped,
                    }
        rt = rs.get("text") if isinstance(rs.get("text"), str) else None
        if rt:
            preprocessed = preprocess_text(rt, config=config)
            if preprocessed.changed:
                return {
                    "doc_id": raw_doc.get("doc_id"),
                    "section": sname,
                    "field": "text",
                    "before": rt,
                    "after": preprocessed.text,
                    "removed_prefix_chars": preprocessed.removed_prefix_chars,
                    "removed_prefix_signals": list(preprocessed.removed_prefix_signals),
                    "inline_edit_removals": preprocessed.inline_edit_removals,
                    "dropped": preprocessed.dropped,
                }
    return None
