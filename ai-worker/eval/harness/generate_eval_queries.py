"""Generate retrieval-eval queries from a namu-wiki anime corpus.

Three backends share the same output schema and CLI:

  * ``--generator deterministic`` (default) — pure-Python templates over
    the corpus fields. Zero LLM calls, byte-identical reruns under a
    fixed ``--seed``. Quality is ``silver`` (best-effort). Use for the
    bulk of generation when reproducibility and speed matter more than
    natural phrasing.

  * ``--generator llm`` — wraps the existing
    ``scripts.dataset.generate_anime_queries`` LLM script (Claude Haiku /
    Sonnet / Ollama Gemma via the Anthropic SDK). Requires
    ``ANTHROPIC_API_KEY`` or ``AIPIPELINE_WORKER_ANTHROPIC_API_KEY``.

  * ``--generator claude-cli`` — drives the local ``claude`` CLI in
    headless ``--print`` mode. Re-uses Claude Code's existing OAuth
    auth, no separate API key required. Choose model with
    ``--llm-backend opus|sonnet|haiku``. Each call processes a batch of
    ~5 docs to amortize subprocess startup.

Both backends emit one JSON object per line with the schema:

    {
      "id":                       "anime-silver-<idx>",
      "query":                    "...",
      "language":                 "ko",
      "expected_doc_ids":         ["<doc_id>"],
      "expected_section_keywords":["<kw>", ...],
      "answer_type":              "summary_plot" | "title_lookup" |
                                  "character_relation" | "body_excerpt" |
                                  "theme_genre" | "setting_worldbuilding",
      "difficulty":               "easy" | "medium" | "hard",
      "tags":                     ["anime", "silver", "synthetic",
                                   "<answer_type>", "<generator>"]
    }

Stratification (deterministic backend) is calibrated to the **actual**
field coverage of ``namu_anime_v3.jsonl``:

  +-----------------------+-----------+---------+
  | type                  | coverage  | target% |
  +-----------------------+-----------+---------+
  | summary_plot          | 100%      | 40%     |
  | character_relation    | 27.8%     | 20%     |
  | title_lookup          | 100%      | 15%     |
  | body_excerpt          | 99.9%     | 10%     |
  | theme_genre           | 100%      | 10%     |
  | setting_worldbuilding | 1.6%      | 5%      |
  +-----------------------+-----------+---------+

Coverage-poor types (``setting_worldbuilding``) are capped at the
number of docs that actually carry the source signal — the generator
never fabricates a section that doesn't exist in the corpus.

Usage::

    python -m eval.harness.generate_eval_queries \\
        --corpus eval/corpora/anime_namu_v3/corpus.jsonl \\
        --out    eval/eval_queries/anime_silver_200.jsonl \\
        --target 200 \\
        --generator deterministic \\
        --seed 42

    python -m eval.harness.generate_eval_queries \\
        --corpus eval/corpora/anime_namu_v3/corpus.jsonl \\
        --out    eval/eval_queries/anime_silver_200_llm.jsonl \\
        --target 200 \\
        --generator llm \\
        --llm-backend claude:haiku
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

log = logging.getLogger("eval.harness.generate_eval_queries")


# ---------------------------------------------------------------------------
# Stratification (calibrated to actual field coverage in namu_anime_v3).
# ---------------------------------------------------------------------------


# Order matters: deterministic generation iterates types in this order
# so a smaller --target N still gets a representative spread (and not,
# say, all summary_plot rows just because that bucket is biggest).
ANSWER_TYPES: Tuple[str, ...] = (
    "summary_plot",
    "title_lookup",
    "character_relation",
    "body_excerpt",
    "theme_genre",
    "setting_worldbuilding",
)

DEFAULT_RATIOS: Dict[str, float] = {
    "summary_plot": 0.40,
    "title_lookup": 0.15,
    "character_relation": 0.20,
    "body_excerpt": 0.10,
    "theme_genre": 0.10,
    "setting_worldbuilding": 0.05,
}

DIFFICULTY_BY_TYPE: Dict[str, str] = {
    "summary_plot": "medium",
    "title_lookup": "easy",
    "character_relation": "medium",
    "body_excerpt": "hard",
    "theme_genre": "medium",
    "setting_worldbuilding": "hard",
}


# ---------------------------------------------------------------------------
# Corpus loader.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    title: str
    summary: str
    summary_bullets: List[str]
    sections: Dict[str, Dict[str, Any]]
    section_order: List[str]


def _iter_corpus(corpus_path: Path) -> Iterator[CorpusDoc]:
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}. See "
            "ai-worker/eval/corpora/<name>/README.md to re-stage."
        )
    with corpus_path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as ex:
                log.warning(
                    "Skipping malformed line %d in %s: %s", line_no, corpus_path, ex
                )
                continue
            doc_id = str(d.get("doc_id") or "").strip()
            title = str(d.get("title") or "").strip()
            if not doc_id or not title:
                continue
            sections = d.get("sections") or {}
            if not isinstance(sections, dict):
                sections = {}
            sec_order = d.get("section_order") or list(sections.keys())
            if not isinstance(sec_order, list):
                sec_order = list(sections.keys())
            bullets = (
                d.get("summary_bullets")
                or d.get("sum_bullets")
                or []
            )
            if not isinstance(bullets, list):
                bullets = []
            yield CorpusDoc(
                doc_id=doc_id,
                title=title,
                summary=str(d.get("summary") or "").strip(),
                summary_bullets=[str(b).strip() for b in bullets if str(b).strip()],
                sections=sections,
                section_order=[str(s) for s in sec_order],
            )


# ---------------------------------------------------------------------------
# Text helpers.
# ---------------------------------------------------------------------------


# Korean noun-ish token: hangul block(s) of length >= 2 OR ascii word.
# This is intentionally crude — synthetic queries are silver, not gold.
_NOUN_LIKE_RE = re.compile(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+|(?<=[다요죠])\s+(?=[A-Z가-힣])")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = _CONTROL_RE.sub(" ", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _split_sentences(text: str, *, max_sentences: int = 6) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    out: List[str] = []
    for p in parts:
        p = p.strip(" .!?")
        if 8 <= len(p) <= 240:
            out.append(p)
        if len(out) >= max_sentences:
            break
    return out


# Common stopwords / fillers we don't want as expected_section_keywords.
# These appear in nearly every doc's summary and would make hits trivial
# and uninformative. Purely pragmatic — extend if a future eval surfaces
# noise.
_NOUN_STOPLIST = {
    "이야기", "작품", "내용", "주인공", "사람", "이상", "정도", "그것",
    "이번", "관련", "다음", "사이", "일이", "상황", "사실", "경우",
    "anime", "manga", "story", "the", "and", "for", "with", "from",
}


def _extract_nouns(text: str, *, limit: int = 6) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    seen: List[str] = []
    seen_set: set[str] = set()
    for match in _NOUN_LIKE_RE.finditer(text):
        token = match.group(0)
        norm = unicodedata.normalize("NFKC", token).strip()
        if not norm or norm.lower() in _NOUN_STOPLIST:
            continue
        if norm in seen_set:
            continue
        seen.append(norm)
        seen_set.add(norm)
        if len(seen) >= limit:
            break
    return seen


# Character-name heuristic for the 등장인물 section: lines that begin
# with a hangul block of 2-4 chars followed by a delimiter (`:`, `-`,
# `(`) or a particle (`은/는/이/가`). Each line in a character section
# is one character entry by namu-wiki convention.
_CHAR_LINE_RE = re.compile(
    r"^([가-힣]{2,4}|[A-Za-z][A-Za-z0-9 ]{1,18})\s*[:：\-\(은는이가]"
)


def _extract_character_names(section_text: str, *, limit: int = 4) -> List[str]:
    if not section_text:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for line in section_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _CHAR_LINE_RE.match(line)
        if not m:
            continue
        name = m.group(1).strip()
        if not name or name in seen or name in _NOUN_STOPLIST:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Per-type query builders.
#
# Every builder returns (query_text, expected_section_keywords) or None
# when the doc lacks the source signal. Builders never fabricate text
# that isn't in the doc — if the keywords aren't substrings of the
# source, the row is dropped.
# ---------------------------------------------------------------------------


def _build_summary_plot(doc: CorpusDoc) -> Optional[Tuple[str, List[str]]]:
    sentences = _split_sentences(doc.summary)
    if not sentences:
        return None
    nouns = _extract_nouns(doc.summary, limit=4)
    if len(nouns) < 2:
        return None
    # Phrase the question around the leading noun phrase, anchored by a
    # second noun so a broad single-noun query doesn't trivially match.
    head, tail = nouns[0], nouns[1]
    query = f"{head}와(과) {tail}이(가) 등장하는 작품의 줄거리를 알려주세요."
    return (query, nouns[:3])


def _build_title_lookup(doc: CorpusDoc) -> Optional[Tuple[str, List[str]]]:
    title = doc.title.strip()
    if not title:
        return None
    nouns = _extract_nouns(doc.summary, limit=2)
    keywords = [title] + nouns
    query = f"{title}이(가) 어떤 작품인가요?"
    return (query, keywords)


def _build_character_relation(doc: CorpusDoc) -> Optional[Tuple[str, List[str]]]:
    section = doc.sections.get("등장인물")
    if not isinstance(section, dict):
        return None
    text = str(section.get("text") or "")
    names = _extract_character_names(text)
    if not names:
        return None
    if len(names) >= 2:
        a, b = names[0], names[1]
        query = f"{doc.title}에서 {a}과(와) {b}의 관계를 설명해 주세요."
        return (query, [a, b, doc.title])
    a = names[0]
    query = f"{doc.title}의 {a}은(는) 어떤 인물인가요?"
    return (query, [a, doc.title])


def _build_body_excerpt(doc: CorpusDoc) -> Optional[Tuple[str, List[str]]]:
    section = doc.sections.get("본문")
    if not isinstance(section, dict):
        return None
    chunks = section.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return None
    # Take the first chunk that yields >= 2 distinct nouns. Skip the
    # very short opening fragments that usually bleed in from layout.
    for chunk in chunks[:6]:
        text = _clean_text(str(chunk))
        if len(text) < 60:
            continue
        nouns = _extract_nouns(text, limit=3)
        if len(nouns) < 2:
            continue
        head, tail = nouns[0], nouns[1]
        query = f"{doc.title} 본문에서 {head}와(과) {tail}이(가) 어떻게 다루어지나요?"
        return (query, [head, tail, doc.title])
    return None


def _build_theme_genre(doc: CorpusDoc) -> Optional[Tuple[str, List[str]]]:
    if not doc.summary_bullets:
        return None
    # Pick the first bullet that has at least one non-stopword noun.
    for bullet in doc.summary_bullets:
        nouns = _extract_nouns(bullet, limit=3)
        if not nouns:
            continue
        head = nouns[0]
        query = f"{doc.title}의 주요 주제 중 '{head}'에 대해 설명해 주세요."
        return (query, [head, doc.title])
    return None


def _build_setting_worldbuilding(doc: CorpusDoc) -> Optional[Tuple[str, List[str]]]:
    section = doc.sections.get("설정") or doc.sections.get("세계관")
    if not isinstance(section, dict):
        return None
    text = str(section.get("text") or "")
    nouns = _extract_nouns(text, limit=3)
    if len(nouns) < 2:
        return None
    head, tail = nouns[0], nouns[1]
    query = f"{doc.title}의 세계관에서 {head}와(과) {tail}이(가) 어떻게 설정되어 있나요?"
    return (query, [head, tail, doc.title])


_BUILDERS = {
    "summary_plot": _build_summary_plot,
    "title_lookup": _build_title_lookup,
    "character_relation": _build_character_relation,
    "body_excerpt": _build_body_excerpt,
    "theme_genre": _build_theme_genre,
    "setting_worldbuilding": _build_setting_worldbuilding,
}


# ---------------------------------------------------------------------------
# Deterministic generator.
# ---------------------------------------------------------------------------


def generate_deterministic(
    docs: List[CorpusDoc],
    *,
    target: int,
    seed: int = 42,
    ratios: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Build up to ``target`` query rows over ``docs``.

    For each answer_type, computes a budget ``round(ratio * target)`` and
    walks docs (shuffled by ``seed``) calling the type's builder. Skips
    docs the builder can't handle (no source signal). Stops a type early
    when its budget is filled. Final list is shuffled deterministically
    so the on-disk JSONL doesn't have all 80 summary_plot rows in a
    single block at the top.
    """
    if not docs:
        return []
    ratios = ratios or DEFAULT_RATIOS
    rng = random.Random(seed)

    rows: List[Dict[str, Any]] = []
    used_keys: set[Tuple[str, str]] = set()  # (doc_id, answer_type)

    for atype in ANSWER_TYPES:
        budget = max(0, int(round(ratios.get(atype, 0.0) * target)))
        if budget <= 0:
            continue
        builder = _BUILDERS[atype]
        # Shuffle a per-type doc copy so the same docs don't dominate
        # every type's prefix when targets are small.
        shuffled = list(docs)
        rng.shuffle(shuffled)
        produced = 0
        for doc in shuffled:
            key = (doc.doc_id, atype)
            if key in used_keys:
                continue
            built = builder(doc)
            if built is None:
                continue
            query, keywords = built
            keywords = [k for k in keywords if k]
            if not keywords:
                continue
            row = {
                "id": f"anime-silver-{atype}-{produced+1:04d}",
                "query": query,
                "language": "ko",
                "expected_doc_ids": [doc.doc_id],
                "expected_section_keywords": keywords[:5],
                "answer_type": atype,
                "difficulty": DIFFICULTY_BY_TYPE.get(atype, "medium"),
                "tags": ["anime", "silver", "synthetic", atype, "deterministic"],
            }
            rows.append(row)
            used_keys.add(key)
            produced += 1
            if produced >= budget:
                break
        if produced < budget:
            log.info(
                "Type %s: requested %d, produced %d (corpus signal limited)",
                atype, budget, produced,
            )

    # Final shuffle so on-disk row order isn't grouped by type.
    rng.shuffle(rows)
    # Re-id sequentially after shuffle so consumers see ascending ids.
    for i, row in enumerate(rows, start=1):
        row["id"] = f"anime-silver-{i:04d}"
    return rows


# ---------------------------------------------------------------------------
# LLM backend wrapper.
#
# The existing scripts.dataset.generate_anime_queries already does the
# Claude/Ollama plumbing, evidence-grounded prompting, and retry logic.
# Re-implementing that here would just duplicate ~600 lines. Instead we
# invoke it as a subprocess and re-shape its output into the new schema.
# Schema crosswalk:
#   old.query                → new.query
#   old.question_type        → new.answer_type   (mapped, see below)
#   old.expected_doc_ids     → new.expected_doc_ids
#   old.expected_keywords    → new.expected_section_keywords
#   (none)                   → new.id, language, difficulty, tags
# ---------------------------------------------------------------------------


# Mapping from the legacy script's question_type vocabulary to ours.
# The legacy script's "factoid" overlaps with our title_lookup but is
# broader; we tag it as summary_plot since most factoid rows in the
# legacy output are answerable from the summary.
_LLM_TYPE_MAP = {
    "factoid": "summary_plot",
    "plot": "summary_plot",
    "character": "character_relation",
    "theme": "theme_genre",
    "setting": "setting_worldbuilding",
    "title": "title_lookup",
    "body": "body_excerpt",
}


def generate_llm(
    *,
    corpus_path: Path,
    target: int,
    out_path: Path,
    backend: str = "claude:haiku",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run the legacy LLM script and re-shape its output.

    Two-step: invoke the legacy script (which writes to a sibling
    ``_raw.jsonl``), then read it back and produce rows in our schema.
    The raw file is preserved for audit — silver runs benefit from
    keeping the original generator's evidence_chunk + generator metadata
    around for debugging mismatched queries.

    NOTE: requires Anthropic API credentials when ``backend`` starts
    with ``claude:``, or a running Ollama daemon when it starts with
    ``ollama:``. The legacy script handles both — see
    ``scripts/dataset/generate_anime_queries.py`` for the full flag
    surface.
    """
    raw_path = out_path.with_suffix(".raw.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    queries_per_title = max(1, target // max(1, _count_lines(corpus_path)))

    cmd = [
        sys.executable,
        "-m",
        "scripts.dataset.generate_anime_queries",
        "--corpus",
        str(corpus_path),
        "--out",
        str(raw_path),
        "--queries-per-title",
        str(queries_per_title),
        "--generator",
        backend,
        "--seed",
        str(seed),
    ]
    log.info("LLM mode: invoking %s", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Legacy LLM generator failed with exit code {completed.returncode}. "
            "See its stderr above. Most common cause: missing ANTHROPIC_API_KEY "
            "or Ollama daemon not running."
        )

    rows: List[Dict[str, Any]] = []
    if not raw_path.exists():
        raise RuntimeError(
            f"Legacy generator produced no output at {raw_path}. "
            "Check the generation log for failures."
        )
    with raw_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                src = json.loads(line)
            except json.JSONDecodeError:
                continue
            atype = _LLM_TYPE_MAP.get(
                str(src.get("question_type") or "").lower(), "summary_plot"
            )
            row = {
                "id": "",  # set after final length is known
                "query": str(src.get("query") or "").strip(),
                "language": "ko",
                "expected_doc_ids": list(src.get("expected_doc_ids") or []),
                "expected_section_keywords": list(src.get("expected_keywords") or []),
                "answer_type": atype,
                "difficulty": DIFFICULTY_BY_TYPE.get(atype, "medium"),
                "tags": ["anime", "silver", "synthetic", atype, f"llm:{backend}"],
            }
            if not row["query"] or not row["expected_doc_ids"]:
                continue
            rows.append(row)
            if len(rows) >= target:
                break

    for i, row in enumerate(rows, start=1):
        row["id"] = f"anime-silver-llm-{i:04d}"
    return rows


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fp:
        for _ in fp:
            n += 1
    return n


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.harness.generate_eval_queries",
        description="Generate retrieval-eval queries from an anime corpus.",
    )
    parser.add_argument("--corpus", required=True, type=Path,
                        help="Path to the corpus JSONL.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL path.")
    parser.add_argument("--target", type=int, default=200,
                        help="Approximate number of rows to emit (default: 200).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic seed (default: 42).")
    parser.add_argument("--generator", choices=["deterministic", "llm", "claude-cli"],
                        default="deterministic",
                        help="Generator backend (default: deterministic).")
    parser.add_argument("--llm-backend", type=str, default="claude:haiku",
                        help="LLM provider passed to the legacy script when "
                             "--generator llm (default: claude:haiku). When "
                             "--generator claude-cli, this is the model alias "
                             "(e.g. opus, sonnet, haiku) passed to claude --model.")
    parser.add_argument("--claude-cli", type=str, default=None,
                        help="Path to the `claude` executable (default: $PATH lookup).")
    parser.add_argument("--cli-batch-size", type=int, default=5,
                        help="Docs per claude-cli batch call (default: 5).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="DEBUG logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    if args.generator == "deterministic":
        log.info("Loading corpus from %s", args.corpus)
        docs = list(_iter_corpus(args.corpus))
        log.info("Loaded %d docs; generating up to %d queries (seed=%d)",
                 len(docs), args.target, args.seed)
        rows = generate_deterministic(docs, target=args.target, seed=args.seed)
    elif args.generator == "claude-cli":
        from eval.harness.llm_subprocess_generator import generate_via_claude_cli

        log.info("Loading corpus from %s", args.corpus)
        # Re-use the corpus iterator but feed the raw dict back to the
        # CLI generator (it has its own excerpt projection that needs
        # the section dict structure).
        docs_raw: List[Dict[str, Any]] = []
        with args.corpus.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    docs_raw.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        log.info(
            "Loaded %d docs; generating up to %d queries via claude-cli (model=%s, seed=%d)",
            len(docs_raw), args.target, args.llm_backend, args.seed,
        )
        log_path = args.out.with_suffix(".cli_log.jsonl")
        rows = generate_via_claude_cli(
            docs_raw,
            target=args.target,
            model=args.llm_backend,
            seed=args.seed,
            claude_cli_path=args.claude_cli,
            log_path=log_path,
            docs_per_batch=int(args.cli_batch_size),
        )
        # Re-assign sequential ids in case generator returned sparse fills.
        for i, row in enumerate(rows, start=1):
            row["id"] = f"anime-silver-llm-{i:04d}"
    else:
        rows = generate_llm(
            corpus_path=args.corpus,
            target=args.target,
            out_path=args.out,
            backend=args.llm_backend,
            seed=args.seed,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d rows to %s", len(rows), args.out)

    # Per-type breakdown for the operator's eyes.
    by_type: Dict[str, int] = {}
    for row in rows:
        by_type[row["answer_type"]] = by_type.get(row["answer_type"], 0) + 1
    print()
    print(f"Generated {len(rows)} queries -> {args.out}")
    print("  per-type breakdown:")
    for atype in ANSWER_TYPES:
        n = by_type.get(atype, 0)
        pct = (n / len(rows) * 100) if rows else 0.0
        print(f"    {atype:25s} {n:4d}  ({pct:5.1f}%)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
