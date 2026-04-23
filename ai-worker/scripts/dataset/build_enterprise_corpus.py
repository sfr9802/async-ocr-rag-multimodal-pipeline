"""Build a synthetic Korean enterprise-document corpus via Claude Sonnet 4.6.

This is Phase 9b's counterpart to Phase 9a's anime corpus. It generates
N Korean internal-policy documents per category across HR / finance /
IT / product / legal and lays them down in the same ingest-ready shape
as ``anime_corpus_kr.jsonl`` so the existing RAG ingest consumes both
without branching.

Usage (from ai-worker/)::

    python -m scripts.dataset.build_enterprise_corpus \\
        --out fixtures/enterprise_corpus_kr/ \\
        --categories hr,finance,it,product,legal \\
        --per-category 25 --seed 42 \\
        --generator claude:sonnet-4-6

Design notes
------------
* **Prompts live as committed files** under
  ``scripts/dataset/prompts/enterprise/<category>.md`` so the tone of
  each category is reviewable in git. The script substitutes
  ``{doc_id}`` / ``{seed}`` / ``{min_chars}`` / ``{max_chars}`` into
  the template at call time.
* **Resumable.** Each doc is a standalone JSON file on disk. A second
  run re-reads existing files, rebuilds ``index.jsonl`` from them, and
  only calls Claude for gaps.
* **Diversity guard.** After generating a category's missing docs,
  compute pairwise bge-m3 cosine across all docs in the category. Any
  pair with cosine > 0.88 → the younger doc (higher index or later
  ``generated_ts``) is queued for regeneration with a perturbed seed.
  Up to 3 regeneration passes per category; residuals are logged as
  unresolved-duplicates, not silently ignored.
* **Audit trail.** Every Claude call (success OR failure OR validation
  rejection) appends a row to ``generation_log.jsonl`` alongside the
  corpus. Token counts and latency are included so the committed log
  doubles as a cost ledger.
* **Budget.** 25MB total committed size is the ceiling; exceeding it
  emits a warning but the script still finishes.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.dataset._common import (
    ClaudeResponseError,
    GenerationLog,
    RateLimiter,
    ResumableJsonlWriter,
    claude_json_call,
    configure_logging,
    load_anthropic_client,
    log_call,
    stable_seed,
)

log = logging.getLogger("scripts.dataset.build_enterprise_corpus")


_SUPPORTED_CATEGORIES = ("hr", "finance", "it", "product", "legal")
_SIZE_BUDGET_MB = 25
_DIVERSITY_THRESHOLD = 0.88
_MAX_DIVERSITY_PASSES = 3

_SYSTEM_PROMPT = (
    "You are a professional Korean enterprise-document writer producing "
    "realistic internal policy / spec / runbook documents for an IT "
    "company headquartered in Seoul. Every document must feel like it "
    "was written by a Korean corporate author: proper honorific register "
    "(합니다/습니다), RFC-style numbered clauses, concrete figures "
    "(dates, amounts, retention periods, SLAs), named roles (팀장, CFO, "
    "CISO, 법무팀, 재무팀), and cross-references to related documents.\n\n"
    "Return a single JSON object matching the schema the user describes. "
    "Do NOT wrap the JSON in Markdown code fences. Do NOT add any prose "
    "before or after the JSON."
)

# Claude Sonnet 4.6 list pricing (USD per 1M tokens). Used only for a
# best-effort cost estimate in the final log — no rate-enforcement.
_CLAUDE_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


# ---------------------------------------------------------------------------
# Prompt + doc id helpers
# ---------------------------------------------------------------------------


def _prompt_path(category: str) -> Path:
    return (
        Path(__file__).resolve().parent
        / "prompts"
        / "enterprise"
        / f"{category}.md"
    )


def _load_prompt(category: str) -> str:
    path = _prompt_path(category)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing enterprise prompt for category={category!r}: {path}"
        )
    return path.read_text(encoding="utf-8")


def _render_prompt(
    template: str,
    *,
    doc_id: str,
    seed: int,
    min_chars: int,
    max_chars: int,
) -> str:
    # Use a narrow keyword whitelist so unrelated ``{...}`` literals
    # inside the markdown (code examples, Korean braces) don't explode.
    for key, value in (
        ("{doc_id}", doc_id),
        ("{seed}", str(seed)),
        ("{min_chars}", str(min_chars)),
        ("{max_chars}", str(max_chars)),
    ):
        template = template.replace(key, value)
    return template


def _doc_id_for(category: str, index: int) -> str:
    return f"kr-{category}-{index:03d}"


def _category_dir(out_dir: Path, category: str) -> Path:
    return out_dir / category


def _doc_path(out_dir: Path, category: str, doc_id: str) -> Path:
    return _category_dir(out_dir, category) / f"{doc_id}.json"


def resolve_generator(generator: str) -> Tuple[str, str]:
    """Parse a ``--generator claude:sonnet-4-6`` into ``(provider, model_id)``.

    Accepts plain model ids (``claude-sonnet-4-6``) unchanged. The
    provider half is purely informational today — only ``claude`` is
    supported.
    """
    if ":" in generator:
        provider, suffix = generator.split(":", 1)
    else:
        provider, suffix = "claude", generator
    if provider not in ("claude", "anthropic"):
        raise ValueError(f"Unsupported generator provider: {provider!r}")
    aliases = {
        "sonnet": "claude-sonnet-4-6",
        "sonnet-4-6": "claude-sonnet-4-6",
        "opus": "claude-opus-4-7",
        "opus-4-7": "claude-opus-4-7",
        "haiku": "claude-haiku-4-5-20251001",
        "haiku-4-5": "claude-haiku-4-5-20251001",
    }
    if suffix in aliases:
        return "claude", aliases[suffix]
    if suffix.startswith("claude-"):
        return "claude", suffix
    return "claude", f"claude-{suffix}"


# ---------------------------------------------------------------------------
# Schema validation + index row shaping
# ---------------------------------------------------------------------------


_VALID_DOC_ID = re.compile(r"^kr-[a-z]+-\d{3,}$")


def _validate_doc(
    raw: Dict[str, Any],
    *,
    expected_doc_id: str,
    min_chars: int,
    max_chars: int,
) -> Dict[str, Any]:
    for field in ("doc_id", "title", "sections"):
        if field not in raw:
            raise ClaudeResponseError(f"Missing required field: {field}")
    if str(raw["doc_id"]).strip() != expected_doc_id:
        raise ClaudeResponseError(
            f"doc_id mismatch: expected {expected_doc_id!r}, got {raw['doc_id']!r}"
        )

    sections = raw["sections"]
    if not isinstance(sections, list) or not sections:
        raise ClaudeResponseError("'sections' must be a non-empty list.")
    if not (3 <= len(sections) <= 6):
        raise ClaudeResponseError(f"Expected 3-6 sections, got {len(sections)}")

    cleaned_sections: List[Dict[str, str]] = []
    total_chars = 0
    for i, section in enumerate(sections):
        if not isinstance(section, dict):
            raise ClaudeResponseError(f"section {i} is not an object.")
        heading = str(section.get("heading", "")).strip()
        text = str(section.get("text", "")).strip()
        if not heading or not text:
            raise ClaudeResponseError(f"section {i} missing heading or text.")
        total_chars += len(text)
        cleaned_sections.append({"heading": heading, "text": text})

    lower_bound = max(120, min_chars // 2)
    if total_chars < lower_bound:
        raise ClaudeResponseError(
            f"Body too short: {total_chars} chars (expected >= {lower_bound})"
        )
    if total_chars > max_chars * 2:
        log.warning(
            "Doc %s body length %d exceeds max_chars*2=%d — accepting.",
            expected_doc_id, total_chars, max_chars * 2,
        )

    exceptions = raw.get("exception_clauses") or []
    if not isinstance(exceptions, list):
        exceptions = []
    exceptions = [str(e).strip() for e in exceptions if str(e).strip()]

    related = raw.get("related_docs") or []
    if not isinstance(related, list):
        related = []
    related = [str(r).strip() for r in related if str(r).strip()]

    return {
        "doc_id": expected_doc_id,
        "title": str(raw["title"]).strip()[:200],
        "sections": cleaned_sections,
        "exception_clauses": exceptions,
        "related_docs": related,
    }


def _to_index_row(
    doc: Dict[str, Any],
    *,
    category: str,
    seed: int,
    generated_ts: str,
    source_label: str,
) -> Dict[str, Any]:
    """Shape a validated doc into the ingest-ready index.jsonl row.

    The ingest reads ``sections`` as a dict keyed by section name. We
    disambiguate duplicate Korean headings with a ``#N`` suffix so the
    dict keys stay unique without losing author intent.
    """
    sections_map: Dict[str, Dict[str, str]] = {}
    section_order: List[str] = []
    for item in doc["sections"]:
        heading = item["heading"]
        key = heading
        suffix = 2
        while key in sections_map:
            key = f"{heading}#{suffix}"
            suffix += 1
        sections_map[key] = {"text": item["text"]}
        section_order.append(key)
    return {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "sections": sections_map,
        "section_order": section_order,
        "domain": "enterprise",
        "category": category,
        "language": "ko",
        "source": source_label,
        "generated_ts": generated_ts,
        "seed": seed,
        "exception_clauses": doc.get("exception_clauses", []),
        "related_docs": doc.get("related_docs", []),
    }


# ---------------------------------------------------------------------------
# Diversity guard
# ---------------------------------------------------------------------------


@dataclass
class _DuplicatePair:
    """A cosine-over-threshold pair inside a single category."""

    category: str
    doc_a: str
    doc_b: str
    cosine: float
    younger: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "doc_a": self.doc_a,
            "doc_b": self.doc_b,
            "cosine": round(self.cosine, 4),
            "younger": self.younger,
        }


def _doc_embedding_text(doc: Dict[str, Any]) -> str:
    # Concatenate title + all section texts. Headings are repeated as
    # weak signal (they often carry the topic name).
    parts: List[str] = [str(doc.get("title", ""))]
    for s in doc.get("sections", []):
        parts.append(str(s.get("heading", "")))
        parts.append(str(s.get("text", "")))
    return "\n".join(p for p in parts if p)


def _build_embedder(model_name: str) -> Any:
    # Local import so callers who pass --skip-diversity never have to
    # install sentence-transformers.
    from app.capabilities.rag.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_name=model_name)


def _pairwise_cosine(vectors) -> Any:
    # sentence-transformers already normalizes, so matmul == cosine.
    import numpy as np

    if vectors.shape[0] < 2:
        return np.zeros((vectors.shape[0], vectors.shape[0]), dtype="float32")
    return (vectors @ vectors.T).astype("float32", copy=False)


def find_duplicate_pairs(
    docs: List[Dict[str, Any]],
    *,
    category: str,
    threshold: float,
    embedder: Any,
) -> List[_DuplicatePair]:
    """Return pairwise cosine > threshold inside one category.

    Younger-side identification uses doc index (higher idx = younger)
    with ``generated_ts`` as a tiebreaker. A doc whose creation order is
    clearly older is preferred as the keeper.
    """
    if len(docs) < 2:
        return []
    order_by_doc_id = {doc["doc_id"]: i for i, doc in enumerate(docs)}
    texts = [_doc_embedding_text(d) for d in docs]
    vectors = embedder.embed_passages(texts)
    cos = _pairwise_cosine(vectors)
    pairs: List[_DuplicatePair] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            c = float(cos[i, j])
            if c > threshold:
                a_id, b_id = docs[i]["doc_id"], docs[j]["doc_id"]
                a_ts = str(docs[i].get("generated_ts") or "")
                b_ts = str(docs[j].get("generated_ts") or "")
                if order_by_doc_id[a_id] == order_by_doc_id[b_id]:
                    younger = a_id if a_ts >= b_ts else b_id
                else:
                    younger = (
                        a_id if order_by_doc_id[a_id] > order_by_doc_id[b_id] else b_id
                    )
                pairs.append(
                    _DuplicatePair(
                        category=category, doc_a=a_id, doc_b=b_id,
                        cosine=c, younger=younger,
                    )
                )
    return pairs


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def _read_existing_doc(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as ex:  # noqa: BLE001
        log.warning("Corrupt doc on disk — will regenerate: %s (%s)", path, ex)
        return None


def _write_doc_json(out_dir: Path, category: str, payload: Dict[str, Any]) -> Path:
    target = _doc_path(out_dir, category, payload["doc_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def _collect_category_docs(out_dir: Path, category: str) -> List[Dict[str, Any]]:
    cat_dir = _category_dir(out_dir, category)
    if not cat_dir.exists():
        return []
    docs: List[Dict[str, Any]] = []
    for path in sorted(cat_dir.glob("kr-*.json")):
        existing = _read_existing_doc(path)
        if existing is None:
            continue
        docs.append(existing)
    return docs


def _total_bytes(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    total = 0
    for path in out_dir.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


# ---------------------------------------------------------------------------
# Doc generation
# ---------------------------------------------------------------------------


@dataclass
class CostBook:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    failures: int = 0

    def add(self, prompt: Optional[int], completion: Optional[int]) -> None:
        if prompt:
            self.prompt_tokens += int(prompt)
        if completion:
            self.completion_tokens += int(completion)
        self.calls += 1

    def estimated_usd(self, model: str) -> float:
        rates = _CLAUDE_PRICING.get(model)
        if rates is None:
            return 0.0
        in_rate, out_rate = rates
        return (
            self.prompt_tokens * in_rate + self.completion_tokens * out_rate
        ) / 1_000_000.0


def _generate_one_doc(
    *,
    client: Any,
    model: str,
    template: str,
    doc_id: str,
    seed: int,
    min_chars: int,
    max_chars: int,
    gen_log: GenerationLog,
    costs: CostBook,
    max_retries: int,
    script_label: str,
) -> Optional[Dict[str, Any]]:
    """Call Claude and validate. Returns the validated doc dict or None."""
    for attempt in range(1, max_retries + 1):
        user_prompt = _render_prompt(
            template,
            doc_id=doc_id,
            seed=seed + (attempt - 1) * 7919,  # perturb on retry
            min_chars=min_chars,
            max_chars=max_chars,
        )
        try:
            with log_call(
                gen_log,
                script=script_label,
                provider="claude",
                model=model,
                seed=seed,
                note=f"{doc_id} attempt={attempt}",
            ) as slot:
                parsed = claude_json_call(
                    client,
                    model=model,
                    system=_SYSTEM_PROMPT,
                    user=user_prompt,
                    max_tokens=2048,
                    temperature=0.75,
                )
                usage = parsed.pop("_usage", {})
                slot["prompt_tokens"] = usage.get("input_tokens")
                slot["completion_tokens"] = usage.get("output_tokens")
            doc = _validate_doc(
                parsed,
                expected_doc_id=doc_id,
                min_chars=min_chars,
                max_chars=max_chars,
            )
            costs.add(usage.get("input_tokens"), usage.get("output_tokens"))
            return doc
        except (ClaudeResponseError, Exception) as ex:  # noqa: BLE001
            log.warning(
                "Generation for %s attempt %d/%d failed: %s: %s",
                doc_id, attempt, max_retries, type(ex).__name__, ex,
            )
    costs.failures += 1
    return None


def _build_category_pass(
    *,
    client: Any,
    model: str,
    category: str,
    per_category: int,
    min_chars: int,
    max_chars: int,
    out_dir: Path,
    limiter: RateLimiter,
    gen_log: GenerationLog,
    costs: CostBook,
    max_retries: int,
    source_label: str,
    index_writer: ResumableJsonlWriter,
    dry_run: bool,
) -> Tuple[int, int]:
    """Generate all missing docs for one category. Returns ``(new, existing)``."""
    template = _load_prompt(category)
    new_count = 0
    existing_count = 0
    for idx in range(1, per_category + 1):
        doc_id = _doc_id_for(category, idx)
        seed = stable_seed(category, idx)
        doc_path = _doc_path(out_dir, category, doc_id)
        if doc_path.exists():
            existing = _read_existing_doc(doc_path)
            if existing is not None:
                if not index_writer.has(doc_id):
                    row = _to_index_row(
                        {
                            "doc_id": existing["doc_id"],
                            "title": existing["title"],
                            "sections": existing["sections"],
                            "exception_clauses": existing.get("exception_clauses", []),
                            "related_docs": existing.get("related_docs", []),
                        },
                        category=category,
                        seed=existing.get("seed", seed),
                        generated_ts=existing.get("generated_ts")
                            or datetime.now(timezone.utc).isoformat(),
                        source_label=existing.get("source", source_label),
                    )
                    index_writer.append(row)
                existing_count += 1
                log.info("[skip] %s already on disk", doc_id)
                continue
        if dry_run:
            log.info("[dry-run] would generate %s (seed=%d)", doc_id, seed)
            continue
        if client is None:
            raise RuntimeError("Anthropic client not initialised.")
        limiter.wait()
        doc = _generate_one_doc(
            client=client,
            model=model,
            template=template,
            doc_id=doc_id,
            seed=seed,
            min_chars=min_chars,
            max_chars=max_chars,
            gen_log=gen_log,
            costs=costs,
            max_retries=max_retries,
            script_label="build_enterprise_corpus",
        )
        if doc is None:
            continue
        generated_ts = datetime.now(timezone.utc).isoformat()
        disk_payload = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "category": category,
            "domain": "enterprise",
            "language": "ko",
            "sections": doc["sections"],
            "exception_clauses": doc["exception_clauses"],
            "related_docs": doc["related_docs"],
            "seed": seed,
            "generated_ts": generated_ts,
            "source": source_label,
            "generator_model": model,
        }
        _write_doc_json(out_dir, category, disk_payload)
        row = _to_index_row(
            doc,
            category=category,
            seed=seed,
            generated_ts=generated_ts,
            source_label=source_label,
        )
        index_writer.append(row)
        new_count += 1
        log.info("[new] %s (%d sections)", doc_id, len(doc["sections"]))
    return new_count, existing_count


def _regenerate_duplicates(
    *,
    client: Any,
    model: str,
    pairs: List[_DuplicatePair],
    category: str,
    pass_number: int,
    per_category: int,
    min_chars: int,
    max_chars: int,
    out_dir: Path,
    limiter: RateLimiter,
    gen_log: GenerationLog,
    costs: CostBook,
    max_retries: int,
    source_label: str,
    index_writer: ResumableJsonlWriter,
) -> int:
    """Regenerate the younger doc from each duplicate pair. Returns regen count.

    We delete the younger file (and drop it from index.jsonl by rewriting)
    and call Claude again with a seed-perturbed prompt. The rewritten
    row replaces the old index entry.
    """
    template = _load_prompt(category)
    youngers = {p.younger for p in pairs}
    regenerated = 0
    for younger_id in sorted(youngers):
        m = re.match(r"kr-([a-z]+)-(\d+)$", younger_id)
        if not m:
            log.warning("Skipping regen of malformed doc_id %s", younger_id)
            continue
        idx = int(m.group(2))
        if idx > per_category:
            continue
        seed = stable_seed(category, idx) + pass_number * 10_000
        doc_path = _doc_path(out_dir, category, younger_id)
        if doc_path.exists():
            try:
                doc_path.unlink()
            except OSError as ex:
                log.warning("Could not unlink %s: %s", doc_path, ex)
                continue
        index_writer.drop(younger_id)
        limiter.wait()
        doc = _generate_one_doc(
            client=client,
            model=model,
            template=template,
            doc_id=younger_id,
            seed=seed,
            min_chars=min_chars,
            max_chars=max_chars,
            gen_log=gen_log,
            costs=costs,
            max_retries=max_retries,
            script_label="build_enterprise_corpus.diversity",
        )
        if doc is None:
            log.warning("Regen failed for %s (pass %d)", younger_id, pass_number)
            continue
        generated_ts = datetime.now(timezone.utc).isoformat()
        disk_payload = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "category": category,
            "domain": "enterprise",
            "language": "ko",
            "sections": doc["sections"],
            "exception_clauses": doc["exception_clauses"],
            "related_docs": doc["related_docs"],
            "seed": seed,
            "generated_ts": generated_ts,
            "source": source_label,
            "generator_model": model,
            "regen_pass": pass_number,
        }
        _write_doc_json(out_dir, category, disk_payload)
        row = _to_index_row(
            doc,
            category=category,
            seed=seed,
            generated_ts=generated_ts,
            source_label=source_label,
        )
        index_writer.append(row)
        regenerated += 1
        log.info("[regen pass=%d] %s", pass_number, younger_id)
    return regenerated


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def build_enterprise_corpus(
    *,
    out_dir: Path,
    categories: List[str],
    per_category: int,
    generator: str,
    min_chars: int,
    max_chars: int,
    rate_per_sec: float,
    max_retries: int,
    embed_model: str,
    diversity_threshold: float,
    diversity_passes: int,
    skip_diversity: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_log = GenerationLog(out_dir / "generation_log.jsonl")
    index_writer = ResumableJsonlWriter(
        out_dir / "index.jsonl", key_fn=lambda row: str(row.get("doc_id", ""))
    )
    limiter = RateLimiter(rate_per_sec)
    provider, model = resolve_generator(generator)
    source_label = f"synthesized-by-{model}"

    client = None if dry_run else load_anthropic_client()
    costs = CostBook()

    totals_new = 0
    totals_existing = 0
    unresolved_dupes: List[_DuplicatePair] = []

    embedder = None
    if not skip_diversity and not dry_run:
        try:
            embedder = _build_embedder(embed_model)
        except Exception as ex:  # noqa: BLE001
            log.warning("Embedder unavailable (%s) — diversity guard disabled.", ex)
            embedder = None

    for category in categories:
        if category not in _SUPPORTED_CATEGORIES:
            log.error("Unknown category %r — skipping", category)
            continue

        new_count, existing_count = _build_category_pass(
            client=client,
            model=model,
            category=category,
            per_category=per_category,
            min_chars=min_chars,
            max_chars=max_chars,
            out_dir=out_dir,
            limiter=limiter,
            gen_log=gen_log,
            costs=costs,
            max_retries=max_retries,
            source_label=source_label,
            index_writer=index_writer,
            dry_run=dry_run,
        )
        totals_new += new_count
        totals_existing += existing_count

        if embedder is None or dry_run:
            continue

        docs = _collect_category_docs(out_dir, category)
        if len(docs) < 2:
            continue

        for pass_number in range(1, diversity_passes + 1):
            pairs = find_duplicate_pairs(
                docs,
                category=category,
                threshold=diversity_threshold,
                embedder=embedder,
            )
            if not pairs:
                log.info("[diversity] %s pass=%d clean", category, pass_number)
                break
            log.info(
                "[diversity] %s pass=%d found %d pair(s) over %.2f",
                category, pass_number, len(pairs), diversity_threshold,
            )
            regenerated = _regenerate_duplicates(
                client=client,
                model=model,
                pairs=pairs,
                category=category,
                pass_number=pass_number,
                per_category=per_category,
                min_chars=min_chars,
                max_chars=max_chars,
                out_dir=out_dir,
                limiter=limiter,
                gen_log=gen_log,
                costs=costs,
                max_retries=max_retries,
                source_label=source_label,
                index_writer=index_writer,
            )
            if regenerated == 0:
                log.warning(
                    "[diversity] %s pass=%d: no regens succeeded, stopping",
                    category, pass_number,
                )
                unresolved_dupes.extend(pairs)
                break
            docs = _collect_category_docs(out_dir, category)
        else:
            # 3 passes exhausted; record the residual
            pairs = find_duplicate_pairs(
                docs,
                category=category,
                threshold=diversity_threshold,
                embedder=embedder,
            )
            if pairs:
                log.warning(
                    "[diversity] %s: %d residual duplicate pair(s) after %d passes",
                    category, len(pairs), diversity_passes,
                )
                unresolved_dupes.extend(pairs)

    total_bytes = _total_bytes(out_dir)
    size_mb = total_bytes / (1024 * 1024)
    if size_mb > _SIZE_BUDGET_MB:
        log.warning(
            "Corpus size %.2f MB exceeds budget %d MB",
            size_mb, _SIZE_BUDGET_MB,
        )

    cost_usd = costs.estimated_usd(model)
    summary = {
        "docs_new": totals_new,
        "docs_existing": totals_existing,
        "docs_total": totals_new + totals_existing,
        "model": model,
        "provider": provider,
        "prompt_tokens": costs.prompt_tokens,
        "completion_tokens": costs.completion_tokens,
        "claude_calls": costs.calls,
        "claude_failures": costs.failures,
        "estimated_cost_usd": round(cost_usd, 4),
        "size_mb": round(size_mb, 3),
        "size_over_budget": size_mb > _SIZE_BUDGET_MB,
        "unresolved_duplicate_pairs": [p.to_json() for p in unresolved_dupes],
    }
    (out_dir / "corpus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(
        "Corpus build done: new=%d existing=%d failures=%d cost≈$%.3f size=%.2fMB",
        totals_new, totals_existing, costs.failures, cost_usd, size_mb,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_categories(raw: str) -> List[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def _default_out_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "fixtures"
        / "enterprise_corpus_kr"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=_default_out_dir())
    parser.add_argument(
        "--categories", type=str, default="hr,finance,it,product,legal",
    )
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed base (fed into stable_seed per doc)")
    parser.add_argument("--generator", type=str, default="claude:sonnet-4-6")
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--rate-per-sec", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-m3",
                        help="Model used for the diversity cosine check")
    parser.add_argument("--diversity-threshold", type=float,
                        default=_DIVERSITY_THRESHOLD)
    parser.add_argument("--diversity-passes", type=int,
                        default=_MAX_DIVERSITY_PASSES)
    parser.add_argument("--skip-diversity", action="store_true",
                        help="Skip the bge-m3 pairwise-cosine check")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    categories = _parse_categories(args.categories)
    if not categories:
        log.error("No categories supplied.")
        return 2

    try:
        summary = build_enterprise_corpus(
            out_dir=args.out,
            categories=categories,
            per_category=args.per_category,
            generator=args.generator,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            rate_per_sec=args.rate_per_sec,
            max_retries=args.max_retries,
            embed_model=args.embed_model,
            diversity_threshold=args.diversity_threshold,
            diversity_passes=args.diversity_passes,
            skip_diversity=args.skip_diversity,
            dry_run=args.dry_run,
        )
    except Exception as ex:  # noqa: BLE001
        log.exception("Enterprise corpus build failed: %s", ex)
        return 1

    target = len(categories) * args.per_category
    if summary["docs_total"] < target * 0.5 and not args.dry_run:
        log.warning(
            "Only %d docs on disk (target %d) — check generation_log.jsonl",
            summary["docs_total"], target,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
