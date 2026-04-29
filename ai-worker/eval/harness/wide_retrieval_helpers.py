"""Eval-only helpers for wide retrieval experiments.

Provides three pure helpers used by ``WideRetrievalEvalAdapter`` and the
``eval_wide_mmr_titlecap_sweep`` driver. None of these touch production
code:

  1. ``apply_title_cap`` — limit how many chunks per title (or doc_id
     fallback) survive into a downstream stage. Useful both for capping
     the rerank-input pool and for the final top-k.

  2. ``mmr_select_score_fallback`` — score + diversity-by-doc/title MMR
     selector. The production ``Retriever._mmr_select`` uses a
     doc_id-only diversity penalty over rerank/bi-encoder relevance;
     this helper extends the same idea to (a) optional title-level
     penalty when a title resolver is supplied, and (b) explicit
     ``lambda_val``, ``doc_id_penalty``, ``title_penalty`` knobs.
     This is a *score-fallback* MMR — it does not need candidate
     embeddings, only relevance scores and doc/title metadata. The
     fallback name is intentional so the report wording does not over-
     claim "vector-MMR" when the harness can't see embedding vectors.

  3. ``DocTitleResolver`` — small adapter that maps ``doc_id`` to a
     title string by reading the original corpus jsonl once and
     caching the lookup. Falls back to the doc_id when the corpus row
     has no title field, so callers can pass it through unconditionally
     without branching on "is title available?".

The helpers are deliberately small, deterministic, and pure-Python so
the unit tests can pin behaviour without needing FAISS / embedder /
reranker. The wider-retrieval adapter wires them together; the sweep
driver orchestrates a grid over (candidate_k × mmr_on × title_cap ×
rerank_in × final_top_k).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)


# Default doc_id penalty mirrors the production Retriever's
# ``_MMR_DOC_ID_PENALTY`` constant so an MMR run with the eval-only
# helper at the production lambda reproduces the same diversity
# trade-off the production retriever would have made.
DEFAULT_DOC_ID_PENALTY = 0.6
# Title penalty defaults below the doc penalty: same-title chunks
# from different doc_ids are still "the same anime" but less
# duplicative than chunks within one doc.
DEFAULT_TITLE_PENALTY = 0.4


TitleProvider = Callable[[Any], Optional[str]]
"""``(chunk_like) -> title or None`` — eval-only adapter contract.

Returns the chunk's title string when known, ``None`` to signal "no
title metadata for this chunk; fall back to doc_id." The caller is
responsible for falling back; the helpers below treat ``None`` as
"no title penalty / cap key" and continue.
"""


@dataclass
class DocTitleResolver:
    """Lazy doc_id -> title map built from a corpus jsonl.

    Reads each row of the corpus, pulls ``title`` (falling back to
    ``seed`` then to ``doc_id``), and indexes by the same ``doc_id``
    rule the offline corpus builder uses (see
    ``eval.harness.offline_corpus``: ``doc_id or seed or title``).
    The map is built once and cached on the instance.

    Title fallback chain:
      1. exact map hit
      2. raw ``doc_id`` returned (so MMR/title cap still groups
         things; just doesn't get the *titled* grouping)

    The resolver is not strictly required — title cap can run on
    doc_id alone — but supplying it lets MMR / title cap collapse
    multi-doc anime franchises (e.g. seasons of the same series)
    onto a single penalty key when the dataset author intended that.
    """

    corpus_path: Path
    _cache: Dict[str, str]

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = Path(corpus_path)
        self._cache = {}
        self._loaded = False

    @classmethod
    def from_corpus(cls, corpus_path: Path) -> "DocTitleResolver":
        return cls(Path(corpus_path))

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.corpus_path.exists():
            return
        with self.corpus_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                doc_id = str(
                    raw.get("doc_id")
                    or raw.get("seed")
                    or raw.get("title")
                    or ""
                ).strip()
                if not doc_id:
                    continue
                title = raw.get("title")
                if title is None:
                    continue
                self._cache[doc_id] = str(title).strip()

    def title_for_doc(self, doc_id: str) -> Optional[str]:
        """Return the title for ``doc_id`` or ``None`` if unknown."""
        self._ensure_loaded()
        if not doc_id:
            return None
        return self._cache.get(str(doc_id))

    def title_provider(self) -> TitleProvider:
        """Closure form for handing into ``apply_title_cap`` / MMR."""

        def _provider(chunk: Any) -> Optional[str]:
            doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
            return self.title_for_doc(doc_id)

        return _provider


def _key_for_chunk(
    chunk: Any,
    *,
    title_provider: Optional[TitleProvider],
) -> str:
    """Resolve the cap / penalty key for ``chunk``.

    Preference: title (when title_provider supplied AND it returned a
    non-empty string) > doc_id. Empty / unknown both fall back to
    doc_id, and an empty doc_id stays empty (the caller treats empty
    keys as "no group"). Returns a normalized lowercase string so
    casing drift in raw titles doesn't split "TITLE" and "title" into
    two cap buckets.
    """
    if title_provider is not None:
        try:
            t = title_provider(chunk)
        except Exception:
            t = None
        if t:
            t_str = str(t).strip().casefold()
            if t_str:
                return t_str
    doc_id = str(getattr(chunk, "doc_id", "") or "").strip().casefold()
    return doc_id


def apply_title_cap(
    chunks: Sequence[Any],
    *,
    cap: Optional[int],
    title_provider: Optional[TitleProvider] = None,
) -> List[Any]:
    """Limit the number of chunks per title (doc_id fallback) to ``cap``.

    Order is preserved: the first ``cap`` chunks per key win. Chunks
    with an empty key (no doc_id and no title) bypass the cap so the
    caller doesn't accidentally drop everything when metadata is
    missing — those are surfaced unchanged at the tail. ``cap=None``
    or ``cap <= 0`` is a no-op, returning a fresh list copy.

    Use cases:
      - cap the rerank-input pool so the cross-encoder doesn't burn
        cycles on 30 near-duplicate chunks of one franchise season.
      - cap the final top-k for diversity (cap=1: at most one chunk
        per title; cap=2: balance between diversity and depth).
    """
    if cap is None or int(cap) <= 0:
        return list(chunks)
    cap_int = int(cap)
    counts: Dict[str, int] = {}
    out: List[Any] = []
    for chunk in chunks:
        key = _key_for_chunk(chunk, title_provider=title_provider)
        if not key:
            out.append(chunk)
            continue
        used = counts.get(key, 0)
        if used >= cap_int:
            continue
        out.append(chunk)
        counts[key] = used + 1
    return out


def _relevance_score(chunk: Any) -> float:
    """Pick the live ranking signal for ``chunk``.

    Mirrors the production Retriever helper: prefer ``rerank_score``
    when populated; fall back to bi-encoder ``score``; final fallback
    is 0.0 so a candidate without either signal still has a stable
    (deterministic) seat in the MMR ordering.
    """
    rs = getattr(chunk, "rerank_score", None)
    if rs is not None:
        try:
            return float(rs)
        except (TypeError, ValueError):
            pass
    s = getattr(chunk, "score", None)
    if s is None:
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def mmr_select_score_fallback(
    candidates: Sequence[Any],
    *,
    top_k: int,
    lambda_val: float,
    doc_id_penalty: float = DEFAULT_DOC_ID_PENALTY,
    title_penalty: float = DEFAULT_TITLE_PENALTY,
    title_provider: Optional[TitleProvider] = None,
) -> List[Any]:
    """Score-only MMR selector with doc_id + title diversity penalties.

    For each un-selected candidate compute::

        value = lambda_val * relevance
              - (1 - lambda_val) * max(doc_penalty, title_penalty)

    where:
      - relevance = chunk.rerank_score if present else chunk.score
      - doc_penalty = ``doc_id_penalty`` when chunk.doc_id is in the
        already-selected doc_id set, else 0.0
      - title_penalty = ``title_penalty`` when chunk's title (resolved
        via ``title_provider``) is in the already-selected title set,
        else 0.0; skipped entirely when ``title_provider`` is None

    The penalty term is the *max* of the two so a chunk that shares
    BOTH the doc and the title with a selected one is penalized at the
    same magnitude as one that only shares one — the goal is "don't
    pile chunks of one franchise"; a single shared signal is enough.

    Determinism: ties on the value score keep the input list's order
    (we walk the list left-to-right and take the first max each pass).
    Pure-Python O(top_k * len(candidates)); candidate lists are bounded
    so the quadratic factor is fine.

    Why "score_fallback": this helper does NOT use candidate embedding
    vectors. It is a *fallback* MMR usable when the eval harness
    cannot reach into FAISS and re-compute pairwise similarity. The
    Chroma-era prior strategy used cosine-similarity-based MMR; this
    version trades that for relevance + categorical-overlap penalty,
    which is sufficient to surface the *qualitative* effect the spec
    asks for (does diversity help? does title-level penalty help?).
    """
    k = max(0, int(top_k))
    if k == 0 or not candidates:
        return []
    lam = max(0.0, min(1.0, float(lambda_val)))
    doc_pen = max(0.0, float(doc_id_penalty))
    title_pen = max(0.0, float(title_penalty))

    remaining: List[Any] = list(candidates)
    selected: List[Any] = []
    selected_doc_ids: set = set()
    selected_titles: set = set()

    while remaining and len(selected) < k:
        best_idx = -1
        best_value = float("-inf")
        for i, cand in enumerate(remaining):
            relevance = _relevance_score(cand)
            doc_id = str(getattr(cand, "doc_id", "") or "").strip().casefold()
            title_key: Optional[str] = None
            if title_provider is not None:
                try:
                    t_raw = title_provider(cand)
                except Exception:
                    t_raw = None
                if t_raw:
                    title_key = str(t_raw).strip().casefold() or None
            this_doc_pen = doc_pen if (doc_id and doc_id in selected_doc_ids) else 0.0
            this_title_pen = (
                title_pen if (title_key and title_key in selected_titles) else 0.0
            )
            penalty = max(this_doc_pen, this_title_pen)
            value = lam * relevance - (1.0 - lam) * penalty
            if value > best_value:
                best_value = value
                best_idx = i
        if best_idx < 0:
            break
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        chosen_doc = str(getattr(chosen, "doc_id", "") or "").strip().casefold()
        if chosen_doc:
            selected_doc_ids.add(chosen_doc)
        if title_provider is not None:
            try:
                t_raw = title_provider(chosen)
            except Exception:
                t_raw = None
            if t_raw:
                t_key = str(t_raw).strip().casefold()
                if t_key:
                    selected_titles.add(t_key)
    return selected


def count_keys(
    chunks: Iterable[Any],
    *,
    title_provider: Optional[TitleProvider] = None,
) -> Dict[str, int]:
    """Count chunks per cap-key — diagnostic helper for tests / reports."""
    counts: Dict[str, int] = {}
    for chunk in chunks:
        key = _key_for_chunk(chunk, title_provider=title_provider)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts
