"""Eval-only cap policy abstraction for the rerank-input pool.

Extends the behaviour already encoded in
``eval.harness.wide_retrieval_helpers.apply_title_cap`` with named,
strategy-style policies that decide *what counts as a duplicate group*:

  - :class:`TitleCapPolicy` — title via ``DocTitleResolver``; doc_id
    fallback when title is missing. Same key as ``apply_title_cap`` so
    the existing wide-MMR sweep keeps reading consistent group keys.
  - :class:`DocIdCapPolicy` — strict doc_id grouping. Two chunks of the
    same doc_id collapse together; two doc_ids that share a title (e.g.
    multiple seasons of one anime franchise) do **not**.
  - :class:`NoCapPolicy` — passthrough; never drops anything. Used to
    verify whether any cap is helpful at all on a given dataset.
  - :class:`SectionPathCapPolicy` — group by ``(doc_id, section)``.
    Two chunks from the same section of the same doc collapse; two
    different sections of the same doc do not. Useful when the corpus
    has section-level structural diversity (e.g. ``요약`` vs ``본문``)
    that title cap collapses too aggressively.

The existing ``apply_title_cap(...)`` helper keeps working unchanged;
this module is additive. The ``WideRetrievalEvalAdapter`` accepts a
``cap_policy_rerank_input`` override on top of the legacy
``title_cap_rerank_input`` integer for backward compatibility — the
override wins when supplied.

All policies are pure Python and operate on chunk-like objects exposing
``doc_id`` / ``section`` (and use the supplied ``title_provider`` when
appropriate). No FAISS / embedder / reranker dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from eval.harness.wide_retrieval_helpers import TitleProvider


class CapPolicy(ABC):
    """Strategy interface: assign a group key, then cap the count.

    Subclasses set ``name`` (stable string used in CSV / JSON / report
    column headers) and ``cap`` (positive int for "at most N per group",
    None or 0 for no-op). They override :meth:`key_for` to define the
    grouping rule.

    The default :meth:`apply` walks the input list left-to-right, keeps
    the first ``cap`` chunks per non-empty key, and surfaces empty-key
    chunks unchanged at the tail. This mirrors ``apply_title_cap``'s
    contract so behaviour is identical for the legacy title-cap call.
    """

    # Stable label used by the report writer + CSV columns. Override
    # in subclasses; do NOT reuse strings — they are looked up by name
    # in the verdict logic.
    name: str = "cap_policy"
    cap: Optional[int] = None

    @abstractmethod
    def key_for(self, chunk: Any) -> str:
        """Return the cap-group key for ``chunk``.

        Empty string is the "no group" sentinel — chunks with an empty
        key bypass the cap and surface unchanged. This matches
        ``apply_title_cap``'s behaviour for chunks with neither a title
        nor a doc_id (typically test stubs / corrupt rows).
        """

    def apply(self, chunks: Sequence[Any]) -> List[Any]:
        """Return ``chunks`` filtered down to ``cap`` per group."""
        if self.cap is None or int(self.cap) <= 0:
            return list(chunks)
        cap_int = int(self.cap)
        counts: Dict[str, int] = {}
        out: List[Any] = []
        for chunk in chunks:
            key = self.key_for(chunk)
            if not key:
                out.append(chunk)
                continue
            used = counts.get(key, 0)
            if used >= cap_int:
                continue
            out.append(chunk)
            counts[key] = used + 1
        return out

    def group_sizes(self, chunks: Sequence[Any]) -> Dict[str, int]:
        """Diagnostic — count chunks per non-empty group key."""
        sizes: Dict[str, int] = {}
        for chunk in chunks:
            key = self.key_for(chunk)
            if not key:
                continue
            sizes[key] = sizes.get(key, 0) + 1
        return sizes

    def cap_out_records(
        self, chunks: Sequence[Any],
    ) -> Dict[str, List[Any]]:
        """Per-group "would have been dropped" listings.

        Used by the audit pass to answer "which chunks did this policy
        cap out?" without re-running the cap. Each value list is in
        input order; the first ``cap`` entries are the kept ones, the
        rest are the capped-out chunks.
        """
        if self.cap is None or int(self.cap) <= 0:
            return {}
        cap_int = int(self.cap)
        bucket: Dict[str, List[Any]] = {}
        dropped: Dict[str, List[Any]] = {}
        for chunk in chunks:
            key = self.key_for(chunk)
            if not key:
                continue
            seen = bucket.setdefault(key, [])
            if len(seen) < cap_int:
                seen.append(chunk)
            else:
                dropped.setdefault(key, []).append(chunk)
        return dropped

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "cap": self.cap}


# ---------------------------------------------------------------------------
# Concrete policies
# ---------------------------------------------------------------------------


class TitleCapPolicy(CapPolicy):
    """Cap by title (via title_provider) with doc_id fallback.

    Same grouping rule as ``apply_title_cap``: title takes precedence
    when the provider returns a non-empty string; doc_id fills in
    otherwise. Both keys are casefolded so case drift in raw titles
    doesn't split a single bucket into two.
    """

    name = "title"

    def __init__(
        self,
        cap: Optional[int],
        *,
        title_provider: Optional[TitleProvider] = None,
    ) -> None:
        self.cap = cap
        self.title_provider = title_provider

    def key_for(self, chunk: Any) -> str:
        if self.title_provider is not None:
            try:
                title = self.title_provider(chunk)
            except Exception:  # noqa: BLE001 — defensive
                title = None
            if title:
                t_str = str(title).strip().casefold()
                if t_str:
                    return t_str
        return str(getattr(chunk, "doc_id", "") or "").strip().casefold()

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cap": self.cap,
            "title_provider": (
                "supplied" if self.title_provider is not None else None
            ),
        }


class DocIdCapPolicy(CapPolicy):
    """Cap strictly by ``doc_id``.

    Stricter than :class:`TitleCapPolicy`: same-titled but distinct
    ``doc_id`` rows survive separately. Useful when the title axis
    collapses too many rows (e.g. a long-running franchise where each
    season is a separate doc) and the eval wants per-doc diversity
    rather than per-franchise diversity.
    """

    name = "doc_id"

    def __init__(self, cap: Optional[int]) -> None:
        self.cap = cap

    def key_for(self, chunk: Any) -> str:
        return str(getattr(chunk, "doc_id", "") or "").strip().casefold()


class NoCapPolicy(CapPolicy):
    """Passthrough policy — never drops anything.

    Implemented by overriding :meth:`apply` directly. Group sizes still
    work for diagnostic purposes (so the report can show "no cap, but
    this is the largest group").
    """

    name = "no_cap"

    def __init__(self) -> None:
        self.cap = None

    def key_for(self, chunk: Any) -> str:
        return str(getattr(chunk, "doc_id", "") or "").strip().casefold()

    def apply(self, chunks: Sequence[Any]) -> List[Any]:
        return list(chunks)

    def cap_out_records(
        self, chunks: Sequence[Any],
    ) -> Dict[str, List[Any]]:
        return {}


class SectionPathCapPolicy(CapPolicy):
    """Cap by ``(doc_id, section)`` composite key.

    Two chunks of the same section of the same document collapse
    together; two chunks of *different* sections in the same document
    do not. Provides a "soft" diversity signal for corpora whose chunks
    carry meaningful section structure (the anime corpus has ``요약`` /
    ``본문`` for example) and where a per-doc cap would over-collapse.
    """

    name = "section_path"

    def __init__(self, cap: Optional[int]) -> None:
        self.cap = cap

    def key_for(self, chunk: Any) -> str:
        doc = str(getattr(chunk, "doc_id", "") or "").strip().casefold()
        section = str(getattr(chunk, "section", "") or "").strip().casefold()
        if not doc and not section:
            return ""
        if not doc:
            return f"_|{section}"
        if not section:
            return doc
        return f"{doc}|{section}"


# ---------------------------------------------------------------------------
# Functional sugar
# ---------------------------------------------------------------------------


def apply_cap_policy(
    chunks: Sequence[Any], *, policy: CapPolicy,
) -> List[Any]:
    """Apply ``policy`` to ``chunks``; thin wrapper for callers that
    don't already hold a policy reference.
    """
    return policy.apply(chunks)
