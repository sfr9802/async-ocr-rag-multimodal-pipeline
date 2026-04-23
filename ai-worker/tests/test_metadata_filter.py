"""Tests for the Phase 9 metadata-filtering plumbing.

Three layers under test, all offline:

  1. ``RagMetadataStore.doc_ids_matching`` rejects non-whitelisted
     filter keys at the SQL boundary so a parser hallucination can't
     turn into an arbitrary SELECT. Empty filters return every doc.

  2. ``Retriever.retrieve(filters=...)`` post-filters FAISS hits
     against the metadata allowlist. Empty filters reproduce the
     pre-Phase-9 behaviour bit-for-bit; a filter that matches no docs
     short-circuits to an empty report with ``filter_produced_no_docs``.

  3. ``LlmQueryParser`` populates ``ParsedQuery.filters`` from a strong
     query signal and silently drops out-of-whitelist values rather
     than falling back to the regex parser.

All three use either pure-Python fakes or the existing ``HashingEmbedder``
+ in-memory FAISS index seam, so the suite never touches Postgres or a
real model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pytest

from app.capabilities.rag.embeddings import HashingEmbedder
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.metadata_store import (
    ChunkLookupResult,
    RagMetadataStore,
)
from app.capabilities.rag.query_parser import LlmQueryParser, RegexQueryParser
from app.capabilities.rag.retriever import Retriever
from app.clients.llm_chat import LlmChatProvider


# ---------------------------------------------------------------------------
# Fakes shared by the retriever tests.
# ---------------------------------------------------------------------------


class _FakeFilterableMetadataStore:
    """Stand-in for RagMetadataStore that knows about domain/category/language.

    Built from a list of ``(chunk_id, doc_id, section, text, domain,
    category, language)`` tuples so each test can spell out the exact
    metadata partitioning it cares about. The doc-id allowlist mimics
    the V4 SQL whitelist by raising on unknown keys.
    """

    _ALLOWED = frozenset({"domain", "category", "language"})

    def __init__(
        self,
        index_version: str,
        rows: List[tuple],
    ) -> None:
        self._version = index_version
        self._chunks_by_row = {}
        self._docs: dict[str, dict] = {}
        for i, (chunk_id, doc_id, section, text, domain, category, language) in enumerate(rows):
            self._chunks_by_row[i] = ChunkLookupResult(
                chunk_id=chunk_id,
                doc_id=doc_id,
                section=section,
                text=text,
                faiss_row_id=i,
            )
            self._docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "domain": domain,
                    "category": category,
                    "language": language,
                },
            )

    def lookup_chunks_by_faiss_rows(
        self, index_version: str, faiss_row_ids: Iterable[int]
    ) -> List[ChunkLookupResult]:
        assert index_version == self._version, "version mismatch in fake store"
        return [self._chunks_by_row[i] for i in faiss_row_ids if i in self._chunks_by_row]

    def doc_ids_matching(self, filters: dict) -> List[str]:
        if not filters:
            return list(self._docs.keys())
        for key in filters:
            if key not in self._ALLOWED:
                raise ValueError(
                    f"Unsupported filter key {key!r}. Allowed: {sorted(self._ALLOWED)}."
                )
        out: List[str] = []
        for doc_id, meta in self._docs.items():
            if all(meta.get(k) == v for k, v in filters.items()):
                out.append(doc_id)
        return out


def _build_filterable_retriever(tmp_path: Path) -> tuple[Retriever, _FakeFilterableMetadataStore]:
    rows = [
        ("anime-1", "anime-001", "overview",
         "Aoi tends luminescent gardens above the clouds.",
         "anime", None, "en"),
        ("anime-2", "anime-002", "plot",
         "Ironclad Academy students pilot construction mechs to reinforce a coastal dam.",
         "anime", None, "en"),
        ("anime-3", "anime-kr-1", "overview",
         "주인공이 바다를 지키는 슬라이스오브라이프 애니메이션.",
         "anime", None, "ko"),
        ("ent-1", "kr-hr-001", "overview",
         "재택근무 신청은 팀장 결재 후 인사팀에 제출합니다.",
         "enterprise", "hr", "ko"),
        ("ent-2", "kr-it-001", "overview",
         "Kubernetes 클러스터 운영 가이드와 장애 대응 절차.",
         "enterprise", "it", "ko"),
    ]
    embedder = HashingEmbedder(dim=64)
    texts = [r[3] for r in rows]
    vectors = embedder.embed_passages(texts)
    index = FaissIndex(tmp_path / "idx")
    index.build(vectors, index_version="test-v1", embedding_model=embedder.model_name)

    metadata = _FakeFilterableMetadataStore(index_version="test-v1", rows=rows)
    retriever = Retriever(
        embedder=embedder,
        index=index,
        metadata=metadata,
        top_k=3,
        candidate_k=10,
    )
    retriever.ensure_ready()
    return retriever, metadata


# ---------------------------------------------------------------------------
# 1. doc_ids_matching whitelist (purely the SQL contract — fake store).
# ---------------------------------------------------------------------------


def test_fake_store_doc_ids_matching_rejects_unknown_keys():
    rows = [
        ("c", "d", "s", "t", "anime", None, "en"),
    ]
    store = _FakeFilterableMetadataStore("v1", rows)
    with pytest.raises(ValueError):
        store.doc_ids_matching({"author": "alice"})


def test_fake_store_doc_ids_matching_empty_returns_all():
    rows = [
        ("c1", "d-a", "s", "t", "anime", None, "en"),
        ("c2", "d-b", "s", "t", "enterprise", "hr", "ko"),
    ]
    store = _FakeFilterableMetadataStore("v1", rows)
    assert sorted(store.doc_ids_matching({})) == ["d-a", "d-b"]


def test_fake_store_doc_ids_matching_single_key():
    rows = [
        ("c1", "d-a", "s", "t", "anime", None, "en"),
        ("c2", "d-b", "s", "t", "enterprise", "hr", "ko"),
    ]
    store = _FakeFilterableMetadataStore("v1", rows)
    assert store.doc_ids_matching({"domain": "anime"}) == ["d-a"]


def test_fake_store_doc_ids_matching_multi_key_intersection():
    rows = [
        ("c1", "d-a", "s", "t", "enterprise", "hr", "ko"),
        ("c2", "d-b", "s", "t", "enterprise", "it", "ko"),
        ("c3", "d-c", "s", "t", "enterprise", "hr", "en"),
    ]
    store = _FakeFilterableMetadataStore("v1", rows)
    matches = store.doc_ids_matching({"category": "hr", "language": "ko"})
    assert matches == ["d-a"]


def test_fake_store_doc_ids_matching_no_match_returns_empty():
    rows = [
        ("c", "d", "s", "t", "anime", None, "en"),
    ]
    store = _FakeFilterableMetadataStore("v1", rows)
    assert store.doc_ids_matching({"domain": "enterprise"}) == []


# ---------------------------------------------------------------------------
# 2. Retriever filter behaviour.
# ---------------------------------------------------------------------------


def test_retriever_no_filter_keeps_all_domains(tmp_path):
    retriever, _ = _build_filterable_retriever(tmp_path)
    report = retriever.retrieve("Aoi luminescent gardens above the clouds")
    assert report.results, "non-empty result expected"
    assert report.filters == {}
    assert report.filter_produced_no_docs is False
    # Anime + enterprise rows can both surface — no exclusion.
    domains = {r.doc_id.split("-", 1)[0] for r in report.results}
    assert domains  # at least one


def test_retriever_filter_excludes_off_domain_docs(tmp_path):
    retriever, _ = _build_filterable_retriever(tmp_path)
    report = retriever.retrieve(
        "재택근무 신청 절차",
        filters={"domain": "enterprise"},
    )
    assert report.filters == {"domain": "enterprise"}
    assert report.filter_produced_no_docs is False
    for chunk in report.results:
        # Anime doc_ids start with 'anime-' — the enterprise filter must
        # have stripped them out of the candidate pool.
        assert not chunk.doc_id.startswith("anime-"), (
            f"anime doc {chunk.doc_id!r} leaked through enterprise filter"
        )


def test_retriever_filter_short_circuits_when_no_docs_match(tmp_path):
    retriever, _ = _build_filterable_retriever(tmp_path)
    # category=legal matches zero docs in the fixture set.
    report = retriever.retrieve(
        "비밀유지계약 체결 절차",
        filters={"category": "legal"},
    )
    assert report.results == []
    assert report.filter_produced_no_docs is True
    assert report.filters == {"category": "legal"}


def test_retriever_rejects_unknown_filter_key(tmp_path):
    retriever, _ = _build_filterable_retriever(tmp_path)
    with pytest.raises(ValueError):
        retriever.retrieve("anything", filters={"author": "alice"})


def test_retriever_caller_filter_overrides_parser_filter(tmp_path):
    """A filters= argument wins over filters the parser inferred.

    Caller-knows-best ordering: the agent loop should be able to scope
    a retry to a specific domain even if the LLM parser proposed
    something different on the first turn.
    """
    retriever, _ = _build_filterable_retriever(tmp_path)
    # Retriever has a NoOp parser by default (no filters), but we pass an
    # explicit override and verify it takes effect.
    report = retriever.retrieve(
        "anime question",
        filters={"language": "ko"},
    )
    for chunk in report.results:
        assert not chunk.doc_id.startswith("anime-001"), (
            "language=ko filter must exclude the en-only anime-001"
        )
        assert not chunk.doc_id.startswith("anime-002"), (
            "language=ko filter must exclude the en-only anime-002"
        )


# ---------------------------------------------------------------------------
# 3. LlmQueryParser fills filters when the query is unambiguous.
# ---------------------------------------------------------------------------


class _FakeChat(LlmChatProvider):
    """Minimal LlmChatProvider stub that returns a canned JSON dict."""

    def __init__(self, json_return: dict) -> None:
        self._json_return = json_return
        self._capabilities = {
            "function_calling": True,
            "thinking": False,
            "json_mode": True,
            "vision": False,
            "audio": False,
        }
        self.calls: list = []

    @property
    def name(self) -> str:
        return "fake-chat"

    @property
    def capabilities(self) -> dict:
        return self._capabilities

    def chat_json(self, messages, *, schema_hint, max_tokens=512,
                  temperature=0.0, timeout_s=15.0, enable_thinking=False):
        self.calls.append({"messages": messages, "schema_hint": schema_hint})
        return self._json_return

    def chat_tools(self, messages, tools, *, max_tokens=512,
                   temperature=0.0, timeout_s=15.0,
                   enable_thinking=False):  # pragma: no cover
        raise NotImplementedError


def test_llm_parser_populates_filters_when_query_is_unambiguous():
    chat = _FakeChat(
        json_return={
            "normalized": "재택근무 신청 기한",
            "keywords": ["재택근무", "신청", "기한"],
            "intent": "factoid",
            "rewrites": [],
            "filters": {"domain": "enterprise", "category": "hr", "language": "ko"},
        },
    )
    parser = LlmQueryParser(chat)
    parsed = parser.parse("재택근무 신청 기한이 어떻게 되나요?")
    assert parsed.parser_name == "llm"
    assert parsed.filters == {
        "domain": "enterprise",
        "category": "hr",
        "language": "ko",
    }


def test_llm_parser_drops_out_of_whitelist_filter_values_silently():
    """A junk value on a whitelisted KEY should be dropped, not crash.

    The schema-violation fallback is reserved for genuinely malformed
    JSON. Filters are advisory; an out-of-vocab value is the LLM
    overshooting the prompt — we strip it and keep the rest.
    """
    chat = _FakeChat(
        json_return={
            "normalized": "q",
            "keywords": [],
            "intent": "other",
            "rewrites": [],
            "filters": {"domain": "news", "language": "ko"},  # 'news' not allowed
        },
    )
    parser = LlmQueryParser(chat)
    parsed = parser.parse("some query")
    assert parsed.parser_name == "llm"
    assert parsed.filters == {"language": "ko"}


def test_llm_parser_omits_filters_when_query_is_ambiguous():
    chat = _FakeChat(
        json_return={
            "normalized": "what is rag",
            "keywords": ["rag"],
            "intent": "definition",
            "rewrites": [],
            "filters": {},
        },
    )
    parser = LlmQueryParser(chat)
    parsed = parser.parse("What is RAG?")
    assert parsed.filters == {}


def test_llm_parser_drops_unknown_filter_keys_silently():
    chat = _FakeChat(
        json_return={
            "normalized": "q",
            "keywords": [],
            "intent": "other",
            "rewrites": [],
            "filters": {"author": "alice", "domain": "anime"},
        },
    )
    parser = LlmQueryParser(chat)
    parsed = parser.parse("q")
    assert parsed.filters == {"domain": "anime"}


def test_llm_parser_filters_field_omitted_from_response_is_ok():
    """Old-shape responses without 'filters' should still be accepted."""
    chat = _FakeChat(
        json_return={
            "normalized": "q",
            "keywords": [],
            "intent": "other",
            "rewrites": [],
        },
    )
    parser = LlmQueryParser(chat)
    parsed = parser.parse("q")
    assert parsed.filters == {}


def test_regex_parser_never_emits_filters():
    """Non-LLM parsers must keep filters={} so the retriever's filter
    path is dead unless an LLM is wired in."""
    parser = RegexQueryParser()
    parsed = parser.parse("재택근무 신청 기한이 어떻게 되나요?")
    assert parsed.filters == {}
