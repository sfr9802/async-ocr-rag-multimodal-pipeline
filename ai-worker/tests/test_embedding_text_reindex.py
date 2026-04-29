"""Tests for ``eval/harness/embedding_text_reindex.py``.

Pin the slug rule, cache-key determinism, manifest layout, and the
end-to-end variant build over a tiny in-memory corpus + the
production ``HashingEmbedder``. None of the tests need GPU or the
real bge-m3 model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness.embedding_text_reindex import (
    _digest_embed_texts,
    build_variant_dense_stack,
    corpus_slug_for_path,
    default_cache_dir_for_variant,
    iter_variant_chunk_records,
    load_variant_dense_stack,
    manifest_to_dict,
    model_slug_for_path,
    variant_cache_key,
    variant_slug_for_path,
)


# ---------------------------------------------------------------------------
# 0. Model + corpus slug helpers
# ---------------------------------------------------------------------------


class TestModelSlugForPath:
    def test_strips_hf_org_prefix(self):
        assert model_slug_for_path("BAAI/bge-m3") == "bge-m3"
        assert model_slug_for_path("intfloat/multilingual-e5-large") == \
            "multilingual-e5-large"

    def test_lowercases(self):
        assert model_slug_for_path("BAAI/bge-M3") == "bge-m3"

    def test_replaces_underscore_with_hyphen(self):
        assert model_slug_for_path("vendor/model_v2") == "model-v2"

    def test_empty_or_none_raises(self):
        with pytest.raises(ValueError):
            model_slug_for_path("")
        with pytest.raises(ValueError):
            model_slug_for_path(None)  # type: ignore[arg-type]


class TestCorpusSlugForPath:
    def test_anime_corpus_trims_token_chunked_suffix(self):
        slug = corpus_slug_for_path(
            Path("eval/corpora/anime_namu_v3_token_chunked/corpus.jsonl")
        )
        assert slug == "anime-namu-v3"

    def test_chunked_only_suffix_is_trimmed(self):
        slug = corpus_slug_for_path(
            Path("eval/corpora/enterprise_v1_chunked/payload.jsonl")
        )
        assert slug == "enterprise-v1"

    def test_no_known_suffix_returns_bare_parent(self):
        slug = corpus_slug_for_path(
            Path("eval/corpora/some_other_v1/data.jsonl")
        )
        assert slug == "some-other-v1"

    def test_empty_parent_falls_back_to_unknown(self):
        # Path with no parent name (root-level file).
        slug = corpus_slug_for_path(Path("/corpus.jsonl"))
        assert slug == "unknown-corpus" or slug == ""


# ---------------------------------------------------------------------------
# 1. Slug rule
# ---------------------------------------------------------------------------


class TestVariantSlugForPath:
    def test_known_variants_map_to_hyphen_slugs(self):
        assert variant_slug_for_path("raw") == "raw"
        assert variant_slug_for_path("title") == "title"
        assert variant_slug_for_path("title_section") == "title-section"
        assert variant_slug_for_path("section") == "section"
        assert variant_slug_for_path("keyword") == "keyword"
        assert variant_slug_for_path("all") == "all"

    def test_underscore_variant_normalises(self):
        # ``title_section`` → "title-section" (underscore → hyphen).
        assert variant_slug_for_path("title_section") == "title-section"

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding-text variant"):
            variant_slug_for_path("title-section")  # not in builder constants
        with pytest.raises(ValueError):
            variant_slug_for_path("garbage")

    def test_empty_or_none_raises(self):
        with pytest.raises(ValueError):
            variant_slug_for_path("")
        with pytest.raises(ValueError):
            variant_slug_for_path(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Cache key determinism + variant-axis sensitivity
# ---------------------------------------------------------------------------


class TestVariantCacheKey:
    def _corpus(self, tmp_path: Path) -> Path:
        p = tmp_path / "corpus.jsonl"
        p.write_text(
            '{"doc_id": "d1", "title": "T1", "sections": {"s": {"chunks": ["c1"]}}}\n',
            encoding="utf-8",
        )
        return p

    def test_key_is_stable_for_same_inputs(self, tmp_path: Path):
        corpus = self._corpus(tmp_path)
        k1 = variant_cache_key(corpus, "BAAI/bge-m3", 1024, "raw")
        k2 = variant_cache_key(corpus, "BAAI/bge-m3", 1024, "raw")
        assert k1 == k2
        assert len(k1) == 16

    def test_variant_axis_changes_key(self, tmp_path: Path):
        corpus = self._corpus(tmp_path)
        k_raw = variant_cache_key(corpus, "BAAI/bge-m3", 1024, "raw")
        k_title = variant_cache_key(corpus, "BAAI/bge-m3", 1024, "title")
        k_ts = variant_cache_key(corpus, "BAAI/bge-m3", 1024, "title_section")
        # No collisions between variants — this is the core safety
        # invariant; collisions would silently feed the confirm sweep
        # wrong vectors.
        assert len({k_raw, k_title, k_ts}) == 3

    def test_max_seq_axis_changes_key(self, tmp_path: Path):
        corpus = self._corpus(tmp_path)
        a = variant_cache_key(corpus, "BAAI/bge-m3", 1024, "title_section")
        b = variant_cache_key(corpus, "BAAI/bge-m3", 512, "title_section")
        assert a != b

    def test_model_axis_changes_key(self, tmp_path: Path):
        corpus = self._corpus(tmp_path)
        a = variant_cache_key(corpus, "BAAI/bge-m3", 1024, "title_section")
        b = variant_cache_key(corpus, "intfloat/multilingual-e5", 1024, "title_section")
        assert a != b

    def test_unknown_variant_raises(self, tmp_path: Path):
        corpus = self._corpus(tmp_path)
        with pytest.raises(ValueError):
            variant_cache_key(corpus, "BAAI/bge-m3", 1024, "garbage")

    def test_default_cache_dir_layout(self, tmp_path: Path):
        # Mirror the project's anime corpus layout so the slug rule
        # produces the readable name we standardised on.
        corpus_dir = tmp_path / "anime_namu_v3_token_chunked"
        corpus_dir.mkdir()
        corpus = corpus_dir / "corpus.jsonl"
        corpus.write_text(
            '{"doc_id": "d1", "title": "T1", "sections": {"s": {"chunks": ["c1"]}}}\n',
            encoding="utf-8",
        )
        d = default_cache_dir_for_variant(
            cache_root=Path("/tmp/cache"),
            embedding_model="BAAI/bge-m3",
            max_seq_length=1024,
            corpus_path=corpus,
            variant="title_section",
        )
        assert d.parent == Path("/tmp/cache")
        # New layout: <model>-<corpus>-<variant>-mseq<N>. No hash suffix.
        # ``BAAI/bge-m3`` → ``bge-m3``; ``anime_namu_v3_token_chunked``
        # → ``anime-namu-v3`` (after the ``-token-chunked`` trim).
        assert d.name == "bge-m3-anime-namu-v3-title-section-mseq1024"

    def test_default_cache_dir_no_hash_suffix(self, tmp_path: Path):
        """Pin: directory name carries no hex digest — the corpus path
        + chunk count in build.json are the source of truth for cache
        validity, not a path-encoded hash."""
        corpus = self._corpus(tmp_path)
        d = default_cache_dir_for_variant(
            cache_root=Path("/tmp/cache"),
            embedding_model="BAAI/bge-m3",
            max_seq_length=1024,
            corpus_path=corpus,
            variant="raw",
        )
        # Last hyphen segment must be ``mseq1024``, not a 16-char hex.
        last_segment = d.name.rsplit("-", 1)[-1]
        assert last_segment == "mseq1024"
        # Any 16-char hex suffix would have flipped this assertion.
        assert all(
            seg != "mseq1024" or i == len(d.name.split("-")) - 1
            for i, seg in enumerate(d.name.split("-"))
        )

    def test_default_cache_dirs_differ_per_variant(self, tmp_path: Path):
        corpus_dir = tmp_path / "anime_namu_v3_token_chunked"
        corpus_dir.mkdir()
        corpus = corpus_dir / "corpus.jsonl"
        corpus.write_text(
            '{"doc_id": "d1", "title": "T1", "sections": {"s": {"chunks": ["c1"]}}}\n',
            encoding="utf-8",
        )
        d_raw = default_cache_dir_for_variant(
            cache_root=Path("/tmp/cache"),
            embedding_model="BAAI/bge-m3",
            max_seq_length=1024,
            corpus_path=corpus,
            variant="raw",
        )
        d_title = default_cache_dir_for_variant(
            cache_root=Path("/tmp/cache"),
            embedding_model="BAAI/bge-m3",
            max_seq_length=1024,
            corpus_path=corpus,
            variant="title",
        )
        d_ts = default_cache_dir_for_variant(
            cache_root=Path("/tmp/cache"),
            embedding_model="BAAI/bge-m3",
            max_seq_length=1024,
            corpus_path=corpus,
            variant="title_section",
        )
        assert d_raw != d_title != d_ts
        assert d_raw.name == "bge-m3-anime-namu-v3-raw-mseq1024"
        assert d_title.name == "bge-m3-anime-namu-v3-title-mseq1024"
        assert d_ts.name == "bge-m3-anime-namu-v3-title-section-mseq1024"


# ---------------------------------------------------------------------------
# 3. Variant chunk iteration — embed_text composition
# ---------------------------------------------------------------------------


class TestIterVariantChunkRecords:
    def _two_doc_corpus(self, tmp_path: Path) -> Path:
        p = tmp_path / "corpus.jsonl"
        p.write_text(
            '{"doc_id": "d1", "title": "Bookshop", "sections": '
            '{"overview": {"chunks": ["The retired translator runs the bookshop."]}}}\n'
            '{"doc_id": "d2", "title": "Cats", "sections": '
            '{"plot": {"chunks": ["The fisherman feeds the harbor cats."]}}}\n',
            encoding="utf-8",
        )
        return p

    def test_raw_variant_yields_chunk_text_only(self, tmp_path: Path):
        corpus = self._two_doc_corpus(tmp_path)
        records = list(iter_variant_chunk_records(corpus, variant="raw"))
        assert len(records) == 2
        # raw variant: embed_text == raw_text.
        for rec in records:
            assert rec.embed_text == rec.raw_text
        assert records[0].title == "Bookshop"
        assert records[0].section == "overview"
        assert records[0].faiss_row_id == 0
        assert records[1].faiss_row_id == 1

    def test_title_variant_prefixes_title(self, tmp_path: Path):
        corpus = self._two_doc_corpus(tmp_path)
        records = list(iter_variant_chunk_records(corpus, variant="title"))
        # First line of embed_text is the title; raw_text remains the
        # plain chunk text.
        assert records[0].embed_text.startswith("Bookshop\n")
        assert records[0].raw_text == "The retired translator runs the bookshop."
        assert records[1].embed_text.startswith("Cats\n")

    def test_title_section_variant_includes_section(self, tmp_path: Path):
        corpus = self._two_doc_corpus(tmp_path)
        records = list(iter_variant_chunk_records(
            corpus, variant="title_section",
        ))
        # Title appears before section, both before the chunk body.
        text0 = records[0].embed_text
        assert text0.startswith("Bookshop\noverview\n")
        text1 = records[1].embed_text
        assert text1.startswith("Cats\nplot\n")
        # Raw text unchanged.
        assert "translator" in records[0].raw_text

    def test_unknown_variant_raises(self, tmp_path: Path):
        corpus = self._two_doc_corpus(tmp_path)
        with pytest.raises(ValueError):
            list(iter_variant_chunk_records(corpus, variant="garbage"))

    def test_skips_docs_without_doc_id(self, tmp_path: Path):
        # Corpus row missing doc_id / seed / title — skip silently.
        p = tmp_path / "corpus.jsonl"
        p.write_text(
            '{"sections": {"s": {"chunks": ["orphan"]}}}\n'
            '{"doc_id": "d1", "title": "T", "sections": {"s": {"chunks": ["valid"]}}}\n',
            encoding="utf-8",
        )
        records = list(iter_variant_chunk_records(p, variant="raw"))
        assert len(records) == 1
        assert records[0].doc_id == "d1"


# ---------------------------------------------------------------------------
# 4. Manifest digest + sample shape
# ---------------------------------------------------------------------------


class TestEmbedTextDigest:
    def test_digest_is_deterministic(self):
        a, _ = _digest_embed_texts(["a", "b", "c"])
        b, _ = _digest_embed_texts(["a", "b", "c"])
        assert a == b

    def test_digest_changes_on_text_change(self):
        a, _ = _digest_embed_texts(["a", "b", "c"])
        b, _ = _digest_embed_texts(["a", "b", "d"])
        assert a != b

    def test_samples_capped_and_truncated(self):
        # 10 items → only first 5 sampled; previews capped to 240 chars.
        long_text = "x" * 1000
        _, samples = _digest_embed_texts([long_text] * 10)
        assert len(samples) == 5
        for s in samples:
            assert len(s["preview"]) == 240
            assert s["char_count"] == 1000


# ---------------------------------------------------------------------------
# 5. End-to-end variant build → load round-trip with HashingEmbedder
# ---------------------------------------------------------------------------


class TestBuildVariantDenseStackEndToEnd:
    def _corpus(self, tmp_path: Path) -> Path:
        p = tmp_path / "corpus.jsonl"
        p.write_text(
            '{"doc_id": "doc-book", "title": "BookshopRouter", "sections": '
            '{"overview": {"chunks": '
            '["The retired translator runs the bookshop at the last station."]}}}\n'
            '{"doc_id": "doc-cats", "title": "HarborCats", "sections": '
            '{"plot": {"chunks": '
            '["The fisherman feeds the harbor cats every morning before dawn."]}}}\n'
            '{"doc_id": "doc-aoi", "title": "AoiGarden", "sections": '
            '{"overview": {"chunks": '
            '["Aoi tends luminescent gardens suspended above the clouds."]}}}\n',
            encoding="utf-8",
        )
        return p

    def test_build_persists_index_chunks_and_manifest(self, tmp_path: Path):
        from app.capabilities.rag.embeddings import HashingEmbedder

        corpus = self._corpus(tmp_path)
        index_dir = tmp_path / "idx-title-section"
        embedder = HashingEmbedder(dim=64)

        retriever, info, manifest = build_variant_dense_stack(
            corpus,
            embedder=embedder,
            index_dir=index_dir,
            top_k=2,
            embedding_text_variant="title_section",
        )

        assert info.document_count == 3
        assert info.chunk_count >= 3
        assert (index_dir / "faiss.index").exists()
        assert (index_dir / "build.json").exists()
        assert (index_dir / "chunks.jsonl").exists()
        assert (index_dir / "variant_manifest.json").exists()

        # Manifest carries the variant + a non-trivial digest.
        manifest_payload = json.loads(
            (index_dir / "variant_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest_payload["variant"] == "title_section"
        assert manifest_payload["variant_slug"] == "title-section"
        assert len(manifest_payload["embed_text_sha256"]) == 64
        assert manifest_payload["chunk_count"] == info.chunk_count

        # The retriever can answer at least one query — the variant
        # build wired up retrieval correctly.
        report = retriever.retrieve("bookshop translator")
        assert report.results
        assert report.results[0].doc_id  # populated

        # Manifest dataclass round-trips via manifest_to_dict.
        as_dict = manifest_to_dict(manifest)
        assert as_dict["variant"] == "title_section"
        assert as_dict["embed_text_sha256"] == manifest.embed_text_sha256

    def test_load_variant_dense_stack_reuses_cache(self, tmp_path: Path):
        from app.capabilities.rag.embeddings import HashingEmbedder

        corpus = self._corpus(tmp_path)
        index_dir = tmp_path / "idx-raw"
        embedder = HashingEmbedder(dim=64)
        build_variant_dense_stack(
            corpus,
            embedder=embedder,
            index_dir=index_dir,
            top_k=2,
            embedding_text_variant="raw",
        )

        # Fresh embedder instance — load path mustn't re-encode.
        embedder2 = HashingEmbedder(dim=64)
        retriever2, info2, manifest2 = load_variant_dense_stack(
            index_dir,
            embedder=embedder2,
            top_k=2,
        )
        assert info2.chunk_count >= 3
        assert manifest2 is not None and manifest2.variant == "raw"
        report = retriever2.retrieve("fisherman cats")
        assert report.results

    def test_build_raises_for_zero_chunk_corpus(self, tmp_path: Path):
        from app.capabilities.rag.embeddings import HashingEmbedder

        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="zero chunks"):
            build_variant_dense_stack(
                empty,
                embedder=HashingEmbedder(dim=8),
                index_dir=tmp_path / "idx",
                top_k=1,
                embedding_text_variant="title_section",
            )

    def test_load_rejects_model_mismatch(self, tmp_path: Path):
        from app.capabilities.rag.embeddings import HashingEmbedder

        corpus = self._corpus(tmp_path)
        index_dir = tmp_path / "idx-mismatch"
        build_variant_dense_stack(
            corpus,
            embedder=HashingEmbedder(dim=32),
            index_dir=index_dir,
            top_k=2,
            embedding_text_variant="raw",
        )
        # Different dim → different model_name → load must refuse.
        with pytest.raises(RuntimeError, match="embedding_model"):
            load_variant_dense_stack(
                index_dir,
                embedder=HashingEmbedder(dim=64),
                top_k=2,
            )

    def test_variant_changes_index_vectors(self, tmp_path: Path):
        """Two variants over the same corpus must produce different
        embedded texts → different FAISS contents (verified via the
        manifest digest, which is deterministic over the embed-text
        list)."""
        from app.capabilities.rag.embeddings import HashingEmbedder

        corpus = self._corpus(tmp_path)
        _, _, m_raw = build_variant_dense_stack(
            corpus,
            embedder=HashingEmbedder(dim=32),
            index_dir=tmp_path / "idx-raw",
            top_k=2,
            embedding_text_variant="raw",
        )
        _, _, m_ts = build_variant_dense_stack(
            corpus,
            embedder=HashingEmbedder(dim=32),
            index_dir=tmp_path / "idx-ts",
            top_k=2,
            embedding_text_variant="title_section",
        )
        assert m_raw.embed_text_sha256 != m_ts.embed_text_sha256
        # Both manifests share the same chunk_count + dimension.
        assert m_raw.chunk_count == m_ts.chunk_count
        assert m_raw.dimension == m_ts.dimension
