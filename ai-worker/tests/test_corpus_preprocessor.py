"""Tests for the ingest-side corpus preprocessor.

Coverage map:

- ``detect_prefix_span`` returns None when the anchor is missing or
  too far inside the text.
- The strip removes a real-shaped namu-wiki page-prefix block
  including the news-headline tail before the anchor.
- The strip refuses on text that has no anchor — in-domain prose is
  preserved byte-for-byte.
- A short prefix-only chunk (≈100 chars of pure metadata) is dropped
  entirely; the result text is empty and ``dropped`` is True.
- Inline ``[편집]`` markers are stripped without touching ordinary
  bracketed text like ``[엑's 이슈]`` (substring match) or ``[1]``
  citations.
- ``preprocess_text`` is idempotent: ``preprocess(preprocess(x).text)``
  matches ``preprocess(x).text`` for representative inputs.
- ``preprocess_section_payload`` filters dropped chunks out of the
  emitted ``chunks`` list and reports separate text-blob / list
  results.
- ``preprocess_document_payload`` rolls per-section results into the
  doc summary correctly (counts of changes, drops, prefix strips,
  inline removals).
"""

from __future__ import annotations

import pytest

from eval.harness.corpus_preprocessor import (
    DROP_THRESHOLD_RATIO,
    MAX_PREFIX_STRIP_CHARS,
    PAGE_META_ANCHOR,
    PREPROCESS_VERSION,
    CorpusPreprocessSummary,
    PreprocessConfig,
    detect_prefix_span,
    fold_document_summary,
    preprocess_document_payload,
    preprocess_section_payload,
    preprocess_text,
    strip_inline_edit_markers,
)


# Real-shape prefix block adapted from Phase 1A's long-chunk audit
# samples. Includes the news-headline tail, the duplicated title, the
# timestamp, post-count, navbar, and the perm-warning notice — all
# glued onto the body by the namu-wiki dumper.
PREFIX_BLOCK_PERM = (
    "'일상이 화보' 지드래곤, 셔츠&가디건 매치로 색다른 느낌 뽐낸 공항패션! "
    "박시후가 가정 파탄?...결국 3명 다 입장 냈다 [엑's 이슈] "
    "야쿠자: 야쿠자 최근 수정 시각: 2025-08-07 22:58:06 76 편집 요청 편집 요청 "
    "편집 권한이 부족합니다. 로그인된 사용자(이)여야 합니다. "
    "해당 문서의 ACL 탭 을 확인하시기 바랍니다. "
)

PREFIX_BLOCK_NAVBAR = (
    "에르네스티 에체바르리아: 에르네스티 에체바르리아 "
    "최근 수정 시각: 2025-07-06 23:27:56 23 편집 토론 역사 분류 "
)

BODY_TEXT = (
    "주인공 테이토 클라인은 잃어버린 과거의 기억을 떠올리고, 제국군 사관학교를 "
    "탈출한다. 사쿠라 같은 인물도 등장한다."
)


# --- detect_prefix_span -------------------------------------------------


class TestDetectPrefixSpan:
    def test_anchor_with_perm_warning_is_swept(self):
        text = PREFIX_BLOCK_PERM + BODY_TEXT
        span = detect_prefix_span(text)
        assert span is not None
        # The strip should include the news headline AND the
        # perm-warning sentence — anchor.start() is mid-text but the
        # sweep always starts at offset 0.
        assert span.start == 0
        assert span.end > 0
        # The body portion must remain in text[span.end:].
        assert "주인공 테이토 클라인" in text[span.end:]

    def test_anchor_with_navbar_only_is_swept(self):
        text = PREFIX_BLOCK_NAVBAR + BODY_TEXT
        span = detect_prefix_span(text)
        assert span is not None
        assert span.start == 0
        assert "주인공 테이토 클라인" in text[span.end:]

    def test_no_anchor_returns_none(self):
        text = "주인공 테이토 클라인은 사관학교를 탈출한다. 그 후 미카게의 도움을 받는다."
        assert detect_prefix_span(text) is None

    def test_anchor_buried_after_search_window_returns_none(self):
        # Anchor lives at offset > PREFIX_SEARCH_HEAD_CHARS — must not
        # trigger a strip. We pad with in-domain text to push the
        # anchor out of the head window.
        pad = "본문은 길게 이어진다. " * 400  # well over 3000 chars
        text = pad + PREFIX_BLOCK_NAVBAR + BODY_TEXT
        assert len(pad) > 3000
        assert detect_prefix_span(text) is None

    def test_empty_string_returns_none(self):
        assert detect_prefix_span("") is None
        assert detect_prefix_span("   ") is None

    def test_signal_list_includes_anchor_marker(self):
        text = PREFIX_BLOCK_NAVBAR + BODY_TEXT
        span = detect_prefix_span(text)
        assert span is not None
        assert "page_meta_anchor" in span.signals


# --- preprocess_text: prefix strip --------------------------------------


class TestPrefixStrip:
    def test_long_prefix_chunk_keeps_body(self):
        text = PREFIX_BLOCK_PERM + BODY_TEXT
        result = preprocess_text(
            text, config=PreprocessConfig(strip_page_prefix=True)
        )
        assert result.changed
        assert not result.dropped
        assert result.removed_prefix_chars > 0
        assert BODY_TEXT in result.text
        # The news headline is gone.
        assert "지드래곤" not in result.text
        # The "최근 수정 시각" anchor is gone.
        assert "최근 수정 시각" not in result.text

    def test_short_prefix_only_chunk_is_dropped(self):
        # ≈80-110-char "prefix only" pre-chunk — common in the dump.
        text = "야쿠자: 야쿠자 최근 수정 시각: 2025-08-07 22:58:06 76 편집 토론 역사 분류"
        result = preprocess_text(
            text, config=PreprocessConfig(strip_page_prefix=True)
        )
        assert result.changed
        assert result.dropped
        assert result.text == ""

    def test_no_anchor_returns_unchanged(self):
        text = BODY_TEXT
        result = preprocess_text(
            text, config=PreprocessConfig(strip_page_prefix=True)
        )
        assert not result.changed
        assert not result.dropped
        assert result.text == text
        assert result.removed_prefix_chars == 0

    def test_prefix_strip_disabled_does_nothing(self):
        text = PREFIX_BLOCK_PERM + BODY_TEXT
        result = preprocess_text(text, config=PreprocessConfig())
        assert not result.changed
        assert result.text == text

    def test_prefix_signals_listed_in_result(self):
        text = PREFIX_BLOCK_NAVBAR + BODY_TEXT
        result = preprocess_text(
            text, config=PreprocessConfig(strip_page_prefix=True)
        )
        assert "page_meta_anchor" in result.removed_prefix_signals

    def test_post_count_does_not_eat_first_category_year(self):
        # Real-world failure mode: anchor + post_count + navbar*4 was
        # then followed by "2015년 작품 ...", and a free-floating
        # ``\d+`` rule in the sweep used to match "2015", chewing the
        # leading digits off the first category. The post-count is
        # part of the anchor itself; this test pins that fix.
        text = (
            "최근 수정 시각: 2025-08-05 13:22:05 83 편집 토론 역사 분류 "
            "2015년 작품 부시로드 밴드물"
        )
        result = preprocess_text(
            text, config=PreprocessConfig(strip_page_prefix=True)
        )
        # Categories survive intact.
        assert "2015년 작품" in result.text
        assert "부시로드 밴드물" in result.text
        # Anchor + revcount + navbar are gone.
        assert "최근 수정 시각" not in result.text
        assert " 83 " not in result.text


# --- strip_inline_edit_markers ------------------------------------------


class TestInlineEditStrip:
    def test_basic_marker_stripped(self):
        text = "주인공 [편집] 은 테이토 클라인이다."
        cleaned, count = strip_inline_edit_markers(text)
        assert "[편집]" not in cleaned
        assert "주인공" in cleaned
        assert "테이토 클라인" in cleaned
        assert count == 1

    def test_variants_stripped(self):
        text = "본문 [편집] [원본 편집] [소스 편집] 끝"
        cleaned, count = strip_inline_edit_markers(text)
        assert count == 3
        for token in ("[편집]", "[원본 편집]", "[소스 편집]"):
            assert token not in cleaned

    def test_unrelated_brackets_preserved(self):
        # [엑's 이슈] is a publication name, [1] is a citation, [pdf]
        # is a doc-type tag — none of these should be stripped.
        text = "본문 [엑's 이슈] 내용 [1] 끝 [pdf] 표시"
        cleaned, count = strip_inline_edit_markers(text)
        assert count == 0
        assert "[엑's 이슈]" in cleaned
        assert "[1]" in cleaned
        assert "[pdf]" in cleaned

    def test_double_space_is_compressed(self):
        text = "주인공  [편집]  본문 끝"
        cleaned, _ = strip_inline_edit_markers(text)
        assert "  " not in cleaned

    def test_empty_input_returns_empty(self):
        cleaned, count = strip_inline_edit_markers("")
        assert cleaned == ""
        assert count == 0


# --- preprocess_text combination + idempotency --------------------------


class TestCombinedAndIdempotency:
    def test_combined_strips_prefix_and_inline(self):
        text = PREFIX_BLOCK_PERM + "주인공 [편집] 은 테이토. " + BODY_TEXT
        result = preprocess_text(
            text,
            config=PreprocessConfig(
                strip_page_prefix=True, strip_inline_edit=True
            ),
        )
        assert result.changed
        assert "[편집]" not in result.text
        assert "지드래곤" not in result.text
        assert "테이토" in result.text

    def _idempotent(self, text: str, config: PreprocessConfig) -> None:
        first = preprocess_text(text, config=config)
        second = preprocess_text(first.text, config=config)
        assert second.text == first.text, (
            f"non-idempotent under {config}\n"
            f"first:  {first.text!r}\n"
            f"second: {second.text!r}"
        )

    def test_idempotent_clean_prose(self):
        for cfg in (
            PreprocessConfig(strip_page_prefix=True),
            PreprocessConfig(strip_inline_edit=True),
            PreprocessConfig(strip_page_prefix=True, strip_inline_edit=True),
        ):
            self._idempotent(BODY_TEXT, cfg)

    def test_idempotent_with_full_prefix(self):
        text = PREFIX_BLOCK_PERM + "주인공 [편집] 본문. " + BODY_TEXT
        for cfg in (
            PreprocessConfig(strip_page_prefix=True),
            PreprocessConfig(strip_inline_edit=True),
            PreprocessConfig(strip_page_prefix=True, strip_inline_edit=True),
        ):
            self._idempotent(text, cfg)

    def test_idempotent_on_inline_only_text(self):
        text = "주인공 [편집] [편집] [편집] 본문 끝"
        cfg = PreprocessConfig(strip_inline_edit=True)
        self._idempotent(text, cfg)


# --- preprocess_section_payload ----------------------------------------


class TestSectionPayload:
    def test_dropped_chunks_filtered_out(self):
        payload = {
            "chunks": [
                "야쿠자: 야쿠자 최근 수정 시각: 2025-08-07 22:58:06 76 편집 토론 역사 분류",
                BODY_TEXT,
            ]
        }
        outcome = preprocess_section_payload(
            payload, config=PreprocessConfig(strip_page_prefix=True)
        )
        # The prefix-only chunk is dropped from new_payload['chunks'].
        assert len(outcome.new_payload["chunks"]) == 1
        assert outcome.new_payload["chunks"][0] == BODY_TEXT
        # Both per-chunk results are returned for bookkeeping.
        assert len(outcome.chunk_results) == 2
        assert outcome.chunk_results[0].dropped is True
        assert outcome.chunk_results[1].dropped is False

    def test_text_blob_result_is_separate(self):
        payload = {
            "chunks": [BODY_TEXT],
            "text": PREFIX_BLOCK_PERM + BODY_TEXT,
        }
        outcome = preprocess_section_payload(
            payload, config=PreprocessConfig(strip_page_prefix=True)
        )
        assert outcome.text_blob_result is not None
        assert outcome.text_blob_result.changed
        # The chunk_results list does NOT contain the text-blob result.
        assert len(outcome.chunk_results) == 1
        assert outcome.chunk_results[0].changed is False

    def test_list_entries_processed(self):
        payload = {
            "list": [
                {"name": "테이토 [편집]", "desc": "주인공 [편집] 이다."},
                {"name": "미카게", "desc": "친구"},
            ]
        }
        outcome = preprocess_section_payload(
            payload, config=PreprocessConfig(strip_inline_edit=True)
        )
        new_list = outcome.new_payload["list"]
        assert "[편집]" not in new_list[0]["name"]
        assert "[편집]" not in new_list[0]["desc"]
        assert new_list[1]["name"] == "미카게"  # untouched
        # 2 entries × 2 fields where both have content → 4 results
        # (4 results total: 2 with changed=True, 2 unchanged).
        assert len(outcome.list_results) == 4
        assert sum(1 for r in outcome.list_results if r.changed) == 2

    def test_non_dict_section_passthrough(self):
        payload = {"chunks": "not a list"}
        outcome = preprocess_section_payload(
            payload, config=PreprocessConfig(strip_page_prefix=True)
        )
        # Non-list 'chunks' field is preserved as-is via shallow copy.
        assert outcome.new_payload["chunks"] == "not a list"


# --- preprocess_document_payload + summary fold ------------------------


class TestDocumentPayload:
    def test_doc_summary_counts_changes(self):
        doc = {
            "doc_id": "d1",
            "title": "T",
            "sections": {
                "s1": {
                    "chunks": [
                        "야쿠자: 야쿠자 최근 수정 시각: 2025-08-07 22:58:06 76 편집 토론 역사 분류",
                        PREFIX_BLOCK_NAVBAR + BODY_TEXT,
                        BODY_TEXT,
                    ]
                },
                "s2": {
                    "list": [
                        {"name": "테이토 [편집]", "desc": "주인공"},
                    ]
                },
            },
        }
        new_doc, summary = preprocess_document_payload(
            doc,
            config=PreprocessConfig(
                strip_page_prefix=True, strip_inline_edit=True
            ),
        )
        assert summary.doc_id == "d1"
        assert summary.sections_processed == 2
        assert summary.chunks_processed == 3
        assert summary.chunks_dropped == 1     # the prefix-only chunk
        assert summary.chunks_changed == 2     # dropped + body-with-prefix
        assert summary.list_entries_changed == 1
        assert summary.prefix_strip_count >= 2

        # The chunker would only see two chunks now in s1 (third chunk
        # is unchanged BODY_TEXT, second is BODY_TEXT after prefix
        # strip). Confirm the new doc reflects that.
        assert len(new_doc["sections"]["s1"]["chunks"]) == 2

    def test_corpus_summary_fold(self):
        corpus = CorpusPreprocessSummary(
            source_corpus="x",
            config=PreprocessConfig(strip_page_prefix=True),
        )
        doc1 = {
            "doc_id": "d1",
            "sections": {
                "s": {"chunks": [PREFIX_BLOCK_NAVBAR + BODY_TEXT]},
            },
        }
        doc2 = {
            "doc_id": "d2",
            "sections": {"s": {"chunks": [BODY_TEXT]}},  # no change
        }
        for doc in (doc1, doc2):
            _, ds = preprocess_document_payload(
                doc, config=corpus.config
            )
            fold_document_summary(corpus, ds)
        assert corpus.document_count == 2
        assert corpus.chunks_processed == 2
        assert corpus.chunks_changed == 1
        assert corpus.prefix_strip_count == 1


# --- Variant label / metadata ------------------------------------------


class TestConfigVariantLabel:
    def test_raw(self):
        assert PreprocessConfig().variant_label == "raw"

    def test_prefix_only(self):
        cfg = PreprocessConfig(strip_page_prefix=True)
        assert cfg.variant_label == f"prefix-{PREPROCESS_VERSION}"

    def test_inline_only(self):
        cfg = PreprocessConfig(strip_inline_edit=True)
        assert cfg.variant_label == f"inline-edit-{PREPROCESS_VERSION}"

    def test_combined(self):
        cfg = PreprocessConfig(strip_page_prefix=True, strip_inline_edit=True)
        assert (
            cfg.variant_label
            == f"prefix-{PREPROCESS_VERSION}.inline-edit-{PREPROCESS_VERSION}"
        )


# --- Anchor regex + cap constants pinned -------------------------------


class TestPinnedConstants:
    def test_anchor_matches_real_shape(self):
        assert PAGE_META_ANCHOR.search(
            "최근 수정 시각: 2025-08-07 22:58:06"
        )
        assert PAGE_META_ANCHOR.search(
            "최근 수정 시각: 2025-07-06 23:27:56"
        )
        # Date-only without time must not match (the timestamp shape
        # is the load-bearing signature).
        assert not PAGE_META_ANCHOR.search("최근 수정 시각: 2025-08-07")

    def test_drop_ratio_constant(self):
        assert 0.0 < DROP_THRESHOLD_RATIO < 1.0

    def test_max_prefix_strip_constant(self):
        assert MAX_PREFIX_STRIP_CHARS >= 1500
