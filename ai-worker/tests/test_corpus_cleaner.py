"""Tests for the conservative corpus cleaner.

Covers:
  - Each line-kill pattern actually drops the offending line.
  - Each inline marker is stripped without dropping its host line.
  - In-domain prose passes through completely unchanged.
  - Idempotency: ``clean(clean(x)).text == clean(x).text``.
  - The drop_reason field is set correctly when the cleaner produces
    an empty chunk.
  - Repeated-line collapse fires only on 3+ runs of non-trivial lines.
"""

from __future__ import annotations

from eval.harness.corpus_cleaner import (
    DROP_REASON_EMPTY_AFTER_CLEAN,
    DROP_REASON_EMPTY_INPUT,
    clean_chunk,
    clean_chunks,
    cleaning_result_to_dict,
)


# --- Line-kill patterns ------------------------------------------------


class TestLineKills:
    def test_category_footer_dropped(self):
        text = "주인공은 테이토 클라인이다.\n분류: 일본 애니메이션\n분류: 2010년 작품"
        result = clean_chunk(text)
        assert "분류" not in result.text
        assert "테이토 클라인" in result.text
        assert result.removed_lines == 2

    def test_redirect_notice_dropped(self):
        text = "본문 시작.\n이 문서는 다른 항목으로 리다이렉트되어 있습니다.\n본문 끝."
        result = clean_chunk(text)
        assert "리다이렉트" not in result.text
        assert "본문 시작" in result.text
        assert "본문 끝" in result.text
        assert result.removed_lines == 1

    def test_powered_by_dropped(self):
        text = "본문\nPowered by namu-wiki"
        result = clean_chunk(text)
        assert "Powered" not in result.text
        assert "본문" in result.text

    def test_license_footer_dropped(self):
        text = "본문 끝.\nCC BY-NC-SA 라이선스"
        result = clean_chunk(text)
        assert "CC BY" not in result.text

    def test_ad_phrase_line_dropped(self):
        text = "기사 본문입니다.\n광고 문의 070-0000-0000"
        result = clean_chunk(text)
        assert "광고" not in result.text
        assert "본문" in result.text

    def test_ui_link_only_lines_dropped(self):
        text = "본문 시작\n최근 변경\n역사 보기\n본문 끝"
        result = clean_chunk(text)
        assert "최근 변경" not in result.text
        assert "역사 보기" not in result.text
        assert "본문 시작" in result.text
        assert "본문 끝" in result.text

    def test_delimiter_run_line_dropped(self):
        text = "본문\n----\n다음 내용\n=====\n끝"
        result = clean_chunk(text)
        # Both delimiter-only lines drop, the prose lines stay.
        assert "----" not in result.text
        assert "=====" not in result.text
        assert "본문" in result.text
        assert "다음 내용" in result.text


# --- Inline markers -----------------------------------------------------


class TestInlineMarkers:
    def test_edit_marker_stripped_inline(self):
        text = "주인공 [편집] 은 테이토 클라인이다."
        result = clean_chunk(text)
        # Inline marker gone, sentence preserved.
        assert "[편집]" not in result.text
        assert "주인공" in result.text
        assert "테이토 클라인" in result.text
        # Line was kept (not dropped) — removed_lines counts only line drops.
        assert result.removed_lines == 0

    def test_collapse_toggle_stripped_inline(self):
        text = "내용 [접기] 본문 [펼치기] 끝"
        result = clean_chunk(text)
        for token in ("[접기]", "[펼치기]"):
            assert token not in result.text
        assert "본문" in result.text


# --- Negative cases (in-domain content untouched) -----------------------


class TestPreservesInDomainProse:
    def test_clean_prose_passes_through(self):
        text = (
            "주인공 테이토 클라인은 잃어버린 과거의 기억을 떠올리고, "
            "제국군 사관학교를 탈출한다."
        )
        result = clean_chunk(text)
        assert result.text == text
        assert result.removed_lines == 0
        assert result.collapsed_repeats == 0
        assert result.drop_reason is None

    def test_dialogue_dashes_not_treated_as_delimiter(self):
        # 1-3 dashes intermixed with text are not pure delimiter lines.
        text = "주인공 — 테이토. 부주인공 - 미카게."
        result = clean_chunk(text)
        assert result.text == text

    def test_word_starting_with_분류_not_dropped(self):
        # "분류" without a colon is just a noun.
        text = "분류는 일반적으로 두 가지로 나뉜다."
        result = clean_chunk(text)
        assert result.text == text

    def test_short_dialogue_repeats_not_collapsed(self):
        # 5 short identical lines must survive — they're below the
        # length floor.
        text = "응.\n응.\n응.\n응.\n응."
        result = clean_chunk(text)
        assert result.text.count("응.") == 5
        assert result.collapsed_repeats == 0


# --- Repeated-line collapse --------------------------------------------


class TestRepeatCollapse:
    def test_3_plus_consecutive_long_lines_collapse_to_one(self):
        line = "이 문장은 분명히 반복됩니다 그리고 충분히 깁니다."
        text = "\n".join([line, line, line, "다른 문장"])
        result = clean_chunk(text)
        assert result.text.count(line) == 1
        assert result.collapsed_repeats == 2  # 3 - 1

    def test_two_consecutive_repeats_kept(self):
        # Only 2 repeats — below the threshold; both must survive.
        line = "이 문장은 두 번만 반복됩니다."
        text = "\n".join([line, line, "다음"])
        result = clean_chunk(text)
        assert result.text.count(line) == 2
        assert result.collapsed_repeats == 0

    def test_non_consecutive_repeats_kept(self):
        # Same line repeats 3 times but with prose in between — not a run.
        line = "반복되지만 연속이 아닌 문장입니다."
        text = "\n".join([line, "다른 1", line, "다른 2", line])
        result = clean_chunk(text)
        assert result.text.count(line) == 3
        assert result.collapsed_repeats == 0


# --- Idempotency --------------------------------------------------------


class TestIdempotency:
    def _idempotent(self, text: str) -> None:
        first = clean_chunk(text)
        second = clean_chunk(first.text)
        assert second.text == first.text, (
            f"clean(clean(x)) != clean(x)\nfirst:  {first.text!r}\n"
            f"second: {second.text!r}"
        )

    def test_idempotent_on_clean_prose(self):
        self._idempotent(
            "주인공 테이토 클라인은 사관학교를 탈출한다."
        )

    def test_idempotent_with_inline_markers(self):
        self._idempotent("주인공 [편집] [접기] 본문 [펼치기]")

    def test_idempotent_with_line_kills(self):
        self._idempotent(
            "본문\n분류: 애니\n----\nPowered by namu-wiki\nCC BY-NC-SA"
        )

    def test_idempotent_with_repeats(self):
        line = "이 문장은 분명히 반복됩니다."
        self._idempotent("\n".join([line] * 5 + ["끝"]))

    def test_idempotent_on_mixed_real_world_chunk(self):
        text = (
            "[편집] 본문 시작.\n"
            "주인공 테이토 클라인은 사관학교를 탈출한다. [접기]\n"
            "분류: 애니메이션\n"
            "----\n"
            "Powered by namu-wiki\n"
            "이 문서는 다른 항목으로 리다이렉트되어 있습니다.\n"
            "본문 끝."
        )
        self._idempotent(text)


# --- Drop reasons -------------------------------------------------------


class TestDropReasons:
    def test_empty_input_drop_reason(self):
        result = clean_chunk("")
        assert result.text == ""
        assert result.drop_reason == DROP_REASON_EMPTY_INPUT

    def test_whitespace_only_input_drop_reason(self):
        result = clean_chunk("   \n  \n\t")
        assert result.text == ""
        assert result.drop_reason == DROP_REASON_EMPTY_INPUT

    def test_pure_noise_chunk_drops_to_empty(self):
        text = "분류: 애니\n----\nPowered by namu-wiki\nCC BY-NC-SA"
        result = clean_chunk(text)
        assert result.text == ""
        assert result.drop_reason == DROP_REASON_EMPTY_AFTER_CLEAN
        assert result.removed_lines >= 4

    def test_drop_reason_is_none_when_chunk_survives(self):
        result = clean_chunk("본문 한 줄.")
        assert result.drop_reason is None


# --- Helpers -----------------------------------------------------------


class TestHelpers:
    def test_clean_chunks_runs_over_iterable(self):
        results = clean_chunks(["abc", "분류: x", ""])
        assert len(results) == 3
        assert results[0].text == "abc"
        assert results[1].text == ""
        assert results[1].drop_reason == DROP_REASON_EMPTY_AFTER_CLEAN
        assert results[2].drop_reason == DROP_REASON_EMPTY_INPUT

    def test_to_dict_round_trip(self):
        result = clean_chunk("hello world")
        payload = cleaning_result_to_dict(result)
        assert payload["text"] == "hello world"
        assert payload["drop_reason"] is None
