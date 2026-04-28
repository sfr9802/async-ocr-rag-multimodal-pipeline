"""Tests for the corpus noise-signal detector.

The detector ships with a deliberately narrow pattern set. These tests
lock in two properties:

1. The patterns we *do* ship fire on the exact namu-wiki residue we
   want to flag, and they fire with the right occurrence count.
2. The patterns we *don't* ship — i.e. plain in-domain Korean prose,
   character names, dialogue — produce zero signals. This is the
   important one: false positives here become silent data loss after
   the cleaner runs.
"""

from __future__ import annotations

from eval.harness.corpus_noise_signals import (
    DEFAULT_NOISE_PATTERNS,
    NoiseSignal,
    aggregate_signals,
    detect_noise_signals,
    signal_to_dict,
)


# --- Positive detections ------------------------------------------------


class TestDetectsKnownNoise:
    def test_edit_marker_basic(self):
        signals = detect_noise_signals("본문 [편집] 다음 내용")
        names = [s.name for s in signals]
        assert "ui_edit_marker" in names
        edit = next(s for s in signals if s.name == "ui_edit_marker")
        assert edit.occurrences == 1

    def test_edit_marker_variants(self):
        signals = detect_noise_signals(
            "[편집] [원본 편집] [소스 편집]"
        )
        edit = next(s for s in signals if s.name == "ui_edit_marker")
        assert edit.occurrences == 3

    def test_collapse_toggle(self):
        signals = detect_noise_signals("[접기] 본문 [펼치기] 끝 [숨기기]")
        toggle = next(s for s in signals if s.name == "ui_collapse_toggle")
        assert toggle.occurrences == 3

    def test_recent_changes_and_login(self):
        text = "최근 변경 | 최근 토론 | 로그인 | 회원가입"
        signals = detect_noise_signals(text)
        names = {s.name for s in signals}
        assert "ui_recent_changes" in names
        assert "ui_login_links" in names

    def test_category_footer_line(self):
        text = "본문 끝.\n분류: 일본 애니메이션\n분류: 2010년 작품"
        signals = detect_noise_signals(text)
        cat = next(s for s in signals if s.name == "ui_category_footer")
        assert cat.occurrences == 2

    def test_redirect_notice(self):
        text = "이 문서는 다른 문서로 리다이렉트되어 있습니다."
        signals = detect_noise_signals(text)
        names = {s.name for s in signals}
        assert "ui_redirect_notice" in names

    def test_powered_by_and_license(self):
        text = "Powered by namu-wiki — 라이선스: CC BY-NC-SA"
        signals = detect_noise_signals(text)
        names = {s.name for s in signals}
        assert "ui_powered_by" in names
        assert "license_footer" in names

    def test_ad_phrases(self):
        text = "광고 문의 070-0000-0000\n기사 제보는 메일로\n구독하기 클릭"
        signals = detect_noise_signals(text)
        ad = next(s for s in signals if s.name == "ad_phrase")
        # All three phrases live behind one ad_phrase signal.
        assert ad.occurrences == 3

    def test_delimiter_run(self):
        text = "------\n본문\n=====\n본문2\n____\n본문3"
        signals = detect_noise_signals(text)
        delim = next(s for s in signals if s.name == "delimiter_run")
        assert delim.occurrences >= 3

    def test_repeated_sentence_fires_on_3_plus(self):
        line = "중요한 문장이 반복됩니다."
        text = "\n".join([line, line, line, "다른 문장"])
        signals = detect_noise_signals(text)
        rep = next(s for s in signals if s.name == "repeated_sentence")
        # 3 occurrences ⇒ 2 extras counted.
        assert rep.occurrences == 2

    def test_short_dialogue_lines_not_flagged_as_repeated(self):
        # Short interjections must not trip the repeated_sentence check.
        text = "응.\n응.\n응.\n응.\n응."
        signals = detect_noise_signals(text)
        names = {s.name for s in signals}
        assert "repeated_sentence" not in names

    def test_history_link(self):
        signals = detect_noise_signals("역사 보기 | 분류")
        names = {s.name for s in signals}
        assert "ui_history_link" in names


# --- Negative cases (must not fire) ------------------------------------


class TestNoFalsePositivesOnInDomainProse:
    def test_clean_korean_prose_yields_no_signals(self):
        text = (
            "주인공 테이토 클라인은 잃어버린 과거의 기억을 떠올리고, "
            "제국군 사관학교를 탈출한다. 후속작에서는 자유도가 증가하면서 "
            "PK 난무 등의 문제점이 발생한다."
        )
        assert detect_noise_signals(text) == []

    def test_character_dialogue_yields_no_signals(self):
        text = '"무엇을 원하는가?"\n"진실을 알고 싶다."\n"그렇다면 따라오라."'
        assert detect_noise_signals(text) == []

    def test_summary_with_dashes_does_not_fire_delimiter_run(self):
        # 1-3 dashes are valid Korean punctuation usage.
        text = "주인공 — 테이토 클라인. 부주인공 - 미카게."
        signals = detect_noise_signals(text)
        names = {s.name for s in signals}
        assert "delimiter_run" not in names

    def test_section_heading_does_not_fire_category(self):
        # A line starting with "분류" but not "분류:" must not fire.
        text = "분류는 일반적으로 두 가지로 나뉜다."
        signals = detect_noise_signals(text)
        names = {s.name for s in signals}
        assert "ui_category_footer" not in names

    def test_empty_text_yields_no_signals(self):
        assert detect_noise_signals("") == []
        assert detect_noise_signals("   ") == []


# --- Aggregation + serialization ---------------------------------------


class TestAggregationAndSerialization:
    def test_aggregate_signals_sums_per_name(self):
        a = [NoiseSignal("ui_edit_marker", "x", 2)]
        b = [
            NoiseSignal("ui_edit_marker", "x", 3),
            NoiseSignal("ad_phrase", "y", 1),
        ]
        c: list[NoiseSignal] = []
        totals = aggregate_signals([a, b, c])
        assert totals == {"ui_edit_marker": 5, "ad_phrase": 1}

    def test_signal_to_dict_round_trip(self):
        s = NoiseSignal("ui_edit_marker", "edit", 7)
        payload = signal_to_dict(s)
        assert payload == {
            "name": "ui_edit_marker",
            "description": "edit",
            "occurrences": 7,
        }


# --- Registry sanity ----------------------------------------------------


class TestPatternRegistry:
    def test_pattern_names_are_unique(self):
        names = [p.name for p in DEFAULT_NOISE_PATTERNS]
        assert len(names) == len(set(names))

    def test_every_pattern_is_compiled(self):
        for spec in DEFAULT_NOISE_PATTERNS:
            assert spec.pattern.search is not None
