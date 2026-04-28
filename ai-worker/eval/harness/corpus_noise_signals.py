"""Conservative noise signal detection over corpus chunks.

Why this exists
---------------
Phase 0 surfaced two corpus-quality concerns: a long-tail of multi-thousand
token chunks, and namu-wiki UI/boilerplate residue mixed into otherwise
in-domain text. Before we run any cleaner over the corpus we want a pure
*detection* pass that only flags suspect lines and counts how often each
signal fires. The detector deliberately does no rewriting — that is the
cleaner's job (``corpus_cleaner.py``) — so the audit and the cleaner can
be reasoned about independently.

What counts as a signal
-----------------------
The pattern set is intentionally narrow. Each signal is something a human
reader would unambiguously identify as namu-wiki UI / advertisement /
license / template residue, not in-domain prose:

  - ``ui_edit_marker``     — ``[편집]`` / ``[원본 편집]`` / ``[소스 편집]``
  - ``ui_collapse_toggle`` — ``[접기]`` / ``[펼치기]`` / ``[숨기기]``
  - ``ui_recent_changes``  — "최근 변경", "최근 토론" UI link text
  - ``ui_login_links``     — "로그인" / "회원가입" / "로그아웃"
  - ``ui_history_link``    — "역사 보기"
  - ``ui_category_footer`` — "분류:" footer at the start of a line
  - ``ui_redirect_notice`` — "이 문서는 ... 리다이렉트 ..."
  - ``ui_powered_by``      — "Powered by ..."
  - ``license_footer``     — "CC BY-NC-SA" license boilerplate
  - ``ad_phrase``          — "광고 문의" / "기사 제보" / "구독하기"
  - ``delimiter_run``      — 4+ run of ``-``, ``=``, ``_``, or ``*``
  - ``repeated_sentence``  — same non-trivial line repeats 3+ times

We do **not** flag normal section headings, character names, summaries,
quoted dialogue, or anything that overlaps with in-domain content. The
golden rule: if a reader could plausibly want the line in their search
result, the detector must leave it alone.

Public surface
--------------
- ``NoiseSignal``           — one (name, description, occurrences) tuple
- ``NoisePatternSpec``      — registry entry; not consumed externally
- ``DEFAULT_NOISE_PATTERNS``— the registry the detector ships with
- ``detect_noise_signals``  — main entry point
- ``signal_to_dict``        — JSON-friendly serializer
- ``aggregate_signals``     — fold a list of signal-lists into a counter

Implementation notes
--------------------
- Patterns are case-sensitive by default; namu-wiki Korean residue is
  always exact-form so case folding would only widen the false-positive
  surface (and there is no Korean case to fold).
- ``repeated_sentence`` is line-based: lines under 6 characters are
  ignored so we don't flag legitimate dialogue runs ("응.", "...").
- The detector is read-only over the input string — never mutates it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class NoiseSignal:
    name: str
    description: str
    occurrences: int


@dataclass(frozen=True)
class NoisePatternSpec:
    name: str
    description: str
    pattern: re.Pattern[str]


# Each entry is (signal_name, human description, compiled regex). The
# detector iterates this list in order, so adjacent kindred patterns
# (all ``ui_*``) live next to each other for readability.
DEFAULT_NOISE_PATTERNS: Tuple[NoisePatternSpec, ...] = (
    NoisePatternSpec(
        name="ui_edit_marker",
        description="namu-wiki edit / source-edit UI marker",
        pattern=re.compile(r"\[\s*(?:원본\s*편집|소스\s*편집|편집)\s*\]"),
    ),
    NoisePatternSpec(
        name="ui_collapse_toggle",
        description="collapse / expand / hide UI toggle",
        pattern=re.compile(r"\[\s*(?:접기|펼치기|숨기기)\s*\]"),
    ),
    NoisePatternSpec(
        name="ui_recent_changes",
        description="recent-changes or recent-discussion UI link",
        pattern=re.compile(r"최근\s*(?:변경|토론)"),
    ),
    NoisePatternSpec(
        name="ui_login_links",
        description="login / signup / logout UI link",
        pattern=re.compile(r"(?:로그인|회원가입|로그아웃)"),
    ),
    NoisePatternSpec(
        name="ui_history_link",
        description="history-view UI link",
        pattern=re.compile(r"역사\s*보기"),
    ),
    NoisePatternSpec(
        name="ui_category_footer",
        description="category footer line",
        pattern=re.compile(r"(?:^|\n)\s*분류\s*:\s*\S"),
    ),
    NoisePatternSpec(
        name="ui_redirect_notice",
        description="redirect notice line",
        pattern=re.compile(r"이\s*문서는[^\n]{0,80}리다이렉트"),
    ),
    NoisePatternSpec(
        name="ui_powered_by",
        description="powered-by footer",
        pattern=re.compile(r"Powered\s*by", re.IGNORECASE),
    ),
    NoisePatternSpec(
        name="license_footer",
        description="CC license footer boilerplate",
        pattern=re.compile(r"CC\s*BY[\-\s]*NC[\-\s]*SA", re.IGNORECASE),
    ),
    NoisePatternSpec(
        name="ad_phrase",
        description="advertisement / press-release phrase",
        pattern=re.compile(r"(?:광고\s*문의|기사\s*제보|구독\s*하기|구독\s*신청)"),
    ),
    NoisePatternSpec(
        name="delimiter_run",
        description="long delimiter run (4+ of -, =, _, *)",
        pattern=re.compile(r"([\-=_*])\1{3,}"),
    ),
)


# Lines shorter than this are ignored when counting repeated_sentence
# signals; otherwise short interjections in dialogue ("응.", "...") get
# falsely flagged. Tunable but intentionally generous.
_REPEATED_SENTENCE_MIN_LEN = 6
# How many times the *same* line must appear within one chunk before the
# repeated_sentence signal fires. 3 is the smallest count that excludes
# the common "X. X." pattern in summary lines and tracks the sort of
# template repetition we actually saw in Phase 0 spot-checks.
_REPEATED_SENTENCE_MIN_COUNT = 3


def detect_noise_signals(
    text: str,
    *,
    patterns: Sequence[NoisePatternSpec] = DEFAULT_NOISE_PATTERNS,
) -> List[NoiseSignal]:
    """Detect noise signals in ``text``. Read-only — does not mutate.

    Returns a list of ``NoiseSignal`` for each pattern that fired at
    least once, ordered by the registry order (so the report stays
    stable across runs). The returned list is empty for clean text.
    """
    if not isinstance(text, str) or not text:
        return []

    signals: List[NoiseSignal] = []
    for spec in patterns:
        matches = spec.pattern.findall(text)
        if matches:
            signals.append(NoiseSignal(
                name=spec.name,
                description=spec.description,
                occurrences=len(matches),
            ))

    repeats = _count_repeated_sentences(text)
    if repeats > 0:
        signals.append(NoiseSignal(
            name="repeated_sentence",
            description=(
                "same non-trivial line repeats "
                f"{_REPEATED_SENTENCE_MIN_COUNT}+ times in chunk"
            ),
            occurrences=repeats,
        ))

    return signals


def _count_repeated_sentences(text: str) -> int:
    """Count *extra* occurrences of any repeated non-trivial line.

    For a line that occurs ``n`` times where ``n >=
    _REPEATED_SENTENCE_MIN_COUNT``, contributes ``n - 1`` to the total.
    Lines under ``_REPEATED_SENTENCE_MIN_LEN`` characters are ignored.
    """
    counts: Counter[str] = Counter()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) < _REPEATED_SENTENCE_MIN_LEN:
            continue
        counts[line] += 1
    extra = 0
    for count in counts.values():
        if count >= _REPEATED_SENTENCE_MIN_COUNT:
            extra += count - 1
    return extra


def signal_to_dict(signal: NoiseSignal) -> Dict[str, Any]:
    return asdict(signal)


def aggregate_signals(
    signal_lists: Iterable[Sequence[NoiseSignal]],
) -> Dict[str, int]:
    """Fold a sequence of per-chunk signal lists into a name->count map.

    Useful for the "what fired across the corpus" summary — both the
    detector and the cleaner can produce per-chunk lists, and this is
    the canonical way to roll them up for the audit report.
    """
    totals: Counter[str] = Counter()
    for signals in signal_lists:
        for s in signals:
            totals[s.name] += int(s.occurrences)
    return dict(totals)
