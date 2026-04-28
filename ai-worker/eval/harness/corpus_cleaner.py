"""Conservative cleaner for corpus chunks.

Why this exists
---------------
``corpus_noise_signals.detect_noise_signals`` flags suspect lines but
never rewrites them. This module is the matching *rewrite* step: it
removes only the categories of noise the detector positively identifies,
and it leaves any line of in-domain prose alone. This conservatism is
deliberate — Phase 1A's job is to make the corpus a fair retrieval input,
not to maximize a downstream metric. Aggressive cleaning would make the
upcoming retrieval experiments harder to interpret because the cleaner
would become a confounder.

Cleaning policy
---------------
1. Line-level: drop a line if and only if it is *purely* one of:
     - a category footer (``분류: …``)
     - a redirect notice (``이 문서는 … 리다이렉트 …``)
     - a ``Powered by …`` footer
     - a CC license footer
     - an ad / press-release phrase line
     - a delimiter run line (just dashes/equals/etc.)
   The "purely" qualifier is enforced by checking the line minus the
   matched span has no other meaningful characters left.

2. Inline: strip these inline markers but keep the rest of the line:
     - ``[편집]`` / ``[원본 편집]`` / ``[소스 편집]``
     - ``[접기]`` / ``[펼치기]`` / ``[숨기기]``
   Inline markers are short, stable strings; removing them never costs
   in-domain tokens.

3. Repetition: collapse 3+ consecutive identical non-trivial lines down
   to a single occurrence. Lines under 6 chars are exempt (dialogue).

4. Whitespace: trim trailing whitespace per line; collapse 3+ blank
   lines into 2; strip leading/trailing whitespace from the chunk.

Idempotency
-----------
``clean_chunk(clean_chunk(x).text) == clean_chunk(x).text`` for all x.
This is what lets us safely re-run the cleaner during experimentation.
The tests pin this for representative inputs.

Public surface
--------------
- ``CleaningResult``    — text + bookkeeping for one chunk
- ``DROP_REASON_*``     — string constants for the dropped-chunk audit
- ``clean_chunk``       — main entry
- ``clean_chunks``      — convenience batch
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


# Drop reasons used in the audit report when the cleaner empties a chunk.
DROP_REASON_EMPTY_INPUT = "empty_input"
DROP_REASON_EMPTY_AFTER_CLEAN = "empty_after_clean"


@dataclass(frozen=True)
class CleaningResult:
    text: str
    char_count_before: int
    char_count_after: int
    removed_lines: int
    collapsed_repeats: int
    drop_reason: Optional[str]


# --- Patterns ----------------------------------------------------------

# Inline markers stripped via simple regex substitution. The patterns
# are intentionally small and exact to keep their behavior obvious.
_INLINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[\s*(?:원본\s*편집|소스\s*편집|편집)\s*\]"),
    re.compile(r"\[\s*(?:접기|펼치기|숨기기)\s*\]"),
)

# Patterns whose match span — when stripped from the line — leaves no
# meaningful residue. The patterns deliberately span the whole line
# (anchored to ``^`` and using ``.*`` / ``\s*$`` to consume the tail)
# so that pattern.sub("", line) returns whitespace only when the line
# is genuinely a noise line. This guard is what protects in-domain
# sentences that happen to *contain* the kill phrase as a substring.
_LINE_KILL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*분류\s*:.*$"),                              # category footer
    re.compile(r"^\s*이\s*문서는[^\n]{0,80}리다이렉트.*$"),       # redirect notice
    re.compile(r"^\s*Powered\s*by\b.*$", re.IGNORECASE),          # powered-by
    re.compile(
        r"^\s*(?:라이선스|License)\s*:\s*CC\s*BY[\-\s\w]*.*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*CC\s*BY[\-\s]*NC[\-\s]*SA.*$", re.IGNORECASE),
    re.compile(
        r"^\s*(?:광고\s*문의|기사\s*제보|구독\s*하기|구독\s*신청).*$"
    ),
    re.compile(r"^\s*(?:최근\s*변경|최근\s*토론|역사\s*보기)\s*$"),
    re.compile(r"^\s*(?:로그인|회원가입|로그아웃)\s*$"),
    # Pure delimiter run line: only -, =, _, *, spaces. Min 4-char run.
    re.compile(r"^\s*[\-=_*]{4,}\s*$"),
)

_REPEATED_LINE_MIN_LEN = 6
_BLANK_LINE_RUN_LIMIT = 2  # collapse 3+ blanks into this many


def clean_chunk(text: str) -> CleaningResult:
    """Run the conservative cleaner over a single chunk.

    Returns a ``CleaningResult`` with the cleaned text and bookkeeping.
    The cleaner is idempotent: feeding the result back in produces the
    same output (modulo bookkeeping counters, which all become zero on
    the second pass since there's nothing left to clean).
    """
    char_before = len(text or "")
    if not text or not text.strip():
        return CleaningResult(
            text="",
            char_count_before=char_before,
            char_count_after=0,
            removed_lines=0,
            collapsed_repeats=0,
            drop_reason=DROP_REASON_EMPTY_INPUT,
        )

    removed_lines = 0
    kept: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if _is_full_line_noise(line):
            removed_lines += 1
            continue
        line = _strip_inline_markers(line)
        # If inline-stripping removed everything *and* the original line
        # had no other content, drop the now-empty residue too rather
        # than emit a bare blank.
        if not line.strip() and raw_line.strip():
            removed_lines += 1
            continue
        kept.append(line)

    # Pass 2: collapse 3+ consecutive identical non-trivial lines to 1,
    # and collapse 3+ blank lines to two.
    collapsed_lines, collapsed_repeats = _collapse_repeats(kept)
    collapsed_lines = _collapse_blank_runs(collapsed_lines, _BLANK_LINE_RUN_LIMIT)

    cleaned = "\n".join(collapsed_lines).strip()

    if not cleaned:
        return CleaningResult(
            text="",
            char_count_before=char_before,
            char_count_after=0,
            removed_lines=removed_lines,
            collapsed_repeats=collapsed_repeats,
            drop_reason=DROP_REASON_EMPTY_AFTER_CLEAN,
        )

    return CleaningResult(
        text=cleaned,
        char_count_before=char_before,
        char_count_after=len(cleaned),
        removed_lines=removed_lines,
        collapsed_repeats=collapsed_repeats,
        drop_reason=None,
    )


def clean_chunks(texts: Iterable[str]) -> List[CleaningResult]:
    """Convenience: run ``clean_chunk`` over a sequence of strings."""
    return [clean_chunk(t) for t in texts]


def cleaning_result_to_dict(result: CleaningResult) -> Dict[str, Any]:
    return asdict(result)


# --- Internals ---------------------------------------------------------


def _is_full_line_noise(line: str) -> bool:
    if not line.strip():
        return False  # blanks handled by the run-collapse pass
    for pattern in _LINE_KILL_PATTERNS:
        if pattern.search(line):
            # Confirm the kill: after the match span and any whitespace,
            # nothing else of substance remains. This protects against
            # rare edge cases where a redirect-style phrase is embedded
            # in a longer in-domain sentence.
            stripped_residue = pattern.sub("", line).strip()
            if not stripped_residue:
                return True
    return False


def _strip_inline_markers(line: str) -> str:
    for pattern in _INLINE_PATTERNS:
        line = pattern.sub("", line)
    # Compress accidental double spaces created by the strip; keep
    # tabs alone since corpora rarely use them.
    return re.sub(r"  +", " ", line).rstrip()


def _collapse_repeats(lines: Sequence[str]) -> tuple[List[str], int]:
    """Collapse 3+ consecutive identical lines down to one.

    Lines under ``_REPEATED_LINE_MIN_LEN`` chars are exempt. Returns the
    new list plus the count of *removed* repeats (so the audit can
    report "we collapsed N lines worth of repetition").
    """
    out: List[str] = []
    collapsed = 0
    i = 0
    n = len(lines)
    while i < n:
        cur = lines[i]
        run_end = i + 1
        while run_end < n and lines[run_end] == cur:
            run_end += 1
        run = run_end - i
        if run >= 3 and len(cur.strip()) >= _REPEATED_LINE_MIN_LEN:
            out.append(cur)
            collapsed += run - 1
        else:
            out.extend(lines[i:run_end])
        i = run_end
    return out, collapsed


def _collapse_blank_runs(lines: Sequence[str], limit: int) -> List[str]:
    """Limit any run of blank/whitespace-only lines to ``limit`` lines."""
    out: List[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run <= limit:
                out.append(line)
        else:
            blank_run = 0
            out.append(line)
    return out
