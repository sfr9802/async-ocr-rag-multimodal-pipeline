"""Regenerate a natural-surface v4 RAG silver query set.

This generator builds retrieval-evaluation queries from the canonical v4
``pages_v4.jsonl`` and ``rag_chunks.jsonl`` artifacts, but intentionally keeps
the query surface closer to what a user would type: shortened titles, aliases,
plot memories, character hints, setting terms, and a small number of
wrong-assumption / ambiguous probes.

It writes the user-facing manifest schema requested for the Phase 7 natural
silver set. It does not run retrieval, indexing, tuning, or scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_CORPUS_DIR = Path("eval/corpora/namu-v4-structured-combined")
DEFAULT_REPORT_DIR = Path("eval/reports/phase7/7.11_silver_natural_regen")
DEFAULT_TARGET_TOTAL = 500
DEFAULT_SEED = 20260502

QUERY_TYPES = {
    "title_direct",
    "title_partial",
    "alias",
    "character_question",
    "setting_question",
    "plot_memory",
    "theme_question",
    "comparison_like",
    "vague_recall",
    "wrong_assumption",
    "unanswerable",
    "ambiguous",
}
TITLE_LEVELS = {"exact_title", "partial_title", "alias", "none"}
ENTITY_LEVELS = {"explicit", "partial", "implicit", "none"}
DIFFICULTIES = {"easy", "medium", "hard"}
ANSWERABILITY = {"answerable", "partially_answerable", "unanswerable"}

TARGET_COUNTS: dict[str, int] = {
    "title_direct": 90,
    "title_partial": 55,
    "alias": 45,
    "character_question": 75,
    "setting_question": 75,
    "plot_memory": 75,
    "vague_recall": 30,
    "theme_question": 20,
    "comparison_like": 10,
    "wrong_assumption": 15,
    "ambiguous": 10,
}

GENERIC_PAGE_TITLES = {
    "등장인물",
    "평가",
    "OST",
    "기타",
    "회차",
    "에피소드",
    "주제가",
    "음악",
    "회차 목록",
    "에피소드 가이드",
    "미디어 믹스",
    "기타 등장인물",
    "설정",
    "줄거리",
    "스태프",
    "성우진",
    "애니메이션",
    "특징",
}

STOP_LINES = {
    "자세한 내용은",
    "문서를 참고",
    "이전 역사 보러 가기",
    "스포일러",
    "회차",
    "제목",
    "각본",
    "콘티",
    "연출",
    "작화감독",
    "총작화감독",
    "방영일",
    "노래",
    "작사",
    "작곡",
    "편곡",
    "원화",
    "일본",
    "한국",
    "성우",
    "등장인물",
    "주요 성우진",
}

STOP_TOKENS = {
    "자세한",
    "내용",
    "문서",
    "참고",
    "작품",
    "애니메이션",
    "시리즈",
    "등장인물",
    "성우",
    "일본",
    "한국",
    "미국",
    "방영일",
    "각본",
    "콘티",
    "연출",
    "작화감독",
    "총작화감독",
    "노래",
    "작사",
    "작곡",
    "편곡",
    "가사",
    "평점",
    "별점",
    "기준",
    "목록",
    "설정",
    "문단",
    "이전",
    "역사",
    "보러",
    "이렇게",
    "다만",
    "일부",
    "같은",
    "내용은",
    "사용중인",
    "정작",
    "감독",
    "영어판",
    "원문",
    "시절",
    "전작",
    "등의",
    "작중",
}

TITLE_ALIAS_RULES: tuple[tuple[str, str], ...] = (
    ("CLANNAD", "클라나드"),
    ("ARIA", "아리아"),
    ("블루 아카이브", "블루아카이브"),
    ("약속의 네버랜드", "약속의 네버랜드"),
    ("스파이 패밀리", "스파이패밀리"),
    ("SPY×FAMILY", "스파이패밀리"),
    ("SPY x FAMILY", "스파이패밀리"),
    ("SPY FAMILY", "스파이패밀리"),
    ("D.Gray-man", "디그레이맨"),
    ("강철의 연금술사 FULLMETAL ALCHEMIST", "강철의 연금술사 FA"),
    ("소드 아트 온라인", "소드아트"),
    ("카구야 님", "카구야"),
    ("이 멋진 세계에 축복을", "코노스바"),
    ("하이큐", "하이큐"),
    ("바이올렛 에버가든", "바이올렛 에버가든"),
    ("나의 히어로 아카데미아", "나히아"),
    ("귀멸의 칼날", "귀멸"),
    ("주술회전", "주술회전"),
    ("포켓몬스터", "포켓몬"),
)

TABLE_MARKERS = (
    "각본 콘티",
    "연출 작화감독",
    "총작화감독",
    "방영일",
    "日:",
    "韓:",
    "러닝 타임",
    "TV ver",
    "Full ver",
    "MV ver",
    "제1화",
    "제2화",
    "제3화",
    "회차 제목",
)


@dataclass(frozen=True)
class PageMeta:
    page_id: str
    page_title: str
    work_title: str
    display_title: str
    retrieval_title: str
    page_type: str
    relation: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    query: str
    expected_doc_id: str | None
    expected_title: str | None
    expected_section_path: list[str] | None
    expected_chunk_ids: list[str]
    query_type: str
    title_mention_level: str
    entity_mention_level: str
    difficulty: str
    answerability: str
    source_evidence: str
    generation_note: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate natural-surface v4 RAG silver queries."
    )
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"(?<=[A-Za-z0-9가-힣」』])\s+(과의|와의|의|을|를)(?=\s|[,.;:!?]|$)", r"\1", text)
    return text


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _strip_parentheticals(title: str) -> str:
    out = re.sub(r"\([^)]*\)", "", title or "")
    out = re.sub(r"\[[^\]]*\]", "", out)
    return _clean_text(out)


def _parent_title_from_retrieval_title(retrieval_title: str, page_title: str) -> str:
    rt = (retrieval_title or "").strip()
    pt = (page_title or "").strip()
    for sep in (" / ", "/"):
        if sep in rt:
            head, _, tail = rt.partition(sep)
            if head.strip() and tail.strip() and head.strip() != pt:
                return head.strip()
    return rt or pt


def _expected_title(chunk: Mapping[str, Any]) -> str:
    return str(
        chunk.get("retrieval_title")
        or chunk.get("display_title")
        or chunk.get("title")
        or ""
    )


def _retrieval_tail(title: str) -> str:
    title = (title or "").strip()
    for sep in (" / ", "/"):
        if sep in title:
            return title.rsplit(sep, 1)[-1].strip()
    return ""


def _query_anchor_title(chunk: Mapping[str, Any]) -> str:
    title = _expected_title(chunk)
    page_title = str(chunk.get("title") or "")
    return _parent_title_from_retrieval_title(title, page_title)


def _is_generic_title(title: str) -> bool:
    if not title:
        return False
    return title in GENERIC_PAGE_TITLES or any(x in title for x in GENERIC_PAGE_TITLES)


def _title_surface(raw_title: str) -> tuple[str, str]:
    title = _clean_text(raw_title)
    if not title:
        return "", "none"

    for needle, alias in TITLE_ALIAS_RULES:
        if _title_rule_matches(title, needle):
            return alias, "alias"

    simplified = _strip_parentheticals(title)
    simplified = simplified.replace(" The Animation", "")
    simplified = simplified.replace(" THE ANIMATION", "")
    simplified = simplified.replace(" TVA", "")
    simplified = re.sub(r"\s+", " ", simplified).strip()

    if "~" in simplified:
        simplified = simplified.split("~", 1)[0].strip()

    if simplified.startswith("극장판 "):
        rest = simplified.removeprefix("극장판 ").strip()
        tokens = rest.split()
        if len(tokens) >= 2:
            simplified = f"{tokens[0]} 극장판"
        else:
            simplified = rest

    if len(simplified) > 18:
        tokens = simplified.split()
        simplified = " ".join(tokens[:2]) if len(tokens) >= 2 else simplified[:12]
    simplified = _strip_korean_particle(simplified)

    if not simplified:
        return "", "none"
    if _norm(simplified) == _norm(title):
        if " " in simplified and len(simplified) <= 12:
            return simplified.replace(" ", ""), "alias"
        return simplified, "exact_title"
    if _norm(simplified) in _norm(title) or _norm(title).startswith(_norm(simplified)):
        return simplified, "partial_title"
    return simplified, "alias"


def _title_rule_matches(title: str, needle: str) -> bool:
    title_l = title.lower()
    needle_l = needle.lower()
    if re.fullmatch(r"[a-z0-9 .×!-]+", needle_l):
        return re.search(
            rf"(?<![a-z0-9]){re.escape(needle_l)}(?![a-z0-9])",
            title_l,
        ) is not None
    return needle_l in title_l


def _contains_exact_title(query: str, expected_title: str | None) -> bool:
    if not expected_title:
        return False
    title = _norm(expected_title)
    if len(title) < 4:
        return False
    return title in _norm(query)


def _evidence(text: str, *, limit: int = 180) -> str:
    text = _clean_text(text)
    text = re.sub(r"대덕구컴퓨터.*", "", text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return cut + "..."


def _good_chunk(chunk: Mapping[str, Any]) -> bool:
    text = _clean_text(str(chunk.get("chunk_text") or ""))
    meta = chunk.get("metadata") or {}
    if len(text) < 45:
        return False
    if meta.get("is_stub") is True:
        return False
    if float(meta.get("noise_score") or 0.0) > 0.02:
        return False
    if "대덕구컴퓨터" in text or "www.omypc.co.kr" in text:
        return False
    return True


def _sentence_candidates(text: str) -> list[str]:
    text = _clean_text(text)
    text = text.replace("이 문서에 스포일러 가 포함되어 있습니다.", "")
    if any(marker in text for marker in TABLE_MARKERS):
        # 회차표/스태프표/주제가 표는 사용자가 기억으로 묻는 plot surface로
        # 쓰기엔 너무 기계적이라 제외한다.
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", text)
    out: list[str] = []
    for part in parts:
        part = _clean_text(part)
        if not part:
            continue
        part = _clean_text(
            re.sub(r"^(그리고|그러나|하지만|또한|다만)\s+", "", part)
        )
        if re.match(r"^(처럼|을|를|에서|으로|에게|부터|까지|난 후|한 후|된 후)\b", part):
            continue
        if re.search(r"제\d+화", part):
            continue
        if re.match(r"^[,.;:)\]]", part):
            continue
        if any(
            stop in part
            for stop in (
                "자세한 내용은",
                "이전 역사",
                "대덕구컴퓨터",
                "회차 목록",
                "줄거리, 결말, 반전",
                "직·간접적으로 포함",
                "본 문서",
                "복붙",
                "TVING",
                "VOD",
            )
        ):
            continue
        if any(marker in part for marker in TABLE_MARKERS):
            continue
        if 35 <= len(part) <= 125:
            out.append(part)
    if out:
        return out
    if 35 <= len(text) <= 150:
        return [text]
    if len(text) > 150:
        return [text[:115].rsplit(" ", 1)[0].strip()]
    return []


def _remove_title_from_fragment(fragment: str, titles: Iterable[str]) -> str:
    out = fragment
    for title in titles:
        title = _clean_text(title)
        if title and len(title) >= 3:
            out = out.replace(title, "이 작품")
            out = out.replace(_strip_parentheticals(title), "이 작품")
    return _clean_text(out)


def _trim_memory_fragment(fragment: str, *, limit: int = 115) -> str:
    fragment = _clean_text(fragment)
    if len(fragment) <= limit:
        return fragment
    cut = fragment[:limit].rsplit(" ", 1)[0].strip()
    cut = cut.rstrip(",.;:")
    return cut + "..."


def _line_entities(text: str, *, limit: int = 5) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        line = _clean_text(raw)
        if "CV." in line:
            line = line.split("CV.", 1)[0].strip()
        if "성우:" in line:
            line = line.split("성우:", 1)[0].strip()
        line = _strip_korean_particle(line)
        if re.match(r"(?i)^cv\b", line):
            continue
        if not line or len(line) < 2 or len(line) > 28:
            continue
        if re.match(r"^(에서|으로|에게|부터|까지|처럼|그리고|그러나|하지만|또한|다만)\b", line):
            continue
        if "," in line or "，" in line:
            continue
        if line.startswith("-") or line.startswith("※"):
            continue
        if len(line.split()) > 3:
            continue
        if re.search(r"\d+회|에서는$|으로$|에게$|부터$|까지$|처럼$|관련$|출처|전국대회|본편|대학|작중", line):
            continue
        if any(stop in line for stop in STOP_LINES) or "문서" in line:
            continue
        if any(marker in line for marker in ("▼", "소개", "대사", "하이라이트")):
            continue
        if re.search(
            r"[.!?]|입니다|한다|했다|있다|된다|이며|이고|되어|되는|나오는|보이는|얼굴|시체|장면|모습|고통|완전히|절단|감독",
            line,
        ):
            continue
        if re.search(r"https?://|www\.|[0-9]{4}\.[0-9]{2}|[0-9]+화", line):
            continue
        if len(re.sub(r"[^A-Za-z가-힣ぁ-んァ-ヶ一-龯]", "", line)) < 2:
            continue
        if line in seen:
            continue
        seen.add(line)
        entities.append(line)
        if len(entities) >= limit:
            break
    return entities


def _term_entities(text: str, *, limit: int = 5, allow_regex: bool = True) -> list[str]:
    lines = _line_entities(text, limit=limit * 2)
    terms: list[str] = []
    seen: set[str] = set()
    for line in lines:
        line = _strip_korean_particle(line)
        if len(line.split()) > 3:
            continue
        if any(tok in line for tok in STOP_TOKENS):
            continue
        if line in {"일부", "같은", "이하", "주역", "사람", "문제", "정도"}:
            continue
        if line in seen:
            continue
        seen.add(line)
        terms.append(line)
        if len(terms) >= limit:
            return terms

    if not allow_regex:
        return terms

    text = _clean_text(text)
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9+\-]{1,20}|[가-힣]{2,12}", text):
        token = _strip_korean_particle(match.group(0))
        if token in STOP_TOKENS or token in seen:
            continue
        if token in {"일부", "같은", "이하", "주역", "사람", "문제", "정도"}:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _section_setting_terms(section_path: Iterable[Any], *, limit: int = 3) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in section_path:
        text = _clean_text(str(raw))
        if not text or text in GENERIC_PAGE_TITLES:
            continue
        for part in re.split(r"[>/,ㆍ·]", text):
            part = _clean_text(part)
            part = re.sub(r"\([^)]*\)", "", part).strip()
            if not part or part in GENERIC_PAGE_TITLES:
                continue
            if len(part) < 2 or len(part) > 24:
                continue
            if any(stop in part for stop in STOP_TOKENS):
                continue
            if part in seen:
                continue
            seen.add(part)
            terms.append(part)
            if len(terms) >= limit:
                return terms
    return terms


def _strip_korean_particle(text: str) -> str:
    text = _clean_text(text)
    text = re.sub(r"\s+(으로|로|에서|에게|부터|까지|처럼|의|은|는|이|가|을|를|도|과|와)$", "", text)
    if len(text) > 3:
        text = re.sub(r"(으로|에서|에게|부터|까지|처럼|의|은|는|이|가|을|를|도|과|와)$", "", text)
    return text.strip()


def _make_row(
    *,
    query: str,
    chunk: Mapping[str, Any],
    query_type: str,
    title_mention_level: str,
    entity_mention_level: str,
    difficulty: str,
    answerability: str = "answerable",
    evidence_text: str | None = None,
    note: str,
) -> Candidate | None:
    expected = _expected_title(chunk)
    query = _clean_text(query)
    if not query:
        return None
    if _bad_query_surface(query):
        return None
    if _contains_exact_title(query, expected):
        return None
    if query_type not in QUERY_TYPES:
        return None
    if title_mention_level not in TITLE_LEVELS:
        return None
    if entity_mention_level not in ENTITY_LEVELS:
        return None
    if difficulty not in DIFFICULTIES:
        return None
    if answerability not in ANSWERABILITY:
        return None
    return Candidate(
        query=query,
        expected_doc_id=str(chunk.get("doc_id") or ""),
        expected_title=expected,
        expected_section_path=list(chunk.get("section_path") or []),
        expected_chunk_ids=[str(chunk.get("chunk_id") or "")],
        query_type=query_type,
        title_mention_level=title_mention_level,
        entity_mention_level=entity_mention_level,
        difficulty=difficulty,
        answerability=answerability,
        source_evidence=_evidence(evidence_text or str(chunk.get("chunk_text") or "")),
        generation_note=note,
    )


def _bad_query_surface(query: str) -> bool:
    if len(query) > 240:
        return True
    bad_literals = (
        "이 문서가 설명하는 작품",
        "줄거리, 결말, 반전",
        "출처:",
        "설정랑",
        "내용은랑",
        "에서는랑",
        "의랑",
        "완전히랑",
        "절단된 같은",
        "와 배경인",
        "랑 와 ",
        "랑 의 ",
        "랑 에서",
        "랑 작중",
        "CV는랑",
        "CV랑",
        "이 문서에",
    )
    if any(x in query for x in bad_literals):
        return True
    if re.match(r"^(처럼|을|를|에서|으로|에게|부터|까지|난 후|한 후|된 후|그리고|그러나|하지만|또한|다만)\b", query):
        return True
    if re.search(r"제\d+화", query):
        return True
    if re.search(r"랑\s*(와|의|은|는|이|가|을|를|에서|으로|에게|부터|까지|처럼|배경인|같은|작중)\b", query):
        return True
    if re.search(r"(에서는|으로|에게|부터|까지|처럼)랑", query):
        return True
    return False


def _add_candidate(
    by_type: dict[str, list[Candidate]],
    candidate: Candidate | None,
) -> None:
    if candidate is not None:
        by_type[candidate.query_type].append(candidate)


def _direct_candidates(chunk: Mapping[str, Any], by_type: dict[str, list[Candidate]]) -> None:
    if str(chunk.get("section_type") or "") != "summary":
        return
    section_path = list(chunk.get("section_path") or [])
    if not section_path or section_path[0] not in {"개요", "소개"}:
        return
    tail = _retrieval_tail(_expected_title(chunk))
    if tail and tail not in {"애니메이션"}:
        return
    text = _clean_text(str(chunk.get("chunk_text") or ""))
    title = _query_anchor_title(chunk)
    surface, level = _title_surface(title)
    if not surface or level == "exact_title":
        return

    if "감독은" in text and "방영 시기는" in text:
        q = f"{surface} 애니 감독이랑 방영 시기만 확인해줘"
    elif "제작사는" in text and ("방영 시기" in text or "방영" in text):
        q = f"{surface} 애니 제작사랑 언제 방영했는지 봐줘"
    elif "원작" in text:
        q = f"{surface} 애니 원작이 뭐였는지 알려줘"
    else:
        q = f"{surface} 애니가 어떤 작품이었는지 짧게 찾아줘"

    _add_candidate(
        by_type,
        _make_row(
            query=q,
            chunk=chunk,
            query_type="title_direct",
            title_mention_level=level,
            entity_mention_level="none",
            difficulty="medium",
            note="정식 제목의 괄호/시즌/영문 부제를 제거한 사용자 입력형으로 생성.",
        ),
    )

    if "1기" in _expected_title(chunk) or "제1기" in text:
        q2 = f"{surface} 1기 쪽 기본 정보만 다시 확인하고 싶어"
    elif "2기" in _expected_title(chunk) or "제2기" in text:
        q2 = f"{surface} 2기였던 편 방영 정보 찾아줘"
    elif "3기" in _expected_title(chunk) or "제3기" in text:
        q2 = f"{surface} 3기 TV판 정보가 필요해"
    else:
        q2 = f"{surface} 쪽 애니판 정보 찾고 있어"
    _add_candidate(
        by_type,
        _make_row(
            query=q2,
            chunk=chunk,
            query_type="title_partial",
            title_mention_level=level,
            entity_mention_level="partial",
            difficulty="medium",
            note="풀타이틀 대신 축약된 작품 표면형과 시즌 단서만 남김.",
        ),
    )

    if level == "alias":
        _add_candidate(
            by_type,
            _make_row(
                query=f"{surface}, 이 이름으로 찾으면 어느 애니 문서가 맞아?",
                chunk=chunk,
                query_type="alias",
                title_mention_level="alias",
                entity_mention_level="partial",
                difficulty="medium",
                note="띄어쓰기 제거/통용 약칭/한글 별칭 기반 alias query.",
            ),
        )


def _character_candidates(
    chunk: Mapping[str, Any], by_type: dict[str, list[Candidate]]
) -> None:
    if str(chunk.get("section_type") or "") != "character":
        return
    text = str(chunk.get("chunk_text") or "")
    entities = _line_entities(text, limit=5)
    if len(entities) < 2:
        return
    e1, e2 = entities[0], entities[1]
    templates = [
        f"{e1}하고 {e2} 성우진 같이 나오는 애니 문서 찾아줘",
        f"{e1} 나오는 작품 캐릭터 정리 어디였지?",
        f"{e1}하고 {e2}가 같이 적힌 등장 캐릭터 문서 찾고 싶어",
    ]
    for idx, q in enumerate(templates[:2]):
        _add_candidate(
            by_type,
            _make_row(
                query=q,
                chunk=chunk,
                query_type="character_question",
                title_mention_level="none",
                entity_mention_level="explicit",
                difficulty="medium" if idx == 0 else "hard",
                note="작품명 대신 캐릭터/성우 표면형으로 목표 문서를 가리킴.",
            ),
        )


def _setting_candidates(
    chunk: Mapping[str, Any], by_type: dict[str, list[Candidate]]
) -> None:
    section_type = str(chunk.get("section_type") or "")
    if section_type not in {"setting", "worldview", "concept"}:
        return
    section_terms = _section_setting_terms(chunk.get("section_path") or [], limit=3)
    text_terms = _term_entities(str(chunk.get("chunk_text") or ""), limit=4, allow_regex=False)
    terms = section_terms or text_terms
    if not terms:
        return
    title = _query_anchor_title(chunk)
    surface, level = _title_surface(title)
    if len(terms) >= 2:
        q = f"{terms[0]}랑 {terms[1]} 설정이 같이 나오는 세계관 문서 찾아줘"
    else:
        q = f"{terms[0]} 설정이 나오는 세계관 문서 찾아줘"
    _add_candidate(
        by_type,
        _make_row(
            query=q,
            chunk=chunk,
            query_type="setting_question",
            title_mention_level="none",
            entity_mention_level="explicit",
            difficulty="hard",
            note="작품명보다 section_path의 설정 항목명을 전면에 둔 검색창식 질문.",
        ),
    )

    if surface and level != "exact_title":
        q2 = f"{surface}에서 {terms[0]} 설정이 어디에 정리돼 있지?"
        _add_candidate(
            by_type,
            _make_row(
                query=q2,
                chunk=chunk,
                query_type="setting_question",
                title_mention_level=level,
                entity_mention_level="explicit",
                difficulty="medium",
                note="짧은 작품 표면형과 설정 용어를 결합함.",
            ),
        )


def _plot_candidates(chunk: Mapping[str, Any], by_type: dict[str, list[Candidate]]) -> None:
    if str(chunk.get("section_type") or "") != "synopsis":
        return
    section_path = list(chunk.get("section_path") or [])
    if section_path and section_path[0] in {"회차 목록", "음악", "주제가"}:
        return
    title = _query_anchor_title(chunk)
    surface, _ = _title_surface(title)
    for sentence in _sentence_candidates(str(chunk.get("chunk_text") or ""))[:2]:
        fragment = _remove_title_from_fragment(
            sentence,
            [title, _expected_title(chunk), surface],
        )
        fragment = _trim_memory_fragment(fragment)
        if len(fragment) < 25:
            continue
        q = f"{fragment} 이런 내용 나오는 애니 뭐였더라"
        _add_candidate(
            by_type,
            _make_row(
                query=q,
                chunk=chunk,
                query_type="plot_memory",
                title_mention_level="none",
                entity_mention_level="implicit",
                difficulty="hard",
                evidence_text=sentence,
                note="줄거리 문장을 제목 없이 기억 단서 형태로 변환.",
            ),
        )
        break


def _evaluation_candidates(
    chunk: Mapping[str, Any], by_type: dict[str, list[Candidate]]
) -> None:
    section_type = str(chunk.get("section_type") or "")
    if section_type not in {"evaluation", "summary", "trivia"}:
        return
    if section_type != "evaluation":
        return
    text = str(chunk.get("chunk_text") or "")
    sentences = _sentence_candidates(text)
    if not sentences:
        return
    title = _query_anchor_title(chunk)
    surface, _ = _title_surface(title)

    for sentence in sentences:
        if any(key in sentence for key in ("평가", "호평", "비판", "작화", "연출", "완성도", "흥행", "인기")):
            if any(noise in sentence for noise in ("평점", "별점", "XXX위", "자세한 내용은")):
                continue
            fragment = _remove_title_from_fragment(sentence, [title, _expected_title(chunk), surface])
            fragment = _trim_memory_fragment(fragment)
            q = f"{fragment} 이런 평가 받던 애니가 뭐였지?"
            _add_candidate(
                by_type,
                _make_row(
                    query=q,
                    chunk=chunk,
                    query_type="vague_recall",
                    title_mention_level="none",
                    entity_mention_level="implicit",
                    difficulty="hard",
                    evidence_text=sentence,
                    note="평가 문장을 작품명 없이 희미한 기억형 query로 변환.",
                ),
            )
            break

    terms = _term_entities(text, limit=3)
    if terms:
        q2 = f"{terms[0]} 얘기가 나오는 분위기나 주제 설명을 찾고 있어"
        _add_candidate(
            by_type,
            _make_row(
                query=q2,
                chunk=chunk,
                query_type="theme_question",
                title_mention_level="none",
                entity_mention_level="partial",
                difficulty="hard",
                note="제목 대신 주제/평가 키워드 중심으로 생성.",
            ),
        )

    for sentence in sentences:
        if any(key in sentence for key in ("원작", "전작", "후속작", "비해", "달리", "비교")):
            fragment = _remove_title_from_fragment(sentence, [title, _expected_title(chunk), surface])
            fragment = _trim_memory_fragment(fragment)
            q3 = f"원작이나 다른 판본이랑 비교해서 {fragment} 이런 설명 있던 문서 찾아줘"
            _add_candidate(
                by_type,
                _make_row(
                    query=q3,
                    chunk=chunk,
                    query_type="comparison_like",
                    title_mention_level="none",
                    entity_mention_level="implicit",
                    difficulty="hard",
                    evidence_text=sentence,
                    note="비교 표현이 있는 근거 문장을 자연 검색어로 변환.",
                ),
            )
            break


def _wrong_assumption_candidates(
    chunk: Mapping[str, Any], by_type: dict[str, list[Candidate]]
) -> None:
    if str(chunk.get("section_type") or "") != "summary":
        return
    text = _clean_text(str(chunk.get("chunk_text") or ""))
    match = re.search(r"방영 시기는\s*(\d{4})년", text)
    if not match:
        return
    actual_year = int(match.group(1))
    wrong_year = actual_year + 1
    title = _query_anchor_title(chunk)
    surface, level = _title_surface(title)
    if not surface or level == "exact_title":
        return
    q = f"{surface} 애니 {wrong_year}년에 방영 시작한 거 맞아?"
    _add_candidate(
        by_type,
        _make_row(
            query=q,
            chunk=chunk,
            query_type="wrong_assumption",
            title_mention_level=level,
            entity_mention_level="none",
            difficulty="hard",
            answerability="partially_answerable",
            evidence_text=f"실제 근거에는 방영 시기가 {actual_year}년으로 적혀 있다. {text}",
            note="방영 연도를 일부러 한 해 틀리게 둔 wrong-assumption query.",
        ),
    )


def _ambiguous_candidates(
    chunk: Mapping[str, Any], by_type: dict[str, list[Candidate]]
) -> None:
    text = _clean_text(str(chunk.get("chunk_text") or ""))
    if not text:
        return
    if "건담" in text and "선라이즈" in text:
        q = "건담 패러디가 많고 선라이즈 얘기도 나오는 애니가 뭐였지?"
    elif "건담" in text and "패러디" in text:
        q = "건담 패러디가 유난히 많았던 애니 쪽 문서 찾고 싶어"
    elif "학생회" in text and "고백" in text:
        q = "학생회에서 서로 고백 문제로 머리싸움하는 러브코미디 찾고 있어"
    elif "고아원" in text and "엄마" in text:
        q = "고아원에서 엄마라고 부르던 사람이 부모가 아니었던 그 애니 뭐였지?"
    elif "위장 가족" in text and ("스파이" in text or "아냐" in text):
        q = "스파이가 위장 가족 만들고 아이가 마음 읽는 애니 문서 찾고 있어"
    elif "농구선수" in text and "벚꽃" in text:
        q = "농구 그만둔 불량아가 벚꽃길에서 여자애 만나는 애니가 뭐였지?"
    elif "총을 든 학생" in text and "학원도시" in text:
        q = "총 든 학생들이 다니는 거대 학원도시 배경 애니 찾고 있어"
    elif "폐교" in text and "대책위원회" in text:
        q = "폐교 위기 학교를 살리려는 대책위원회 나오는 애니 찾아줘"
    elif "5대국" in text and "카게" in text:
        q = "닌자 마을 수장을 5대국에서 뽑는다는 설정 문서가 뭐였지?"
    elif "GN 드라이브" in text and "궤도 엘리베이터" in text:
        q = "GN 드라이브랑 궤도 엘리베이터 설정 같이 나오는 건담 문서가 뭐였지?"
    elif "연극부" in text and "나기사" in text:
        q = "연극부 만들려는 몸 약한 여자애 나오는 학원물 애니가 뭐였지?"
    else:
        return
    _add_candidate(
        by_type,
        _make_row(
            query=q,
            chunk=chunk,
            query_type="ambiguous",
            title_mention_level="none",
            entity_mention_level="partial",
            difficulty="hard",
            answerability="partially_answerable",
            note="복수 문서와 충돌할 수 있는 단서만 남긴 ambiguous query.",
        ),
    )


def _load_pages(path: Path) -> dict[str, PageMeta]:
    pages: dict[str, PageMeta] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            rec = json.loads(line)
            page_id = str(rec.get("page_id") or "")
            if not page_id:
                continue
            pages[page_id] = PageMeta(
                page_id=page_id,
                page_title=str(rec.get("page_title") or ""),
                work_title=str(rec.get("work_title") or ""),
                display_title=str(rec.get("display_title") or ""),
                retrieval_title=str(rec.get("retrieval_title") or ""),
                page_type=str(rec.get("page_type") or ""),
                relation=str(rec.get("relation") or ""),
                aliases=tuple(str(x) for x in rec.get("aliases") or []),
            )
    return pages


def _collect_candidates(rag_chunks: Path) -> dict[str, list[Candidate]]:
    by_type: dict[str, list[Candidate]] = defaultdict(list)
    with rag_chunks.open("r", encoding="utf-8") as fp:
        for line in fp:
            chunk = json.loads(line)
            if not _good_chunk(chunk):
                continue
            _direct_candidates(chunk, by_type)
            _character_candidates(chunk, by_type)
            _setting_candidates(chunk, by_type)
            _plot_candidates(chunk, by_type)
            _evaluation_candidates(chunk, by_type)
            _wrong_assumption_candidates(chunk, by_type)
            _ambiguous_candidates(chunk, by_type)
    return by_type


def _select_candidates(
    by_type: Mapping[str, list[Candidate]],
    *,
    target_counts: Mapping[str, int],
    seed: int,
) -> list[Candidate]:
    rng = random.Random(seed)
    used_queries: set[str] = set()
    used_chunks: set[str] = set()
    selected: list[Candidate] = []

    for qtype, target in target_counts.items():
        pool = list(by_type.get(qtype) or [])
        rng.shuffle(pool)
        picked = 0
        for cand in pool:
            qnorm = _norm(cand.query)
            chunk_key = "|".join(cand.expected_chunk_ids)
            if qnorm in used_queries:
                continue
            # Prefer distinct chunks, but allow reuse if a later fill pass needs it.
            if chunk_key in used_chunks:
                continue
            selected.append(cand)
            used_queries.add(qnorm)
            used_chunks.add(chunk_key)
            picked += 1
            if picked >= target:
                break
        if picked < target:
            for cand in pool:
                qnorm = _norm(cand.query)
                if qnorm in used_queries:
                    continue
                selected.append(cand)
                used_queries.add(qnorm)
                picked += 1
                if picked >= target:
                    break
        if picked < target:
            raise SystemExit(
                f"not enough candidates for {qtype}: needed {target}, got {picked}"
            )

    rng.shuffle(selected)
    return selected


def _to_rows(candidates: Iterable[Candidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates, start=1):
        row = {
            "query_id": f"v4-silver-natural-{idx:04d}",
            "query": cand.query,
            "expected_doc_id": cand.expected_doc_id,
            "expected_title": cand.expected_title,
            "expected_section_path": cand.expected_section_path,
            "expected_chunk_ids": cand.expected_chunk_ids,
            "query_type": cand.query_type,
            "title_mention_level": cand.title_mention_level,
            "entity_mention_level": cand.entity_mention_level,
            "difficulty": cand.difficulty,
            "answerability": cand.answerability,
            "source_evidence": cand.source_evidence,
            "generation_note": cand.generation_note,
        }
        rows.append(row)
    return rows


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    pages: Mapping[str, PageMeta],
    rag_chunks: Path,
    target_total: int,
) -> list[str]:
    chunk_to_doc: dict[str, str] = {}
    with rag_chunks.open("r", encoding="utf-8") as fp:
        for line in fp:
            rec = json.loads(line)
            chunk_to_doc[str(rec.get("chunk_id") or "")] = str(rec.get("doc_id") or "")

    errors: list[str] = []
    if len(rows) != target_total:
        errors.append(f"expected {target_total} rows, got {len(rows)}")
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("query_id") or "")
        if not qid:
            errors.append(f"row {idx} missing query_id")
        if qid in seen_ids:
            errors.append(f"duplicate query_id {qid}")
        seen_ids.add(qid)

        qtype = row.get("query_type")
        if qtype not in QUERY_TYPES:
            errors.append(f"{qid} bad query_type {qtype}")
        if row.get("title_mention_level") not in TITLE_LEVELS:
            errors.append(f"{qid} bad title_mention_level")
        if row.get("entity_mention_level") not in ENTITY_LEVELS:
            errors.append(f"{qid} bad entity_mention_level")
        if row.get("difficulty") not in DIFFICULTIES:
            errors.append(f"{qid} bad difficulty")
        if row.get("answerability") not in ANSWERABILITY:
            errors.append(f"{qid} bad answerability")
        if not str(row.get("query") or "").strip():
            errors.append(f"{qid} empty query")
        doc_id = row.get("expected_doc_id")
        if row.get("answerability") != "unanswerable":
            if not doc_id or doc_id not in pages:
                errors.append(f"{qid} missing expected doc {doc_id}")
            chunk_ids = row.get("expected_chunk_ids") or []
            if not chunk_ids:
                errors.append(f"{qid} missing chunk ids")
            for chunk_id in chunk_ids:
                actual_doc = chunk_to_doc.get(str(chunk_id))
                if not actual_doc:
                    errors.append(f"{qid} missing chunk {chunk_id}")
                elif actual_doc != doc_id:
                    errors.append(f"{qid} chunk/doc mismatch {chunk_id}")
        if _contains_exact_title(str(row.get("query") or ""), row.get("expected_title")):
            errors.append(f"{qid} query contains exact expected_title")
    return errors


def _write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary(
    *,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    summary_json: Path,
    summary_md: Path,
) -> None:
    counts = {
        "query_type": Counter(row["query_type"] for row in rows),
        "title_mention_level": Counter(row["title_mention_level"] for row in rows),
        "difficulty": Counter(row["difficulty"] for row in rows),
        "answerability": Counter(row["answerability"] for row in rows),
    }
    payload = {
        "total": len(rows),
        "counts": {k: dict(v) for k, v in counts.items()},
        "exact_title_ratio": (
            counts["title_mention_level"].get("exact_title", 0) / len(rows)
            if rows
            else 0.0
        ),
        "validation": manifest["validation"],
        "sample": rows[:10],
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Phase 7 v4 Natural Silver 500 Summary",
        "",
        f"- total: {len(rows)}",
        f"- exact_title_ratio: {payload['exact_title_ratio']:.3f}",
        f"- validation_errors: {len(manifest['validation']['errors'])}",
        "",
        "## query_type",
    ]
    for key, value in sorted(counts["query_type"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## title_mention_level"])
    for key, value in sorted(counts["title_mention_level"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## difficulty"])
    for key, value in sorted(counts["difficulty"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## answerability"])
    for key, value in sorted(counts["answerability"].items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.target_total != sum(TARGET_COUNTS.values()):
        raise SystemExit(
            f"this profile expects --target-total {sum(TARGET_COUNTS.values())}"
        )
    corpus_dir = args.corpus_dir
    pages_v4 = corpus_dir / "pages_v4.jsonl"
    rag_chunks = corpus_dir / "rag_chunks.jsonl"
    if not pages_v4.exists():
        raise SystemExit(f"missing pages_v4: {pages_v4}")
    if not rag_chunks.exists():
        raise SystemExit(f"missing rag_chunks: {rag_chunks}")

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = report_dir / "queries_v4_silver_natural_500.jsonl"
    manifest_path = report_dir / "manifest.json"
    summary_json = report_dir / "queries_v4_silver_natural_500.summary.json"
    summary_md = report_dir / "queries_v4_silver_natural_500.summary.md"

    pages = _load_pages(pages_v4)
    candidates = _collect_candidates(rag_chunks)
    selected = _select_candidates(
        candidates,
        target_counts=TARGET_COUNTS,
        seed=args.seed,
    )
    rows = _to_rows(selected)
    errors = _validate_rows(
        rows,
        pages=pages,
        rag_chunks=rag_chunks,
        target_total=args.target_total,
    )
    if errors:
        raise SystemExit("validation failed:\n" + "\n".join(errors[:50]))

    _write_jsonl(rows, out_jsonl)

    counts = {
        "query_type": dict(Counter(row["query_type"] for row in rows)),
        "title_mention_level": dict(
            Counter(row["title_mention_level"] for row in rows)
        ),
        "difficulty": dict(Counter(row["difficulty"] for row in rows)),
        "answerability": dict(Counter(row["answerability"] for row in rows)),
    }
    manifest = {
        "generator": Path(__file__).as_posix(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "target_total": args.target_total,
        "source_artifacts": {
            "pages_v4": {
                "path": str(pages_v4),
                "size_bytes": pages_v4.stat().st_size,
                "sha256": _sha256(pages_v4),
            },
            "rag_chunks": {
                "path": str(rag_chunks),
                "size_bytes": rag_chunks.stat().st_size,
                "sha256": _sha256(rag_chunks),
            },
        },
        "outputs": {
            "queries": str(out_jsonl),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
        "target_counts": TARGET_COUNTS,
        "actual_counts": counts,
        "validation": {
            "errors": errors,
            "exact_title_count": counts["title_mention_level"].get(
                "exact_title", 0
            ),
            "exact_title_ratio": counts["title_mention_level"].get(
                "exact_title", 0
            )
            / len(rows),
            "doc_and_chunk_ids_verified": True,
            "note": (
                "Queries are generated from rag_chunks.jsonl and page metadata "
                "from pages_v4.jsonl; exact expected_title string matches in "
                "query text are rejected."
            ),
        },
        "candidate_pool_counts": {
            key: len(value) for key, value in sorted(candidates.items())
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(
        rows=rows,
        manifest=manifest,
        summary_json=summary_json,
        summary_md=summary_md,
    )

    print(json.dumps({
        "queries": str(out_jsonl),
        "manifest": str(manifest_path),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "counts": counts,
        "validation": manifest["validation"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
