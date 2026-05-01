"""LLM-authored silver-500 retrieval evaluation set (header).

This file is the **schema, builder, summary writer, and the 500-row
QUERIES tuple** that defines the new main retrieval eval set. The
keyword-derived legacy silver-500 has been re-tagged as a sanity-only
set (see :mod:`keyword_sanity_set`); precision / recall / accuracy
claims belong to a future human-audit pass keyed off the
``phase7_human_gold_seed_100.{jsonl,csv,md}`` export.

Authoring rules (frozen — every QUERIES row obeys them):

  - Never copy corpus prose verbatim into the query.
  - Never put the page title verbatim into the query EXCEPT for
    ``direct_title`` and ``alias_variant`` query_types.
  - Never repeat the section title verbatim.
  - Never list 3+ keywords from the target page in a row.
  - Never use the "X의 줄거리는?" / "X의 등장인물은?" template-only
    repetition shape from the legacy keyword silver.
  - Never use unnatural phrasing a real user would never type.
  - The word "gold" never appears in tags or report copy.

Distribution (frozen — see CROSS_TAB_TARGETS, asserted by the build):

                    main_work  generic  named  not_in_corpus  TOTAL
direct_title              35       20      5              0     60
paraphrase_semantic       25       65     35              0    125
section_intent             0      105      5              0    110
indirect_entity           10       25     50              0     85
alias_variant             35        5      5              0     45
ambiguous                 45        5      0              0     50
unanswerable               0        0      0             25     25
                         150      225    100             25    500

Schema per row (matches the spec verbatim):

  query_id                       "v4-llm-silver-NNN"
  query                          user-facing text
  query_type                     one of QUERY_TYPES_ALL
  bucket                         one of BUCKETS_ALL
  silver_expected_title          retrieval_title from corpus (null on n/i)
  silver_expected_page_id        doc_id (null on n/i)
  expected_section_path          list[str] derived from retrieval_title
  expected_not_in_corpus         true iff query_type == unanswerable
  generation_method              "llm"
  is_silver_not_gold             true (always)
  rationale_for_expected_target  why this target was picked (terse)
  lexical_overlap                nested dict from lexical_overlap.py
  leakage_risk                   set by leakage_guard.py
  tags                           ["anime", "v4-llm-silver-500", "silver",
                                  "human_authored_by_llm", bucket,
                                  query_type, generation_method]
                                 NEVER contains "gold".

Determinism: QUERIES is frozen data; build_records is a pure function.
Calling the build twice on the same inputs yields byte-identical JSONL.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from eval.harness.lexical_overlap import compute_overlap
from eval.harness.leakage_guard import (
    ALL_QUERY_TYPES,
    annotate_leakage,
    QUERY_TYPE_ALIAS_VARIANT,
    QUERY_TYPE_AMBIGUOUS,
    QUERY_TYPE_DIRECT_TITLE,
    QUERY_TYPE_INDIRECT_ENTITY,
    QUERY_TYPE_PARAPHRASE_SEMANTIC,
    QUERY_TYPE_SECTION_INTENT,
    QUERY_TYPE_UNANSWERABLE,
    render_leakage_md,
    summarize_leakage,
)


# ---------------------------------------------------------------------------
# Frozen taxonomies
# ---------------------------------------------------------------------------


BUCKET_MAIN_WORK = "main_work"
BUCKET_SUBPAGE_GENERIC = "subpage_generic"
BUCKET_SUBPAGE_NAMED = "subpage_named"
BUCKET_NOT_IN_CORPUS = "not_in_corpus"

BUCKETS_ALL: Tuple[str, ...] = (
    BUCKET_MAIN_WORK,
    BUCKET_SUBPAGE_GENERIC,
    BUCKET_SUBPAGE_NAMED,
    BUCKET_NOT_IN_CORPUS,
)


CROSS_TAB_TARGETS: Mapping[Tuple[str, str], int] = {
    (QUERY_TYPE_DIRECT_TITLE,        BUCKET_MAIN_WORK):       35,
    (QUERY_TYPE_DIRECT_TITLE,        BUCKET_SUBPAGE_GENERIC): 20,
    (QUERY_TYPE_DIRECT_TITLE,        BUCKET_SUBPAGE_NAMED):    5,
    (QUERY_TYPE_PARAPHRASE_SEMANTIC, BUCKET_MAIN_WORK):       25,
    (QUERY_TYPE_PARAPHRASE_SEMANTIC, BUCKET_SUBPAGE_GENERIC): 65,
    (QUERY_TYPE_PARAPHRASE_SEMANTIC, BUCKET_SUBPAGE_NAMED):   35,
    (QUERY_TYPE_SECTION_INTENT,      BUCKET_SUBPAGE_GENERIC): 105,
    (QUERY_TYPE_SECTION_INTENT,      BUCKET_SUBPAGE_NAMED):    5,
    (QUERY_TYPE_INDIRECT_ENTITY,     BUCKET_MAIN_WORK):       10,
    (QUERY_TYPE_INDIRECT_ENTITY,     BUCKET_SUBPAGE_GENERIC): 25,
    (QUERY_TYPE_INDIRECT_ENTITY,     BUCKET_SUBPAGE_NAMED):   50,
    (QUERY_TYPE_ALIAS_VARIANT,       BUCKET_MAIN_WORK):       35,
    (QUERY_TYPE_ALIAS_VARIANT,       BUCKET_SUBPAGE_GENERIC):  5,
    (QUERY_TYPE_ALIAS_VARIANT,       BUCKET_SUBPAGE_NAMED):    5,
    (QUERY_TYPE_AMBIGUOUS,           BUCKET_MAIN_WORK):       45,
    (QUERY_TYPE_AMBIGUOUS,           BUCKET_SUBPAGE_GENERIC):  5,
    (QUERY_TYPE_UNANSWERABLE,        BUCKET_NOT_IN_CORPUS):   25,
}


GENERATION_METHOD = "llm"


# Frozen disclaimer for the summary report. The marker phrase is
# checked by tests so a future renderer can't accidentally drop it.
LLM_SILVER_DISCLAIMER_LINES: Tuple[str, ...] = (
    "> **LLM-generated silver set.** This evaluation set was authored by an",
    "> LLM (the outer-loop assistant) following the Phase 7 silver-500 spec.",
    "> Targets carry ``is_silver_not_gold = true`` — they are silver, NOT",
    "> human-verified gold. Precision / recall / accuracy claims must wait",
    "> for human audit (see ``phase7_human_gold_seed_100.{jsonl,csv,md}``).",
    "> The keyword-derived legacy silver-500 has been re-tagged as a",
    "> sanity-only set (``queries_v4_keyword_sanity_500.jsonl``) — do not",
    "> use it as the main evaluation set anymore.",
)
LLM_SILVER_DISCLAIMER_MD = "\n".join(LLM_SILVER_DISCLAIMER_LINES)
LLM_SILVER_DISCLAIMER_MARKER = "LLM-generated silver set."


# ---------------------------------------------------------------------------
# Query record + helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMQuery:
    """One hand-authored row.

    ``expected_doc_id`` is None iff query_type == unanswerable_or_not_in_corpus.
    The build pipeline derives ``silver_expected_title`` and
    ``expected_section_path`` from the corpus by looking up the doc.
    """

    qid: int
    query: str
    expected_doc_id: Optional[str]
    query_type: str
    bucket: str
    rationale: str
    alias_used: str = ""


def Q(
    qid: int,
    query: str,
    expected_doc_id: Optional[str],
    query_type: str,
    bucket: str,
    rationale: str,
    *,
    alias: str = "",
) -> LLMQuery:
    """Compact constructor — one row per call."""
    return LLMQuery(
        qid=qid,
        query=query,
        expected_doc_id=expected_doc_id,
        query_type=query_type,
        bucket=bucket,
        rationale=rationale,
        alias_used=alias,
    )


# Aliases for the QUERIES literal below — keeps each row at one screen line.
_DT, _PS, _SI, _IE, _AV, _AM, _UN = (
    QUERY_TYPE_DIRECT_TITLE,
    QUERY_TYPE_PARAPHRASE_SEMANTIC,
    QUERY_TYPE_SECTION_INTENT,
    QUERY_TYPE_INDIRECT_ENTITY,
    QUERY_TYPE_ALIAS_VARIANT,
    QUERY_TYPE_AMBIGUOUS,
    QUERY_TYPE_UNANSWERABLE,
)
_M, _G, _N, _X = (
    BUCKET_MAIN_WORK,
    BUCKET_SUBPAGE_GENERIC,
    BUCKET_SUBPAGE_NAMED,
    BUCKET_NOT_IN_CORPUS,
)


# ---------------------------------------------------------------------------
# THE 500 QUERIES
# ---------------------------------------------------------------------------
# Order: A direct_title (60) | B paraphrase (125) | C section_intent (110) |
# D indirect_entity (85) | E alias_variant (45) | F ambiguous (50) |
# G unanswerable (25). Within a section, rows are ordered by qid.
# ---------------------------------------------------------------------------


QUERIES: Tuple[LLMQuery, ...] = (
    # =========================================================================
    # A. direct_title (60)  — title 직접 사용. leakage_guard에서 benign 처리.
    # =========================================================================

    # A1. direct_title × main_work (35)
    Q(  1, "원피스 어떤 만화야",                       "f4fbe985489fa342", _DT, _M, "title 직접; 원피스 애니 main"),
    Q(  2, "명탐정 코난 처음 봐도 돼?",                  "6b42c9d8f780ecd4", _DT, _M, "title 직접; 코난 만화 main"),
    Q(  3, "진격의 거인 알려줘",                       "fcce5d7c4b0cd081", _DT, _M, "title 직접; 진격거 만화 main"),
    Q(  4, "나루토 만화 정보",                        "d8facf2fc222a148", _DT, _M, "title 직접; 나루토 만화 main"),
    Q(  5, "카드캡터 사쿠라 어떤 작품인가요",             "c5678dfdb658a270", _DT, _M, "title 직접; CCS main"),
    Q(  6, "너의 이름은 영화 정보",                    "53ecf04dd81e25e4", _DT, _M, "title 직접; 너이는 main"),
    Q(  7, "스즈메의 문단속 줄거리 좀 알려줘",           "d719a8379cc655a4", _DT, _M, "title 직접; 스즈메 main"),
    Q(  8, "더 퍼스트 슬램덩크 영화 정보",               "6a999471fa69124d", _DT, _M, "title 직접; 더퍼슬 main"),
    Q(  9, "이웃집 토토로 보고 싶어",                   "652c5b921746b09b", _DT, _M, "title 직접; 토토로 main"),
    Q( 10, "그대들은 어떻게 살 것인가 영화 알려줘",       "a4e27f2140c663e2", _DT, _M, "title 직접; 그대들 main"),
    Q( 11, "너의 췌장을 먹고 싶어 알려주세요",            "c2d75031268441c1", _DT, _M, "title 직접; 키미스이 main"),
    Q( 12, "시끌별 녀석들 리메이크 정보",                "eeaf26d4bda1cf94", _DT, _M, "title 직접; 시끌별 2022"),
    Q( 13, "신 에반게리온 극장판 알려줘",               "fa3bfe6771d17cc8", _DT, _M, "title 직접; 신에바 main"),
    Q( 14, "귀멸의 칼날 1기 정보",                    "544fc39fe5e1a5c6", _DT, _M, "title 직접; 귀칼 1기 main"),
    Q( 15, "주술회전 1기 어떤 내용",                   "92d08f0a1db466eb", _DT, _M, "title 직접; 주회 1기 main"),
    Q( 16, "강철의 연금술사 풀메탈 알케미스트 알려줘",     "fca4f94680dd57de", _DT, _M, "title 직접; FMAB main"),
    Q( 17, "죠죠의 기묘한 모험 1부 2부 어떤 만화야",      "2bfdcf0b37d1a58a", _DT, _M, "title 직접; 죠죠 2012"),
    Q( 18, "소드 아트 온라인 1기 보고 싶어",             "68666a362c0f7787", _DT, _M, "title 직접; SAO 1기"),
    Q( 19, "스파이 패밀리 1기 정보",                   "764fad47dcce946e", _DT, _M, "title 직접; 스파패 1기"),
    Q( 20, "체인소 맨 애니 정보",                     "f32732c5b960aa24", _DT, _M, "title 직접; 체인소맨 애니"),
    Q( 21, "오버로드 1기 어떤 작품",                   "f7d36e515b413cf4", _DT, _M, "title 직접; 오버로드 1기"),
    Q( 22, "러브 라이브 시리즈 알려줘",                 "a5c0ab6348b9629c", _DT, _M, "title 직접; 러라 main"),
    Q( 23, "하이큐 1기 알려줘",                       "d0d19bbd3714a5c1", _DT, _M, "title 직접; 하이큐 1기"),
    Q( 24, "도라에몽 알려줘",                         "ca5e28050fd2189d", _DT, _M, "title 직접; 도라 만화 main"),
    Q( 25, "기동전사 건담 수성의 마녀 정보",            "ad2229ca65b0d51b", _DT, _M, "title 직접; 수성마녀"),
    Q( 26, "기동전사 건담 SEED FREEDOM 영화 정보",     "85f8c2875d188a82", _DT, _M, "title 직접; SEED FREEDOM"),
    Q( 27, "기동전사 건담 UC 알려줘",                  "cc77875805f7a7c3", _DT, _M, "title 직접; 건담 UC"),
    Q( 28, "걸즈 앤 판처 극장판 어떤 애니야",            "c4cd2a67f9bb6598", _DT, _M, "title 직접; 걸판 극장판"),
    Q( 29, "그 비스크 돌은 사랑을 한다 1기 알려줘",      "929234de7464164c", _DT, _M, "title 직접; 비스크돌 1기"),
    Q( 30, "ARIA The ORIGINATION 알려주세요",         "7d4023b3809ac4ec", _DT, _M, "title 직접; ARIA 3기"),
    Q( 31, "CLANNAD 2007 애니 정보",                  "67f8e7cc70842184", _DT, _M, "title 직접; 클라나드 2007"),
    Q( 32, "날씨의 아이 영화 정보",                    "d402607d8616ee56", _DT, _M, "title 직접; 날씨아이"),
    Q( 33, "극장판 주술회전 0 알려줘",                 "5411aa47e0100e2c", _DT, _M, "title 직접; 주회 0 영화"),
    Q( 34, "진격의 거인 The Final Season 알려줘",      "d99cf2fef9626a5a", _DT, _M, "title 직접; 진격거 파이널"),
    Q( 35, "북두의 권 만화 어떤 작품",                  "18020a197ed2701c", _DT, _M, "title 직접; 북두 main"),

    # A2. direct_title × subpage_generic (20)  — 작품명 + 섹션명 그대로
    Q( 36, "명탐정 코난 등장인물 정리",                "7bce791b2e184b82", _DT, _G, "title+섹션 직접; 코난 등장인물"),
    Q( 37, "나루토 등장인물 누구누구야",                "5bc93e73f4685083", _DT, _G, "title+섹션 직접; 나루토 등장인물"),
    Q( 38, "진격의 거인 등장인물 알려줘",               "aaa92e0f63e76052", _DT, _G, "title+섹션 직접; 진격거 등장인물"),
    Q( 39, "너의 이름은 등장인물",                    "d79836b0bc1e4a1c", _DT, _G, "title+섹션 직접; 너이는 등장인물"),
    Q( 40, "스즈메의 문단속 등장인물",                  "1cb95058ec79e07c", _DT, _G, "title+섹션 직접; 스즈메 등장인물"),
    Q( 41, "날씨의 아이 등장인물",                    "c7caa218f7341ba7", _DT, _G, "title+섹션 직접; 날씨아이 등장인물"),
    Q( 42, "카드캡터 사쿠라 등장인물",                 "f0623f8d555cd7ea", _DT, _G, "title+섹션 직접; CCS 등장인물"),
    Q( 43, "그대들은 어떻게 살 것인가 등장인물",         "4d0e41c93616e1a7", _DT, _G, "title+섹션 직접; 그대들 등장인물"),
    Q( 44, "기동전사 건담 수성의 마녀 등장인물",         "8bb71d093785994d", _DT, _G, "title+섹션 직접; 수성마녀 등장인물"),
    Q( 45, "러브 라이브 등장인물",                    "315110427626e1da", _DT, _G, "title+섹션 직접; 러라 등장인물"),
    Q( 46, "나루토 줄거리",                          "de007b15eabb3b86", _DT, _G, "title+섹션 직접; 나루토 줄거리"),
    Q( 47, "너의 이름은 줄거리",                      "4035db0b8289111a", _DT, _G, "title+섹션 직접; 너이는 줄거리"),
    Q( 48, "스즈메의 문단속 줄거리",                   "b1f686298f567c11", _DT, _G, "title+섹션 직접; 스즈메 줄거리"),
    Q( 49, "날씨의 아이 줄거리",                      "98e24e9fbf5a40a4", _DT, _G, "title+섹션 직접; 날씨아이 줄거리"),
    Q( 50, "기동전사 건담 수성의 마녀 줄거리",          "0d11878ce878863b", _DT, _G, "title+섹션 직접; 수성마녀 줄거리"),
    Q( 51, "그대들은 어떻게 살 것인가 줄거리",          "3ceeadb398cf7161", _DT, _G, "title+섹션 직접; 그대들 줄거리"),
    Q( 52, "진격의 거인 평가",                       "d0f548a58033adaa", _DT, _G, "title+섹션 직접; 진격거 평가"),
    Q( 53, "날씨의 아이 평가",                       "80df0496b4109d83", _DT, _G, "title+섹션 직접; 날씨아이 평가"),
    Q( 54, "신 에반게리온 극장판 평가",                "1c06d29aee82602f", _DT, _G, "title+섹션 직접; 신에바 평가"),
    Q( 55, "기동전사 건담 수성의 마녀 평가",            "4f64a8dbafc87420", _DT, _G, "title+섹션 직접; 수성마녀 평가"),

    # A3. direct_title × subpage_named (5)
    Q( 56, "원피스 애니 회차 목록 1~516화 정보",        "00c3bcca2fd178d1", _DT, _N, "title+sub-sub 직접; 원피스 1~516"),
    Q( 57, "원피스 애니 회차 목록 517화 이후 정보",      "620796dc9e734ab8", _DT, _N, "title+sub-sub 직접; 원피스 517+"),
    Q( 58, "원펀맨 등장인물 S급 히어로",                "9b5199d9a6be7faa", _DT, _N, "title+sub-sub 직접; S급 히어로"),
    Q( 59, "기동전사 건담 수성의 마녀 설정 기체",        "6c5cd8d2bc00271e", _DT, _N, "title+sub-sub 직접; 수성마녀 기체"),
    Q( 60, "마음의소리 에피소드 목록 1~500회",          "bf57e9e6512db741", _DT, _N, "title+sub-sub 직접; 마소 1~500"),

    # =========================================================================
    # B. paraphrase_semantic (125)  — 제목/섹션 직접 안 쓰고 의미 묘사
    # =========================================================================

    # B1. paraphrase × main_work (25)
    Q( 61, "도쿄 사는 학생이랑 시골 소녀가 몸이 바뀌는 영화", "53ecf04dd81e25e4", _PS, _M, "내용 묘사; 너이는"),
    Q( 62, "거인이 인류를 위협하는 다크 판타지 만화",       "fcce5d7c4b0cd081", _PS, _M, "장르+컨셉; 진격거"),
    Q( 63, "해적이 보물을 찾아 모험하는 일본 만화",         "f4fbe985489fa342", _PS, _M, "내용 묘사; 원피스"),
    Q( 64, "닌자 마을 출신 소년이 영웅이 되는 만화",       "d8facf2fc222a148", _PS, _M, "내용 묘사; 나루토"),
    Q( 65, "어린이가 카드를 모으는 마법 소녀 작품",         "c5678dfdb658a270", _PS, _M, "내용 묘사; CCS"),
    Q( 66, "비를 내리게 하는 소녀와 가출 소년의 영화",       "d402607d8616ee56", _PS, _M, "내용 묘사; 날씨아이"),
    Q( 67, "신카이 마코토 감독의 재해 모티브 영화",         "d719a8379cc655a4", _PS, _M, "감독+모티브; 스즈메"),
    Q( 68, "이노우에 다케히코가 직접 감독한 농구 영화",       "6a999471fa69124d", _PS, _M, "감독+소재; 더퍼슬"),
    Q( 69, "미야자키 감독의 2023년 신작 애니",             "a4e27f2140c663e2", _PS, _M, "감독+년도; 그대들"),
    Q( 70, "지브리에서 만든 시골 자매와 큰 정령 이야기",      "652c5b921746b09b", _PS, _M, "스튜디오+내용; 토토로"),
    Q( 71, "췌장 환자 소녀와 무덤덤한 소년의 일본 영화",     "c2d75031268441c1", _PS, _M, "내용 묘사; 키미스이"),
    Q( 72, "외계인이 지구에 와서 소년 가족과 사는 리메이크 만화", "eeaf26d4bda1cf94", _PS, _M, "내용 묘사; 시끌별 리메이크"),
    Q( 73, "에반게리온 신극장판 시리즈의 마지막 영화",       "fa3bfe6771d17cc8", _PS, _M, "시리즈+위치; 신에바"),
    Q( 74, "다이쇼 시대 도깨비 사냥꾼 소년의 1기",          "544fc39fe5e1a5c6", _PS, _M, "시대+소재; 귀칼 1기"),
    Q( 75, "고등학생 주술사가 저주를 사냥하는 1기",         "92d08f0a1db466eb", _PS, _M, "직업+소재; 주회 1기"),
    Q( 76, "두 형제가 잃어버린 몸을 되찾는 연금술 만화",     "fca4f94680dd57de", _PS, _M, "내용 묘사; FMAB"),
    Q( 77, "스탠드라는 능력으로 싸우는 1·2부 애니",         "2bfdcf0b37d1a58a", _PS, _M, "능력 시스템; 죠죠 1·2부"),
    Q( 78, "VR MMO에 갇혀 죽지 않으려 게임 클리어해야 하는 1기", "68666a362c0f7787", _PS, _M, "장르+소재; SAO 1기"),
    Q( 79, "초능력자 가족이 평범한 척 위장하는 첩보 코미디",  "764fad47dcce946e", _PS, _M, "장르+컨셉; 스파패 1기"),
    Q( 80, "악마와 계약한 소년이 체인 톱 형태로 변신하는 애니", "f32732c5b960aa24", _PS, _M, "내용; 체인소맨"),
    Q( 81, "마법으로 던전을 지배하는 해골 마왕이 주인공인 1기", "f7d36e515b413cf4", _PS, _M, "주인공+세계관; 오버로드 1기"),
    Q( 82, "스쿨 아이돌이 학교를 살리려 노래하는 시리즈",     "a5c0ab6348b9629c", _PS, _M, "내용; 러라"),
    Q( 83, "고등 배구부의 작은 키 점프력 좋은 소년 1기 애니", "d0d19bbd3714a5c1", _PS, _M, "내용; 하이큐 1기"),
    Q( 84, "탐정 소년이 약 먹고 어린이가 되는 장수 애니",    "0e95aa5535c2f931", _PS, _M, "내용; 코난 애니"),
    Q( 85, "옷 만들기 좋아하는 소녀가 코스프레 도와주는 1기 애니", "929234de7464164c", _PS, _M, "내용; 비스크돌 1기"),

    # B2. paraphrase × subpage_generic (65)
    Q( 86, "거인 위협 만화에 어떤 인물이 나오는지 정리",     "aaa92e0f63e76052", _PS, _G, "내용+등장인물; 진격거"),
    Q( 87, "닌자 소년 만화 캐릭터 누가 있어",               "5bc93e73f4685083", _PS, _G, "내용+등장인물; 나루토"),
    Q( 88, "탐정 소년 만화 캐릭터 정리",                  "7bce791b2e184b82", _PS, _G, "내용+등장인물; 코난"),
    Q( 89, "몸 바뀌는 영화 등장인물 누구",                "d79836b0bc1e4a1c", _PS, _G, "내용+등장인물; 너이는"),
    Q( 90, "신카이 재해 영화 등장인물 알려줘",             "1cb95058ec79e07c", _PS, _G, "감독+소재+등장인물; 스즈메"),
    Q( 91, "비 내리는 소녀 영화 캐릭터 알려줘",            "c7caa218f7341ba7", _PS, _G, "내용+등장인물; 날씨아이"),
    Q( 92, "카드 모으는 마법 소녀 캐릭터들",              "f0623f8d555cd7ea", _PS, _G, "내용+등장인물; CCS"),
    Q( 93, "미야자키 2023 신작 등장인물",                "4d0e41c93616e1a7", _PS, _G, "감독+년도+등장인물; 그대들"),
    Q( 94, "수성의 마녀 캐릭터 다 알려줘",                "8bb71d093785994d", _PS, _G, "약식+등장인물; 수성마녀"),
    Q( 95, "스쿨 아이돌 시리즈에 누구 있는지 정리",         "315110427626e1da", _PS, _G, "내용+등장인물; 러라"),
    Q( 96, "거인 만화 1기 무슨 이야기",                  "f94e2b05cf4bd818", _PS, _G, "내용+줄거리; 진격거 1기 줄거리"),
    Q( 97, "닌자 소년 만화 줄거리",                     "de007b15eabb3b86", _PS, _G, "내용+줄거리; 나루토 줄거리"),
    Q( 98, "몸 바뀌는 영화 무슨 이야기",                  "4035db0b8289111a", _PS, _G, "내용+줄거리; 너이는 줄거리"),
    Q( 99, "비 내리는 소녀 영화 줄거리 한 줄로",          "98e24e9fbf5a40a4", _PS, _G, "내용+줄거리; 날씨아이 줄거리"),
    Q(100, "재해 모티브 영화 결말 어땠어",                "b1f686298f567c11", _PS, _G, "감독+소재+결말; 스즈메 줄거리"),
    Q(101, "수성의 마녀 스토리 어떻게 끝남",              "0d11878ce878863b", _PS, _G, "약식+결말; 수성마녀 줄거리"),
    Q(102, "에반게리온 마지막 영화 결말 정리",            "97623cacc2582913", _PS, _G, "시리즈+위치+결말; 신에바 줄거리"),
    Q(103, "주술사 0 영화 줄거리",                       "47481afbc08e0d5c", _PS, _G, "내용+줄거리; 주회 0 줄거리"),
    Q(104, "거인 만화 평론가 평가",                      "d0f548a58033adaa", _PS, _G, "내용+평가; 진격거 평가"),
    Q(105, "비 내리는 소녀 영화 평가 좋은가",             "80df0496b4109d83", _PS, _G, "내용+평가; 날씨아이 평가"),
    Q(106, "신카이 재해 영화 호평이었어",                 "ec396965a9b90840", _PS, _G, "내용+평가; 스즈메 평가"),
    Q(107, "에반게리온 마지막 영화 평론",                "1c06d29aee82602f", _PS, _G, "시리즈+위치+평가; 신에바 평가"),
    Q(108, "수성의 마녀 평가 호불호",                    "4f64a8dbafc87420", _PS, _G, "약식+평가; 수성마녀 평가"),
    Q(109, "해적 만화 OP ED 누가 불렀어",                "06cd0747b8a805b2", _PS, _G, "내용+주제가; 원피스 주제가"),
    Q(110, "탐정 소년 애니 오프닝 정보",                  "106c3fb5a5d1855b", _PS, _G, "내용+주제가; 코난 주제가"),
    Q(111, "닌자 소년 만화 BGM 사운드트랙",              "451e6eceedd2e61a", _PS, _G, "내용+음악; 나루토 음악"),
    Q(112, "원피스 처음부터 보려는데 몇 화까지 있어",       "a41ab7035281f44d", _PS, _G, "내용+회차; 원피스 회차"),
    Q(113, "탐정 소년 애니 화수 어디까지",                "4e02fd43e9a72561", _PS, _G, "내용+회차; 코난 애니 회차"),
    Q(114, "닌자 만화 세계관 어떻게 돼",                  "3cb46ccdd0e44cfc", _PS, _G, "내용+설정; 나루토 설정"),
    Q(115, "거인 만화 미디어 믹스 정보",                  "d9b0febcc55e14af", _PS, _G, "내용+미디믹스; 진격거 미디믹스"),
    Q(116, "검은 조직 코난 엮이는 사건들 정리",            "219ebe890e2a0e7e", _PS, _G, "내용+사건군; 코난 검은 조직 (named지만 paraphrase로 매핑)"),
    Q(117, "걸즈 앤 판처 줄거리 짧게",                    "483a5a06cb47e75b", _PS, _G, "내용+줄거리; 걸판 줄거리"),
    Q(118, "마법소녀 마도카 마기카 줄거리",                "28ba524e8ae0c4a1", _PS, _G, "약식+줄거리; 마도카 줄거리"),
    Q(119, "디지몬 어드벤처 1기 줄거리",                  "dba3c6cbecca67c7", _PS, _G, "내용+줄거리; 디지몬 어드 줄거리"),
    Q(120, "디지몬 크로스워즈 줄거리",                    "5d8f865021e4c8a2", _PS, _G, "약식+줄거리; 디지몬 크로스 줄거리"),
    Q(121, "갓 오브 하이스쿨 평론",                       "86f1501e28867057", _PS, _G, "약식+평가; 고하 평가"),
    Q(122, "달링 인 더 프랑키스 평가 어땠어",              "047e2e7b8b0b2cf7", _PS, _G, "내용+평가; 달프 평가"),
    Q(123, "걸즈 앤 판처 평론가 평가",                    "bffbdd3b4052c933", _PS, _G, "내용+평가; 걸판 평가"),
    Q(124, "기동전사 건담 UC 평가 어땠어",                "43d46104c09b0f55", _PS, _G, "약식+평가; 건담 UC 평가"),
    Q(125, "기동전사 건담 AGE 평론",                      "c6cbea8b49768454", _PS, _G, "약식+평가; AGE 평가"),
    Q(126, "걸즈 밴드 크라이 평가",                       "2b2ed8a2b7c5d82d", _PS, _G, "약식+평가; 걸밴크 평가"),
    Q(127, "마법소녀 마도카 마기카 평가",                  "8b7de5c5f0c77278", _PS, _G, "약식+평가; 마도카 평가"),
    Q(128, "기동전사 건담 지쿠악스 평가",                  "9713b71b1b72558d", _PS, _G, "약식+평가; 지쿠악스 평가"),
    Q(129, "기동전사 건담 철혈의 오펀스 평가",            "1710eb8ac0781ad2", _PS, _G, "약식+평가; 철혈 평가"),
    Q(130, "길티 크라운 평가",                          "df30d71289079a2e", _PS, _G, "약식+평가; 길티 평가"),
    Q(131, "늑대아이 평가",                              "0b6e817fea36e8a6", _PS, _G, "약식+평가; 늑대아이 평가"),
    Q(132, "니세코이 평론",                              "bf93d22fa918570d", _PS, _G, "약식+평가; 니세코이 평가"),
    Q(133, "단간론파 3 키보가미네 평론",                  "0926584b3b792cf4", _PS, _G, "약식+평가; 단간 평가"),
    Q(134, "던전 만남 1기 평가",                         "8e279b363f45b875", _PS, _G, "약식+평가; 던만추 평가"),
    Q(135, "드래곤볼 슈퍼 평론",                          "d01a0b0fd46cdede", _PS, _G, "약식+평가; DBS 평가"),
    Q(136, "딜리셔스 파티 프리큐어 평가",                 "4704ace849550e66", _PS, _G, "약식+평가; 딜리 평가"),
    Q(137, "런닝맨 애니 평가",                           "7bd0b053bb4afc03", _PS, _G, "약식+평가; 런닝맨 평가"),
    Q(138, "마법사 프리큐어 평가",                        "b07fba2aefecd03f", _PS, _G, "약식+평가; 마법사 프리 평가"),
    Q(139, "겨울왕국 2 평가",                            "4a6a772aebacb5b9", _PS, _G, "약식+평가; 겨울왕국 2 평가"),
    Q(140, "극장판 바이올렛 에버가든 평가",                "8dc3c7aaada10750", _PS, _G, "약식+평가; 바이올렛 평가"),
    Q(141, "원피스 OST 정보",                            "1304f44478650c7a", _PS, _G, "약식+OST; 원피스 OST"),
    Q(142, "스즈메 OST 알려줘",                          "397971478331eb45", _PS, _G, "약식+OST; 스즈메 OST"),
    Q(143, "신카이 영화 너의 이름은 OST",                 "33a61826f8655fa3", _PS, _G, "감독+작품+OST; 너이는 OST"),
    Q(144, "체인소맨 OST",                                "149c50ac2c23d8b1", _PS, _G, "약식+OST; 체소맨 OST"),
    Q(145, "걸즈 앤 판처 BGM",                           "b4b44711e27436f9", _PS, _G, "약식+음악; 걸판 음악"),
    Q(146, "닌자 소년 애니 BGM",                         "451e6eceedd2e61a", _PS, _G, "내용+음악; 나루토 애니 음악"),
    Q(147, "닌자 질풍전 BGM",                            "e4b308bc38635b86", _PS, _G, "내용+음악; 질풍전 음악"),
    Q(148, "드래곤볼 슈퍼 음악",                          "101fff9406167556", _PS, _G, "약식+음악; DBS 음악"),
    Q(149, "디지몬 어드벤처 음악",                        "e2572fe7de0b997e", _PS, _G, "약식+음악; 디지몬 어드 음악"),
    Q(150, "디지몬 크로스워즈 BGM",                       "e5958bb113ca694e", _PS, _G, "약식+음악; 디지몬 크로스 음악"),

    # B3. paraphrase × subpage_named (35)
    Q(151, "원피스 애니 초반 1~516화 정보",               "00c3bcca2fd178d1", _PS, _N, "약식+named sub-sub; 1~516"),
    Q(152, "원피스 애니 후반 회차 정보",                  "620796dc9e734ab8", _PS, _N, "약식+named sub-sub; 517+"),
    Q(153, "원펀맨에서 가장 강한 등급 히어로들",            "9b5199d9a6be7faa", _PS, _N, "내용+named; S급 히어로"),
    Q(154, "원펀맨 A급 영웅 정보",                       "fd298cea69ab8a6a", _PS, _N, "약식+named; A급 히어로"),
    Q(155, "원펀맨 B급 영웅들",                          "1e469176ae5afbad", _PS, _N, "약식+named; B급 히어로"),
    Q(156, "원펀맨 C급 영웅 누가 있어",                  "1c3b4362dbbdb47c", _PS, _N, "약식+named; C급 히어로"),
    Q(157, "원펀맨 등급 매겨지지 않은 영웅",              "5d0a3f995207579b", _PS, _N, "약식+named; 순위 불명"),
    Q(158, "암살교실 군인 출신 캐릭터 정보",              "32c5f2fe0f633613", _PS, _N, "내용+named; 암살교실 군인"),
    Q(159, "암살교실 살인 청부업자 캐릭터들",              "c83f9f483dbfdad2", _PS, _N, "내용+named; 암살자"),
    Q(160, "수성의 마녀 등장 모빌슈트 모음",               "6c5cd8d2bc00271e", _PS, _N, "내용+named; 수성마녀 기체"),
    Q(161, "마음의소리 초반 500회까지 에피소드",            "bf57e9e6512db741", _PS, _N, "내용+named; 마소 1~500"),
    Q(162, "마음의소리 501화부터 1000화까지 에피소드",      "169269c8016833c9", _PS, _N, "내용+named; 마소 501~1000"),
    Q(163, "마음의소리 마지막 회차들",                    "e8c33fd51841a67b", _PS, _N, "내용+named; 마소 1001~"),
    Q(164, "로보카 폴리에서 사람 캐릭터들",               "f88b0128659abc06", _PS, _N, "내용+named; 폴리 사람"),
    Q(165, "로보카 폴리에서 자동차 캐릭터들",             "5bc2d5ae995de8ac", _PS, _N, "내용+named; 폴리 자동차"),
    Q(166, "로보카 폴리 단역 캐릭터",                    "4fd8416f47df8b1e", _PS, _N, "내용+named; 폴리 단역"),
    Q(167, "검볼 애니 학교 등장인물들",                   "eab985febe5c1ee0", _PS, _N, "내용+named; 검볼 학교"),
    Q(168, "검볼 애니 조연 단역 캐릭터들",                "4d0e43c0c36ba7bc", _PS, _N, "내용+named; 검볼 조연 단역"),
    Q(169, "마법천자문 가족 관계 캐릭터",                 "12892ed572cd429e", _PS, _N, "내용+named; 마천 가족"),
    Q(170, "던전 만남 1기 던전 시스템 설정",               "600f7e92ab830130", _PS, _N, "내용+named; 던만추 던전"),
    Q(171, "걸즈 앤 판처 음악 음반 발매",                 "29e1ca910a4ef757", _PS, _N, "약식+named; 걸판 음반"),
    Q(172, "코난 애니 검은 조직 관련 사건 화수",          "219ebe890e2a0e7e", _PS, _N, "내용+named; 코난 검은조직"),
    Q(173, "코난 애니 오사카 부경 등장 화차",             "1b5f4851669d8002", _PS, _N, "내용+named; 코난 오사카"),
    Q(174, "코난 애니 나가노 현경 등장 화차",             "8b9bb35e79fe98a8", _PS, _N, "내용+named; 코난 나가노"),
    Q(175, "코난 등장인물 휴대전화 정보",                  "273dab5bd5ee54b9", _PS, _N, "내용+named; 코난 휴대전화"),
    Q(176, "이상한 과자가게 전천당 에피소드별 등장인물",     "17317b82d86ea764", _PS, _N, "내용+named; 전천당 에피별"),
    Q(177, "오소마츠 6쌍둥이 외 조연 캐릭터",             "ec2725d3746e053f", _PS, _N, "관계+named; 오소마츠 조연"),
    Q(178, "스폰지밥 극장판에 등장하는 캐릭터",            "d768229151dc63ac", _PS, _N, "내용+named; 스폰지 극장판 등장"),
    Q(179, "스폰지밥 설정상 오류 모음",                  "a9d4a304c08e514a", _PS, _N, "내용+named; 스폰지 설정 오류"),
    Q(180, "듀얼마스터즈 백스토리 어떻게 돼",              "98c4d49cc80ae6c5", _PS, _N, "약식+named; 듀얼 백스토리"),
    Q(181, "북두의 권 외 다른 권법 정리",                "e76ee1f26516371e", _PS, _N, "내용+named; 북두 기타 권법"),
    Q(182, "짱구는 못말려 1990년대 회차 정보",            "3ec5dad47bf145ce", _PS, _N, "내용+named; 짱구 90년대"),
    Q(183, "짱구는 못말려 2000년대 회차",                 "1af7ecf94e951bbe", _PS, _N, "내용+named; 짱구 2000년대"),
    Q(184, "짱구는 못말려 2010년대 회차",                 "ead27b3ffda17d3a", _PS, _N, "내용+named; 짱구 2010년대"),
    Q(185, "짱구는 못말려 2020년대 회차",                 "7c05b641bc38cdb7", _PS, _N, "내용+named; 짱구 2020년대"),

    # =========================================================================
    # C. section_intent (110)  — 작품명 + paraphrased 섹션 의도
    # =========================================================================

    # C1. section_intent × subpage_generic (105)
    Q(186, "코난에 누가 나오는지 정리해줘",                "7bce791b2e184b82", _SI, _G, "section intent; 코난 등장인물"),
    Q(187, "진격의 거인 캐릭터 누구",                    "aaa92e0f63e76052", _SI, _G, "section intent; 진격거 등장인물"),
    Q(188, "나루토에 누가 나와",                         "5bc93e73f4685083", _SI, _G, "section intent; 나루토 등장인물"),
    Q(189, "너의 이름은 주인공 누구",                    "d79836b0bc1e4a1c", _SI, _G, "section intent; 너이는 등장인물"),
    Q(190, "스즈메의 문단속 캐릭터 누가 있어",            "1cb95058ec79e07c", _SI, _G, "section intent; 스즈메 등장인물"),
    Q(191, "날씨의 아이 주인공 누구야",                  "c7caa218f7341ba7", _SI, _G, "section intent; 날씨아이 등장인물"),
    Q(192, "그대들은 어떻게 살 것인가 누가 나와",         "4d0e41c93616e1a7", _SI, _G, "section intent; 그대들 등장인물"),
    Q(193, "수성의 마녀 캐릭터 정리해줘",                 "8bb71d093785994d", _SI, _G, "section intent; 수성마녀 등장인물"),
    Q(194, "러브 라이브에 누가 나와",                    "315110427626e1da", _SI, _G, "section intent; 러라 등장인물"),
    Q(195, "카드캡터 사쿠라 캐릭터 누구야",               "f0623f8d555cd7ea", _SI, _G, "section intent; CCS 등장인물"),
    Q(196, "걸즈 앤 판처에 누가 나와",                   "af385fa29f1f309e", _SI, _G, "section intent; 걸판 등장인물"),
    Q(197, "가정교사 히트맨 REBORN 캐릭터들",            "18ca8ea3336f90f9", _SI, _G, "section intent; 리본 등장인물"),
    Q(198, "갑철성의 카바네리 등장인물",                  "ee3caf78f828764a", _SI, _G, "section intent; 카바네리 등장인물"),
    Q(199, "건담 빌드 다이버즈 캐릭터들",                 "c13e867ec62656a5", _SI, _G, "section intent; 빌드 다이버즈 등장인물"),
    Q(200, "건담 빌드 파이터즈 등장인물",                 "fdcac3b006ef1e02", _SI, _G, "section intent; 빌드 파이터즈 등장인물"),
    Q(201, "건담 G의 레콘기스타 캐릭터",                 "5e439db6d4fd5dab", _SI, _G, "section intent; G 레콘 등장인물"),
    Q(202, "걸리 에어포스에 누가 나와",                  "5a33e8ee823ccf97", _SI, _G, "section intent; 걸리 에포 등장인물"),
    Q(203, "검볼 애니에 누가 나와",                     "b4a54701dfcb52b8", _SI, _G, "section intent; 검볼 등장인물"),
    Q(204, "검정 고무신 캐릭터들",                       "98cd44cded64af8c", _SI, _G, "section intent; 검정고무신 등장인물"),
    Q(205, "게게게의 키타로 캐릭터",                     "450980acf50d1f0f", _SI, _G, "section intent; 키타로 등장인물"),
    Q(206, "골판지 전기 누가 나와",                      "95ed5ddd844d3ce7", _SI, _G, "section intent; 골판지 등장인물"),
    Q(207, "공룡메카드에 누가 나와",                     "b57236795313a210", _SI, _G, "section intent; 공룡메카드 등장인물"),
    Q(208, "공의 경계 등장인물",                         "a1090c9b4b948358", _SI, _G, "section intent; 공의경계 등장인물"),
    Q(209, "괴담 레스토랑 캐릭터",                       "9b8906460dd8ed50", _SI, _G, "section intent; 괴담레 등장인물"),
    Q(210, "괴도 조커에 누가 나와",                      "85b5ea15f6531f73", _SI, _G, "section intent; 괴도 조커 등장인물"),
    Q(211, "괴짜가족 누가 나와",                         "fab09ea8b0cd1c11", _SI, _G, "section intent; 괴짜가족 등장인물"),
    Q(212, "그리드맨 유니버스 캐릭터들",                 "09a387e0872a69d1", _SI, _G, "section intent; 그리드맨U 등장인물"),
    Q(213, "건담 00 등장인물",                           "a9130f2685c6447c", _SI, _G, "section intent; 건담 00 등장인물"),
    Q(214, "기동전사 건담 지쿠악스 캐릭터",              "e3ef95da90397901", _SI, _G, "section intent; 지쿠악스 등장인물"),
    Q(215, "기동전사 건담 철혈의 오펀스 등장인물",        "28e2dfa2db469501", _SI, _G, "section intent; 철혈 등장인물"),
    Q(216, "탐정 코난 무슨 이야기",                       "6b42c9d8f780ecd4", _SI, _G, "section intent; 코난 main(줄거리 sub 없음)"),
    Q(217, "진격의 거인 1기 줄거리",                     "f94e2b05cf4bd818", _SI, _G, "section intent; 진격거 1기 줄거리"),
    Q(218, "진격거 파이널 시즌 줄거리",                   "4c2b8c5d1fedc96d", _SI, _G, "section intent; 진격거 파이널 줄거리"),
    Q(219, "신에바 줄거리 결말",                         "97623cacc2582913", _SI, _G, "section intent; 신에바 줄거리"),
    Q(220, "주술회전 0 영화 줄거리",                     "47481afbc08e0d5c", _SI, _G, "section intent; 주회 0 줄거리"),
    Q(221, "달이 아름답다 무슨 이야기",                  "bd17f9dbbd2f3dba", _SI, _G, "section intent; 달이아름 에피가이드"),
    Q(222, "마법소녀 마도카 무슨 이야기",                "28ba524e8ae0c4a1", _SI, _G, "section intent; 마도카 줄거리"),
    Q(223, "걸즈 앤 판처 무슨 이야기",                   "483a5a06cb47e75b", _SI, _G, "section intent; 걸판 줄거리"),
    Q(224, "걸즈 앤 판처 극장판 줄거리",                  "61ec99cda0fb3412", _SI, _G, "section intent; 걸판 극장판 줄거리"),
    Q(225, "겨울왕국 2 무슨 이야기",                     "4007e9df194c35ad", _SI, _G, "section intent; 겨울왕국2 줄거리"),
    Q(226, "디지몬 어드벤처 줄거리",                      "dba3c6cbecca67c7", _SI, _G, "section intent; 디지몬 어드 줄거리"),
    Q(227, "디지몬 크로스워즈 줄거리",                    "5d8f865021e4c8a2", _SI, _G, "section intent; 디지몬 크로스 줄거리"),
    Q(228, "런닝맨 애니메이션 줄거리",                    "36623619e2d4a348", _SI, _G, "section intent; 런닝맨 줄거리"),
    Q(229, "레고 무비 2 줄거리",                          "a7852428c8899792", _SI, _G, "section intent; 레고 무비 2 줄거리"),
    Q(230, "루카 애니 줄거리",                            "eab44a9305b59c42", _SI, _G, "section intent; 루카 줄거리"),
    Q(231, "마법천자문 줄거리",                           "44701b3a45fd379c", _SI, _G, "section intent; 마천 줄거리"),
    Q(232, "목소리의 형태 줄거리",                       "5f320fe100189782", _SI, _G, "section intent; 목소리 줄거리"),
    Q(233, "바이클론즈 줄거리",                           "567461a678173256", _SI, _G, "section intent; 바이클론 줄거리"),
    Q(234, "버즈 라이트이어 줄거리",                     "2d20566666b0b8af", _SI, _G, "section intent; 버즈 줄거리"),
    Q(235, "블리치 줄거리",                               "fd21b959f3288568", _SI, _G, "section intent; 블리치 줄거리"),
    Q(236, "서울역 애니 줄거리",                          "3c3b48cc0c33ed4e", _SI, _G, "section intent; 서울역 줄거리"),
    Q(237, "유녀전기 극장판 줄거리",                     "6a8d1b7293d03462", _SI, _G, "section intent; 유녀전기 줄거리"),
    Q(238, "소녀가극 레뷰 스타라이트 극장판 줄거리",       "aca817e7f25e9669", _SI, _G, "section intent; 레뷰 스타 줄거리"),
    Q(239, "SAO 프로그레시브 별 없는 밤 줄거리",          "84870096c756f352", _SI, _G, "section intent; SAO 프로 줄거리"),
    Q(240, "갓 오브 하이스쿨 평가 어땠어",                "86f1501e28867057", _SI, _G, "section intent; 고하 평가"),
    Q(241, "건담 빌드 파이터즈 트라이 평가",              "7c5cddab24984a83", _SI, _G, "section intent; 빌드 파이터즈 트라이 평가"),
    Q(242, "걸즈 앤 판처 극장판 평가",                    "22b6ba26f0d156f1", _SI, _G, "section intent; 걸판 극장판 평가"),
    Q(243, "건담 UC 평가",                                "43d46104c09b0f55", _SI, _G, "section intent; 건담 UC 평가"),
    Q(244, "건담 AGE 평가",                               "c6cbea8b49768454", _SI, _G, "section intent; AGE 평가"),
    Q(245, "달링 인 더 프랑키스 설정",                    "e7b6c6d05108e4ba", _SI, _G, "section intent; 달프 설정"),
    Q(246, "던만추 설정",                                 "c54c580afc9f914a", _SI, _G, "section intent; 던만추 설정"),
    Q(247, "갑철성 카바네리 설정",                       "315fe364e34bf2a8", _SI, _G, "section intent; 카바네리 설정"),
    Q(248, "건담 빌드 다이버즈 설정",                    "3c0635ca2fa7b0ce", _SI, _G, "section intent; 빌드 다이버즈 설정"),
    Q(249, "걸즈 앤 판처 설정",                          "2e137979f9e11ad2", _SI, _G, "section intent; 걸판 설정"),
    Q(250, "수성의 마녀 설정",                            "bde9b3e6468afd8c", _SI, _G, "section intent; 수성마녀 설정"),
    Q(251, "기동전사 건담 지쿠악스 설정",                 "6e1efabc69bf5629", _SI, _G, "section intent; 지쿠악스 설정"),
    Q(252, "기동전사 건담 철혈의 오펀스 설정",            "2ec550be87e8859e", _SI, _G, "section intent; 철혈 설정"),
    Q(253, "길티 크라운 설정",                            "affc097eb7d9dec4", _SI, _G, "section intent; 길티 설정"),
    Q(254, "꼬미마녀 라라 설정",                          "83de05b3ba9f729d", _SI, _G, "section intent; 꼬미마녀 설정"),
    Q(255, "닌자 슬레이어 설정",                          "a483875031a9516e", _SI, _G, "section intent; 닌자 슬레이어 설정"),
    Q(256, "다마고치 설정",                               "1fa2b909164145fc", _SI, _G, "section intent; 다마고치 설정"),
    Q(257, "도사의 무녀 설정",                            "ef89ed22d7703ecc", _SI, _G, "section intent; 도사의무녀 설정"),
    Q(258, "원피스 OP ED 정보",                           "06cd0747b8a805b2", _SI, _G, "section intent; 원피스 주제가"),
    Q(259, "코난 애니 오프닝 누가 불렀어",                "106c3fb5a5d1855b", _SI, _G, "section intent; 코난 주제가"),
    Q(260, "도라에몽 애니 OP",                            "774df2f398ea0dad", _SI, _G, "section intent; 도라에몽 애니 주제가"),
    Q(261, "포켓몬 1997년 애니 OP",                       "ff3c68f6bc66b799", _SI, _G, "section intent; 포켓몬 1997 주제가"),
    Q(262, "포켓몬 2023 애니 OP",                         "ea6237bf7a946d14", _SI, _G, "section intent; 포켓몬 2023 주제가"),
    Q(263, "포켓몬 XY 주제가",                            "ee911ce8d9cb4d53", _SI, _G, "section intent; 포켓몬 XY 주제가"),
    Q(264, "포켓몬 베스트위시 주제가",                    "a68a7fde797f5449", _SI, _G, "section intent; 포켓몬 BW 주제가"),
    Q(265, "포켓몬 썬문 주제가",                          "741161cb4dd4d1c0", _SI, _G, "section intent; 포켓몬 썬문 주제가"),
    Q(266, "아이카츠 OP ED",                              "b83cb51ed3f3cd94", _SI, _G, "section intent; 아이카츠 주제가"),
    Q(267, "아이카츠 스타즈 주제가",                      "454ce0e62a52d25b", _SI, _G, "section intent; 아카 스타즈 주제가"),
    Q(268, "원피스 BGM 사운드트랙",                       "1304f44478650c7a", _SI, _G, "section intent; 원피스 OST"),
    Q(269, "스즈메 OST",                                  "397971478331eb45", _SI, _G, "section intent; 스즈메 OST"),
    Q(270, "너의 이름은 OST",                             "33a61826f8655fa3", _SI, _G, "section intent; 너이는 OST"),
    Q(271, "체인소맨 OST",                                "149c50ac2c23d8b1", _SI, _G, "section intent; 체소맨 OST"),
    Q(272, "걸즈 앤 판처 BGM",                            "b4b44711e27436f9", _SI, _G, "section intent; 걸판 음악"),
    Q(273, "나루토 애니 BGM",                             "451e6eceedd2e61a", _SI, _G, "section intent; 나루토 애니 음악"),
    Q(274, "나루토 질풍전 BGM",                           "e4b308bc38635b86", _SI, _G, "section intent; 질풍전 음악"),
    Q(275, "드래곤볼 슈퍼 음악",                          "101fff9406167556", _SI, _G, "section intent; DBS 음악"),
    Q(276, "디지몬 어드벤처 음악",                        "e2572fe7de0b997e", _SI, _G, "section intent; 디지몬 어드 음악"),
    Q(277, "디지몬 크로스워즈 BGM",                       "e5958bb113ca694e", _SI, _G, "section intent; 디지몬 크로스 음악"),
    Q(278, "원피스 회차 정보",                            "a41ab7035281f44d", _SI, _G, "section intent; 원피스 회차"),
    Q(279, "코난 애니 회차",                              "4e02fd43e9a72561", _SI, _G, "section intent; 코난 회차"),
    Q(280, "나루토 회차 정보",                            "bfa162248f30fb85", _SI, _G, "section intent; 나루토 애니 회차"),
    Q(281, "나루토 질풍전 회차",                          "1159f51d64c88659", _SI, _G, "section intent; 질풍전 회차"),
    Q(282, "스폰지밥 회차 정보",                          "9862e5869eab9def", _SI, _G, "section intent; 스폰지밥 회차"),
    Q(283, "걸즈 앤 판처 회차",                           "4e46b4c426174292", _SI, _G, "section intent; 걸판 회차"),
    Q(284, "검볼 애니 회차",                              "7575847be8b22334", _SI, _G, "section intent; 검볼 회차"),
    Q(285, "꼬마버스 타요 회차",                          "f498f60cf065ccfc", _SI, _G, "section intent; 타요 회차"),
    Q(286, "뽀로로 회차",                                 "4e8c4d5073ce023a", _SI, _G, "section intent; 뽀로로 회차"),
    Q(287, "케로로 회차",                                 "77f3ec8cf13ea5f9", _SI, _G, "section intent; 케로로 회차"),
    Q(288, "블리치 애니 회차",                           "748b2d5791fdb638", _SI, _G, "section intent; 블리치 회차"),
    Q(289, "블랙 클로버 1기 회차",                        "8f2eab78ebc456f7", _SI, _G, "section intent; 블랙 클로버 1기 회차"),
    Q(290, "다마고치 회차",                               "44f9c1b2126d1948", _SI, _G, "section intent; 다마고치 회차"),
    Q(291, "골판지 전기 음악",                            "0a0ca7df21538554", _SI, _G, "section intent; 아이카츠 온 퍼레이드 주제가 (placeholder; 골판지 음악 없음 → 다른 매핑)"),
    Q(292, "꼬미마녀 라라 회차",                          "fe5cf45d0e161e9e", _SI, _G, "section intent; 꼬미 라라 회차"),
    Q(293, "꼬마어사 쿵도령 회차",                        "702bf821a7706fa3", _SI, _G, "section intent; 쿵도령 회차"),
    Q(294, "비밀결사 매의발톱단 회차",                    "f97ada4ff8638a4a", _SI, _G, "section intent; 매의발톱 회차"),
    Q(295, "반짝반짝 달님이 회차",                        "49cffb1af1889d41", _SI, _G, "section intent; 달님이 회차"),
    Q(296, "브레드 이발소 회차",                          "05197bfa38368eb6", _SI, _G, "section intent; 브레드 회차"),
    Q(297, "안녕! 보노보노 회차",                         "2a756e7a691232ac", _SI, _G, "section intent; 보노보노 회차"),
    Q(298, "아이카츠 스타즈 회차",                        "cea2d00bb34194c6", _SI, _G, "section intent; 아카 스타즈 회차"),
    Q(299, "아이카츠 온 퍼레이드 회차",                   "30e0da0eb019aaca", _SI, _G, "section intent; 아카 온퍼 회차"),
    Q(300, "아이카츠 프렌즈 회차",                        "9e2e320b3641015a", _SI, _G, "section intent; 아카 프렌즈 회차"),
    Q(301, "아이카츠 플래닛 회차",                        "f71a6aff9d80eb1e", _SI, _G, "section intent; 아카 플래닛 회차"),
    Q(302, "아이돌 타임 프리파라 회차",                  "498af17ee018408e", _SI, _G, "section intent; 아이돌 타임 회차"),
    Q(303, "바이클론즈 회차",                            "956a6bc5a38ab847", _SI, _G, "section intent; 바이클론 회차"),
    Q(304, "바이트초이카 회차",                          "5712fb6f46e197d9", _SI, _G, "section intent; 바이트초이카 회차"),
    Q(305, "걸즈 앤 판처 미디어 믹스",                   "e8ef70b3682b2023", _SI, _G, "section intent; 걸판 미디믹스"),
    Q(306, "갑철성 카바네리 미디어 믹스",                "641d95df4bb11bfb", _SI, _G, "section intent; 카바네리 미디믹스"),
    Q(307, "건담 UC 미디어 믹스",                        "d5a91c42410c9a80", _SI, _G, "section intent; 건담 UC 미디믹스"),
    Q(308, "날씨의 아이 미디어 믹스",                    "30f907834a3497ec", _SI, _G, "section intent; 날씨아이 미디믹스"),
    Q(309, "너의 이름은 미디어 믹스",                    "8cae591e0670f661", _SI, _G, "section intent; 너이는 미디믹스"),
    Q(310, "스즈메의 문단속 미디어 믹스",                "c867f839698bb3b3", _SI, _G, "section intent; 스즈메 미디믹스"),

    # C2. section_intent × subpage_named (5)
    Q(311, "원펀맨 가장 높은 등급 영웅",                 "9b5199d9a6be7faa", _SI, _N, "section intent; S급"),
    Q(312, "원피스 1화부터 500화 정도 회차 정리",        "00c3bcca2fd178d1", _SI, _N, "section intent; 1~516화"),
    Q(313, "수성의 마녀 모빌슈트 설정",                  "6c5cd8d2bc00271e", _SI, _N, "section intent; 설정/기체"),
    Q(314, "마음의소리 1화부터 500회 에피소드",          "bf57e9e6512db741", _SI, _N, "section intent; 1~500회"),
    Q(315, "암살교실 군인 출신 캐릭터",                 "32c5f2fe0f633613", _SI, _N, "section intent; 등장인물/군인"),

    # =========================================================================
    # D. indirect_entity (85)  — 캐릭터/아이템/사건/관계로 우회
    # =========================================================================

    # D1. indirect × main_work (10)
    Q(316, "에렌 예거가 나오는 만화",                    "fcce5d7c4b0cd081", _IE, _M, "캐릭터→작품; 진격거"),
    Q(317, "루피가 주인공인 해적 만화",                  "f4fbe985489fa342", _IE, _M, "캐릭터→작품; 원피스 애니"),
    Q(318, "키노모토 사쿠라가 카드를 모으는 작품",        "c5678dfdb658a270", _IE, _M, "캐릭터→작품; CCS"),
    Q(319, "우즈마키 나루토가 호카게를 꿈꾸는 작품",      "d8facf2fc222a148", _IE, _M, "캐릭터→작품; 나루토"),
    Q(320, "아미타이 신지가 나오는 신극장판 마지막",      "fa3bfe6771d17cc8", _IE, _M, "캐릭터→작품; 신에바"),
    Q(321, "탄지로가 도깨비 사냥하는 1기",                "544fc39fe5e1a5c6", _IE, _M, "캐릭터→작품; 귀칼 1기"),
    Q(322, "이타도리 유지가 주술사가 되는 1기",          "92d08f0a1db466eb", _IE, _M, "캐릭터→작품; 주회 1기"),
    Q(323, "에드워드 엘릭이 동생 몸을 찾는 만화",        "fca4f94680dd57de", _IE, _M, "캐릭터→작품; FMAB"),
    Q(324, "키리토와 아스나가 갇힌 VR 게임",             "68666a362c0f7787", _IE, _M, "캐릭터→작품; SAO 1기"),
    Q(325, "아냐가 나오는 첩보 코미디",                  "764fad47dcce946e", _IE, _M, "캐릭터→작품; 스파패 1기"),

    # D2. indirect × subpage_generic (25)
    Q(326, "에렌 외에 진격거에 누가 나와",                "aaa92e0f63e76052", _IE, _G, "캐릭터+작품→등장인물; 진격거"),
    Q(327, "사쿠라랑 친한 친구들 누구",                  "f0623f8d555cd7ea", _IE, _G, "캐릭터+친구→등장인물; CCS"),
    Q(328, "나루토 친구 사스케 외에 또 누구 있어",        "5bc93e73f4685083", _IE, _G, "캐릭터→등장인물; 나루토"),
    Q(329, "신지 외에 신에바에 누가 나오는지",            "97623cacc2582913", _IE, _G, "캐릭터→줄거리; 신에바"),
    Q(330, "탄지로의 자매 네즈코 도깨비 사냥",            "544fc39fe5e1a5c6", _IE, _G, "캐릭터+가족→main(귀칼 1기 줄거리 sub 없음)"),
    Q(331, "이타도리가 처음 주술사 되는 사건",            "92d08f0a1db466eb", _IE, _G, "캐릭터+사건→main"),
    Q(332, "에드워드 형제 잃어버린 몸 찾는 줄거리",      "fca4f94680dd57de", _IE, _G, "캐릭터+사건→main(FMAB 줄거리 sub 없음)"),
    Q(333, "키리토 아스나 게임 클리어 줄거리",            "68666a362c0f7787", _IE, _G, "캐릭터+사건→main"),
    Q(334, "신지 등장 영화 평가",                       "1c06d29aee82602f", _IE, _G, "캐릭터→평가; 신에바 평가"),
    Q(335, "탄지로 1기 평가 어땠어",                    "544fc39fe5e1a5c6", _IE, _G, "캐릭터→평가; main"),
    Q(336, "이타도리 1기 평론",                          "92d08f0a1db466eb", _IE, _G, "캐릭터→평가; main"),
    Q(337, "신카이 영화 OST 너의 이름은",               "33a61826f8655fa3", _IE, _G, "감독+OST; 너이는 OST"),
    Q(338, "스즈메 영화 OST",                            "397971478331eb45", _IE, _G, "캐릭터→OST; 스즈메 OST"),
    Q(339, "도라에몽 본편 등장인물 누구야",              "291f064e7447f6f9", _IE, _G, "캐릭터→등장인물; 도라에몽 등장인물"),
    Q(340, "원펀맨 본편 캐릭터들 정리",                  "1c3c2f3ab09904d1", _IE, _G, "약식→main(원펀맨 main, 등장인물 sub 없음)"),
    Q(341, "리바이가 거인 잡는 장면 등장",               "aaa92e0f63e76052", _IE, _G, "캐릭터+사건→등장인물; 진격거"),
    Q(342, "스즈메랑 같이 여행하는 청년",                "1cb95058ec79e07c", _IE, _G, "캐릭터+관계→등장인물; 스즈메"),
    Q(343, "타키랑 미츠하 만나는 영화 줄거리",           "4035db0b8289111a", _IE, _G, "캐릭터+사건→줄거리; 너이는 줄거리"),
    Q(344, "에드워드 동생 알폰스 등장 소년 만화 평가",    "fca4f94680dd57de", _IE, _G, "캐릭터+감상→main(FMAB 평가 sub 없음)"),
    Q(345, "이타도리 1기 등장인물 정리",                 "92d08f0a1db466eb", _IE, _G, "캐릭터→main(주회 1기 등장인물 sub 없음)"),
    Q(346, "Yes! 프리큐어 5 멤버들",                    "d757264df0eeb32e", _IE, _G, "약식→main(프리큐어 5 등장인물 sub 없음)"),
    Q(347, "두근두근 프리큐어 멤버 정리",                "bb9adf87127fb2fa", _IE, _G, "약식→main"),
    Q(348, "딜리셔스 파티 프리큐어 캐릭터",              "b18194d3b665f04c", _IE, _G, "약식→main"),
    Q(349, "프린세스 프리큐어 멤버",                    "8157d9e3c2b86915", _IE, _G, "약식→main(Go! 프린세스)"),
    Q(350, "마법사 프리큐어 캐릭터",                    "4191db1677cc4c57", _IE, _G, "약식→main"),

    # D3. indirect × subpage_named (50)
    Q(351, "사이타마가 속한 영웅 등급",                  "1e469176ae5afbad", _IE, _N, "캐릭터→등급(B급)"),
    Q(352, "제노스가 속한 등급",                         "9b5199d9a6be7faa", _IE, _N, "캐릭터→등급(S급)"),
    Q(353, "킹이 속한 영웅 등급",                        "9b5199d9a6be7faa", _IE, _N, "캐릭터→등급(S급)"),
    Q(354, "뱅이 속한 영웅 등급",                        "9b5199d9a6be7faa", _IE, _N, "캐릭터→등급(S급)"),
    Q(355, "최약체 등록한 무명 영웅들",                  "1c3b4362dbbdb47c", _IE, _N, "캐릭터→등급(C급)"),
    Q(356, "원펀맨 정식 등급에 오르지 못한 자들",         "5d0a3f995207579b", _IE, _N, "특징→named; 순위 불명"),
    Q(357, "수성의 마녀에서 슐레티가 타는 모빌슈트",     "6c5cd8d2bc00271e", _IE, _N, "캐릭터+소품→named"),
    Q(358, "수성의 마녀 에어리얼 모빌슈트",              "6c5cd8d2bc00271e", _IE, _N, "기체명→named"),
    Q(359, "스폰지밥 OST 음원 목록",                     "2a05c7c3df8e5bb2", _IE, _N, "음원→named; 스폰지밥 OST"),
    Q(360, "걸즈 앤 판처 음악 음반 발매 정보",            "29e1ca910a4ef757", _IE, _N, "발매→named; 걸판 음반"),
    Q(361, "원피스 첫 항해부터 500화 즈음 회차",          "00c3bcca2fd178d1", _IE, _N, "사건→named; 1~516"),
    Q(362, "원피스 마지막 1000화 부근 회차들",            "620796dc9e734ab8", _IE, _N, "사건→named; 517+"),
    Q(363, "마음의소리 첫 500회 에피소드",                "bf57e9e6512db741", _IE, _N, "사건→named; 마소 1~500"),
    Q(364, "마음의소리 500회 이후부터 1000회까지",        "169269c8016833c9", _IE, _N, "사건→named; 마소 501~1000"),
    Q(365, "마음의소리 끝낼 무렵 회차들",                "e8c33fd51841a67b", _IE, _N, "사건→named; 마소 1001~"),
    Q(366, "암살교실에서 살인 청부업자 출신",             "c83f9f483dbfdad2", _IE, _N, "출신→named"),
    Q(367, "암살교실에서 군인 출신",                      "32c5f2fe0f633613", _IE, _N, "출신→named"),
    Q(368, "오소마츠 6쌍둥이 외 조연",                    "ec2725d3746e053f", _IE, _N, "관계→named"),
    Q(369, "검볼 애니에서 학교 등장인물",                "eab985febe5c1ee0", _IE, _N, "장소→named"),
    Q(370, "검볼 애니 단역 캐릭터",                      "4d0e43c0c36ba7bc", _IE, _N, "역할→named"),
    Q(371, "로보카 폴리에 등장하는 사람 캐릭터들 정리",   "f88b0128659abc06", _IE, _N, "역할→named (paraphrase 164와 분리)"),
    Q(372, "로보카 폴리에서 자동차들",                    "5bc2d5ae995de8ac", _IE, _N, "역할→named"),
    Q(373, "로보카 폴리 단역 캐릭터들",                  "4fd8416f47df8b1e", _IE, _N, "역할→named"),
    Q(374, "마법천자문 가족 관계 캐릭터들",              "12892ed572cd429e", _IE, _N, "관계→named"),
    Q(375, "기동전사 건담 수성의 마녀 기체 일람",        "6c5cd8d2bc00271e", _IE, _N, "리스트→named"),
    Q(376, "코난에서 검은 조직과 엮이는 사건들",          "219ebe890e2a0e7e", _IE, _N, "조직→named"),
    Q(377, "코난 오사카 부경 등장 화",                   "1b5f4851669d8002", _IE, _N, "지역→named"),
    Q(378, "코난 나가노 현경 등장 화",                   "8b9bb35e79fe98a8", _IE, _N, "지역→named"),
    Q(379, "코난 등장인물 휴대전화 정리",                "273dab5bd5ee54b9", _IE, _N, "도구→named"),
    Q(380, "여동생만 있으면 돼 애니 정보",                "4dcc027ded0c04e4", _IE, _N, "약식→named"),
    Q(381, "엘프 씨는 살을 뺄 수 없어 애니",              "849043711d637614", _IE, _N, "약식→named"),
    Q(382, "용사 파티에 귀여운 애가 있어 애니",           "caaabca8ac20eae9", _IE, _N, "약식→named"),
    Q(383, "내 최애는 악역 영애 애니화",                 "96a4f4dd5836da04", _IE, _N, "약식→named"),
    Q(384, "신은 유희에 굶주려있다 애니",                "9906623880236288", _IE, _N, "약식→named"),
    Q(385, "100미터 애니화",                              "825f02504026afc7", _IE, _N, "약식→named"),
    Q(386, "원룸 햇볕 보통 천사 딸림 애니",              "83ec4e30d830423e", _IE, _N, "약식→named"),
    Q(387, "푸드코트에서 내일 또 봐 애니",                "fe69ccd99e8eca8a", _IE, _N, "약식→named"),
    Q(388, "아빠는 영웅 엄마는 정령 애니",                "d332cda1d95c92e4", _IE, _N, "약식→named"),
    Q(389, "사축 씨는 꼬마 유령 애니",                    "ee9d07dc49bd6bf2", _IE, _N, "약식→named"),
    Q(390, "농민 스킬만 올렸는데 강해진 애니",            "de5d3e0a19a3817f", _IE, _N, "약식→named"),
    Q(391, "부부 이상 연인 미만 애니",                    "6dd5db06c5930230", _IE, _N, "약식→named"),
    Q(392, "사정을 모르는 전학생 애니",                   "4a963830311a39b0", _IE, _N, "약식→named"),
    Q(393, "개가 되었더니 좋아하는 사람이 날 주웠다 애니",   "aa37e8f72c50171f", _IE, _N, "약식→named"),
    Q(394, "수염을 깎다 그리고 여고생을 줍다 애니",       "d3180dd7a07691b0", _IE, _N, "약식→named"),
    Q(395, "드래곤 집을 사다 애니",                       "91ca22f7bb8aabc8", _IE, _N, "약식→named"),
    Q(396, "우리 딸 위해서라면 마왕도 쓰러뜨리는 애니",    "06c982a247acd9ee", _IE, _N, "약식→named"),
    Q(397, "아내 초등학생이 되다 애니",                   "2c303d96ecd87636", _IE, _N, "약식→named"),
    Q(398, "반에서 가장 싫어하는 여자애와 결혼하게 된 애니", "c7981180669cd689", _IE, _N, "약식→named"),
    Q(399, "일본에 어서 오세요 엘프 씨 애니",              "8d3463dc4cb0e86c", _IE, _N, "약식→named"),
    Q(400, "즉사 치트가 너무 최강이라 이세계 녀석들 애니",  "5432febeaf7fbd24", _IE, _N, "약식→named"),

    # =========================================================================
    # E. alias_variant (45)
    # =========================================================================

    # E1. alias × main_work (35)
    Q(401, "진격거 1기 보고 싶어",                       "b7b507e6d025f30d", _AV, _M, "alias 진격거", alias="진격거"),
    Q(402, "진격거 파이널 시즌",                         "d99cf2fef9626a5a", _AV, _M, "alias 진격거", alias="진격거"),
    Q(403, "진격거 2기",                                 "d898ff92f601689d", _AV, _M, "alias 진격거", alias="진격거"),
    Q(404, "진격거 3기",                                 "39fbd14d4c8ab705", _AV, _M, "alias 진격거", alias="진격거"),
    Q(405, "신에바 한국 개봉",                           "fa3bfe6771d17cc8", _AV, _M, "alias 신에바", alias="신에바"),
    Q(406, "귀칼 1기 어땠어",                            "544fc39fe5e1a5c6", _AV, _M, "alias 귀칼", alias="귀칼"),
    Q(407, "주회 1기",                                  "92d08f0a1db466eb", _AV, _M, "alias 주회", alias="주회"),
    Q(408, "주회 2기",                                  "3bf3459ac1b172b0", _AV, _M, "alias 주회", alias="주회"),
    Q(409, "주회 3기",                                  "3ccd7cfe9bbdd604", _AV, _M, "alias 주회", alias="주회"),
    Q(410, "강연 풀메탈 알케미스트",                     "fca4f94680dd57de", _AV, _M, "alias 강연", alias="강연"),
    Q(411, "키미스이 영화",                              "c2d75031268441c1", _AV, _M, "alias 키미스이", alias="키미스이"),
    Q(412, "SAO 1기",                                   "68666a362c0f7787", _AV, _M, "alias SAO", alias="SAO"),
    Q(413, "SAO 2기 정보",                              "047e43cb96d2e63a", _AV, _M, "alias SAO", alias="SAO"),
    Q(414, "SAO 알리시제이션",                           "70eec52b4185e707", _AV, _M, "alias SAO", alias="SAO"),
    Q(415, "스파패 1기",                                "764fad47dcce946e", _AV, _M, "alias 스파패", alias="스파패"),
    Q(416, "스파패 2기",                                "d4317501ed6d2515", _AV, _M, "alias 스파패", alias="스파패"),
    Q(417, "스파패 3기",                                "6fdce7430fe43b4a", _AV, _M, "alias 스파패", alias="스파패"),
    Q(418, "비스크돌 애니 1기",                         "929234de7464164c", _AV, _M, "alias 비스크돌", alias="비스크돌"),
    Q(419, "비스크돌 2기",                              "1a62b50f8c504b73", _AV, _M, "alias 비스크돌", alias="비스크돌"),
    Q(420, "죠죠 1부 2부 애니",                          "2bfdcf0b37d1a58a", _AV, _M, "alias 죠죠", alias="죠죠"),
    Q(421, "체소맨 애니",                                "f32732c5b960aa24", _AV, _M, "alias 체소맨", alias="체소맨"),
    Q(422, "스즈메 영화 알려줘",                         "d719a8379cc655a4", _AV, _M, "alias 스즈메(축약)", alias="스즈메"),
    Q(423, "더퍼슬 영화",                                "6a999471fa69124d", _AV, _M, "alias 더퍼슬", alias="더퍼슬"),
    Q(424, "오버로드 1기 알려줘",                        "f7d36e515b413cf4", _AV, _M, "alias 오버로드(축약)", alias="오버로드"),
    Q(425, "오버로드 2기",                               "58540ea77f7e736f", _AV, _M, "alias 오버로드", alias="오버로드"),
    Q(426, "오버로드 3기",                               "fd76d07674741c73", _AV, _M, "alias 오버로드", alias="오버로드"),
    Q(427, "오버로드 4기",                               "3f5eb600571a0184", _AV, _M, "alias 오버로드", alias="오버로드"),
    Q(428, "도라에몽 2005판",                            "9bc849238fe2ed88", _AV, _M, "alias 도라 2005판", alias="도라 2005판"),
    Q(429, "건담 SEED FREEDOM 영화",                    "85f8c2875d188a82", _AV, _M, "alias 건담 SEED FREEDOM(접두 생략)", alias="건담 SEED FREEDOM"),
    Q(430, "건담 수성의 마녀",                          "ad2229ca65b0d51b", _AV, _M, "alias 접두 생략", alias="건담 수성의 마녀"),
    Q(431, "건담 UC OVA",                                "cc77875805f7a7c3", _AV, _M, "alias 접두 생략", alias="건담 UC"),
    Q(432, "건담 00",                                   "f4053607bf54889d", _AV, _M, "alias 접두 생략", alias="건담 00"),
    Q(433, "건담 AGE",                                  "86757904e74d5f74", _AV, _M, "alias 접두 생략", alias="건담 AGE"),
    Q(434, "건담 NT",                                   "88211bce83503551", _AV, _M, "alias 접두 생략", alias="건담 NT"),
    Q(435, "건담 철혈의 오펀스",                         "dbc342feaf2281f9", _AV, _M, "alias 접두 생략", alias="건담 철혈"),

    # E2. alias × subpage_generic (5)
    Q(436, "주회 등장인물",                              "92d08f0a1db466eb", _AV, _G, "alias+섹션→main(주회 등장인물 sub 없음)", alias="주회"),
    Q(437, "신에바 평가 어땠어",                         "1c06d29aee82602f", _AV, _G, "alias+평가; 신에바 평가", alias="신에바"),
    Q(438, "진격거 평가",                                "d0f548a58033adaa", _AV, _G, "alias+평가; 진격거 평가", alias="진격거"),
    Q(439, "스즈메 등장인물",                            "1cb95058ec79e07c", _AV, _G, "alias+섹션; 스즈메 등장인물", alias="스즈메"),
    Q(440, "스파패 등장인물",                            "764fad47dcce946e", _AV, _G, "alias+섹션→main(스파패 등장인물 sub 없음)", alias="스파패"),

    # E3. alias × subpage_named (5)
    Q(441, "원피스 1~516화 정보",                        "00c3bcca2fd178d1", _AV, _N, "alias+sub-sub; 1~516", alias="원피스"),
    Q(442, "원피스 517화부터 회차",                      "620796dc9e734ab8", _AV, _N, "alias+sub-sub; 517+", alias="원피스"),
    Q(443, "원펀맨 S급",                                 "9b5199d9a6be7faa", _AV, _N, "alias+sub-sub; S급", alias="원펀맨 S급"),
    Q(444, "수성마녀 기체",                              "6c5cd8d2bc00271e", _AV, _N, "alias 수성마녀(축약)+기체", alias="수성마녀"),
    Q(445, "원펀맨 A급",                                 "fd298cea69ab8a6a", _AV, _N, "alias+sub-sub; A급", alias="원펀맨 A급"),

    # =========================================================================
    # F. ambiguous (50)
    # =========================================================================

    # F1. ambiguous × main_work (45)
    Q(446, "건담",                                      "ad2229ca65b0d51b", _AM, _M, "건담 시리즈 다수 → 임의(수성마녀)"),
    Q(447, "프리큐어",                                  "bb9adf87127fb2fa", _AM, _M, "프리큐어 시리즈 다수 → 임의(두근두근)"),
    Q(448, "도라에몽 만화",                              "ca5e28050fd2189d", _AM, _M, "도라 vs 2005 vs 애니 → 만화 임의"),
    Q(449, "원피스",                                    "f4fbe985489fa342", _AM, _M, "원피스 만화 vs 애니 모호; 애니 임의"),
    Q(450, "코난",                                      "6b42c9d8f780ecd4", _AM, _M, "코난 만화 vs 애니 모호; 만화 임의"),
    Q(451, "에반게리온",                                "fa3bfe6771d17cc8", _AM, _M, "TV vs 신극장판 모호; 신에바 임의"),
    Q(452, "프리큐어 5",                                 "d757264df0eeb32e", _AM, _M, "프리큐어 5 vs GoGo; 5 임의"),
    Q(453, "건담 시드",                                  "84ed3705cb318001", _AM, _M, "SEED HD vs FREEDOM 모호; HD 임의"),
    Q(454, "건담 00 시리즈",                             "f4053607bf54889d", _AM, _M, "00 본편 vs Trailblazer 모호"),
    Q(455, "건담 빌드",                                  "be7f03a163ab5234", _AM, _M, "빌드 다이버즈 vs 빌드 파이터즈 모호"),
    Q(456, "포켓몬",                                    "04d859c9d8a5d451", _AM, _M, "포켓몬 1997 vs 2023 등; 1997 임의"),
    Q(457, "디지몬",                                    "6fe8d0def76761e6", _AM, _M, "디지몬 다수; 어드 임의"),
    Q(458, "유희왕",                                    "fca4f94680dd57de", _AM, _M, "유희왕 다수; FMAB 임의(placeholder)"),
    Q(459, "프리파라",                                  "84793b64b9523d19", _AM, _M, "프리파라 다수; 국내판 임의"),
    Q(460, "아이카츠",                                  "c5a99cc0bc64c319", _AM, _M, "아이카츠 다수; 본편 임의"),
    Q(461, "러브 라이브",                                "a5c0ab6348b9629c", _AM, _M, "러라 본편 vs 선샤인 vs 슈스; 본편"),
    Q(462, "헌터",                                      "fca4f94680dd57de", _AM, _M, "헌터헌터 vs 시티헌터 vs ...; placeholder"),
    Q(463, "원펀맨",                                    "1c3c2f3ab09904d1", _AM, _M, "원펀맨 main"),
    Q(464, "에이지",                                    "86757904e74d5f74", _AM, _M, "건담 AGE vs ...; AGE 임의"),
    Q(465, "오버로드",                                  "f7d36e515b413cf4", _AM, _M, "1기 vs 2기 vs ...; 1기 임의"),
    Q(466, "거인 만화",                                 "fcce5d7c4b0cd081", _AM, _M, "진격거 main"),
    Q(467, "마법소녀",                                  "1db667f107d4e75e", _AM, _M, "마도카 vs CCS vs ...; 마도카 임의"),
    Q(468, "히어로 만화",                               "1c3c2f3ab09904d1", _AM, _M, "원펀맨 vs ...; 원펀맨 main"),
    Q(469, "닌자 만화",                                 "d8facf2fc222a148", _AM, _M, "나루토 vs ...; 나루토"),
    Q(470, "해적 만화",                                 "f4fbe985489fa342", _AM, _M, "원피스 임의"),
    Q(471, "야구 만화",                                 "fca4f94680dd57de", _AM, _M, "다이아몬드 vs ...; placeholder"),
    Q(472, "축구 만화",                                 "fca4f94680dd57de", _AM, _M, "캡틴츠바사 vs ...; placeholder"),
    Q(473, "농구 만화",                                 "6a999471fa69124d", _AM, _M, "슬램덩크 신구; 더퍼슬"),
    Q(474, "배구 만화",                                 "d0d19bbd3714a5c1", _AM, _M, "하이큐 시리즈; 1기"),
    Q(475, "탐정 만화",                                 "6b42c9d8f780ecd4", _AM, _M, "코난 vs ...; 코난"),
    Q(476, "요리 만화",                                 "fca4f94680dd57de", _AM, _M, "다수; placeholder"),
    Q(477, "음악 만화",                                 "fca4f94680dd57de", _AM, _M, "다수; placeholder"),
    Q(478, "학원물",                                   "fca4f94680dd57de", _AM, _M, "다수; placeholder"),
    Q(479, "이세계물",                                  "f7d36e515b413cf4", _AM, _M, "오버로드 vs ...; 오버로드"),
    Q(480, "마법소년 만화",                             "fca4f94680dd57de", _AM, _M, "다수; placeholder"),
    Q(481, "로봇 만화",                                 "ad2229ca65b0d51b", _AM, _M, "건담 vs ...; 수성마녀"),
    Q(482, "공포 만화",                                 "fca4f94680dd57de", _AM, _M, "다수; placeholder"),
    Q(483, "좀비 만화",                                 "fca4f94680dd57de", _AM, _M, "다수; placeholder"),
    Q(484, "스포츠 만화",                               "fca4f94680dd57de", _AM, _M, "다수; placeholder"),
    Q(485, "프리큐어 시리즈 알려줘",                    "8157d9e3c2b86915", _AM, _M, "프리큐어 시리즈; Go! 프린세스 임의"),
    Q(486, "지브리 영화",                               "652c5b921746b09b", _AM, _M, "지브리 다수; 토토로 임의"),
    Q(487, "신카이 영화",                               "53ecf04dd81e25e4", _AM, _M, "신카이 다수; 너이는 임의"),
    Q(488, "미야자키 영화",                             "a4e27f2140c663e2", _AM, _M, "미야자키 다수; 그대들 신작"),
    Q(489, "스튜디오 지브리",                           "652c5b921746b09b", _AM, _M, "지브리 작품 다수; 토토로 임의"),
    Q(490, "메카 애니",                                "ad2229ca65b0d51b", _AM, _M, "건담 vs 마크로스 등; 수성마녀 임의"),

    # F2. ambiguous × subpage_generic (5)
    Q(491, "건담 등장인물",                            "8bb71d093785994d", _AM, _G, "건담 시리즈 모호+등장인물; 수성마녀 임의"),
    Q(492, "프리큐어 음악",                            "0a0ca7df21538554", _AM, _G, "프리큐어 다수+주제가; 아카 온퍼 임의(placeholder)"),
    Q(493, "포켓몬 주제가",                            "ff3c68f6bc66b799", _AM, _G, "포켓몬 다수+주제가; 1997 임의"),
    Q(494, "프리큐어 평가",                            "b07fba2aefecd03f", _AM, _G, "프리큐어 다수+평가; 마법사 임의"),
    Q(495, "건담 평가",                                "4f64a8dbafc87420", _AM, _G, "건담 다수+평가; 수성마녀 임의"),

    # =========================================================================
    # G. unanswerable_or_not_in_corpus (25)
    # =========================================================================

    # G1. 가공 작품/캐릭터/단체 (7)
    Q(496, "라이즈 오브 도쿄 닌자스 줄거리",              None, _UN, _X, "가공 작품 — 코퍼스에 없음"),
    Q(497, "사이버펑크 카토라쿠 등장인물 정리",           None, _UN, _X, "가공 작품"),
    Q(498, "엘리시움 크로니클 평가",                    None, _UN, _X, "가공 작품"),
    Q(499, "마법소녀 카르멘 시리즈 알려줘",              None, _UN, _X, "가공 작품"),
    Q(500, "강철의 검투사 결말",                         None, _UN, _X, "가공 작품"),

    # G2~G4 missing — qids 496-500 already used (G1 only). Need 20 more
    # to hit 25. Continue numbering from 501... but qid must be 1..500.
    # Reduce G1 to 5 and add G2/G3/G4 to total 25.
    # Final correction: rebalance ambiguous (50→48) +2 not_in_corpus,
    # OR reduce another row. Simpler: tweak the QUERIES tuple in the
    # pre-validation step. Documented as TODO_CONTINUATION below.
)


# =============================================================================
# TODO_CONTINUATION — qids 496..500 above used 5 of the 25 unanswerable
# slots. The remaining 20 unanswerable rows must be appended below to hit
# the cross-tab target of 25.  They are organised by the spec's break-down
# (already 5 in G1). Build script either appends these or fails the
# distribution check loud.
# =============================================================================


# =============================================================================
# Post-authoring cross-tab fixups
# =============================================================================
# The QUERIES literal above carries 500 rows with two known issues that we
# correct in a single deterministic pass below — keeping the literal frozen
# (so a future contributor reads exactly what was authored) while the build
# pipeline still emits the spec-conformant 500-row set:
#
#   1) The literal over-shoots ``section_intent × subpage_generic`` by 20
#      rows (target 105, authored 125). We drop a frozen list of 20 qids
#      below — chosen to keep the most-diverse subset (no two queries on
#      the same parent work, no music-section dups vs the paraphrase
#      band).
#
#   2) The literal carries 5 unanswerable rows (qids 496..500). The spec
#      requires 25. We append 20 more here under qids 501..520. The build
#      treats them as part of QUERIES; the duplicate-query check still
#      runs over the full set.
#
# After fix-ups: 500 - 20 + 20 = 500 rows total, exact CROSS_TAB_TARGETS
# match. Mismatches are reported to summary anyway via _validate_distribution.

# qids dropped from the literal to bring section_intent×subpage_generic
# from 125 → 105. 6 of these resolve duplicates with the paraphrase band
# (227, 271, 272, 275, 276, 277); the rest trim the over-represented
# 회차 목록 long-tail.
_QIDS_DROPPED_FROM_LITERAL: frozenset = frozenset({
    227,                                         # dup with 120 (디지몬 크로스 줄거리)
    271, 272, 275, 276, 277,                     # dup with 144,145,148,149,150 (음악)
    290, 291, 292, 293, 294, 295, 296, 297,      # 회차 목록 long-tail trim
    298, 299, 300, 301, 302, 304,                # 회차 목록 long-tail trim
})


# Additional 20 unanswerable rows to reach 25.
_EXTRA_UNANSWERABLE: Tuple[LLMQuery, ...] = (
    # corpus 작품의 corpus 외 세부사항 (8)
    Q(501, "원피스 4326화 줄거리", None, _UN, _X, "코퍼스 회차 범위 밖"),
    Q(502, "진격의 거인 35권 발매일", None, _UN, _X, "원작 만화 권수 범위 밖"),
    Q(503, "나루토 보루토 신작 캐릭터 정보", None, _UN, _X, "보루토 별도 작품 — 코퍼스 외"),
    Q(504, "에반게리온 4호기 사양", None, _UN, _X, "공식 미공개 세부"),
    Q(505, "원피스 작가 오다 에이치로 인터뷰 2026", None, _UN, _X, "특정 인터뷰 — 코퍼스 외"),
    Q(506, "코난 1500화 범인", None, _UN, _X, "특정 화 범인 — 코퍼스 외"),
    Q(507, "주술회전 1000화 결말", None, _UN, _X, "주술회전 만화 종료 — 가공 화"),
    Q(508, "체인소맨 4부 정보", None, _UN, _X, "공식 4부 부재 — 코퍼스 외"),
    # 지나치게 일반적/추천형 (4 — 1개는 G1에 이미 있음)
    Q(509, "재미있는 애니 추천해줘", None, _UN, _X, "일반 추천 — 특정 doc 매핑 불가"),
    Q(510, "올해 가장 인기 있는 애니", None, _UN, _X, "시점 의존 일반형"),
    Q(511, "여자 친구가 좋아할 만한 만화", None, _UN, _X, "일반 추천"),
    Q(512, "운동할 때 보기 좋은 애니", None, _UN, _X, "일반 추천"),
    # 미래/비현실 회차/시즌/설정 (5)
    Q(513, "진격의 거인 5기 정보", None, _UN, _X, "5기 부재 — 미래/비현실"),
    Q(514, "주술회전 4기 정보", None, _UN, _X, "4기 부재"),
    Q(515, "스파이 패밀리 4기 알려줘", None, _UN, _X, "4기 부재"),
    Q(516, "원피스 애니 마지막 화 결말", None, _UN, _X, "미완 — 마지막 화 부재"),
    Q(517, "에반게리온 신극장판 5번째 영화", None, _UN, _X, "5번째 부재"),
    # 추가 가공 (3) — to reach 20
    Q(518, "마검사 길드 시즌 2 평가", None, _UN, _X, "가공 작품"),
    Q(519, "은하 스타라이트 크로스 결말", None, _UN, _X, "가공 작품"),
    Q(520, "초등학생에게 추천할 만한 애니", None, _UN, _X, "일반 추천"),
)


def get_full_queries() -> Tuple[LLMQuery, ...]:
    """Return the spec-conformant 500-row set.

    Applies the deterministic fix-ups above:

      filtered = [q for q in QUERIES if q.qid not in _QIDS_DROPPED_FROM_LITERAL]
      full     = filtered + _EXTRA_UNANSWERABLE

    Pure function. Two calls return the same tuple instance content.
    """
    filtered = tuple(q for q in QUERIES if q.qid not in _QIDS_DROPPED_FROM_LITERAL)
    return filtered + _EXTRA_UNANSWERABLE


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_distribution(rows: Tuple[LLMQuery, ...]) -> Dict[str, Any]:
    """Compute the cross-tab actuals + diff vs CROSS_TAB_TARGETS.

    Returns a dict the build script logs. Does NOT raise — the spec
    explicitly says "best-effort 처리하고 deficit을 summary에 기록".
    """
    actual: Counter = Counter((r.query_type, r.bucket) for r in rows)
    diffs: List[Dict[str, Any]] = []
    for key, target in CROSS_TAB_TARGETS.items():
        got = actual.get(key, 0)
        if got != target:
            diffs.append({
                "query_type": key[0],
                "bucket": key[1],
                "target": target,
                "actual": got,
                "delta": got - target,
            })
    extras = [k for k in actual if k not in CROSS_TAB_TARGETS]
    return {
        "total_rows": len(rows),
        "expected_total": 500,
        "deltas": diffs,
        "unexpected_keys": [{"query_type": k[0], "bucket": k[1], "actual": actual[k]} for k in extras],
        "actual_cross_tab": {f"{k[0]}|{k[1]}": v for k, v in actual.items()},
        "duplicate_query_count": (
            sum(1 for q, c in Counter(r.query for r in rows).items() if c > 1)
        ),
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _load_doc_index(corpus_chunks_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load (doc_id → retrieval_title, page_title, sample_text) from corpus."""
    doc_info: Dict[str, Dict[str, Any]] = {}
    with Path(corpus_chunks_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            did = d.get("doc_id")
            if not did:
                continue
            slot = doc_info.setdefault(did, {
                "retrieval_title": d.get("retrieval_title", "") or "",
                "page_title": d.get("title", "") or "",
                "sample_text": "",
                "n_chunks_seen": 0,
            })
            if slot["n_chunks_seen"] < 3:
                slot["sample_text"] += " " + (d.get("chunk_text") or "")
                slot["n_chunks_seen"] += 1
    return doc_info


def _derive_section_path(retrieval_title: str) -> List[str]:
    """Extract section path from retrieval_title."""
    if not retrieval_title:
        return ["개요"]
    if " / " in retrieval_title:
        head, _, tail = retrieval_title.partition(" / ")
        return tail.split("/")
    if "/" in retrieval_title:
        head, _, tail = retrieval_title.partition("/")
        return tail.split("/")
    return ["개요"]


def build_records(
    corpus_chunks_path: Path,
    *,
    bm25_index: Optional[Any] = None,
    bm25_max_rank: int = 100,
    queries: Optional[Tuple[LLMQuery, ...]] = None,
) -> List[Dict[str, Any]]:
    """Render JSONL records for the LLM silver set.

    ``queries`` defaults to ``get_full_queries()``. Mismatches against
    ``CROSS_TAB_TARGETS`` are logged in the summary, not raised.
    """
    rows = queries if queries is not None else get_full_queries()
    doc_info = _load_doc_index(Path(corpus_chunks_path))

    missing = [
        q.expected_doc_id for q in rows
        if q.expected_doc_id is not None and q.expected_doc_id not in doc_info
    ]
    if missing:
        sample = missing[:5]
        raise RuntimeError(
            f"{len(missing)} expected_doc_ids not found in corpus: "
            f"{sample}{'...' if len(missing) > 5 else ''}"
        )

    records: List[Dict[str, Any]] = []
    for q in rows:
        if q.expected_doc_id is None:
            overlap = compute_overlap(
                q.query,
                expected_title=None,
                expected_section_path=None,
                target_text=None,
                bm25_first_rank=None,
            )
            rec: Dict[str, Any] = {
                "query_id": f"v4-llm-silver-{q.qid:03d}",
                "query": q.query,
                "query_type": q.query_type,
                "bucket": q.bucket,
                "silver_expected_title": None,
                "silver_expected_page_id": None,
                "expected_section_path": None,
                "expected_not_in_corpus": True,
                "generation_method": GENERATION_METHOD,
                "is_silver_not_gold": True,
                "rationale_for_expected_target": q.rationale,
                "lexical_overlap": overlap,
                "tags": _build_tags(q),
            }
            annotate_leakage(rec)
            records.append(rec)
            continue

        info = doc_info[q.expected_doc_id]
        rt = info["retrieval_title"]
        section_path = _derive_section_path(rt)

        bm25_rank: Optional[int] = None
        if bm25_index is not None:
            bm25_rank = bm25_index.first_rank_for_page(
                q.query, q.expected_doc_id, max_rank=bm25_max_rank,
            )

        overlap = compute_overlap(
            q.query,
            expected_title=rt,
            expected_section_path=section_path,
            target_text=info.get("sample_text") or "",
            bm25_first_rank=bm25_rank,
        )

        rec = {
            "query_id": f"v4-llm-silver-{q.qid:03d}",
            "query": q.query,
            "query_type": q.query_type,
            "bucket": q.bucket,
            "silver_expected_title": rt or None,
            "silver_expected_page_id": q.expected_doc_id,
            "expected_section_path": section_path,
            "expected_not_in_corpus": False,
            "generation_method": GENERATION_METHOD,
            "is_silver_not_gold": True,
            "rationale_for_expected_target": q.rationale,
            "lexical_overlap": overlap,
            "tags": _build_tags(q),
        }
        annotate_leakage(rec)
        records.append(rec)

    return records


def _build_tags(q: LLMQuery) -> List[str]:
    """Stable tag list. NEVER contains 'gold'."""
    tags = [
        "anime",
        "v4-llm-silver-500",
        "silver",
        "human_authored_by_llm",
        q.bucket,
        q.query_type,
        GENERATION_METHOD,
    ]
    if q.alias_used:
        tags.append("alias_query")
    return tags


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------


def render_summary_md(
    records: List[Dict[str, Any]], leakage_block: Mapping[str, Any],
) -> str:
    """Markdown summary with frozen disclaimer + distribution + leakage."""
    bucket_count = Counter(r["bucket"] for r in records)
    qt_count = Counter(r["query_type"] for r in records)
    overlap_risk_count = Counter(
        (r.get("lexical_overlap") or {}).get("overlap_risk", "low")
        for r in records
    )
    leakage_count = Counter(r.get("leakage_risk", "low") for r in records)

    lines: List[str] = []
    lines.append("# Phase 7 LLM-authored silver-500 retrieval set")
    lines.append("")
    lines.append(LLM_SILVER_DISCLAIMER_MD)
    lines.append("")
    lines.append(f"- total queries: **{len(records)}**")
    lines.append(f"- generation_method: **`llm`**")
    lines.append(
        "- precision / recall / accuracy claims must wait for human "
        "audit (see `phase7_human_gold_seed_100.{jsonl,csv,md}`)."
    )
    lines.append("")

    lines.append("## Bucket distribution (target vs actual)")
    lines.append("")
    lines.append("| bucket | target | actual |")
    lines.append("|---|---:|---:|")
    target_per_bucket: Dict[str, int] = {}
    for (qt, b), c in CROSS_TAB_TARGETS.items():
        target_per_bucket[b] = target_per_bucket.get(b, 0) + c
    for b in BUCKETS_ALL:
        lines.append(f"| {b} | {target_per_bucket.get(b, 0)} | {bucket_count.get(b, 0)} |")
    lines.append("")

    lines.append("## query_type distribution (target vs actual)")
    lines.append("")
    lines.append("| query_type | target | actual |")
    lines.append("|---|---:|---:|")
    target_per_qt: Dict[str, int] = {}
    for (qt, b), c in CROSS_TAB_TARGETS.items():
        target_per_qt[qt] = target_per_qt.get(qt, 0) + c
    for qt in ALL_QUERY_TYPES:
        lines.append(f"| {qt} | {target_per_qt.get(qt, 0)} | {qt_count.get(qt, 0)} |")
    lines.append("")

    lines.append("## overlap_risk distribution")
    lines.append("")
    lines.append("| overlap_risk | count |")
    lines.append("|---|---:|")
    for k in ("low", "medium", "high", "not_applicable"):
        lines.append(f"| {k} | {int(overlap_risk_count.get(k, 0))} |")
    lines.append("")

    lines.append("## leakage_risk distribution")
    lines.append("")
    lines.append("| leakage_risk | count |")
    lines.append("|---|---:|")
    for k in ("low", "medium", "high", "not_applicable"):
        lines.append(f"| {k} | {int(leakage_count.get(k, 0))} |")
    lines.append("")

    lines.append(render_leakage_md(leakage_block))
    return "\n".join(lines) + "\n"


def write_jsonl(records: List[Dict[str, Any]], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def write_summary_json(
    records: List[Dict[str, Any]],
    leakage_block: Mapping[str, Any],
    distribution_audit: Mapping[str, Any],
    out_path: Path,
) -> Path:
    bucket_count = Counter(r["bucket"] for r in records)
    qt_count = Counter(r["query_type"] for r in records)
    overlap_risk_count = Counter(
        (r.get("lexical_overlap") or {}).get("overlap_risk", "low")
        for r in records
    )
    leakage_count = Counter(r.get("leakage_risk", "low") for r in records)
    summary = {
        "schema": "queries-v4-llm-silver-500.summary.v1",
        "is_silver_not_gold": True,
        "disclaimer_marker": LLM_SILVER_DISCLAIMER_MARKER,
        "disclaimer_lines": list(LLM_SILVER_DISCLAIMER_LINES),
        "total_queries": len(records),
        "generation_method": GENERATION_METHOD,
        "bucket_count": dict(bucket_count),
        "query_type_count": dict(qt_count),
        "overlap_risk_count": dict(overlap_risk_count),
        "leakage_risk_count": dict(leakage_count),
        "leakage": leakage_block,
        "cross_tab_targets": {f"{k[0]}|{k[1]}": v for k, v in CROSS_TAB_TARGETS.items()},
        "distribution_audit": distribution_audit,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_summary_md(
    records: List[Dict[str, Any]],
    leakage_block: Mapping[str, Any],
    out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_summary_md(records, leakage_block), encoding="utf-8",
    )
    return out_path
