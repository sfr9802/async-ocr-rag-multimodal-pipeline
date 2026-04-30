"""Phase 7.2 — production embedding-text builder + ingest parity tests.

Three goals pinned by this suite:

  1. **Byte-perfect parity** between the eval-side Phase 7.0 export
     path and the production ingest path on the SAME canonical input.
     Phase 7.0 measured a +22pt hit@1 lift driven by the embedding
     text shape; if the production builder's output drifts from the
     eval export by even one byte the gain is invalidated.

  2. **No regression** on:
       - v3 prefix variants (``raw`` / ``title`` / ``section`` /
         ``title_section`` / ``keyword`` / ``all``) — eval-only,
         pinned because Phase 2 and earlier eval reports referenced
         the exact byte output.
       - Phase 7.0 v4 ``retrieval_title_section`` output — every
         already-built dense index whose
         ``embed_text_sha256`` was computed under the old eval-only
         module must remain reproducible.

  3. **Production-ingest contract**:
       - default variant is ``retrieval_title_section``
       - ``title_section`` rollback is one config flag
       - unknown variant fails loud
       - ``ingest_manifest.json`` is written with the variant + the
         builder version + the same sha256 the eval export computes
       - ``ChunkRow.text`` continues to carry the raw chunk text
         (reranker / generation contract unaffected)

The IngestService is exercised end-to-end with a HashingEmbedder + a
fake metadata store + a real FaissIndex on tmp_path. No live model,
no Postgres.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest

# Production canonical builder (Phase 7.2 owner).
from app.capabilities.rag.embedding_text_builder import (
    DEFAULT_PRODUCTION_VARIANT,
    EMBEDDING_TEXT_BUILDER_VERSION,
    PRODUCTION_VARIANTS,
    V4_BODY_LABEL,
    V4_SECTION_LABEL,
    V4_SECTION_PATH_JOINER,
    V4_SECTION_TYPE_LABEL,
    V4_TITLE_LABEL,
    V4EmbeddingTextInput as ProductionV4Input,
    VARIANT_RETRIEVAL_TITLE_SECTION,
    VARIANT_TITLE_SECTION,
    build_embedding_text_from_v3_chunk,
    build_v4_embedding_text as production_build_v4,
    is_known_production_variant,
)
from app.capabilities.rag.embeddings import HashingEmbedder
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.ingest import (
    IngestService,
    load_ingest_manifest,
)
from app.capabilities.rag.metadata_store import ChunkLookupResult, ChunkRow, DocumentRow

# Eval-side wrapper (Phase 7.0 byte producer that production must match).
from eval.harness.embedding_text_builder import (
    DEFAULT_KEYWORD_LIMIT,
    EMBEDDING_TEXT_VARIANTS,
    EmbeddingTextInput,
    V4EmbeddingTextInput as EvalV4Input,
    VARIANT_ALL,
    VARIANT_KEYWORD,
    VARIANT_RAW,
    VARIANT_SECTION,
    VARIANT_TITLE,
    build_embedding_text,
    build_v4_embedding_text as eval_build_v4,
    is_known_variant,
)
from eval.harness.v4_chunk_export import recompute_embedding_text


# ---------------------------------------------------------------------------
# Builder identity: production module owns the v4 byte format
# ---------------------------------------------------------------------------


def test_eval_v4_input_is_production_v4_input():
    """Eval re-export must return the same dataclass identity as production.

    Pinning this prevents a future split where eval and production
    define two different ``V4EmbeddingTextInput`` types and silently
    diverge their fields.
    """
    assert EvalV4Input is ProductionV4Input


def test_eval_build_v4_is_production_build_v4():
    assert eval_build_v4 is production_build_v4


def test_production_default_variant_is_retrieval_title_section():
    assert DEFAULT_PRODUCTION_VARIANT == VARIANT_RETRIEVAL_TITLE_SECTION


def test_production_variants_registry_pinned():
    assert set(PRODUCTION_VARIANTS) == {
        VARIANT_TITLE_SECTION, VARIANT_RETRIEVAL_TITLE_SECTION,
    }


def test_builder_version_is_phase7_2_canonical():
    """Bumping this is a cache-invalidating change. Pinning catches it."""
    assert EMBEDDING_TEXT_BUILDER_VERSION == "v4-1"


def test_v4_format_constants_are_phase7_0_canonical():
    """Pin the four labels + section joiner — every cached index whose
    embed_text_sha256 was computed under the previous values would
    silently desync if any of them changed.
    """
    assert V4_TITLE_LABEL == "제목"
    assert V4_SECTION_LABEL == "섹션"
    assert V4_SECTION_TYPE_LABEL == "섹션타입"
    assert V4_BODY_LABEL == "본문"
    assert V4_SECTION_PATH_JOINER == " > "


# ---------------------------------------------------------------------------
# Byte-perfect output: known Phase 7.0 cases
# ---------------------------------------------------------------------------


_EXPECTED_PHASE_7_0_OUTPUTS: List[Dict[str, str]] = [
    # 1. main work: retrieval_title == page_title
    dict(
        case="main_work",
        page_title="ARIA The ORIGINATION",
        retrieval_title="ARIA The ORIGINATION",
        section_path=("개요",),
        section_type="summary",
        chunk_text="일본의 만화 ARIA를 원작으로 하는 TV 애니메이션.",
        expected_title_section=(
            "제목: ARIA The ORIGINATION\n"
            "섹션: 개요\n"
            "섹션타입: summary\n"
            "\n"
            "본문:\n"
            "일본의 만화 ARIA를 원작으로 하는 TV 애니메이션."
        ),
        expected_retrieval_title_section=(
            "제목: ARIA The ORIGINATION\n"
            "섹션: 개요\n"
            "섹션타입: summary\n"
            "\n"
            "본문:\n"
            "일본의 만화 ARIA를 원작으로 하는 TV 애니메이션."
        ),
    ),
    # 2. generic subpage 등장인물 — Phase 6.3 audit's canonical example
    dict(
        case="subpage_generic_chr",
        page_title="등장인물",
        retrieval_title="가난뱅이 신이! / 등장인물",
        section_path=("개요",),
        section_type="summary",
        chunk_text="만화 가난뱅이 신이! 의 등장인물 일람.",
        expected_title_section=(
            "제목: 등장인물\n"
            "섹션: 개요\n"
            "섹션타입: summary\n"
            "\n"
            "본문:\n"
            "만화 가난뱅이 신이! 의 등장인물 일람."
        ),
        expected_retrieval_title_section=(
            "제목: 가난뱅이 신이! / 등장인물\n"
            "섹션: 개요\n"
            "섹션타입: summary\n"
            "\n"
            "본문:\n"
            "만화 가난뱅이 신이! 의 등장인물 일람."
        ),
    ),
    # 3. generic subpage 평가
    dict(
        case="subpage_generic_review",
        page_title="평가",
        retrieval_title="신비아파트 고스트볼 ZERO / 평가",
        section_path=("총평",),
        section_type="review",
        chunk_text="시리즈 누적 평점은 양호한 편.",
        expected_title_section=(
            "제목: 평가\n"
            "섹션: 총평\n"
            "섹션타입: review\n"
            "\n"
            "본문:\n"
            "시리즈 누적 평점은 양호한 편."
        ),
        expected_retrieval_title_section=(
            "제목: 신비아파트 고스트볼 ZERO / 평가\n"
            "섹션: 총평\n"
            "섹션타입: review\n"
            "\n"
            "본문:\n"
            "시리즈 누적 평점은 양호한 편."
        ),
    ),
    # 4. generic subpage 줄거리
    dict(
        case="subpage_generic_plot",
        page_title="줄거리",
        retrieval_title="걸즈 앤 판처 / 줄거리",
        section_path=("1화",),
        section_type="plot",
        chunk_text="주인공이 학교로 전학을 온다.",
        expected_title_section=(
            "제목: 줄거리\n"
            "섹션: 1화\n"
            "섹션타입: plot\n"
            "\n"
            "본문:\n"
            "주인공이 학교로 전학을 온다."
        ),
        expected_retrieval_title_section=(
            "제목: 걸즈 앤 판처 / 줄거리\n"
            "섹션: 1화\n"
            "섹션타입: plot\n"
            "\n"
            "본문:\n"
            "주인공이 학교로 전학을 온다."
        ),
    ),
    # 5. generic subpage 회차
    dict(
        case="subpage_generic_episode",
        page_title="회차",
        retrieval_title="괴담 레스토랑 / 회차",
        section_path=("3화",),
        section_type="episode",
        chunk_text="레스토랑에 새 손님이 찾아온다.",
        expected_title_section=(
            "제목: 회차\n"
            "섹션: 3화\n"
            "섹션타입: episode\n"
            "\n"
            "본문:\n"
            "레스토랑에 새 손님이 찾아온다."
        ),
        expected_retrieval_title_section=(
            "제목: 괴담 레스토랑 / 회차\n"
            "섹션: 3화\n"
            "섹션타입: episode\n"
            "\n"
            "본문:\n"
            "레스토랑에 새 손님이 찾아온다."
        ),
    ),
    # 6. generic subpage 음악 / 주제가
    dict(
        case="subpage_generic_music",
        page_title="음악",
        retrieval_title="ARIA / 음악",
        section_path=("주제가", "OP"),
        section_type="music",
        chunk_text="오프닝 주제가는 'Undine' 이다.",
        expected_title_section=(
            "제목: 음악\n"
            "섹션: 주제가 > OP\n"
            "섹션타입: music\n"
            "\n"
            "본문:\n"
            "오프닝 주제가는 'Undine' 이다."
        ),
        expected_retrieval_title_section=(
            "제목: ARIA / 음악\n"
            "섹션: 주제가 > OP\n"
            "섹션타입: music\n"
            "\n"
            "본문:\n"
            "오프닝 주제가는 'Undine' 이다."
        ),
    ),
    # 7. generic subpage 설정
    dict(
        case="subpage_generic_setting",
        page_title="설정",
        retrieval_title="마법과고교의 열등생 / 설정",
        section_path=("세계관",),
        section_type="setting",
        chunk_text="설정의 핵심은 마법공학 시스템이다.",
        expected_title_section=(
            "제목: 설정\n"
            "섹션: 세계관\n"
            "섹션타입: setting\n"
            "\n"
            "본문:\n"
            "설정의 핵심은 마법공학 시스템이다."
        ),
        expected_retrieval_title_section=(
            "제목: 마법과고교의 열등생 / 설정\n"
            "섹션: 세계관\n"
            "섹션타입: setting\n"
            "\n"
            "본문:\n"
            "설정의 핵심은 마법공학 시스템이다."
        ),
    ),
    # 8. named subpage (non-generic page_title)
    dict(
        case="subpage_named",
        page_title="목소리의 형태(애니메이션)",
        retrieval_title="목소리의 형태 / 목소리의 형태(애니메이션)",
        section_path=("원작과의 차이점",),
        section_type="comparison",
        chunk_text="원작에 비해 일부 장면이 추가되었다.",
        expected_title_section=(
            "제목: 목소리의 형태(애니메이션)\n"
            "섹션: 원작과의 차이점\n"
            "섹션타입: comparison\n"
            "\n"
            "본문:\n"
            "원작에 비해 일부 장면이 추가되었다."
        ),
        expected_retrieval_title_section=(
            "제목: 목소리의 형태 / 목소리의 형태(애니메이션)\n"
            "섹션: 원작과의 차이점\n"
            "섹션타입: comparison\n"
            "\n"
            "본문:\n"
            "원작에 비해 일부 장면이 추가되었다."
        ),
    ),
    # 9. retrieval_title empty → fallback to page_title
    dict(
        case="retrieval_title_empty_fallback",
        page_title="원더풀 프리큐어!",
        retrieval_title="",
        section_path=("개요",),
        section_type="summary",
        chunk_text="2024년에 방영된 프리큐어 시리즈.",
        expected_title_section=(
            "제목: 원더풀 프리큐어!\n"
            "섹션: 개요\n"
            "섹션타입: summary\n"
            "\n"
            "본문:\n"
            "2024년에 방영된 프리큐어 시리즈."
        ),
        # When retrieval_title is empty, retrieval_title_section falls
        # back to page_title; output is byte-equal to title_section.
        expected_retrieval_title_section=(
            "제목: 원더풀 프리큐어!\n"
            "섹션: 개요\n"
            "섹션타입: summary\n"
            "\n"
            "본문:\n"
            "2024년에 방영된 프리큐어 시리즈."
        ),
    ),
    # 10. section_path missing → section line dropped
    dict(
        case="section_path_missing",
        page_title="등장인물",
        retrieval_title="천관사복 / 등장인물",
        section_path=(),
        section_type="character",
        chunk_text="주연 캐릭터들의 관계도.",
        expected_title_section=(
            "제목: 등장인물\n"
            "섹션타입: character\n"
            "\n"
            "본문:\n"
            "주연 캐릭터들의 관계도."
        ),
        expected_retrieval_title_section=(
            "제목: 천관사복 / 등장인물\n"
            "섹션타입: character\n"
            "\n"
            "본문:\n"
            "주연 캐릭터들의 관계도."
        ),
    ),
    # 11. section_type missing → section_type line dropped
    dict(
        case="section_type_missing",
        page_title="등장인물",
        retrieval_title="쿠로무쿠로 / 등장인물",
        section_path=("주역",),
        section_type="",
        chunk_text="검은 갑옷의 검사가 등장한다.",
        expected_title_section=(
            "제목: 등장인물\n"
            "섹션: 주역\n"
            "\n"
            "본문:\n"
            "검은 갑옷의 검사가 등장한다."
        ),
        expected_retrieval_title_section=(
            "제목: 쿠로무쿠로 / 등장인물\n"
            "섹션: 주역\n"
            "\n"
            "본문:\n"
            "검은 갑옷의 검사가 등장한다."
        ),
    ),
]


@pytest.mark.parametrize("case", _EXPECTED_PHASE_7_0_OUTPUTS, ids=lambda c: c["case"])
def test_production_builder_byte_matches_phase7_0_expected_output(case):
    """Pin the byte output for both variants on every Phase 7.0 bucket."""
    chunk = ProductionV4Input(
        chunk_text=case["chunk_text"],
        page_title=case["page_title"],
        retrieval_title=case["retrieval_title"],
        section_path=tuple(case["section_path"]),
        section_type=case["section_type"],
    )
    out_ts = production_build_v4(chunk, variant=VARIANT_TITLE_SECTION)
    out_rts = production_build_v4(chunk, variant=VARIANT_RETRIEVAL_TITLE_SECTION)
    assert out_ts == case["expected_title_section"], case["case"]
    assert out_rts == case["expected_retrieval_title_section"], case["case"]


# ---------------------------------------------------------------------------
# Eval / production parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _EXPECTED_PHASE_7_0_OUTPUTS, ids=lambda c: c["case"])
def test_eval_v4_export_matches_production_byte_for_byte(case):
    """v4_chunk_export.recompute_embedding_text must agree with production."""
    record: Dict[str, Any] = {
        "chunk_text": case["chunk_text"],
        "title": case["page_title"],
        "retrieval_title": case["retrieval_title"],
        "section_path": list(case["section_path"]),
        "section_type": case["section_type"],
    }
    eval_ts = recompute_embedding_text(record, variant=VARIANT_TITLE_SECTION)
    eval_rts = recompute_embedding_text(
        record, variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )

    chunk = ProductionV4Input(
        chunk_text=case["chunk_text"],
        page_title=case["page_title"],
        retrieval_title=case["retrieval_title"],
        section_path=tuple(case["section_path"]),
        section_type=case["section_type"],
    )
    prod_ts = production_build_v4(chunk, variant=VARIANT_TITLE_SECTION)
    prod_rts = production_build_v4(chunk, variant=VARIANT_RETRIEVAL_TITLE_SECTION)

    assert eval_ts == prod_ts, case["case"]
    assert eval_rts == prod_rts, case["case"]


def test_v3_ingest_chunk_helper_matches_production_v4_input():
    """``build_embedding_text_from_v3_chunk`` is a thin wrapper — pin equality."""
    expected = production_build_v4(
        ProductionV4Input(
            chunk_text="본문 내용입니다.",
            page_title="등장인물",
            retrieval_title="가난뱅이 신이! / 등장인물",
            section_path=("개요",),
            section_type="summary",
        ),
        variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    actual = build_embedding_text_from_v3_chunk(
        chunk_text="본문 내용입니다.",
        title="등장인물",
        retrieval_title="가난뱅이 신이! / 등장인물",
        section_name="개요",
        section_type="summary",
        variant=VARIANT_RETRIEVAL_TITLE_SECTION,
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# v3 prefix variants — eval-only, regression pinned
# ---------------------------------------------------------------------------


def _v3_chunk(
    text: str = "본문",
    title: Optional[str] = "제목값",
    section: Optional[str] = "섹션값",
    keywords: Iterable[str] = (),
) -> EmbeddingTextInput:
    return EmbeddingTextInput(
        text=text, title=title, section=section,
        keywords=tuple(keywords),
    )


def test_v3_variant_raw_unchanged():
    out = build_embedding_text(_v3_chunk(text="hello"), variant=VARIANT_RAW)
    assert out == "hello"


def test_v3_variant_title_unchanged():
    out = build_embedding_text(
        _v3_chunk(text="hello", title="Foo", section=None),
        variant=VARIANT_TITLE,
    )
    assert out == "Foo\nhello"


def test_v3_variant_section_unchanged():
    out = build_embedding_text(
        _v3_chunk(text="hello", title=None, section="overview"),
        variant=VARIANT_SECTION,
    )
    assert out == "overview\nhello"


def test_v3_variant_title_section_unchanged():
    out = build_embedding_text(
        _v3_chunk(text="hello", title="Foo", section="overview"),
        variant=VARIANT_TITLE_SECTION,
    )
    assert out == "Foo\noverview\nhello"


def test_v3_variant_keyword_unchanged():
    out = build_embedding_text(
        _v3_chunk(text="hello", title=None, section=None,
                  keywords=("alpha", "beta", "alpha")),
        variant=VARIANT_KEYWORD,
    )
    # alpha dedup + space-joined keyword segment, then chunk text on next line.
    assert out == "alpha beta\nhello"


def test_v3_variant_all_unchanged():
    out = build_embedding_text(
        _v3_chunk(text="hello", title="Foo", section="overview",
                  keywords=("a", "b")),
        variant=VARIANT_ALL,
    )
    assert out == "Foo\noverview\na b\nhello"


def test_v3_builder_rejects_v4_variant():
    """Calling the v3 builder with the v4-only variant must error loudly.

    Silently treating retrieval_title_section as title_section would
    conflate variants in any sweep that mixes them.
    """
    with pytest.raises(ValueError):
        build_embedding_text(
            _v3_chunk(),
            variant=VARIANT_RETRIEVAL_TITLE_SECTION,
        )


def test_v3_builder_rejects_unknown_variant():
    with pytest.raises(ValueError):
        build_embedding_text(_v3_chunk(), variant="bogus")


def test_eval_variant_registry_includes_phase7_0_variant():
    assert VARIANT_RETRIEVAL_TITLE_SECTION in EMBEDDING_TEXT_VARIANTS


def test_eval_is_known_variant_helper():
    for v in EMBEDDING_TEXT_VARIANTS:
        assert is_known_variant(v)
    assert not is_known_variant("bogus")


# ---------------------------------------------------------------------------
# Production builder — error paths
# ---------------------------------------------------------------------------


def test_production_builder_rejects_unknown_variant():
    with pytest.raises(ValueError):
        production_build_v4(
            ProductionV4Input(chunk_text="x", page_title="y"),
            variant="bogus",
        )


def test_production_builder_rejects_v3_only_variant():
    """The v3 ``raw`` / ``title`` / etc. variants are not production knobs."""
    for v in (VARIANT_RAW, VARIANT_TITLE, VARIANT_SECTION,
              VARIANT_KEYWORD, VARIANT_ALL):
        with pytest.raises(ValueError):
            production_build_v4(
                ProductionV4Input(chunk_text="x", page_title="y"),
                variant=v,
            )


def test_production_is_known_production_variant():
    assert is_known_production_variant(VARIANT_TITLE_SECTION)
    assert is_known_production_variant(VARIANT_RETRIEVAL_TITLE_SECTION)
    assert not is_known_production_variant(VARIANT_RAW)
    assert not is_known_production_variant("bogus")


# ---------------------------------------------------------------------------
# Ingest end-to-end: variant selection + manifest
# ---------------------------------------------------------------------------


class _FakeIngestStore:
    """Captures whatever ``replace_all`` was handed.

    Lets the test inspect ``ChunkRow.text`` to confirm raw chunk text is
    still what the metadata layer sees (reranker / generation contract
    unchanged in Phase 7.2).
    """

    def __init__(self) -> None:
        self.documents: List[DocumentRow] = []
        self.chunks: List[ChunkRow] = []
        self.replace_all_called = False

    def replace_all(
        self, *, documents, chunks, index_version, embedding_model,
        embedding_dim, faiss_index_path, notes=None,
    ) -> None:
        self.documents = list(documents)
        self.chunks = list(chunks)
        self.replace_all_called = True


_INGEST_FIXTURE_ROWS: List[Dict[str, Any]] = [
    {
        "doc_id": "doc-aria",
        "title": "ARIA The ORIGINATION",
        # No ``retrieval_title`` (v3 corpus shape) — production builder
        # falls back to title under retrieval_title_section so the
        # output is the same under either variant for v3 fixtures.
        "sections": {
            "overview": {
                "chunks": [
                    "일본의 만화 ARIA를 원작으로 하는 TV 애니메이션.",
                ],
            },
            "characters": {
                "chunks": [
                    "주인공은 미즈나시 아카리.",
                ],
            },
        },
    },
    {
        "doc_id": "doc-poor-god",
        "title": "가난뱅이 신이!",
        # Future-proof: the row carries retrieval_title at the doc
        # level so the ingest path's v4-shape capture wires through.
        "retrieval_title": "가난뱅이 신이!",
        "sections": {
            "characters": {
                "chunks": [
                    "신과 인간이 동거하는 코미디.",
                ],
            },
        },
    },
]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _build_ingest(
    tmp_path: Path,
    *,
    variant: Optional[str] = None,
):
    embedder = HashingEmbedder(dim=64)
    index = FaissIndex(tmp_path / "idx")
    store = _FakeIngestStore()
    service = IngestService(
        embedder=embedder,
        metadata_store=store,
        index=index,
        embedding_text_variant=variant,
    )
    return service, store, embedder, index


def test_ingest_default_variant_is_retrieval_title_section(tmp_path: Path):
    service, _, _, _ = _build_ingest(tmp_path)
    assert service.embedding_text_variant == VARIANT_RETRIEVAL_TITLE_SECTION


def test_ingest_accepts_explicit_title_section_rollback(tmp_path: Path):
    service, _, _, _ = _build_ingest(tmp_path, variant=VARIANT_TITLE_SECTION)
    assert service.embedding_text_variant == VARIANT_TITLE_SECTION


def test_ingest_rejects_unknown_variant(tmp_path: Path):
    with pytest.raises(ValueError):
        _build_ingest(tmp_path, variant="bogus")


def test_ingest_rejects_v3_only_variant(tmp_path: Path):
    with pytest.raises(ValueError):
        _build_ingest(tmp_path, variant=VARIANT_RAW)


def test_ingest_writes_manifest_with_default_variant(tmp_path: Path):
    """Default ingest run lands a manifest with variant=retrieval_title_section."""
    jsonl = _write_jsonl(tmp_path / "in.jsonl", _INGEST_FIXTURE_ROWS)
    service, _, _, index = _build_ingest(tmp_path)
    result = service.ingest_jsonl(
        jsonl, source_label="phase7_2-fixture", index_version="test-v1",
    )

    # Manifest exists at the expected path
    manifest_path = tmp_path / "idx" / "ingest_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Schema fields the spec asks for
    for key in (
        "embedding_text_variant", "embedding_text_builder_version",
        "embedding_model", "max_seq_length", "chunk_count",
        "document_count", "dimension", "index_version", "corpus_path",
        "embed_text_sha256", "embed_text_samples",
    ):
        assert key in payload, f"manifest missing {key}"

    assert payload["embedding_text_variant"] == VARIANT_RETRIEVAL_TITLE_SECTION
    assert payload["embedding_text_builder_version"] == EMBEDDING_TEXT_BUILDER_VERSION
    assert payload["index_version"] == "test-v1"
    assert payload["chunk_count"] == 3  # 2 + 1 chunks across the fixture
    assert payload["document_count"] == 2
    assert isinstance(payload["embed_text_sha256"], str)
    assert len(payload["embed_text_sha256"]) == 64  # hex sha256 = 64 chars
    assert isinstance(payload["embed_text_samples"], list)
    assert payload["embed_text_samples"]

    # IngestResult also carries the manifest dataclass
    assert result.manifest is not None
    assert result.manifest.embedding_text_variant == VARIANT_RETRIEVAL_TITLE_SECTION


def test_ingest_writes_manifest_with_title_section_rollback(tmp_path: Path):
    jsonl = _write_jsonl(tmp_path / "in.jsonl", _INGEST_FIXTURE_ROWS)
    service, _, _, _ = _build_ingest(tmp_path, variant=VARIANT_TITLE_SECTION)
    service.ingest_jsonl(
        jsonl, source_label="phase7_2-rollback", index_version="rb-1",
    )
    payload = json.loads(
        (tmp_path / "idx" / "ingest_manifest.json").read_text(encoding="utf-8"),
    )
    assert payload["embedding_text_variant"] == VARIANT_TITLE_SECTION


def test_ingest_chunkrow_text_is_raw_not_embedding_text(tmp_path: Path):
    """Reranker / generation paths must continue to see raw chunk text.

    Phase 7.2 only changes the string handed to the embedder; whatever
    gets persisted under ``ChunkRow.text`` must still be the raw chunk
    body so the cross-encoder reranker scores the same passage strings
    it always has.
    """
    jsonl = _write_jsonl(tmp_path / "in.jsonl", _INGEST_FIXTURE_ROWS)
    service, store, _, _ = _build_ingest(tmp_path)
    service.ingest_jsonl(
        jsonl, source_label="phase7_2-rawtext", index_version="rt-1",
    )
    assert store.replace_all_called
    expected_chunk_texts = {
        "일본의 만화 ARIA를 원작으로 하는 TV 애니메이션.",
        "주인공은 미즈나시 아카리.",
        "신과 인간이 동거하는 코미디.",
    }
    actual_chunk_texts = {c.text for c in store.chunks}
    assert actual_chunk_texts == expected_chunk_texts
    # No embedding_text "제목: ..." prefix should leak into ChunkRow.text.
    for c in store.chunks:
        assert not c.text.startswith("제목:"), c.text


def test_ingest_default_and_rollback_produce_different_embed_sha(tmp_path: Path):
    """When the v4 doc-level retrieval_title differs from page_title, the
    two variants produce different vectors → different sha256.

    Uses a fixture row where retrieval_title carries a parent-work
    prefix so the variant choice actually matters.
    """
    rows = [
        {
            "doc_id": "doc-generic",
            "title": "등장인물",
            "retrieval_title": "가난뱅이 신이! / 등장인물",
            "sections": {
                "주역": {"chunks": ["주인공의 친구이자 라이벌."]},
            },
        },
    ]
    jsonl = _write_jsonl(tmp_path / "in.jsonl", rows)

    # Default (retrieval_title_section)
    service_def, _, _, _ = _build_ingest(tmp_path / "default")
    service_def.ingest_jsonl(jsonl, source_label="def", index_version="d1")
    sha_default = json.loads(
        (tmp_path / "default" / "idx" / "ingest_manifest.json").read_text(
            encoding="utf-8",
        ),
    )["embed_text_sha256"]

    # Rollback (title_section)
    service_rb, _, _, _ = _build_ingest(
        tmp_path / "rollback", variant=VARIANT_TITLE_SECTION,
    )
    service_rb.ingest_jsonl(jsonl, source_label="rb", index_version="r1")
    sha_rollback = json.loads(
        (tmp_path / "rollback" / "idx" / "ingest_manifest.json").read_text(
            encoding="utf-8",
        ),
    )["embed_text_sha256"]

    assert sha_default != sha_rollback


def test_ingest_v3_only_corpus_yields_same_sha_under_either_variant(tmp_path: Path):
    """v3 corpora (no retrieval_title) should produce the same byte string
    under either variant — retrieval_title_section falls back to
    page_title, so the hash MUST match.
    """
    rows = [
        {
            "doc_id": "doc-aria",
            "title": "ARIA The ORIGINATION",
            # NO retrieval_title field
            "sections": {
                "overview": {"chunks": ["일본의 만화 ARIA를 원작으로 하는 TV 애니메이션."]},
            },
        },
    ]
    jsonl = _write_jsonl(tmp_path / "in.jsonl", rows)

    service_def, _, _, _ = _build_ingest(tmp_path / "rts")
    service_def.ingest_jsonl(jsonl, source_label="x", index_version="v1")
    sha_rts = json.loads(
        (tmp_path / "rts" / "idx" / "ingest_manifest.json").read_text(
            encoding="utf-8",
        ),
    )["embed_text_sha256"]

    service_rb, _, _, _ = _build_ingest(
        tmp_path / "ts", variant=VARIANT_TITLE_SECTION,
    )
    service_rb.ingest_jsonl(jsonl, source_label="x", index_version="v1")
    sha_ts = json.loads(
        (tmp_path / "ts" / "idx" / "ingest_manifest.json").read_text(
            encoding="utf-8",
        ),
    )["embed_text_sha256"]

    assert sha_rts == sha_ts


# ---------------------------------------------------------------------------
# Byte-perfect parity: ingest's embed_text_sha256 = eval-side digest
# ---------------------------------------------------------------------------


def _eval_style_digest(strings: List[str]) -> str:
    """Replicate the v4_chunk_export.embed_text_sha256 convention.

    Each string is utf-8 encoded then a literal ``b"\\n"`` is appended;
    the SHA-256 hex digest is the result.
    """
    h = hashlib.sha256()
    for s in strings:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def test_ingest_manifest_sha256_matches_eval_export_convention(tmp_path: Path):
    """Pin the byte-perfect parity between production ingest and the
    eval-side ``embed_text_sha256`` convention.

    Produces 3 chunks via IngestService, then independently builds the
    same canonical strings off the same source data and digests them
    with the eval-side convention. The two hashes must match — that is
    the contract that lets the Phase 7.0 export sha256 be compared
    directly against the production ingest's manifest sha256.
    """
    rows = [
        {
            "doc_id": "doc-1",
            "title": "원더풀 프리큐어!",
            "sections": {
                "overview": {"chunks": ["2024년에 방영된 프리큐어 시리즈."]},
            },
        },
        {
            "doc_id": "doc-2",
            "title": "등장인물",
            "retrieval_title": "가난뱅이 신이! / 등장인물",
            "sections": {
                "주역": {"chunks": ["주인공의 친구이자 라이벌."]},
                "엑스트라": {"chunks": ["조연으로 등장한 인물들."]},
            },
        },
    ]
    jsonl = _write_jsonl(tmp_path / "in.jsonl", rows)

    # Drive the ingest path
    service, _, _, _ = _build_ingest(tmp_path)
    service.ingest_jsonl(
        jsonl, source_label="parity", index_version="par-1",
    )
    payload = json.loads(
        (tmp_path / "idx" / "ingest_manifest.json").read_text(encoding="utf-8"),
    )
    actual_sha = payload["embed_text_sha256"]

    # Independently produce the same strings via the production builder
    # using the v3-shape helper. Order MUST match the ingest loop's
    # iteration order: doc-1.overview chunk 0, doc-2.주역 chunk 0,
    # doc-2.엑스트라 chunk 0 (Python dict preserves insertion order
    # since 3.7, JSON preserves source order, and iteration over
    # _iter_documents respects the file order).
    expected_strings: List[str] = []
    for row in rows:
        for section_name, section_raw in row["sections"].items():
            for chunk_text in section_raw["chunks"]:
                expected_strings.append(
                    build_embedding_text_from_v3_chunk(
                        chunk_text=chunk_text,
                        title=row.get("title") or "",
                        section_name=section_name,
                        retrieval_title=row.get("retrieval_title") or "",
                        section_type="",
                        variant=VARIANT_RETRIEVAL_TITLE_SECTION,
                    )
                )
    expected_sha = _eval_style_digest(expected_strings)

    assert actual_sha == expected_sha


# ---------------------------------------------------------------------------
# load_ingest_manifest helper
# ---------------------------------------------------------------------------


def test_load_ingest_manifest_round_trip(tmp_path: Path):
    jsonl = _write_jsonl(tmp_path / "in.jsonl", _INGEST_FIXTURE_ROWS)
    service, _, _, _ = _build_ingest(tmp_path)
    service.ingest_jsonl(
        jsonl, source_label="rt", index_version="rt-1",
    )
    loaded = load_ingest_manifest(tmp_path / "idx")
    assert loaded is not None
    assert loaded.embedding_text_variant == VARIANT_RETRIEVAL_TITLE_SECTION
    assert loaded.embedding_text_builder_version == EMBEDDING_TEXT_BUILDER_VERSION
    assert loaded.chunk_count == 3
    assert loaded.document_count == 2
    assert len(loaded.embed_text_sha256) == 64


def test_load_ingest_manifest_returns_none_when_absent(tmp_path: Path):
    """No manifest file → None (don't raise; old indexes predate Phase 7.2)."""
    (tmp_path / "idx").mkdir(parents=True)
    assert load_ingest_manifest(tmp_path / "idx") is None


# ---------------------------------------------------------------------------
# Settings wiring
# ---------------------------------------------------------------------------


def test_settings_default_variant_is_retrieval_title_section(monkeypatch):
    """Pin the production default at the config layer.

    Pulled into a separate test so a future settings refactor that
    accidentally flips the default (e.g. via a typo in the Field
    default kwarg) trips this immediately.
    """
    # Import inside the test to avoid leaking get_settings cache state.
    from app.core.config import WorkerSettings
    s = WorkerSettings()
    assert s.rag_embedding_text_variant == VARIANT_RETRIEVAL_TITLE_SECTION


def test_settings_env_override_to_title_section(monkeypatch):
    monkeypatch.setenv(
        "AIPIPELINE_WORKER_RAG_EMBEDDING_TEXT_VARIANT", "title_section",
    )
    from app.core.config import WorkerSettings
    s = WorkerSettings()
    assert s.rag_embedding_text_variant == VARIANT_TITLE_SECTION
