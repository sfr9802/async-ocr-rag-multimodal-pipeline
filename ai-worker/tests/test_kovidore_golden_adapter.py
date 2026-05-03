from __future__ import annotations

import json

import pytest

from ai_worker.evals.golden_retrieval.adapters.kovidore import (
    KovidoreQrel,
    KovidoreQuery,
    convert_kovidore_dataset,
    corpus_page_to_search_unit,
    corpus_pages_to_extracted_artifacts,
    parse_corpus_row,
    qrels_to_golden_queries,
)
from ai_worker.evals.golden_retrieval.import_kovidore import (
    load_fixture_bundle,
    upsert_fixture_rows,
)
from ai_worker.evals.golden_retrieval.matching import matches_result
from ai_worker.evals.golden_retrieval.run import (
    GoldenQuery,
    evaluate_golden_queries,
    result_matches_spec,
)


def _corpus_row(**overrides):
    row = {
        "corpus_id": 913,
        "doc_id": "2022_12_recent_economic_trends",
        "markdown": "# 금융 시장\n\n국고채 금리는 하락",
        "elements": "[{'category': 'heading1', 'content': {'markdown': '# 금융 시장'}}]",
        "page_number_in_doc": 52,
        "image": {"src": "https://example.test/page.jpg", "height": 10, "width": 10},
    }
    row.update(overrides)
    return row


def _query_row(**overrides):
    row = {
        "query_id": 0,
        "query": "2022년 금융 시장의 금리 변화는?",
        "language": "ko",
        "query_types": ["extractive", "numerical"],
        "query_format": "question",
        "source_type": "context",
        "query_type_for_generation": "extractive",
        "answer": "국고채 금리는 하락했습니다.",
    }
    row.update(overrides)
    return row


def test_kovidore_corpus_row_parses_to_page_search_unit():
    page = parse_corpus_row(_corpus_row())
    unit = corpus_page_to_search_unit(page)

    assert page.source_file_name == "2022_12_recent_economic_trends"
    assert page.unit_type == "PAGE"
    assert page.unit_key == "page:52"
    assert unit["sourceFileId"].startswith("kovi-src-")
    assert unit["sourceFileName"] == "2022_12_recent_economic_trends"
    assert unit["extractedArtifactId"].startswith("kovi-art-")
    assert unit["unitType"] == "PAGE"
    assert unit["unitKey"] == "page:52"
    assert unit["embeddingStatus"] == "PENDING"
    assert unit["textContent"] == "# 금융 시장\n\n국고채 금리는 하락"
    assert unit["metadataJson"]["dataset"] == "kovidore-v2-economic-beir"
    assert unit["metadataJson"]["corpusId"] == 913
    assert unit["metadataJson"]["docId"] == "2022_12_recent_economic_trends"
    assert unit["metadataJson"]["elements"][0]["category"] == "heading1"


def test_kovidore_qrels_map_to_expected_page_search_units():
    page = parse_corpus_row(_corpus_row())
    query = KovidoreQuery(
        query_id="0",
        query="금리 변화",
        query_types=["extractive"],
        query_format="question",
        source_type="context",
        query_type_for_generation="extractive",
        answer="금리는 하락",
        language="ko",
    )

    rows = qrels_to_golden_queries(
        [query],
        [KovidoreQrel(query_id="0", corpus_id=913, score=2)],
        {913: page},
    )

    assert rows[0]["id"] == "kovidore-q-0"
    assert rows[0]["category"] == "extractive"
    assert rows[0]["answerHint"] == "금리는 하락"
    assert rows[0]["expected"] == [
        {
            "sourceFileName": "2022_12_recent_economic_trends",
            "unitType": "PAGE",
            "unitKey": "page:52",
            "pageStart": 52,
            "pageEnd": 52,
            "relevanceScore": 2,
            "datasetCorpusId": 913,
        }
    ]
    assert rows[0]["acceptable"] == []


def test_kovidore_qrels_split_expected_and_acceptable_by_grade():
    page_52 = parse_corpus_row(_corpus_row())
    page_53 = parse_corpus_row(_corpus_row(corpus_id=914, page_number_in_doc=53))
    query = KovidoreQuery(
        query_id="0",
        query="금리 변화",
        query_types=["extractive"],
        query_format="question",
        source_type="context",
        query_type_for_generation="extractive",
        answer=None,
        language="ko",
    )

    rows = qrels_to_golden_queries(
        [query],
        [
            KovidoreQrel(query_id="0", corpus_id=913, score=2),
            KovidoreQrel(query_id="0", corpus_id=914, score=1),
        ],
        {913: page_52, 914: page_53},
    )

    assert rows[0]["expected"][0]["unitKey"] == "page:52"
    assert rows[0]["expected"][0]["relevanceScore"] == 2
    assert rows[0]["acceptable"][0]["unitKey"] == "page:53"
    assert rows[0]["acceptable"][0]["relevanceScore"] == 1


def test_kovidore_score_one_promotes_to_expected_when_no_score_two():
    page = parse_corpus_row(_corpus_row())
    query = KovidoreQuery(
        query_id="0",
        query="금리 변화",
        query_types=["extractive"],
        query_format="question",
        source_type="context",
        query_type_for_generation="extractive",
        answer=None,
        language="ko",
    )

    rows = qrels_to_golden_queries(
        [query],
        [KovidoreQrel(query_id="0", corpus_id=913, score=1)],
        {913: page},
    )

    assert rows[0]["expected"][0]["unitKey"] == "page:52"
    assert rows[0]["expected"][0]["relevanceScore"] == 1
    assert rows[0]["acceptable"] == []


def test_kovidore_conversion_writes_golden_queries_jsonl(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    out_dir = tmp_path / "out"
    _write_jsonl(corpus_path, [{"row": _corpus_row()}])
    _write_jsonl(queries_path, [{"row": _query_row()}])
    _write_jsonl(qrels_path, [{"row": {"query_id": 0, "corpus_id": 913, "score": 1}}])

    summary = convert_kovidore_dataset(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        out_dir=out_dir,
        limit_docs=1,
        limit_queries=1,
    )

    golden_rows = _read_jsonl(out_dir / "golden_queries.jsonl")
    extracted_artifacts = _read_jsonl(out_dir / "extracted_artifacts.jsonl")
    search_units = _read_jsonl(out_dir / "search_units.jsonl")
    source_files = _read_jsonl(out_dir / "source_files.jsonl")
    source_manifest = json.loads((out_dir / "source_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary.search_units == 1
    assert summary.golden_queries == 1
    assert golden_rows[0]["expected"][0]["unitKey"] == "page:52"
    assert golden_rows[0]["expected"][0]["relevanceScore"] == 1
    assert extracted_artifacts[0]["artifactType"] == "OCR_RESULT_JSON"
    assert extracted_artifacts[0]["payloadJson"]["pages"][0]["corpusId"] == 913
    assert search_units[0]["metadataJson"]["elements"][0]["category"] == "heading1"
    assert source_files[0]["sourceFileName"] == "2022_12_recent_economic_trends"
    assert source_files[0]["status"] == "READY"
    assert source_manifest["sources"][0]["expectedUnits"] == [
        {"unitType": "PAGE", "unitKey": "page:52"}
    ]
    assert manifest["datasetId"] == "kovidore-v2-economic-beir"
    assert manifest["primarySearchUnitType"] == "PAGE"
    assert "CC BY 4.0" in manifest["licenseNote"]


def test_kovidore_conversion_accepts_dataset_path(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    _write_jsonl(dataset_path / "corpus.jsonl", [{"row": _corpus_row()}])
    _write_jsonl(dataset_path / "queries.jsonl", [{"row": _query_row()}])
    _write_jsonl(dataset_path / "qrels.jsonl", [{"row": {"query_id": 0, "corpus_id": 913, "score": 2}}])

    summary = convert_kovidore_dataset(
        dataset_path=dataset_path,
        out_dir=tmp_path / "out",
        limit_docs=1,
        limit_queries=1,
    )

    assert summary.source_files == 1
    assert summary.search_units == 1
    assert summary.golden_queries == 1


def test_stable_identity_matching_drives_hit_at_k():
    query = GoldenQuery(
        id="kovidore-economic-0",
        query="금리 변화",
        expected=[
            {
                "sourceFileName": "2022_12_recent_economic_trends",
                "unitType": "PAGE",
                "unitKey": "page:52",
                "pageStart": 52,
                "pageEnd": 52,
            }
        ],
    )

    def retrieve(_query: str, top_k: int):
        return [
            {
                "sourceFileName": "other_doc",
                "unitType": "PAGE",
                "unitKey": "page:1",
                "pageStart": 1,
                "pageEnd": 1,
            },
            {
                "sourceFileName": "2022_12_recent_economic_trends",
                "unitType": "PAGE",
                "unitKey": "page:52",
                "pageStart": 52,
                "pageEnd": 52,
            },
        ][:top_k]

    report = evaluate_golden_queries([query], retrieve=retrieve, top_k=2)

    assert report["rows"][0]["firstMatchRank"] == 2
    assert report["metrics"]["hit@1"] == pytest.approx(0.0)
    assert report["metrics"]["hit@3"] == pytest.approx(1.0)
    assert report["metrics"]["ndcg@5"] > 0.0
    assert report["metrics"]["citation_match@5"] == pytest.approx(1.0)


def test_result_matching_falls_back_to_source_file_and_page_range():
    spec = {
        "sourceFileName": "2022_12_recent_economic_trends",
        "unitType": "PAGE",
        "unitKey": "page:52",
        "pageStart": 52,
        "pageEnd": 52,
    }

    assert result_matches_spec(
        {
            "sourceFileName": "2022_12_recent_economic_trends",
            "unitType": "PAGE",
            "pageStart": 52,
            "pageEnd": 52,
        },
        spec,
    )
    assert matches_result(
        {
            "sourceFileName": "2022_12_recent_economic_trends",
            "unitType": "PAGE",
            "pageStart": 52,
            "pageEnd": 52,
        },
        spec,
    )
    assert not result_matches_spec(
        {
            "sourceFileName": "2022_12_recent_economic_trends",
            "unitType": "TABLE",
            "pageStart": 52,
            "pageEnd": 52,
        },
        spec,
    )
    assert not result_matches_spec(
        {
            "sourceFileName": "2022_12_recent_economic_trends",
            "unitType": "PAGE",
            "unitKey": "page:53",
            "pageStart": 52,
            "pageEnd": 52,
        },
        spec,
    )


def test_extracted_artifact_groups_pages_by_doc_id():
    page_52 = parse_corpus_row(_corpus_row())
    page_53 = parse_corpus_row(_corpus_row(corpus_id=914, page_number_in_doc=53))

    artifacts = corpus_pages_to_extracted_artifacts([page_53, page_52])

    assert len(artifacts) == 1
    assert artifacts[0]["artifactType"] == "OCR_RESULT_JSON"
    assert artifacts[0]["artifactKey"] == "kovidore-v2-economic-beir:2022_12_recent_economic_trends"
    assert [page["pageNumberInDoc"] for page in artifacts[0]["payloadJson"]["pages"]] == [52, 53]


def test_import_fixture_upserts_are_stable_and_idempotent(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    out_dir = tmp_path / "out"
    _write_jsonl(corpus_path, [{"row": _corpus_row()}])
    _write_jsonl(queries_path, [{"row": _query_row()}])
    _write_jsonl(qrels_path, [{"row": {"query_id": 0, "corpus_id": 913, "score": 2}}])
    convert_kovidore_dataset(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        out_dir=out_dir,
    )
    bundle = load_fixture_bundle(out_dir)
    cursor = _RecordingCursor()

    upsert_fixture_rows(cursor, bundle, batch_size=1)
    first_params = list(cursor.params)
    upsert_fixture_rows(cursor, bundle, batch_size=1)

    assert bundle.source_files[0]["sourceFileId"].startswith("kovi-src-")
    assert bundle.search_units[0]["embeddingStatus"] == "PENDING"
    assert any("ON CONFLICT (id) DO UPDATE" in sql for sql in cursor.sql)
    assert any("ON CONFLICT (source_file_id, unit_type, unit_key)" in sql for sql in cursor.sql)
    assert first_params[1][0] == cursor.params[len(first_params) + 1][0]


class _RecordingCursor:
    def __init__(self):
        self.sql = []
        self.params = []

    def execute(self, sql, params):
        self.sql.append(sql)
        self.params.append(params)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
