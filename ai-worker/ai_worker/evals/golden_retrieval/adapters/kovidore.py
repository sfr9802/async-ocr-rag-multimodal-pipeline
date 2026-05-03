"""KoViDoRe v2 economic BEIR adapter for SearchUnit golden fixtures."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from ai_worker.evals.golden_retrieval.adapters.base import AdapterSummary


DATASET_ID = "kovidore-v2-economic-beir"
DATASET_ALIAS = "kovidore-economic"
HF_REPO = "whybe-choi/kovidore-v2-economic-beir"
PRIMARY_SEARCH_UNIT_TYPE = "PAGE"
PIPELINE_VERSION = "kovidore-fixture-v1"
LICENSE_NOTE = (
    "Annotations/qrels/metadata are CC BY 4.0. Original source documents and "
    "parsed markdown inherit publisher-specific licenses from document_metadata."
)
LICENSE_AND_TERMS = {
    "basis": "Hugging Face dataset card License information section",
    "annotationsQueriesQrelsAndRelatedMetadata": {
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "appliesTo": [
            "annotations",
            "queries",
            "query-document relevance judgments (qrels)",
            "related metadata generated for this corpus",
        ],
    },
    "sourceDocumentsAndParsedMarkdown": {
        "license": "Inherited from each original publisher",
        "appliesTo": [
            "original source documents",
            "parsed text in corpus.markdown",
        ],
        "perDocumentLicenseField": "document_metadata.license",
    },
}


@dataclass(frozen=True)
class KovidoreCorpusPage:
    corpus_id: int
    doc_id: str
    markdown: str
    elements: Any
    page_number_in_doc: int
    image: Optional[dict[str, Any]] = None

    @property
    def source_file_name(self) -> str:
        return self.doc_id

    @property
    def source_file_id(self) -> str:
        return _stable_fixture_id("kovi-src", self.doc_id)

    @property
    def artifact_id(self) -> str:
        return _stable_fixture_id("kovi-art", self.doc_id)

    @property
    def unit_type(self) -> str:
        return PRIMARY_SEARCH_UNIT_TYPE

    @property
    def unit_key(self) -> str:
        return f"page:{self.page_number_in_doc}"


@dataclass(frozen=True)
class KovidoreQuery:
    query_id: str
    query: str
    query_types: list[str]
    query_format: Optional[str]
    source_type: Optional[str]
    query_type_for_generation: Optional[str]
    answer: Optional[str]
    language: Optional[str]


@dataclass(frozen=True)
class KovidoreQrel:
    query_id: str
    corpus_id: int
    score: int


@dataclass(frozen=True)
class KovidoreInputs:
    corpus_rows: list[dict[str, Any]]
    query_rows: list[dict[str, Any]]
    qrel_rows: list[dict[str, Any]]
    document_metadata_rows: list[dict[str, Any]]
    input_manifest: dict[str, Optional[str]]


def parse_corpus_row(row: dict[str, Any]) -> KovidoreCorpusPage:
    raw = _unwrap_row(row)
    return KovidoreCorpusPage(
        corpus_id=_required_int(raw, "corpus_id"),
        doc_id=_required_text(raw, "doc_id"),
        markdown=str(raw.get("markdown") or ""),
        elements=parse_elements(raw.get("elements")),
        page_number_in_doc=_required_int(raw, "page_number_in_doc"),
        image=_json_object_or_none(raw.get("image")),
    )


def parse_query_row(row: dict[str, Any]) -> KovidoreQuery:
    raw = _unwrap_row(row)
    return KovidoreQuery(
        query_id=_required_text(raw, "query_id"),
        query=_required_text(raw, "query"),
        query_types=_list_of_text(raw.get("query_types")),
        query_format=_text_or_none(raw.get("query_format")),
        source_type=_text_or_none(raw.get("source_type")),
        query_type_for_generation=_text_or_none(raw.get("query_type_for_generation")),
        answer=_text_or_none(raw.get("answer")),
        language=_text_or_none(raw.get("language")),
    )


def parse_qrel_row(row: dict[str, Any]) -> KovidoreQrel:
    raw = _unwrap_row(row)
    return KovidoreQrel(
        query_id=_required_text(raw, "query_id"),
        corpus_id=_required_int(raw, "corpus_id"),
        score=_required_int(raw, "score"),
    )


def parse_elements(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(parsed, (list, dict)):
        return parsed
    return text


def corpus_page_to_search_unit(page: KovidoreCorpusPage) -> dict[str, Any]:
    metadata = {
        "dataset": DATASET_ID,
        "hfRepo": HF_REPO,
        "fileType": "pdf",
        "corpusId": page.corpus_id,
        "docId": page.doc_id,
        "pageNumberInDoc": page.page_number_in_doc,
        "elements": page.elements,
    }
    if page.image:
        metadata["image"] = page.image

    return {
        "searchUnitId": _search_unit_id(page),
        "logicalSearchUnitId": f"{DATASET_ALIAS}:{page.source_file_name}:{page.unit_key}",
        "sourceFileId": page.source_file_id,
        "sourceFileName": page.source_file_name,
        "extractedArtifactId": page.artifact_id,
        "indexId": _stable_index_id(page.source_file_id, page.unit_type, page.unit_key),
        "unitType": page.unit_type,
        "unitKey": page.unit_key,
        "title": f"{page.source_file_name} p.{page.page_number_in_doc}",
        "sectionPath": None,
        "pageStart": page.page_number_in_doc,
        "pageEnd": page.page_number_in_doc,
        "textContent": page.markdown,
        "embeddingStatus": "PENDING",
        "contentSha256": _sha256_text(_normalize_text_for_hash(page.markdown)),
        "metadataJson": metadata,
        "indexMetadata": {
            "dataset": DATASET_ID,
            "fileType": "pdf",
            "source_file_id": page.source_file_id,
            "source_file_name": page.source_file_name,
            "unit_type": page.unit_type,
            "unit_key": page.unit_key,
            "page_start": page.page_number_in_doc,
            "page_end": page.page_number_in_doc,
            "corpus_id": page.corpus_id,
            "doc_id": page.doc_id,
        },
    }


def corpus_pages_to_source_files(
    pages: Iterable[KovidoreCorpusPage],
    *,
    document_metadata: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[KovidoreCorpusPage]] = defaultdict(list)
    for page in pages:
        grouped[page.source_file_name].append(page)

    rows: list[dict[str, Any]] = []
    for source_file_name in sorted(grouped):
        source_pages = sorted(
            grouped[source_file_name],
            key=lambda page: page.page_number_in_doc,
        )
        first_page = source_pages[0]
        metadata = {
            "dataset": DATASET_ID,
            "hfRepo": HF_REPO,
            "docId": source_file_name,
            "language": "ko",
            "pageCountInFixture": len(source_pages),
            "pageNumbersInFixture": [page.page_number_in_doc for page in source_pages],
        }
        source_metadata = (document_metadata or {}).get(source_file_name)
        if source_metadata:
            metadata["sourceDocument"] = source_metadata
        rows.append({
            "sourceFileId": first_page.source_file_id,
            "logicalSourceId": source_file_name,
            "sourceFileName": source_file_name,
            "originalFilename": f"{source_file_name}.pdf",
            "fileType": "pdf",
            "mimeType": "application/pdf",
            "status": "READY",
            "storageUri": f"fixture://{DATASET_ID}/{source_file_name}.pdf",
            "metadataJson": metadata,
        })
    return rows


def corpus_pages_to_extracted_artifacts(
    pages: Iterable[KovidoreCorpusPage],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[KovidoreCorpusPage]] = defaultdict(list)
    for page in pages:
        grouped[page.doc_id].append(page)

    rows: list[dict[str, Any]] = []
    for doc_id in sorted(grouped):
        doc_pages = sorted(grouped[doc_id], key=lambda page: page.page_number_in_doc)
        first_page = doc_pages[0]
        payload = {
            "dataset": DATASET_ID,
            "hfRepo": HF_REPO,
            "docId": doc_id,
            "pages": [
                {
                    "corpusId": page.corpus_id,
                    "pageNumber": page.page_number_in_doc,
                    "pageNumberInDoc": page.page_number_in_doc,
                    "markdown": page.markdown,
                    "elements": page.elements,
                }
                for page in doc_pages
            ],
        }
        payload_text = json.dumps(_json_sanitize(payload), ensure_ascii=False, sort_keys=True)
        rows.append({
            "artifactId": first_page.artifact_id,
            "jobId": _stable_fixture_id("kovi-job", doc_id),
            "sourceFileId": first_page.source_file_id,
            "logicalSourceId": doc_id,
            "sourceFileName": doc_id,
            "artifactType": "OCR_RESULT_JSON",
            "artifactKey": f"{DATASET_ID}:{doc_id}",
            "storageUri": f"fixture://{DATASET_ID}/{doc_id}/ocr-result.json",
            "pipelineVersion": PIPELINE_VERSION,
            "checksumSha256": _sha256_text(payload_text),
            "payloadJson": payload,
            "metadataJson": {
                "dataset": DATASET_ID,
                "docId": doc_id,
                "source": "kovidore-corpus-markdown",
            },
        })
    return rows


def qrels_to_golden_queries(
    queries: Iterable[KovidoreQuery],
    qrels: Iterable[KovidoreQrel],
    corpus_by_id: dict[int, KovidoreCorpusPage],
    *,
    limit_queries: Optional[int] = None,
) -> list[dict[str, Any]]:
    qrels_by_query: dict[str, list[KovidoreQrel]] = defaultdict(list)
    for qrel in qrels:
        if qrel.score > 0:
            qrels_by_query[qrel.query_id].append(qrel)

    rows: list[dict[str, Any]] = []
    for query in queries:
        relevant = [
            qrel
            for qrel in sorted(
                qrels_by_query.get(query.query_id, []),
                key=lambda item: (-item.score, item.corpus_id),
            )
            if qrel.corpus_id in corpus_by_id
        ]
        if not relevant:
            continue

        has_fully_relevant = any(qrel.score == 2 for qrel in relevant)
        expected_qrels = [
            qrel for qrel in relevant
            if qrel.score == 2 or (qrel.score == 1 and not has_fully_relevant)
        ]
        acceptable_qrels = [
            qrel for qrel in relevant
            if qrel.score == 1 and has_fully_relevant
        ]

        row = {
            "id": f"kovidore-q-{query.query_id}",
            "query": query.query,
            "category": ",".join(query.query_types) if query.query_types else query.query_format,
            "expected": [
                _qrel_to_target(qrel, corpus_by_id[qrel.corpus_id])
                for qrel in expected_qrels
            ],
            "acceptable": [
                _qrel_to_target(qrel, corpus_by_id[qrel.corpus_id])
                for qrel in acceptable_qrels
            ],
            "mustNot": [],
            "answerHint": query.answer,
            "metadata": {
                "dataset": DATASET_ID,
                "queryId": _int_or_text(query.query_id),
                "language": query.language,
                "queryTypes": query.query_types,
                "queryFormat": query.query_format,
                "sourceType": query.source_type,
                "queryTypeForGeneration": query.query_type_for_generation,
            },
        }
        rows.append(_drop_none(row))

        if limit_queries is not None and len(rows) >= limit_queries:
            break
    return rows


def convert_kovidore_dataset(
    *,
    out_dir: Path,
    dataset_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    queries_path: Optional[Path] = None,
    qrels_path: Optional[Path] = None,
    limit_docs: Optional[int] = None,
    limit_queries: Optional[int] = None,
    document_metadata_path: Optional[Path] = None,
    hf_split: str = "train",
) -> AdapterSummary:
    inputs = load_kovidore_inputs(
        dataset_path=dataset_path,
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        document_metadata_path=document_metadata_path,
        hf_split=hf_split,
    )

    pages = [parse_corpus_row(row) for row in inputs.corpus_rows]
    pages = _limit_distinct_docs(pages, limit_docs)
    corpus_by_id = {page.corpus_id: page for page in pages}

    queries = [parse_query_row(row) for row in inputs.query_rows]
    qrels = [parse_qrel_row(row) for row in inputs.qrel_rows]
    document_metadata = read_document_metadata_rows(inputs.document_metadata_rows)

    source_files = corpus_pages_to_source_files(
        pages,
        document_metadata=document_metadata,
    )
    extracted_artifacts = corpus_pages_to_extracted_artifacts(pages)
    search_units = [corpus_page_to_search_unit(page) for page in pages]
    golden_queries = qrels_to_golden_queries(
        queries,
        qrels,
        corpus_by_id,
        limit_queries=limit_queries,
    )
    source_manifest = build_source_manifest(search_units)
    manifest = build_manifest(
        source_files=source_files,
        extracted_artifacts=extracted_artifacts,
        search_units=search_units,
        golden_queries=golden_queries,
        input_manifest=inputs.input_manifest,
        limit_docs=limit_docs,
        limit_queries=limit_queries,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "source_files.jsonl", source_files)
    write_jsonl(out_dir / "extracted_artifacts.jsonl", extracted_artifacts)
    write_jsonl(out_dir / "search_units.jsonl", search_units)
    write_json(out_dir / "source_manifest.json", source_manifest)
    write_jsonl(out_dir / "golden_queries.jsonl", golden_queries)
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "conversion_manifest.json", manifest)

    return AdapterSummary(
        dataset_id=DATASET_ID,
        out_dir=out_dir,
        source_files=len(source_files),
        search_units=len(search_units),
        golden_queries=len(golden_queries),
    )


def load_kovidore_inputs(
    *,
    dataset_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    queries_path: Optional[Path] = None,
    qrels_path: Optional[Path] = None,
    document_metadata_path: Optional[Path] = None,
    hf_split: str = "train",
) -> KovidoreInputs:
    if dataset_path is not None:
        paths = {
            "corpus": _resolve_subset_path(dataset_path, "corpus"),
            "queries": _resolve_subset_path(dataset_path, "queries"),
            "qrels": _resolve_subset_path(dataset_path, "qrels"),
            "documentMetadata": _resolve_subset_path(
                dataset_path,
                "document_metadata",
                required=False,
            ),
        }
        return KovidoreInputs(
            corpus_rows=read_rows(paths["corpus"]),
            query_rows=read_rows(paths["queries"]),
            qrel_rows=read_rows(paths["qrels"]),
            document_metadata_rows=read_rows(paths["documentMetadata"]) if paths["documentMetadata"] else [],
            input_manifest={key: str(value) if value is not None else None for key, value in paths.items()},
        )

    if corpus_path is not None and queries_path is not None and qrels_path is not None:
        return KovidoreInputs(
            corpus_rows=read_rows(corpus_path),
            query_rows=read_rows(queries_path),
            qrel_rows=read_rows(qrels_path),
            document_metadata_rows=read_rows(document_metadata_path) if document_metadata_path else [],
            input_manifest={
                "corpus": str(corpus_path),
                "queries": str(queries_path),
                "qrels": str(qrels_path),
                "documentMetadata": str(document_metadata_path) if document_metadata_path else None,
            },
        )

    return KovidoreInputs(
        corpus_rows=load_huggingface_subset_rows("corpus", hf_split),
        query_rows=load_huggingface_subset_rows("queries", hf_split),
        qrel_rows=load_huggingface_subset_rows("qrels", hf_split),
        document_metadata_rows=load_huggingface_subset_rows("document_metadata", hf_split, required=False),
        input_manifest={
            "corpus": f"hf://{HF_REPO}/corpus/{hf_split}",
            "queries": f"hf://{HF_REPO}/queries/{hf_split}",
            "qrels": f"hf://{HF_REPO}/qrels/{hf_split}",
            "documentMetadata": f"hf://{HF_REPO}/document_metadata/{hf_split}",
        },
    )


def load_huggingface_subset_rows(
    subset: str,
    split: str,
    *,
    required: bool = True,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        if not required:
            return []
        raise ValueError(
            "Hugging Face loading requires the optional 'datasets' package. "
            "Use --dataset-path or explicit --corpus/--queries/--qrels files for offline tests."
        ) from exc

    try:
        dataset = load_dataset(HF_REPO, subset, split=split)
    except Exception as exc:
        if not required:
            return []
        raise ValueError(f"Failed to load Hugging Face subset {subset!r} split {split!r}") from exc
    return [_json_sanitize(row) for row in dataset]


def build_manifest(
    *,
    source_files: list[dict[str, Any]],
    extracted_artifacts: list[dict[str, Any]],
    search_units: list[dict[str, Any]],
    golden_queries: list[dict[str, Any]],
    input_manifest: dict[str, Optional[str]],
    limit_docs: Optional[int],
    limit_queries: Optional[int],
) -> dict[str, Any]:
    return {
        "datasetId": DATASET_ID,
        "source": HF_REPO,
        "language": "ko",
        "domain": "economic",
        "documentType": "periodic_reports",
        "primarySearchUnitType": PRIMARY_SEARCH_UNIT_TYPE,
        "matchingIdentity": ["sourceFileName", "unitType", "unitKey"],
        "licenseNote": LICENSE_NOTE,
        "licenseAndTerms": LICENSE_AND_TERMS,
        "fixtureKind": "dev-eval-search-unit",
        "artifactTypes": ["OCR_RESULT_JSON"],
        "input": input_manifest,
        "limits": {
            "limitDocs": limit_docs,
            "limitQueries": limit_queries,
        },
        "outputs": {
            "sourceFiles": "source_files.jsonl",
            "extractedArtifacts": "extracted_artifacts.jsonl",
            "searchUnits": "search_units.jsonl",
            "sourceManifest": "source_manifest.json",
            "goldenQueries": "golden_queries.jsonl",
            "manifest": "manifest.json",
        },
        "counts": {
            "sourceFiles": len(source_files),
            "extractedArtifacts": len(extracted_artifacts),
            "searchUnits": len(search_units),
            "goldenQueries": len(golden_queries),
        },
    }


def build_source_manifest(search_units: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for unit in search_units:
        source_file_name = str(unit["sourceFileName"])
        by_source[source_file_name].append({
            "unitType": str(unit["unitType"]),
            "unitKey": str(unit["unitKey"]),
        })
    return {
        "dataset": DATASET_ID,
        "sources": [
            {
                "logicalSourceId": source_file_name,
                "fileName": source_file_name,
                "sourceFileName": source_file_name,
                "type": "pdf-page-fixture",
                "expectedUnits": sorted(
                    units,
                    key=lambda item: (item["unitType"], item["unitKey"]),
                ),
            }
            for source_file_name, units in sorted(by_source.items())
        ],
    }


def read_document_metadata(path: Path) -> dict[str, dict[str, Any]]:
    return read_document_metadata_rows(read_rows(path))


def read_document_metadata_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_doc_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw = _unwrap_row(row)
        doc_id = _text_or_none(raw.get("doc_id") or raw.get("docId") or raw.get("document_id"))
        file_name = _text_or_none(raw.get("file_name") or raw.get("fileName"))
        if not doc_id and file_name:
            doc_id = file_name[:-4] if file_name.lower().endswith(".pdf") else file_name
        if not doc_id:
            continue
        by_doc_id[doc_id] = _json_sanitize(raw)
    return by_doc_id


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return read_dataset_directory_rows(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(_unwrap_row(json.loads(stripped)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
        return rows
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [_unwrap_row(item) for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            if isinstance(raw.get("rows"), list):
                return [_unwrap_row(item) for item in raw["rows"] if isinstance(item, dict)]
            if isinstance(raw.get("data"), list):
                return [_unwrap_row(item) for item in raw["data"] if isinstance(item, dict)]
            return [_unwrap_row(raw)]
        raise ValueError(f"{path} must contain an object, rows array, or JSON array")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix == ".parquet":
        return read_parquet_rows(path)
    raise ValueError(f"Unsupported KoViDoRe input format: {path}")


def read_dataset_directory_rows(path: Path) -> list[dict[str, Any]]:
    try:
        from datasets import load_from_disk  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(f"Directory input requires datasets.load_from_disk: {path}") from exc
    dataset = load_from_disk(str(path))
    if hasattr(dataset, "keys") and "train" in dataset:
        dataset = dataset["train"]
    return [_json_sanitize(row) for row in dataset]


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "Reading parquet requires optional pandas/pyarrow. "
            "Export a small JSONL sample from the Hugging Face Dataset Viewer instead."
        ) from exc
    frame = pd.read_parquet(path)
    return [_json_sanitize(row) for row in frame.to_dict(orient="records")]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(_json_sanitize(row), ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = convert_kovidore_dataset(
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        corpus_path=Path(args.corpus) if args.corpus else None,
        queries_path=Path(args.queries) if args.queries else None,
        qrels_path=Path(args.qrels) if args.qrels else None,
        document_metadata_path=Path(args.document_metadata) if args.document_metadata else None,
        out_dir=Path(args.out_dir),
        limit_docs=args.limit_docs,
        limit_queries=args.limit_queries,
        hf_split=args.hf_split,
    )
    print(json.dumps({
        "dataset": summary.dataset_id,
        "outDir": str(summary.out_dir),
        "sourceFiles": summary.source_files,
        "searchUnits": summary.search_units,
        "goldenQueries": summary.golden_queries,
    }, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert KoViDoRe v2 economic BEIR rows to SearchUnit golden fixtures.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Local KoViDoRe dataset root. Looks for corpus/queries/qrels/document_metadata files first.",
    )
    parser.add_argument("--corpus", default=None, help="Path to KoViDoRe corpus rows JSON/JSONL/CSV/TSV/parquet.")
    parser.add_argument("--queries", default=None, help="Path to KoViDoRe queries rows JSON/JSONL/CSV/TSV/parquet.")
    parser.add_argument("--qrels", default=None, help="Path to KoViDoRe qrels rows JSON/JSONL/CSV/TSV/parquet.")
    parser.add_argument("--document-metadata", default=None, help="Optional document_metadata rows path.")
    parser.add_argument("--hf-split", default="train", help="Hugging Face split name when no local inputs are given.")
    parser.add_argument("--limit-docs", type=int, default=None, help="Keep only the first N distinct doc_id values.")
    parser.add_argument("--limit-queries", type=int, default=None, help="Write only the first N queries with expected units.")
    parser.add_argument("--out-dir", required=True, help="Directory for generated fixtures.")
    return parser


def _resolve_subset_path(dataset_path: Path, subset: str, *, required: bool = True) -> Optional[Path]:
    candidates: list[Path] = []
    for suffix in (".jsonl", ".json", ".csv", ".tsv", ".parquet"):
        candidates.append(dataset_path / f"{subset}{suffix}")
    subset_dir = dataset_path / subset
    candidates.extend([
        subset_dir / f"{subset}.jsonl",
        subset_dir / f"{subset}.json",
        subset_dir / "train.jsonl",
        subset_dir / "train.json",
        subset_dir / "data.jsonl",
        subset_dir / "data.json",
    ])
    candidates.extend(sorted(subset_dir.glob("*.jsonl")) if subset_dir.exists() else [])
    candidates.extend(sorted(subset_dir.glob("*.json")) if subset_dir.exists() else [])
    candidates.extend(sorted(subset_dir.glob("*.parquet")) if subset_dir.exists() else [])
    if subset_dir.exists():
        candidates.append(subset_dir)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not required:
        return None
    searched = ", ".join(str(path) for path in candidates[:8])
    raise FileNotFoundError(f"Could not find KoViDoRe subset {subset!r} under {dataset_path}. Tried: {searched}")


def _limit_distinct_docs(
    pages: list[KovidoreCorpusPage],
    limit_docs: Optional[int],
) -> list[KovidoreCorpusPage]:
    if limit_docs is None:
        return pages
    if limit_docs < 1:
        return []
    kept: set[str] = set()
    limited: list[KovidoreCorpusPage] = []
    for page in pages:
        if page.doc_id not in kept:
            if len(kept) >= limit_docs:
                continue
            kept.add(page.doc_id)
        if page.doc_id in kept:
            limited.append(page)
    return limited


def _unwrap_row(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("row"), dict):
        unwrapped = dict(row["row"])
        if "row_idx" in row:
            unwrapped.setdefault("row_idx", row["row_idx"])
        return unwrapped
    return dict(row)


def _qrel_to_target(qrel: KovidoreQrel, page: KovidoreCorpusPage) -> dict[str, Any]:
    return {
        "sourceFileName": page.source_file_name,
        "unitType": page.unit_type,
        "unitKey": page.unit_key,
        "pageStart": page.page_number_in_doc,
        "pageEnd": page.page_number_in_doc,
        "relevanceScore": qrel.score,
        "datasetCorpusId": qrel.corpus_id,
    }


def _search_unit_id(page: KovidoreCorpusPage) -> str:
    return _stable_fixture_id("kovi-su", page.source_file_name, page.unit_key)


def _stable_index_id(source_file_id: str, unit_type: str, unit_key: str) -> str:
    return f"source_file:{source_file_id}:unit:{unit_type}:{unit_key}"


def _stable_fixture_id(prefix: str, *parts: Any) -> str:
    raw = "\u001f".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text_for_hash(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _required_text(row: dict[str, Any], key: str) -> str:
    text = _text_or_none(row.get(key))
    if text is None:
        raise ValueError(f"KoViDoRe row missing required field {key!r}: {row}")
    return text


def _required_int(row: dict[str, Any], key: str) -> int:
    value = _int_or_none(row.get(key))
    if value is None:
        raise ValueError(f"KoViDoRe row missing integer field {key!r}: {row}")
    return value


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_text(value: str) -> int | str:
    parsed = _int_or_none(value)
    return parsed if parsed is not None else value


def _text_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _list_of_text(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, list):
            value = parsed
        elif "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        else:
            return [stripped]
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _text_or_none(item))]


def _json_object_or_none(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, dict):
        return _json_sanitize(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return _json_sanitize(parsed) if isinstance(parsed, dict) else None
    return None


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, dict):
            cleaned[key] = _drop_none(item)
        else:
            cleaned[key] = item
    return cleaned


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
