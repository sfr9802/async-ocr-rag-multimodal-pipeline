"""Build (or rebuild) the FAISS RAG index.

Reads a JSONL dataset in the port/rag schema, chunks each document with
the same constants port/rag uses, embeds every chunk via
sentence-transformers, builds a FAISS IndexFlatIP, and persists both the
vectors (on disk) and the document/chunk metadata (PostgreSQL ragmeta
schema).

Usage (from ai-worker/):

  # fixture dataset (small, committed for tests + first-run smoke)
  python -m scripts.build_rag_index --fixture

  # real dataset from port/rag
  python -m scripts.build_rag_index --input D:/port/rag/app/scripts/namu_anime_v3.jsonl

The worker must be RESTARTED after a rebuild so that the long-lived
RagCapability picks up the new index. Rebuilds are not hot-reloaded.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from app.capabilities.rag.embeddings import SentenceTransformerEmbedder
from app.capabilities.rag.faiss_index import FaissIndex
from app.capabilities.rag.ingest import IngestService
from app.capabilities.rag.metadata_store import RagMetadataStore
from app.core.config import get_settings
from app.core.logging import configure_logging

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
DEFAULT_FIXTURE = _FIXTURES_DIR / "anime_sample.jsonl"
KR_FIXTURE = _FIXTURES_DIR / "kr_sample.jsonl"
ANIME_CORPUS_KR_FIXTURE = _FIXTURES_DIR / "anime_corpus_kr.jsonl"
ANIME_KR_FIXTURE = _FIXTURES_DIR / "anime_kr.jsonl"
ENTERPRISE_INDEX_FIXTURE = _FIXTURES_DIR / "corpus_kr" / "index.jsonl"


# Per-fixture default (domain, language) applied when a row doesn't
# already carry the field. B.1's constraint ("anime fixture CONTENT must
# not change") forces us to inject these at ingest time instead of
# rewriting the JSONL in place.
_FIXTURE_DEFAULTS: dict[Path, dict[str, str]] = {
    DEFAULT_FIXTURE:         {"domain": "anime",      "language": "en"},
    KR_FIXTURE:              {"domain": "enterprise", "language": "ko"},
    ANIME_CORPUS_KR_FIXTURE: {"domain": "anime",      "language": "ko"},
    ANIME_KR_FIXTURE:        {"domain": "anime",      "language": "ko"},
    ENTERPRISE_INDEX_FIXTURE: {"domain": "enterprise", "language": "ko"},
}


def main() -> int:
    configure_logging()
    log = logging.getLogger("scripts.build_rag_index")

    parser = argparse.ArgumentParser(description="Build or rebuild the RAG index.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a JSONL dataset in the port/rag schema.",
    )
    parser.add_argument(
        "--fixture",
        nargs="?",
        const="en",
        default=None,
        choices=[
            "en", "kr", "both", "anime_corpus_kr",
            "anime", "enterprise", "all",
        ],
        help=(
            "Use committed fixtures instead of --input. "
            "'en' = anime_sample.jsonl (default), "
            "'kr' = kr_sample.jsonl, "
            "'both' = merge en + kr into a single index, "
            "'anime_corpus_kr' = anime_corpus_kr.jsonl "
            "(300-title Korean anime corpus sampled from port/rag), "
            "'anime' (Phase 9) = anime_sample.jsonl + anime_kr.jsonl, "
            "'enterprise' (Phase 9) = fixtures/corpus_kr/index.jsonl, "
            "'all' (Phase 9) = anime_sample + anime_kr + corpus_kr/index.jsonl."
        ),
    )
    parser.add_argument(
        "--index-version",
        type=str,
        default=None,
        help="Override the index version label. Defaults to v-<epoch>.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Free-form notes recorded on the index_builds row.",
    )
    parser.add_argument(
        "--with-images",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing images + image_manifest.jsonl. "
            "Builds a CLIP image index alongside the text index."
        ),
    )
    args = parser.parse_args()

    # Resolve the input path(s).
    input_paths: list[Path] = []
    if args.fixture is not None:
        fixture_lang = args.fixture
        if fixture_lang == "anime_corpus_kr":
            input_paths.append(ANIME_CORPUS_KR_FIXTURE)
        elif fixture_lang == "anime":
            input_paths.append(DEFAULT_FIXTURE)
            input_paths.append(ANIME_KR_FIXTURE)
        elif fixture_lang == "enterprise":
            input_paths.append(ENTERPRISE_INDEX_FIXTURE)
        elif fixture_lang == "all":
            # Unified Phase 9 index. Includes the two anime corpora (en +
            # ko), the pre-Phase-9 KR enterprise-ish placeholder (so
            # rag_sample_kr.jsonl baseline still reproduces), AND the
            # new synthetic enterprise corpus under fixtures/corpus_kr/.
            input_paths.append(DEFAULT_FIXTURE)
            input_paths.append(ANIME_KR_FIXTURE)
            input_paths.append(KR_FIXTURE)
            input_paths.append(ENTERPRISE_INDEX_FIXTURE)
        else:
            if fixture_lang in ("en", "both"):
                input_paths.append(DEFAULT_FIXTURE)
            if fixture_lang in ("kr", "both"):
                input_paths.append(KR_FIXTURE)
    elif args.input is not None:
        input_paths.append(args.input)
    else:
        parser.error("Provide --input <path> or --fixture")
        return 2

    for p in input_paths:
        if not p.exists():
            log.error("Dataset not found: %s", p)
            return 2

    # Merge and inject Phase 9 default (domain, language) per-fixture so
    # kr_sample.jsonl rows (which predate the V4 schema) don't land in
    # ragmeta with NULL domain/language after a rebuild. Rows that
    # already carry the fields are left untouched — the injection is
    # strictly a fill-forward.
    input_path = _prepare_merged_fixture(
        input_paths, log, fixture_label=args.fixture,
    )

    settings = get_settings()
    log.info("Dataset:        %s", input_path)
    log.info("Index dir:      %s", settings.rag_index_dir)
    log.info("Embed model:    %s", settings.rag_embedding_model)
    log.info("Query prefix:   %r", settings.rag_embedding_prefix_query)
    log.info("Passage prefix: %r", settings.rag_embedding_prefix_passage)
    log.info("DB DSN:         %s", _redact(settings.rag_db_dsn))

    embedder = SentenceTransformerEmbedder(
        model_name=settings.rag_embedding_model,
        query_prefix=settings.rag_embedding_prefix_query,
        passage_prefix=settings.rag_embedding_prefix_passage,
    )
    metadata = RagMetadataStore(settings.rag_db_dsn)
    metadata.ping()

    index = FaissIndex(Path(settings.rag_index_dir))
    ingester = IngestService(
        embedder=embedder,
        metadata_store=metadata,
        index=index,
    )

    started = time.time()
    result = ingester.ingest_jsonl(
        input_path,
        source_label=input_path.name,
        index_version=args.index_version,
        notes=args.notes,
    )
    elapsed = time.time() - started

    # Log the final build identity from the index itself, not from settings —
    # both should agree, but we want the log line to reflect exactly what was
    # written into build.json and ragmeta.index_builds so ops can grep for it.
    log.info(
        "Ingest done in %.1fs: %d documents, %d chunks, version=%s, "
        "embedding_model=%s, embedding_dim=%d",
        elapsed,
        result.document_count,
        result.chunk_count,
        result.info.index_version,
        result.info.embedding_model,
        result.info.dimension,
    )

    # Record languages + domains in build.json for downstream tools.
    languages = ["en"]
    domains = ["anime"]
    if args.fixture == "kr":
        languages = ["kr"]
        domains = ["enterprise"]
    elif args.fixture == "both":
        languages = ["en", "kr"]
        domains = ["anime", "enterprise"]
    elif args.fixture == "anime_corpus_kr":
        languages = ["ko"]
        domains = ["anime"]
    elif args.fixture == "anime":
        languages = ["en", "ko"]
        domains = ["anime"]
    elif args.fixture == "enterprise":
        languages = ["ko"]
        domains = ["enterprise"]
    elif args.fixture == "all":
        languages = ["en", "ko"]
        domains = ["anime", "enterprise"]
    build_json_path = Path(settings.rag_index_dir) / "build.json"
    if build_json_path.exists():
        import json as _json
        build_meta = _json.loads(build_json_path.read_text(encoding="utf-8"))
        build_meta["languages"] = languages
        build_meta["domains"] = domains
        build_json_path.write_text(
            _json.dumps(build_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("build.json updated with languages=%s domains=%s", languages, domains)

    stats = metadata.stats()
    log.info("ragmeta stats: %s", stats)

    # -- optional: build CLIP image index --------------------------------
    if args.with_images is not None:
        _build_image_index(
            args.with_images,
            settings=settings,
            index_version=result.info.index_version,
            log=log,
        )

    log.info(
        "Next steps: restart the worker so it picks up the new index. "
        "The worker will refuse to register the RAG capability if its "
        "configured embedding model does not match %r.",
        result.info.embedding_model,
    )
    return 0


def _prepare_merged_fixture(
    input_paths: list[Path],
    log: logging.Logger,
    *,
    fixture_label: str | None,
) -> Path:
    """Merge input JSONL files, injecting per-fixture default metadata.

    For each row in each fixture: if the row already carries ``domain``
    or ``language`` (e.g. ``anime_kr.jsonl`` sets both explicitly), the
    field is left untouched. Otherwise the injection table in
    ``_FIXTURE_DEFAULTS`` fills them in. Rows from unrecognized inputs
    (``--input <path>``) pass through unchanged so external corpora
    don't silently pick up a wrong label.

    Always returns a temp file path; even the single-fixture case goes
    through the injection pass so the ragmeta rows land with consistent
    domain/language metadata.
    """
    import json as _json
    import tempfile

    # Single-fixture ingest with no recognized default: pass the file
    # through directly to preserve exact byte-for-byte behaviour (this
    # matches the pre-Phase-9 path when --input was used on an ad-hoc
    # dataset).
    if len(input_paths) == 1 and input_paths[0] not in _FIXTURE_DEFAULTS:
        return input_paths[0]

    merged = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
    )
    try:
        total = 0
        for fixture_path in input_paths:
            defaults = _FIXTURE_DEFAULTS.get(fixture_path, {})
            rows_written = 0
            with fixture_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith("#"):
                        continue
                    try:
                        row = _json.loads(stripped)
                    except _json.JSONDecodeError:
                        log.warning("Skipping malformed line in %s", fixture_path.name)
                        continue
                    if isinstance(row, dict) and defaults:
                        for key, value in defaults.items():
                            if row.get(key) is None:
                                row[key] = value
                    merged.write(_json.dumps(row, ensure_ascii=False))
                    merged.write("\n")
                    rows_written += 1
            log.info(
                "merged %s (%d rows, defaults=%s)",
                fixture_path.name, rows_written, defaults or "{}",
            )
            total += rows_written
    finally:
        merged.close()
    log.info(
        "Prepared merged fixture (%s): %d rows from %d files -> %s",
        fixture_label or "custom", total, len(input_paths), merged.name,
    )
    return Path(merged.name)


def _build_image_index(
    images_dir: Path,
    *,
    settings,
    index_version: str,
    log: logging.Logger,
) -> None:
    """Build a CLIP image FAISS index from a manifest directory."""
    import hashlib as _hashlib
    import json as _json

    from app.capabilities.rag.image_embeddings import ClipImageEmbedder
    from app.capabilities.rag.image_index import ImageFaissIndex
    from app.capabilities.rag.image_metadata_store import (
        ImageChunkRow,
        ImageMetadataStore,
    )

    manifest_path = images_dir / "image_manifest.jsonl"
    if not manifest_path.exists():
        log.error("image_manifest.jsonl not found in %s", images_dir)
        return

    # Parse manifest
    entries = []
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(_json.loads(line))
        except _json.JSONDecodeError as e:
            log.warning("Invalid JSON on line %d of manifest: %s — skipping", line_no, e)
    log.info("Image manifest: %d entries from %s", len(entries), manifest_path)

    if not entries:
        log.warning("Empty image manifest — skipping image index build.")
        return

    # Read image bytes
    all_bytes: list[bytes] = []
    valid_entries: list[dict] = []
    for entry in entries:
        img_path = images_dir / entry["path"]
        if not img_path.exists():
            log.warning("Image not found: %s — skipping", img_path)
            continue
        all_bytes.append(img_path.read_bytes())
        valid_entries.append(entry)

    if not valid_entries:
        log.warning("No valid images found — skipping image index build.")
        return

    # Encode
    clip_model = settings.cross_modal_clip_model
    log.info("CLIP model: %s — encoding %d images ...", clip_model, len(all_bytes))
    embedder = ClipImageEmbedder(model_name=clip_model)
    started = time.time()
    vectors = embedder.encode_images(all_bytes)
    log.info("CLIP encoding done in %.1fs (dim=%d)", time.time() - started, vectors.shape[1])

    # Build FAISS image index
    image_index = ImageFaissIndex(Path(settings.rag_index_dir))
    image_index.build(
        vectors,
        index_version=index_version,
        embedding_model=clip_model,
    )

    # Write image metadata to DB
    rows: list[ImageChunkRow] = []
    for i, entry in enumerate(valid_entries):
        sha = _hashlib.sha256(all_bytes[i]).hexdigest()
        rows.append(ImageChunkRow(
            image_id=entry.get("image_id", f"img-{i:04d}"),
            doc_id=entry["doc_id"],
            page_number=entry.get("page_number"),
            source_uri=entry.get("source_uri"),
            sha256=sha,
            caption=entry.get("caption"),
            section_hint=entry.get("section_hint"),
            faiss_row_id=i,
            index_version=index_version,
        ))

    img_meta = ImageMetadataStore(settings.rag_db_dsn)
    img_meta.replace_all(
        images=rows,
        index_version=index_version,
        embedding_model=clip_model,
        embedding_dim=int(vectors.shape[1]),
        faiss_index_path=str(Path(settings.rag_index_dir) / "image"),
    )
    log.info("Image index built: %d images, version=%s", len(rows), index_version)


def _redact(dsn: str) -> str:
    # Never print passwords in startup logs.
    out = []
    for part in dsn.split():
        if part.startswith("password="):
            out.append("password=****")
        else:
            out.append(part)
    return " ".join(out)


if __name__ == "__main__":
    sys.exit(main())
