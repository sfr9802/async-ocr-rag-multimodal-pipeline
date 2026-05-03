"""Postgres DAO for the ragmeta schema.

Kept deliberately procedural — psycopg2 + hand-written SQL. SQLAlchemy
would add layers we don't need at phase-2 scale, and we want the RAG data
path to have zero overlap with the Spring side's JPA stack.

The schema (ragmeta.documents / ragmeta.chunks / ragmeta.index_builds) is
created by the core-api's Flyway migration V2. This module only reads and
writes rows; it never creates tables.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import psycopg2
import psycopg2.extras

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentRow:
    doc_id: str
    title: Optional[str]
    source: Optional[str]
    category: Optional[str]
    metadata: Optional[dict]
    domain: Optional[str] = None
    language: Optional[str] = None


# Whitelist of filter keys honored by ``doc_ids_matching``. We hard-code
# this rather than letting callers SELECT against arbitrary columns —
# the LLM query parser is the primary populator and we don't want a
# parser hallucination to turn into a SQL injection vector or accidental
# scan against an unindexed column.
_FILTER_COLUMNS = frozenset({"domain", "category", "language"})


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    doc_id: str
    section: Optional[str]
    chunk_order: int
    text: str
    token_count: Optional[int]
    faiss_row_id: int
    index_version: str
    extra: Optional[dict] = None


@dataclass(frozen=True)
class ChunkLookupResult:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    faiss_row_id: int
    extra: Optional[dict] = None


class RagMetadataStore:
    """Connection factory + DAO for ragmeta.*."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    # ------------------------------------------------------------------
    # connection management
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Fail fast at startup if the DB is unreachable or the schema is missing."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'ragmeta'")
            if cur.fetchone() is None:
                raise RuntimeError(
                    "ragmeta schema is missing. Has the core-api Flyway V2 migration run?"
                )

    @contextmanager
    def _connect(self):
        conn = psycopg2.connect(self._dsn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    def replace_all(
        self,
        *,
        documents: List[DocumentRow],
        chunks: List[ChunkRow],
        index_version: str,
        embedding_model: str,
        embedding_dim: int,
        faiss_index_path: str,
        notes: Optional[str] = None,
    ) -> None:
        """Atomically replace the ragmeta contents with a fresh build.

        The contract is: after this call, the DB holds EXACTLY the rows
        from this build. Older documents/chunks from previous builds are
        removed so there is no ambiguity about which version is live.
        `index_builds` keeps history across versions.
        """
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        with self._connect() as conn, conn.cursor() as cur:
            # Wipe previous live data (history survives in index_builds).
            cur.execute("DELETE FROM ragmeta.chunks")
            cur.execute("DELETE FROM ragmeta.documents")

            if documents:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO ragmeta.documents
                      (doc_id, title, source, category, metadata_json,
                       domain, language, created_at, updated_at)
                    VALUES %s
                    """,
                    [
                        (
                            d.doc_id,
                            d.title,
                            d.source,
                            d.category,
                            json.dumps(d.metadata) if d.metadata is not None else None,
                            d.domain,
                            d.language,
                            now,
                            now,
                        )
                        for d in documents
                    ],
                )

            if chunks:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO ragmeta.chunks
                      (chunk_id, doc_id, section, chunk_order, text,
                       token_count, faiss_row_id, index_version, extra_json, created_at)
                    VALUES %s
                    """,
                    [
                        (
                            c.chunk_id,
                            c.doc_id,
                            c.section,
                            c.chunk_order,
                            c.text,
                            c.token_count,
                            c.faiss_row_id,
                            c.index_version,
                            json.dumps(c.extra) if c.extra is not None else None,
                            now,
                        )
                        for c in chunks
                    ],
                )

            cur.execute(
                """
                INSERT INTO ragmeta.index_builds
                  (index_version, embedding_model, embedding_dim,
                   chunk_count, document_count, faiss_index_path, notes, built_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (index_version) DO UPDATE SET
                  embedding_model = EXCLUDED.embedding_model,
                  embedding_dim = EXCLUDED.embedding_dim,
                  chunk_count = EXCLUDED.chunk_count,
                  document_count = EXCLUDED.document_count,
                  faiss_index_path = EXCLUDED.faiss_index_path,
                  notes = EXCLUDED.notes,
                  built_at = EXCLUDED.built_at
                """,
                (
                    index_version,
                    embedding_model,
                    embedding_dim,
                    len(chunks),
                    len(documents),
                    faiss_index_path,
                    notes,
                    now,
                ),
            )
        log.info(
            "ragmeta replaced: %d documents, %d chunks, version=%s",
            len(documents), len(chunks), index_version,
        )

    def upsert_index_rows(
        self,
        *,
        documents: List[DocumentRow],
        chunks: List[ChunkRow],
    ) -> None:
        """Upsert SearchUnit-backed rows without rebuilding all metadata.

        The vector index writer owns FAISS row allocation. This method only
        makes the ragmeta lookup rows idempotent for a stable chunk_id/index_id.
        """
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        with self._connect() as conn, conn.cursor() as cur:
            if documents:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO ragmeta.documents
                      (doc_id, title, source, category, metadata_json,
                       domain, language, created_at, updated_at)
                    VALUES %s
                    ON CONFLICT (doc_id) DO UPDATE SET
                      title = EXCLUDED.title,
                      source = EXCLUDED.source,
                      category = EXCLUDED.category,
                      metadata_json = EXCLUDED.metadata_json,
                      domain = EXCLUDED.domain,
                      language = EXCLUDED.language,
                      updated_at = EXCLUDED.updated_at
                    """,
                    [
                        (
                            d.doc_id,
                            d.title,
                            d.source,
                            d.category,
                            json.dumps(d.metadata) if d.metadata is not None else None,
                            d.domain,
                            d.language,
                            now,
                            now,
                        )
                        for d in documents
                    ],
                )

            if chunks:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO ragmeta.chunks
                      (chunk_id, doc_id, section, chunk_order, text,
                       token_count, faiss_row_id, index_version, extra_json, created_at)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                      doc_id = EXCLUDED.doc_id,
                      section = EXCLUDED.section,
                      chunk_order = EXCLUDED.chunk_order,
                      text = EXCLUDED.text,
                      token_count = EXCLUDED.token_count,
                      faiss_row_id = EXCLUDED.faiss_row_id,
                      index_version = EXCLUDED.index_version,
                      extra_json = EXCLUDED.extra_json
                    """,
                    [
                        (
                            c.chunk_id,
                            c.doc_id,
                            c.section,
                            c.chunk_order,
                            c.text,
                            c.token_count,
                            c.faiss_row_id,
                            c.index_version,
                            json.dumps(c.extra) if c.extra is not None else None,
                            now,
                        )
                        for c in chunks
                    ],
                )
        log.info(
            "ragmeta search-unit rows upserted: %d documents, %d chunks",
            len(documents),
            len(chunks),
        )

    def record_index_build(
        self,
        *,
        index_version: str,
        embedding_model: str,
        embedding_dim: int,
        chunk_count: int,
        document_count: int,
        faiss_index_path: str,
        notes: Optional[str] = None,
    ) -> None:
        """Upsert the latest build metadata for a live index rewrite."""
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ragmeta.index_builds
                  (index_version, embedding_model, embedding_dim,
                   chunk_count, document_count, faiss_index_path, notes, built_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (index_version) DO UPDATE SET
                  embedding_model = EXCLUDED.embedding_model,
                  embedding_dim = EXCLUDED.embedding_dim,
                  chunk_count = EXCLUDED.chunk_count,
                  document_count = EXCLUDED.document_count,
                  faiss_index_path = EXCLUDED.faiss_index_path,
                  notes = EXCLUDED.notes,
                  built_at = EXCLUDED.built_at
                """,
                (
                    index_version,
                    embedding_model,
                    embedding_dim,
                    int(chunk_count),
                    int(document_count),
                    faiss_index_path,
                    notes,
                    now,
                ),
            )

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def lookup_chunks_by_faiss_rows(
        self, index_version: str, faiss_row_ids: Iterable[int]
    ) -> List[ChunkLookupResult]:
        ids = list(faiss_row_ids)
        if not ids:
            return []
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_id, section, text, faiss_row_id, extra_json
                  FROM ragmeta.chunks
                 WHERE index_version = %s
                   AND faiss_row_id = ANY(%s)
                """,
                (index_version, ids),
            )
            rows = cur.fetchall()
        by_row = {r[4]: ChunkLookupResult(
            chunk_id=r[0], doc_id=r[1], section=r[2] or "", text=r[3], faiss_row_id=r[4],
            extra=_json_dict_or_none(r[5]),
        ) for r in rows}
        # Preserve the FAISS ranking order of the input ids.
        return [by_row[i] for i in ids if i in by_row]

    def list_chunks(self, index_version: str) -> List[ChunkRow]:
        """Return chunks for an index version ordered by FAISS row id."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_id, section, chunk_order, text,
                       token_count, faiss_row_id, index_version, extra_json
                  FROM ragmeta.chunks
                 WHERE index_version = %s
                 ORDER BY faiss_row_id ASC
                """,
                (index_version,),
            )
            rows = cur.fetchall()
        return [
            ChunkRow(
                chunk_id=row[0],
                doc_id=row[1],
                section=row[2],
                chunk_order=int(row[3]),
                text=row[4],
                token_count=int(row[5]) if row[5] is not None else None,
                faiss_row_id=int(row[6]),
                index_version=row[7],
                extra=_json_dict_or_none(row[8]),
            )
            for row in rows
        ]

    def doc_ids_matching(self, filters: dict) -> List[str]:
        """Return doc_ids whose metadata matches every key in ``filters``.

        Only keys in ``_FILTER_COLUMNS`` (``domain`` / ``category`` /
        ``language``) are accepted; anything else raises ``ValueError``
        so a caller can't SELECT against arbitrary columns. Empty or
        None filters return the full doc_id list — the retriever uses
        that for the "no filter" short-circuit without having to
        special-case it here.
        """
        if not filters:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT doc_id FROM ragmeta.documents")
                return [str(row[0]) for row in cur.fetchall()]

        for key in filters:
            if key not in _FILTER_COLUMNS:
                raise ValueError(
                    f"Unsupported filter key {key!r}. Allowed: "
                    f"{sorted(_FILTER_COLUMNS)}."
                )

        # Parameterised WHERE: the column name is whitelisted above so
        # f-string interpolation is safe; the value flows through %s so
        # psycopg2 handles quoting/escape.
        clauses = [f"{key} = %s" for key in filters.keys()]
        params = [filters[key] for key in filters.keys()]
        sql = (
            "SELECT doc_id FROM ragmeta.documents "
            "WHERE " + " AND ".join(clauses)
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return [str(row[0]) for row in cur.fetchall()]

    def stats(self) -> dict:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ragmeta.documents")
            docs = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM ragmeta.chunks")
            chunks = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT index_version, embedding_model, embedding_dim, chunk_count, built_at
                  FROM ragmeta.index_builds
                 ORDER BY built_at DESC LIMIT 1
                """
            )
            latest = cur.fetchone()
        return {
            "documents": docs,
            "chunks": chunks,
            "latest_build": (
                {
                    "index_version": latest[0],
                    "embedding_model": latest[1],
                    "embedding_dim": latest[2],
                    "chunk_count": latest[3],
                    "built_at": latest[4].isoformat() if latest[4] else None,
                }
                if latest else None
            ),
        }


def _json_dict_or_none(value) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None
