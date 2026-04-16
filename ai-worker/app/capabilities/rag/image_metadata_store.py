"""Postgres DAO for ragmeta.image_chunks.

Same procedural psycopg2 pattern as metadata_store.py. The table is
created by the core-api Flyway migration V3; this module only reads and
writes rows.
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
class ImageChunkRow:
    image_id: str
    doc_id: str
    page_number: Optional[int]
    source_uri: Optional[str]
    sha256: Optional[str]
    caption: Optional[str]
    section_hint: Optional[str]
    faiss_row_id: int
    index_version: str
    extra: Optional[dict] = None


@dataclass(frozen=True)
class ImageLookupResult:
    image_id: str
    doc_id: str
    page_number: Optional[int]
    caption: Optional[str]
    section_hint: Optional[str]
    faiss_row_id: int


class ImageMetadataStore:
    """Connection factory + DAO for ragmeta.image_chunks."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def ping(self) -> None:
        """Fail fast if the image_chunks table is missing."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'ragmeta' AND table_name = 'image_chunks'"
            )
            if cur.fetchone() is None:
                raise RuntimeError(
                    "ragmeta.image_chunks table is missing. "
                    "Has the core-api Flyway V3 migration run?"
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
        images: List[ImageChunkRow],
        index_version: str,
        embedding_model: str,
        embedding_dim: int,
        faiss_index_path: str,
        notes: Optional[str] = None,
    ) -> None:
        """Atomically replace all image chunk rows with a fresh build."""
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM ragmeta.image_chunks")

            if images:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO ragmeta.image_chunks
                      (image_id, doc_id, page_number, source_uri, sha256,
                       caption, section_hint, faiss_row_id, index_version,
                       extra_json, created_at)
                    VALUES %s
                    """,
                    [
                        (
                            img.image_id,
                            img.doc_id,
                            img.page_number,
                            img.source_uri,
                            img.sha256,
                            img.caption,
                            img.section_hint,
                            img.faiss_row_id,
                            img.index_version,
                            json.dumps(img.extra) if img.extra else None,
                            now,
                        )
                        for img in images
                    ],
                )
        log.info(
            "ragmeta.image_chunks replaced: %d images, version=%s",
            len(images), index_version,
        )

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def lookup_by_faiss_rows(
        self, index_version: str, faiss_row_ids: Iterable[int]
    ) -> List[ImageLookupResult]:
        ids = list(faiss_row_ids)
        if not ids:
            return []
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT image_id, doc_id, page_number, caption,
                       section_hint, faiss_row_id
                  FROM ragmeta.image_chunks
                 WHERE index_version = %s
                   AND faiss_row_id = ANY(%s)
                """,
                (index_version, ids),
            )
            rows = cur.fetchall()
        by_row = {
            r[5]: ImageLookupResult(
                image_id=r[0],
                doc_id=r[1],
                page_number=r[2],
                caption=r[3],
                section_hint=r[4],
                faiss_row_id=r[5],
            )
            for r in rows
        }
        return [by_row[i] for i in ids if i in by_row]
