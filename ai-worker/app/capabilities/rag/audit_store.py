"""Best-effort persistence for retrieval audit traces."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import psycopg2

from app.capabilities.rag.retrieval_contract import citation_payload, preview
from app.capabilities.rag.retriever import RetrievalReport


class RetrievalAuditStore:
    """Writes lightweight RetrievalRun/RetrievalHit traces to ragmeta.

    Callers are expected to wrap this in best-effort error handling so an
    audit table outage never makes the user-facing RAG request fail.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

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

    def record_retrieval(
        self,
        report: RetrievalReport,
        *,
        request_id: str,
        user_id: Optional[str] = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        normalized_query = (
            report.parsed_query.normalized
            if report.parsed_query is not None
            else None
        )
        latency_ms = _latency_ms(report)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ragmeta.retrieval_runs
                  (id, request_id, user_id, query, normalized_query,
                   retriever_version, embedding_model, reranker_model,
                   top_k, filters_json, latency_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                (
                    run_id,
                    request_id,
                    user_id,
                    report.query,
                    normalized_query,
                    report.index_version,
                    report.embedding_model,
                    report.reranker_name,
                    report.top_k,
                    json.dumps(report.filters or {}),
                    latency_ms,
                    now,
                ),
            )

            for rank, chunk in enumerate(report.results, start=1):
                citation = citation_payload(chunk)
                final_score = (
                    chunk.rerank_score
                    if chunk.rerank_score is not None
                    else chunk.score
                )
                rrf_score = (
                    chunk.score
                    if report.parsed_query is not None and report.parsed_query.rewrites
                    else None
                )
                cur.execute(
                    """
                    INSERT INTO ragmeta.retrieval_hits
                      (id, retrieval_run_id, rank, search_unit_id,
                       source_file_id, unit_type, unit_key, dense_score,
                       sparse_score, rrf_score, rerank_score, final_score,
                       citation_json, snippet, selected_for_context,
                       created_at)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       %s::jsonb, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        run_id,
                        rank,
                        chunk.search_unit_id,
                        chunk.source_file_id or chunk.doc_id,
                        chunk.unit_type or "CHUNK",
                        chunk.unit_key,
                        chunk.dense_score if chunk.dense_score is not None else chunk.score,
                        chunk.sparse_score,
                        rrf_score,
                        chunk.rerank_score,
                        final_score,
                        json.dumps(citation),
                        preview(chunk.text),
                        True,
                        now,
                    ),
                )
        return run_id


def _latency_ms(report: RetrievalReport) -> Optional[int]:
    values = [
        value
        for value in (report.dense_retrieval_ms, report.rerank_ms)
        if value is not None
    ]
    if not values:
        return None
    return int(round(sum(values)))
