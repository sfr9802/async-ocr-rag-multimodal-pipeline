"""Stable identity matching helpers for golden retrieval evals."""

from __future__ import annotations

from typing import Any

from ai_worker.evals.golden_retrieval.run import result_matches_spec


def matches_result(
    result: dict[str, Any],
    spec: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> bool:
    return result_matches_spec(result, spec, manifest or {})
