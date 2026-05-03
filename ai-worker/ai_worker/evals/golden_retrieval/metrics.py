"""Metric helpers for stable-identity golden retrieval evals."""

from __future__ import annotations

import math
from typing import Any

from ai_worker.evals.golden_retrieval.matching import matches_result


def ndcg_at_k(
    results: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    acceptable: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    k: int,
) -> float:
    """Compute graded nDCG using expected/acceptable SearchUnit targets."""
    if k <= 0:
        return 0.0
    grades = [
        relevance_for_result(result, expected, acceptable, manifest=manifest)
        for result in results[:k]
    ]
    ideal_grades = sorted(
        [
            *(_spec_relevance(spec, default=2) for spec in expected),
            *(_spec_relevance(spec, default=1) for spec in acceptable),
        ],
        reverse=True,
    )[:k]
    ideal = _dcg(ideal_grades)
    if ideal <= 0.0:
        return 0.0
    return _dcg(grades) / ideal


def relevance_for_result(
    result: dict[str, Any],
    expected: list[dict[str, Any]],
    acceptable: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> int:
    relevance = 0
    for spec in expected:
        if matches_result(result, spec, manifest):
            relevance = max(relevance, _spec_relevance(spec, default=2))
    for spec in acceptable:
        if matches_result(result, spec, manifest):
            relevance = max(relevance, _spec_relevance(spec, default=1))
    return relevance


def _spec_relevance(spec: dict[str, Any], *, default: int) -> int:
    for key in ("relevanceScore", "relevance", "score"):
        if key in spec:
            try:
                return int(spec[key])
            except (TypeError, ValueError):
                return default
    return default


def _dcg(grades: list[int]) -> float:
    total = 0.0
    for idx, grade in enumerate(grades, start=1):
        if grade <= 0:
            continue
        total += (2.0**grade - 1.0) / math.log2(idx + 1)
    return total
