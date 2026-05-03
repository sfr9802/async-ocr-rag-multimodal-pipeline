"""Shared adapter contracts for public golden retrieval datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterSummary:
    dataset_id: str
    out_dir: Path
    source_files: int
    search_units: int
    golden_queries: int
