"""FUNSD adapter skeleton.

FUNSD is intentionally not implemented in this first pass. It will be used
only for a small OCR/layout import smoke fixture, with PAGE as the primary
SearchUnit type.
"""

from __future__ import annotations


DATASET_ID = "funsd"
PRIMARY_SEARCH_UNIT_TYPE = "PAGE"


def convert(*_args, **_kwargs) -> None:
    raise NotImplementedError("FUNSD adapter is a skeleton; implement after KoViDoRe is stable.")
