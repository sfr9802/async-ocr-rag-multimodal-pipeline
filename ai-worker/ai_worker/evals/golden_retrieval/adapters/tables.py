"""PubTables/TableBank adapter skeleton.

The first table dataset pass should use only a small sample and emit TABLE
SearchUnits. No full PubTables-1M or TableBank download is automated here.
"""

from __future__ import annotations


DATASET_IDS = ("pubtables-1m-sample", "tablebank-sample")
PRIMARY_SEARCH_UNIT_TYPE = "TABLE"


def convert(*_args, **_kwargs) -> None:
    raise NotImplementedError("Table adapter is a skeleton; implement after KoViDoRe is stable.")
