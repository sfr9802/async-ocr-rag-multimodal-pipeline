"""Tests for the pure-Python helpers in
``app.capabilities.rag.embeddings``.

Scope is intentionally limited to the helpers that don't touch
``torch.cuda``:

  - ``resolve_max_seq_length`` — translates the
    ``rag_embedding_max_seq_length`` setting into a concrete
    ``SentenceTransformerEmbedder`` argument.
  - ``apply_cuda_alloc_conf`` — sets ``PYTORCH_CUDA_ALLOC_CONF`` in
    ``os.environ``; respects an operator-set value.

The CUDA-touching helpers (``cuda_memory_stats``,
``reset_cuda_peak_stats``, ``_is_cuda_oom_exception``) and the OOM
fallback inside ``SentenceTransformerEmbedder._embed`` are exercised
end-to-end by the retrieval CLI and the live worker. Putting unit
tests around them in this module triggered intermittent CUDA-driver
init in the same pytest process and segfaulted later torch-using
tests (``test_rag_capability``). The Phase 1C log lines + manifest
output are the production verification path; if a regression is
caught later, add a pytest-forked variant rather than re-introducing
the in-process test that destabilised the suite.
"""

from __future__ import annotations

import os

import pytest

from app.capabilities.rag.embeddings import (
    apply_cuda_alloc_conf,
    resolve_max_seq_length,
)


# ``apply_cuda_alloc_conf`` writes directly to ``os.environ``. Without
# isolation, leaving ``PYTORCH_CUDA_ALLOC_CONF`` set across tests can
# influence later tests that import torch / sentence_transformers and
# touch CUDA. Snapshot + restore the variable around every test in
# this module.
@pytest.fixture(autouse=True)
def _alloc_conf_isolation():
    original = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = original


# ---------------------------------------------------------------------------
# resolve_max_seq_length
# ---------------------------------------------------------------------------


def test_resolves_default_1024():
    assert resolve_max_seq_length(1024) == 1024


def test_zero_is_no_cap():
    """0 is the documented escape hatch — use the model's own default."""
    assert resolve_max_seq_length(0) is None


def test_negative_is_no_cap():
    assert resolve_max_seq_length(-1) is None
    assert resolve_max_seq_length(-512) is None


def test_none_is_no_cap():
    assert resolve_max_seq_length(None) is None


def test_passes_through_positive_int():
    assert resolve_max_seq_length(512) == 512
    assert resolve_max_seq_length(2048) == 2048
    assert resolve_max_seq_length(8192) == 8192


def test_string_int_is_coerced():
    """Pydantic gives us int, but defending against env-string injection
    keeps the helper robust if the setting is ever read raw."""
    assert resolve_max_seq_length("1024") == 1024


def test_invalid_string_is_no_cap():
    assert resolve_max_seq_length("not-a-number") is None


# ---------------------------------------------------------------------------
# apply_cuda_alloc_conf
# ---------------------------------------------------------------------------


def test_alloc_conf_none_is_noop():
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    assert apply_cuda_alloc_conf(None) is None
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_alloc_conf_empty_is_noop():
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    assert apply_cuda_alloc_conf("") is None
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_alloc_conf_sets_when_unset():
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    out = apply_cuda_alloc_conf("expandable_segments:True")
    assert out == "expandable_segments:True"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_alloc_conf_existing_value_wins():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    out = apply_cuda_alloc_conf("expandable_segments:True")
    # Operator-set value should win; helper returns whatever is in effect.
    assert out == "max_split_size_mb:256"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:256"


def test_alloc_conf_idempotent_when_same():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    out = apply_cuda_alloc_conf("expandable_segments:True")
    assert out == "expandable_segments:True"
