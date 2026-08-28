# tests/test_pooled_narwhals.py
"""Unit tests for the narwhals pooled engine internals."""

import mlforecast.pooled as mp
from mlforecast._pooled_keys import _NULL_SENTINEL, add_bucket_id


def test_engine_constant_is_valid():
    assert mp.POOLED_ENGINE in ("narwhals", "numpy")


def test_shared_key_helpers_importable_from_both_engines():
    from mlforecast import _pooled_legacy

    assert _pooled_legacy.add_bucket_id is add_bucket_id
    assert _NULL_SENTINEL == "\x00__MLF_NULL__"
