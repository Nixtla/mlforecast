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


import narwhals as nw
import numpy as np
import polars as pl
import pytest

from mlforecast._pooled_engine import PooledCtx, build_agg_table, grouped_accumulate

BACKENDS = ["polars", "pandas"]


def _panel(backend, n_buckets=3, n_times=10, n_series_per_bucket=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_buckets):
        for s in range(n_series_per_bucket):
            for t in range(n_times):
                rows.append((f"b{b}_s{s}", t, b, rng.normal(10, 2)))
    df = pl.DataFrame(rows, schema=["unique_id", "ds", "store", "y"], orient="row")
    return df if backend == "polars" else df.to_pandas()


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_cum_sum_resets_per_bucket(backend):
    df = _panel(backend, n_buckets=2, n_times=3, n_series_per_bucket=1)
    out = grouped_accumulate(df, ["store"], ["y"], "cum_sum", ["Ey"])
    o = out if isinstance(out, pl.DataFrame) else pl.from_pandas(out)
    o = o.sort(["store", "ds"])
    for b in (0, 1):
        blk = o.filter(pl.col("store") == b)
        assert blk["Ey"].to_list() == pytest.approx(
            np.cumsum(blk["y"].to_numpy()).tolist()
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_rejects_unknown_op(backend):
    with pytest.raises(ValueError, match="unsupported accumulate op"):
        grouped_accumulate(_panel(backend), ["store"], ["y"], "cum_prod", ["Ey"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_build_agg_table_shape_and_aggregates(backend):
    df = _panel(backend, n_buckets=3, n_times=10, n_series_per_bucket=2)
    tbl = build_agg_table(df, ["store"], "ds", "y", {None})
    o = tbl if isinstance(tbl, pl.DataFrame) else pl.from_pandas(tbl)
    assert o.height == 3 * 10, "one row per (bucket, timestamp)"
    for c in ["store", "ds", "ord", "s", "c", "q", "mn", "mx", "Es", "Ec", "Eq"]:
        assert c in o.columns
    # every timestamp had 2 contributing series
    assert o["c"].to_list() == [2.0] * 30
    # per-bucket prefix sums restart at each bucket
    first_of_bucket = o.sort(["store", "ord"]).group_by("store").first()
    assert first_of_bucket["Es"].to_list() == pytest.approx(
        first_of_bucket["s"].to_list()
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_build_agg_table_emits_no_complex_groupby_warning(backend):
    """A composite expression inside .agg() costs 250x on pandas and is
    invisible on polars. Guard it here, permanently."""
    import warnings

    df = _panel(backend, n_buckets=4, n_times=20)
    # NOTE: pytest.warns(None) was removed in pytest 8; this repo runs pytest 9.
    # catch_warnings(record=True) is the supported way to assert NO warning.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        build_agg_table(df, ["store"], "ds", "y", {None, "sum", "mean"})
    complex_gb = [w for w in record if "complex group-by expression" in str(w.message)]
    assert not complex_gb, (
        "build_agg_table fell back to narwhals' slow pandas group-by path: "
        f"{[str(w.message) for w in complex_gb]}"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_ctx_col_respects_time_agg_suffix(backend):  # noqa: ARG001 (parametrized for uniformity; backend-independent)
    plain = PooledCtx(keys=["store"], lag=1, min_samples=7, time_agg=None)
    agg = PooledCtx(keys=["store"], lag=1, min_samples=7, time_agg="sum")
    assert plain.col("Es") == "Es"
    assert agg.col("Es") == "Es__sum"


@pytest.mark.parametrize("backend", BACKENDS)
def test_ctx_window_matches_manual_rolling_sum(backend):
    df = _panel(backend, n_buckets=2, n_times=12, n_series_per_bucket=1)
    tbl = build_agg_table(df, ["store"], "ds", "y", {None})
    ctx = PooledCtx(keys=["store"], lag=1, min_samples=1, time_agg=None)
    t = nw.from_native(tbl, eager_only=True)
    got = t.with_columns(ctx.window("s", 3).alias("w3")).to_native()
    o = got if isinstance(got, pl.DataFrame) else pl.from_pandas(got)
    o = o.sort(["store", "ord"])
    for b in (0, 1):
        blk = o.filter(pl.col("store") == b)
        s = blk["s"].to_numpy()
        for i in range(len(s)):
            lo, hi = max(i - 1 - 3 + 1, 0), i - 1 + 1  # ordinals (t-lag-w, t-lag]
            want = s[lo:hi].sum() if hi > lo else 0.0
            assert blk["w3"][i] == pytest.approx(want, abs=1e-12), f"bucket {b} row {i}"
