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
def test_nan_targets_do_not_poison_prefix_sums(backend):
    """NaN is not null: an unguarded sum() would make one NaN target turn that
    bucket's ENTIRE prefix sum into NaN, and count() would count it as present.
    The legacy engine treats NaN as missing; so must we."""
    df = _panel(backend, n_buckets=2, n_times=5, n_series_per_bucket=2)
    o = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    # Poison exactly one raw row: series "b0_s0" at ds=0. There are 2 series
    # per bucket in this fixture, so (store=0, ds=0) has 2 contributing rows;
    # poisoning only one of them (not both, via unique_id) leaves 1 surviving
    # observation, which is what the assertions below check against.
    o = o.with_columns(
        pl.when((pl.col("unique_id") == "b0_s0") & (pl.col("ds") == 0))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    ).with_columns(pl.col("y").fill_null(float("nan")))
    src = o if isinstance(df, pl.DataFrame) else o.to_pandas()

    tbl = build_agg_table(src, ["store"], "ds", "y", {None})
    t = tbl if isinstance(tbl, pl.DataFrame) else pl.from_pandas(tbl)
    t = t.sort(["store", "ord"])
    b0 = t.filter(pl.col("store") == 0)
    assert b0["c"][0] == 1.0, "the NaN row must not be counted as an observation"
    es = b0["Es"].cast(pl.Float64).to_numpy()
    assert not np.isnan(es).any(), f"NaN leaked into the prefix sum: {es}"
    assert b0["s"][0] == pytest.approx(
        o.filter((pl.col("store") == 0) & (pl.col("ds") == 0) & ~pl.col("y").is_nan())[
            "y"
        ].sum()
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_rejects_empty_keys(backend):
    with pytest.raises(ValueError, match="non-empty"):
        grouped_accumulate(_panel(backend), [], ["y"], "cum_sum", ["Ey"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_ewm_mean_requires_explicit_adjust(backend):
    with pytest.raises(ValueError, match="adjust"):
        grouped_accumulate(
            _panel(backend, n_buckets=1, n_times=5, n_series_per_bucket=1),
            ["store"],
            ["y"],
            "ewm_mean",
            ["Ey"],
            alpha=0.5,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_ewm_mean_requires_explicit_alpha(backend):
    with pytest.raises(ValueError, match="alpha"):
        grouped_accumulate(
            _panel(backend, n_buckets=1, n_times=5, n_series_per_bucket=1),
            ["store"],
            ["y"],
            "ewm_mean",
            ["Ey"],
            adjust=False,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_grouped_accumulate_ewm_mean_requires_explicit_ignore_nulls(backend):
    with pytest.raises(ValueError, match="ignore_nulls"):
        grouped_accumulate(
            _panel(backend, n_buckets=1, n_times=5, n_series_per_bucket=1),
            ["store"],
            ["y"],
            "ewm_mean",
            ["Ey"],
            alpha=0.5,
            adjust=False,
        )


def test_grouped_accumulate_ewm_mean_backends_agree_with_explicit_adjust():
    """With alpha, adjust, and ignore_nulls all explicit, polars and pandas
    must agree -- otherwise the shim silently reintroduces the divergence it
    exists to prevent (polars' native ewm_mean defaults adjust=True and
    ignore_nulls=False; the two backends' unadjusted-vs-adjusted recursions,
    and their null-handling, diverge without explicit values)."""
    polars_df = _panel(backend="polars", n_buckets=1, n_times=6, n_series_per_bucket=1)
    pandas_df = polars_df.to_pandas()

    pl_out = grouped_accumulate(
        polars_df,
        ["store"],
        ["y"],
        "ewm_mean",
        ["Ey"],
        alpha=0.5,
        adjust=False,
        ignore_nulls=True,
    )
    pd_out = grouped_accumulate(
        pandas_df,
        ["store"],
        ["y"],
        "ewm_mean",
        ["Ey"],
        alpha=0.5,
        adjust=False,
        ignore_nulls=True,
    )
    pl_vals = pl_out.sort("ds")["Ey"].to_list()
    pd_vals = pl.from_pandas(pd_out).sort("ds")["Ey"].to_list()
    assert pl_vals == pytest.approx(pd_vals)


def test_grouped_accumulate_ewm_mean_forward_fills_gaps_per_bucket():
    """Legacy `_ewm_from_agg` assigns the running EWM at EVERY ordinal,
    updating only where an observation exists, so gap ordinals carry the
    previous value forward. polars' raw ewm_mean leaves those rows null;
    this must be forward-filled WITHIN each bucket (never bleeding across
    buckets) to match. Two buckets with interior gaps, offset so a naive
    frame-level (rather than per-bucket) ffill would leak bucket 0's last
    value into bucket 1's leading gap and fail this test."""
    # bucket 0: [1.0, None, 3.0, 4.0, None, 6.0] -> legacy fold
    #   [1.0, 1.0, 2.0, 3.0, 3.0, 4.5]
    # bucket 1: [None, 10.0, None, None, 20.0, None] -- starts with a gap, so
    # a cross-bucket ffill would wrongly carry bucket 0's final value (4.5)
    # into this bucket's first row instead of leaving it null (no observation
    # yet in this bucket).
    rows = []
    b0_y = [1.0, None, 3.0, 4.0, None, 6.0]
    b1_y = [None, 10.0, None, None, 20.0, None]
    for t, y in enumerate(b0_y):
        rows.append(("s0", t, 0, y))
    for t, y in enumerate(b1_y):
        rows.append(("s1", t, 1, y))
    df = pl.DataFrame(rows, schema=["unique_id", "ds", "store", "y"], orient="row")

    want_b0 = [1.0, 1.0, 2.0, 3.0, 3.0, 4.5]
    want_b1 = [None, 10.0, 10.0, 10.0, 15.0, 15.0]

    for backend in BACKENDS:
        src = df if backend == "polars" else df.to_pandas()
        out = grouped_accumulate(
            src,
            ["store"],
            ["y"],
            "ewm_mean",
            ["Ey"],
            alpha=0.5,
            adjust=False,
            ignore_nulls=True,
        )
        o = out if isinstance(out, pl.DataFrame) else pl.from_pandas(out)
        o = o.sort(["store", "ds"])
        got_b0 = o.filter(pl.col("store") == 0)["Ey"].to_list()
        got_b1 = o.filter(pl.col("store") == 1)["Ey"].to_list()
        assert got_b0 == pytest.approx(want_b0), f"{backend}: bucket 0 mismatch"
        assert got_b1[0] is None, (
            f"{backend}: bucket 1 must not inherit bucket 0's tail"
        )
        assert got_b1[1:] == pytest.approx(want_b1[1:]), f"{backend}: bucket 1 mismatch"


def test_grouped_accumulate_cum_min_forward_fills_gaps_per_bucket():
    """Legacy `_expanding_min_from_agg` is `np.fmin.accumulate(agg.mins)`;
    `np.fmin` IGNORES NaN, so the running minimum carries THROUGH a gap
    ordinal. polars' `cum_min` and pandas' `cummin` instead leave that exact
    position null, so `grouped_accumulate(..., "cum_min", ...)` must
    forward-fill to match -- same defect class as the ewm_mean case above.
    Two buckets, the second STARTING with a gap, so a frame-level (rather
    than per-bucket) ffill would wrongly carry bucket 0's trailing minimum
    into bucket 1's leading gap instead of leaving it null (no observation
    yet in that bucket)."""
    # bucket 0: [5.0, None, 2.0, None, 8.0, 1.0] -> running min
    #   [5.0, 5.0, 2.0, 2.0, 2.0, 1.0]
    # bucket 1: [None, 10.0, None, None, 3.0, None] -- starts with a gap, so
    # bucket 1's first row must stay null (no observation yet in THIS
    # bucket), not inherit bucket 0's final running min of 1.0.
    rows = []
    b0_y = [5.0, None, 2.0, None, 8.0, 1.0]
    b1_y = [None, 10.0, None, None, 3.0, None]
    for t, y in enumerate(b0_y):
        rows.append(("s0", t, 0, y))
    for t, y in enumerate(b1_y):
        rows.append(("s1", t, 1, y))
    df = pl.DataFrame(rows, schema=["unique_id", "ds", "store", "y"], orient="row")

    want_b0 = [5.0, 5.0, 2.0, 2.0, 2.0, 1.0]
    want_b1 = [None, 10.0, 10.0, 10.0, 3.0, 3.0]

    for backend in BACKENDS:
        src = df if backend == "polars" else df.to_pandas()
        out = grouped_accumulate(src, ["store"], ["y"], "cum_min", ["Amn"])
        o = out if isinstance(out, pl.DataFrame) else pl.from_pandas(out)
        o = o.sort(["store", "ds"])
        got_b0 = o.filter(pl.col("store") == 0)["Amn"].to_list()
        got_b1 = o.filter(pl.col("store") == 1)["Amn"].to_list()
        assert got_b0 == pytest.approx(want_b0), f"{backend}: bucket 0 mismatch"
        assert got_b1[0] is None, (
            f"{backend}: bucket 1 must not inherit bucket 0's tail minimum"
        )
        assert got_b1[1:] == pytest.approx(want_b1[1:]), f"{backend}: bucket 1 mismatch"


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
