# tests/test_pooled_differential.py
"""Both engines, identical inputs, identical outputs.

The pooled suite proves the narwhals engine is correct in absolute terms; this
file proves the two engines agree, so any divergence names itself.
"""

import importlib
import os

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingStd,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingStd,
)

BACKENDS = ["polars", "pandas"]


def _panel(backend, n_series=40, n_times=60, n_groups=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1),
        pl.datetime(2020, 1, 1) + pl.duration(days=n_times - 1),
        interval="1d",
        eager=True,
    )
    df = pl.DataFrame(
        {
            "unique_id": np.repeat([f"id_{i}" for i in range(n_series)], n_times),
            "ds": np.tile(dates.to_numpy(), n_series),
            "y": rng.normal(10, 2, n_series * n_times),
            "store": np.repeat([i % n_groups for i in range(n_series)], n_times),
        }
    ).sort("unique_id", "ds")
    return df if backend == "polars" else df.to_pandas()


def _preprocess_with_engine(engine, df, tfms, statics):
    """Reimport the engine module with the env var set, then preprocess."""
    prev = os.environ.get("MLFORECAST_POOLED_ENGINE")
    os.environ["MLFORECAST_POOLED_ENGINE"] = engine
    try:
        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        import mlforecast.core

        importlib.reload(mlforecast.core)
        fcst = MLForecast(models=[LinearRegression()], freq="1d", lag_transforms=tfms)
        return fcst.preprocess(df, static_features=statics, dropna=False)
    finally:
        if prev is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = prev
        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        import mlforecast.core

        importlib.reload(mlforecast.core)


def assert_engines_agree(df, tfms, statics, atol=1e-10):
    a = _preprocess_with_engine("numpy", df, tfms, statics)
    b = _preprocess_with_engine("narwhals", df, tfms, statics)
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    assert sorted(a.columns) == sorted(b.columns), (
        f"column mismatch: numpy-only={set(a.columns) - set(b.columns)}, "
        f"narwhals-only={set(b.columns) - set(a.columns)}"
    )
    feat_cols = [c for c in a.columns if c not in ("unique_id", "ds", "store", "y")]
    assert feat_cols, "no feature columns produced"
    a = a.sort("unique_id", "ds")
    b = b.sort("unique_id", "ds")
    for c in feat_cols:
        av = a[c].cast(pl.Float64).to_numpy()
        bv = b[c].cast(pl.Float64).to_numpy()
        np.testing.assert_allclose(av, bv, atol=atol, equal_nan=True, err_msg=c)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rolling_mean_groupby_engines_agree(backend):
    assert_engines_agree(
        _panel(backend), {1: [RollingMean(7, groupby=["store"])]}, ["store"]
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_rolling_mean_global_engines_agree(backend):
    assert_engines_agree(
        _panel(backend), {1: [RollingMean(7, global_=True)]}, ["store"]
    )


ROLLING_EXPANDING = [
    ("rolling_std", RollingStd(14, groupby=["store"])),
    ("rolling_min", RollingMin(14, groupby=["store"])),
    ("rolling_max", RollingMax(14, groupby=["store"])),
    ("expanding_mean", ExpandingMean(groupby=["store"])),
    ("expanding_std", ExpandingStd(groupby=["store"])),
    ("expanding_min", ExpandingMin(groupby=["store"])),
    ("expanding_max", ExpandingMax(groupby=["store"])),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "label,tfm", ROLLING_EXPANDING, ids=[x[0] for x in ROLLING_EXPANDING]
)
def test_rolling_expanding_engines_agree(backend, label, tfm):  # noqa: ARG001
    assert_engines_agree(_panel(backend), {1: [tfm]}, ["store"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_rolling_std_window1_min_samples1_no_inf(backend):
    """Finding 1 regression: at cnt == 1, `sq - num*num/cnt` is not exactly 0
    -- it carries a cancellation residue from subtracting large prefix sums
    -- so dividing by `(cnt - 1) == 0` under a bare `cnt > 0` guard produced
    +-inf (not NaN) on both backends. `RollingStd(window_size=1,
    min_samples=1)` is a legal, reachable public-API configuration where
    `cnt == 1` is the common case. The pooled expression must mask on
    `cnt > 1`, matching legacy's own `(win_cnt >= min_samples) & (win_cnt >
    1)`."""
    df = _panel(backend, n_series=12, n_times=30, n_groups=4)
    tfms = {1: [RollingStd(1, min_samples=1, groupby=["store"])]}
    b = _preprocess_with_engine("narwhals", df, tfms, ["store"])
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    feat_cols = [c for c in b.columns if c not in ("unique_id", "ds", "store", "y")]
    assert feat_cols, "no feature columns produced"
    for c in feat_cols:
        vals = b[c].cast(pl.Float64).to_numpy()
        assert not np.isinf(vals).any(), f"{c}: found {np.isinf(vals).sum()} infinities"
    assert_engines_agree(df, tfms, ["store"])


def _gap_panel():
    """store 0: 12 days with an INTERIOR gap (both underlying series are NaN
    on day 5, so that bucket-timestamp has zero valid observations). store 1
    starts 3 days later than store 0 -- exercises per-bucket containment of
    the cum_min/cum_max forward-fill (Finding 2) at the same time as the
    interior-gap carry-through. Plain integer ``ds`` and no ``unique_id``
    column: this test drives ``build_agg_table``/``grouped_accumulate``/
    ``PooledCtx`` directly rather than through ``MLForecast.preprocess()``,
    because ``TimeSeries._fit`` unconditionally rejects any NaN in the raw
    target column (``core.py``'s ``if ufp.is_nan_or_none(df[target_col])
    .any(): raise ValueError``) regardless of engine -- so an interior gap in
    the RAW target can never reach either engine through the public
    ``preprocess()`` path today. This mirrors exactly how Finding 2 itself was
    confirmed: against ``build_agg_table`` + ``grouped_accumulate`` +
    ``PooledCtx``, not the top-level API."""
    rng = np.random.default_rng(3)
    rows = []
    for day in range(12):
        for s in ("a", "b"):
            y = float(rng.normal(10, 2))
            if day == 5:
                y = float("nan")
            rows.append((0, day, y))
    for day in range(3, 12):
        for s in ("a", "b"):
            y = float(rng.normal(10, 2))
            rows.append((1, day, y))
    return pl.DataFrame(rows, schema=["store", "ds", "y"], orient="row")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "tfm,legacy_fn_name",
    [
        (ExpandingMin(groupby=["store"]), "_expanding_min_from_agg"),
        (ExpandingMax(groupby=["store"]), "_expanding_max_from_agg"),
    ],
    ids=["min", "max"],
)
def test_expanding_minmax_interior_gap_matches_legacy(backend, tfm, legacy_fn_name):
    """Finding 2 regression: legacy carries the running min/max THROUGH an
    interior all-NaN timestamp (`np.fmin.accumulate`/`np.fmax.accumulate`
    ignore NaN); the narwhals engine must forward-fill `cum_min`/`cum_max`
    per bucket to match. store 0 has an interior gap at day 5; store 1
    starts 3 days later than store 0, so this also covers per-bucket
    containment of the fill. Exercises the shipped `_pooled_expr` and the
    real `build_agg_table`/`grouped_accumulate`/`PooledCtx` path, compared
    directly against the legacy aggregate helper (`_expanding_min_from_agg`/
    `_expanding_max_from_agg`) -- see `_gap_panel`'s docstring for why this
    bypasses `MLForecast.preprocess()`."""
    import narwhals as nw

    from mlforecast import lag_transforms as lt
    from mlforecast._pooled_engine import PooledCtx, build_agg_table, grouped_accumulate
    from mlforecast._pooled_legacy import _build_ts_aggs

    df_pl = _gap_panel()
    src = df_pl if backend == "polars" else df_pl.to_pandas()

    # legacy ground truth, built the same way the differential harness
    # elsewhere in this file builds it (sorted, per-bucket dense ordinals)
    d = df_pl.sort(["store", "ds"])
    bid = d["store"].to_numpy().astype(np.int64)
    y_arr = d["y"].to_numpy().astype(float)
    ts = d["ds"].to_numpy()
    ordv = np.empty(len(ts), dtype=np.int64)
    for b in np.unique(bid):
        m = bid == b
        u = np.unique(ts[m])
        ordv[m] = np.searchsorted(u, ts[m])
    aggs = _build_ts_aggs(bid, ordv, y_arr)
    legacy_fn = getattr(lt, legacy_fn_name)
    lag = 1
    want = {b: legacy_fn(a, lag) for b, a in aggs.items()}

    base, op = tfm._pooled_accumulate
    tbl = build_agg_table(src, ["store"], "ds", "y", {None})
    tbl = grouped_accumulate(tbl, ["store"], [base], op, [f"A{base}"])
    ctx = PooledCtx(keys=["store"], lag=lag, min_samples=1, time_agg=None)
    expr = tfm._pooled_expr(ctx)
    t = nw.from_native(tbl, eager_only=True).with_columns(expr.alias("v"))
    o = t.to_native()
    o = o if isinstance(o, pl.DataFrame) else pl.from_pandas(o)
    o = o.sort(["store", "ord"])
    for b in (0, 1):
        got = (
            o.filter(pl.col("store") == b).sort("ord")["v"].cast(pl.Float64).to_numpy()
        )
        np.testing.assert_allclose(
            got, want[b], atol=1e-9, equal_nan=True, err_msg=f"store {b}"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_target_engines_agree(backend):
    """A float32 target reaches `ga.data.dtype` as float32, and BOTH engines must
    round-trip y through it so float32 rounding matches bit-for-bit. Nothing else
    in the suite or this file exercises a non-float64 target, so without this test
    a dtype regression here is invisible."""
    df = _panel(backend, n_series=12, n_times=40)
    d = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    # large magnitudes make float32 rounding visible
    d = d.with_columns((pl.col("y") * 1000.0).cast(pl.Float32).alias("y"))
    src = d if isinstance(df, pl.DataFrame) else d.to_pandas()
    assert_engines_agree(
        src,
        {1: [RollingMean(4, groupby=["store"]), RollingMean(9, groupby=["store"])]},
        ["store"],
    )
