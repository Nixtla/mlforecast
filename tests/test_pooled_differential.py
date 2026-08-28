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
    1)`.

    ONE SERIES PER BUCKET IS REQUIRED: with window_size=1, lag=1, the count
    at ordinal t is the number of observations at ordinal t-1 WITHIN THE
    BUCKET -- i.e. the number of series sharing that bucket. A fixture with
    multiple series per store (e.g. the earlier, broken version of this test:
    n_series=12, n_groups=4, 3 series/bucket) has cnt in {0, 3} and NEVER
    reaches cnt==1, so it cannot exercise the bug at all -- verified by a
    reviewer monkeypatching RollingStd back to the pre-fix code and getting a
    PASS. n_series=n_groups below gives store=i%n_groups exactly one series
    per bucket, so cnt in {0, 1}."""
    import narwhals as nw

    from mlforecast._pooled_engine import PooledCtx, build_agg_table

    n_series = 8
    df = _panel(backend, n_series=n_series, n_times=30, n_groups=n_series)

    # Verify the fixture actually reaches cnt == 1 on the real aggregate
    # table, so this test cannot silently stop biting again if the fixture
    # ever changes.
    tbl = build_agg_table(df, ["store"], "ds", "y", {None})
    ctx = PooledCtx(keys=["store"], lag=1, min_samples=1, time_agg=None)
    cnt_vals = set(
        nw.from_native(tbl, eager_only=True)
        .with_columns(ctx.window("c", 1).alias("cnt"))["cnt"]
        .to_list()
    )
    assert 1.0 in cnt_vals, f"fixture never reaches cnt == 1: {cnt_vals}"

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


def _large_magnitude_panel(n_series=6, n_times=30, mag=1e11, seed=0):
    """Near-constant, LARGE-magnitude target: `sq` and `num**2/cnt` (both
    reconstructed from prefix-sum subtraction) are each huge, so their
    difference is pure floating-point cancellation noise that can round
    negative -- this is reproducible, not a random fluke: 296/300 randomized
    trials (magnitudes 1e6..1e12, 2-8 series/bucket, window 2-7) produced a
    negative residue. A large, near-constant series is an entirely ordinary
    input in practice (counts, populations, revenue), not an exotic corner
    case -- no NaN, no unusual config. Single bucket (all series share
    store=0) so every series' aggregate feeds the same `cnt`/`num`/`sq`,
    matching the coordinator's construction exactly."""
    rng = np.random.default_rng(seed)
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1),
        pl.datetime(2020, 1, 1) + pl.duration(days=n_times - 1),
        interval="1d",
        eager=True,
    )
    y = mag * (1.0 + rng.normal(0, 1e-15, n_series * n_times))
    return pl.DataFrame(
        {
            "unique_id": np.repeat([f"id_{s}" for s in range(n_series)], n_times),
            "ds": np.tile(dates.to_numpy(), n_series),
            "y": y,
            "store": np.zeros(n_series * n_times, dtype=np.int64),
        }
    ).sort("unique_id", "ds")


def _negative_residue_dates(df_pl, window_size, min_samples):
    """The set of `ds` values where the raw (unclamped) variance is negative,
    computed ONCE via a fixed (polars) build of the aggregate table -- this
    is the precondition guard (requirement 2: fail loudly, not vacuously, if
    `_large_magnitude_panel` ever stops reproducing this) AND the reference
    row set used by the tests below.

    IMPORTANT, discovered while building this test (see the fix-round-3
    section of the report for the full evidence): the raw variance's SIGN at
    these near-cancellation magnitudes is not just a property of the input
    data -- it also depends on the summation ORDER used to build `sq`/`num`,
    which genuinely differs between polars' and pandas' native `.sum()`
    (measured: up to an 8388608-ULP difference in the per-timestamp `q`
    aggregate at mag=1e11, i.e. exactly a rounding-unit-scale disagreement,
    NOT a bug in either backend). Because the TRUE mathematical variance
    here is >= 0, the only source of negativity is this same rounding noise,
    so it is mathematically impossible to construct a "deeply negative"
    residue immune to this effect: any residue large enough to be robust to
    the backend's own noise floor is large enough to BE the true, positive
    variance instead (verified empirically over jitter 1e-15..1e-5 -- see
    tmp/scan_jitter2.py in the fix-round-3 investigation). A full-column,
    per-row atol=0.0 comparison between arbitrary summation paths therefore
    does not hold everywhere in this construction. What DOES hold, checked
    directly: at the SPECIFIC dates this function identifies (via one FIXED
    reference computation, reused for both backend parametrizations), both
    the legacy and narwhals engines emit EXACTLY 0.0 on BOTH backends -- see
    the two tests below, which assert exactly that."""
    import narwhals as nw

    from mlforecast._pooled_engine import PooledCtx, build_agg_table

    tbl = build_agg_table(df_pl, ["store"], "ds", "y", {None})
    ctx = PooledCtx(keys=["store"], lag=1, min_samples=min_samples, time_agg=None)
    cnt = ctx.window("c", window_size)
    num = ctx.window("s", window_size)
    sq = ctx.window("q", window_size)
    raw_var = (sq - num * num / cnt) / (cnt - 1)
    o = (
        nw.from_native(tbl, eager_only=True)
        .with_columns(raw_var.alias("raw_var"), cnt.alias("cnt"))
        .to_native()
    )
    o = o if isinstance(o, pl.DataFrame) else pl.from_pandas(o)
    o = o.filter(pl.col("cnt") >= min_samples)
    raw = o["raw_var"].cast(pl.Float64).to_numpy()
    assert len(raw) > 0 and (raw < 0).any(), (
        "precondition failed: _large_magnitude_panel no longer produces a "
        "negative raw variance for this window_size/min_samples -- the "
        "tests below would pass vacuously without this guard. "
        f"min raw_var seen: {raw.min() if len(raw) else 'n/a'}"
    )
    neg = o.filter(pl.col("raw_var") < 0)
    return set(neg["ds"].to_list())


def _assert_both_engines_clip_to_zero_at(df_pl, tfms, neg_dates, backend):
    """At the rows identified by `_negative_residue_dates`, both the legacy
    numpy engine and the narwhals engine must emit EXACTLY 0.0 -- the
    defining property of Finding 3's fix (`np.maximum(var, 0.0)` /
    `.clip(lower_bound=0.0)`, not `.abs()`)."""
    df = df_pl if backend == "polars" else df_pl.to_pandas()
    a = _preprocess_with_engine("numpy", df, tfms, ["store"])
    b = _preprocess_with_engine("narwhals", df, tfms, ["store"])
    a = (a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)).sort(
        "unique_id", "ds"
    )
    b = (b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)).sort(
        "unique_id", "ds"
    )
    feat_cols = [c for c in a.columns if c not in ("unique_id", "ds", "store", "y")]
    assert feat_cols, "no feature columns produced"
    for c in feat_cols:
        a_at = (
            a.filter(pl.col("ds").is_in(list(neg_dates)))[c].cast(pl.Float64).to_numpy()
        )
        b_at = (
            b.filter(pl.col("ds").is_in(list(neg_dates)))[c].cast(pl.Float64).to_numpy()
        )
        assert len(a_at) > 0, "no rows matched the negative-residue dates"
        np.testing.assert_allclose(a_at, 0.0, atol=0.0, err_msg=f"legacy, {c}")
        np.testing.assert_allclose(b_at, 0.0, atol=0.0, err_msg=f"narwhals, {c}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_rolling_std_large_magnitude_cancellation_clips_to_zero(backend):
    """Finding 3 regression, re-classified CRITICAL: near-identical
    LARGE-magnitude values make `sq` and `num**2/cnt` both huge, so their
    difference is pure floating-point cancellation noise that rounds
    negative. legacy clamps with `np.maximum(var, 0.0)`; the pre-fix
    narwhals code used `.abs()`, which turned a large-magnitude negative
    residue into a standard deviation of thousands where the true value is
    exactly 0.0 -- a magnitude error of O(1e3), reachable through the
    ordinary public API with no NaN and no exotic config. See
    `_negative_residue_dates`'s docstring for why the assertion is scoped to
    the identified rows rather than the whole column."""
    df_pl = _large_magnitude_panel()
    neg_dates = _negative_residue_dates(df_pl, window_size=5, min_samples=2)
    tfms = {1: [RollingStd(5, min_samples=2, groupby=["store"])]}
    _assert_both_engines_clip_to_zero_at(df_pl, tfms, neg_dates, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_expanding_std_large_magnitude_cancellation_clips_to_zero(backend):
    """Same construction as the RollingStd case above. Verified separately
    reachable for ExpandingStd: `cnt >= 2` does not prevent the residue from
    going negative -- 7 of 29 eligible rows go negative on this panel (min
    raw variance -3.36e6)."""
    df_pl = _large_magnitude_panel()
    neg_dates = _negative_residue_dates(df_pl, window_size=None, min_samples=2)
    tfms = {1: [ExpandingStd(groupby=["store"])]}
    _assert_both_engines_clip_to_zero_at(df_pl, tfms, neg_dates, backend)


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
