# tests/test_pooled_differential.py
"""Both engines, identical inputs, identical outputs.

The pooled suite proves the narwhals engine is correct in absolute terms; this
file proves the two engines agree, so any divergence names itself.
"""

import importlib
import operator
import os

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    Combine,
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingQuantile,
    ExpandingStd,
    ExponentiallyWeightedMean,
    LookupLag,
    Offset,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingQuantile,
    RollingStd,
    SeasonalRollingMax,
    SeasonalRollingMean,
    SeasonalRollingMin,
    SeasonalRollingQuantile,
    SeasonalRollingStd,
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


SEASONAL = [
    (
        "seas_mean",
        SeasonalRollingMean(season_length=7, window_size=4, groupby=["store"]),
    ),
    ("seas_std", SeasonalRollingStd(season_length=7, window_size=4, groupby=["store"])),
    ("seas_min", SeasonalRollingMin(season_length=7, window_size=4, groupby=["store"])),
    ("seas_max", SeasonalRollingMax(season_length=7, window_size=4, groupby=["store"])),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("label,tfm", SEASONAL, ids=[x[0] for x in SEASONAL])
def test_seasonal_engines_agree(backend, label, tfm):  # noqa: ARG001
    assert_engines_agree(_panel(backend, n_times=90), {1: [tfm]}, ["store"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_seasonal_rolling_std_window1_min_samples1_no_inf(backend):
    """Finding 1's defect (RollingStd), reproduced for the seasonal family:
    with ``window_size=1`` the seasonal count reduces to a single shifted
    per-ordinal count (``_pooled_count`` sums ``sum_horizontal`` over exactly
    one offset, ``lag``), so a lone series per bucket reaches ``cnt == 1``
    just as reliably as ``RollingStd(window_size=1)`` does. At ``cnt == 1``
    the numerator is not exactly 0 -- it carries a cancellation residue from
    subtracting shifted sums -- so dividing by ``(cnt - 1) == 0`` under a
    bare ``cnt > 0`` guard gives +-inf, not NaN. The mask must be
    ``cnt > 1``, matching ``_seasonal_stat``'s own ``len(vals) > 1`` guard.

    ONE SERIES PER BUCKET IS REQUIRED, exactly as in
    ``test_rolling_std_window1_min_samples1_no_inf``: ``n_series=n_groups``
    gives ``store=i%n_groups`` exactly one series per bucket, so
    ``cnt in {0, 1}``."""
    import narwhals as nw

    from mlforecast._pooled_engine import PooledCtx, build_agg_table

    n_series = 8
    df = _panel(backend, n_series=n_series, n_times=30, n_groups=n_series)

    tfm = SeasonalRollingStd(
        season_length=3, window_size=1, min_samples=1, groupby=["store"]
    )

    # Verify the fixture actually reaches cnt == 1 on the real aggregate
    # table, so this test cannot silently stop biting again if the fixture
    # ever changes.
    tbl = build_agg_table(df, ["store"], "ds", "y", {None})
    ctx = PooledCtx(keys=["store"], lag=1, min_samples=1, time_agg=None)
    cnt_vals = set(
        nw.from_native(tbl, eager_only=True)
        .with_columns(tfm._pooled_count(ctx).alias("cnt"))["cnt"]
        .to_list()
    )
    assert 1.0 in cnt_vals, f"fixture never reaches cnt == 1: {cnt_vals}"

    tfms = {1: [tfm]}
    b = _preprocess_with_engine("narwhals", df, tfms, ["store"])
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    feat_cols = [c for c in b.columns if c not in ("unique_id", "ds", "store", "y")]
    assert feat_cols, "no feature columns produced"
    for c in feat_cols:
        vals = b[c].cast(pl.Float64).to_numpy()
        assert not np.isinf(vals).any(), f"{c}: found {np.isinf(vals).sum()} infinities"
    assert_engines_agree(df, tfms, ["store"])


def _seasonal_negative_residue_dates(df_pl, tfm, min_samples):
    """Same precondition-and-reference role as ``_negative_residue_dates``,
    but built from the seasonal family's own ``sum_horizontal``-based
    ``cnt``/``num``/``sq`` (offsets ``lag + k*season_length``), which is a
    different summation path than ``RollingStd``'s prefix-sum window
    subtraction and so must be checked independently rather than assumed to
    share ``_negative_residue_dates``'s result set."""
    import narwhals as nw

    from mlforecast._pooled_engine import PooledCtx, build_agg_table

    tbl = build_agg_table(df_pl, ["store"], "ds", "y", {None})
    ctx = PooledCtx(keys=["store"], lag=1, min_samples=min_samples, time_agg=None)
    offs = tfm._pooled_offsets(ctx)
    cnt = tfm._pooled_count(ctx)
    num = nw.sum_horizontal(*[ctx.shift("s", o).alias(f"_ss{o}") for o in offs])
    sq = nw.sum_horizontal(*[ctx.shift("q", o).alias(f"_sq{o}") for o in offs])
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
        "precondition failed: this construction no longer produces a "
        "negative raw variance for SeasonalRollingStd -- the test below "
        "would pass vacuously without this guard. "
        f"min raw_var seen: {raw.min() if len(raw) else 'n/a'}"
    )
    neg = o.filter(pl.col("raw_var") < 0)
    return set(neg["ds"].to_list())


def _assert_narwhals_clips_to_zero_at(df_pl, tfms, neg_dates, backend):
    """At the rows identified by ``_seasonal_negative_residue_dates``, the
    narwhals engine must emit EXACTLY 0.0 -- the defining property of
    Finding 3's fix (``.clip(lower_bound=0.0)``, not ``.abs()``).

    This deliberately checks ONLY the narwhals side, unlike
    ``_assert_both_engines_clip_to_zero_at`` (used by RollingStd/
    ExpandingStd's analogous test) which checks both. That is not a
    weakened test -- it is the correct one, for a reason specific to this
    family: RollingStd/ExpandingStd's LEGACY path shares the exact same
    sum-of-squares aggregate formula as narwhals (see
    ``_rolling_std_from_agg``), so legacy ALSO clips to 0 at these dates and
    the two-sided check is meaningful. SeasonalRollingStd's legacy path has
    NO aggregate fast path at all (see the class docstring) -- it computes
    ``np.std(vals, ddof=1)`` directly on raw values, a numerically STABLE
    two-pass computation. At this magnitude (near-constant data around
    1e11), legacy therefore reports the TRUE small nonzero std (order 1e-4),
    not 0. Verified by measurement (see the task-5 report): every valid row
    in this panel diverges between engines by more than 1e-6, up to 1863 in
    absolute terms, because the sum/sum_sq aggregates lose the entire true
    signal (~1e-8 in the sum-of-squared-deviations term) to floating-point
    noise (~1e7 at this magnitude) long before either engine's `.clip`
    guard runs. No choice of clip/abs recovers that signal -- this is
    catastrophic cancellation, not a bug -- so asserting full engine
    agreement here would be false, not just strict. What IS true and
    revert-proof is that the fix in this file behaves as designed: even
    though the narwhals computation itself is numerically ruined at this
    magnitude, it still clips its (meaningless) negative residue to exactly
    0.0 rather than reporting some wildly-wrong large value via `.abs()`."""
    df = df_pl if backend == "polars" else df_pl.to_pandas()
    b = _preprocess_with_engine("narwhals", df, tfms, ["store"])
    b = (b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)).sort(
        "unique_id", "ds"
    )
    feat_cols = [c for c in b.columns if c not in ("unique_id", "ds", "store", "y")]
    assert feat_cols, "no feature columns produced"
    for c in feat_cols:
        b_at = (
            b.filter(pl.col("ds").is_in(list(neg_dates)))[c].cast(pl.Float64).to_numpy()
        )
        assert len(b_at) > 0, "no rows matched the negative-residue dates"
        np.testing.assert_allclose(b_at, 0.0, atol=0.0, err_msg=f"narwhals, {c}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_seasonal_rolling_std_large_magnitude_cancellation_clips_to_zero(backend):
    """Finding 3's defect (RollingStd/ExpandingStd large-magnitude
    cancellation), reproduced for SeasonalRollingStd: ``season_length=1``
    makes the seasonal offsets ``lag, lag+1, ..., lag+window_size-1`` -- the
    same window ``RollingStd(5, ...)`` would use -- so the identical
    near-constant, large-magnitude construction (``mag=1e11``, 6 series in
    one bucket, see ``_large_magnitude_panel``) reaches a negative raw
    variance residue through the seasonal family's own
    ``sum_horizontal``-based ``cnt``/``num``/``sq``, independently confirmed
    by ``_seasonal_negative_residue_dates`` rather than assumed from the
    rolling case.

    NOTE (disclosed finding, see the task-5 report for the full writeup):
    this test does NOT assert full engine agreement, because at this
    magnitude engine agreement genuinely does not hold for
    SeasonalRollingStd -- unlike RollingStd/ExpandingStd, whose legacy path
    shares the same unstable aggregate formula, SeasonalRollingStd's legacy
    path is numerically stable (direct ``np.std`` on raw values) and reports
    the true small nonzero std, while the narwhals aggregate formula loses
    that signal entirely to floating-point noise. This is an inherent
    limitation of computing variance from ``sum``/``sum_sq`` aggregates at
    extreme magnitude / near-constant data, not fixable by any clip/abs
    choice. What this test DOES verify, and what IS revert-proof: the
    ``.clip(lower_bound=0.0)`` fix still behaves correctly here, so the
    already-unreliable narwhals computation degrades to exactly 0.0 rather
    than to some wildly-wrong large value via ``.abs()``."""
    df_pl = _large_magnitude_panel()
    tfm = SeasonalRollingStd(
        season_length=1, window_size=5, min_samples=2, groupby=["store"]
    )
    neg_dates = _seasonal_negative_residue_dates(df_pl, tfm, min_samples=2)
    tfms = {1: [tfm]}
    _assert_narwhals_clips_to_zero_at(df_pl, tfms, neg_dates, backend)


MISC = [
    ("ewm", ExponentiallyWeightedMean(alpha=0.3, groupby=["store"])),
    ("offset", Offset(RollingMean(7, groupby=["store"]), n=2)),
    (
        "combine",
        Combine(
            RollingMean(7, groupby=["store"]),
            RollingMean(14, groupby=["store"]),
            operator.truediv,
        ),
    ),
]

TIME_AGGS = [
    ("ta_sum", RollingMean(7, groupby=["store"], time_agg="sum")),
    ("ta_mean", RollingMean(7, groupby=["store"], time_agg="mean")),
    ("ta_count", RollingMean(7, groupby=["store"], time_agg="count")),
    ("ta_min", RollingMean(7, groupby=["store"], time_agg="min")),
    ("ta_max", RollingMean(7, groupby=["store"], time_agg="max")),
    ("ta_std", RollingStd(14, groupby=["store"], time_agg="sum")),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("label,tfm", MISC, ids=[x[0] for x in MISC])
def test_misc_transforms_engines_agree(backend, label, tfm):  # noqa: ARG001
    assert_engines_agree(_panel(backend), {1: [tfm]}, ["store"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("label,tfm", TIME_AGGS, ids=[x[0] for x in TIME_AGGS])
def test_time_agg_engines_agree(backend, label, tfm):  # noqa: ARG001
    assert_engines_agree(_panel(backend), {1: [tfm]}, ["store"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_time_aggs_in_one_state_agree(backend):
    """One state, several time_aggs -> several suffixed column families."""
    assert_engines_agree(
        _panel(backend),
        {
            1: [
                RollingMean(7, groupby=["store"]),
                RollingMean(7, groupby=["store"], time_agg="sum"),
                RollingMean(7, groupby=["store"], time_agg="mean"),
            ]
        },
        ["store"],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_lookup_lag_pooled_expr_is_positional_on_sparse_table(backend):
    """``LookupLag`` has no narwhals state-building path yet: partition_by
    states always build a legacy ``PooledState`` regardless of
    ``MLFORECAST_POOLED_ENGINE`` (``core.py`` builds every partition-mode
    state via the hardcoded ``PooledState.from_partition``, not the
    engine-dispatched ``_pooled_state_cls()`` -- Task 8 adds a narwhals
    ``from_partition``). So ``assert_engines_agree`` through the full
    ``MLForecast`` pipeline would only ever compare legacy against legacy and
    never invoke ``LookupLag._pooled_expr`` at all -- it would pass whether or
    not the expression exists, which is not evidence of anything.

    Exercise the expression directly against a hand-built SPARSE
    (bucket, timestamp) table instead -- one row per *observed* timestamp,
    with real gaps in the calendar -- and confirm the lag counts occurrences
    (row position within the bucket), not calendar ordinals: shifting by
    ``lag`` skips the missing ordinals entirely, exactly matching legacy
    ``_compute_latest_from_aggs``'s positional ``agg.sums[-lag]`` indexing.
    """
    import narwhals as nw

    from mlforecast._pooled_engine import PooledCtx

    # bucket "a": 4 observed rows (real ordinals 0, 1, 3, 6 -- gaps at 2, 4, 5,
    # but the table only ever stores observed rows, so those gaps are simply
    # absent, not nulled).
    # bucket "b": 2 observed rows.
    data = {
        "bucket": ["a", "a", "a", "a", "b", "b"],
        "s": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0],
        "c": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    df = pl.DataFrame(data) if backend == "polars" else pl.DataFrame(data).to_pandas()

    tfm = LookupLag(partition_by=["dummy"])
    ctx = PooledCtx(keys=["bucket"], lag=2, min_samples=1)
    t = nw.from_native(df, eager_only=True)
    expr = tfm._pooled_expr(ctx)
    assert expr is not None, "LookupLag must implement _pooled_expr"
    out = t.with_columns(expr.alias("feat")).to_native()
    out_pl = out if isinstance(out, pl.DataFrame) else pl.from_pandas(out)
    got = out_pl["feat"].to_numpy()

    # Row i's feature is row (i - lag)'s value, purely by position within the
    # bucket -- 2 occurrences back, not 2 calendar steps back.
    expected = np.array([np.nan, np.nan, 10.0, 20.0, np.nan, np.nan])
    np.testing.assert_allclose(got, expected, equal_nan=True)


GLOBAL_ACCUMULATE = [
    ("expanding_min_global", ExpandingMin(global_=True)),
    ("expanding_max_global", ExpandingMax(global_=True)),
    ("ewm_global", ExponentiallyWeightedMean(alpha=0.4, global_=True)),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "label,tfm", GLOBAL_ACCUMULATE, ids=[x[0] for x in GLOBAL_ACCUMULATE]
)
def test_global_mode_accumulate_engines_agree(backend, label, tfm):  # noqa: ARG001
    """The `global_=True` (no ``keys``) branch of ``ensure_accumulates`` --
    the one this task's fix touched. Before the fix it called the
    accumulate op with NO kwargs at all (silently using backend-native
    ``ewm_mean`` defaults instead of the required
    ``alpha``/``adjust``/``ignore_nulls``); the first attempted fix then
    called a `.forward_fill()` that doesn't exist on a narwhals `Expr`
    (`AttributeError`). Nothing else in this suite exercises this branch:
    every other case uses `groupby=["store"]`, which goes through
    `grouped_accumulate` instead. Passing `statics=[]` (no static features)
    is required for `global_=True` to fit at all.
    """
    assert_engines_agree(_panel(backend), {1: [tfm]}, [])


def test_ewm_positional_shift_guard_fires_for_gapped_partition_lag_gt_1():
    """Runtime backstop for the defect in
    ``ExponentiallyWeightedMean._pooled_expr``: it shifts by row POSITION,
    which only matches legacy's calendar-ORDINAL threshold fold when the
    bucket is dense (``global_``/``groupby`` modes, which legacy renumbers to
    ``0..n-1``) or when ``lag == 1``. A ``partition_by`` state keeps real,
    non-renumbered parent-calendar ordinals and can be gapped -- measured:
    at ``lag=2`` on a gapped partition bucket this expression returns
    11.0/31.0 where legacy returns 16.0/36.0, a silent wrong number on both
    backends.

    No ``partition_by`` state is ever built through ``NarwhalsPooledState``
    today (``core.py`` routes every ``partition_by`` key through the legacy
    ``PooledState.from_partition`` regardless of engine), so this can't be
    reached through the real ``MLForecast`` pipeline yet -- construct a
    minimal state directly, with ``mode="nonlocal"`` (what a future
    ``partition_by`` wiring would use), and confirm
    ``NarwhalsPooledState.feature_frame`` refuses rather than silently
    computing a wrong number. Also confirm the two cases that must NOT
    raise: ``lag == 1`` even in a non-dense mode, and a known-dense mode
    (``groupby``) even at ``lag > 1``.
    """
    import numpy as np

    from mlforecast.pooled import NarwhalsPooledState

    ewm_lag2 = ExponentiallyWeightedMean(alpha=0.5)._set_core_tfm(2)
    ewm_lag1 = ExponentiallyWeightedMean(alpha=0.5)._set_core_tfm(1)

    df = pl.DataFrame({"ds": [0, 1, 2], "y": [1.0, 2.0, 3.0]})
    state = NarwhalsPooledState(
        agg=None,
        groups=None,
        group_cols=None,
        series_bucket_id=np.zeros(1, dtype=np.int64),
        join_cols=["unique_id", "ds"],
        keys=[],
        time_col="ds",
        mode="nonlocal",
    )._build(df, "ds", "y", np.float64)

    with pytest.raises(NotImplementedError, match="row-position"):
        state.feature_frame({"feat": ewm_lag2})

    # lag == 1 is safe even in a non-dense mode.
    state.feature_frame({"feat": ewm_lag1})

    # a known-dense mode is safe even at lag > 1.
    state.mode = "groupby"
    state.feature_frame({"feat": ewm_lag2})


# ---------------------------------------------------------------------------
# Task 7: the quantile shim. Quantiles have no sufficient statistic (unlike
# every other pooled family in this suite), so both engines run
# `np.quantile` over the identical value multiset -- any divergence is a pure
# windowing bug, hence `atol=0.0` everywhere below, not the suite's usual
# `atol=1e-10`.
# ---------------------------------------------------------------------------

QUANTILES = [
    ("roll_q", RollingQuantile(p=0.5, window_size=14, groupby=["store"])),
    ("roll_q90", RollingQuantile(p=0.9, window_size=14, groupby=["store"])),
    ("exp_q", ExpandingQuantile(p=0.5, groupby=["store"])),
    (
        "seas_q",
        SeasonalRollingQuantile(
            p=0.5, season_length=7, window_size=4, groupby=["store"]
        ),
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("label,tfm", QUANTILES, ids=[x[0] for x in QUANTILES])
def test_quantile_engines_agree(backend, label, tfm):  # noqa: ARG001
    """Quantiles must match EXACTLY -- both engines run np.quantile on the
    same value multiset, so any difference is a windowing bug."""
    assert_engines_agree(_panel(backend, n_times=90), {1: [tfm]}, ["store"], atol=0.0)


# Task 6's retrospective (see its report) found that every required test used
# `lag=1`, which hid a wrong-offset bug entirely, and separately that a
# happy-path N(10, 2) fixture alone missed a defect that produced 2405 where
# 0.0 was correct. Cover: lag > 1, the extreme quantiles p=0.0/1.0, the
# `keys=[]` (global_) branch of `_quantile_columns` (the groupby-only
# QUANTILES list above never reaches it), and window_size=1 (a window with at
# most one element per series).
QUANTILE_PARAM_CASES = [
    ("roll_q_lag3", 3, RollingQuantile(p=0.25, window_size=10, groupby=["store"])),
    ("roll_q_p0", 1, RollingQuantile(p=0.0, window_size=10, groupby=["store"])),
    ("roll_q_p1", 1, RollingQuantile(p=1.0, window_size=10, groupby=["store"])),
    (
        "roll_q_window1",
        1,
        RollingQuantile(p=0.5, window_size=1, min_samples=1, groupby=["store"]),
    ),
    ("exp_q_lag4", 4, ExpandingQuantile(p=0.75, groupby=["store"])),
    ("exp_q_global", 1, ExpandingQuantile(p=0.5, global_=True)),
    (
        "seas_q_lag2",
        2,
        SeasonalRollingQuantile(
            p=0.3, season_length=5, window_size=3, groupby=["store"]
        ),
    ),
    (
        "seas_q_p1",
        1,
        SeasonalRollingQuantile(
            p=1.0, season_length=7, window_size=4, groupby=["store"]
        ),
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "label,lag,tfm",
    QUANTILE_PARAM_CASES,
    ids=[x[0] for x in QUANTILE_PARAM_CASES],
)
def test_quantile_engines_agree_params(backend, label, lag, tfm):  # noqa: ARG001
    statics = [] if getattr(tfm, "global_", False) else ["store"]
    assert_engines_agree(_panel(backend, n_times=90), {lag: [tfm]}, statics, atol=0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_quantile_engines_agree_large_magnitude(backend):
    """Data-regime check (Task 6's retrospective: an N(10, 2) fixture alone
    missed a defect elsewhere in this plan). Reuses `_large_magnitude_panel`
    (near-constant values around 1e11). Unlike the Std families, quantiles
    read raw values directly -- no sum-of-squares cancellation is possible --
    so exact engine agreement is expected here too, not just "clips to
    zero"."""
    df_pl = _large_magnitude_panel(n_series=6, n_times=40)
    df = df_pl if backend == "polars" else df_pl.to_pandas()
    tfms = {
        1: [RollingQuantile(p=0.5, window_size=5, min_samples=2, groupby=["store"])]
    }
    assert_engines_agree(df, tfms, ["store"], atol=0.0)


def test_quantile_shim_matches_legacy_row_level_directly():
    """Direct proof against the legacy row-level implementation (not just
    engine-vs-engine agreement), covering RollingQuantile, ExpandingQuantile,
    and SeasonalRollingQuantile's `_bucket_feature_rows_impl`/`_window_stat`/
    `_expanding_stat`/`_seasonal_stat` bodies at once, on a panel with more
    than one series per bucket (so `min_samples` counts across series) and
    lag > 1."""
    import narwhals as nw

    from mlforecast.pooled import NarwhalsPooledState

    n_series, n_times = 6, 40
    df = _panel("polars", n_series=n_series, n_times=n_times, n_groups=3)
    store = df["store"].to_numpy().astype(np.int64)
    ds = df["ds"].to_numpy()
    y_arr = df["y"].to_numpy().astype(float)
    ordv = np.zeros(len(df), dtype=np.int64)
    for b in np.unique(store):
        m = store == b
        u = np.unique(ds[m])
        ordv[m] = np.searchsorted(u, ds[m])

    cases = [
        (RollingQuantile(p=0.4, window_size=6, min_samples=3, groupby=["store"]), 2),
        (ExpandingQuantile(p=0.6, groupby=["store"]), 3),
        (
            SeasonalRollingQuantile(
                p=0.2,
                season_length=4,
                window_size=3,
                min_samples=2,
                groupby=["store"],
            ),
            1,
        ),
    ]
    for tfm, lag in cases:
        tfm = tfm._set_core_tfm(lag)
        want = tfm._bucket_feature_rows_impl(store, ordv, y_arr)

        state = NarwhalsPooledState(
            agg=None,
            groups=None,
            group_cols=None,
            series_bucket_id=np.zeros(1, dtype=np.int64),
            join_cols=["unique_id", "ds"],
            keys=["store"],
            time_col="ds",
            mode="groupby",
        )._build(df, "ds", "y", np.float64)
        got_cols = state._quantile_columns({"feat": tfm})
        agg_nw = nw.from_native(state.agg, eager_only=True)
        agg_store = agg_nw.get_column("store").to_numpy().astype(np.int64)
        agg_ord = agg_nw.get_column("ord").to_numpy().astype(np.int64)
        # one row per (store, ord) in `state.agg`; build a lookup and apply it
        # positionally to every original row via its own (store, ord).
        lut = {
            (int(b), int(o)): v for b, o, v in zip(agg_store, agg_ord, got_cols["feat"])
        }
        got = np.array([lut[(int(b), int(o))] for b, o in zip(store, ordv)])

        np.testing.assert_allclose(
            got,
            want,
            atol=0.0,
            equal_nan=True,
            err_msg=f"{type(tfm).__name__} lag={lag}",
        )


def test_quantile_time_agg_slow_path_literal_matches_legacy_fit_values():
    """Fix-round-1 regression: `time_agg` + quantile must be SUPPORTED (legacy
    routes it through `_compute_bucket_feature_collapsed`), not refused. This
    is the same literal scenario as the pre-existing, unmodifiable
    `tests/test_pooled.py::test_time_agg_quantile_slow_path_literal`
    (daily sums `[11,22,33,44,55,66]`, `RollingQuantile(window_size=3)` ->
    `[nan,nan,nan,22,33,44]`), but calling the narwhals engine's FIT stage
    directly (`_preprocess_with_engine`, no `_predict_setup`/
    `_update_features`) so the assertion is actually reached: the real test
    always crashes first at Task 9's un-wired predict-path `AttributeError`
    (`state._ts_aggs`), before ever checking its own value assertion, so it
    cannot itself prove the fit-stage number is right or wrong either way.

    Revert-proof: with the (now-removed) `time_agg is not None: raise
    NotImplementedError` guard restored, this errors instead of computing
    `[nan, nan, nan, 22.0, 33.0, 44.0]`:
        NotImplementedError: RollingQuantile(time_agg='sum') ... is not yet
        supported by the narwhals pooled engine's quantile shim: ...
    Before that guard existed at all, the same scenario silently returned
    `[nan, nan, 6.0, 6.5, 12.0, 17.5]` (median of raw per-row values, ignoring
    the sum collapse) -- see the guard commit for that transcript. With the
    collapsed-store fix, it now matches the expected value exactly."""
    y_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 6), interval="1d", eager=True
    )
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 6 + ["b"] * 6,
            "ds": np.tile(dates.to_numpy(), 2),
            "y": y_a + y_b,
            "grp": ["X"] * 12,
        }
    )
    tfm = RollingQuantile(p=0.5, window_size=3, groupby=["grp"], time_agg="sum")
    out = _preprocess_with_engine("narwhals", df, {1: [tfm]}, ["grp"])
    out = out if isinstance(out, pl.DataFrame) else pl.from_pandas(out)
    col = tfm._get_name(1)
    got = (
        out.filter(pl.col("unique_id") == "a")
        .sort("ds")[col]
        .cast(pl.Float64)
        .to_numpy()
    )
    np.testing.assert_allclose(
        got, [np.nan, np.nan, np.nan, 22.0, 33.0, 44.0], atol=0.0, equal_nan=True
    )


TIME_AGG_QUANTILES = [
    (
        "roll_q_ta_sum",
        RollingQuantile(p=0.5, window_size=10, groupby=["store"], time_agg="sum"),
    ),
    (
        "exp_q_ta_mean",
        ExpandingQuantile(p=0.4, groupby=["store"], time_agg="mean"),
    ),
    (
        "seas_q_ta_max",
        SeasonalRollingQuantile(
            p=0.6, season_length=5, window_size=3, groupby=["store"], time_agg="max"
        ),
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "label,tfm", TIME_AGG_QUANTILES, ids=[x[0] for x in TIME_AGG_QUANTILES]
)
def test_quantile_time_agg_engines_agree(backend, label, tfm):  # noqa: ARG001
    """`time_agg` requested by the coordinator's fix-round-1: one case per
    quantile family (`RollingQuantile`/`ExpandingQuantile`/
    `SeasonalRollingQuantile`), each with a DIFFERENT `time_agg` value
    (sum/mean/max) to exercise `_time_agg_value_expr`'s distinct branches, at
    the family's usual `atol=0.0` (both engines run `np.quantile` on the same
    time_agg-collapsed value multiset, so any difference is a windowing bug,
    exactly as for the non-time_agg case)."""
    assert_engines_agree(_panel(backend, n_times=90), {1: [tfm]}, ["store"], atol=0.0)


def test_quantile_engines_agree_mixed_with_expression_transforms():
    """A state holding BOTH a quantile transform (no `_pooled_expr`, routed
    through `_quantile_columns`) and an ordinary expression transform in the
    SAME `feature_frame` call -- proves `feature_frame`'s split between
    `with_columns(exprs)` and the `nw.new_series` quantile attachment doesn't
    clobber or misalign either family."""
    df = _panel("polars", n_times=90)
    tfms = {
        1: [
            RollingQuantile(p=0.5, window_size=14, groupby=["store"]),
            RollingMean(14, groupby=["store"]),
        ]
    }
    assert_engines_agree(df, tfms, ["store"], atol=1e-9)


def _assert_narwhals_partition_path():
    """Every `partition_by` differential test below relies on `from_partition`
    actually routing through `NarwhalsPooledState` -- otherwise both sides of
    `assert_engines_agree` run the SAME legacy engine and the test passes
    vacuously regardless of what the narwhals engine does. Confirmed by
    reverting `mlforecast/{core,pooled,_pooled_engine}.py` to pre-Task-8 HEAD
    (`git stash`) and rerunning: every test below still PASSED (core.py still
    hardcoded `PooledState.from_partition`), which is exactly this failure
    mode -- hence this explicit assertion in each one."""
    import mlforecast.pooled as mp

    assert hasattr(mp.NarwhalsPooledState, "from_partition"), (
        "narwhals partition path missing"
    )


def _partition_df(backend, n_series, n_times, seed, n_promo_values=2):
    """`_panel` plus a `promo` partition_by column, `n_promo_values` distinct values."""
    df = _panel(backend, n_series=n_series, n_times=n_times)
    rng = np.random.default_rng(seed)
    promo = rng.integers(0, n_promo_values, len(df))
    if isinstance(df, pl.DataFrame):
        df = df.with_columns(promo=pl.Series(promo))
    else:
        df = df.assign(promo=promo)
    return df


@pytest.mark.parametrize("backend", BACKENDS)
def test_partition_by_global_engines_agree(backend):
    df = _partition_df(backend, n_series=30, n_times=60, seed=3)
    import mlforecast.pooled as mp

    assert hasattr(mp.NarwhalsPooledState, "from_partition"), (
        "narwhals partition path missing"
    )
    assert_engines_agree(
        df,
        {1: [RollingMean(7, global_=True, partition_by=["promo"], min_samples=1)]},
        [],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_partition_by_groupby_engines_agree(backend):
    df = _partition_df(backend, n_series=30, n_times=60, seed=4)
    import mlforecast.pooled as mp

    assert hasattr(mp.NarwhalsPooledState, "from_partition"), (
        "narwhals partition path missing"
    )
    assert_engines_agree(
        df,
        {1: [RollingMean(7, groupby=["store"], partition_by=["promo"], min_samples=1)]},
        ["store"],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_partition_by_local_engines_agree(backend):
    """local mode (no global_/groupby): bucket key is (unique_id, promo), parent
    scope is the series' own calendar -- the third `from_partition` branch,
    not exercised by the two tests above."""
    _assert_narwhals_partition_path()
    df = _partition_df(backend, n_series=24, n_times=50, seed=5)
    assert_engines_agree(
        df,
        {1: [RollingMean(7, partition_by=["promo"], min_samples=1)]},
        [],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_partition_by_quantile_engines_agree(backend):
    _assert_narwhals_partition_path()
    df = _partition_df(backend, n_series=24, n_times=70, seed=7)
    assert_engines_agree(
        df,
        {
            1: [
                RollingQuantile(
                    p=0.5,
                    window_size=7,
                    global_=True,
                    partition_by=["promo"],
                    min_samples=1,
                )
            ]
        },
        [],
        atol=0.0,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_partition_by_expanding_minmax_engines_agree(backend):
    """PUBLIC-API proof for the cum_min/cum_max forward-fill (ruling F17).

    A NaN target is rejected at fit (`ValueError: y column contains null
    values`), so a gap ordinal cannot be produced through plain `fit` -- it
    arises only from partition_by densified holes and predict-time
    placeholders. This test is therefore the first place the gap path is
    reachable through the public API: a time-varying partition key leaves
    each bucket without an observation at many parent-calendar ordinals, and
    legacy's np.fmin.accumulate carries the running extremum THROUGH those
    holes.
    """
    _assert_narwhals_partition_path()
    df = _partition_df(backend, n_series=24, n_times=70, seed=11, n_promo_values=3)
    assert_engines_agree(
        df,
        {
            1: [
                ExpandingMin(global_=True, partition_by=["promo"]),
                ExpandingMax(global_=True, partition_by=["promo"]),
            ]
        },
        [],
        atol=0.0,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_lookup_lag_partition_engines_agree(backend):
    """LookupLag must NOT be densified -- its lag counts OCCURRENCES, not
    calendar ordinals. A gappy partition (n_promo_values=3 on a small panel)
    exercises that directly: densifying would silently convert the
    occurrence lag into an ordinal lag."""
    _assert_narwhals_partition_path()
    df = _partition_df(backend, n_series=24, n_times=70, seed=8, n_promo_values=3)
    assert_engines_agree(df, {1: [LookupLag(partition_by=["promo"])]}, [], atol=0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ewm_partition_gapped_lag_gt1_engines_agree(backend):
    """`ExponentiallyWeightedMean` MUST be densified for partition_by (ruling
    F25): its `_pooled_expr` shifts by row POSITION, which only matches
    legacy's calendar-ORDINAL threshold fold when the grid is dense or
    `lag == 1`. At `lag == 1` the bug is invisible (consecutive ordinals
    differ by at least 1), so this test uses `lag=2` on a deliberately
    gapped partition (n_promo_values=3, small panel -> frequent gaps) --
    exactly the configuration the task brief measured diverging (11.0/31.0
    vs legacy's 16.0/36.0) before densification."""
    _assert_narwhals_partition_path()
    df = _partition_df(backend, n_series=20, n_times=40, seed=13, n_promo_values=3)
    assert_engines_agree(
        df,
        {
            2: [
                ExponentiallyWeightedMean(
                    alpha=0.3, global_=True, partition_by=["promo"]
                )
            ]
        },
        [],
        atol=1e-9,
    )


def test_lookup_lag_and_ewm_same_partition_state_raises():
    """The one configuration the two densification carve-outs cannot both
    satisfy (ruling F17 vs F25): a LookupLag and an EWM(lag>1) sharing the
    SAME partition_by key (mode/groupby/partition_by all identical) forces
    one shared aggregate table to be simultaneously sparse (for LookupLag)
    and dense (for EWM) -- refused loudly rather than computed silently
    wrong."""
    df = _partition_df("polars", n_series=20, n_times=40, seed=13, n_promo_values=3)
    prev = os.environ.get("MLFORECAST_POOLED_ENGINE")
    os.environ["MLFORECAST_POOLED_ENGINE"] = "narwhals"
    try:
        import importlib

        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        import mlforecast.core

        importlib.reload(mlforecast.core)
        fcst = MLForecast(
            models=[LinearRegression()],
            freq="1d",
            # LookupLag has no `global_`/`groupby` param -- it is always
            # "local" mode (bucket = (id_col, *partition_cols)). EWM must
            # match that exact mode/groupby/partition_by combination to land
            # in the SAME pooled state (`_get_pooled_tfms`'s key is
            # `(mode, group_cols, partition_cols)`); `global_=True` (as in
            # the standalone EWM test above) gives "nonlocal" instead and
            # never collides.
            lag_transforms={
                2: [
                    LookupLag(partition_by=["promo"]),
                    ExponentiallyWeightedMean(alpha=0.3, partition_by=["promo"]),
                ]
            },
        )
        with pytest.raises(NotImplementedError, match="LookupLag"):
            fcst.preprocess(df, static_features=[], dropna=False)
    finally:
        if prev is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = prev
        import importlib

        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        import mlforecast.core

        importlib.reload(mlforecast.core)


# ---- Task 9: predict -- tail evaluation + per-bucket seed rows ----


def _predict_with_engine(engine, df, tfms, statics, h):
    prev = os.environ.get("MLFORECAST_POOLED_ENGINE")
    os.environ["MLFORECAST_POOLED_ENGINE"] = engine
    try:
        import mlforecast.pooled, mlforecast.core

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)
        fcst = MLForecast(
            models=[LinearRegression()], freq="1d", lags=[1], lag_transforms=tfms
        )
        fcst.fit(df, static_features=statics)
        return fcst.predict(h)
    finally:
        if prev is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = prev
        import mlforecast.pooled, mlforecast.core

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)


PREDICT_CASES = [
    (
        "p_rolling",
        [RollingMean(7, groupby=["store"]), RollingMean(14, groupby=["store"])],
    ),
    (
        "p_expanding",
        [ExpandingMean(groupby=["store"]), ExpandingStd(groupby=["store"])],
    ),
    (
        "p_expanding_minmax",
        [ExpandingMin(groupby=["store"]), ExpandingMax(groupby=["store"])],
    ),
    ("p_ewm", [ExponentiallyWeightedMean(alpha=0.3, groupby=["store"])]),
    (
        "p_seasonal",
        [SeasonalRollingMean(season_length=7, window_size=4, groupby=["store"])],
    ),
    ("p_quantile", [RollingQuantile(p=0.5, window_size=14, groupby=["store"])]),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("label,tfms", PREDICT_CASES, ids=[x[0] for x in PREDICT_CASES])
def test_predict_engines_agree(backend, label, tfms):
    df = _panel(backend, n_series=30, n_times=90)
    a = _predict_with_engine("numpy", df, {1: tfms}, ["store"], 14)
    b = _predict_with_engine("narwhals", df, {1: tfms}, ["store"], 14)
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    a, b = a.sort("unique_id", "ds"), b.sort("unique_id", "ds")
    np.testing.assert_allclose(
        a["LinearRegression"].to_numpy(),
        b["LinearRegression"].to_numpy(),
        atol=1e-9,
        err_msg=label,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_models_do_not_leak_state(backend):
    """_backup/restore must isolate each model's recursive walk: a leaked
    prefix would make the second model's forecast depend on the first's."""
    from sklearn.tree import DecisionTreeRegressor

    df = _panel(backend, n_series=20, n_times=60)
    tfms = {1: [ExpandingMean(groupby=["store"])]}
    os.environ["MLFORECAST_POOLED_ENGINE"] = "narwhals"
    try:
        import mlforecast.pooled, mlforecast.core

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)
        together = MLForecast(
            models=[LinearRegression(), DecisionTreeRegressor(random_state=0)],
            freq="1d",
            lags=[1],
            lag_transforms=tfms,
        )
        together.fit(df, static_features=["store"])
        both = together.predict(10)
        alone = MLForecast(
            models=[DecisionTreeRegressor(random_state=0)],
            freq="1d",
            lags=[1],
            lag_transforms=tfms,
        )
        alone.fit(df, static_features=["store"])
        solo = alone.predict(10)
        b = both if isinstance(both, pl.DataFrame) else pl.from_pandas(both)
        s = solo if isinstance(solo, pl.DataFrame) else pl.from_pandas(solo)
        np.testing.assert_allclose(
            b.sort("unique_id", "ds")["DecisionTreeRegressor"].to_numpy(),
            s.sort("unique_id", "ds")["DecisionTreeRegressor"].to_numpy(),
            atol=0.0,
        )
    finally:
        os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        import mlforecast.pooled, mlforecast.core

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)


# ---- Extra Task 9 coverage beyond the brief's PREDICT_CASES ----
#
# Lesson from earlier tasks: a test suite using only ``lag=1`` hid a
# wrong-number bug (Lesson 2). EWM's positional shift is explicitly the
# family whose row-vs-ordinal divergence was measured at ``lag > 1``
# (`_guard_ewm_positional_shift`'s docstring); this exercises EWM predict at
# ``lag=1``, ``lag=3`` and ``lag=5`` over a horizon that outlasts a small
# window, so a step where the query references an already-*predicted* row
# (not just historical data) is actually reached.


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("lag", [1, 3, 5])
def test_predict_ewm_lag_gt_1_engines_agree(backend, lag):
    df = _panel(backend, n_series=16, n_times=40)
    tfms = {lag: [ExponentiallyWeightedMean(alpha=0.4, groupby=["store"])]}
    a = _predict_with_engine("numpy", df, tfms, ["store"], 8)
    b = _predict_with_engine("narwhals", df, tfms, ["store"], 8)
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    a, b = a.sort("unique_id", "ds"), b.sort("unique_id", "ds")
    np.testing.assert_allclose(
        a["LinearRegression"].to_numpy(),
        b["LinearRegression"].to_numpy(),
        atol=1e-9,
        err_msg=f"lag={lag}",
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_predict_rolling_min_max_horizon_exceeds_window_engines_agree(backend):
    """A horizon longer than ``lag + window_size`` forces later recursive
    steps to read predicted (pending), not historical, rows out of the
    RollingMin/RollingMax raw ``mn``/``mx`` columns -- the one family whose
    ``_pooled_expr`` shifts a raw (not prefix-summed) column, so this is the
    sharpest test of the retention/seed boundary (`_pooled_retention`)."""
    df = _panel(backend, n_series=16, n_times=40)
    tfms = {1: [RollingMin(3, groupby=["store"]), RollingMax(3, groupby=["store"])]}
    a = _predict_with_engine("numpy", df, tfms, ["store"], 10)
    b = _predict_with_engine("narwhals", df, tfms, ["store"], 10)
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    a, b = a.sort("unique_id", "ds"), b.sort("unique_id", "ds")
    np.testing.assert_allclose(
        a["LinearRegression"].to_numpy(),
        b["LinearRegression"].to_numpy(),
        atol=1e-9,
    )


# ---- Task 10: update()/append_observations and keep_last_n trimming ----


def _update_then_predict(engine, df, new_df, tfms, statics, h, lags=None):
    prev = os.environ.get("MLFORECAST_POOLED_ENGINE")
    os.environ["MLFORECAST_POOLED_ENGINE"] = engine
    try:
        import mlforecast.pooled, mlforecast.core

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)
        fcst = MLForecast(
            models=[LinearRegression()],
            freq="1d",
            lags=lags or [1],
            lag_transforms=tfms,
        )
        fcst.fit(df, static_features=statics)
        fcst.update(new_df)
        return fcst.predict(h)
    finally:
        if prev is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = prev
        import mlforecast.pooled, mlforecast.core

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)


@pytest.mark.parametrize("backend", BACKENDS)
def test_update_then_predict_engines_agree(backend):
    df = _panel(backend, n_series=20, n_times=60)
    d = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    last = d["ds"].max()
    nxt = d.filter(pl.col("ds") == last).with_columns(
        (pl.col("ds") + pl.duration(days=1)).alias("ds")
    )
    new_df = nxt if isinstance(df, pl.DataFrame) else nxt.to_pandas()
    tfms = {1: [RollingMean(7, groupby=["store"]), ExpandingMean(groupby=["store"])]}
    a = _update_then_predict("numpy", df, new_df, tfms, ["store"], 7)
    b = _update_then_predict("narwhals", df, new_df, tfms, ["store"], 7)
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    np.testing.assert_allclose(
        a.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        b.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        atol=1e-9,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_update_then_predict_lag_gt_1_engines_agree(backend):
    """Lesson from earlier tasks: a suite using only ``lag=1`` hid a
    wrong-number bug. Vary the transform's own lag (not the plain ``lags=``
    regressor feature) so ``append_observations``'s ordinal recompute is
    exercised with a lag that reaches back past the single newly appended
    timestamp."""
    df = _panel(backend, n_series=16, n_times=40)
    d = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    last = d["ds"].max()
    nxt = d.filter(pl.col("ds") == last).with_columns(
        (pl.col("ds") + pl.duration(days=1)).alias("ds")
    )
    new_df = nxt if isinstance(df, pl.DataFrame) else nxt.to_pandas()
    tfms = {3: [RollingMean(5, groupby=["store"]), ExpandingMean(groupby=["store"])]}
    a = _update_then_predict("numpy", df, new_df, tfms, ["store"], 5, lags=[3])
    b = _update_then_predict("narwhals", df, new_df, tfms, ["store"], 5, lags=[3])
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    np.testing.assert_allclose(
        a.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        b.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        atol=1e-9,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_keep_last_n_trims_expanding_state(backend):
    """The seed row makes prefix-dependent states trimmable -- the limitation
    _trim_pooled_states documents today."""
    prev = os.environ.get("MLFORECAST_POOLED_ENGINE")
    os.environ["MLFORECAST_POOLED_ENGINE"] = "narwhals"
    try:
        import mlforecast.pooled, mlforecast.core

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)
        df = _panel(backend, n_series=20, n_times=120)
        fcst = MLForecast(
            models=[LinearRegression()],
            freq="1d",
            lags=[1],
            lag_transforms={1: [ExpandingMean(groupby=["store"])]},
        )
        fcst.fit(df, static_features=["store"], keep_last_n=30)
        state = next(iter(fcst.ts._pooled_states.values()))
        t = (
            state.agg
            if isinstance(state.agg, pl.DataFrame)
            else pl.from_pandas(state.agg)
        )
        assert t.height < 5 * 120, (
            f"expanding state kept {t.height} aggregate rows; the seed row "
            "should have allowed a trim"
        )
    finally:
        if prev is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = prev
        import mlforecast.pooled, mlforecast.core

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)


@pytest.mark.parametrize("backend", BACKENDS)
def test_keep_last_n_trim_predict_engines_agree(backend):
    """Structural-invariant test alone (the previous test) can't catch a trim
    that shrinks the table WRONG -- e.g. drops the seed row too, or mis-seeds
    it -- only that it shrinks at all. This closes that gap: fit with a small
    explicit ``keep_last_n`` (small enough to force a real trim on a fixture
    this size) on BOTH engines, then predict, and require identical output --
    legacy never trims an Expanding state at all (`_is_finite_window` is
    False for it), so this is really "trimmed narwhals vs. untrimmed legacy,
    same numbers", which can only hold if the seed row is exactly correct.

    A prior version of this test compared predictions ONLY: with the trim
    skipped entirely (Task 3's temporary guard, reverted-source check), the
    narwhals state stays untrimmed too, and untrimmed-vs-untrimmed agrees
    trivially -- a test that cannot fail. The explicit row-count assertion
    below closes that gap by pinning that a trim actually happened.
    """
    # groupby="store" (5 groups by _panel's default n_groups) pools every
    # series sharing a store into ONE aggregate row per (store, timestamp) --
    # the untrimmed height is n_groups * n_times = 5*80=400, NOT
    # n_series*n_times (a first version of this bound used the latter, which
    # is so much larger than 400 that it could never catch a no-op trim --
    # see the module docstring's "tests that cannot fail" lesson). Retention
    # is max(keep_last_n=20, RollingMean(5)'s own lag+window=6) = 20, plus
    # one seed row per store -> 5*21=105 expected after a real trim.
    n_groups, n_times = 5, 80
    df = _panel(backend, n_series=16, n_times=n_times, n_groups=n_groups)
    tfms = {1: [ExpandingMean(groupby=["store"]), RollingMean(5, groupby=["store"])]}
    a, _ = _predict_with_engine_keep_last_n("numpy", df, tfms, ["store"], 6, 20)
    b, b_state = _predict_with_engine_keep_last_n(
        "narwhals", df, tfms, ["store"], 6, 20
    )
    b_agg = (
        b_state.agg
        if isinstance(b_state.agg, pl.DataFrame)
        else pl.from_pandas(b_state.agg)
    )
    untrimmed_height = n_groups * n_times
    assert b_agg.height < untrimmed_height // 2, (
        f"narwhals state kept {b_agg.height} aggregate rows out of "
        f"{untrimmed_height} untrimmed; keep_last_n=20 on an 80-step "
        "fixture should have forced a real trim (expected ~105)"
    )
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    np.testing.assert_allclose(
        a.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        b.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        atol=1e-9,
    )


def _predict_with_engine_keep_last_n(engine, df, tfms, statics, h, keep_last_n):
    prev = os.environ.get("MLFORECAST_POOLED_ENGINE")
    os.environ["MLFORECAST_POOLED_ENGINE"] = engine
    try:
        import mlforecast.pooled, mlforecast.core

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)
        fcst = MLForecast(
            models=[LinearRegression()], freq="1d", lags=[1], lag_transforms=tfms
        )
        fcst.fit(df, static_features=statics, keep_last_n=keep_last_n)
        state = next(iter(fcst.ts._pooled_states.values()))
        return fcst.predict(h), state
    finally:
        if prev is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = prev
        import mlforecast.pooled, mlforecast.core

        importlib.reload(mlforecast.pooled)
        importlib.reload(mlforecast.core)
