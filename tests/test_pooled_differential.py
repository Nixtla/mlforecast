# tests/test_pooled_differential.py
"""Both engines, identical inputs, identical outputs.

The pooled suite proves the narwhals engine is correct in absolute terms; this
file proves the two engines agree, so any divergence names itself.
"""

import operator

import narwhals as nw
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

from ._pooled_engine_env import pooled_engine

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
    """Select ``engine`` for the duration of one preprocess call.

    ``pooled_engine`` restores every reloaded module object by snapshot, so
    this leaves no residue for later tests -- see
    ``tests/_pooled_engine_env.py``.
    """
    with pooled_engine(engine):
        fcst = MLForecast(models=[LinearRegression()], freq="1d", lag_transforms=tfms)
        return fcst.preprocess(df, static_features=statics, dropna=False)


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


def _values_by_ordinal(df_pl):
    """``(ordered dates, [[raw y at that date], ...])`` for a one-bucket panel.

    ``_large_magnitude_panel`` puts every series in ``store=0``, so a pooled
    window over ``w`` ordinals is a window over every series' raw value at
    those ``w`` dates -- exactly the value set the exact oracle below needs.
    """
    d = df_pl.sort("ds")
    dates = sorted(set(d["ds"].to_list()))
    return dates, [d.filter(pl.col("ds") == ts)["y"].to_list() for ts in dates]


def _exact_std_by_date(df_pl, ordinals_for):
    """``{ds: exact sample std}`` -- the TRUE value, to 60 significant digits.

    Legacy is deliberately NOT the reference here. Its ``RollingStd``/
    ``ExpandingStd`` aggregate fast paths (``_rolling_std_from_agg`` /
    ``_expanding_std_from_agg``) compute the variance as
    ``sum(y**2) - sum(y)**2/n``, which at this magnitude is pure rounding
    noise -- these tests used to assert that the narwhals engine reproduced
    that noise (clipped to 0.0). It no longer does, on purpose: the shifted
    moments (``sK``/``qK``, see ``_pooled_engine.compute_kref``) recover the
    true std, so the only reference that can settle the result is an exact
    one. float64 values are exact rationals, so ``Fraction`` gives the true
    variance with no rounding at all.
    """
    from .test_pooled_narwhals import _exact_sample_std

    dates, per_ord = _values_by_ordinal(df_pl)
    out = {}
    for t in range(len(dates)):
        vals = [v for i in ordinals_for(t, len(dates)) for v in per_ord[i]]
        if len(vals) > 1:
            out[dates[t]] = _exact_sample_std(vals)
    return out


def _rolling_ordinals(window_size):
    return lambda t, n: list(range(max(t - window_size, 0), t))  # noqa: ARG005


def _expanding_ordinals():
    return lambda t, n: list(range(0, t))  # noqa: ARG005


def _seasonal_ordinals(season_length, window_size, lag=1):
    offs = [lag + k * season_length for k in range(window_size)]
    return lambda t, n: [t - o for o in offs if 0 <= t - o < n]


# MEASURED over both backends and all three families on
# `_large_magnitude_panel` (see the fix report): worst relative error against
# the exact std is 1.1e-15. ~5x headroom for a backend's own summation order.
_EXACT_STD_TOL = 5e-15


def _assert_matches_exact_std(df_pl, tfms, backend, ordinals_for):
    """The narwhals engine's std column vs the EXACT std, end to end.

    Returns legacy's own worst relative error on the same rows, so each
    caller can state -- and assert -- what legacy does here rather than
    assuming it.
    """
    from .test_pooled_narwhals import _rel_err

    want = _exact_std_by_date(df_pl, ordinals_for)
    df = df_pl if backend == "polars" else df_pl.to_pandas()
    a = _preprocess_with_engine("numpy", df, tfms, ["store"])
    b = _preprocess_with_engine("narwhals", df, tfms, ["store"])
    a = (a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)).sort(
        "unique_id", "ds"
    )
    b = (b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)).sort(
        "unique_id", "ds"
    )
    feat_cols = [c for c in b.columns if c not in ("unique_id", "ds", "store", "y")]
    assert feat_cols, "no feature columns produced"
    worst_nw, worst_nw_at, worst_legacy, checked = 0.0, None, 0.0, 0
    for c in feat_cols:
        rows = b.select(["ds", c]).rows()
        legacy_rows = a.select(["ds", c]).rows()
        for (ds, got), (_, got_legacy) in zip(rows, legacy_rows):
            if got is None or ds not in want:
                continue
            checked += 1
            err = _rel_err(float(got), want[ds])
            if err > worst_nw:
                worst_nw, worst_nw_at = err, (c, ds)
            if got_legacy is not None:
                worst_legacy = max(worst_legacy, _rel_err(float(got_legacy), want[ds]))
    assert checked > 0, "no rows compared against the oracle"
    assert worst_nw < _EXACT_STD_TOL, (
        f"narwhals vs exact std: worst relative error {worst_nw:g} at "
        f"{worst_nw_at} over {checked} rows (limit {_EXACT_STD_TOL:g})"
    )
    return worst_legacy, checked


@pytest.mark.parametrize("backend", BACKENDS)
def test_rolling_std_large_magnitude_matches_the_exact_std(backend):
    """Finding 3's fixture, re-pointed at the EXACT std.

    This test used to assert that the narwhals engine clipped to 0.0 here,
    matching legacy's aggregate path. That pinned the defect: the true std of
    these windows is small but nonzero, and both engines were losing it
    entirely to the cancellation in ``sum(y**2) - sum(y)**2/n``. With the
    shifted moments the narwhals engine now returns the true value, so the
    assertion is against arbitrary-precision truth -- and the legacy engine,
    which still uses the raw two-moment formula, is asserted to be
    catastrophically WRONG on the very same rows. That second assertion is
    what proves the fixture still reaches the ill-conditioned regime: if it
    ever stops doing so, this test starts passing vacuously against the
    pre-fix code, and the guard fails instead.
    """
    df_pl = _large_magnitude_panel()
    tfms = {1: [RollingStd(5, min_samples=2, groupby=["store"])]}
    worst_legacy, _ = _assert_matches_exact_std(
        df_pl, tfms, backend, _rolling_ordinals(5)
    )
    assert worst_legacy > 0.5, (
        "precondition failed: legacy's raw two-moment formula is still "
        f"accurate on this panel (worst relative error {worst_legacy:g}) -- "
        "the fixture no longer reaches the regime this fix exists for"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_expanding_std_large_magnitude_matches_the_exact_std(backend):
    """Same construction and same two-sided assertion as the RollingStd case."""
    df_pl = _large_magnitude_panel()
    tfms = {1: [ExpandingStd(groupby=["store"])]}
    worst_legacy, _ = _assert_matches_exact_std(
        df_pl, tfms, backend, _expanding_ordinals()
    )
    assert worst_legacy > 0.5, (
        "precondition failed: legacy's raw two-moment formula is still "
        f"accurate on this panel (worst relative error {worst_legacy:g}) -- "
        "the fixture no longer reaches the regime this fix exists for"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_std_of_constant_data_is_exactly_zero(backend):
    """A constant bucket at 1e11: the narwhals engine returns EXACTLY 0.0.

    With the shifted moments this holds by construction -- ``y - K`` is
    identically 0, so ``sK`` and ``qK`` are exact zeros and no cancellation
    is possible -- and it pins the observable half of the variance clamp
    (``.clip(lower_bound=0.0)``, never ``.abs()``): no NaN out of a negative
    sqrt, no tiny positive dust.

    The legacy engine is checked too, in the opposite direction: its raw
    two-moment formula returns values in the THOUSANDS here (measured 2634.8
    against a true std of exactly 0.0), which is both the reason this file no
    longer treats legacy as the reference for ill-conditioned data and the
    guard that this fixture really is in the regime the fix addresses.
    """
    df_pl = _large_magnitude_panel().with_columns(pl.lit(1e11).alias("y"))
    df = df_pl if backend == "polars" else df_pl.to_pandas()
    tfms = {
        1: [
            RollingStd(5, min_samples=2, groupby=["store"]),
            ExpandingStd(groupby=["store"]),
            SeasonalRollingStd(
                season_length=1, window_size=5, min_samples=2, groupby=["store"]
            ),
        ]
    }
    worst_legacy = 0.0
    for engine in ("numpy", "narwhals"):
        out = _preprocess_with_engine(engine, df, tfms, ["store"])
        out = out if isinstance(out, pl.DataFrame) else pl.from_pandas(out)
        feat_cols = [
            c for c in out.columns if c not in ("unique_id", "ds", "store", "y")
        ]
        assert feat_cols, "no feature columns produced"
        for c in feat_cols:
            v = out[c].cast(pl.Float64).to_numpy()
            v = v[~np.isnan(v)]
            assert len(v) > 0, f"{engine}/{c}: every row was null"
            if engine == "numpy":
                worst_legacy = max(worst_legacy, float(np.abs(v).max()))
                continue
            np.testing.assert_array_equal(v, np.zeros_like(v), err_msg=f"{engine}, {c}")
    assert worst_legacy > 100.0, (
        "precondition failed: the legacy two-moment formula no longer blows "
        f"up on constant 1e11 data (worst |value| {worst_legacy:g}) -- this "
        "fixture would no longer distinguish the fix from the defect"
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_seasonal_rolling_std_large_magnitude_matches_the_exact_std(backend):
    """The 1863x seasonal divergence from legacy is GONE, not documented.

    This test used to assert only that the narwhals engine clipped its
    (meaningless) residue to 0.0 here, with a long note explaining that full
    engine agreement was impossible: ``SeasonalRollingStd`` has no legacy
    aggregate fast path, so legacy computes ``np.std(vals, ddof=1)`` directly
    on the raw values -- numerically stable, and reporting the true small
    nonzero std -- while the narwhals aggregate formula lost that signal
    entirely to cancellation (measured: every valid row diverged, up to 1863
    in absolute terms).

    With the shifted moments there is no such limitation, so this asserts the
    two things that used to be impossible:

    * the narwhals engine matches the EXACT std (arbitrary precision), and
    * legacy -- which is the accurate one for THIS family -- also matches it,
      i.e. the two engines now agree, which is the direct inverse of the old
      behaviour this test was written to document.
    """
    from .test_pooled_narwhals import _rel_err

    df_pl = _large_magnitude_panel()
    tfm = SeasonalRollingStd(
        season_length=1, window_size=5, min_samples=2, groupby=["store"]
    )
    tfms = {1: [tfm]}
    worst_legacy, checked = _assert_matches_exact_std(
        df_pl, tfms, backend, _seasonal_ordinals(1, 5)
    )
    assert checked > 0
    # 1e-1, not 1e-15: legacy is the ACCURATE engine for this family, but
    # only to the precision of `np.std`'s own mean subtraction, and this
    # panel's spread is ~4 ULP of its magnitude -- measured, legacy lands
    # 4.2e-2 relative off the exact value where the shifted moments land
    # 1e-15 off. The bound that matters is the comparison with what this
    # test used to document: an engine divergence of 1863 ABSOLUTE on a true
    # std of ~1e-4, i.e. 1.8e7 relative.
    assert worst_legacy < 0.1, (
        "legacy's stable np.std path should now be within its own precision "
        f"of the exact std for this family; got relative error {worst_legacy:g}"
    )

    # And the same panel through the RollingStd family, whose LEGACY path
    # does use the unstable aggregate formula, must still be wrecked -- the
    # precondition that this magnitude is genuinely ill-conditioned, checked
    # here so the assertion above cannot pass vacuously on well-behaved data.
    dates, per_ord = _values_by_ordinal(df_pl)
    from .test_pooled_narwhals import _exact_sample_std, _naive_sample_std

    worst_naive = 0.0
    for t in range(len(dates)):
        vals = [v for i in _seasonal_ordinals(1, 5)(t, len(dates)) for v in per_ord[i]]
        if len(vals) > 1:
            worst_naive = max(
                worst_naive, _rel_err(_naive_sample_std(vals), _exact_sample_std(vals))
            )
    assert worst_naive > 0.5, (
        "precondition failed: the naive two-moment formula is still accurate "
        f"on this panel (worst relative error {worst_naive:g})"
    )


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
    with pooled_engine("narwhals"):
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


# ---- Task 9: predict -- tail evaluation + per-bucket seed rows ----


def _predict_with_engine(engine, df, tfms, statics, h):
    with pooled_engine(engine):
        fcst = MLForecast(
            models=[LinearRegression()], freq="1d", lags=[1], lag_transforms=tfms
        )
        fcst.fit(df, static_features=statics)
        return fcst.predict(h)


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
    with pooled_engine("narwhals"):
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
    with pooled_engine(engine):
        fcst = MLForecast(
            models=[LinearRegression()],
            freq="1d",
            lags=lags or [1],
            lag_transforms=tfms,
        )
        fcst.fit(df, static_features=statics)
        fcst.update(new_df)
        return fcst.predict(h)


@pytest.mark.parametrize("backend", BACKENDS)
def test_update_then_predict_engines_agree(backend):
    df = _panel(backend, n_series=20, n_times=60)
    d = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    last = d["ds"].max()
    nxt = d.filter(pl.col("ds") == last).with_columns(
        (pl.col("ds") + pl.duration(days=1)).alias("ds")
    )
    new_df = nxt if isinstance(df, pl.DataFrame) else nxt.to_pandas()
    # ExpandingMin/ExpandingMax/EWM are here deliberately. Until this fix
    # wave this test used only RollingMean + ExpandingMean -- the two families
    # that never read an `A`-prefixed accumulate column -- so it could not
    # fail against `append_observations` dropping those columns on `update()`.
    tfms = {
        1: [
            RollingMean(7, groupby=["store"]),
            ExpandingMean(groupby=["store"]),
            ExpandingMin(groupby=["store"]),
            ExpandingMax(groupby=["store"]),
            ExponentiallyWeightedMean(alpha=0.3, groupby=["store"]),
        ]
    }
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
    tfms = {
        3: [
            RollingMean(5, groupby=["store"]),
            ExpandingMean(groupby=["store"]),
            ExpandingMin(groupby=["store"]),
            ExpandingMax(groupby=["store"]),
            ExponentiallyWeightedMean(alpha=0.3, groupby=["store"]),
        ]
    }
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
    with pooled_engine("narwhals"):
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
    with pooled_engine(engine):
        fcst = MLForecast(
            models=[LinearRegression()], freq="1d", lags=[1], lag_transforms=tfms
        )
        fcst.fit(df, static_features=statics, keep_last_n=keep_last_n)
        state = next(iter(fcst.ts._pooled_states.values()))
        return fcst.predict(h), state


# ---- Task 10 fix round 3: dynamic partition_by reassignment at h>=3 ----
#
# `h=1`/`h=2` are USELESS for this regression: the broken code (dense
# `append_predictions` bucket range bounded by `bids.max() + 1`, not the
# state's full registered bucket count) only under-advances a bucket's
# shared-parent calendar on a step where NO series is currently pointing at
# it -- and the resulting misalignment only bites once some LATER step reads
# a window against that now-behind bucket. Both engines agree trivially at
# h<=2 even against the broken code (see the revert-proof in the task-10
# report's third fix round), so this test requires h>=3.


def _panel_with_cycling_promo(backend, n_series=20, n_times=60, n_groups=4):
    """`_panel` plus a `promo` column that cycles 0,1,2 by TIME (identical
    pattern for every series, exactly like the isolating reproduction in
    ``tmp/repro_h_gt_1_partition_divergence.py``): every series shares the
    SAME promo value at a given timestamp, so a `groupby=["store"]` state's
    per-store buckets are reassigned in lockstep as promo cycles, and a
    `local` (no groupby) state's per-series buckets are reassigned
    identically for every series too.
    """
    d = _panel(backend, n_series=n_series, n_times=n_times, n_groups=n_groups)
    d = d if isinstance(d, pl.DataFrame) else pl.from_pandas(d)
    d = d.with_columns(
        ((pl.col("ds").rank("dense") - 1) % 3).cast(pl.Int64).alias("promo")
    )
    return d if backend == "polars" else d.to_pandas()


def _dynamic_reassignment_x_df(backend, df, h):
    """Future `X_df` continuing the fit-time promo cycle -- reassigns EVERY
    series between existing (store, promo) / (id, promo) buckets each step,
    never introducing a brand-new partition value (every promo value 0/1/2
    was already observed by every series during fit)."""
    d = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    uids = d.get_column("unique_id").unique(maintain_order=True).to_list()
    last = d.get_column("ds").max()
    n_times = d.filter(pl.col("unique_id") == uids[0]).height
    dates = pl.datetime_range(
        last + pl.duration(days=1),
        last + pl.duration(days=h),
        interval="1d",
        eager=True,
    )
    promo = [(n_times + step) % 3 for step in range(h)]
    x_df = pl.DataFrame(
        {
            "unique_id": np.repeat(uids, h),
            "ds": np.tile(dates.to_numpy(), len(uids)),
            "promo": np.tile(np.asarray(promo, dtype=np.int64), len(uids)),
        }
    )
    return x_df if backend == "polars" else x_df.to_pandas()


def _predict_with_engine_x_df(engine, df, tfms, statics, x_df, h):
    with pooled_engine(engine):
        fcst = MLForecast(
            models=[LinearRegression()], freq="1d", lags=[1], lag_transforms=tfms
        )
        fcst.fit(df, static_features=statics)
        return fcst.predict(h, X_df=x_df)


PARTITION_REASSIGN_CASES = [
    (
        "groupby_partition_by",
        {
            1: [
                RollingMean(7, groupby=["store"], partition_by=["promo"]),
                ExpandingMean(groupby=["store"], partition_by=["promo"]),
            ]
        },
        ["store"],
    ),
    (
        "local_partition_by",
        {
            1: [
                RollingMean(7, partition_by=["promo"]),
                ExpandingMean(partition_by=["promo"]),
            ]
        },
        ["store"],
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "label,tfms,statics",
    PARTITION_REASSIGN_CASES,
    ids=[c[0] for c in PARTITION_REASSIGN_CASES],
)
def test_dynamic_partition_reassignment_engines_agree_h_gt_2(
    backend, label, tfms, statics
):
    """Regression for a Critical found downstream of Task 10: a series
    dynamically REASSIGNED between already-existing `partition_by` buckets
    across a recursive predict walk diverged between engines starting at
    h=3 (measured: max|diff| up to ~0.05 on `groupby+partition_by`, ~0.02 on
    `local+partition_by`), while freezing the assignment (same series, same
    buckets, never moved) agreed exactly at every horizon -- isolating the
    mechanism to the per-step bucket-reassignment path, not the
    densification/predict-tail machinery (which still runs identically
    either way).

    Mechanism: `NarwhalsPooledState._aggregate_predictions_by_bucket`'s
    dense branch (every non-`LookupLag` `partition_by`/densified state)
    bounded its synthetic per-step bucket range by
    `self.series_bucket_id.max() + 1` -- but `series_bucket_id` has one
    entry per CURRENT series, not per registered bucket. A bucket sharing a
    parent (shared) calendar with the CURRENTLY active ones, but with no
    series pointing at it THIS step, was silently excluded whenever its own
    id happened to be numerically higher than every currently-active
    bucket's id -- which happens routinely once promo/partition values
    cycle series between LOW-id and HIGH-id buckets over time. That
    bucket's shared-parent ordinal then permanently falls one step behind
    on every predict step it was excluded from, with no error raised; a
    LATER step reading a window against it (once some series moves back)
    silently computes off a stale ordinal. h=1/h=2 don't yet reach a window
    boundary far enough back to expose the drift (see the module comment
    above) -- hence h>=3 is required to catch this.

    Fixed by bounding the dense branch's bucket range by
    `len(self.groups)` (the state's full registered bucket count) instead.
    """
    n_series, n_times, n_groups, h = 20, 60, 4, 7
    df = _panel_with_cycling_promo(
        backend, n_series=n_series, n_times=n_times, n_groups=n_groups
    )
    x_df = _dynamic_reassignment_x_df(backend, df, h)

    # Precondition: the reassignment this test exists to exercise must
    # actually happen, or this test would quietly stop testing anything --
    # every series' own promo value must differ across at least two steps
    # of the horizon (each series shares the SAME promo pattern, so
    # checking one series suffices).
    x_nw = x_df if isinstance(x_df, pl.DataFrame) else pl.from_pandas(x_df)
    one_series = x_nw.filter(pl.col("unique_id") == x_nw["unique_id"][0])
    assert one_series["promo"].n_unique() > 1, (
        "precondition failed: the future X_df never reassigns a series to a "
        "different partition_by bucket across the horizon -- this test "
        "would pass even against the broken code"
    )

    a = _predict_with_engine_x_df("numpy", df, tfms, statics, x_df, h)
    b = _predict_with_engine_x_df("narwhals", df, tfms, statics, x_df, h)
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    a, b = a.sort("unique_id", "ds"), b.sort("unique_id", "ds")
    for h_idx in (3, 7):
        a_h = a.filter(pl.col("ds") == a["ds"].unique().sort()[h_idx - 1])
        b_h = b.filter(pl.col("ds") == b["ds"].unique().sort()[h_idx - 1])
        np.testing.assert_allclose(
            a_h["LinearRegression"].to_numpy(),
            b_h["LinearRegression"].to_numpy(),
            atol=1e-9,
            err_msg=f"{label}: engines diverge at h={h_idx}",
        )


# ---------------------------------------------------------------------------
# Fix wave: the accumulate baseline across ALL THREE state-entry paths.
#
# `ExpandingMin`, `ExpandingMax` and `ExponentiallyWeightedMean` are the only
# pooled families whose predict seed reads an `A`-prefixed accumulate column
# (`_accumulate_specs`). Two paths built a state without ever materializing
# those columns, and the seed substitution was guarded by
# `if the column is present`, so the result was a WRONG NUMBER with no error:
#
#   * `core.py:_initialize_lag_transform_states` (`history_warmup` and
#     `predict(new_df=...)`) ran `ensure_time_aggs` + `ensure_densified` but
#     not `ensure_accumulates`;
#   * `NarwhalsPooledState.append_observations` (`update()`) took its column
#     list from a FRESH `build_agg_table`, which never emits `A*`, and so
#     dropped an existing one.
#
# Measured against the legacy engine before the fix, on pandas at atol=1e-8:
#   ExpandingMin  new_df 6.771 vs 1.516 | update 10.078 vs 1.516
#   ExpandingMax  new_df 14.515 vs 19.969
#   EWM           new_df 10.301 vs 10.272 | update 12.801 vs 11.031
# ---------------------------------------------------------------------------

ACCUMULATE_TFM_CASES = [
    ("expanding_min", lambda: ExpandingMin(groupby=["store"])),
    ("expanding_max", lambda: ExpandingMax(groupby=["store"])),
    ("ewm", lambda: ExponentiallyWeightedMean(alpha=0.3, groupby=["store"])),
    # global_ mode takes the un-partitioned `apply_accumulate` branch
    ("expanding_min_global", lambda: ExpandingMin(global_=True)),
    ("expanding_max_global", lambda: ExpandingMax(global_=True)),
    ("ewm_global", lambda: ExponentiallyWeightedMean(alpha=0.3, global_=True)),
    # controls: families that never touch an accumulate column. They matched
    # on every path even against the broken code, which is exactly why a
    # suite built only from them could not fail.
    ("rolling_mean_control", lambda: RollingMean(7, groupby=["store"])),
    ("expanding_mean_control", lambda: ExpandingMean(groupby=["store"])),
]


def _accumulate_panel(backend):
    return _panel(backend, n_series=12, n_times=40)


def _next_timestamp_rows(df):
    d = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    last = d["ds"].max()
    nxt = d.filter(pl.col("ds") == last).with_columns(
        (pl.col("ds") + pl.duration(days=1)).alias("ds")
    )
    return nxt if isinstance(df, pl.DataFrame) else nxt.to_pandas()


def _predict_via_path(engine, path, df, new_rows, tfms, statics, h):
    """Reach the pooled state through one of the three entry paths.

    ``plain``   -- fit_transform materializes the features (the path that was
                   always correct, since `feature_frame` settles accumulates).
    ``new_df``  -- `TimeSeries` rebuilt from history with no feature pass, so
                   `_initialize_lag_transform_states` is what must settle them.
    ``update``  -- `append_observations` rebuilds the aggregate table.
    """
    with pooled_engine(engine):
        fcst = MLForecast(
            models=[LinearRegression()], freq="1d", lags=[1], lag_transforms=tfms
        )
        fcst.fit(df, static_features=statics)
        if path == "plain":
            return fcst.predict(h)
        if path == "new_df":
            full = (
                pl.concat([df, new_rows])
                if isinstance(df, pl.DataFrame)
                else __import__("pandas").concat([df, new_rows], ignore_index=True)
            )
            return fcst.predict(h, new_df=full)
        if path == "update":
            fcst.update(new_rows)
            return fcst.predict(h)
        raise AssertionError(path)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("path", ["plain", "new_df", "update"])
@pytest.mark.parametrize(
    "label,make_tfm", ACCUMULATE_TFM_CASES, ids=[c[0] for c in ACCUMULATE_TFM_CASES]
)
def test_accumulate_families_agree_on_every_state_entry_path(
    backend, path, label, make_tfm
):
    """Can fail: reverting either half of the fix (dropping
    `state.ensure_accumulates(leaves)` from
    `core.py:_initialize_lag_transform_states`, or letting
    `append_observations` take its `base_cols` from the fresh
    `build_agg_table` again) makes the three accumulate families diverge on
    `new_df`/`update` while the two controls keep passing."""
    df = _accumulate_panel(backend)
    new_rows = _next_timestamp_rows(df)
    tfm = make_tfm()
    statics = ["store"]
    tfms = {1: [tfm]}
    a = _predict_via_path("numpy", path, df, new_rows, tfms, statics, 5)
    b = _predict_via_path("narwhals", path, df, new_rows, tfms, statics, 5)
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    np.testing.assert_allclose(
        a.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        b.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        atol=1e-8,
        err_msg=f"{label} diverges on the {path!r} path",
    )


@pytest.mark.parametrize("path", ["new_df", "update"])
def test_missing_accumulate_column_raises_rather_than_seeding_wrong(path):
    """The general fix, not just its two instances.

    `_make_seeds` and `trim_to_last` used to substitute the seed row's running
    accumulate value only *if* the `A` column happened to be there. That turns
    a missing prerequisite into a silently wrong number. They now raise. This
    reproduces the missing prerequisite directly -- delete the `A` columns off
    a settled state -- and requires a loud failure.
    """
    with pooled_engine("narwhals"):
        import narwhals as nw

        df = _accumulate_panel("polars")
        fcst = MLForecast(
            models=[LinearRegression()],
            freq="1d",
            lags=[1],
            lag_transforms={1: [ExpandingMin(groupby=["store"])]},
        )
        fcst.fit(df, static_features=["store"])
        key, pooled_tfms = next(iter(fcst.ts._get_pooled_tfms().items()))
        state = fcst.ts._pooled_states[key]
        assert state._accumulates, (
            "precondition: the fixture must actually require an accumulate "
            "column, or this test proves nothing"
        )
        agg = nw.from_native(state.agg, eager_only=True)
        accum_cols = [c for c in agg.columns if c in set(state._accumulates.values())]
        assert accum_cols, f"precondition: no A* column on {agg.columns}"
        n_ord_before = int(agg.get_column("ord").max()) + 1
        state.agg = agg.drop(accum_cols).to_native()
        state._seeds = None
        state._pending = []
        with pytest.raises(RuntimeError, match="missing accumulate column"):
            if path == "update":
                # strictly fewer ordinals than the state holds, or
                # `trim_to_last` short-circuits before reaching the seed step
                assert n_ord_before > 1, n_ord_before
                state.trim_to_last(n_ord_before - 1)
            else:
                state._make_seeds(pooled_tfms)


# ---------------------------------------------------------------------------
# Fix wave: composites over quantiles.
# ---------------------------------------------------------------------------

QUANTILE_COMPOSITE_CASES = [
    (
        "offset_rolling_quantile",
        Offset(RollingQuantile(0.5, 7, groupby=["store"]), 2),
    ),
    (
        "offset_expanding_quantile",
        Offset(ExpandingQuantile(0.5, groupby=["store"]), 2),
    ),
    (
        "offset_seasonal_rolling_quantile",
        Offset(SeasonalRollingQuantile(0.5, 7, 3, groupby=["store"]), 2),
    ),
    (
        "combine_quantile_quantile",
        Combine(
            RollingQuantile(0.5, 7, groupby=["store"]),
            RollingQuantile(0.9, 7, groupby=["store"]),
            operator.truediv,
        ),
    ),
    (
        "combine_quantile_rolling_mean",
        Combine(
            RollingQuantile(0.5, 7, groupby=["store"]),
            RollingMean(7, groupby=["store"]),
            operator.truediv,
        ),
    ),
    (
        "combine_rolling_mean_quantile",
        Combine(
            RollingMean(7, groupby=["store"]),
            RollingQuantile(0.5, 7, groupby=["store"]),
            operator.truediv,
        ),
    ),
    (
        "combine_offset_quantile_rolling_mean",
        Combine(
            Offset(RollingQuantile(0.5, 7, groupby=["store"]), 2),
            RollingMean(7, groupby=["store"]),
            operator.truediv,
        ),
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "label,tfm",
    QUANTILE_COMPOSITE_CASES,
    ids=[c[0] for c in QUANTILE_COMPOSITE_CASES],
)
def test_quantile_composites_engines_agree(backend, label, tfm):  # noqa: ARG001
    """Before the fix every one of these raised
    ``AttributeError: 'NoneType' object has no attribute 'alias'`` under
    narwhals while the numpy engine computed them -- composites forwarded
    ``_pooled_expr`` but not the quantile marker, so the expression branch got
    the base class's ``None``."""
    assert_engines_agree(_panel(backend), {1: [tfm]}, ["store"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_quantile_composite_predict_engines_agree(backend):
    """`latest_features` evaluates the same `feature_frame`, so the predict
    path must agree too -- and it exercises the seed/tail rebuild, where the
    materialized quantile column is recomputed over the tail rather than the
    full table."""
    tfms = {
        1: [
            Offset(RollingQuantile(0.5, 5, groupby=["store"]), 2),
            Combine(
                RollingQuantile(0.5, 5, groupby=["store"]),
                RollingMean(5, groupby=["store"]),
                operator.truediv,
            ),
        ]
    }
    a = _predict_with_engine("numpy", _panel(backend), tfms, ["store"], 5)
    b = _predict_with_engine("narwhals", _panel(backend), tfms, ["store"], 5)
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    np.testing.assert_allclose(
        a.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        b.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Fix wave: ordinary `partition_by` cardinalities must densify, not refuse.
#
# `should_densify`'s `k=4` refused any partition whose values are mutually
# exclusive over the calendar with cardinality >= 5 -- day-of-week (7)
# included, which is documented usage. Recalibrated to k=64 (plus an absolute
# 20M dense-row cap); see `_pooled_engine.should_densify`.
# ---------------------------------------------------------------------------


def _calendar_partition_panel(backend, n_series=20, n_times=90, n_groups=4):
    d = _panel(backend, n_series=n_series, n_times=n_times, n_groups=n_groups)
    d = d if isinstance(d, pl.DataFrame) else pl.from_pandas(d)
    d = d.with_columns(
        pl.col("ds").dt.weekday().cast(pl.Int64).alias("dow"),
        pl.col("ds").dt.month().cast(pl.Int64).alias("month"),
    )
    return d if backend == "polars" else d.to_pandas()


CALENDAR_PARTITION_CASES = [
    # cardinality 7: dense/sparse ratio is exactly 7, which k=4 refused
    ("dow_global", [RollingMean(7, min_samples=1, global_=True, partition_by=["dow"])]),
    (
        "dow_groupby",
        [RollingMean(7, min_samples=1, groupby=["store"], partition_by=["dow"])],
    ),
    ("dow_local", [RollingMean(7, min_samples=1, partition_by=["dow"])]),
    (
        "dow_expanding_minmax",
        [
            ExpandingMin(global_=True, partition_by=["dow"]),
            ExpandingMax(global_=True, partition_by=["dow"]),
        ],
    ),
    # cardinality 3 (the fixture spans 3 months): under the old k=4 too
    (
        "month_global",
        [RollingMean(3, min_samples=1, global_=True, partition_by=["month"])],
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "label,tfm_list",
    CALENDAR_PARTITION_CASES,
    ids=[c[0] for c in CALENDAR_PARTITION_CASES],
)
def test_calendar_cardinality_partition_by_computes(backend, label, tfm_list):  # noqa: ARG001
    """Reverting `should_densify`'s `k` to 4 makes every `dow_*` case here
    raise ``NotImplementedError: ... exceeds the should_densify size guard``
    under narwhals while the numpy engine computes them."""
    df = _calendar_partition_panel(backend)
    assert_engines_agree(df, {1: tfm_list}, ["store"], atol=1e-9)


def test_should_densify_accepts_calendar_cardinalities_and_refuses_pathological():
    """The guard's own arithmetic, stated as the property it must have.

    For a partition whose values are mutually exclusive over the calendar,
    dense/sparse == cardinality exactly, so the bound is a bound on the
    partition's cardinality.
    """
    from mlforecast._pooled_engine import _MAX_DENSE_ROWS, should_densify

    n_calendar = 730
    for cardinality in (2, 7, 12, 24, 31, 53, 64):
        n_sparse = n_calendar  # every timestamp observed by exactly one bucket
        assert should_densify(cardinality, n_calendar, n_sparse), (
            f"cardinality {cardinality} (dense/sparse == {cardinality}) must "
            "densify -- it is ordinary calendar partitioning"
        )
    for cardinality in (65, 365, 10_000):
        assert not should_densify(cardinality, n_calendar, n_calendar), (
            f"cardinality {cardinality} exceeds the ratio bound and must be "
            "refused rather than materialized"
        )
    # the absolute memory cap bites even when the ratio is comfortable
    assert not should_densify(_MAX_DENSE_ROWS, 2, _MAX_DENSE_ROWS), (
        "a dense grid past the absolute row cap must be refused however "
        "proportionate it is to the sparse data"
    )


# ---------------------------------------------------------------------------
# Fix wave, finding 4: the LookupLag-mixed-with-other-families refusal STAYS
# (one aggregate table cannot be sparse for LookupLag and dense for everything
# else at the same time, and a dual-representation state would have to double
# the whole Task 9 predict path). What changed is the message: it now names a
# workaround that PRESERVES the buckets, instead of telling the user to change
# `partition_by` -- which changes what is being computed.
# ---------------------------------------------------------------------------


def _holiday_panel(backend, n_series=8, n_times=60):
    d = _panel(backend, n_series=n_series, n_times=n_times)
    d = d if isinstance(d, pl.DataFrame) else pl.from_pandas(d)
    d = d.with_columns((pl.col("ds").dt.day() % 5 == 0).cast(pl.Int64).alias("holiday"))
    # a byte-identical duplicate: same buckets, different pooled state key
    d = d.with_columns(pl.col("holiday").alias("holiday_lookup"))
    return d if backend == "polars" else d.to_pandas()


def test_lookup_lag_mixed_with_other_families_refusal_names_a_real_workaround():
    """The refusal must be actionable. Pins that it offers the duplicate
    partition column, not "use a different partition_by" (which would change
    the buckets and therefore the answer)."""
    df = _holiday_panel("polars")
    with pooled_engine("narwhals"):
        fcst = MLForecast(
            models=[LinearRegression()],
            freq="1d",
            lags=[1],
            lag_transforms={
                1: [
                    LookupLag(partition_by=["holiday"]),
                    RollingMean(3, min_samples=1, partition_by=["holiday"]),
                ]
            },
        )
        with pytest.raises(NotImplementedError) as excinfo:
            fcst.preprocess(df, static_features=[], dropna=False)
    msg = str(excinfo.value)
    assert "DUPLICATE" in msg, msg
    assert "'holiday'" in msg, msg
    assert "MLFORECAST_POOLED_ENGINE=numpy" in msg, msg


@pytest.mark.parametrize("backend", BACKENDS)
def test_lookup_lag_duplicate_column_workaround_matches_legacy_mixed_state(backend):
    """The workaround the refusal recommends must actually reproduce what the
    legacy engine computes for the refused configuration -- otherwise the
    message is sending users somewhere that changes their numbers.

    Legacy computes the MIXED state (both transforms on ``holiday``);
    narwhals computes the SPLIT one (LookupLag on the duplicate column). The
    two must agree on values. Feature NAMES differ (they follow the column),
    so this compares the arrays directly rather than through
    ``assert_engines_agree``.
    """
    df = _holiday_panel(backend)
    mixed = {
        1: [
            LookupLag(partition_by=["holiday"]),
            RollingMean(3, min_samples=1, partition_by=["holiday"]),
        ]
    }
    split = {
        1: [
            LookupLag(partition_by=["holiday_lookup"]),
            RollingMean(3, min_samples=1, partition_by=["holiday"]),
        ]
    }
    a = _preprocess_with_engine("numpy", df, mixed, [])
    b = _preprocess_with_engine("narwhals", df, split, [])
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    a, b = a.sort("unique_id", "ds"), b.sort("unique_id", "ds")
    skip = ("unique_id", "ds", "y", "store", "holiday", "holiday_lookup", "lag1")
    a_cols = sorted(c for c in a.columns if c not in skip)
    b_cols = sorted(c for c in b.columns if c not in skip)
    assert len(a_cols) == len(b_cols) == 2, (a_cols, b_cols)
    for ca, cb in zip(a_cols, b_cols):
        np.testing.assert_allclose(
            a[ca].cast(pl.Float64).to_numpy(),
            b[cb].cast(pl.Float64).to_numpy(),
            atol=1e-10,
            equal_nan=True,
            err_msg=f"{ca} (legacy, mixed state) vs {cb} (narwhals, split state)",
        )


# ---------------------------------------------------------------------------
# The frozen centring reference (`NarwhalsPooledState._kref`) across the two
# in-process paths that REWRITE the aggregate table. Both use `ExpandingStd`
# on purpose: its window reaches back to ordinal 0 through the `E`-prefixed
# cumulative columns, so it is the only std family that actually reads a
# `trim_to_last` seed row's carried moments and the only one whose result
# depends on every appended row having been centred on the SAME reference.
# `RollingStd(5)` never reaches either and cannot fail these.
# ---------------------------------------------------------------------------


def _exact_std_of_all(df_pl):
    from .test_pooled_narwhals import _exact_sample_std

    _, per_ord = _values_by_ordinal(df_pl)
    return _exact_sample_std([v for row in per_ord for v in row])


def _fit_narwhals(df, tfms, **fit_kw):
    fcst = MLForecast(models=[LinearRegression()], freq="1d", lag_transforms=tfms)
    fcst.fit(df, static_features=["store"], **fit_kw)
    return fcst


def _latest_pooled_feature(fcst):
    """``(feature value at the next ordinal, aggregate rows before the call)``.

    One bucket in these fixtures, so every series carries the same value.
    ``latest_features`` reduces ``state.agg`` to the predict tail, hence the
    row count is read first.
    """
    ts = fcst.ts
    key = next(iter(ts._pooled_states))
    state = ts._pooled_states[key]
    n_rows = len(nw.from_native(state.agg, eager_only=True))
    feats = state.latest_features(ts._get_pooled_tfms()[key], len(ts.uids))
    assert len(feats) == 1, f"expected one pooled feature, got {sorted(feats)}"
    arr = np.asarray(next(iter(feats.values())), dtype=float)
    assert (arr == arr[0]).all(), "one bucket: every series must share the value"
    return float(arr[0]), n_rows


@pytest.mark.parametrize("backend", BACKENDS)
def test_trim_to_last_keeps_the_expanding_std_exact(backend):
    """`keep_last_n` drops a prefix; the frozen reference must survive it.

    Two things are pinned. (1) The reference is NEVER re-derived: it is a
    per-bucket mean, so recomputing it over the retained suffix would move it
    and leave every surviving `qK` centred on a value that no longer matches
    the new ones. (2) `trim_to_last` must substitute the seed row's `sK`/`qK`
    with their own `EsK`/`EqK` cumulative values, exactly as it already did
    for `s`/`c` -- that substitution loop is driven by `_PREFIX_AGGS`, and
    with the old hard-coded `("s", "c", "q")` list the two shifted moments
    are silently skipped (the loop only substitutes columns it finds), so the
    trimmed state reports the std of the RETAINED WINDOW instead of the whole
    history.
    """
    df_pl = _large_magnitude_panel(n_times=40)
    df = df_pl if backend == "polars" else df_pl.to_pandas()
    tfms = {1: [ExpandingStd(groupby=["store"])]}
    want = _exact_std_of_all(df_pl)

    with pooled_engine("narwhals"):
        from .test_pooled_narwhals import _rel_err

        got_auto, rows_auto = _latest_pooled_feature(_fit_narwhals(df, tfms))
        got_trim, rows_trim = _latest_pooled_feature(
            _fit_narwhals(df, tfms, keep_last_n=8)
        )
        # `fit` infers a `keep_last_n` of its own, so BOTH of these are
        # trimmed -- at two different depths, from a 40-ordinal history.
        # Whichever depth, the expanding std must still be the one over the
        # WHOLE history, which is the property the seed row carries.
        assert rows_auto < 40 and rows_trim < 40 and rows_auto != rows_trim, (
            "precondition failed: no prefix was dropped, or both fits kept "
            f"the same rows ({rows_auto} / {rows_trim} of 40 ordinals)"
        )
        for label, got in (("auto keep_last_n", got_auto), ("keep_last_n=8", got_trim)):
            err = _rel_err(got, want)
            assert err < _EXACT_STD_TOL, (
                f"{label}: expanding std {got!r} is {err:g} off the exact "
                f"value {float(want)!r}"
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_update_keeps_the_expanding_std_exact(backend):
    """`update()` appends a suffix; the frozen reference must survive it.

    `append_observations` re-aggregates only the NEW rows. It must centre
    them on the reference the fit-time rows already used -- computing a fresh
    one for the new rows (which is what happens if `kref` is not threaded
    through, or if `_extend_kref` is allowed to overwrite an existing
    bucket's entry) leaves the bucket's prefix sums adding `sum((y-K)**2)`
    terms taken about two different centres, which is not any variance at
    all.
    """
    df_pl = _large_magnitude_panel(n_times=40)
    dates = sorted(set(df_pl["ds"].to_list()))
    head_pl = df_pl.filter(pl.col("ds") <= dates[-6])
    tail_pl = df_pl.filter(pl.col("ds") > dates[-6])
    assert len(tail_pl) > 0 and len(head_pl) > 0
    head = head_pl if backend == "polars" else head_pl.to_pandas()
    tail = tail_pl if backend == "polars" else tail_pl.to_pandas()
    tfms = {1: [ExpandingStd(groupby=["store"])]}
    want = _exact_std_of_all(df_pl)

    with pooled_engine("narwhals"):
        from .test_pooled_narwhals import _rel_err

        fcst = _fit_narwhals(head, tfms)
        key = next(iter(fcst.ts._pooled_states))
        k_before = (
            nw.from_native(fcst.ts._pooled_states[key]._kref, eager_only=True)
            .get_column("K")
            .to_numpy()
            .copy()
        )
        assert (np.abs(k_before) > 1e10).all(), (
            "precondition failed: the fixture is not at a magnitude where the "
            "centring reference matters"
        )
        fcst.update(tail)
        k_after = (
            nw.from_native(fcst.ts._pooled_states[key]._kref, eager_only=True)
            .get_column("K")
            .to_numpy()
        )
        np.testing.assert_array_equal(
            k_after, k_before, err_msg="update() moved a frozen reference"
        )
        got, _ = _latest_pooled_feature(fcst)
    err = _rel_err(got, want)
    assert err < _EXACT_STD_TOL, (
        f"expanding std after update() is {err:g} off the exact value "
        f"{float(want)!r} (got {got!r})"
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("time_agg", ["mean", "sum", "count", "min", "max"])
def test_time_agg_std_predict_engines_agree(backend, time_agg):
    """Recursive predict for a ``time_agg`` std family.

    This is the path where each step's PENDING row has to derive its own
    ``sK__<agg>``/``qK__<agg>`` from the bucket's frozen reference for that
    family (``_pending_agg_frame`` -> ``_derive_time_agg_family``). Every
    ``time_agg`` is swept because each has its own reference on its own
    scale -- ``count`` in particular holds small integers, and centring it on
    the target's magnitude would manufacture the very cancellation the
    reference exists to remove.
    """
    df = _panel(backend, n_series=12, n_times=40, n_groups=2)
    tfms = {1: [RollingStd(5, min_samples=2, groupby=["store"], time_agg=time_agg)]}
    a = _predict_with_engine("numpy", df, tfms, ["store"], 5)
    b = _predict_with_engine("narwhals", df, tfms, ["store"], 5)
    a = (a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)).sort(
        "unique_id", "ds"
    )
    b = (b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)).sort(
        "unique_id", "ds"
    )
    np.testing.assert_allclose(
        a["LinearRegression"].to_numpy(), b["LinearRegression"].to_numpy(), atol=1e-9
    )
