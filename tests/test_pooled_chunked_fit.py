"""G4 guards for computing the fit block a column range at a time (PR 4).

At fit the engine materialises a ``(n_buckets, width)`` block per channel and
several same-shaped temporaries inside ``combine``, scatters ``n_rows`` values
out of the result and drops the rest. ``PooledState.fit_values`` instead walks
the calendar in chunks and scatters each one, which bounds that transient at
``n_buckets x (lookback + chunk)`` however wide the calendar is.

It applies only to kernels whose window reach is bounded *and* whose inners
carry nothing between calls -- ``Rolling*``, ``SeasonalRolling*``, ``Lag``, the
set ``_PooledKernel.lookback()`` is defined for. ``Expanding*``/``EWM`` re-derive
from ordinal 0, so a chunk would need the whole prefix and nothing is bounded.

Chunking cannot be bit-identical for the averaging kernels: coreforecast's
rolling path carries a running accumulator, so starting it at a different offset
sums the same values through different-magnitude partials. That is the same
float-associativity noise G2.1 documents for trimming. What *is* exact is the
part anything downstream depends on:

* **G4.1 three-tier equivalence** -- the NaN mask is exact for every kernel;
  ``Min``/``Max``/``Lag`` are bit-identical outright (they read order-independent
  channels); only the mean/std *values* move, by ~1e-14 relative.
* **G4.2 dropna row identity** -- ``_transform``'s ``keep_rows`` is driven by
  ``np.isnan``, so an exact mask means chunking cannot change which rows survive
  ``dropna``, which series get dropped, or what the model trains on. This is the
  guarantee that makes the value drift acceptable.
* **G4.3 primes_state is honest** -- ``transform`` leaves inner state behind iff
  the kernel says it does. ``_initialize_lag_transform_states`` skips the ones
  that say no, so a coreforecast release that started caching inside a rolling
  ``transform`` has to fail here rather than silently unprime.
* **G4.4 lookback matches retention** -- ``lookback()`` and
  ``_pooled_retention`` are the same distance measured from different origins,
  so they must agree wherever both are finite.
* **G4.5 the cap gates** -- a state above the cell budget chunks, one below it
  does not. Small panels keeping the single-chunk path byte for byte is what
  makes this change opt-in by size.
* **G4.6 ineligible kernels never chunk** -- including on a key shared with an
  eligible one.
* **G4.7 boundaries** -- a lookback longer than the calendar, a chunk of one, a
  lag past the end, buckets with one row and with none, and seasonal strides
  that do not divide the chunk.
* **G4.8 the transient is bounded** -- asserted against the block size, not
  merely reported, so a regression to the full-width path fails here.
"""

import operator
import pickle
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import mlforecast.lag_transforms as L
import mlforecast.pooled as pooled_mod
from mlforecast import MLForecast
from mlforecast.pooled import PooledState, _chunk_cols, base_channels, get_kernel

ID, TIME, TARGET = "unique_id", "ds", "y"
_D = pd.Timestamp("2020-01-01")


@pytest.fixture
def force_chunk(monkeypatch):
    """Shrink the cell budget so ordinary test panels take the chunked path."""

    def _set(cells):
        monkeypatch.setattr(pooled_mod, "_MAX_CHUNK_CELLS", cells)

    return _set


def _panel(n_series=24, length=60, seed=0):
    """Ragged starts, interior gaps, aligned ends -- what the engine allows.

    Aligned ends are required (`_check_aligned_ends`); the gaps are what make
    empty cells, and so the ``count`` channel, actually exercised.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        start = int(rng.integers(0, 8))
        for t in range(start, length):
            if t < length - 1 and rng.random() < 0.08:
                continue
            rows.append(
                (
                    f"s{i}",
                    _D + pd.Timedelta(days=int(t)),
                    float(rng.normal(10, 3)),
                    f"g{i % 4}",
                    str(rng.integers(0, 3)),
                )
            )
    return pd.DataFrame(rows, columns=[ID, TIME, TARGET, "brand", "promo"])


def _feature(df, transform, cap, dropna=False):
    """The fit-time feature column, computed with ``_MAX_CHUNK_CELLS = cap``."""
    prev = pooled_mod._MAX_CHUNK_CELLS
    pooled_mod._MAX_CHUNK_CELLS = cap
    try:
        fcst = MLForecast(
            freq="D", models=[LinearRegression()], lag_transforms={1: [transform]}
        )
        out = fcst.preprocess(
            df,
            id_col=ID,
            time_col=TIME,
            target_col=TARGET,
            static_features=["brand"],
            dropna=dropna,
            validate_data=False,
        )
    finally:
        pooled_mod._MAX_CHUNK_CELLS = prev
    known = {ID, TIME, TARGET, "brand", "promo"}
    col = [c for c in out.columns if c not in known][0]
    return out, col


_UNCHUNKED = 1 << 30


def _bounded():
    """The chunkable kernels, as ``(id, factory)`` taking the pooling mode."""
    return [
        ("rolling_mean", lambda **kw: L.RollingMean(window_size=5, **kw)),
        ("rolling_std", lambda **kw: L.RollingStd(window_size=5, **kw)),
        ("rolling_min", lambda **kw: L.RollingMin(window_size=5, **kw)),
        ("rolling_max", lambda **kw: L.RollingMax(window_size=5, **kw)),
        (
            "seasonal_rolling_mean",
            lambda **kw: L.SeasonalRollingMean(season_length=7, window_size=3, **kw),
        ),
        (
            "seasonal_rolling_std",
            lambda **kw: L.SeasonalRollingStd(season_length=7, window_size=3, **kw),
        ),
        (
            "seasonal_rolling_min",
            lambda **kw: L.SeasonalRollingMin(season_length=7, window_size=3, **kw),
        ),
    ]


_BOUNDED_IDS = [n for n, _ in _bounded()]
#: kernels reading order-independent channels, so chunking cannot move them
_EXACT_VALUE_KERNELS = ("min", "max", "lag")


def _assert_equivalent(a, b, name):
    """Mask exact always; values exact for min/max; float noise for the rest.

    ``std`` needs an absolute tolerance as well: it is computed as
    ``sqrt((s2 - s1**2/n) / (n-1))``, so on a bucket whose values are nearly
    identical the subtraction cancels and a last-ulp move in either moment is a
    large *relative* move in a near-zero result. The absolute error stays at the
    same ~1e-11 as everywhere else, which is what the guarantee is about.
    """
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=f"{name}:mask")
    if any(k in name for k in _EXACT_VALUE_KERNELS):
        np.testing.assert_array_equal(a, b, err_msg=f"{name}:values")
    else:
        atol = 1e-9 if "std" in name else 0.0
        np.testing.assert_allclose(
            a, b, rtol=1e-12, atol=atol, equal_nan=True, err_msg=name
        )


# --------------------------------------------------------------------------- #
# G4.1 -- mask exact always, values exact for min/max, ~1e-14 for mean/std.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mode",
    [
        {"global_": True},
        {"groupby": ["brand"]},
        {"partition_by": ["promo"]},
        {"global_": True, "partition_by": ["promo"]},
        {"groupby": ["brand"], "partition_by": ["promo"]},
    ],
    ids=[
        "global",
        "groupby",
        "local_partition",
        "global_partition",
        "groupby_partition",
    ],
)
@pytest.mark.parametrize("name,make", _bounded(), ids=_BOUNDED_IDS)
def test_g4_1_chunked_matches_unchunked(name, make, mode):
    """Asserted in tiers, strongest first, so a regression says which broke."""
    df = _panel()
    ref, col = _feature(df, make(**mode), _UNCHUNKED)
    got, _ = _feature(df, make(**mode), 3)
    _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), name)


@pytest.mark.parametrize("time_agg", [None, "sum", "mean", "min"])
def test_g4_1_time_agg_views_chunk_consistently(time_agg):
    """`_collapse` is elementwise, so a collapsed column slice must equal the
    slice of the collapsed view -- the property the chunked path relies on to
    avoid building a full-width view at fit."""
    df = _panel()
    ref, col = _feature(
        df,
        L.RollingMean(window_size=5, groupby=["brand"], time_agg=time_agg),
        _UNCHUNKED,
    )
    got, _ = _feature(
        df, L.RollingMean(window_size=5, groupby=["brand"], time_agg=time_agg), 3
    )
    _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), "rolling_mean")


@pytest.mark.parametrize("min_samples", [None, 1, 8])
def test_g4_1_min_samples_gating_is_unaffected(min_samples):
    """`k` is evaluated on absolute ordinals, so the gate cannot shift with the
    chunk origin. A relative `k` would silently blank or admit cells."""
    df = _panel()
    kw = dict(window_size=5, groupby=["brand"], min_samples=min_samples)
    ref, col = _feature(df, L.RollingMean(**kw), _UNCHUNKED)
    got, _ = _feature(df, L.RollingMean(**kw), 3)
    np.testing.assert_array_equal(
        np.isnan(ref[col].to_numpy()), np.isnan(got[col].to_numpy())
    )


def test_g4_1_wrappers_chunk_like_their_leaves():
    """`Offset`/`Combine` resolve to pooled leaves and must follow them."""
    df = _panel()
    for tfm_factory in (
        lambda: L.Offset(L.RollingMean(window_size=5, groupby=["brand"]), 2),
        lambda: L.Combine(
            L.RollingMean(window_size=5, groupby=["brand"]),
            L.RollingMean(window_size=3, groupby=["brand"]),
            operator.truediv,
        ),
    ):
        ref, col = _feature(df, tfm_factory(), _UNCHUNKED)
        got, _ = _feature(df, tfm_factory(), 3)
        _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), "wrapper")


# --------------------------------------------------------------------------- #
# G4.2 -- an exact mask means dropna keeps exactly the same rows.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,make", _bounded(), ids=_BOUNDED_IDS)
def test_g4_2_dropna_keeps_identical_rows(name, make):
    """The concrete downstream consequence of the mask being exact.

    If chunking could move a value across the null boundary it would change the
    training set, which no tolerance on the values themselves would catch.
    """
    df = _panel()
    ref, _ = _feature(df, make(groupby=["brand"]), _UNCHUNKED, dropna=True)
    got, _ = _feature(df, make(groupby=["brand"]), 3, dropna=True)
    assert len(ref) == len(got), name
    np.testing.assert_array_equal(ref[ID].to_numpy(), got[ID].to_numpy())
    np.testing.assert_array_equal(ref[TIME].to_numpy(), got[TIME].to_numpy())


# --------------------------------------------------------------------------- #
# G4.3 / G4.4 -- the kernel metadata the chunking decision rests on.
# --------------------------------------------------------------------------- #
def _all_kernel_specs():
    return [
        ("RollingMean", L.RollingMean(window_size=4, global_=True)),
        ("RollingStd", L.RollingStd(window_size=4, global_=True)),
        ("RollingMin", L.RollingMin(window_size=4, global_=True)),
        ("RollingMax", L.RollingMax(window_size=4, global_=True)),
        (
            "SeasonalRollingMean",
            L.SeasonalRollingMean(season_length=3, window_size=2, global_=True),
        ),
        (
            "SeasonalRollingStd",
            L.SeasonalRollingStd(season_length=3, window_size=2, global_=True),
        ),
        (
            "SeasonalRollingMin",
            L.SeasonalRollingMin(season_length=3, window_size=2, global_=True),
        ),
        (
            "SeasonalRollingMax",
            L.SeasonalRollingMax(season_length=3, window_size=2, global_=True),
        ),
        ("ExpandingMean", L.ExpandingMean(global_=True)),
        ("ExpandingStd", L.ExpandingStd(global_=True)),
        ("ExpandingMin", L.ExpandingMin(global_=True)),
        ("ExpandingMax", L.ExpandingMax(global_=True)),
        ("EWM", L.ExponentiallyWeightedMean(alpha=0.5, global_=True)),
        ("RollingQuantile", L.RollingQuantile(p=0.5, window_size=4, global_=True)),
        ("ExpandingQuantile", L.ExpandingQuantile(p=0.5, global_=True)),
        (
            "SeasonalRollingQuantile",
            L.SeasonalRollingQuantile(
                p=0.5, season_length=3, window_size=2, global_=True
            ),
        ),
        ("LookupLag", L.LookupLag(partition_by=["promo"])),
        ("Lag", L.Lag(1)),
    ]


def _state_for(kernel, tfm, n_buckets=5, width=16):
    rng = np.random.default_rng(0)
    n = 200
    return PooledState.build(
        mode="global",
        group_cols=[],
        partition_cols=[],
        bucket_id_by_row=rng.integers(0, n_buckets, n),
        ordinal_by_row=rng.integers(0, width, n),
        y=rng.normal(10, 1, n),
        n_buckets=n_buckets,
        n_ordinals=width,
        series_bucket_id=np.zeros(n_buckets, dtype=np.int64),
        needed=base_channels(kernel.channels, tfm.time_agg),
        needs_rows=getattr(kernel, "needs_rows", False),
    )


@pytest.mark.parametrize(
    "name,tfm", _all_kernel_specs(), ids=[n for n, _ in _all_kernel_specs()]
)
def test_g4_3_primes_state_matches_reality(name, tfm):
    """`transform` must leave inner state behind exactly when it claims to.

    The priming skip in `_initialize_lag_transform_states` trusts this flag; if
    a coreforecast release started caching inside a rolling `transform`, the
    skip would silently leave it unprimed and this is where that shows up.
    """
    tfm = tfm._set_core_tfm(2) if hasattr(tfm, "_set_core_tfm") else tfm
    kernel = get_kernel(tfm)
    inner = kernel.make_inner()
    state = _state_for(kernel, tfm)
    before = pickle.dumps(inner)
    state.transform(kernel, inner)
    changed = pickle.dumps(inner) != before
    assert changed == kernel.primes_state, name


@pytest.mark.parametrize(
    "name,tfm", _all_kernel_specs(), ids=[n for n, _ in _all_kernel_specs()]
)
def test_g4_4_lookback_agrees_with_retention(name, tfm):
    """Same distance, measured from the output ordinal vs the last stored column.

    Pinned together so a change to one has to confront the other; retention is
    also finite for Expanding*/EWM, which is why it cannot be the chunk
    predicate on its own.
    """
    tfm = tfm._set_core_tfm(2) if hasattr(tfm, "_set_core_tfm") else tfm
    kernel = get_kernel(tfm)
    lookback = kernel.lookback()
    if lookback is None:
        return
    assert lookback == tfm._pooled_retention, name
    assert not kernel.primes_state, f"{name} is chunkable but primes state"


# --------------------------------------------------------------------------- #
# G4.5 / G4.6 -- when the chunked path is taken, and when it is not.
# --------------------------------------------------------------------------- #
def test_g4_5_small_states_keep_the_single_chunk_path():
    """The early return is what keeps ordinary panels byte for byte."""
    assert _chunk_cols(n_buckets=1, width=730, lookback=7) == 730
    assert _chunk_cols(n_buckets=500, width=730, lookback=7) == 730


def test_g4_5_large_states_chunk_within_the_budget():
    """The source block a chunk reads is `lookback + chunk` columns wide."""
    for n_buckets, width, lookback in [(25_000, 730, 7), (97_426, 730, 7)]:
        chunk = _chunk_cols(n_buckets, width, lookback)
        assert chunk < width
        assert n_buckets * (lookback + chunk) <= 4 * pooled_mod._MAX_CHUNK_CELLS


def test_g4_5_chunk_never_falls_below_the_lookback():
    """Holding the chunk at the lookback keeps the re-read rework under 2x."""
    for lookback in (1, 7, 160):
        assert _chunk_cols(10**7, 10**4, lookback) >= lookback


@pytest.mark.parametrize(
    "name,tfm",
    [
        ("expanding_mean", L.ExpandingMean(groupby=["brand"])),
        ("ewm", L.ExponentiallyWeightedMean(alpha=0.5, groupby=["brand"])),
        (
            "rolling_quantile",
            L.RollingQuantile(p=0.5, window_size=5, groupby=["brand"]),
        ),
        ("lookup_lag", L.LookupLag(partition_by=["promo"])),
    ],
    ids=["expanding_mean", "ewm", "rolling_quantile", "lookup_lag"],
)
def test_g4_6_ineligible_kernels_never_chunk(name, tfm, monkeypatch):
    """`lookback() is None` has to keep them on the whole-block path.

    Expanding* would need the prefix from ordinal 0 in every chunk, so chunking
    it bounds nothing; the row kernels do not read the channel blocks at all.
    """
    calls = []
    original = PooledState._transform_range
    monkeypatch.setattr(
        PooledState,
        "_transform_range",
        lambda self, k, i, a, b: (calls.append(name), original(self, k, i, a, b))[1],
    )
    df = _panel()
    _feature(df, tfm, 3)
    assert calls == [], f"{name} took the chunked path"


def test_g4_6_mixed_key_chunks_only_the_eligible_leaf(monkeypatch):
    """Two leaves share one state; eligibility is per leaf, not per state."""
    calls = []
    original = PooledState._transform_range
    monkeypatch.setattr(
        PooledState,
        "_transform_range",
        lambda self, k, i, a, b: (
            calls.append(type(k).__name__),
            original(self, k, i, a, b),
        )[1],
    )
    df = _panel()
    prev = pooled_mod._MAX_CHUNK_CELLS
    pooled_mod._MAX_CHUNK_CELLS = 3
    try:
        fcst = MLForecast(
            freq="D",
            models=[LinearRegression()],
            lag_transforms={
                1: [
                    L.RollingMean(window_size=5, groupby=["brand"]),
                    L.ExpandingMean(groupby=["brand"]),
                ]
            },
        )
        fcst.preprocess(
            df,
            id_col=ID,
            time_col=TIME,
            target_col=TARGET,
            static_features=["brand"],
            dropna=False,
            validate_data=False,
        )
    finally:
        pooled_mod._MAX_CHUNK_CELLS = prev
    assert calls, "the rolling leaf should have chunked"
    assert set(calls) == {"RollingMeanK"}, set(calls)


# --------------------------------------------------------------------------- #
# G4.7 -- boundaries.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window_size", [1, 2, 40, 80])
def test_g4_7_lookback_around_and_beyond_the_calendar(window_size):
    """A window longer than the calendar clamps `lo` to 0 for every chunk."""
    df = _panel(length=40)
    kw = dict(window_size=window_size, groupby=["brand"], min_samples=1)
    ref, col = _feature(df, L.RollingMean(**kw), _UNCHUNKED)
    got, _ = _feature(df, L.RollingMean(**kw), 3)
    _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), "rolling_mean")


@pytest.mark.parametrize("season_length", [5, 7])
@pytest.mark.parametrize("window_size", [2, 3])
@pytest.mark.parametrize("cap", [3, 40, 400])
def test_g4_7_seasonal_strides_need_no_phase_alignment(season_length, window_size, cap):
    """coreforecast's seasonal window is origin-invariant once complete.

    So a chunk boundary landing mid-season is fine and the slice needs no
    rounding to a multiple of `season_length`.
    """
    df = _panel(length=50)
    kw = dict(
        season_length=season_length,
        window_size=window_size,
        groupby=["brand"],
        min_samples=1,
    )
    ref, col = _feature(df, L.SeasonalRollingMean(**kw), _UNCHUNKED)
    got, _ = _feature(df, L.SeasonalRollingMean(**kw), cap)
    _assert_equivalent(ref[col].to_numpy(), got[col].to_numpy(), "rolling_mean")


def test_g4_7_sparse_buckets():
    """A bucket with one observation, and a bucket with none in a chunk.

    Empty chunks are skipped by the row index rather than transformed, so a
    calendar stretch nobody observed must not shift the rows that follow.
    """
    rows = [("s0", _D, 1.0, "g0", "0"), ("s1", _D, 2.0, "g1", "0")]
    for t in (1, 2, 3, 20, 21):
        rows += [
            ("s0", _D + pd.Timedelta(days=t), float(t), "g0", "0"),
            ("s1", _D + pd.Timedelta(days=t), float(t * 2), "g1", "0"),
        ]
    df = pd.DataFrame(rows, columns=[ID, TIME, TARGET, "brand", "promo"])
    kw = dict(window_size=3, groupby=["brand"], min_samples=1)
    ref, col = _feature(df, L.RollingMean(**kw), _UNCHUNKED)
    got, _ = _feature(df, L.RollingMean(**kw), 3)
    np.testing.assert_allclose(
        ref[col].to_numpy(), got[col].to_numpy(), rtol=1e-12, equal_nan=True
    )


# --------------------------------------------------------------------------- #
# G4.8 -- the transient is bounded, not merely smaller.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel_name", ["rolling_mean", "rolling_std"])
def test_g4_8_fit_transient_is_bounded_by_the_chunk(kernel_name, force_chunk):
    """Asserted against the chunk, so a regression to full width fails here.

    The unchunked path peaks at several multiples of the *whole* block; chunked
    it must stay within a small multiple of one chunk's source block.
    """
    # deliberately sparse -- many buckets over a wide calendar with far fewer
    # rows, which is the partitioned shape the block is oversized for and the
    # only one where the O(n_rows) floor does not swamp the measurement
    n_buckets, width, lookback, n = 1500, 500, 5, 40_000
    tfm = (
        L.RollingMean(window_size=lookback, global_=True)
        if kernel_name == "rolling_mean"
        else L.RollingStd(window_size=lookback, global_=True)
    )
    tfm = tfm._set_core_tfm(1)
    kernel = get_kernel(tfm)
    rng = np.random.default_rng(0)
    bid = rng.integers(0, n_buckets, n)
    ordi = rng.integers(0, width, n)
    state = PooledState.build(
        mode="global",
        group_cols=[],
        partition_cols=[],
        bucket_id_by_row=bid,
        ordinal_by_row=ordi,
        y=rng.normal(10, 1, n),
        n_buckets=n_buckets,
        n_ordinals=width,
        series_bucket_id=np.zeros(n_buckets, dtype=np.int64),
        needed=base_channels(kernel.channels, tfm.time_agg),
    )
    state._fit_row_bid, state._fit_row_ord = bid, ordi

    def peak_of(cap):
        force_chunk(cap)
        tracemalloc.start()
        values = state.fit_values(kernel, kernel.make_inner())
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak, values

    unchunked_peak, reference = peak_of(_UNCHUNKED)
    force_chunk(4096)  # `_chunk_cols` reads the cap, so set it before asking
    chunk = _chunk_cols(n_buckets, width, kernel.lookback())
    assert chunk < width, "the panel must be big enough to chunk"
    chunked_peak, values = peak_of(4096)

    _assert_equivalent(reference, values, kernel_name)

    # the O(n_rows) row index and the output vector are unavoidable either way;
    # what must fall is the part that scales with the calendar
    floor = reference.nbytes + state._fit_row_ord.nbytes + state._fit_row_bid.nbytes
    block_bytes = n_buckets * width * 8
    assert chunked_peak - floor < block_bytes, (
        f"chunked transient {(chunked_peak - floor) / 1e6:.1f}MB is not below one "
        f"full block ({block_bytes / 1e6:.1f}MB)"
    )
    assert chunked_peak < 0.6 * unchunked_peak, (
        f"chunked peak {chunked_peak / 1e6:.1f}MB vs unchunked "
        f"{unchunked_peak / 1e6:.1f}MB"
    )
