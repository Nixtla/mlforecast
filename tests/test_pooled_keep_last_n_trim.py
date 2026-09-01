"""G2 guards for trimming pooled states under ``keep_last_n`` (PR 2).

Once ``keep_last_n`` is resolved, a pooled state whose transforms are all
finite-window drops its unused history prefix, in parity with the
``TimeSeries.ga`` trim. A single byte-identical state hash cannot survive a
trim (it deliberately drops aggregate prefixes), so these guards assert the
*contract* instead:

* **G2.1 prediction-equality** -- predictions from a trimmed model match an
  untrimmed model (the dropped prefix never enters a finite window, so it
  cannot move a forecast). Covers the retention floor (``keep_last_n`` smaller
  than a window). Compared with a tolerance, not byte-for-byte: the rolling
  fast path cumsums the *whole* aggregate vector, so a shorter (trimmed) vector
  accumulates the window sum through different-magnitude partials -- genuine
  float associativity noise (~1e-13), not a regression. The byte-identical
  guarantee lives at the state level (G2.2/G2.3).
* **G2.2 trim == fit-on-slice** -- a trimmed state is byte-identical to a fresh
  state fit on only the retained tail of the input, including after a
  follow-up ``update()`` (exercises the rebuilt ``_ts_aggs`` and the two
  append conventions).
* **G2.3 suffix invariant** -- each retained aggregate vector equals the tail
  of the untrimmed vector and the retained calendar length equals
  ``max(keep_last_n, W_state)``.
* **G2.4 non-trim assertion** -- a state containing any Expanding*/EWM
  transform keeps full history (pooled has no carried accumulator, so it must
  recompute over everything).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast.forecast import MLForecast
from mlforecast.lag_transforms import (
    Combine,
    ExpandingMean,
    ExponentiallyWeightedMean,
    Offset,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingStd,
)

ID, TIME, TARGET = "unique_id", "ds", "y"


def _make_panel(T=20):
    """Balanced panel: every series spans ds=1..T, so the last ``R`` distinct
    timestamps are the last ``R`` ordinals of every (global / group / parent)
    calendar -- which makes ``df[df.ds > T - R]`` an exact ``last-R-ordinals``
    slice across all modes (used by the fit-on-slice contract test)."""
    ids, ds, y, brand, promo = [], [], [], [], []
    series = {"a": ("x", 1.0), "b": ("x", 3.0), "c": ("y", 7.0), "d": ("y", 11.0)}
    for sid, (br, base) in series.items():
        for t in range(1, T + 1):
            ids.append(sid)
            ds.append(t)
            y.append(base + 2.0 * t + 5.0 * ((t * (1 if sid in ("a", "c") else 2)) % 4))
            brand.append(br)
            promo.append(t % 2)
    return pd.DataFrame({ID: ids, TIME: ds, TARGET: y, "brand": brand, "promo": promo})


# Finite-window transforms across every pooled mode. All trimmable, so a fit
# with a small keep_last_n trims every pooled state.
def _finite_lag_transforms():
    return {
        1: [
            RollingMean(4, global_=True),
            RollingMean(4, groupby=["brand"]),
            RollingMean(3, min_samples=1, partition_by=["promo"]),
            RollingMean(3, min_samples=1, global_=True, partition_by=["promo"]),
            RollingMean(3, min_samples=1, groupby=["brand"], partition_by=["promo"]),
            RollingStd(4, min_samples=2, global_=True),
            RollingMin(4, global_=True),
            RollingMax(4, global_=True),
        ]
    }


def _build_fcst(lag_transforms, lags=(1,)):
    return MLForecast(
        models=[LinearRegression()],
        freq=1,
        lags=list(lags),
        lag_transforms=lag_transforms,
    )


def _future_X(h, T=20):
    rows = []
    for sid in ["a", "b", "c", "d"]:
        for t in range(T + 1, T + 1 + h):
            rows.append({ID: sid, TIME: t, "promo": t % 2})
    return pd.DataFrame(rows)


def _sorted_preds(preds):
    preds = preds.sort_values([ID, TIME]).reset_index(drop=True)
    model_cols = [c for c in preds.columns if c not in (ID, TIME)]
    return preds[model_cols].to_numpy().ravel()


_AGG_FIELDS = ("unique_times", "sums", "counts", "sum_sq", "mins", "maxs")


def _assert_state_byte_identical(got, ref, ctx=""):
    """Field-for-field equality of two PooledStates' mutable state.

    The mutable state is what `snapshot`/`restore` round-trips: the aggregate
    channels, the shared calendar length, the bucket vocabulary and the current
    series assignment.
    """
    assert got.n_buckets == ref.n_buckets, f"{ctx}:n_buckets"
    # `n_ordinals` is the absolute calendar position and keeps counting through a
    # trim, so a trimmed state and a fresh fit on the tail differ there by design;
    # what must match is the stored extent and its contents.
    assert got.width == ref.width, f"{ctx}:width"
    np.testing.assert_array_equal(
        got.series_bucket_id, ref.series_bucket_id, err_msg=f"{ctx}:series_bucket_id"
    )
    if ref.bucket_uniques is None:
        assert got.bucket_uniques is None, f"{ctx}:uniques-none"
    else:
        np.testing.assert_array_equal(
            got.bucket_uniques, ref.bucket_uniques, err_msg=f"{ctx}:uniques"
        )
    assert got.base.keys() == ref.base.keys(), f"{ctx}:channels"
    for name in ref.base:
        np.testing.assert_array_equal(
            got.base[name], ref.base[name], err_msg=f"{ctx}:base[{name}]"
        )
    if ref.rows_ord is None:
        assert got.rows_ord is None, f"{ctx}:rows-none"
    else:
        for b, (go, ro) in enumerate(zip(got.rows_ord, ref.rows_ord)):
            np.testing.assert_array_equal(go, ro, err_msg=f"{ctx}:rows_ord[{b}]")
            np.testing.assert_array_equal(
                got.rows_y[b], ref.rows_y[b], err_msg=f"{ctx}:rows_y[{b}]"
            )


def _preprocess_states(df, keep_last_n, lag_transforms, lags=(1,)):
    fcst = _build_fcst(lag_transforms, lags=lags)
    fcst.preprocess(
        df,
        id_col=ID,
        time_col=TIME,
        target_col=TARGET,
        keep_last_n=keep_last_n,
        static_features=["brand"],
        dropna=False,
    )
    return fcst.ts._pooled_states


# A keep_last_n comfortably above every transform window (so the retention
# floor is a no-op and R == keep_last_n), but well below T so a trim happens.
_R = 8
_NO_TRIM = 10_000  # >= calendar length -> trim_to_last is a no-op everywhere


# --------------------------------------------------------------------------- #
# G2.1 -- prediction equality (incl. the retention floor).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("keep_last_n", [2, 6])
def test_g2_1_trimmed_predictions_match_untrimmed(keep_last_n):
    df = _make_panel()
    h = 4
    X_df = _future_X(h)

    fcst_trim = _build_fcst(_finite_lag_transforms())
    fcst_trim.fit(
        df,
        id_col=ID,
        time_col=TIME,
        target_col=TARGET,
        static_features=["brand"],
        keep_last_n=keep_last_n,
    )
    trimmed = _sorted_preds(fcst_trim.predict(h=h, X_df=X_df))

    fcst_full = _build_fcst(_finite_lag_transforms())
    fcst_full.fit(
        df,
        id_col=ID,
        time_col=TIME,
        target_col=TARGET,
        static_features=["brand"],
        keep_last_n=_NO_TRIM,
    )
    full = _sorted_preds(fcst_full.predict(h=h, X_df=X_df))

    # keep_last_n=2 is below the widest window (4); equality here proves the
    # max(keep_last_n, W_state) retention floor kept enough history.
    np.testing.assert_allclose(trimmed, full, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# G2.2 -- trim == fresh fit on the retained tail, before and after update().
# --------------------------------------------------------------------------- #
_MODES = {
    "global": {1: [RollingMean(4, global_=True)]},
    "groupby": {1: [RollingMean(4, groupby=["brand"])]},
    "global+partition": {
        1: [RollingMean(3, min_samples=1, global_=True, partition_by=["promo"])]
    },
    "groupby+partition": {
        1: [RollingMean(3, min_samples=1, groupby=["brand"], partition_by=["promo"])]
    },
    "local+partition": {1: [RollingMean(3, min_samples=1, partition_by=["promo"])]},
}


def _engine(df, engine):
    if engine == "pandas":
        return df
    import polars as pl

    return pl.from_pandas(df)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("name", list(_MODES))
def test_g2_2_trim_equals_fit_on_truncated_slice(name, engine):
    # both engines so the engine-specific bucket_df trim (filter_with_mask) and
    # the narwhals _idsorted_to_bucket_pos rebuild are exercised under each.
    lag_transforms = _MODES[name]
    T = 20
    df = _make_panel(T)
    df_slice = df[df[TIME] > T - _R].reset_index(drop=True)

    trimmed = _preprocess_states(_engine(df, engine), _R, lag_transforms)
    fresh = _preprocess_states(_engine(df_slice, engine), _NO_TRIM, lag_transforms)

    assert trimmed.keys() == fresh.keys()
    for key in trimmed:
        _assert_state_byte_identical(trimmed[key], fresh[key], ctx=f"{name}:{key}")


def _preprocessed_ts(df, keep_last_n, lag_transforms):
    fcst = _build_fcst(lag_transforms)
    fcst.preprocess(
        df,
        id_col=ID,
        time_col=TIME,
        target_col=TARGET,
        keep_last_n=keep_last_n,
        static_features=["brand"],
        dropna=False,
    )
    return fcst.ts


@pytest.mark.parametrize("name", list(_MODES))
def test_g2_2_trim_then_update_matches_fresh_then_update(name):
    """The next update() must extend a trimmed state exactly as it would extend
    a state freshly fit on the retained tail (exercises the rebuilt _ts_aggs and
    both append conventions). Uses preprocess + TimeSeries.update so no model is
    trained -- the contract is purely about state, and a bare static string
    feature would otherwise trip the regressor in the global/local-only modes."""
    lag_transforms = _MODES[name]
    T = 20
    df = _make_panel(T)
    df_slice = df[df[TIME] > T - _R].reset_index(drop=True)

    ts_trim = _preprocessed_ts(df, _R, lag_transforms)
    ts_fresh = _preprocessed_ts(df_slice, _NO_TRIM, lag_transforms)

    # one new timestamp for every series (update requires all series each step)
    new_rows = pd.DataFrame(
        {
            ID: ["a", "b", "c", "d"],
            TIME: [T + 1] * 4,
            TARGET: [101.0, 103.0, 107.0, 111.0],
            "brand": ["x", "x", "y", "y"],
            "promo": [(T + 1) % 2] * 4,
        }
    )
    ts_trim.update(new_rows)
    ts_fresh.update(new_rows)

    assert ts_trim._pooled_states.keys() == ts_fresh._pooled_states.keys()
    for key in ts_trim._pooled_states:
        _assert_state_byte_identical(
            ts_trim._pooled_states[key],
            ts_fresh._pooled_states[key],
            ctx=f"{name}:{key}",
        )


# --------------------------------------------------------------------------- #
# G2.3 -- suffix invariant: trimming only drops a prefix; length == retention.
# --------------------------------------------------------------------------- #
def test_g2_3_suffix_invariant_global():
    T = 20
    df = _make_panel(T)
    lag_transforms = {1: [RollingMean(4, global_=True)]}

    untrimmed = _preprocess_states(df, _NO_TRIM, lag_transforms)
    trimmed = _preprocess_states(df, _R, lag_transforms)

    key = ("global", (), ())
    u_base = untrimmed[key].base
    t_base = trimmed[key].base

    full_len = untrimmed[key].width
    assert full_len == T  # global calendar is the 20 distinct timestamps
    # W_state (= lag+window) <= _R, so retention == keep_last_n == _R
    assert trimmed[key].width == _R
    cutoff = full_len - _R

    for name in u_base:
        np.testing.assert_array_equal(
            t_base[name], u_base[name][:, cutoff:], err_msg=name
        )
    # the stored block is exactly the suffix of the calendar, while the absolute
    # ordinal counter keeps running so future windows still line up
    assert trimmed[key].width == _R
    assert trimmed[key].n_ordinals == T


# --------------------------------------------------------------------------- #
# G2.4 -- unbounded-transform states are never trimmed.
# --------------------------------------------------------------------------- #
def test_g2_4_expanding_and_ewm_states_keep_full_history():
    T = 20
    df = _make_panel(T)
    lag_transforms = {
        1: [
            ExpandingMean(global_=True),
            ExponentiallyWeightedMean(alpha=0.5, groupby=["brand"]),
        ]
    }
    states = _preprocess_states(df, 3, lag_transforms)  # tiny keep_last_n

    exp_state = states[("global", (), ())]
    assert exp_state.width == T  # untouched
    assert exp_state.base["sum"].shape[1] == T

    ewm_state = states[("groupby", ("brand",), ())]
    # every brand bucket keeps its full per-group calendar
    for bid in range(ewm_state.n_buckets):
        nxt = ewm_state.width
        assert nxt == T
        assert ewm_state.base["sum"].shape[1] == T


def test_g2_4_mixed_finite_and_unbounded_state_not_trimmed():
    """A finite and an unbounded transform sharing one mode key produce ONE
    state; because not all its transforms are finite-window, it is not
    trimmed."""
    T = 20
    df = _make_panel(T)
    lag_transforms = {
        1: [
            RollingMean(3, global_=True),  # finite
            ExpandingMean(global_=True),  # unbounded -> blocks the trim
        ]
    }
    states = _preprocess_states(df, 3, lag_transforms)
    state = states[("global", (), ())]  # both transforms share this key
    assert state.width == T
    assert state.base["sum"].shape[1] == T


def test_g2_4_offset_and_combine_respect_inner_transform():
    """Offset/Combine delegate finiteness to their operands: a finite inner
    keeps the state trimmable, an unbounded inner blocks it."""
    T = 20
    df = _make_panel(T)

    finite = {1: [Offset(RollingMean(3, global_=True), 1)]}
    state = _preprocess_states(df, _R, finite)[("global", (), ())]
    assert state.width == _R  # trimmed

    unbounded = {
        1: [
            Combine(
                RollingMean(3, global_=True),
                ExpandingMean(global_=True),
                np.add,
            )
        ]
    }
    state = _preprocess_states(df, 3, unbounded)[("global", (), ())]
    assert state.width == T  # not trimmed
