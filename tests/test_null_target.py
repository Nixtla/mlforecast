"""Tests for `allow_null_target` and the `skipna` lag/target transforms.

The scenario throughout is a product life-cycle: the series is on a dense daily
grid, but for a stretch in the middle the target is unknown (the product didn't
exist / was out of range) rather than zero.
"""

import operator
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.compat import core_supports_skipna
from mlforecast.core import TimeSeries
from mlforecast.lag_transforms import (
    ExpandingMax,
    ExpandingMean,
    ExpandingQuantile,
    ExpandingStd,
    ExponentiallyWeightedMean,
    Offset,
    RollingMean,
    RollingQuantile,
    RollingStd,
    SeasonalRollingMean,
)
from mlforecast.target_transforms import (
    Differences,
    LocalMinMaxScaler,
    LocalRobustScaler,
    LocalStandardScaler,
)

N = 24
# The timestamps whose target is unknown: 2020-01-09..01-11. Selected by date
# rather than by positional slice so both fixtures punch the same hole.
DATES = pd.date_range("2020-01-01", periods=N, freq="D")
GAP_DATES = DATES[8:11]


def _one_series(uid="a", base=1.0, gap=True):
    df = pd.DataFrame(
        {
            "unique_id": uid,
            "ds": DATES,
            "y": base + np.arange(N, dtype=float),
            "brand": "x",
        }
    )
    if gap:
        df.loc[df["ds"].isin(GAP_DATES), "y"] = np.nan
    return df


@pytest.fixture
def series():
    """One series with a 3-day hole in the middle of an otherwise dense grid."""
    return _one_series()[["unique_id", "ds", "y"]]


@pytest.fixture
def two_series():
    """Two aligned series; only `a` has the hole. Used for the pooled paths."""
    return pd.concat(
        [_one_series("a", 1.0), _one_series("b", 100.0, gap=False)], ignore_index=True
    )


@pytest.fixture(autouse=True)
def _quiet_propagation_warning():
    """Silence the advisory 'these transforms propagate nulls' warning.

    The tests that assert on it use `pytest.warns` explicitly, which takes
    precedence over this filter.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*can't skip them.*")
        yield


def _fcst(**kwargs):
    return MLForecast(models=[LinearRegression()], freq="D", **kwargs)


# ---------------------------------------------------------------- the gate


def test_null_target_raises_by_default(series):
    fcst = _fcst(lags=[1])
    with pytest.raises(ValueError, match="y column contains null values"):
        fcst.preprocess(series)


def test_error_points_at_the_flag(series):
    with pytest.raises(ValueError, match="allow_null_target=True"):
        _fcst(lags=[1]).preprocess(series)


def test_allow_null_target_keeps_valid_rows_and_drops_null_ones(series):
    prep = _fcst(lags=[1]).preprocess(series, allow_null_target=True, dropna=False)
    assert not prep["y"].isna().any()
    # the 3 null-target rows are gone, everything else survives
    assert prep.shape[0] == N - 3
    assert set(prep["ds"]) == set(series.loc[series["y"].notna(), "ds"])


def test_null_rows_dropped_even_with_dropna_false(series):
    # dropna=False keeps feature nulls but never target nulls
    prep = _fcst(lags=[2]).preprocess(series, allow_null_target=True, dropna=False)
    assert prep["lag2"].isna().any()  # feature nulls kept
    assert not prep["y"].isna().any()  # target nulls dropped


def test_null_target_still_raises_when_flag_is_false_on_fit(series):
    with pytest.raises(ValueError, match="null values"):
        _fcst(lags=[1]).fit(series)


# ---------------------------------------------------------------- skipna features


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
@pytest.mark.parametrize(
    "tfm, pandas_stat, min_periods",
    [
        (RollingMean(3, 1, skipna=True), "mean", 1),
        (RollingStd(3, 2, skipna=True), "std", 2),
        (RollingQuantile(0.5, 3, 1, skipna=True), "median", 1),
    ],
)
def test_skipna_matches_pandas_skipna(series, tfm, pandas_stat, min_periods):
    """A skipna rolling feature equals pandas' NaN-skipping rolling, shifted by the lag."""
    fcst = _fcst(lag_transforms={1: [tfm]})
    prep = fcst.preprocess(series, allow_null_target=True, dropna=False)
    feat_name = [c for c in prep.columns if c not in ("unique_id", "ds", "y")][0]

    roll = series["y"].rolling(3, min_periods=min_periods)
    expected = getattr(roll, pandas_stat)().shift(1)
    expected = expected[series["y"].notna()].to_numpy()
    np.testing.assert_allclose(prep[feat_name].to_numpy(), expected, rtol=1e-12)


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_skipna_does_not_inflate_the_sample_count(series):
    """The whole point: nulls must not count as observations.

    Imputing the hole with 0 would divide by 3 instead of 1 for a window that
    covers 2 unknown days and 1 real one.
    """
    fcst = _fcst(lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    prep = fcst.preprocess(series, allow_null_target=True, dropna=False)
    name = "rolling_mean_lag1_window_size3_min_samples1_skipnaTrue"
    row = prep.loc[prep["ds"].eq(pd.Timestamp("2020-01-13")), name]
    # window covers 01-10..01-12: 01-10 and 01-11 are unknown, 01-12 is 12.0
    assert row.item() == pytest.approx(12.0)

    zero_filled = series.copy()
    zero_filled["y"] = zero_filled["y"].fillna(0.0)
    naive = _fcst(lag_transforms={1: [RollingMean(3, 1)]}).preprocess(
        zero_filled, dropna=False
    )
    naive_row = naive.loc[
        naive["ds"].eq(pd.Timestamp("2020-01-13")),
        "rolling_mean_lag1_window_size3_min_samples1",
    ]
    assert naive_row.item() == pytest.approx(4.0)  # (0 + 0 + 12) / 3


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_explicit_skipna_false_propagates(series):
    """The opt-out: `skipna=False` keeps coreforecast's propagating semantics."""
    prep = _fcst(lag_transforms={1: [RollingMean(3, 1, skipna=False)]}).preprocess(
        series, allow_null_target=True, dropna=False
    )
    tail = prep.loc[
        prep["ds"].gt(pd.Timestamp("2020-01-11")),
        "rolling_mean_lag1_window_size3_min_samples1_skipnaFalse",
    ]
    assert tail.isna().all()


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_skipna_absent_from_feature_name_at_default():
    """Existing feature names must not change."""
    assert (
        RollingMean(3, 1)._get_name(1) == "rolling_mean_lag1_window_size3_min_samples1"
    )
    assert RollingMean(3, 1, skipna=True)._get_name(1).endswith("_skipnaTrue")


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_seasonal_and_expanding_quantile_skipna(series):
    fcst = _fcst(
        lag_transforms={
            1: [
                SeasonalRollingMean(7, 2, 1, skipna=True),
                ExpandingQuantile(0.5, skipna=True),
            ]
        }
    )
    prep = fcst.preprocess(series, allow_null_target=True, dropna=False)
    # both produce non-null values past the hole, unlike the propagating default
    for col in prep.columns:
        if col in ("unique_id", "ds", "y"):
            continue
        assert prep.loc[prep["ds"].gt(pd.Timestamp("2020-01-15")), col].notna().all()


# ------------------------------------------------- transforms with a broken update


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
@pytest.mark.parametrize(
    "make",
    [
        lambda: ExpandingMean(skipna=True),
        lambda: ExpandingStd(skipna=True),
        lambda: ExpandingMax(skipna=True),
        lambda: ExponentiallyWeightedMean(0.5, skipna=True),
    ],
)
def test_local_skipna_rejected_when_core_update_ignores_it(make):
    """These would train on NaN-skipping features and then predict with
    NaN-propagating ones. Better to refuse than to diverge silently."""
    with pytest.raises(ValueError, match="not supported in local mode"):
        _fcst(lag_transforms={1: [make()]})


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
@pytest.mark.parametrize(
    "scope", [{"global_": True}, {"groupby": ["brand"]}, {"partition_by": ["brand"]}]
)
def test_same_transforms_allowed_in_pooled_scopes(scope):
    """Pooled scopes keep their own NaN-aware aggregates, so they're exempt."""
    _fcst(lag_transforms={1: [ExpandingMean(skipna=True, **scope)]})


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_pooled_skips_nulls_regardless_of_skipna(two_series):
    """Pooled aggregates already exclude nulls; skipna is a no-op there."""
    fcst = _fcst(lag_transforms={1: [RollingMean(3, 1, global_=True)]})
    prep = fcst.preprocess(two_series, allow_null_target=True, dropna=False)
    name = "global_rolling_mean_lag1_window_size3_min_samples1"
    # at 01-12 the window is 01-09..01-11, where only `b` contributed
    row = prep.loc[
        prep["ds"].eq(pd.Timestamp("2020-01-12")) & prep["unique_id"].eq("a"), name
    ]
    assert row.item() == pytest.approx(np.mean([108.0, 109.0, 110.0]))


# ---------------------------------------------------------------- warning


@pytest.mark.parametrize(
    "make", [lambda: ExpandingMean(), lambda: ExponentiallyWeightedMean(0.5)]
)
def test_warns_when_inferred_skipna_cannot_be_honored(series, make):
    """Inferred skipna falls back to propagating rather than raising."""
    with pytest.warns(UserWarning, match="can't skip them"):
        _fcst(lag_transforms={1: [make()]}).preprocess(series, allow_null_target=True)


def test_no_warning_for_transforms_that_can_skip(series):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _fcst(lag_transforms={1: [RollingMean(3, 1)]}).preprocess(
            series, allow_null_target=True
        )


def test_no_warning_for_plain_lags(series):
    """A lag of a null is genuinely null, so Lag must not be flagged."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _fcst(lags=[1, 7]).preprocess(series, allow_null_target=True)


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_no_warning_when_skipna_is_set(series):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _fcst(lag_transforms={1: [RollingMean(3, 1, skipna=True)]}).preprocess(
            series, allow_null_target=True
        )


# ---------------------------------------------------------------- target transforms


@pytest.mark.parametrize(
    "tfm", [Differences([1]), LocalStandardScaler(), LocalMinMaxScaler()]
)
def test_nan_propagating_target_transforms_rejected(series, tfm):
    with pytest.raises(ValueError, match="not supported with these target transforms"):
        _fcst(lags=[1], target_transforms=[tfm]).preprocess(
            series, allow_null_target=True
        )


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
@pytest.mark.parametrize(
    "tfm",
    [
        LocalStandardScaler(skipna=True),
        LocalMinMaxScaler(skipna=True),
        LocalRobustScaler("iqr", skipna=True),
    ],
)
def test_skipna_target_transforms_accepted(series, tfm):
    prep = _fcst(
        lags=[1],
        target_transforms=[tfm],
        lag_transforms={1: [RollingMean(3, 1, skipna=True)]},
    ).preprocess(series, allow_null_target=True)
    assert not prep["y"].isna().any()
    assert prep.shape[0] > 0


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_scaler_skipna_statistics_ignore_nulls(series):
    """The scaler's mean/std come from the observed values only."""
    scaler = LocalStandardScaler(skipna=True)
    _fcst(
        lags=[1],
        target_transforms=[scaler],
        lag_transforms={1: [RollingMean(3, 1, skipna=True)]},
    ).preprocess(series, allow_null_target=True)
    observed = series["y"].dropna()
    np.testing.assert_allclose(
        scaler.scaler_.stats_.ravel(),
        [observed.mean(), observed.std(ddof=0)],
        rtol=1e-10,
    )


# ---------------------------------------------------------------- end to end


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_fit_predict_round_trip(series):
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    fcst.fit(series, allow_null_target=True)
    preds = fcst.predict(3)
    assert preds.shape == (3, 3)
    assert preds["LinearRegression"].notna().all()


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_fitted_values_exclude_null_rows(series):
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    fcst.fit(series, fitted=True, allow_null_target=True)
    fitted = fcst.forecast_fitted_values()
    assert not fitted["y"].isna().any()
    null_dates = set(series.loc[series["y"].isna(), "ds"])
    assert not (set(fitted["ds"]) & null_dates)


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_max_horizon_path(series):
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    fcst.fit(series, max_horizon=2, allow_null_target=True)
    preds = fcst.predict(2)
    assert preds.shape[0] == 2
    assert preds["LinearRegression"].notna().all()


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_cross_validation_path(series):
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    cv = fcst.cross_validation(
        series,
        n_windows=2,
        h=2,
        allow_null_target=True,
    )
    assert cv.shape[0] == 4
    assert cv["LinearRegression"].notna().all()


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_history_warmup_path(series):
    trained = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    trained.fit(series, allow_null_target=True)
    expected = trained.predict(2)

    warmed = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    warmed.models_ = trained.models_
    warmed.history_warmup(series, allow_null_target=True)
    pd.testing.assert_frame_equal(warmed.predict(2), expected)


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_polars_parity(series):
    pl_series = pl.from_pandas(series)
    kwargs = dict(allow_null_target=True, dropna=False)
    tfms = {1: [RollingMean(3, 1, skipna=True)]}
    pd_prep = MLForecast(
        models=[LinearRegression()], freq="D", lags=[1], lag_transforms=tfms
    ).preprocess(series, **kwargs)
    # polars needs a polars offset string
    pl_prep = MLForecast(
        models=[LinearRegression()], freq="1d", lags=[1], lag_transforms=tfms
    ).preprocess(pl_series, **kwargs)
    np.testing.assert_allclose(
        pd_prep.drop(columns=["unique_id", "ds"]).to_numpy(),
        pl_prep.drop(["unique_id", "ds"]).to_numpy(),
        rtol=1e-12,
    )


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_timeseries_level_api(series):
    ts = TimeSeries(
        freq="D", lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]}
    )
    out = ts.fit_transform(
        series,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        allow_null_target=True,
    )
    assert not out["y"].isna().any()


# ------------------------------------------ internal rebuild paths inherit the flag


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_flag_is_recorded_on_the_timeseries(series):
    fcst = _fcst(lags=[1])
    fcst.fit(series, allow_null_target=True)
    assert fcst.ts.allow_null_target is True
    other = _fcst(lags=[1])
    other.fit(_one_series(gap=False)[["unique_id", "ds", "y"]])
    assert other.ts.allow_null_target is False


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_predict_with_new_df(series):
    """predict(new_df=...) rebuilds history and must not re-reject the nulls."""
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    fcst.fit(series, allow_null_target=True)
    preds = fcst.predict(2, new_df=series)
    assert preds.shape[0] == 2
    assert preds["LinearRegression"].notna().all()


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_cross_validation_without_refit(series):
    """refit=False routes windows through predict(new_df=...)."""
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    cv = fcst.cross_validation(
        series, n_windows=2, h=2, refit=False, allow_null_target=True
    )
    assert cv.shape[0] == 4
    assert cv["LinearRegression"].notna().all()


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_recursive_multistep_fitted_values_with_leading_nulls():
    """forecast_fitted_values(h>1) refits a temporary history per origin."""
    df = _one_series(gap=False)[["unique_id", "ds", "y"]]
    df.loc[df["ds"].isin(DATES[:2]), "y"] = np.nan  # leading nulls
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    fcst.fit(df, fitted=True, allow_null_target=True)
    fitted = fcst.forecast_fitted_values(h=2)
    assert fitted.shape[0] > 0
    assert not fitted["y"].isna().any()


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_lightgbm_cv_finalization():
    pytest.importorskip("lightgbm")
    from mlforecast.lgb_cv import LightGBMCV

    df = pd.concat([_one_series("a", 1.0), _one_series("b", 50.0)], ignore_index=True)[
        ["unique_id", "ds", "y"]
    ]
    cv = LightGBMCV(
        freq="D", lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]}
    )
    # runs to completion, including the final `ts._fit` on the full history
    hist = cv.fit(
        df,
        n_windows=2,
        h=2,
        num_iterations=2,
        eval_every=2,
        params={"verbosity": -1},
        allow_null_target=True,
    )
    assert hist
    assert np.isfinite(hist[-1][1])


# ------------------------------------------------- trailing nulls / dropped series


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_trailing_null_does_not_mark_series_as_dropped():
    """A series ending on a null still has valid history; it isn't 'dropped'."""
    df = _one_series(gap=False)[["unique_id", "ds", "y"]]
    df.loc[df["ds"].eq(DATES[-1]), "y"] = np.nan
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    fcst.preprocess(df, allow_null_target=True)
    assert fcst.ts._dropped_series is None


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_trailing_null_with_scaler_and_fitted_values():
    """Regression: the scaler state was discarded for a series with a trailing null."""
    df = _one_series(gap=False)[["unique_id", "ds", "y"]]
    df.loc[df["ds"].eq(DATES[-1]), "y"] = np.nan
    fcst = _fcst(
        lags=[1],
        target_transforms=[LocalStandardScaler(skipna=True)],
        lag_transforms={1: [RollingMean(3, 1, skipna=True)]},
    )
    fcst.fit(df, fitted=True, allow_null_target=True)
    fitted = fcst.forecast_fitted_values()
    assert not fitted["y"].isna().any()
    assert fitted.shape[0] > 0


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_fully_null_series_is_still_reported_as_dropped():
    """The original behaviour must survive: an all-null series is dropped."""
    a = _one_series("a", 1.0, gap=False)[["unique_id", "ds", "y"]]
    b = _one_series("b", 50.0, gap=False)[["unique_id", "ds", "y"]]
    b["y"] = np.nan
    df = pd.concat([a, b], ignore_index=True)
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    with pytest.warns(UserWarning, match="dropped completely"):
        fcst.preprocess(df, allow_null_target=True)
    assert fcst.ts._dropped_series is not None
    assert list(fcst.ts.uids[fcst.ts._dropped_series]) == ["b"]


# ---------------------------------------------------------- prediction intervals


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_prediction_intervals_with_observed_tail():
    """Nulls confined to older history must not block interval calibration."""
    from mlforecast.utils import PredictionIntervals

    df = pd.concat([_one_series("a", 1.0), _one_series("b", 50.0)], ignore_index=True)[
        ["unique_id", "ds", "y"]
    ]
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    fcst.fit(
        df,
        allow_null_target=True,
        prediction_intervals=PredictionIntervals(n_windows=2, h=2),
    )
    preds = fcst.predict(2, level=[80])
    assert preds["LinearRegression-lo-80"].notna().all()
    assert preds["LinearRegression-hi-80"].notna().all()


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_prediction_intervals_reject_null_calibration_tail():
    """A null inside the calibration windows would silently give null intervals."""
    from mlforecast.utils import PredictionIntervals

    df = _one_series(gap=False)[["unique_id", "ds", "y"]]
    df.loc[df["ds"].eq(DATES[-1]), "y"] = np.nan  # lands in the last CV window
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    with pytest.raises(ValueError, match="calibration windows contain unobserved"):
        fcst.fit(
            df,
            allow_null_target=True,
            prediction_intervals=PredictionIntervals(n_windows=2, h=2),
        )


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_cv_returns_null_actuals_rather_than_dropping_them():
    """Documented contract: row count stays h * n_windows per serie."""
    df = _one_series(gap=False)[["unique_id", "ds", "y"]]
    df.loc[df["ds"].eq(DATES[-1]), "y"] = np.nan
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    cv = fcst.cross_validation(df, n_windows=2, h=2, allow_null_target=True)
    assert cv.shape[0] == 4
    assert cv["y"].isna().sum() == 1
    assert cv["LinearRegression"].notna().all()


# --------------------------------------------------------- transfer conformal


def _two_series_no_gap():
    return pd.concat(
        [_one_series("a", 1.0, gap=False), _one_series("b", 50.0, gap=False)],
        ignore_index=True,
    )[["unique_id", "ds", "y"]]


def _fitted_with_intervals(
    train, method="conformal_distribution", scale_estimator=None
):
    from mlforecast.utils import PredictionIntervals

    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1, skipna=True)]})
    fcst.fit(
        train,
        allow_null_target=True,
        prediction_intervals=PredictionIntervals(
            n_windows=2, h=2, method=method, scale_estimator=scale_estimator
        ),
    )
    return fcst


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
@pytest.mark.parametrize("method", ["recalibrate", "error_scaled"])
def test_transfer_conformal_rejects_null_calibration_tail(method):
    """P1: a null in new_df's validation tail silently produced null intervals."""
    train = _two_series_no_gap()
    fcst = _fitted_with_intervals(train)

    new_df = _two_series_no_gap()
    new_df.loc[new_df["ds"].eq(DATES[-1]), "y"] = np.nan  # inside the backtest tail
    with pytest.raises(ValueError, match="calibration windows contain unobserved"):
        fcst.predict(2, new_df=new_df, level=[80], transfer_conformal=method)


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
@pytest.mark.parametrize("method", ["recalibrate", "error_scaled"])
def test_transfer_conformal_works_with_observed_tail(method):
    """Nulls in older history are fine; only the calibration tail must be observed."""
    train = _two_series_no_gap()
    fcst = _fitted_with_intervals(train)

    new_df = pd.concat(
        [_one_series("a", 1.0), _one_series("b", 50.0)], ignore_index=True
    )[["unique_id", "ds", "y"]]  # holes at 01-09..01-11 only
    preds = fcst.predict(2, new_df=new_df, level=[80], transfer_conformal=method)
    assert preds["LinearRegression-lo-80"].notna().all()
    assert preds["LinearRegression-hi-80"].notna().all()


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
@pytest.mark.parametrize("method", ["weighted_conformal", "scale_aligned_weighted"])
def test_weighted_transfer_inherits_allow_null_target(method):
    """P2: preprocess_fn(new_df) must not re-reject the accepted nulls."""
    train = _two_series_no_gap()
    # scale_aligned_weighted additionally requires a source scale estimator
    fcst = _fitted_with_intervals(
        train,
        method="weighted_conformal_distribution",
        scale_estimator="mad" if method == "scale_aligned_weighted" else None,
    )

    new_df = pd.concat(
        [_one_series("a", 1.0), _one_series("b", 50.0)], ignore_index=True
    )[["unique_id", "ds", "y"]]
    preds = fcst.predict(2, new_df=new_df, level=[80], transfer_conformal=method)
    assert preds["LinearRegression-lo-80"].notna().all()
    assert preds["LinearRegression-hi-80"].notna().all()


# ------------------------------------------------------------ tri-state skipna


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_skipna_inferred_from_allow_null_target(series):
    """`allow_null_target=True` alone is enough; no per-transform annotation."""
    prep = _fcst(lag_transforms={1: [RollingMean(3, 1)]}).preprocess(
        series, allow_null_target=True, dropna=False
    )
    col = "rolling_mean_lag1_window_size3_min_samples1"
    # would be NaN from 01-12 onwards if the nulls were propagating
    assert prep.loc[prep["ds"].gt(pd.Timestamp("2020-01-12")), col].notna().all()
    # and equals what an explicit skipna=True produces
    explicit = _fcst(lag_transforms={1: [RollingMean(3, 1, skipna=True)]}).preprocess(
        series, allow_null_target=True, dropna=False
    )
    np.testing.assert_allclose(
        prep[col].to_numpy(),
        explicit[col + "_skipnaTrue"].to_numpy(),
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_default_still_propagates_without_the_flag():
    """Without `allow_null_target` nothing changes for existing users."""
    df = _one_series(gap=False)[["unique_id", "ds", "y"]]
    prep = _fcst(lag_transforms={1: [RollingMean(3, 1)]}).preprocess(df, dropna=False)
    tfm = next(
        iter(_fcst(lag_transforms={1: [RollingMean(3, 1)]}).ts.transforms.values())
    )
    assert tfm.skipna is None  # declared value untouched
    assert "rolling_mean_lag1_window_size3_min_samples1" in prep.columns


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_inference_does_not_change_feature_names(series):
    """The name reflects the declared spec, so it can't move under the flag."""
    clean = _one_series(gap=False)[["unique_id", "ds", "y"]]
    with_flag = _fcst(lag_transforms={1: [RollingMean(3, 1)]}).preprocess(
        series, allow_null_target=True, dropna=False
    )
    without = _fcst(lag_transforms={1: [RollingMean(3, 1)]}).preprocess(
        clean, dropna=False
    )
    assert list(with_flag.columns) == list(without.columns)


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_refitting_re_resolves_the_flag(series):
    """The declared None is never overwritten, so a second fit re-decides."""
    fcst = _fcst(lag_transforms={1: [RollingMean(3, 1)]})
    fcst.preprocess(series, allow_null_target=True, dropna=False)
    tfm = next(iter(fcst.ts.transforms.values()))
    assert tfm.skipna is None and tfm._core_tfm.skipna is True

    clean = _one_series(gap=False)[["unique_id", "ds", "y"]]
    fcst.preprocess(clean, dropna=False)
    tfm = next(iter(fcst.ts.transforms.values()))
    assert tfm.skipna is None and tfm._core_tfm.skipna is False


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_inference_reaches_wrapped_transforms(series):
    """Offset/Combine take no skipna of their own and must delegate."""
    from mlforecast.lag_transforms import Combine

    fcst = _fcst(
        lag_transforms={
            1: [
                Offset(RollingMean(3, 1), 1),
                Combine(RollingMean(3, 1), RollingMean(2, 1), operator.truediv),
            ]
        }
    )
    prep = fcst.preprocess(series, allow_null_target=True, dropna=False)
    feats = [c for c in prep.columns if c not in ("unique_id", "ds", "y")]
    assert len(feats) == 2
    for col in feats:
        assert prep.loc[prep["ds"].eq(DATES[-1]), col].notna().all()


@pytest.mark.skipif(
    not core_supports_skipna(), reason="coreforecast without skipna support"
)
def test_inferred_skipna_survives_predict(series):
    """Resolution happens in `_fit`, which every rebuild path goes through."""
    fcst = _fcst(lags=[1], lag_transforms={1: [RollingMean(3, 1)]})
    fcst.fit(series, allow_null_target=True)
    direct = fcst.predict(2)
    rebuilt = fcst.predict(2, new_df=series)
    pd.testing.assert_frame_equal(direct, rebuilt)
