from functools import cache

import lightgbm
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals, TransferConformal, generate_daily_series


HORIZON = 14
N_WINDOWS = 10
N_SOURCE_SERIES = 45
N_TARGET_SERIES = 35
LEVELS = [80, 90, 95]
TRANSFER_METHODS = ["recalibrate", "scale_aligned", "error_scaled"]
MODEL = "LGBMRegressor"
_PREDICTION_CACHE: dict[tuple, pd.DataFrame] = {}


# ---------------------------------------------------------------------------
# Task 1: data-model additions
# ---------------------------------------------------------------------------
def test_transfer_conformal_step_size_validation():
    with pytest.raises(ValueError, match="step_size"):
        TransferConformal(step_size=0)
    with pytest.raises(ValueError, match="step_size"):
        TransferConformal(step_size=-1)
    tc = TransferConformal(step_size=2)
    assert tc.step_size == 2


def test_transfer_result_signed_default():
    from mlforecast.conformal_prediction import TransferResult
    dummy = pd.DataFrame({"unique_id": ["a"], "ds": [1], "m": [0.0]})
    tr = TransferResult(cs_df=dummy)
    assert tr.signed is False
    tr2 = TransferResult(cs_df=dummy, signed=True)
    assert tr2.signed is True


def test_compute_conformity_scores_signed():
    from mlforecast.conformal_prediction import compute_conformity_scores
    cv = pd.DataFrame({
        "unique_id": ["a", "a"],
        "ds":        [1, 2],
        "cutoff":    [0, 0],
        "y":         [3.0, 5.0],
        "m":         [1.0, 7.0],
    })
    # unsigned: |y - pred|
    unsigned = compute_conformity_scores(cv.copy(), ["m"], "y")
    assert list(unsigned["m"]) == [2.0, 2.0]

    # signed: y - pred (target - model)
    signed = compute_conformity_scores(cv.copy(), ["m"], "y", signed=True)
    assert list(signed["m"]) == [2.0, -2.0]


@pytest.fixture(scope="module")
def transfer_cp_setup():
    source_train = generate_daily_series(
        N_SOURCE_SERIES, min_length=400, max_length=400, seed=0
    )
    target = generate_daily_series(
        N_TARGET_SERIES, min_length=400, max_length=400, seed=1
    )

    source_train["unique_id"] = "src_" + source_train["unique_id"].astype(str)
    target["unique_id"] = "tgt_" + target["unique_id"].astype(str)

    target_test = target.groupby("unique_id", observed=True).tail(HORIZON)
    target_train = target.drop(target_test.index)

    @cache
    def fit(method):
        mlf = MLForecast(
            models=lightgbm.LGBMRegressor(
                n_estimators=20,
                random_state=0,
                verbosity=-1,
            ),
            lags=[1, 7, 14],
            lag_transforms={1: [ExpandingMean()]},
            freq="D",
            num_threads=1,
        )
        return mlf.fit(
            source_train,
            prediction_intervals=PredictionIntervals(
                n_windows=N_WINDOWS,
                h=HORIZON,
                method="conformal_error",
                transfer=method,
            ),
        )

    return (
        fit,
        target_train.reset_index(drop=True),
        target_test.reset_index(drop=True),
    )


def compute_coverage(
    preds: pd.DataFrame,
    actuals: pd.DataFrame,
    model: str,
    levels: list[int],
) -> dict[int, float]:
    merged = preds.merge(
        actuals[["unique_id", "ds", "y"]],
        on=["unique_id", "ds"],
        how="inner",
    )
    if len(merged) != len(actuals):
        raise AssertionError(
            f"Expected {len(actuals)} matched forecast rows, got {len(merged)}."
        )

    coverage = {}
    for level in levels:
        covered = merged["y"].between(
            merged[f"{model}-lo-{level}"],
            merged[f"{model}-hi-{level}"],
        )
        coverage[level] = float(covered.mean())
    return coverage


def _predict_transfer(transfer_cp_setup, method: str) -> pd.DataFrame:
    fit, target_train, _ = transfer_cp_setup
    cache_key = (id(fit), method)
    if cache_key not in _PREDICTION_CACHE:
        _PREDICTION_CACHE[cache_key] = (
            fit(method).transfer(target_train).predict(h=HORIZON, level=[*LEVELS])
        )
    return _PREDICTION_CACHE[cache_key].copy()


@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_coverage_within_tolerance(transfer_cp_setup, method):
    _, _, target_test = transfer_cp_setup
    preds = _predict_transfer(transfer_cp_setup, method)
    coverage = compute_coverage(preds, target_test, MODEL, LEVELS)

    for level, empirical in coverage.items():
        nominal = level / 100
        # error_scaled uses a robust IQR ratio (more conservative than std) so
        # slight over-coverage is expected; allow up to 0.06 deviation.
        tol = 0.06 if method == "error_scaled" else 0.05
        assert abs(empirical - nominal) <= tol, (
            f"{method} level {level} empirical coverage {empirical:.3f} "
            f"differs from nominal {nominal:.3f} by more than {tol}."
        )


@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_coverage_monotonicity(transfer_cp_setup, method):
    _, _, target_test = transfer_cp_setup
    preds = _predict_transfer(transfer_cp_setup, method)
    coverage = compute_coverage(preds, target_test, MODEL, LEVELS)

    assert coverage[80] <= coverage[90] <= coverage[95]


@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_interval_columns_present(transfer_cp_setup, method):
    preds = _predict_transfer(transfer_cp_setup, method)
    interval_columns = [
        f"{MODEL}-{bound}-{level}"
        for level in LEVELS
        for bound in ("lo", "hi")
    ]

    assert set(interval_columns).issubset(preds.columns)
    assert not preds[interval_columns].isna().any().any()
    assert np.isfinite(preds[interval_columns].to_numpy()).all()


@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_interval_nesting(transfer_cp_setup, method):
    preds = _predict_transfer(transfer_cp_setup, method)

    nested = (
        (preds[f"{MODEL}-lo-95"] <= preds[f"{MODEL}-lo-90"])
        & (preds[f"{MODEL}-lo-90"] <= preds[f"{MODEL}-lo-80"])
        & (preds[f"{MODEL}-lo-80"] <= preds[f"{MODEL}-hi-80"])
        & (preds[f"{MODEL}-hi-80"] <= preds[f"{MODEL}-hi-90"])
        & (preds[f"{MODEL}-hi-90"] <= preds[f"{MODEL}-hi-95"])
    )
    assert nested.all()


def test_equal_counts_pooled_uniform_widths():
    """n_source == n_target must produce uniform pooled interval widths (item 1 bug fix)."""
    n = 5
    source_train = generate_daily_series(n, min_length=200, max_length=200, seed=10)
    target_train = generate_daily_series(n, min_length=200, max_length=200, seed=11)
    source_train["unique_id"] = "src_" + source_train["unique_id"].astype(str)
    target_train["unique_id"] = "tgt_" + target_train["unique_id"].astype(str)

    mlf = MLForecast(
        models=lightgbm.LGBMRegressor(n_estimators=10, random_state=0, verbosity=-1),
        lags=[1, 7],
        freq="D",
        num_threads=1,
    )
    mlf.fit(
        source_train,
        prediction_intervals=PredictionIntervals(
            n_windows=3, h=7, transfer="error_scaled"
        ),
    )

    preds = mlf.transfer(target_train).predict(h=7, level=[90])
    # For each forecast date (horizon step), all target series must share the same
    # interval width (pooled quantile is the same scalar for every series at that step).
    for ds, group in preds.groupby("ds", sort=False):
        step_widths = group[f"{MODEL}-hi-90"] - group[f"{MODEL}-lo-90"]
        assert np.allclose(step_widths, step_widths.iloc[0], rtol=1e-6), (
            f"Pooled transfer with equal source/target counts must produce uniform widths "
            f"across series at each horizon step; ds={ds} got "
            f"[{step_widths.min():.4f}, {step_widths.max():.4f}]"
        )


def test_h1_transfer_raises():
    """PredictionIntervals(h=1) + transfer + h > 1 must raise ValueError (item 2 bug fix)."""
    n = 5
    source_train = generate_daily_series(n, min_length=100, max_length=100, seed=20)
    target_train = generate_daily_series(n, min_length=100, max_length=100, seed=21)
    source_train["unique_id"] = "src_" + source_train["unique_id"].astype(str)
    target_train["unique_id"] = "tgt_" + target_train["unique_id"].astype(str)

    mlf = MLForecast(
        models=lightgbm.LGBMRegressor(n_estimators=10, random_state=0, verbosity=-1),
        lags=[1],
        freq="D",
        num_threads=1,
    )
    mlf.fit(
        source_train,
        prediction_intervals=PredictionIntervals(
            n_windows=2, h=1, transfer="error_scaled"
        ),
    )

    with pytest.raises(ValueError, match="requires PredictionIntervals"):
        mlf.transfer(target_train).predict(h=7, level=[90])


def test_methods_produce_different_widths(transfer_cp_setup):
    """Different transfer methods must produce meaningfully different interval widths."""
    preds_r = _predict_transfer(transfer_cp_setup, "recalibrate")
    preds_s = _predict_transfer(transfer_cp_setup, "scale_aligned")
    widths_r = (preds_r[f"{MODEL}-hi-90"] - preds_r[f"{MODEL}-lo-90"]).mean()
    widths_s = (preds_s[f"{MODEL}-hi-90"] - preds_s[f"{MODEL}-lo-90"]).mean()
    assert not np.isclose(widths_r, widths_s, rtol=0.01), (
        f"'recalibrate' and 'scale_aligned' produced nearly identical mean widths "
        f"({widths_r:.4f} vs {widths_s:.4f}); expected them to differ."
    )


# ---------------------------------------------------------------------------
# Item 5: n_windows tests
# ---------------------------------------------------------------------------

def test_error_scaled_n_windows_1_works():
    """error_scaled with n_windows=1 completes without error."""
    n = 5
    h = 4
    source = generate_daily_series(n, min_length=60, max_length=60, seed=30)
    source["unique_id"] = "src_" + source["unique_id"].astype(str)
    target = generate_daily_series(n, min_length=h + 2, max_length=h + 2, seed=31)
    target["unique_id"] = "tgt_" + target["unique_id"].astype(str)

    mlf = MLForecast(
        models=lightgbm.LGBMRegressor(n_estimators=5, random_state=0, verbosity=-1),
        lags=[1],
        freq="D",
        num_threads=1,
    )
    tc = TransferConformal(method="error_scaled", n_windows=1)
    mlf.fit(
        source,
        prediction_intervals=PredictionIntervals(n_windows=2, h=h, transfer=tc),
    )
    preds = mlf.transfer(target).predict(h=h, level=[90])
    assert f"{MODEL}-lo-90" in preds.columns
    assert np.isfinite(preds[f"{MODEL}-lo-90"].to_numpy()).all()


def test_recalibrate_n_windows_1_raises():
    """recalibrate with n_windows=1 raises a clear ValueError."""
    n = 5
    h = 4
    source = generate_daily_series(n, min_length=60, max_length=60, seed=32)
    source["unique_id"] = "src_" + source["unique_id"].astype(str)
    target = generate_daily_series(n, min_length=60, max_length=60, seed=33)
    target["unique_id"] = "tgt_" + target["unique_id"].astype(str)

    mlf = MLForecast(
        models=lightgbm.LGBMRegressor(n_estimators=5, random_state=0, verbosity=-1),
        lags=[1],
        freq="D",
        num_threads=1,
    )
    tc = TransferConformal(method="recalibrate", n_windows=1)
    mlf.fit(
        source,
        prediction_intervals=PredictionIntervals(n_windows=2, h=h, transfer=tc),
    )
    with pytest.raises(ValueError, match="requires at least 2"):
        mlf.transfer(target)


# ---------------------------------------------------------------------------
# Task 2: _frozen_backtest
# ---------------------------------------------------------------------------
def test_frozen_backtest_min_length_validation():
    """Too-short target raises ValueError naming the shortfall."""
    n, h = 3, 5
    source = generate_daily_series(n, min_length=60, max_length=60, seed=40)
    source["unique_id"] = "src_" + source["unique_id"].astype(str)
    target = generate_daily_series(n, min_length=6, max_length=6, seed=41)
    target["unique_id"] = "tgt_" + target["unique_id"].astype(str)

    mlf = MLForecast(
        models=lightgbm.LGBMRegressor(n_estimators=5, random_state=0, verbosity=-1),
        lags=[1],
        freq="D",
        num_threads=1,
    )
    tc = TransferConformal(method="recalibrate", n_windows=2)
    mlf.fit(
        source,
        prediction_intervals=PredictionIntervals(n_windows=2, h=h, transfer=tc),
    )
    # need h + (2-1)*1 + 1 + 1 = 8 time steps; target only has 6
    with pytest.raises(ValueError, match="time steps"):
        mlf.transfer(target)


def test_frozen_backtest_uses_source_model():
    """Recalibrate intervals are WIDE when source model is bad on target (100x scale).

    If the bug were present (target re-trained), the target model would fit the 100x
    data well and residuals would be small → intervals narrow.
    With the frozen source model, residuals are huge → intervals wide.
    """
    n, h = 5, 3
    src = generate_daily_series(n, min_length=50, max_length=50, seed=0)
    src["unique_id"] = "src_" + src["unique_id"].astype(str)

    tgt = generate_daily_series(n, min_length=30, max_length=30, seed=1)
    tgt["unique_id"] = "tgt_" + tgt["unique_id"].astype(str)
    tgt["y"] = tgt["y"] * 100  # 100x scale — source model will predict ~src values

    mlf = MLForecast(
        models=lightgbm.LGBMRegressor(n_estimators=10, random_state=0, verbosity=-1),
        lags=[1],
        freq="D",
        num_threads=1,
    )
    mlf.fit(
        src,
        prediction_intervals=PredictionIntervals(
            n_windows=2, h=h, transfer="recalibrate"
        ),
    )

    src_preds = mlf.predict(h=h, level=[90])
    src_width = float(
        (src_preds["LGBMRegressor-hi-90"] - src_preds["LGBMRegressor-lo-90"]).mean()
    )

    tgt_preds = mlf.transfer(tgt).predict(h=h, level=[90])
    tgt_width = float(
        (tgt_preds["LGBMRegressor-hi-90"] - tgt_preds["LGBMRegressor-lo-90"]).mean()
    )

    # Frozen-model recalibrate: residuals ≈ 100x source values → much wider intervals.
    assert tgt_width > src_width * 5, (
        f"Recalibrate on 100x target (width={tgt_width:.3f}) should be far wider than "
        f"source intervals (width={src_width:.3f}). Frozen-model invariant violated."
    )


@pytest.mark.parametrize("method", ["recalibrate", "error_scaled"])
def test_frozen_backtest_unequal_length_series(method):
    """Backtest windows must be computed per series, not from global times.

    With unaligned series ends (e.g. M4-style data where every series starts at the
    same origin but has its own length), global cutoffs place the validation window
    past the end of all but the longest series, producing NaN conformity scores and
    NaN intervals. Regression test for that bug: intervals must be finite and
    properly ordered for every series.
    """
    h = 4
    source = generate_daily_series(3, min_length=60, max_length=60, seed=10)
    source["unique_id"] = "src_" + source["unique_id"].astype(str)
    # Unequal lengths -> unaligned series ends (all start on the same date).
    target = generate_daily_series(3, min_length=30, max_length=55, seed=11)
    target["unique_id"] = "tgt_" + target["unique_id"].astype(str)
    assert target.groupby("unique_id")["ds"].max().nunique() > 1

    mlf = MLForecast(
        models=lightgbm.LGBMRegressor(n_estimators=10, random_state=0, verbosity=-1),
        lags=[1, 2],
        freq="D",
        num_threads=1,
    )
    mlf.fit(
        source,
        prediction_intervals=PredictionIntervals(n_windows=2, h=h, transfer=method),
    )

    preds = mlf.transfer(target).predict(h=h, level=[90])
    lo = preds["LGBMRegressor-lo-90"].to_numpy()
    hi = preds["LGBMRegressor-hi-90"].to_numpy()
    assert np.isfinite(lo).all(), f"{method}: lower bounds contain non-finite values"
    assert np.isfinite(hi).all(), f"{method}: upper bounds contain non-finite values"
    assert (lo <= hi).all()


def test_point_forecasts_invariant_across_transfer_methods():
    """Transfer methods may only change interval widths, never the point forecasts.

    Regression test: target transforms (Differences) store fitted state inside the
    transform objects, which used to be shared between self.ts and the new_df
    TimeSeries. The frozen backtest's nested predict calls then re-fitted that
    state on truncated windows, shifting the outer point forecasts for the
    runs_target_cv methods (recalibrate, error_scaled).
    """
    from mlforecast.target_transforms import Differences

    h = 4
    source = generate_daily_series(3, min_length=60, max_length=60, seed=20)
    source["unique_id"] = "src_" + source["unique_id"].astype(str)
    target = generate_daily_series(3, min_length=30, max_length=55, seed=21)
    target["unique_id"] = "tgt_" + target["unique_id"].astype(str)
    # Strong trend so a clobbered Differences restore produces a visible shift.
    target["y"] = target["y"] + target.groupby("unique_id").cumcount() * 2.0

    def fresh_fit(method):
        mlf = MLForecast(
            models=lightgbm.LGBMRegressor(n_estimators=10, random_state=0, verbosity=-1),
            lags=[1, 2],
            freq="D",
            target_transforms=[Differences([1])],
            num_threads=1,
        )
        mlf.fit(
            source,
            prediction_intervals=PredictionIntervals(
                n_windows=2, h=h, method="conformal_error", transfer=method
            ),
        )
        return mlf

    baseline = fresh_fit(None).transfer(target).predict(h=h)["LGBMRegressor"].to_numpy()
    for method in TRANSFER_METHODS:
        preds = fresh_fit(method).transfer(target).predict(h=h, level=[90])
        np.testing.assert_allclose(
            preds["LGBMRegressor"].to_numpy(),
            baseline,
            err_msg=f"{method}: point forecasts differ from plain prediction",
        )


# ---------------------------------------------------------------------------
# Task 3: _add_signed_transfer_intervals
# ---------------------------------------------------------------------------
def test_add_signed_transfer_intervals_shape_and_nesting():
    import pandas as pd
    from mlforecast.conformal_prediction import _add_signed_transfer_intervals

    n_series, horizon = 3, 2
    n_cal = 5  # windows * series (pooled)
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, size=n_cal * horizon)  # signed residuals

    cs_df = pd.DataFrame({"m": scores})
    fcst_df = pd.DataFrame({
        "unique_id": np.repeat(["a", "b", "c"], horizon),
        "ds": list(range(horizon)) * n_series,
        "m": rng.normal(5, 1, n_series * horizon),
    })

    result = _add_signed_transfer_intervals(
        fcst_df, cs_df, model_names=["m"], level=[80, 90], horizon=horizon
    )

    # Columns present
    for lv in [80, 90]:
        assert f"m-lo-{lv}" in result.columns
        assert f"m-hi-{lv}" in result.columns

    # lo <= hi for all rows
    for lv in [80, 90]:
        assert (result[f"m-lo-{lv}"] <= result[f"m-hi-{lv}"]).all()

    # Nesting: lo-90 <= lo-80 <= hi-80 <= hi-90
    assert (result["m-lo-90"] <= result["m-lo-80"]).all()
    assert (result["m-lo-80"] <= result["m-hi-80"]).all()
    assert (result["m-hi-80"] <= result["m-hi-90"]).all()


# ---------------------------------------------------------------------------
# Task 4: _recalibrate_transfer signed residuals
# ---------------------------------------------------------------------------
def test_recalibrate_transfer_result_is_signed():
    """_recalibrate_transfer must return TransferResult(signed=True) with signed scores."""
    import pandas as pd
    from mlforecast.conformal_prediction import (
        _recalibrate_transfer, PredictionIntervals, TransferConformal,
    )
    backtest = pd.DataFrame({
        "unique_id": ["a", "a", "a", "a"],
        "ds":        [2, 3, 1, 2],
        "cutoff":    [1, 1, 0, 0],
        "y":         [3.0, 5.0, 2.0, 4.0],
        "m":         [1.0, 7.0, 3.0, 3.0],
    })
    pi = PredictionIntervals(n_windows=2, h=1)
    tc = TransferConformal(method="recalibrate")
    result = _recalibrate_transfer(
        new_df=backtest,
        prediction_intervals=pi,
        tc=tc,
        backtest_results=backtest,
        model_names=["m"],
        target_col="y",
    )
    assert result.signed is True
    # Signed residuals: y - pred = [3-1, 5-7, 2-3, 4-3] = [2, -2, -1, 1]
    vals = list(result.cs_df["m"])
    assert any(v < 0 for v in vals), "Signed residuals should include negative values"


def test_add_signed_transfer_intervals_bias_warning():
    import pandas as pd
    import warnings as _warnings
    from mlforecast.conformal_prediction import _add_signed_transfer_intervals

    horizon = 2
    # All-negative scores → interval entirely below point forecast
    cs_df = pd.DataFrame({"m": [-5.0, -4.0, -6.0, -5.5, -4.5, -6.5, -5.0, -4.8]})
    fcst_df = pd.DataFrame({
        "unique_id": ["a", "a"],
        "ds": [1, 2],
        "m": [10.0, 10.0],
    })

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _add_signed_transfer_intervals(
            fcst_df, cs_df, model_names=["m"], level=[90], horizon=horizon
        )

    assert any("over-predicts" in str(w.message) for w in caught), (
        "Expected a bias warning when q_hi < 0 (interval below point forecast)"
    )


# ---------------------------------------------------------------------------
# Task 6: dispatch integration
# ---------------------------------------------------------------------------
def test_recalibrate_step_size_param():
    """TransferConformal(step_size=2) runs without error and produces valid intervals."""
    n, h = 5, 3
    src = generate_daily_series(n, min_length=60, max_length=60, seed=50)
    src["unique_id"] = "src_" + src["unique_id"].astype(str)
    tgt = generate_daily_series(n, min_length=40, max_length=40, seed=51)
    tgt["unique_id"] = "tgt_" + tgt["unique_id"].astype(str)

    mlf = MLForecast(
        models=lightgbm.LGBMRegressor(n_estimators=10, random_state=0, verbosity=-1),
        lags=[1],
        freq="D",
        num_threads=1,
    )
    tc = TransferConformal(method="recalibrate", step_size=2)
    mlf.fit(
        src,
        prediction_intervals=PredictionIntervals(n_windows=3, h=h, transfer=tc),
    )

    preds = mlf.transfer(tgt).predict(h=h, level=[90])
    assert "LGBMRegressor-lo-90" in preds.columns
    assert np.isfinite(preds["LGBMRegressor-lo-90"].to_numpy()).all()
    assert (preds["LGBMRegressor-lo-90"] <= preds["LGBMRegressor-hi-90"]).all()


def _dynamic_exog_setup(method, case="dynamic", engine="pandas"):
    """Source model fit with a dynamic exog `u`, plus target history and future X_df."""
    rng = np.random.default_rng(0)
    n = 60
    frames = []
    for i in range(10):
        u = rng.normal(size=n)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": pd.date_range("2000-01-01", periods=n),
                    "u": u,
                    "y": np.cumsum(u) + rng.normal(size=n),
                    "s": float(i),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    cols = {"id_col": "unique_id", "time_col": "ds", "target_col": "y"}
    static_features = ["s"] if case == "static" else []
    if case != "static":
        df = df.drop(columns="s")
    if case == "renamed":
        df = df.rename(columns={"unique_id": "item", "ds": "date", "y": "sales"})
        cols = {"id_col": "item", "time_col": "date", "target_col": "sales"}
    by_id = df.groupby(cols["id_col"])
    train = by_id.head(30)
    new_df = by_id.head(50)
    X_df = by_id.nth(list(range(50, 55)))[[cols["id_col"], cols["time_col"], "u"]]
    if engine == "polars":
        train, new_df, X_df = (pl.from_pandas(d) for d in (train, new_df, X_df))
    freq = "1d" if engine == "polars" else "D"
    mlf = MLForecast(models=LinearRegression(), freq=freq, lags=[1])
    mlf.fit(
        train,
        static_features=static_features,
        prediction_intervals=PredictionIntervals(n_windows=5, h=5, transfer=method),
        **cols,
    )
    return mlf, new_df, X_df


def _check_dynamic_exog_preds(preds):
    assert len(preds) == 50
    lo = preds["LinearRegression-lo-90"].to_numpy()
    hi = preds["LinearRegression-hi-90"].to_numpy()
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert (lo <= hi).all()


@pytest.mark.parametrize("case", ["dynamic", "static", "renamed"])
@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_transfer_with_dynamic_exog(case, method):
    """Transfer intervals work when the source model uses a dynamic exog (#721)."""
    mlf, new_df, X_df = _dynamic_exog_setup(method, case)
    preds = mlf.transfer(new_df).predict(h=5, X_df=X_df, level=[90])
    _check_dynamic_exog_preds(preds)


@pytest.mark.parametrize("method", ["recalibrate", "scale_aligned"])
def test_transfer_with_dynamic_exog_polars(method):
    mlf, new_df, X_df = _dynamic_exog_setup(method, engine="polars")
    preds = mlf.transfer(new_df).predict(h=5, X_df=X_df, level=[90])
    assert isinstance(preds, pl.DataFrame)
    _check_dynamic_exog_preds(preds)


@pytest.mark.parametrize("method", ["recalibrate", "error_scaled"])
def test_transfer_df_missing_dynamic_exog_raises(method):
    mlf, new_df, _ = _dynamic_exog_setup(method)
    with pytest.raises(ValueError, match=r"`df` must contain .*\['u'\]"):
        mlf.transfer(new_df.drop(columns="u"))


# ---------------------------------------------------------------------------
# MLForecast.transfer
# ---------------------------------------------------------------------------
LR = "LinearRegression"


@cache
def _transfer_setup(method, pi_method="conformal_error", transforms=False):
    src = generate_daily_series(8, min_length=80, max_length=80, seed=0)
    tgt = generate_daily_series(4, min_length=50, max_length=50, seed=1)
    tgt["unique_id"] = "tgt_" + tgt["unique_id"].astype(str)
    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 2],
        lag_transforms={1: [RollingMean(window_size=3)]} if transforms else None,
        target_transforms=[Differences([1])] if transforms else None,
    )
    fcst.fit(
        src,
        prediction_intervals=PredictionIntervals(
            n_windows=5, h=3, method=pi_method, transfer=method
        ),
    )
    return fcst, tgt


@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_transfer_update_keeps_source(method):
    fcst, tgt = _transfer_setup(method)
    obs = tgt.groupby("unique_id", observed=True).tail(2)
    train = tgt.drop(obs.index)
    expected = fcst.predict(3, level=[80])
    transferred = fcst.transfer(train)
    transferred.update(obs)
    preds = transferred.predict(3, level=[80])
    assert preds["ds"].min() > obs["ds"].max()
    assert preds[[f"{LR}-lo-80", f"{LR}-hi-80"]].notna().all().all()
    pd.testing.assert_frame_equal(fcst.predict(3, level=[80]), expected)


@pytest.mark.parametrize("pi_method", ["conformal_error", "conformal_distribution"])
@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_transfer_ids_only_selects_forecasts(pi_method, method):
    fcst, tgt = _transfer_setup(method, pi_method)
    transferred = fcst.transfer(tgt)
    ids = ["tgt_id_2", "tgt_id_0"]
    full = transferred.predict(3, level=[80, 90])
    subset = transferred.predict(3, level=[80, 90], ids=ids)
    expected = full[full["unique_id"].isin(ids)]
    pd.testing.assert_frame_equal(
        subset.sort_values(["unique_id", "ds"]).reset_index(drop=True),
        expected.sort_values(["unique_id", "ds"]).reset_index(drop=True),
    )


@pytest.mark.parametrize("transforms", [False, True])
@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_transfer_chained_adapts_from_source(method, transforms):
    fcst, tgt = _transfer_setup(method, transforms=transforms)
    # other series, lengths and scale, so leftover state from them would show
    first_df = generate_daily_series(3, min_length=60, max_length=70, seed=2)
    first_df["y"] = first_df["y"] * 10 + 100
    chained = fcst.transfer(first_df).transfer(tgt)
    pd.testing.assert_frame_equal(
        chained.predict(3, level=[80]),
        fcst.transfer(tgt).predict(3, level=[80]),
    )


def test_transfer_models_are_not_shared():
    fcst, tgt = _transfer_setup("recalibrate")
    expected = fcst.predict(3)
    transferred = fcst.transfer(tgt)
    transferred.models_[LR] = LinearRegression().fit(np.ones((2, 2)), [0.0, 0.0])
    pd.testing.assert_frame_equal(fcst.predict(3), expected)


@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_transfer_save_load(method, tmp_path):
    fcst, tgt = _transfer_setup(method)
    transferred = fcst.transfer(tgt)
    transferred.save(tmp_path)
    loaded = MLForecast.load(tmp_path)
    pd.testing.assert_frame_equal(
        loaded.predict(3, level=[80]), transferred.predict(3, level=[80])
    )


def test_transfer_without_transfer_method_save_load(tmp_path):
    fcst, tgt = _transfer_setup(None)
    fcst.transfer(tgt).save(tmp_path)
    with pytest.raises(ValueError, match=r"PredictionIntervals\(transfer=\.\.\.\)"):
        MLForecast.load(tmp_path).predict(3, level=[80])
