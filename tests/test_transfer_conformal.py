import lightgbm
import numpy as np
import pandas as pd
import pytest

from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean
from mlforecast.utils import PredictionIntervals, generate_daily_series


HORIZON = 14
N_WINDOWS = 10
N_SOURCE_SERIES = 45
N_TARGET_SERIES = 35
LEVELS = [80, 90, 95]
TRANSFER_METHODS = [
    "recalibrate",
    "scale_aligned",
    "error_scaled",
    "weighted_conformal",
    "scale_aligned_weighted",
]
MODEL = "LGBMRegressor"
_PREDICTION_CACHE: dict[tuple, pd.DataFrame] = {}


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
    mlf.fit(
        source_train,
        prediction_intervals=PredictionIntervals(
            n_windows=N_WINDOWS,
            h=HORIZON,
            method="weighted_conformal_error",
            scale_estimator="mad",
        ),
    )

    return (
        mlf,
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


def _predict_transfer(
    transfer_cp_setup: tuple[MLForecast, pd.DataFrame, pd.DataFrame],
    method: str,
) -> pd.DataFrame:
    mlf, target_train, _ = transfer_cp_setup
    cache_key = (id(mlf), method)
    if cache_key not in _PREDICTION_CACHE:
        _PREDICTION_CACHE[cache_key] = mlf.predict(
            h=HORIZON,
            level=LEVELS,
            new_df=target_train,
            transfer_conformal_method=method,
        )
    return _PREDICTION_CACHE[cache_key].copy()


@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_coverage_within_tolerance(transfer_cp_setup, method):
    _, _, target_test = transfer_cp_setup
    preds = _predict_transfer(transfer_cp_setup, method)
    coverage = compute_coverage(preds, target_test, MODEL, LEVELS)

    for level, empirical in coverage.items():
        nominal = level / 100
        assert abs(empirical - nominal) <= 0.05, (
            f"{method} level {level} empirical coverage {empirical:.3f} "
            f"differs from nominal {nominal:.3f} by more than 0.05."
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
        prediction_intervals=PredictionIntervals(n_windows=3, h=7),
    )

    preds = mlf.predict(
        h=7, level=[90], new_df=target_train, transfer_conformal_method="error_scaled"
    )
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
        prediction_intervals=PredictionIntervals(n_windows=2, h=1),
    )

    with pytest.raises(ValueError, match="requires PredictionIntervals"):
        mlf.predict(
            h=7,
            level=[90],
            new_df=target_train,
            transfer_conformal_method="error_scaled",
        )


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
