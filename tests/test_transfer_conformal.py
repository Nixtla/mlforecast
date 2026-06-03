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
_PREDICTION_CACHE: dict[str, pd.DataFrame] = {}


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
    if method not in _PREDICTION_CACHE:
        mlf, target_train, _ = transfer_cp_setup
        _PREDICTION_CACHE[method] = mlf.predict(
            h=HORIZON,
            level=LEVELS,
            new_df=target_train,
            transfer_conformal_method=method,
        )
    return _PREDICTION_CACHE[method].copy()


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
