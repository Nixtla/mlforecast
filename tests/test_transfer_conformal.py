import lightgbm
import numpy as np
import pandas as pd
import pytest

from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean
from mlforecast.utils import PredictionIntervals, TransferConformal, generate_daily_series


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
            transfer_conformal=method,
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
        prediction_intervals=PredictionIntervals(n_windows=3, h=7),
    )

    preds = mlf.predict(
        h=7, level=[90], new_df=target_train, transfer_conformal="error_scaled"
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
            transfer_conformal="error_scaled",
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
    mlf.fit(source, prediction_intervals=PredictionIntervals(n_windows=2, h=h))
    preds = mlf.predict(
        h=h, level=[90], new_df=target,
        transfer_conformal=TransferConformal(method="error_scaled", n_windows=1),
    )
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
    mlf.fit(source, prediction_intervals=PredictionIntervals(n_windows=2, h=h))
    with pytest.raises(ValueError, match="requires at least 2"):
        mlf.predict(
            h=h, level=[90], new_df=target,
            transfer_conformal=TransferConformal(method="recalibrate", n_windows=1),
        )


def test_recalibrate_n_windows_default_unchanged(transfer_cp_setup):
    """Omitting n_windows uses pi.n_windows (same result as explicit None)."""
    mlf, target_train, _ = transfer_cp_setup
    preds_default = mlf.predict(h=HORIZON, level=[90], new_df=target_train,
                                transfer_conformal=TransferConformal(method="recalibrate"))
    preds_none = mlf.predict(h=HORIZON, level=[90], new_df=target_train,
                             transfer_conformal=TransferConformal(method="recalibrate", n_windows=None))
    pd.testing.assert_frame_equal(preds_default, preds_none)


# ---------------------------------------------------------------------------
# Item 4: ESS warning test
# ---------------------------------------------------------------------------

def test_ess_no_warning_identical_distributions(transfer_cp_setup):
    """Identical source/target distributions should not trigger ESS warning."""
    import warnings as _warnings
    mlf, target_train, _ = transfer_cp_setup
    with _warnings.catch_warnings(record=True) as record:
        _warnings.simplefilter("always")
        mlf.predict(h=HORIZON, level=[90], new_df=target_train,
                    transfer_conformal=TransferConformal(method="weighted_conformal"))
    ess_warnings = [w for w in record if "ESS" in str(w.message)]
    assert len(ess_warnings) == 0, f"Unexpected ESS warnings: {ess_warnings}"
