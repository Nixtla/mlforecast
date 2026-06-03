import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlforecast import MLForecast
from mlforecast.forecast import _get_conformal_method
from mlforecast.lag_transforms import ExponentiallyWeightedMean
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals, generate_daily_series

warnings.simplefilter('ignore', UserWarning)


# ---------------------------------------------------------------------------
# Local fixtures (M4-data dependent)
# ---------------------------------------------------------------------------

@pytest.fixture
def fcst():
    return MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1),
        freq=1,
        lags=[24 * (i+1) for i in range(7)],
        lag_transforms={
            48: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        num_threads=1,
        target_transforms=[Differences([24])],
    )


@pytest.fixture
def fitted_fcst(fcst, setup_forecast_data):
    _, train, _ = setup_forecast_data
    fcst.fit(train, fitted=True)
    return fcst


@pytest.fixture
def predictions(fitted_fcst):
    return fitted_fcst.predict(48)


@pytest.fixture
def fcst_with_intervals(setup_forecast_data):
    """Forecast object fitted with prediction intervals."""
    _, train, _ = setup_forecast_data
    f = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1),
        freq=1,
        lags=[24 * (i+1) for i in range(7)],
        lag_transforms={
            48: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        num_threads=1,
        target_transforms=[Differences([24])],
    )
    f.fit(train, prediction_intervals=PredictionIntervals(n_windows=3, h=48))
    return f


@pytest.fixture
def predictions_w_intervals(fcst_with_intervals):
    """Predictions with prediction intervals."""
    return fcst_with_intervals.predict(48, level=[50, 80, 95])


# ---------------------------------------------------------------------------
# Conformal method validation
# ---------------------------------------------------------------------------

def test_conformal_method():
    with pytest.raises(ValueError):
        _get_conformal_method('my_method')


# ---------------------------------------------------------------------------
# Prediction interval integration tests
# ---------------------------------------------------------------------------

def test_prediction_intervals_lower_horizon(fcst_with_intervals):
    """Test we can forecast horizon lower than h with prediction intervals."""
    preds_h1 = fcst_with_intervals.predict(1, level=[50, 80, 95])
    preds_h30 = fcst_with_intervals.predict(30, level=[50, 80, 95])

    monotonic_count = preds_h1.filter(regex='lo|hi').apply(
        lambda x: x.is_monotonic_increasing,
        axis=1
    ).sum()
    assert monotonic_count == len(preds_h1)

    monotonic_count = preds_h30.filter(regex='lo|hi').apply(
        lambda x: x.is_monotonic_increasing,
        axis=1
    ).sum()
    assert monotonic_count == len(preds_h30)


def test_prediction_intervals_error_conditions(fcst_with_intervals):
    """Test error conditions for prediction intervals."""
    with pytest.raises(Exception):
        fcst_with_intervals.predict(49, level=[68])


def test_recover_point_forecasts(predictions, predictions_w_intervals):
    """Test we can recover point forecasts from interval predictions."""
    pd.testing.assert_frame_equal(
        predictions,
        predictions_w_intervals[predictions.columns]
    )


def test_recover_mean_forecasts_level_zero(predictions, fcst_with_intervals):
    """Test we can recover mean forecasts with level 0."""
    level_zero_preds = fcst_with_intervals.predict(48, level=[0])
    np.testing.assert_allclose(
        predictions['LGBMRegressor'].values,
        level_zero_preds['LGBMRegressor-lo-0'].values,
    )


def test_prediction_intervals_monotonicity(predictions_w_intervals):
    """Test monotonicity of prediction intervals."""
    monotonic_count = predictions_w_intervals.filter(regex='lo|hi').apply(
        lambda x: x.is_monotonic_increasing,
        axis=1
    ).sum()
    assert monotonic_count == len(predictions_w_intervals)


# ---------------------------------------------------------------------------
# Weighted conformal prediction tests
# ---------------------------------------------------------------------------

def test_weighted_conformal_method_validation():
    """weighted_conformal_error and _distribution should be accepted; invalid names rejected."""
    PredictionIntervals(method="weighted_conformal_error")
    PredictionIntervals(method="weighted_conformal_distribution")
    with pytest.raises(ValueError):
        PredictionIntervals(method="bad_method")


def test_weighted_conformal_features_stored(weighted_conformal_setup):
    """After fitting with weighted_conformal method, _cs_df should contain feature columns."""
    fcst, _, _, _ = weighted_conformal_setup
    non_feat_cols = {fcst.ts.id_col, fcst.ts.time_col, "cutoff"} | set(fcst.models.keys())
    feat_cols = [c for c in fcst._cs_df.columns if c not in non_feat_cols]
    assert len(feat_cols) > 0, "_cs_df must contain feature columns for DRE"


def test_weighted_conformal_uniform_weights_matches_standard(weighted_conformal_setup):
    """Uniform weights should produce intervals close to the unweighted method."""
    fcst, series, h, n_windows = weighted_conformal_setup
    n_series = fcst.ts.ga.n_groups
    n_cal_rows = len(fcst._cs_df)

    preds_std = fcst.predict(h, level=[80])

    uniform_weights = np.ones(n_cal_rows)
    preds_w = fcst.predict(h, level=[80], covariate_shift_weights=uniform_weights)

    pd.testing.assert_series_equal(
        preds_std["LGBMRegressor"], preds_w["LGBMRegressor"]
    )
    assert "LGBMRegressor-lo-80" in preds_w.columns
    assert "LGBMRegressor-hi-80" in preds_w.columns


def test_weighted_conformal_user_array(weighted_conformal_setup):
    """User-supplied weight array flows through; intervals are finite and ordered."""
    fcst, _, h, _ = weighted_conformal_setup
    n_cal_rows = len(fcst._cs_df)
    weights = np.random.default_rng(42).exponential(scale=1.0, size=n_cal_rows)
    preds = fcst.predict(h, level=[80, 95], covariate_shift_weights=weights)

    assert "LGBMRegressor-lo-80" in preds.columns
    assert preds["LGBMRegressor-lo-80"].notna().all()
    assert preds["LGBMRegressor-hi-80"].notna().all()
    assert (preds["LGBMRegressor-lo-80"] <= preds["LGBMRegressor-hi-80"]).all()


def test_weighted_conformal_callable_weights(weighted_conformal_setup):
    """Callable receives source feature matrix and returns weights; intervals are valid."""
    fcst, _, h, _ = weighted_conformal_setup
    weight_fn = lambda X: np.ones(len(X)) if X is not None else np.ones(len(fcst._cs_df))
    preds = fcst.predict(h, level=[80], covariate_shift_weights=weight_fn)

    assert "LGBMRegressor-lo-80" in preds.columns
    assert preds["LGBMRegressor-lo-80"].notna().all()


# ---------------------------------------------------------------------------
# Density ratio estimation tests
# ---------------------------------------------------------------------------

def test_estimate_density_ratio():
    """estimate_density_ratio returns positive weights of correct shape."""
    from mlforecast.conformal_prediction import estimate_density_ratio

    rng = np.random.default_rng(0)
    src = rng.standard_normal((50, 4))
    tgt = rng.standard_normal((30, 4)) + 0.5
    weights = estimate_density_ratio(src, tgt)
    assert weights.shape == (50,)
    assert (weights > 0).all()


def test_estimate_density_ratio_gradient_boosting():
    """estimate_density_ratio works with gradient_boosting estimator."""
    from mlforecast.conformal_prediction import estimate_density_ratio

    rng = np.random.default_rng(1)
    src = rng.standard_normal((30, 3))
    tgt = rng.standard_normal((20, 3)) + 1.0
    weights = estimate_density_ratio(src, tgt, estimator="gradient_boosting")
    assert weights.shape == (30,)
    assert (weights > 0).all()


# ---------------------------------------------------------------------------
# Scale-aligned conformal prediction tests
# ---------------------------------------------------------------------------

def test_prediction_intervals_scale_estimator_validation():
    """scale_estimator accepts 'mad' and 'std'; rejects invalid values."""
    PredictionIntervals(scale_estimator="mad")
    PredictionIntervals(scale_estimator="std")
    PredictionIntervals(scale_estimator=None)
    with pytest.raises(ValueError, match="scale_estimator"):
        PredictionIntervals(scale_estimator="variance")


def test_compute_series_scales_mad():
    """_compute_series_scales returns correct MAD for a known signal."""
    from mlforecast.conformal_prediction import _compute_series_scales

    df = pd.DataFrame({"unique_id": ["s1"] * 5, "ds": range(5), "y": [0.0, 1, 3, 6, 10]})
    scales = _compute_series_scales(df, "unique_id", "ds", "y", method="mad")
    assert "s1" in scales
    assert scales["s1"] > 0


def test_compute_series_scales_std():
    """_compute_series_scales returns correct std for a known signal."""
    from mlforecast.conformal_prediction import _compute_series_scales

    rng = np.random.default_rng(0)
    y = rng.standard_normal(50).cumsum()
    df = pd.DataFrame({"unique_id": ["s1"] * 50, "ds": range(50), "y": y})
    scales = _compute_series_scales(df, "unique_id", "ds", "y", method="std")
    dy = np.diff(y)
    expected = float(np.std(dy, ddof=1))
    assert abs(scales["s1"] - expected) < 1e-10


def test_compute_series_scales_flat_series():
    """Flat series gets a positive floor rather than zero."""
    from mlforecast.conformal_prediction import _compute_series_scales

    df = pd.DataFrame({
        "unique_id": ["flat"] * 10 + ["noisy"] * 10,
        "ds": list(range(10)) * 2,
        "y": [5.0] * 10 + list(np.random.default_rng(0).standard_normal(10)),
    })
    scales = _compute_series_scales(df, "unique_id", "ds", "y", method="mad")
    assert scales["flat"] > 0


def test_compute_series_scales_short_series():
    """Single-observation series gets a positive scale without crashing."""
    from mlforecast.conformal_prediction import _compute_series_scales

    df = pd.DataFrame({
        "unique_id": ["short", "normal"] + ["normal"] * 9,
        "ds": [0] + list(range(10)),
        "y": [42.0] + list(np.arange(10, dtype=float)),
    })
    scales = _compute_series_scales(df, "unique_id", "ds", "y", method="mad")
    assert scales["short"] > 0


def test_compute_series_scales_polars():
    """_compute_series_scales works on a polars DataFrame."""
    from mlforecast.conformal_prediction import _compute_series_scales

    df_pd = pd.DataFrame({
        "unique_id": ["s1"] * 20,
        "ds": range(20),
        "y": np.arange(20, dtype=float) + np.random.default_rng(7).standard_normal(20),
    })
    df_pl = pl.from_pandas(df_pd)
    scales_pd = _compute_series_scales(df_pd, "unique_id", "ds", "y", method="mad")
    scales_pl = _compute_series_scales(df_pl, "unique_id", "ds", "y", method="mad")
    assert abs(scales_pd["s1"] - scales_pl["s1"]) < 1e-10


def test_apply_scale_alignment_ratio_correct():
    """_apply_scale_alignment multiplies residuals by (σ_target / σ_source) per series."""
    from mlforecast.conformal_prediction import _apply_scale_alignment

    cs_df = pd.DataFrame({
        "unique_id": ["s1", "s1", "s2", "s2"],
        "LGBMRegressor": [1.0, 2.0, 3.0, 4.0],
    })
    source_scales = {"s1": 2.0, "s2": 4.0}
    target_scales = np.array([8.0])
    result = _apply_scale_alignment(cs_df, ["LGBMRegressor"], "unique_id", source_scales, target_scales)
    expected_s1 = np.array([1.0, 2.0]) * (8.0 / 2.0)
    expected_s2 = np.array([3.0, 4.0]) * (8.0 / 4.0)
    np.testing.assert_allclose(result["LGBMRegressor"].to_numpy()[:2], expected_s1)
    np.testing.assert_allclose(result["LGBMRegressor"].to_numpy()[2:], expected_s2)


def test_apply_scale_alignment_no_mutation():
    """_apply_scale_alignment does not mutate its input cs_df."""
    from mlforecast.conformal_prediction import _apply_scale_alignment

    cs_df = pd.DataFrame({
        "unique_id": ["s1"] * 4,
        "LGBMRegressor": [1.0, 2.0, 3.0, 4.0],
    })
    original_vals = cs_df["LGBMRegressor"].to_numpy().copy()
    _apply_scale_alignment(cs_df, ["LGBMRegressor"], "unique_id", {"s1": 1.0}, np.array([5.0]))
    np.testing.assert_array_equal(cs_df["LGBMRegressor"].to_numpy(), original_vals)


def test_scale_aligned_source_scales_stored(scale_aligned_setup):
    """Fitting with scale_estimator stores _source_scales on PredictionIntervals."""
    fcst, _, _, _ = scale_aligned_setup
    assert fcst.prediction_intervals._source_scales is not None
    assert len(fcst.prediction_intervals._source_scales) == fcst.ts.ga.n_groups
    for v in fcst.prediction_intervals._source_scales.values():
        assert v > 0


def test_scale_aligned_transfer_large_target(scale_aligned_setup):
    """scale_aligned transfer produces intervals ~400× wider for a ×400 target."""
    fcst, source_series, target_series, h = scale_aligned_setup

    preds_src = fcst.predict(h, level=[80])
    src_width = (preds_src["LGBMRegressor-hi-80"] - preds_src["LGBMRegressor-lo-80"]).mean()

    preds_sa = fcst.predict(
        h, level=[80], new_df=target_series,
        transfer_conformal_method="scale_aligned",
    )
    assert "LGBMRegressor-lo-80" in preds_sa.columns
    assert preds_sa["LGBMRegressor-lo-80"].notna().all()
    assert (preds_sa["LGBMRegressor-lo-80"] <= preds_sa["LGBMRegressor-hi-80"]).all()

    sa_width = (preds_sa["LGBMRegressor-hi-80"] - preds_sa["LGBMRegressor-lo-80"]).mean()
    ratio = float(sa_width / src_width)
    assert ratio > 50, f"Expected scale-aligned intervals >> source intervals, got ratio={ratio:.1f}"


def test_scale_aligned_nontransfer_path_unchanged(scale_aligned_setup):
    """predict() without new_df is unaffected by scale_estimator."""
    fcst, source_series, _, h = scale_aligned_setup

    pi_plain = PredictionIntervals(method="conformal_error", n_windows=2, h=h)
    fcst_plain = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1, n_estimators=5),
        freq="D",
        lags=[1, 7],
        date_features=["dayofweek"],
        num_threads=1,
    )
    fcst_plain.fit(source_series, prediction_intervals=pi_plain)

    preds_with = fcst.predict(h, level=[80])
    preds_without = fcst_plain.predict(h, level=[80])

    pd.testing.assert_frame_equal(
        preds_with[["LGBMRegressor-lo-80", "LGBMRegressor-hi-80"]].reset_index(drop=True),
        preds_without[["LGBMRegressor-lo-80", "LGBMRegressor-hi-80"]].reset_index(drop=True),
        rtol=1e-5,
    )


def test_scale_aligned_weighted_composes(scale_aligned_setup):
    """scale_aligned_weighted combines scale alignment with DRE weights."""
    fcst, source_series, target_series, h = scale_aligned_setup

    pi_w = PredictionIntervals(
        method="weighted_conformal_error", n_windows=2, h=h, scale_estimator="mad"
    )
    fcst_w = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1, n_estimators=5),
        freq="D",
        lags=[1, 7],
        date_features=["dayofweek"],
        num_threads=1,
    )
    fcst_w.fit(source_series, prediction_intervals=pi_w)

    preds = fcst_w.predict(
        h, level=[80],
        new_df=target_series,
        transfer_conformal_method="scale_aligned_weighted",
    )
    assert "LGBMRegressor-lo-80" in preds.columns
    assert preds["LGBMRegressor-lo-80"].notna().all()
    assert (preds["LGBMRegressor-lo-80"] <= preds["LGBMRegressor-hi-80"]).all()
    assert fcst_w.prediction_intervals._target_scales is None


def test_scale_aligned_requires_scale_estimator():
    """scale_aligned transfer raises if model was fit without scale_estimator."""
    series = generate_daily_series(2, min_length=40, max_length=50, seed=3)
    fcst = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1, n_estimators=5),
        freq="D",
        lags=[1],
        num_threads=1,
    )
    fcst.fit(series, prediction_intervals=PredictionIntervals(method="conformal_error"))
    target_df = series.groupby("unique_id").tail(10).reset_index(drop=True)
    with pytest.raises(ValueError, match="scale_estimator"):
        fcst.predict(5, level=[80], new_df=target_df, transfer_conformal_method="scale_aligned")


# ---------------------------------------------------------------------------
# Transfer conformal prediction tests
# ---------------------------------------------------------------------------

def test_weighted_conformal_auto_dre(weighted_conformal_setup):
    """transfer_conformal_method='weighted_conformal' + new_df uses full-feature DRE."""
    fcst, series, h, _ = weighted_conformal_setup
    target_df = series.groupby("unique_id").tail(30).reset_index(drop=True)

    preds = fcst.predict(
        h,
        level=[80],
        new_df=target_df,
        transfer_conformal_method="weighted_conformal",
    )
    assert "LGBMRegressor-lo-80" in preds.columns
    assert preds["LGBMRegressor-lo-80"].notna().all()
    assert (preds["LGBMRegressor-lo-80"] <= preds["LGBMRegressor-hi-80"]).all()


def test_weighted_conformal_auto_dre_requires_weighted_method():
    """auto-DRE should fail gracefully when _cs_df has no feature columns."""
    series = generate_daily_series(2, min_length=40, max_length=50, seed=1)
    fcst = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1, n_estimators=5),
        freq="D",
        lags=[1],
        num_threads=1,
    )
    fcst.fit(series, prediction_intervals=PredictionIntervals(method="conformal_error"))
    target_df = series.groupby("unique_id").tail(10).reset_index(drop=True)
    with pytest.raises(ValueError, match="No feature columns"):
        fcst.predict(5, level=[80], new_df=target_df,
                     transfer_conformal_method="weighted_conformal")
