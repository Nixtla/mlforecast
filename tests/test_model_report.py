import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression
from utilsforecast.losses import smape

from mlforecast import MLForecast
from mlforecast.model_report import FitReport, ModelFitReport
from mlforecast.utils import PredictionIntervals


def _data(backend):
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 5 + ["b"] * 5,
            "ds": list(range(5)) * 2,
            "y": np.arange(10, dtype=np.float64),
        }
    )
    return pl.from_pandas(df) if backend == "polars" else df


def _cv_data(n_steps=12):
    return pd.DataFrame(
        {
            "unique_id": ["a"] * n_steps + ["b"] * n_steps,
            "ds": list(range(n_steps)) * 2,
            "y": np.arange(2 * n_steps, dtype=np.float64),
        }
    )


def test_preprocess_report_for_pandas():
    df = _data("pandas")
    fcst = MLForecast(
        models=LinearRegression(), freq=1, lags=[1], report_level="detailed"
    )

    fcst.preprocess(df, return_X_y=True)

    report = fcst.feature_preparation_report_
    assert report.input_backend == "pandas"
    assert report.output_backend == "pandas"
    assert report.input_shape == (10, 3)
    assert report.output_shape == (8, 1)
    assert report.input_dtypes["y"] == "float64"
    assert report.input_memory_bytes is not None
    assert report.output_memory_bytes is not None
    assert "kept pandas representation" in report.operations


def test_preprocess_report_can_measure_memory_and_numpy_conversion():
    df = _data("polars")
    fcst = MLForecast(
        models=LinearRegression(), freq=1, lags=[1], report_level="detailed"
    )

    fcst.preprocess(df, return_X_y=True, as_numpy=True)

    report = fcst.feature_preparation_report_
    assert report.input_backend == "polars"
    assert report.output_backend == "numpy"
    assert report.output_shape == (8, 1)
    assert report.output_dtypes == {"feature_0": "float64"}
    assert report.input_memory_bytes is not None
    assert report.output_memory_bytes is not None
    assert "converted polars -> numpy because as_numpy=True" in report.operations


def test_fit_and_predict_reports():
    fcst = MLForecast(
        models=LinearRegression(), freq=1, lags=[1], report_level="detailed"
    )

    fcst.fit(_data("pandas"))

    assert fcst.fit_report_.elapsed_seconds >= 0
    assert fcst.model_fit_report_.fit_calls == 1
    assert fcst.model_fit_report_.model_seconds.keys() == {"LinearRegression"}
    assert fcst.model_fit_report_.elapsed_seconds >= sum(
        fcst.model_fit_report_.model_seconds.values()
    )

    fcst.predict(2)

    assert fcst.predict_report_.elapsed_seconds >= 0
    assert fcst.predict_report_.horizon == 2


def test_fit_report_includes_interval_calibration_fits():
    fcst = MLForecast(
        models=LinearRegression(), freq=1, lags=[1], report_level="detailed"
    )

    fcst.fit(
        _data("pandas"),
        prediction_intervals=PredictionIntervals(n_windows=2, h=1),
    )

    report = fcst.fit_report_
    assert report.model_fit_report is not None
    assert report.calibration_model_fit_report is not None
    assert report.final_model_fit_report is not None
    assert report.calibration_model_fit_report.fit_calls == 1
    assert report.final_model_fit_report.fit_calls == 1
    assert report.model_fit_report.fit_calls == 2
    assert report.model_fit_report.model_seconds["LinearRegression"] >= (
        report.calibration_model_fit_report.model_seconds["LinearRegression"]
        + report.final_model_fit_report.model_seconds["LinearRegression"]
    )


def test_reports_are_off_by_default():
    fcst = MLForecast(models=LinearRegression(), freq=1, lags=[1])

    fcst.fit(_data("pandas"))

    assert not hasattr(fcst, "feature_preparation_report_")
    assert not hasattr(fcst, "model_fit_report_")
    assert not hasattr(fcst, "fit_report_")


def test_default_reports_do_not_call_instrumentation(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("instrumentation must be disabled by default")

    monkeypatch.setattr("mlforecast.forecast.perf_counter", fail_if_called)
    monkeypatch.setattr("mlforecast.forecast.get_process_rss_bytes", fail_if_called)
    monkeypatch.setattr("mlforecast.model_report._memory_bytes", fail_if_called)
    fcst = MLForecast(models=LinearRegression(), freq=1, lags=[1])

    fcst.fit(_data("pandas"))


def test_default_cross_validation_does_not_calculate_metrics(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("metrics must be opt-in")

    monkeypatch.setattr(
        "mlforecast.forecast.summarize_model_metrics", fail_if_called
    )
    fcst = MLForecast(models=LinearRegression(), freq=1, lags=[1])

    fcst.cross_validation(_cv_data(), n_windows=3, h=1)

    assert not hasattr(fcst, "cv_model_metrics_")


def _mean_error(df, models, id_col="unique_id", target_col="y"):
    return pd.DataFrame(
        {
            id_col: ["all"],
            **{
                model: [(df[model] - df[target_col]).mean()]
                for model in models
            },
        }
    )


def test_basic_reports_skip_memory_and_rss():
    fcst = MLForecast(
        models=LinearRegression(), freq=1, lags=[1], report_level="basic"
    )

    fcst.fit(_data("pandas"))
    fcst.predict(1)

    assert fcst.feature_preparation_report_.input_memory_bytes is None
    assert fcst.feature_preparation_report_.output_memory_bytes is None
    assert fcst.model_fit_report_.rss_start_bytes is None
    assert fcst.model_fit_report_.rss_end_bytes is None
    assert fcst.fit_report_.rss_start_bytes is None
    assert fcst.predict_report_.rss_end_bytes is None


def test_invalid_report_level_is_rejected():
    with pytest.raises(ValueError, match="report_level"):
        MLForecast(models=LinearRegression(), freq=1, report_level="verbose")


def test_model_fit_report_diff_has_delta_percent_and_threshold():
    baseline = ModelFitReport(1.0, 1, {"model": 0.5}, None, None)
    candidate = ModelFitReport(1.2, 1, {"model": 0.7}, None, None)

    comparison = candidate.diff(baseline, relative_threshold=0.1)

    elapsed = comparison.metrics["elapsed_seconds"]
    assert elapsed.delta == pytest.approx(0.2)
    assert elapsed.delta_percent == pytest.approx(20.0)
    assert elapsed.exceeds_threshold


def test_report_diff_handles_zero_baseline_with_absolute_threshold():
    baseline = ModelFitReport(0.0, 1, {}, None, None)
    candidate = ModelFitReport(1.0, 1, {}, None, None)

    comparison = candidate.diff(baseline, absolute_threshold=0.5)

    elapsed = comparison.metrics["elapsed_seconds"]
    assert elapsed.delta_percent is None
    assert elapsed.exceeds_threshold


def test_report_diff_surfaces_added_and_removed_model_metrics():
    baseline = ModelFitReport(1.0, 1, {"old": 0.5}, None, None)
    candidate = ModelFitReport(1.0, 1, {"new": 0.7}, None, None)

    comparison = candidate.diff(baseline)

    assert comparison.metrics["model_seconds.old"].change_kind == "removed"
    assert comparison.metrics["model_seconds.new"].change_kind == "added"


def test_report_diff_compares_rss_delta_not_process_position():
    baseline = FitReport(1.0, 100, 120, None, None, None)
    candidate = FitReport(1.0, 1_000, 1_020, None, None, None)

    comparison = candidate.diff(baseline)

    assert "rss_start_bytes" not in comparison.metrics
    assert "rss_end_bytes" not in comparison.metrics
    assert comparison.metrics["rss_delta_bytes"].delta == 0


def test_fitted_values_get_explicit_smape_metrics():
    fcst = MLForecast(
        models=LinearRegression(), freq=1, lags=[1], model_metrics=[smape]
    )

    fcst.fit(_data("pandas"), fitted=True)

    assert set(fcst.model_metrics_["LinearRegression"]) == {"smape"}


def test_empty_model_metrics_disables_quality_report():
    fcst = MLForecast(
        models=LinearRegression(), freq=1, lags=[1], model_metrics=[]
    )

    fcst.fit(_data("pandas"), fitted=True)

    assert not hasattr(fcst, "model_metrics_")


def test_multiple_custom_model_metrics_are_retained():
    fcst = MLForecast(
        models=LinearRegression(),
        freq=1,
        lags=[1],
        model_metrics=[smape, _mean_error],
    )

    fcst.fit(_data("pandas"), fitted=True)

    assert set(fcst.model_metrics_["LinearRegression"]) == {"smape", "_mean_error"}


def test_duplicate_metric_names_are_rejected():
    with pytest.raises(ValueError, match="unique names"):
        MLForecast(
            models=LinearRegression(), freq=1, lags=[1], model_metrics=[smape, smape]
        )


def test_metrics_are_cleared_when_a_later_fit_does_not_calculate_them():
    fcst = MLForecast(
        models=LinearRegression(), freq=1, lags=[1], model_metrics=[smape]
    )

    fcst.fit(_data("pandas"), fitted=True)
    fcst.fit(_data("pandas"), fitted=False)

    assert not hasattr(fcst, "model_metrics_")


def test_metrics_are_cleared_when_a_later_cross_validation_disables_them():
    fcst = MLForecast(
        models=LinearRegression(), freq=1, lags=[1], model_metrics=[smape]
    )

    fcst.cross_validation(_cv_data(), n_windows=3, h=1)
    fcst.model_metrics = []
    fcst.cross_validation(_cv_data(), n_windows=3, h=1)

    for attribute in (
        "cv_model_metrics_",
        "cv_model_metrics_by_fold_",
        "cv_model_metrics_mean_",
    ):
        assert not hasattr(fcst, attribute)


def test_cross_validation_keeps_every_refit_report_and_metrics():
    df = _cv_data()
    fcst = MLForecast(
        models=LinearRegression(),
        freq=1,
        lags=[1],
        report_level="basic",
        model_metrics=[smape],
    )

    fcst.cross_validation(df, n_windows=3, h=1)

    assert len(fcst.cv_fit_reports_) == 3
    assert len(fcst.cv_feature_preparation_reports_) == 3
    assert len(fcst.cv_model_fit_reports_) == 3
    assert fcst.cv_report_folds_ == [0, 1, 2]
    assert set(fcst.cv_model_metrics_["LinearRegression"]) == {"smape"}
    assert set(fcst.cv_model_metrics_mean_["LinearRegression"]) == {"smape"}
    assert set(fcst.cv_model_metrics_by_fold_) == {0, 1, 2}
    assert all(
        set(metrics["LinearRegression"]) == {"smape"}
        for metrics in fcst.cv_model_metrics_by_fold_.values()
    )
    fold_smapes = [
        metrics["LinearRegression"]["smape"]
        for metrics in fcst.cv_model_metrics_by_fold_.values()
    ]
    assert fcst.cv_model_metrics_mean_["LinearRegression"]["smape"] == pytest.approx(
        np.mean(fold_smapes)
    )


def test_cross_validation_records_the_fold_for_each_refit_report():
    fcst = MLForecast(
        models=LinearRegression(), freq=1, lags=[1], report_level="basic"
    )

    fcst.cross_validation(_cv_data(), n_windows=3, h=1, refit=2)

    assert fcst.cv_report_folds_ == [0, 2]


def test_cross_validation_with_intervals_keeps_outer_reports():
    fcst = MLForecast(
        models=LinearRegression(),
        freq=1,
        lags=[1],
        report_level="basic",
        model_metrics=[smape],
    )

    fcst.cross_validation(
        _cv_data(),
        n_windows=3,
        h=1,
        prediction_intervals=PredictionIntervals(n_windows=2, h=1),
    )

    assert len(fcst.cv_fit_reports_) == 3
    assert len(fcst.cv_feature_preparation_reports_) == 3
    assert len(fcst.cv_model_fit_reports_) == 3
    assert set(fcst.cv_model_metrics_by_fold_) == {0, 1, 2}


def test_prediction_interval_calibration_does_not_leak_cv_state():
    fcst = MLForecast(
        models=LinearRegression(),
        freq=1,
        lags=[1],
        report_level="basic",
        model_metrics=[smape],
    )

    fcst.fit(
        _cv_data(), prediction_intervals=PredictionIntervals(n_windows=2, h=1)
    )

    for attribute in (
        "cv_fit_reports_",
        "cv_feature_preparation_reports_",
        "cv_model_fit_reports_",
        "cv_report_folds_",
        "cv_models_",
        "cv_fitted_values_",
        "cv_model_metrics_",
        "cv_model_metrics_by_fold_",
        "cv_model_metrics_mean_",
    ):
        assert not hasattr(fcst, attribute)


def test_prediction_interval_calibration_preserves_existing_cv_state():
    fcst = MLForecast(
        models=LinearRegression(),
        freq=1,
        lags=[1],
        report_level="basic",
        model_metrics=[smape],
    )
    df = _cv_data()
    fcst.cross_validation(df, n_windows=3, h=1, fitted=True)
    previous_state = {
        attribute: getattr(fcst, attribute)
        for attribute in (
            "cv_fit_reports_",
            "cv_feature_preparation_reports_",
            "cv_model_fit_reports_",
            "cv_report_folds_",
            "cv_models_",
            "cv_fitted_values_",
            "cv_model_metrics_",
            "cv_model_metrics_by_fold_",
            "cv_model_metrics_mean_",
        )
    }

    fcst.fit(df, prediction_intervals=PredictionIntervals(n_windows=2, h=1))

    for attribute, value in previous_state.items():
        assert getattr(fcst, attribute) is value
