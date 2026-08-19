import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast


def _data(backend):
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 5 + ["b"] * 5,
            "ds": list(range(5)) * 2,
            "y": np.arange(10, dtype=np.float64),
        }
    )
    return pl.from_pandas(df) if backend == "polars" else df


def test_preprocess_report_for_pandas():
    df = _data("pandas")
    fcst = MLForecast(models=LinearRegression(), freq=1, lags=[1])

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
    fcst = MLForecast(models=LinearRegression(), freq=1, lags=[1])

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
    fcst = MLForecast(models=LinearRegression(), freq=1, lags=[1])

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
