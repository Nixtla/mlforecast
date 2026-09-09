"""Artifacts saved by mlforecast 1.1.0 must keep loading and predicting.

The pickles under ``fixtures/v1_1_0`` were produced by 1.1.0, before
``TimeSeries`` grew class-level defaults for its fit settings.
"""

from pathlib import Path

import cloudpickle
import pandas as pd
import pytest

from mlforecast import MLForecast

FIXTURES = Path(__file__).parent / "fixtures" / "v1_1_0"
CASES = ["recursive", "max_horizon", "prediction_intervals", "pooled_diffs"]


@pytest.fixture(scope="module")
def update_df():
    return pd.read_parquet(FIXTURES / "update_df.parquet")


@pytest.mark.parametrize("case", CASES)
def test_predict_matches_saved_output(case):
    fcst = MLForecast.load(str(FIXTURES / case))
    expected = pd.read_parquet(FIXTURES / case / "expected_predict.parquet")
    pd.testing.assert_frame_equal(fcst.predict(4), expected)


@pytest.mark.parametrize("case", CASES)
def test_update_then_predict_matches_saved_output(case, update_df):
    fcst = MLForecast.load(str(FIXTURES / case))
    fcst.update(update_df)
    expected = pd.read_parquet(
        FIXTURES / case / "expected_predict_after_update.parquet"
    )
    pd.testing.assert_frame_equal(fcst.predict(4), expected)


def test_standalone_timeseries_pickle():
    with open(FIXTURES / "ts.pkl", "rb") as f:
        ts = cloudpickle.load(f)
    with open(FIXTURES / "ts_models.pkl", "rb") as f:
        models = cloudpickle.load(f)
    preds = ts.predict(models=models, horizon=4)
    expected = pd.read_parquet(FIXTURES / "ts_expected_predict.parquet")
    pd.testing.assert_frame_equal(preds, expected)


def test_fit_only_instance_falls_back_to_class_defaults():
    # LightGBMCV and the distributed partitions only run `_fit`, so their
    # pickles lack the settings `fit_transform` stores
    with open(FIXTURES / "ts.pkl", "rb") as f:
        ts = cloudpickle.load(f)
    with open(FIXTURES / "ts_models.pkl", "rb") as f:
        models = cloudpickle.load(f)
    for name in ("as_numpy", "max_horizon", "_horizons"):
        ts.__dict__.pop(name, None)
    preds = ts.predict(models=models, horizon=4)
    expected = pd.read_parquet(FIXTURES / "ts_expected_predict.parquet")
    pd.testing.assert_frame_equal(preds, expected)
