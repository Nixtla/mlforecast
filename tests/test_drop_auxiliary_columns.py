"""Behavior of `MLForecast(drop_auxiliary_columns=...)`.

Split out of `bench_pipeline.py`, which is benchmarks only.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.utils import generate_daily_series


@pytest.fixture(scope="module")
def series():
    # three static features: two get dropped explicitly below, one must survive
    return generate_daily_series(
        n_series=20,
        min_length=100,
        max_length=200,
        n_static_features=3,
        static_as_categorical=False,
        equal_ends=True,
    )


def test_drop_auxiliary_columns_default_drops_groupby(series):
    """Default True should auto-drop all groupby columns but keep computed lag features."""
    statics = series.columns.drop(["unique_id", "ds", "y"]).tolist()
    groupby_col = statics[0]

    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1],
        lag_transforms={1: [RollingMean(7, groupby=[groupby_col])]},
    )
    fcst.fit(series, static_features=statics)

    assert groupby_col not in fcst.ts.features_order_
    assert any("rolling_mean" in f for f in fcst.ts.features_order_)

    preds = fcst.predict(7)
    assert groupby_col not in preds.columns


def test_drop_auxiliary_columns_false_keeps_groupby(series):
    """False should keep groupby columns in the model feature matrix."""
    statics = series.columns.drop(["unique_id", "ds", "y"]).tolist()
    groupby_col = statics[0]

    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1],
        lag_transforms={1: [RollingMean(7, groupby=[groupby_col])]},
        drop_auxiliary_columns=False,
    )
    fcst.fit(series, static_features=statics)

    assert groupby_col in fcst.ts.features_order_


def test_drop_auxiliary_columns_explicit_list(series):
    """An explicit list should drop exactly those columns regardless of groupby."""
    statics = series.columns.drop(["unique_id", "ds", "y"]).tolist()
    drop = statics[:2]

    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1, 7],
        drop_auxiliary_columns=drop,
    )
    fcst.fit(series, static_features=statics)

    for col in drop:
        assert col not in fcst.ts.features_order_
    for col in statics[2:]:
        assert col in fcst.ts.features_order_


def test_drop_auxiliary_columns_predict_excludes_column(series):
    """Dropped columns must not appear in the prediction feature matrix."""
    statics = series.columns.drop(["unique_id", "ds", "y"]).tolist()
    groupby_col = statics[0]

    class FeatureRecorder(BaseEstimator):
        def fit(self, X, y=None):  # noqa: ARG002
            return self

        def predict(self, X):  # noqa: ARG002
            self.predict_feature_names_ = (
                list(X.columns) if hasattr(X, "columns") else None
            )
            return np.zeros(len(X))

    fcst = MLForecast(
        models={"rec": FeatureRecorder()},
        freq="D",
        lags=[1],
        lag_transforms={1: [RollingMean(7, groupby=[groupby_col])]},
    )
    fcst.fit(series, static_features=statics)
    fcst.predict(3)

    fitted_model = fcst.models_["rec"]
    assert hasattr(fitted_model, "predict_feature_names_")
    assert fitted_model.predict_feature_names_ is not None
    assert groupby_col not in fitted_model.predict_feature_names_


def test_drop_auxiliary_columns_cross_validation(series):
    """drop_auxiliary_columns should exclude groupby columns from the feature matrix during cross_validation."""
    statics = series.columns.drop(["unique_id", "ds", "y"]).tolist()
    groupby_col = statics[0]

    class FeatureRecorder(BaseEstimator):
        def __init__(self):
            self.fit_feature_names_ = None

        def fit(self, X, y=None):  # noqa: ARG002
            self.fit_feature_names_ = list(X.columns) if hasattr(X, "columns") else None
            return self

        def predict(self, X):  # noqa: ARG002
            return np.zeros(len(X))

    fcst = MLForecast(
        models={"rec": FeatureRecorder()},
        freq="D",
        lags=[1, 7],
        lag_transforms={1: [RollingMean(7, groupby=[groupby_col])]},
    )
    cv_result = fcst.cross_validation(series, n_windows=2, h=7, static_features=statics)
    assert cv_result is not None
    assert groupby_col not in cv_result.columns
    for i in range(2):
        fitted_model = fcst.cv_models_[i]["rec"]
        assert fitted_model.fit_feature_names_ is not None
        assert groupby_col not in fitted_model.fit_feature_names_


def test_drop_auxiliary_columns_unknown_warns(series):
    """A UserWarning should be emitted when an explicit column name doesn't exist."""
    statics = series.columns.drop(["unique_id", "ds", "y"]).tolist()

    fcst = MLForecast(
        models=LinearRegression(),
        freq="D",
        lags=[1],
        drop_auxiliary_columns=["nonexistent_column"],
    )
    with pytest.warns(UserWarning, match="drop_auxiliary_columns"):
        fcst.fit(series, static_features=statics)
