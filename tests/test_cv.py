from sklearn.base import BaseEstimator
import pytest
from mlforecast import MLForecast
from mlforecast.utils import generate_daily_series

class SeasonalNaive(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return X["lag7"]


@pytest.fixture()
def fcst():
    return MLForecast(
        models=[SeasonalNaive()],
        freq="D",
        lags=[1, 7],
    )

@pytest.fixture()
def series():
    n_series = 10
    return generate_daily_series(
        n_series=n_series,
        min_length=50,
        max_length=100,
        equal_ends=True,
    )


def test_cv(fcst: MLForecast, series):
    horizon = 14
    cv_windows = 4
    fitted = True
    _cv_results = fcst.cross_validation(
        df=series,
        n_windows=cv_windows,
        h=horizon,
        max_horizon=horizon,
        fitted=fitted,
    )
