import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_sample_paths_update_lags_and_invert_differences(backend):
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 8),
            "ds": np.tile(np.arange(8), 2),
            "y": np.tile(np.arange(8, dtype=float), 2),
            "price": np.ones(16),
        }
    )
    future = pd.DataFrame({"unique_id": ["b", "b"], "ds": [8, 9], "price": [2.0, 3.0]})
    if backend == "polars":
        df, future = pl.from_pandas(df), pl.from_pandas(future)
    fcst = MLForecast(
        models={"lr": LinearRegression()},
        freq=1,
        lags=[1],
        target_transforms=[Differences([1])],
    ).fit(df, static_features=[])
    opts = dict(X_df=future, ids=["b"])

    def sample(model, X, rng):
        assert model is fcst.models_["lr"]
        return X["lag1"].to_numpy() + X["price"].to_numpy() + rng.normal(size=len(X))

    paths = fcst.predict_samples(2, 3, sample, random_state=42, **opts)
    if backend == "polars":
        assert isinstance(paths, pl.DataFrame)
        paths = paths.to_pandas()
    rng = np.random.default_rng(42)
    for i in range(3):
        first = 1 + 2 + rng.normal()
        second = first + 3 + rng.normal()
        path = paths[paths["sample"] == i]
        assert path["unique_id"].tolist() == ["b", "b"]
        assert path["ds"].tolist() == [8, 9]
        np.testing.assert_allclose(path["lr"], [7 + first, 7 + first + second])
    np.testing.assert_allclose(fcst.predict(2, **opts)["lr"], [8, 9])


def test_sample_shape_and_direct_models():
    df = pd.DataFrame({"unique_id": ["a"] * 12, "ds": range(12), "y": range(12)})
    fcst = MLForecast(models=LinearRegression(), freq=1, lags=[1]).fit(df)
    with pytest.raises(ValueError, match="one value per row"):
        fcst.predict_samples(2, 3, lambda _m, x, _rng: np.ones((len(x), 2)))
    with pytest.raises(ValueError, match="positive integer"):
        fcst.predict_samples(2, 0, lambda m, x, _rng: m.predict(x))
    fcst.fit(df, max_horizon=2)
    with pytest.raises(ValueError, match="recursive"):
        fcst.predict_samples(2, 3, lambda m, x, _rng: m.predict(x))


def test_seeded_paths_with_multiple_models():
    df = pd.DataFrame({"unique_id": ["a"] * 10, "ds": range(10), "y": range(10)})
    fcst = MLForecast(
        models={"a": GaussianProcessRegressor(), "b": BayesianRidge()}, freq=1, lags=[1]
    ).fit(df)

    def sample(model, features, rng):
        mean, std = model.predict(features, return_std=True)
        return rng.normal(mean, std)

    first = fcst.predict_samples(3, 4, sample, random_state=10)
    pd.testing.assert_frame_equal(
        first, fcst.predict_samples(3, 4, sample, random_state=10)
    )
    assert first.index.is_unique
    assert not np.allclose(first["a"], first["b"])
