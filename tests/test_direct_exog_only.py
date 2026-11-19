import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["recursive", "direct", "sparse"])
@pytest.mark.parametrize("n_series", [1, 3])
@pytest.mark.parametrize("n_features", [1, 2])
@pytest.mark.parametrize("static", [False, True])
def test_fit_and_predict_with_only_dynamic_features(
    engine, mode, n_series, n_features, static
):
    rng = np.random.default_rng(847)
    n = 20
    train = pd.DataFrame(
        {
            "unique_id": np.repeat(["A", "B", "C"][:n_series], n),
            "ds": np.tile(np.arange(n), n_series),
            "x": rng.normal(size=n_series * n),
        }
    )
    train["y"] = 1 + 2 * train["x"]
    future = pd.DataFrame(
        {
            "unique_id": np.repeat(["A", "B", "C"][:n_series], 3),
            "ds": np.tile([n, n + 1, n + 2], n_series),
            "x": rng.normal(size=n_series * 3),
        }
    )
    expected = 1 + 2 * future["x"].to_numpy()
    if n_features == 2:
        train["z"] = rng.normal(size=len(train))
        train["y"] += 3 * train["z"]
        future["z"] = rng.normal(size=len(future))
        expected += 3 * future["z"].to_numpy()
    if static:
        train["constant"] = 1.0
    if engine == "polars":
        train = pl.from_pandas(train)
        future = pl.from_pandas(future)
    kwargs = {"max_horizon": 3} if mode == "direct" else {}
    if mode == "sparse":
        kwargs = {"horizons": [1, 3]}
    fcst = MLForecast(models={"lr": LinearRegression()}, freq=1, lags=[])
    fcst.fit(train, static_features=["constant"] if static else [], **kwargs)
    predictions = fcst.predict(3, X_df=future)
    if mode == "sparse":
        expected = expected.reshape(n_series, 3)[:, [0, 2]].ravel()
    np.testing.assert_allclose(predictions["lr"].to_numpy(), expected, atol=1e-10)
