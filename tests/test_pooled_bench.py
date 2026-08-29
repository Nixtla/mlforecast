"""CodSpeed benchmarks for the pooled engine, both backends.

New file rather than an addition to test_pipeline.py, so no existing test file
is modified.

Cases cover the four transform families the narwhals engine speeds up
(rolling, expanding, seasonal, quantile) at both preprocess (fit) and predict
time, on both polars and pandas input. Predict is included deliberately: it
turned out to be the largest win end to end (up to ~22x at audit scale) and
was not in the original plan's headline, so CodSpeed must track it going
forward, not just fit.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMean,
    ExponentiallyWeightedMean,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingQuantile,
    RollingStd,
    SeasonalRollingMean,
)


@pytest.fixture(scope="module")
def pooled_panel():
    rng = np.random.default_rng(0)
    n_series, n_times, n_groups = 500, 365, 50
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1),
        pl.datetime(2020, 1, 1) + pl.duration(days=n_times - 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame(
        {
            "unique_id": np.repeat([f"id_{i}" for i in range(n_series)], n_times),
            "ds": np.tile(dates.to_numpy(), n_series),
            "y": rng.normal(10, 2, n_series * n_times),
            "store": np.repeat([i % n_groups for i in range(n_series)], n_times),
        }
    ).sort("unique_id", "ds")


CASES = {
    "rolling": [
        RollingMean(7, groupby=["store"]),
        RollingMean(28, groupby=["store"]),
    ],
    "expanding": [ExpandingMean(groupby=["store"])],
    "seasonal": [
        SeasonalRollingMean(season_length=7, window_size=4, groupby=["store"])
    ],
    "quantile": [RollingQuantile(p=0.5, window_size=28, groupby=["store"])],
    # A single cheap family (rolling/expanding) has an O(1) legacy update, so
    # narwhals' per-step overhead can dominate at this panel size and predict
    # looks *slower* than legacy there -- a real, worth-tracking number, not a
    # bug in the benchmark. The combined case mirrors the audit workload (8
    # transform families in one MLForecast) where the legacy engine pays its
    # per-transform Python-level overhead 8x per recursive step; that's where
    # predict's largest, and most representative, win shows up.
    "combined": [
        RollingMean(7, groupby=["store"]),
        RollingStd(7, groupby=["store"]),
        RollingMin(7, groupby=["store"]),
        RollingMax(7, groupby=["store"]),
        ExpandingMean(groupby=["store"]),
        SeasonalRollingMean(season_length=7, window_size=4, groupby=["store"]),
        RollingQuantile(p=0.5, window_size=28, groupby=["store"]),
        ExponentiallyWeightedMean(alpha=0.3, groupby=["store"]),
    ],
}


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("case", list(CASES))
def test_pooled_preprocess(benchmark, pooled_panel, backend, case):
    df = pooled_panel if backend == "polars" else pooled_panel.to_pandas()
    fcst = MLForecast(
        models=[LinearRegression()], freq="1d", lag_transforms={1: CASES[case]}
    )
    benchmark(lambda: fcst.preprocess(df, static_features=["store"], dropna=False))


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("case", list(CASES))
def test_pooled_predict(benchmark, pooled_panel, backend, case):
    df = pooled_panel if backend == "polars" else pooled_panel.to_pandas()
    fcst = MLForecast(
        models=[LinearRegression()],
        freq="1d",
        lags=[1],
        lag_transforms={1: CASES[case]},
    )
    fcst.fit(df, static_features=["store"])
    benchmark(lambda: fcst.predict(14))
