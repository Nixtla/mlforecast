# tests/test_pooled_differential.py
"""Both engines, identical inputs, identical outputs.

The pooled suite proves the narwhals engine is correct in absolute terms; this
file proves the two engines agree, so any divergence names itself.
"""

import importlib
import os

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingStd,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingStd,
)

BACKENDS = ["polars", "pandas"]


def _panel(backend, n_series=40, n_times=60, n_groups=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1),
        pl.datetime(2020, 1, 1) + pl.duration(days=n_times - 1),
        interval="1d",
        eager=True,
    )
    df = pl.DataFrame(
        {
            "unique_id": np.repeat([f"id_{i}" for i in range(n_series)], n_times),
            "ds": np.tile(dates.to_numpy(), n_series),
            "y": rng.normal(10, 2, n_series * n_times),
            "store": np.repeat([i % n_groups for i in range(n_series)], n_times),
        }
    ).sort("unique_id", "ds")
    return df if backend == "polars" else df.to_pandas()


def _preprocess_with_engine(engine, df, tfms, statics):
    """Reimport the engine module with the env var set, then preprocess."""
    prev = os.environ.get("MLFORECAST_POOLED_ENGINE")
    os.environ["MLFORECAST_POOLED_ENGINE"] = engine
    try:
        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        import mlforecast.core

        importlib.reload(mlforecast.core)
        fcst = MLForecast(models=[LinearRegression()], freq="1d", lag_transforms=tfms)
        return fcst.preprocess(df, static_features=statics, dropna=False)
    finally:
        if prev is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = prev
        import mlforecast.pooled

        importlib.reload(mlforecast.pooled)
        import mlforecast.core

        importlib.reload(mlforecast.core)


def assert_engines_agree(df, tfms, statics, atol=1e-10):
    a = _preprocess_with_engine("numpy", df, tfms, statics)
    b = _preprocess_with_engine("narwhals", df, tfms, statics)
    a = a if isinstance(a, pl.DataFrame) else pl.from_pandas(a)
    b = b if isinstance(b, pl.DataFrame) else pl.from_pandas(b)
    assert sorted(a.columns) == sorted(b.columns), (
        f"column mismatch: numpy-only={set(a.columns) - set(b.columns)}, "
        f"narwhals-only={set(b.columns) - set(a.columns)}"
    )
    feat_cols = [c for c in a.columns if c not in ("unique_id", "ds", "store", "y")]
    assert feat_cols, "no feature columns produced"
    a = a.sort("unique_id", "ds")
    b = b.sort("unique_id", "ds")
    for c in feat_cols:
        av = a[c].cast(pl.Float64).to_numpy()
        bv = b[c].cast(pl.Float64).to_numpy()
        np.testing.assert_allclose(av, bv, atol=atol, equal_nan=True, err_msg=c)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rolling_mean_groupby_engines_agree(backend):
    assert_engines_agree(
        _panel(backend), {1: [RollingMean(7, groupby=["store"])]}, ["store"]
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_rolling_mean_global_engines_agree(backend):
    assert_engines_agree(
        _panel(backend), {1: [RollingMean(7, global_=True)]}, ["store"]
    )


ROLLING_EXPANDING = [
    ("rolling_std", RollingStd(14, groupby=["store"])),
    ("rolling_min", RollingMin(14, groupby=["store"])),
    ("rolling_max", RollingMax(14, groupby=["store"])),
    ("expanding_mean", ExpandingMean(groupby=["store"])),
    ("expanding_std", ExpandingStd(groupby=["store"])),
    ("expanding_min", ExpandingMin(groupby=["store"])),
    ("expanding_max", ExpandingMax(groupby=["store"])),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "label,tfm", ROLLING_EXPANDING, ids=[x[0] for x in ROLLING_EXPANDING]
)
def test_rolling_expanding_engines_agree(backend, label, tfm):  # noqa: ARG001
    assert_engines_agree(_panel(backend), {1: [tfm]}, ["store"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_target_engines_agree(backend):
    """A float32 target reaches `ga.data.dtype` as float32, and BOTH engines must
    round-trip y through it so float32 rounding matches bit-for-bit. Nothing else
    in the suite or this file exercises a non-float64 target, so without this test
    a dtype regression here is invisible."""
    df = _panel(backend, n_series=12, n_times=40)
    d = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    # large magnitudes make float32 rounding visible
    d = d.with_columns((pl.col("y") * 1000.0).cast(pl.Float32).alias("y"))
    src = d if isinstance(df, pl.DataFrame) else d.to_pandas()
    assert_engines_agree(
        src,
        {1: [RollingMean(4, groupby=["store"]), RollingMean(9, groupby=["store"])]},
        ["store"],
    )
