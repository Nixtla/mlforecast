"""``update`` must leave the stateful lag transforms where a full fit would.

``Expanding*``/``ExponentiallyWeightedMean`` carry a running accumulator instead
of recomputing from the stored array, so appending observations outside the
recursive predict loop has to advance it explicitly (issue #726).
"""

import operator

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.callbacks import SaveFeatures
from mlforecast.lag_transforms import (
    Combine,
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingQuantile,
    ExpandingStd,
    ExponentiallyWeightedMean,
    Offset,
    RollingMean,
    _BaseLagTransform,
)

FREQ = "D"
ENGINE_FREQ = {"pandas": FREQ, "polars": "1d"}
START = pd.Timestamp("2020-01-01")
# ExpandingMean/Std/Min/Max + EWM + the ExpandingMean inside Combine at lag 1,
# ExpandingMean + Offset(ExpandingMax) + EWM at lag 2
N_STATEFUL = 9


def _lag_transforms():
    return {
        1: [
            ExpandingMean(),
            ExpandingStd(),
            ExpandingMin(),
            ExpandingMax(),
            ExponentiallyWeightedMean(alpha=0.3),
            Combine(ExpandingMean(), RollingMean(window_size=3), operator.add),
            RollingMean(window_size=3),
            ExpandingQuantile(p=0.5),
        ],
        2: [
            ExpandingMean(),
            Offset(ExpandingMax(), 1),
            ExponentiallyWeightedMean(alpha=0.5),
        ],
    }


def _series(lengths, seed=0):
    rng = np.random.default_rng(seed)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range(START, periods=n, freq=FREQ),
                    "y": rng.normal(50, 10, n),
                }
            )
            for uid, n in lengths.items()
        ],
        ignore_index=True,
    )


def _fit(df, engine):
    if engine == "polars":
        df = pl.from_pandas(df)
    fcst = MLForecast(
        freq=ENGINE_FREQ[engine],
        models=[LinearRegression()],
        lags=[1, 2, 3],
        lag_transforms=_lag_transforms(),
    )
    return fcst.fit(df, static_features=[])


def _update(fcst, df, engine):
    fcst.update(pl.from_pandas(df) if engine == "polars" else df)


def _stateful_cores(ts):
    return {
        (name, i): core
        for name, tfm in ts.transforms.items()
        if isinstance(tfm, _BaseLagTransform)
        for i, core in enumerate(tfm._stateful_core_tfms())
    }


def _features(fcst, horizon):
    cb = SaveFeatures()
    fcst.predict(horizon, before_predict_callback=cb)
    feats = cb.get_features()
    return feats.to_pandas() if isinstance(feats, pl.DataFrame) else feats


def _assert_matches_full_fit(full, hist, updates, engine, horizon=3):
    expected = _fit(full, engine)
    actual = _fit(hist, engine)
    for update in updates:
        _update(actual, update, engine)

    exp_cores = _stateful_cores(expected.ts)
    act_cores = _stateful_cores(actual.ts)
    assert len(exp_cores) == N_STATEFUL
    assert exp_cores.keys() == act_cores.keys()
    for key, core in exp_cores.items():
        np.testing.assert_allclose(
            act_cores[key].stats_, core.stats_, equal_nan=True, err_msg=str(key)
        )

    # the two forecasters trained on different amounts of data, so compare the
    # features (and the predictions they drive) under a single set of models
    actual.models_ = expected.models_
    pd.testing.assert_frame_equal(
        _features(actual, horizon), _features(expected, horizon), check_dtype=False
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("k", [1, 2, 5])
def test_bulk_update_matches_full_fit(engine, k):
    full = _series({"a": 30, "b": 30})
    cutoff = full["ds"].max() - k * pd.offsets.Day()
    hist = full[full["ds"] <= cutoff]
    tail = full[full["ds"] > cutoff]
    _assert_matches_full_fit(full, hist, [tail], engine)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_sequential_updates_match_full_fit(engine):
    """One timestamp at a time, as the conformal transfer path does."""
    full = _series({"a": 30, "b": 30})
    dates = full["ds"].drop_duplicates().sort_values().to_numpy()
    hist = full[full["ds"] <= dates[-4]]
    updates = [full[full["ds"] == date] for date in dates[-3:]]
    _assert_matches_full_fit(full, hist, updates, engine)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ragged_update_matches_full_fit(engine):
    """Series can gain a different number of observations, or none at all."""
    full = _series({"a": 30, "b": 28, "c": 26})
    cutoff = START + 25 * pd.offsets.Day()
    hist = full[full["ds"] <= cutoff]
    tail = full[full["ds"] > cutoff]
    assert tail.groupby("unique_id").size().nunique() > 1
    _assert_matches_full_fit(full, hist, [tail], engine)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_update_with_new_series_matches_full_fit(engine):
    """A series added by ``update`` grows (and permutes) the accumulators."""
    full = _series({"a": 30, "b": 30, "c": 30})
    cutoff = full["ds"].max() - 2 * pd.offsets.Day()
    # "b" is new and sorts between the existing ids
    known = full["unique_id"].ne("b")
    hist = full[known & (full["ds"] <= cutoff)]
    tail = full[~known | (full["ds"] > cutoff)].sort_values(["unique_id", "ds"])
    _assert_matches_full_fit(full, hist, [tail], engine)


def test_update_with_new_series_does_not_raise():
    """The shape mismatch reported in #726."""
    dates = pd.date_range(START, periods=11, freq=FREQ)
    df = pd.concat(
        [
            pd.DataFrame(
                {"unique_id": uid, "ds": dates[:10], "y": np.arange(1.0, 11) + i}
            )
            for i, uid in enumerate(("a", "b"))
        ]
    )
    fcst = MLForecast(
        freq=FREQ,
        models=[LinearRegression()],
        lags=[1],
        lag_transforms={1: [ExpandingMean()]},
    )
    fcst.fit(df, static_features=[])
    fcst.update(
        pd.concat(
            [
                pd.DataFrame(
                    {"unique_id": ["a", "b"], "ds": dates[10], "y": [11.0, 12.0]}
                ),
                pd.DataFrame({"unique_id": "c", "ds": dates[:3], "y": [5.0, 7.0, 9.0]}),
            ]
        )
    )
    cb = SaveFeatures()
    fcst.predict(1, before_predict_callback=cb)
    np.testing.assert_allclose(
        cb.get_features()["expanding_mean_lag1"].to_numpy(),
        [
            np.arange(1.0, 12).mean(),
            np.arange(2.0, 13).mean(),
            # "c" only ever saw its own three observations
            np.mean([5.0, 7.0, 9.0]),
        ],
    )


@pytest.mark.parametrize(
    "tfm, expected",
    [(ExpandingMin(), 1.0), (ExpandingMean(), 91.75)],
)
def test_update_does_not_skip_observations(tfm, expected):
    """The accumulator used to fall behind by the number of appended values."""
    n = 12
    y = np.full(n, 100.0)
    y[n - 2] = 1.0
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range(START, periods=n, freq=FREQ),
            "y": y,
        }
    )
    fcst = MLForecast(
        freq=FREQ,
        models=[LinearRegression()],
        lags=[1],
        lag_transforms={1: [tfm]},
    )
    fcst.fit(df.iloc[:-1], static_features=[])
    fcst.update(df.iloc[-1:])
    cb = SaveFeatures()
    fcst.predict(1, before_predict_callback=cb)
    assert cb.get_features().iloc[0, -1] == pytest.approx(expected)


@pytest.mark.parametrize("lag", [1, 3])
def test_keep_last_n_covers_the_transform_lag(lag):
    """Each update reads the value ``lag`` back, so it must survive the trim."""
    n = 20
    y = np.arange(1.0, n + 1)
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range(START, periods=n, freq=FREQ),
            "y": y,
        }
    )
    fcst = MLForecast(
        freq=FREQ,
        models=[LinearRegression()],
        lags=[1],
        lag_transforms={lag: [ExpandingMean()]},
    )
    fcst.fit(df, static_features=[])
    assert fcst.ts.keep_last_n >= lag
    cb = SaveFeatures()
    fcst.predict(1, before_predict_callback=cb)
    assert cb.get_features()[f"expanding_mean_lag{lag}"].item() == pytest.approx(
        y[: n - lag + 1].mean()
    )
