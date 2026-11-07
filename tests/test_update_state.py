"""``update`` must leave the stateful lag transforms where a full fit would.

``Expanding*``/``ExponentiallyWeightedMean`` carry a running accumulator instead
of recomputing from the stored array, so appending observations outside the
recursive predict loop has to advance it explicitly (issue #726).
"""

import copy
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
from mlforecast.target_transforms import (
    Differences,
    LocalMinMaxScaler,
    LocalRobustScaler,
    LocalStandardScaler,
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


def _fit(df, engine, lag_transforms=None, keep_last_n=None):
    if engine == "polars":
        df = pl.from_pandas(df)
    fcst = MLForecast(
        freq=ENGINE_FREQ[engine],
        models=[LinearRegression()],
        lags=[1, 2, 3],
        lag_transforms=_lag_transforms() if lag_transforms is None else lag_transforms,
    )
    return fcst.fit(df, static_features=[], keep_last_n=keep_last_n)


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


def _assert_matches_full_fit(
    full,
    hist,
    updates,
    engine,
    horizon=3,
    lag_transforms=None,
    n_stateful=N_STATEFUL,
    keep_last_n=None,
):
    expected = _fit(full, engine, lag_transforms, keep_last_n)
    actual = _fit(hist, engine, lag_transforms, keep_last_n)
    for update in updates:
        _update(actual, update, engine)

    exp_cores = _stateful_cores(expected.ts)
    act_cores = _stateful_cores(actual.ts)
    assert len(exp_cores) == n_stateful
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


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("first", [2, 4])
def test_update_of_series_not_longer_than_the_lag(engine, first):
    """A series of at most ``lag`` values hasn't started accumulating.

    Folding its appended values reads the nan pad, and nan is absorbing for
    every accumulator, so its feature would be nan from then on.
    """
    lag = 4
    full = _series({"a": 20, "b": 20, "c": 7})
    is_new = full["unique_id"].eq("c")
    hist = full[~is_new]
    new = full[is_new]
    # "c" enters at or below the lag and only crosses it on the second update
    _assert_matches_full_fit(
        full,
        hist,
        [new.iloc[:first], new.iloc[first:]],
        engine,
        lag_transforms={lag: [ExpandingMean(), ExponentiallyWeightedMean(alpha=0.3)]},
        n_stateful=2,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_user_keep_last_n_below_the_lag_matches_full_fit(engine):
    """An explicit ``keep_last_n`` can't drop the values the fold reads."""
    lag = 5
    full = _series({"a": 25, "b": 25})
    cutoff = full["ds"].max() - 3 * pd.offsets.Day()
    _assert_matches_full_fit(
        full,
        full[full["ds"] <= cutoff],
        [full[full["ds"] > cutoff]],
        engine,
        lag_transforms={lag: [ExpandingMean(), ExponentiallyWeightedMean(alpha=0.3)]},
        n_stateful=2,
        keep_last_n=2,
    )


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
    "tfm, feature, expected",
    [
        (ExpandingMin(), "expanding_min_lag1", 1.0),
        (ExpandingMean(), "expanding_mean_lag1", 91.75),
    ],
)
def test_update_does_not_skip_observations(tfm, feature, expected):
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
    assert cb.get_features()[feature].item() == pytest.approx(expected)


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


def _fit_target_update(df, engine, target_transforms):
    fcst = MLForecast(
        models=LinearRegression(),
        freq=ENGINE_FREQ[engine],
        lags=[1, 2],
        target_transforms=copy.deepcopy(target_transforms),
        num_threads=1,
    )
    return fcst.fit(
        pl.from_pandas(df) if engine == "polars" else df, static_features=[]
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("validate_new_data", [False, True])
@pytest.mark.parametrize("updated_ids", [("a",), ("b",), ("a", "b")])
@pytest.mark.parametrize(
    "target_transforms",
    [
        None,
        [LocalStandardScaler()],
        [LocalMinMaxScaler()],
        [LocalRobustScaler(scale="mad")],
        [Differences([1])],
        [Differences([1, 2])],
        [LocalStandardScaler(), Differences([1, 2])],
        [Differences([1, 2]), LocalStandardScaler()],
    ],
    ids=[
        "none",
        "standard",
        "minmax",
        "robust",
        "diff",
        "diffs",
        "scale-diff",
        "diff-scale",
    ],
)
def test_target_transform_update_existing_subset(
    engine, validate_new_data, updated_ids, target_transforms
):
    """Omitted series retain their history and inverse-transform state (#809)."""
    full = _series({"a": 14, "b": 13})
    full.loc[full["unique_id"].eq("b"), "y"] *= 100
    hist = full[full["ds"] <= START + 11 * pd.offsets.Day()]
    new = full[
        (full["ds"] > hist["ds"].max()) & full["unique_id"].isin(updated_ids)
    ].iloc[::-1]
    fcst = _fit_target_update(hist, engine, target_transforms)
    before = [fcst.ts.ga[i].copy() for i in range(2)]
    _new = pl.from_pandas(new) if engine == "polars" else new
    fcst.update(_new, validate_new_data=validate_new_data)
    predictions = fcst.predict(3)
    if engine == "polars":
        predictions = predictions.to_pandas()

    assert list(fcst.ts.uids) == ["a", "b"]
    for i, uid in enumerate(("a", "b")):
        # A single-series update is a reference that cannot misalign groups.
        single = _fit_target_update(
            hist[hist["unique_id"].eq(uid)], engine, target_transforms
        )
        if uid in updated_ids:
            update = new[new["unique_id"].eq(uid)]
            single.update(
                pl.from_pandas(update) if engine == "polars" else update,
                validate_new_data=validate_new_data,
            )
        else:
            np.testing.assert_array_equal(fcst.ts.ga[i], before[i])
        np.testing.assert_allclose(fcst.ts.ga[i], single.ts.ga[0], equal_nan=True)
        # Keep the trained model fixed when comparing multi-step predictions.
        single.models_ = fcst.models_
        expected = single.predict(3)
        if engine == "polars":
            expected = expected.to_pandas()
        pd.testing.assert_frame_equal(
            predictions[predictions["unique_id"].eq(uid)].reset_index(drop=True),
            expected.reset_index(drop=True),
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("validate_new_data", [False, True])
def test_target_transform_update_new_series_rejected(engine, validate_new_data):
    hist = _series({"a": 12, "b": 12})
    fcst = _fit_target_update(hist, engine, [Differences([1]), LocalStandardScaler()])
    expected = fcst.predict(3)
    data = fcst.ts.ga.data.copy()
    new = _series({"c": 1})
    with pytest.raises(
        ValueError, match="Can not update target_transforms with new series"
    ):
        fcst.update(
            pl.from_pandas(new) if engine == "polars" else new,
            validate_new_data=validate_new_data,
        )
    np.testing.assert_array_equal(fcst.ts.ga.data, data)
    assert list(fcst.ts.uids) == ["a", "b"]
    actual = fcst.predict(3)
    if engine == "polars":
        actual, expected = actual.to_pandas(), expected.to_pandas()
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("validate_new_data", [False, True])
def test_target_transform_partial_update_rolls_back(
    engine, validate_new_data, monkeypatch
):
    full = _series({"a": 14, "b": 12})
    hist = full[full["ds"] <= START + 11 * pd.offsets.Day()]
    new = full[full["ds"] > hist["ds"].max()]
    if engine == "polars":
        new = pl.from_pandas(new)
    tfms = [Differences([1, 2]), LocalStandardScaler()]
    fcst = _fit_target_update(hist, engine, tfms)
    expected = _fit_target_update(hist, engine, tfms)
    data = fcst.ts.ga.data.copy()
    dates = list(fcst.ts.last_dates)
    diffs = fcst.ts.target_transforms[0]
    tails = [sc.tails_.copy() for sc in diffs.scalers_]

    def fail(self, ga):  # noqa: ARG001
        raise RuntimeError("target update failed")

    with monkeypatch.context() as m:
        m.setattr(LocalStandardScaler, "update", fail)
        with pytest.raises(RuntimeError, match="target update failed"):
            fcst.update(new, validate_new_data=validate_new_data)
    assert fcst.ts.target_transforms[0] is diffs
    for scaler, tail in zip(diffs.scalers_, tails):
        np.testing.assert_array_equal(scaler.tails_, tail)
    np.testing.assert_array_equal(fcst.ts.ga.data, data)
    assert list(fcst.ts.last_dates) == dates
    fcst.update(new, validate_new_data=validate_new_data)
    expected.update(new, validate_new_data=validate_new_data)
    actual_preds, expected_preds = fcst.predict(3), expected.predict(3)
    if engine == "polars":
        actual_preds, expected_preds = (
            actual_preds.to_pandas(),
            expected_preds.to_pandas(),
        )
    pd.testing.assert_frame_equal(actual_preds, expected_preds)
