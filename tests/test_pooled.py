import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlforecast.core import TimeSeries
from mlforecast.lag_transforms import RollingMean


def _make_df(engine, rows, categorical_cols=None):
    if engine == "polars":
        df = pl.DataFrame(rows)
        for col in categorical_cols or []:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
    else:
        df = pd.DataFrame(rows)
        for col in categorical_cols or []:
            df[col] = pd.Categorical(df[col])
    return df


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_new_series_new_group_update_then_predict(engine):
    """Regression: new series in a new group must get correct bucket ID
    and produce valid predictions after update()."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b", "b"],
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        "brand": ["x", "x", "x", "y", "y", "y"],
    })
    tfm = RollingMean(2, groupby=["brand"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    assert ts._pooled_groups[("brand",)] is not None
    state = ts._pooled_groups[("brand",)]
    assert len(np.unique(state.bucket_id)) == 2

    update_df = _make_df(engine, {
        "unique_id": ["a", "b", "c"],
        "ds": [4, 4, 4],
        "y": [40.0, 400.0, 1000.0],
        "brand": ["x", "y", "z"],
    })
    ts.update(update_df)

    state = ts._pooled_groups[("brand",)]
    assert state.series_bucket_id is not None
    assert len(state.series_bucket_id) == 3
    assert len(np.unique(state.series_bucket_id)) == 3
    assert len(state.bucket_df) == len(df) + len(update_df)
    n_buckets = len(np.unique(state.bucket_id))
    assert n_buckets == 3

    statics = ts.static_features_
    if engine == "pandas":
        uid_to_brand = dict(zip(statics["unique_id"], statics["brand"]))
    else:
        uid_to_brand = dict(zip(
            statics["unique_id"].to_list(), statics["brand"].to_list()
        ))
    assert uid_to_brand["a"] == "x"
    assert uid_to_brand["b"] == "y"
    assert uid_to_brand["c"] == "z"

    ts._predict_setup()
    features = ts._update_features()
    col = tfm._get_name(1)
    expected = np.array([35.0, 350.0, np.nan])
    np.testing.assert_allclose(features[col].to_numpy(), expected, equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_global_update_preserves_bucket_df(engine):
    """After update(), bucket_df should include both old and new observations."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "b", "b"],
        "ds": [1, 2, 1, 2],
        "y": [1.0, 2.0, 10.0, 20.0],
    })
    ts = TimeSeries(freq=1, lag_transforms={1: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False,
    )
    assert ts._pooled_global is not None
    orig_len = len(ts._pooled_global.bucket_df)

    update_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [3, 3],
        "y": [3.0, 30.0],
    })
    ts.update(update_df)
    new_len = len(ts._pooled_global.bucket_df)
    assert new_len == orig_len + 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_update_preserves_bucket_df(engine):
    """After update(), group bucket_df should include new observations."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "b", "b"],
        "ds": [1, 2, 1, 2],
        "y": [1.0, 2.0, 10.0, 20.0],
        "brand": ["x", "x", "x", "x"],
    })
    tfm = RollingMean(2, groupby=["brand"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    orig_len = len(ts._pooled_groups[("brand",)].bucket_df)

    update_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [3, 3],
        "y": [3.0, 30.0],
        "brand": ["x", "x"],
    })
    ts.update(update_df)
    new_len = len(ts._pooled_groups[("brand",)].bucket_df)
    assert new_len == orig_len + 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_global_sequential_updates(engine):
    """Sequential update() calls correctly increment time_index."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "b", "b"],
        "ds": [1, 2, 1, 2],
        "y": [1.0, 2.0, 10.0, 20.0],
    })
    ts = TimeSeries(freq=1, lag_transforms={1: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False,
    )
    update1 = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [3, 3],
        "y": [3.0, 30.0],
    })
    ts.update(update1)
    state = ts._pooled_global
    assert state.next_time_index_by_bucket[0] == 3

    update2 = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [4, 4],
        "y": [4.0, 40.0],
    })
    ts.update(update2)
    state = ts._pooled_global
    assert state.next_time_index_by_bucket[0] == 4
    unique_idx = np.unique(state.time_index)
    assert len(unique_idx) == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_staggered_series_start(engine):
    """Series starting at different timestamps don't inject zeros."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b"],
        "ds": [1, 2, 3, 2, 3],
        "y": [1.0, 2.0, 3.0, 20.0, 30.0],
    })
    ts = TimeSeries(freq=1, lag_transforms={1: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False,
    )
    state = ts._pooled_global
    assert len(state.y) == 5
    assert 0.0 not in state.y


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_categorical_groupby_update_with_new_group(engine):
    """Update with a new categorical group value works correctly."""
    df = _make_df(
        engine,
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [1, 2, 1, 2],
            "y": [1.0, 2.0, 10.0, 20.0],
            "brand": ["b", "b", "b", "b"],
        },
        categorical_cols=["brand"],
    )
    tfm = RollingMean(2, groupby=["brand"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    state = ts._pooled_groups[("brand",)]
    assert len(np.unique(state.bucket_id)) == 1

    update_df = _make_df(
        engine,
        {
            "unique_id": ["a", "b", "c"],
            "ds": [3, 3, 3],
            "y": [3.0, 30.0, 100.0],
            "brand": ["b", "b", "a"],
        },
        categorical_cols=["brand"],
    )
    ts.update(update_df)

    state = ts._pooled_groups[("brand",)]
    assert len(np.unique(state.bucket_id)) == 2
    assert len(state.series_bucket_id) == 3
    assert len(np.unique(state.series_bucket_id)) == 2
    assert len(state.bucket_df) == len(df) + len(update_df)

    ts._predict_setup()
    features = ts._update_features()
    col = tfm._get_name(1)
    expected = np.array([13.75, 13.75, np.nan])
    np.testing.assert_allclose(features[col].to_numpy(), expected, equal_nan=True)


def test_compute_pooled_features_raises_for_unsupported():
    """Transforms returning None from _compute_bucket_feature raise NotImplementedError."""
    from mlforecast.pooled import PooledState, compute_pooled_features
    from mlforecast.lag_transforms import _BaseLagTransform
    from mlforecast.grouped_array import GroupedArray

    class DummyTransform(_BaseLagTransform):
        pass

    ga = GroupedArray(np.array([1.0, 2.0]), np.array([0, 2], dtype=np.int32))
    state = PooledState(
        ga=ga,
        bucket_df=pd.DataFrame({"uid": ["a", "a"], "ds": [1, 2], "_bucket_pos": [0, 1]}),
        groups=None,
        group_cols=None,
        series_bucket_id=np.array([0]),
        bucket_id=np.zeros(2, dtype=np.int64),
        time=np.array([1, 2]),
        time_index=np.array([0, 1], dtype=np.int64),
        y=np.array([1.0, 2.0]),
        next_time_index_by_bucket={0: 2},
        join_cols=["uid", "ds"],
    )
    with pytest.raises(NotImplementedError, match="does not support pooled"):
        compute_pooled_features(state, {"dummy": DummyTransform()})
