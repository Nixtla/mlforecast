import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlforecast.core import TimeSeries
from mlforecast.lag_transforms import (
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingStd,
    ExponentiallyWeightedMean,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingStd,
)

_LAGS = [1, 3]


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
@pytest.mark.parametrize("lag", _LAGS)
def test_new_series_new_group_update_then_predict(engine, lag):
    """Regression: new series in a new group must get correct bucket ID
    and produce valid predictions after update()."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b", "b"],
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        "brand": ["x", "x", "x", "y", "y", "y"],
    })
    tfm = RollingMean(2, groupby=["brand"])
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    assert ts._pooled_states[("groupby", ("brand",), ())] is not None
    state = ts._pooled_states[("groupby", ("brand",), ())]
    assert len(np.unique(state.bucket_id)) == 2

    update_df = _make_df(engine, {
        "unique_id": ["a", "b", "c"],
        "ds": [4, 4, 4],
        "y": [40.0, 400.0, 1000.0],
        "brand": ["x", "y", "z"],
    })
    ts.update(update_df)

    state = ts._pooled_states[("groupby", ("brand",), ())]
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
    col = tfm._get_name(lag)
    # brand x: a=[10,20,30,40], brand y: b=[100,200,300,400]
    # At ds=5, rolling(2) over brand x looks at window [5-lag-1, 5-lag]
    #   lag=1: [3,4] → mean(30,40)=35; lag=3: [1,2] → mean(10,20)=15
    # brand z: c has only 1 obs → NaN
    expected_x = {1: 35.0, 3: 15.0}
    expected_y = {1: 350.0, 3: 150.0}
    expected = np.array([expected_x[lag], expected_y[lag], np.nan])
    np.testing.assert_allclose(features[col].to_numpy(), expected, equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_global_update_preserves_bucket_df(engine, lag):
    """After update(), bucket_df should include both old and new observations."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "b", "b"],
        "ds": [1, 2, 1, 2],
        "y": [1.0, 2.0, 10.0, 20.0],
    })
    ts = TimeSeries(freq=1, lag_transforms={lag: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False,
    )
    assert ("global", (), ()) in ts._pooled_states
    orig_len = len(ts._pooled_states[("global", (), ())].bucket_df)

    update_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [3, 3],
        "y": [3.0, 30.0],
    })
    ts.update(update_df)
    new_len = len(ts._pooled_states[("global", (), ())].bucket_df)
    assert new_len == orig_len + 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_group_update_preserves_bucket_df(engine, lag):
    """After update(), group bucket_df should include new observations."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "b", "b"],
        "ds": [1, 2, 1, 2],
        "y": [1.0, 2.0, 10.0, 20.0],
        "brand": ["x", "x", "x", "x"],
    })
    tfm = RollingMean(2, groupby=["brand"])
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    orig_len = len(ts._pooled_states[("groupby", ("brand",), ())].bucket_df)

    update_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [3, 3],
        "y": [3.0, 30.0],
        "brand": ["x", "x"],
    })
    ts.update(update_df)
    new_len = len(ts._pooled_states[("groupby", ("brand",), ())].bucket_df)
    assert new_len == orig_len + 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_global_sequential_updates(engine, lag):
    """Sequential update() calls correctly increment time_index."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "b", "b"],
        "ds": [1, 2, 1, 2],
        "y": [1.0, 2.0, 10.0, 20.0],
    })
    ts = TimeSeries(freq=1, lag_transforms={lag: [RollingMean(2, global_=True)]})
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
    state = ts._pooled_states[("global", (), ())]
    assert state.next_time_index_by_bucket[0] == 3

    update2 = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [4, 4],
        "y": [4.0, 40.0],
    })
    ts.update(update2)
    state = ts._pooled_states[("global", (), ())]
    assert state.next_time_index_by_bucket[0] == 4
    unique_idx = np.unique(state.time_index)
    assert len(unique_idx) == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_staggered_series_start(engine, lag):
    """Series starting at different timestamps don't inject zeros."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b"],
        "ds": [1, 2, 3, 2, 3],
        "y": [1.0, 2.0, 3.0, 20.0, 30.0],
    })
    ts = TimeSeries(freq=1, lag_transforms={lag: [RollingMean(2, global_=True)]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False,
    )
    state = ts._pooled_states[("global", (), ())]
    assert len(state.y) == 5
    assert 0.0 not in state.y


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("lag", _LAGS)
def test_categorical_groupby_update_with_new_group(engine, lag):
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
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    state = ts._pooled_states[("groupby", ("brand",), ())]
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

    state = ts._pooled_states[("groupby", ("brand",), ())]
    assert len(np.unique(state.bucket_id)) == 2
    assert len(state.series_bucket_id) == 3
    assert len(np.unique(state.series_bucket_id)) == 2
    assert len(state.bucket_df) == len(df) + len(update_df)

    ts._predict_setup()
    features = ts._update_features()
    col = tfm._get_name(lag)
    # brand b: [1,2,10,20,3,30] at ds 1,2,1,2,3,3
    # At ds=4, rolling(2) window is [4-lag-1, 4-lag]
    #   lag=1: [2,3] → mean(2,20,3,30)=13.75; lag=3: [0,1] → mean(1,10)=5.5
    # brand a: c has only 1 obs → NaN
    expected_val = {1: 13.75, 3: 5.5}
    expected = np.array([expected_val[lag], expected_val[lag], np.nan])
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


# === partition_by tests ===


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_local_fit_transform(engine):
    """partition_by with local mode (no global_/groupby) creates correct buckets."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "ds": [1, 2, 3, 4, 1, 2, 3, 4],
        "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        "promo": [0, 0, 1, 1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    # partition_by creates a partition state
    part_key = ("local", (), ("promo",))
    assert part_key in ts._pooled_states
    state = ts._pooled_states[part_key]
    assert state.mode == "local"
    assert state.partition_cols == ["promo"]
    # The feature column should exist
    col = tfm._get_name(1)
    assert "partby_promo" in col
    assert col in result.columns if engine == "polars" else col in result.columns


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_global_fit_transform(engine):
    """partition_by with global_ creates global+partition buckets."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "ds": [1, 2, 3, 4, 1, 2, 3, 4],
        "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        "promo": [0, 0, 1, 1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    part_key = ("nonlocal", (), ("promo",))
    assert part_key in ts._pooled_states
    state = ts._pooled_states[part_key]
    assert state.mode == "nonlocal"
    assert state.partition_cols == ["promo"]
    col = tfm._get_name(1)
    assert "global_" in col
    assert "partby_promo" in col


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_groupby_fit_transform(engine):
    """partition_by with groupby creates group+partition buckets."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "ds": [1, 2, 3, 4, 1, 2, 3, 4],
        "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        "brand": ["x", "x", "x", "x", "y", "y", "y", "y"],
        "promo": [0, 0, 1, 1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, groupby=["brand"], partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    part_key = ("nonlocal", ("brand",), ("promo",))
    assert part_key in ts._pooled_states
    state = ts._pooled_states[part_key]
    assert state.mode == "nonlocal"
    # key_cols should include both brand and promo
    assert "brand" in state.key_cols
    assert "promo" in state.key_cols
    col = tfm._get_name(1)
    assert "groupby_brand" in col
    assert "partby_promo" in col


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_local_predict(engine):
    """partition_by local: predict produces features without errors."""
    from mlforecast.forecast import MLForecast
    from sklearn.linear_model import LinearRegression

    df = _make_df(engine, {
        "unique_id": ["a"]*12 + ["b"]*12,
        "ds": list(range(1, 13)) + list(range(1, 13)),
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
              10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
        "promo": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1,
                  0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    })
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=[],
    )
    future_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [13, 13],
        "promo": [1, 0],
    })
    preds = fcst.predict(h=1, X_df=future_df)
    assert len(preds) == 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_not_in_local_tfms(engine):
    """Transforms with partition_by should not appear in local transforms."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b", "b"],
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        "promo": [0, 0, 1, 0, 1, 0],
    })
    from mlforecast.lag_transforms import Lag
    tfm_local = Lag(1)
    tfm_part = RollingMean(2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm_local, tfm_part]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    local_tfms = ts._get_local_tfms(ts.transforms)
    for t in local_tfms.values():
        assert not getattr(t, "partition_by", None)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_update(engine):
    """update() with partition_by states works correctly."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b", "b"],
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        "promo": [0, 0, 1, 0, 1, 0],
    })
    tfm = RollingMean(2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    part_key = ("local", (), ("promo",))
    state = ts._pooled_states[part_key]
    orig_len = len(state.bucket_df)

    update_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [4, 4],
        "y": [4.0, 40.0],
        "promo": [1, 0],
    })
    ts.update(update_df)
    state = ts._pooled_states[part_key]
    assert len(state.bucket_df) == orig_len + 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_local_numeric_values(engine):
    """Verify rolling mean per (id, promo) bucket matches hand-computed values."""
    df = _make_df(engine, {
        "unique_id": ["a"] * 6 + ["b"] * 6,
        "ds": list(range(1, 7)) * 2,
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
              100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        "promo": [0, 0, 0, 1, 0, 1,
                  1, 1, 0, 0, 1, 0],
    })
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    col = tfm._get_name(1)
    vals = result[col].to_numpy()
    expected = np.array([
        np.nan, 10.0, 15.0, np.nan, 30.0, 40.0,   # series a
        np.nan, 100.0, np.nan, 300.0, np.nan, 400.0,  # series b
    ])
    np.testing.assert_allclose(vals, expected, equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_global_numeric_values(engine):
    """Verify rolling mean per (promo) bucket with global parent calendar."""
    df = _make_df(engine, {
        "unique_id": ["a"] * 3 + ["b"] * 3,
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        "promo": [0, 1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    col = tfm._get_name(1)
    vals = result[col].to_numpy()
    expected = np.array([np.nan, 100.0, 105.0, np.nan, 10.0, 60.0])
    np.testing.assert_allclose(vals, expected, equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_ordinals_have_parent_gaps(engine):
    """Verify ordinals are [0,2,4] not [0,1,2] when partition has gaps."""
    df = _make_df(engine, {
        "unique_id": ["a"] * 5,
        "ds": [1, 2, 3, 4, 5],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "promo": [0, 1, 0, 1, 0],
    })
    tfm = RollingMean(2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    state = ts._pooled_states[("local", (), ("promo",))]
    # Bucket (a,0): ds=[1,3,5] → parent ordinals [0,2,4] (NOT [0,1,2])
    # Bucket (a,1): ds=[2,4] → parent ordinals [1,3] (NOT [0,1])
    expected_ordinals = np.array([0, 2, 4, 1, 3])
    np.testing.assert_array_equal(state.time_index, expected_ordinals)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_range_semantics_with_gaps(engine):
    """RANGE (not ROWS) windowing on a partition bucket with timestamp gaps.

    Single series, parent calendar [1,2,3,4,5]. The promo=1 bucket is observed
    only at ts [1,3,5] -> parent ordinals [0,2,4]. With RollingMean(window_size=2)
    at lag 1, the window at ts=5 (ordinal 4) spans parent ordinals [2,3]; only
    ordinal 2 (ts=3) is observed, so the mean is y[ts=3]=30. ROWS semantics would
    instead average the two preceding *observations* (ts=1 and ts=3) -> 20, which
    is what this guards against.
    """
    df = _make_df(engine, {
        "unique_id": ["a"] * 5,
        "ds": [1, 2, 3, 4, 5],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "promo": [1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    out = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    col = tfm._get_name(1)
    if engine == "polars":
        out = out.to_pandas()
    vals = out[out["promo"] == 1].sort_values("ds")[col].to_numpy()
    # ts=1 -> empty window (NaN); ts=3 -> only ts=1 (10); ts=5 -> only ts=3 (RANGE=30, not ROWS=20)
    np.testing.assert_array_equal(np.isnan(vals), [True, False, False])
    np.testing.assert_allclose(vals[1:], [10.0, 30.0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_expanding_with_parent_gaps(engine):
    """ExpandingMean over a gapped partition bucket accumulates by parent ordinal.

    Same gap setup as the RANGE test. With lag 1 the expanding mean at each
    observation averages all bucket observations at parent ordinals strictly
    below the current one: ts=3 -> {ts=1}=10; ts=5 -> {ts=1,ts=3}=20. The lag
    offset is applied in parent-ordinal space, so the gap at ordinal 3 (ts=4,
    unobserved) does not change which observations fall inside the window.
    """
    df = _make_df(engine, {
        "unique_id": ["a"] * 5,
        "ds": [1, 2, 3, 4, 5],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "promo": [1, 0, 1, 0, 1],
    })
    tfm = ExpandingMean(partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    out = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    col = tfm._get_name(1)
    if engine == "polars":
        out = out.to_pandas()
    vals = out[out["promo"] == 1].sort_values("ds")[col].to_numpy()
    np.testing.assert_array_equal(np.isnan(vals), [True, False, False])
    np.testing.assert_allclose(vals[1:], [10.0, 20.0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_dynamic_keys_multistep(engine):
    """Multi-step prediction with changing promo values in X_df."""
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(engine, {
        "unique_id": ["a"] * 10 + ["b"] * 10,
        "ds": list(range(1, 11)) * 2,
        "y": [float(i) for i in range(1, 11)]
           + [float(i * 10) for i in range(1, 11)],
        "promo": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    })
    tfm = RollingMean(3, min_samples=1, partition_by=["promo"])
    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=[],
    )
    future_df = _make_df(engine, {
        "unique_id": ["a", "a", "b", "b"],
        "ds": [11, 12, 11, 12],
        "promo": [1, 0, 0, 1],
    })
    preds = fcst.predict(h=2, X_df=future_df)
    assert len(preds) == 4
    if engine == "pandas":
        pred_vals = preds.iloc[:, -1].values
    else:
        pred_vals = preds[:, -1].to_numpy()
    assert not np.any(np.isnan(pred_vals))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_predict_ids_with_nonlocal_partition_raises(engine):
    """predict(ids=...) must be blocked for nonlocal partition transforms."""
    from mlforecast.forecast import MLForecast
    from sklearn.linear_model import LinearRegression

    df = _make_df(engine, {
        "unique_id": ["a"] * 4 + ["b"] * 4,
        "ds": [1, 2, 3, 4, 1, 2, 3, 4],
        "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        "promo": [0, 0, 1, 1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=[],
    )
    future_df = _make_df(engine, {
        "unique_id": ["a"],
        "ds": [5],
        "promo": [0],
    })
    with pytest.raises(ValueError, match="Cannot use `ids`"):
        fcst.predict(h=1, X_df=future_df, ids=["a"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_default_static_features_with_partition_cols(engine):
    """static_features=None should auto-exclude partition_by columns."""
    df = _make_df(engine, {
        "unique_id": ["a"] * 4 + ["b"] * 4,
        "ds": [1, 2, 3, 4, 1, 2, 3, 4],
        "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        "promo": [0, 0, 1, 1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False,
    )
    assert "promo" not in ts.static_features_.columns
    assert tfm._get_name(1) in result.columns


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_by_backup_restore(engine):
    """_backup() correctly restores partition_by state."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b", "b"],
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        "promo": [0, 0, 1, 0, 1, 0],
    })
    tfm = RollingMean(2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    part_key = ("local", (), ("promo",))
    orig_y_len = len(ts._pooled_states[part_key].y)

    with ts._backup():
        # Simulate some mutation
        ts._predict_setup()
        ts._update_y(np.array([99.0, 99.0]))

    # After backup restore, state should be back to original
    assert len(ts._pooled_states[part_key].y) == orig_y_len


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_partition_prediction_advances_sibling_calendar(engine):
    """Multi-step predict: sibling buckets under same parent advance together.

    Uses RollingMean(2, min_samples=1) partition transform plus a regular
    lag so that fit doesn't drop the series. Captures features at each step
    via before_predict_callback and asserts exact partition feature values.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Series "a": promo alternates 0,1,0,1,...
    # Bucket (a,0): ds=[1,3,5,7,9] y=[10,30,50,70,90] parent ordinals [0,2,4,6,8]
    # Bucket (a,1): ds=[2,4,6,8,10] y=[20,40,60,80,100] parent ordinals [1,3,5,7,9]
    df = _make_df(engine, {
        "unique_id": ["a"] * 10,
        "ds": list(range(1, 11)),
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "promo": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        if hasattr(x, "to_numpy"):
            captured.append(x[col].to_numpy().copy())
        else:
            captured.append(x.copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lags=[1],
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=[],
    )
    # Predict 2 steps: promo=[1, 0]
    # h=0 (ds=11, promo=1): RollingMean(2,min_samples=1) of bucket(a,1)
    #   at parent ordinal 10. lag=1, window=2 → ordinals [8,9].
    #   bucket(a,1) has y=80 at ord 7, y=100 at ord 9. Ord 8 has no obs.
    #   Only ord 9 (y=100) is in [8,9]. → mean([100]) = 100.0
    # h=1 (ds=12, promo=0): bucket(a,0) at parent ordinal 11.
    #   lag=1, window=2 → ordinals [9,10]. bucket(a,0) has last obs at ord 8.
    #   Neither ord 9 nor 10 has an obs for (a,0). → NaN
    #   Key check: parent calendar advanced to length 11 for BOTH buckets.
    future_df = _make_df(engine, {
        "unique_id": ["a", "a"],
        "ds": [11, 12],
        "promo": [1, 0],
    })
    preds = fcst.predict(h=2, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 2
    assert len(captured) == 2
    np.testing.assert_allclose(captured[0][0], 100.0)
    assert np.isnan(captured[1][0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_partition_update_advances_sibling_calendar(engine):
    """update() advances parent calendar so sibling bucket sees new timestamp.

    After updating with promo=1 at ds=6, verify:
    - ALL sibling buckets have next_time_index = 6 (not just the updated one)
    - Feature values are correct when querying after update
    """
    df = _make_df(engine, {
        "unique_id": ["a"] * 5,
        "ds": [1, 2, 3, 4, 5],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "promo": [0, 1, 0, 1, 0],
    })
    # RollingMean(1, min_samples=1): returns lag-1 value within the bucket
    tfm = RollingMean(1, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    part_key = ("local", (), ("promo",))
    state = ts._pooled_states[part_key]
    for bid in state.next_time_index_by_bucket:
        assert state.next_time_index_by_bucket[bid] == 5

    update_df = _make_df(engine, {
        "unique_id": ["a"],
        "ds": [6],
        "y": [60.0],
        "promo": [1],
    })
    ts.update(update_df)
    state = ts._pooled_states[part_key]
    # After update at ds=6 with promo=1, parent calendar = [1,2,3,4,5,6]
    # ALL sibling buckets should now have next_time_index = 6
    for bid in state.next_time_index_by_bucket:
        assert state.next_time_index_by_bucket[bid] == 6

    # Verify feature computation doesn't crash and uses correct ordinals.
    # After update at ds=6 with promo=1:
    #   Bucket (a,0): observations at parent ordinals [0,2,4] (ds=[1,3,5])
    #   Bucket (a,1): observations at parent ordinals [1,3,5] (ds=[2,4,6])
    # Both buckets now at next_time_index=6. A prediction at ds=7 (ordinal 6)
    # with RollingMean(1) lag=1 for bucket(a,1) looks at ordinal 5 → y=60.
    # For bucket(a,0) it looks at ordinal 5 → no observation → NaN.
    ts._predict_setup()
    features = ts._update_features()
    col = tfm._get_name(1)
    feat_val = features[col].to_numpy()[0]
    # Series "a" was last seen with promo=0 (from static features at fit),
    # so it's assigned to bucket(a,0). RollingMean(1) at ordinal 6 lag-1
    # looks at ordinal 5. Bucket(a,0) has no obs at ordinal 5 → NaN.
    assert np.isnan(feat_val)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_new_partition_bucket_uses_existing_parent_calendar(engine):
    """New partition bucket created during prediction inherits parent ordinal."""
    df = _make_df(engine, {
        "unique_id": ["a"] * 4,
        "ds": [1, 2, 3, 4],
        "y": [10.0, 20.0, 30.0, 40.0],
        "promo": [0, 0, 0, 0],
    })
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    part_key = ("local", (), ("promo",))
    state = ts._pooled_states[part_key]
    # Only promo=0 bucket exists, parent calendar = [1,2,3,4], length 4

    # Trigger prediction setup so we can call update_series_bucket_id
    ts._predict_setup()
    if isinstance(df, pd.DataFrame):
        context_df = pd.DataFrame({
            "unique_id": ["a"],
            "promo": [1],
        })
    else:
        context_df = pl.DataFrame({
            "unique_id": ["a"],
            "promo": [1],
        })
    state.update_series_bucket_id(context_df, "unique_id")
    # New bucket for promo=1 should inherit parent calendar length = 4
    new_bid = int(state.series_bucket_id[0])
    assert state.next_time_index_by_bucket[new_bid] == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_global_partition_update_advances_sibling_calendar(engine):
    """Global+partition: update advances all sibling bucket ordinals."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b", "b"],
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        "promo": [0, 1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    part_key = ("nonlocal", (), ("promo",))
    state = ts._pooled_states[part_key]
    # Global parent calendar = [1,2,3], length 3
    for bid in state.next_time_index_by_bucket:
        assert state.next_time_index_by_bucket[bid] == 3

    update_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [4, 4],
        "y": [40.0, 400.0],
        "promo": [1, 1],
    })
    ts.update(update_df)
    state = ts._pooled_states[part_key]
    # Parent calendar now [1,2,3,4], ALL buckets should be at 4
    for bid in state.next_time_index_by_bucket:
        assert state.next_time_index_by_bucket[bid] == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_groupby_partition_update_advances_sibling_calendar(engine):
    """Groupby+partition: update advances sibling buckets within each group."""
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b", "b"],
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        "brand": ["x", "x", "x", "y", "y", "y"],
        "promo": [0, 1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, min_samples=1, groupby=["brand"], partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand"],
    )
    part_key = ("nonlocal", ("brand",), ("promo",))
    state = ts._pooled_states[part_key]
    # Each brand group has parent calendar [1,2,3], length 3
    for bid in state.next_time_index_by_bucket:
        assert state.next_time_index_by_bucket[bid] == 3

    update_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [4, 4],
        "y": [40.0, 400.0],
        "brand": ["x", "y"],
        "promo": [1, 0],
    })
    ts.update(update_df)
    state = ts._pooled_states[part_key]
    for bid in state.next_time_index_by_bucket:
        assert state.next_time_index_by_bucket[bid] == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_assignment_missing_key_error(engine):
    """Missing partition key in X_df and static_features raises ValueError."""
    from mlforecast.forecast import MLForecast
    from sklearn.linear_model import LinearRegression

    df = _make_df(engine, {
        "unique_id": ["a"] * 4,
        "ds": [1, 2, 3, 4],
        "y": [1.0, 2.0, 3.0, 4.0],
        "promo": [0, 1, 0, 1],
    })
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=[],
    )
    # X_df missing "promo" column but has another exogenous feature
    future_df = _make_df(engine, {
        "unique_id": ["a"],
        "ds": [5],
        "other_feature": [1.0],
    })
    with pytest.raises(ValueError, match="X_df is missing future values"):
        fcst.predict(h=1, X_df=future_df)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_predict_requires_x_df(engine):
    """predict(h) without X_df must error when partition_by is configured."""
    from mlforecast.forecast import MLForecast
    from sklearn.linear_model import LinearRegression

    df = _make_df(engine, {
        "unique_id": ["a"] * 4,
        "ds": [1, 2, 3, 4],
        "y": [1.0, 2.0, 3.0, 4.0],
        "promo": [0, 1, 0, 1],
    })
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=[],
    )
    with pytest.raises(ValueError, match="X_df is required for prediction"):
        fcst.predict(h=1)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_unseen_partition_predict(engine):
    """Fit with promo=0 only, predict h=1 with unseen promo=1."""
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(engine, {
        "unique_id": ["a"] * 4,
        "ds": [1, 2, 3, 4],
        "y": [10.0, 20.0, 30.0, 40.0],
        "promo": [0, 0, 0, 0],
    })
    tfm = RollingMean(1, min_samples=1, partition_by=["promo"])
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        if hasattr(x, "to_numpy"):
            captured.append(x[col].to_numpy().copy())
        else:
            captured.append(x.copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=[],
    )
    future_df = _make_df(engine, {
        "unique_id": ["a"],
        "ds": [5],
        "promo": [1],
    })
    preds = fcst.predict(h=1, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 1
    # Unseen bucket promo=1 has no historical data → feature is NaN
    assert np.isnan(captured[0][0])
    # Prediction should still be finite (model handles NaN)
    if engine == "pandas":
        assert not np.isnan(preds.iloc[0, -1])
    else:
        assert not np.isnan(preds[:, -1].to_numpy()[0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_global_partition_unseen_bucket_predict(engine):
    """Regression test for Bug 1: global+partition with unseen bucket at predict.

    Previously raised IndexError because append_predictions passed
    new_groups=zeros for buckets that didn't exist in the GA yet.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "ds": [1, 2, 3, 4, 1, 2, 3, 4],
        "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        "promo": [0, 0, 0, 0, 0, 0, 0, 0],
    })
    tfm = RollingMean(1, min_samples=1, global_=True, partition_by=["promo"])
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        if hasattr(x, "to_numpy"):
            captured.append(x[col].to_numpy().copy())
        else:
            captured.append(x.copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=[],
    )
    future_df = _make_df(engine, {
        "unique_id": ["a", "b"],
        "ds": [5, 5],
        "promo": [1, 1],
    })
    preds = fcst.predict(h=1, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 2
    # Unseen bucket promo=1 has no historical data → feature is NaN
    assert np.all(np.isnan(captured[0]))
    # Predictions should be finite
    if engine == "pandas":
        pred_vals = preds.iloc[:, -1].values
    else:
        pred_vals = preds[:, -1].to_numpy()
    assert not np.any(np.isnan(pred_vals))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_global_partition_new_bucket_inherits_parent_calendar(engine):
    """Regression test for Bug 2: global+partition new bucket gets ordinal 0.

    When parent_scope_cols is None, _resolve_parent_for_bucket must still
    find the global parent (scope_key=()) and inherit its calendar length.
    """
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b", "b"],
        "ds": [1, 2, 3, 1, 2, 3],
        "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        "promo": [0, 0, 0, 0, 0, 0],
    })
    tfm = RollingMean(1, min_samples=1, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    part_key = ("nonlocal", (), ("promo",))
    state = ts._pooled_states[part_key]
    # Global parent calendar = [1,2,3], length 3
    assert state.next_time_index_by_bucket[0] == 3

    # Create unseen promo=1 bucket via update_series_bucket_id
    if isinstance(df, pd.DataFrame):
        context_df = pd.DataFrame({
            "unique_id": ["a", "b"],
            "promo": [1, 1],
        })
    else:
        context_df = pl.DataFrame({
            "unique_id": ["a", "b"],
            "promo": [1, 1],
        })
    state.update_series_bucket_id(context_df, "unique_id")
    new_bid = int(state.series_bucket_id[0])
    # New bucket must inherit global parent calendar length = 3, not 0
    assert state.next_time_index_by_bucket[new_bid] == 3


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_datetime_update_new_bucket(engine):
    """Regression test for Bug 3: datetime dtype mismatch on new parent grid.

    _resolve_parent_for_bucket used to create np.array([], dtype=float)
    which raises DTypePromotionError with datetime timestamps.
    """
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    df = _make_df(engine, {
        "unique_id": ["a"] * 4,
        "ds": dates,
        "y": [10.0, 20.0, 30.0, 40.0],
        "promo": [0, 0, 0, 0],
    })
    tfm = RollingMean(1, min_samples=1, partition_by=["promo"])
    freq = "1d" if engine == "polars" else "D"
    ts = TimeSeries(freq=freq, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    # Update with a new partition value — this triggers _resolve_parent_for_bucket
    # which creates a new parent grid. With the dtype fix, the grid dtype
    # matches the existing datetime64 dtype.
    update_df = _make_df(engine, {
        "unique_id": ["a"],
        "ds": pd.to_datetime(["2020-01-05"]),
        "y": [50.0],
        "promo": [1],
    })
    ts.update(update_df)
    part_key = ("local", (), ("promo",))
    state = ts._pooled_states[part_key]
    # Parent calendar = [2020-01-01..05], length 5
    for bid in state.next_time_index_by_bucket:
        assert state.next_time_index_by_bucket[bid] == 5


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_backup_restore_with_dynamic_buckets(engine):
    """Verify _backup() isolates dynamic bucket mutations from fitted state.

    Prediction can create temporary buckets, extend parent maps, and grow
    parent grids. After predict returns, these mutations must not leak.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(engine, {
        "unique_id": ["a"] * 4,
        "ds": [1, 2, 3, 4],
        "y": [10.0, 20.0, 30.0, 40.0],
        "promo": [0, 0, 0, 0],
    })
    tfm = RollingMean(1, min_samples=1, partition_by=["promo"])
    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=[],
    )
    part_key = ("local", (), ("promo",))
    state_before = fcst.ts._pooled_states[part_key]
    n_groups_before = len(state_before.groups) if state_before.groups is not None else 0
    n_parents_before = len(state_before._parent_time_grids or {})
    n_bucket_map_before = len(state_before._bucket_to_parent_id or {})
    parent_grid_lens_before = {
        pid: len(grid)
        for pid, grid in (state_before._parent_time_grids or {}).items()
    }

    # Predict with unseen promo=1 — creates dynamic bucket inside _backup()
    future_df = _make_df(engine, {
        "unique_id": ["a"],
        "ds": [5],
        "promo": [1],
    })
    fcst.predict(h=1, X_df=future_df)

    # After predict, fitted state should be restored
    state_after = fcst.ts._pooled_states[part_key]
    n_groups_after = len(state_after.groups) if state_after.groups is not None else 0
    assert n_groups_after == n_groups_before
    assert len(state_after._parent_time_grids or {}) == n_parents_before
    assert len(state_after._bucket_to_parent_id or {}) == n_bucket_map_before
    for pid, length in parent_grid_lens_before.items():
        assert len(state_after._parent_time_grids[pid]) == length


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_predict_multi_horizon_multiple_unseen(engine):
    """h=4 prediction where X_df introduces several never-seen partition values.

    Each new promo value spawns a fresh bucket on the fly; the shared global
    parent calendar keeps every bucket aligned, so the brand-new buckets start
    empty (NaN feature) yet predictions stay finite across all horizons.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(engine, {
        "unique_id": ["a"] * 5 + ["b"] * 5,
        "ds": list(range(1, 6)) * 2,
        "y": [float(i) for i in range(1, 6)] + [float(i * 10) for i in range(1, 6)],
        "promo": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    col = tfm._get_name(1)
    captured = []

    def save_features(x):
        captured.append(x[col].to_numpy().copy())
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y", static_features=[])
    # promo values 2 and 3 are never seen during fit; alternate across 4 horizons
    future_df = _make_df(engine, {
        "unique_id": ["a"] * 4 + ["b"] * 4,
        "ds": [6, 7, 8, 9] * 2,
        "promo": [2, 3, 2, 3, 2, 3, 2, 3],
    })
    preds = fcst.predict(h=4, X_df=future_df, before_predict_callback=save_features)
    assert len(preds) == 8
    # step 0 hits the brand-new promo=2 bucket with no history -> NaN feature
    assert np.all(np.isnan(captured[0]))
    if engine == "pandas":
        pred_vals = preds.iloc[:, -1].to_numpy()
    else:
        pred_vals = preds[:, -1].to_numpy()
    assert not np.any(np.isnan(pred_vals))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_partition_recursive_h3_consistency(engine):
    """Local partition recursive predict slides the rolling window correctly.

    Single series, constant promo=1, RollingMean(window_size=2, lag=1). Each
    horizon's feature must equal the mean of the two most recent values (real,
    then predicted), so the window advances over the appended predictions
    rather than resetting or jumping.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_df(engine, {
        "unique_id": ["a"] * 5,
        "ds": [1, 2, 3, 4, 5],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "promo": [1, 1, 1, 1, 1],
    })
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    col = tfm._get_name(1)
    feats = []

    def save_features(x):
        feats.append(float(x[col].to_numpy()[0]))
        return x

    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=10)],
        freq=1,
        lag_transforms={1: [tfm]},
    )
    fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y", static_features=[])
    future_df = _make_df(engine, {
        "unique_id": ["a"] * 3,
        "ds": [6, 7, 8],
        "promo": [1, 1, 1],
    })
    preds = fcst.predict(h=3, X_df=future_df, before_predict_callback=save_features)
    if engine == "pandas":
        p = preds.iloc[:, -1].to_numpy()
    else:
        p = preds[:, -1].to_numpy()
    # h1 uses real y[ds4],y[ds5]; h2 uses y[ds5],pred[ds6]; h3 uses pred[ds6],pred[ds7]
    np.testing.assert_allclose(feats[0], (40.0 + 50.0) / 2)
    np.testing.assert_allclose(feats[1], (50.0 + p[0]) / 2)
    np.testing.assert_allclose(feats[2], (p[0] + p[1]) / 2)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_update_batch_multiple_ids_new_buckets(engine):
    """A single update() carrying several series, each with an unseen partition
    value, registers one new bucket per value and advances every parent calendar.
    """
    df = _make_df(engine, {
        "unique_id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        "ds": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        "promo": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    tfm = RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    key = ("nonlocal", (), ("promo",))
    state = ts._pooled_states[key]
    assert len(state.groups) == 1  # only promo=0 seen at fit

    # one batch at ds=4: three series, three never-seen promo values
    update_df = _make_df(engine, {
        "unique_id": ["a", "b", "c"],
        "ds": [4, 4, 4],
        "y": [4.0, 40.0, 400.0],
        "promo": [1, 2, 3],
    })
    ts.update(update_df)
    state = ts._pooled_states[key]
    assert len(state.groups) == 4  # promo {0, 1, 2, 3}
    # parent calendar advanced to [1,2,3,4]; every sibling bucket sees length 4
    for bid in state.next_time_index_by_bucket:
        assert state.next_time_index_by_bucket[bid] == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_update_sparse_then_dense(engine):
    """Fit on sparse partition transitions, then apply dense per-step updates;
    the resulting state must match a from-scratch fit on the combined data
    (same per-bucket aggregates and parent calendars).
    """
    def _aggs_by_key(state):
        groups = state.groups.to_pandas() if hasattr(state.groups, "to_pandas") else state.groups
        out = {}
        for bid, agg in state._ts_aggs.items():
            bkey = tuple(groups.iloc[bid].tolist())
            out[bkey] = (
                agg.unique_times.tolist(),
                np.round(agg.sums, 6).tolist(),
                agg.counts.tolist(),
            )
        return out

    def _build():
        return TimeSeries(
            freq=1,
            lag_transforms={1: [RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])]},
        )

    base = {
        "unique_id": ["a", "a", "a", "b", "b", "b"], "ds": [1, 2, 3, 1, 2, 3],
        "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0], "promo": [0, 0, 0, 0, 0, 0],
    }
    dense_steps = [
        {"ds": 4, "promo": [1, 0], "y": [4.0, 40.0]},
        {"ds": 5, "promo": [0, 1], "y": [5.0, 50.0]},
        {"ds": 6, "promo": [1, 1], "y": [6.0, 60.0]},
    ]

    ts_incr = _build()
    ts_incr.fit_transform(
        _make_df(engine, base), id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    rows = [base]
    for step in dense_steps:
        u = {"unique_id": ["a", "b"], "ds": [step["ds"], step["ds"]],
             "y": step["y"], "promo": step["promo"]}
        ts_incr.update(_make_df(engine, u))
        rows.append(u)

    combined = {k: sum((r[k] for r in rows), []) for k in base}
    ts_scratch = _build()
    ts_scratch.fit_transform(
        _make_df(engine, combined), id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )

    key = ("nonlocal", (), ("promo",))
    si, ss = ts_incr._pooled_states[key], ts_scratch._pooled_states[key]
    assert _aggs_by_key(si) == _aggs_by_key(ss)
    assert {pid: grid.tolist() for pid, grid in si._parent_time_grids.items()} == \
        {pid: grid.tolist() for pid, grid in ss._parent_time_grids.items()}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_static_features_explicit_with_partition_cols(engine):
    """Explicit static_features are honored while partition columns are excluded.

    With static_features=["brand","region"] and partition_by=["promo"], the
    fitted static set keeps brand/region but drops promo (it is dynamic and
    re-supplied via X_df at predict), and promo never enters features_order_.
    """
    df = _make_df(engine, {
        "unique_id": ["a"] * 4 + ["b"] * 4,
        "ds": [1, 2, 3, 4] * 2,
        "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        "brand": ["x"] * 4 + ["y"] * 4,
        "region": ["N"] * 4 + ["S"] * 4,
        "promo": [0, 1, 0, 1, 1, 0, 1, 0],
    })
    tfm = RollingMean(2, min_samples=1, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["brand", "region"],
    )
    static_cols = set(ts.static_features_.columns)
    assert {"brand", "region"} <= static_cols
    assert "promo" not in static_cols
    assert "promo" not in ts.features_order_


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_rolling_min_samples_boundary(engine):
    """min_samples is the exact coverage threshold inside a partition bucket.

    RollingMean(window_size=3, min_samples=2) at lag 1 over a single promo=1
    bucket: at ds=2 the window holds 1 observation (< min_samples) -> NaN; at
    ds=3 it holds exactly 2 (== min_samples) -> the mean (10+20)/2 = 15.
    """
    df = _make_df(engine, {
        "unique_id": ["a"] * 5,
        "ds": [1, 2, 3, 4, 5],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "promo": [1, 1, 1, 1, 1],
    })
    tfm = RollingMean(3, min_samples=2, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    out = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    col = tfm._get_name(1)
    if engine == "polars":
        out = out.to_pandas()
    vals = out.sort_values("ds")[col].to_numpy()
    # ds=1,2 below threshold -> NaN; ds=3 hits exactly min_samples -> value
    assert np.isnan(vals[0]) and np.isnan(vals[1])
    np.testing.assert_allclose(vals[2], 15.0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_partition_cv_fold_independent(engine):
    """cross_validation runs across folds with partition_by and does not leak
    dynamic buckets between folds.
    """
    from mlforecast.forecast import MLForecast
    from sklearn.ensemble import HistGradientBoostingRegressor

    n_times = 12
    df = _make_df(engine, {
        "unique_id": ["a"] * n_times + ["b"] * n_times,
        "ds": list(range(1, n_times + 1)) * 2,
        "y": [float(i) for i in range(1, n_times + 1)]
           + [float(i * 10) for i in range(1, n_times + 1)],
        "promo": ([0, 1] * (n_times // 2)) * 2,
    })
    fcst = MLForecast(
        models=[HistGradientBoostingRegressor(max_iter=5)],
        freq=1,
        lag_transforms={1: [RollingMean(2, min_samples=1, global_=True, partition_by=["promo"])]},
    )
    cv = fcst.cross_validation(df, n_windows=2, h=2, static_features=[])
    assert len(cv) == 2 * 2 * 2  # n_windows * h * n_series
    pred_vals = cv["HistGradientBoostingRegressor"].to_numpy()
    assert not np.any(np.isnan(pred_vals))
    # no bucket bleed across folds: only promo {0, 1} ever exists
    state = fcst.ts._pooled_states[("nonlocal", (), ("promo",))]
    assert len(state.groups) == 2


# === Tests ported from feature/groupby_with_range_semantics ===


class TestValidateDataWarning:
    """Warning when validate_data=False with pooled transforms."""

    def _make_fcst(self, transforms):
        from sklearn.linear_model import LinearRegression
        from mlforecast.forecast import MLForecast

        return MLForecast(
            models=[LinearRegression()],
            freq=1,
            lags=[1],
            lag_transforms=transforms,
        )

    def _make_simple_df(self):
        return pd.DataFrame(
            {
                "unique_id": ["a"] * 4 + ["b"] * 4,
                "ds": [1, 2, 3, 4] * 2,
                "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                "brand": ["x"] * 4 + ["x"] * 4,
            }
        )

    def test_warns_global(self):
        fcst = self._make_fcst({1: [RollingMean(window_size=2, global_=True)]})
        df = self._make_simple_df()
        with pytest.warns(UserWarning, match="Pooled.*validate_data"):
            fcst.preprocess(df, static_features=["brand"], validate_data=False)

    def test_warns_groupby(self):
        fcst = self._make_fcst(
            {1: [RollingMean(window_size=2, groupby=["brand"])]}
        )
        df = self._make_simple_df()
        with pytest.warns(UserWarning, match="Pooled.*validate_data"):
            fcst.preprocess(df, static_features=["brand"], validate_data=False)

    def test_warns_partition_by(self):
        fcst = self._make_fcst(
            {1: [RollingMean(window_size=2, global_=True, partition_by=["brand"])]}
        )
        df = self._make_simple_df()
        with pytest.warns(UserWarning, match="Pooled.*validate_data"):
            fcst.preprocess(df, static_features=["brand"], validate_data=False)

    def test_no_warning_when_validated(self):
        fcst = self._make_fcst({1: [RollingMean(window_size=2, global_=True)]})
        df = self._make_simple_df()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fcst.preprocess(df, static_features=["brand"], validate_data=True)

    def test_no_warning_without_pooled(self):
        fcst = self._make_fcst({1: [RollingMean(window_size=2)]})
        df = self._make_simple_df()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fcst.preprocess(df, validate_data=False)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ewm_lag_semantics(engine):
    """EWM with lag > 1 must only consume timestamps up to k-lag, not k-1.

    Regression test for the two-pointer fix in _ewm_from_agg.
    """
    df = _make_df(engine, {
        "unique_id": ["a"] * 4 + ["b"] * 4,
        "ds": list(range(4)) * 2,
        "y": [6.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0, 9.0],
    })
    tfm = ExponentiallyWeightedMean(alpha=0.5, global_=True)
    ts = TimeSeries(freq=1, lag_transforms={2: [tfm]})
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y", dropna=False,
    )
    col = tfm._get_name(2)
    if engine == "pandas":
        vals_a = result.loc[result["unique_id"] == "a", col].values
    else:
        vals_a = result.filter(pl.col("unique_id") == "a")[col].to_numpy()
    # Per-timestamp global means: t0=6, t1=7, t2=8, t3=9
    # lag=2, alpha=0.5:
    #   k=0: consume up to ts -2 → nothing → NaN
    #   k=1: consume up to ts -1 → nothing → NaN
    #   k=2: consume up to ts  0 → ewm=6.0
    #   k=3: consume up to ts  1 → ewm=0.5*7 + 0.5*6 = 6.5
    np.testing.assert_allclose(vals_a, [np.nan, np.nan, 6.0, 6.5], equal_nan=True)

    # Prediction: at t=4, consume up to ts 2 → ewm(t0,t1,t2) = 7.25
    ts._predict_setup()
    features = ts._update_features()
    pred_vals = features[col].to_numpy()
    np.testing.assert_allclose(pred_vals, [7.25, 7.25])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ewm_lag_semantics_groupby(engine):
    """EWM lag semantics hold in groupby mode too."""
    df = _make_df(engine, {
        "unique_id": ["a"] * 4 + ["b"] * 4,
        "ds": list(range(4)) * 2,
        "y": [6.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0, 9.0],
        "grp": ["X"] * 8,
    })
    tfm = ExponentiallyWeightedMean(alpha=0.5, groupby=["grp"])
    ts = TimeSeries(freq=1, lag_transforms={2: [tfm]})
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )
    col = tfm._get_name(2)
    if engine == "pandas":
        vals_a = result.loc[result["unique_id"] == "a", col].values
    else:
        vals_a = result.filter(pl.col("unique_id") == "a")[col].to_numpy()
    np.testing.assert_allclose(vals_a, [np.nan, np.nan, 6.0, 6.5], equal_nan=True)

    ts._predict_setup()
    features = ts._update_features()
    np.testing.assert_allclose(features[col].to_numpy(), [7.25, 7.25])


def _fit_and_collect(engine, lag, tfms, y_a, y_b, n_times, grp=None):
    """Helper: fit global or groupby transforms, return per-series-a preprocess
    values and prediction values."""
    data = {
        "unique_id": ["a"] * n_times + ["b"] * n_times,
        "ds": list(range(n_times)) * 2,
        "y": y_a + y_b,
    }
    if grp is not None:
        data["grp"] = [grp] * (2 * n_times)
    df = _make_df(engine, data)

    ts = TimeSeries(freq=1, lag_transforms={lag: tfms})
    static = ["grp"] if grp else None
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=static,
    )
    ts._predict_setup()
    features = ts._update_features()
    out = {}
    for tfm in tfms:
        col = tfm._get_name(lag)
        if engine == "pandas":
            vals_a = result.loc[result["unique_id"] == "a", col].values
        else:
            vals_a = result.filter(pl.col("unique_id") == "a")[col].to_numpy()
        pred = features[col].to_numpy()
        out[col] = {"preprocess": vals_a, "predict": pred}
    return out


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pooled_transforms_lag3_global(engine):
    """All decomposable transforms produce correct values with lag=3 in global mode."""
    y_a = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    y_b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    # Per-ts global means: t0=1.5, t1=3.5, t2=5.5, t3=7.5, t4=9.5, t5=11.5
    lag = 3
    tfms = [
        RollingMean(window_size=3, global_=True),
        RollingStd(window_size=3, global_=True),
        RollingMin(window_size=3, global_=True),
        RollingMax(window_size=3, global_=True),
        ExpandingMean(global_=True),
        ExpandingStd(global_=True),
        ExpandingMin(global_=True),
        ExpandingMax(global_=True),
    ]
    out = _fit_and_collect(engine, lag, tfms, y_a, y_b, 6)

    nan = np.nan
    # Preprocess (series a, 6 timestamps) — lag=3 means feature at k uses obs up to k-3
    # Rolling window=3: uses obs in [k-3-2, k-3] = [k-5, k-3]
    #   k=0..3: not enough → NaN; k=4: t0,t1 → count=4≥3 → mean=2.5; k=5: t0,t1,t2 → count=6≥3 → mean=3.5
    expected = {
        "global_rolling_mean_lag3_window_size3":     ([nan, nan, nan, nan, 2.5, 3.5],       [5.5, 5.5]),
        "global_rolling_std_lag3_window_size3":       ([nan, nan, nan, nan, 1.290994, 1.870829], [1.870829, 1.870829]),
        "global_rolling_min_lag3_window_size3":       ([nan, nan, nan, nan, 1.0, 1.0],       [3.0, 3.0]),
        "global_rolling_max_lag3_window_size3":       ([nan, nan, nan, nan, 4.0, 6.0],       [8.0, 8.0]),
        "global_expanding_mean_lag3":                 ([nan, nan, nan, 1.5, 2.5, 3.5],       [4.5, 4.5]),
        "global_expanding_std_lag3":                  ([nan, nan, nan, 0.707107, 1.290994, 1.870829], [2.449490, 2.449490]),
        "global_expanding_min_lag3":                  ([nan, nan, nan, 1.0, 1.0, 1.0],       [1.0, 1.0]),
        "global_expanding_max_lag3":                  ([nan, nan, nan, 2.0, 4.0, 6.0],       [8.0, 8.0]),
    }
    for col, (exp_pre, exp_pred) in expected.items():
        np.testing.assert_allclose(
            out[col]["preprocess"], exp_pre, atol=1e-5, equal_nan=True,
            err_msg=f"preprocess mismatch for {col}",
        )
        np.testing.assert_allclose(
            out[col]["predict"], exp_pred, atol=1e-5,
            err_msg=f"predict mismatch for {col}",
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pooled_transforms_lag2_groupby(engine):
    """Decomposable transforms produce correct values with lag=2 in groupby mode."""
    # Group X: series a=[1,2,3,4,5], b=[10,20,30,40,50]
    # Per-ts global means in group X: t0=5.5, t1=11, t2=16.5, t3=22, t4=27.5
    y_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_b = [10.0, 20.0, 30.0, 40.0, 50.0]
    lag = 2
    tfms = [
        RollingMean(window_size=3, groupby=["grp"]),
        RollingMin(window_size=3, groupby=["grp"]),
        RollingMax(window_size=3, groupby=["grp"]),
        ExpandingMean(groupby=["grp"]),
        ExpandingMin(groupby=["grp"]),
        ExpandingMax(groupby=["grp"]),
        ExponentiallyWeightedMean(alpha=0.5, groupby=["grp"]),
    ]
    out = _fit_and_collect(engine, lag, tfms, y_a, y_b, 5, grp="X")

    nan = np.nan
    expected = {
        "groupby_grp_rolling_mean_lag2_window_size3":  [nan, nan, nan, 8.25, 11.0],
        "groupby_grp_rolling_min_lag2_window_size3":   [nan, nan, nan, 1.0, 1.0],
        "groupby_grp_rolling_max_lag2_window_size3":   [nan, nan, nan, 20.0, 30.0],
        "groupby_grp_expanding_mean_lag2":             [nan, nan, 5.5, 8.25, 11.0],
        "groupby_grp_expanding_min_lag2":              [nan, nan, 1.0, 1.0, 1.0],
        "groupby_grp_expanding_max_lag2":              [nan, nan, 10.0, 20.0, 30.0],
        "groupby_grp_exponentially_weighted_mean_lag2_alpha0.5": [nan, nan, 5.5, 8.25, 12.375],
    }
    for col, exp_pre in expected.items():
        np.testing.assert_allclose(
            out[col]["preprocess"], exp_pre, atol=1e-5, equal_nan=True,
            err_msg=f"preprocess mismatch for {col}",
        )


@pytest.mark.parametrize("tfm_factory", [
    lambda m: RollingMean(window_size=4, **m),
    lambda m: RollingStd(window_size=4, **m),
    lambda m: RollingMin(window_size=4, **m),
    lambda m: RollingMax(window_size=4, **m),
    lambda m: ExpandingMean(**m),
    lambda m: ExpandingStd(**m),
    lambda m: ExpandingMin(**m),
    lambda m: ExpandingMax(**m),
    lambda m: ExponentiallyWeightedMean(alpha=0.3, **m),
], ids=[
    "RollingMean", "RollingStd", "RollingMin", "RollingMax",
    "ExpandingMean", "ExpandingStd", "ExpandingMin", "ExpandingMax", "EWM",
])
@pytest.mark.parametrize("lag", _LAGS)
def test_fast_vs_slow_equivalence(tfm_factory, lag):
    """Fast path (aggregate-based) matches slow path (row-level) at every lag.

    Exercises all three code paths: fit (_compute_from_aggregates via
    compute_pooled_features), preprocess (_compute_ts_level_from_aggs),
    and predict (_compute_latest_from_aggs).
    """
    from mlforecast.pooled import compute_pooled_features

    rng = np.random.default_rng(42)
    n_series, n_times = 8, 12
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    y = rng.standard_normal(n_series * n_times)
    grps = np.repeat(["A"] * (n_series // 2) + ["B"] * (n_series // 2), n_times)
    df = pd.DataFrame({"unique_id": ids, "ds": times, "y": y, "grp": grps})

    # --- fit path: fast vs slow via compute_pooled_features ---
    tfm_g = tfm_factory({"global_": True})
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm_g]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )
    state = ts._pooled_states[("global", (), ())]
    col = tfm_g._get_name(lag)
    fitted_tfm = ts.transforms[col]

    fast = compute_pooled_features(state, {col: fitted_tfm})
    saved_aggs = state._ts_aggs
    state._ts_aggs = {}
    slow = compute_pooled_features(state, {col: fitted_tfm})
    state._ts_aggs = saved_aggs

    np.testing.assert_allclose(
        fast[col], slow[col], atol=1e-10, equal_nan=True,
        err_msg=f"fit fast vs slow mismatch for {col}",
    )

    # --- preprocess path: global _compute_ts_level_from_aggs ---
    result_fast = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )
    fast_pre = result_fast[col].values

    ts_slow = TimeSeries(
        freq=1,
        lag_transforms={lag: [tfm_factory({"global_": True})]},
    )
    ts_slow._fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=["grp"],
    )
    ts_slow._pooled_states[("global", (), ())]._ts_aggs = {}
    result_slow = ts_slow._transform(df=df, dropna=False)
    slow_pre = result_slow[col].values
    np.testing.assert_allclose(
        fast_pre, slow_pre, atol=1e-10, equal_nan=True,
        err_msg=f"preprocess global fast vs slow for {col}",
    )

    # --- preprocess path: groupby _compute_ts_level_from_aggs ---
    tfm_grp = tfm_factory({"groupby": ["grp"]})
    ts_grp = TimeSeries(freq=1, lag_transforms={lag: [tfm_grp]})
    result_grp = ts_grp.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )
    col_grp = tfm_grp._get_name(lag)
    fast_grp_pre = result_grp[col_grp].values

    ts_grp_slow = TimeSeries(
        freq=1,
        lag_transforms={lag: [tfm_factory({"groupby": ["grp"]})]},
    )
    ts_grp_slow._fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=["grp"],
    )
    for st in ts_grp_slow._pooled_states.values():
        st._ts_aggs = {}
        st._idsorted_to_bucket_pos = None
    result_grp_slow = ts_grp_slow._transform(df=df, dropna=False)
    slow_grp_pre = result_grp_slow[col_grp].values
    np.testing.assert_allclose(
        fast_grp_pre, slow_grp_pre, atol=1e-10, equal_nan=True,
        err_msg=f"preprocess groupby fast vs slow for {col_grp}",
    )

    # --- predict path: _compute_latest_from_aggs ---
    ts2 = TimeSeries(freq=1, lag_transforms={lag: [tfm_factory({"global_": True})]})
    ts2.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )
    ts2._predict_setup()
    features = ts2._update_features()
    pred_col = list(ts2.transforms.keys())[0]
    pred_vals = features[pred_col].to_numpy()
    assert not np.all(np.isnan(pred_vals)), f"predict returned all NaN for {pred_col}"


@pytest.mark.parametrize("tfm_factory", [
    lambda m: RollingMean(window_size=4, **m),
    lambda m: RollingStd(window_size=4, **m),
    lambda m: RollingMin(window_size=4, **m),
    lambda m: RollingMax(window_size=4, **m),
    lambda m: ExpandingMean(**m),
    lambda m: ExpandingStd(**m),
    lambda m: ExpandingMin(**m),
    lambda m: ExpandingMax(**m),
    lambda m: ExponentiallyWeightedMean(alpha=0.3, **m),
], ids=[
    "RollingMean", "RollingStd", "RollingMin", "RollingMax",
    "ExpandingMean", "ExpandingStd", "ExpandingMin", "ExpandingMax", "EWM",
])
@pytest.mark.parametrize("lag", _LAGS)
def test_fast_vs_slow_partition(tfm_factory, lag):
    """Fast path matches slow path for partition_by (global+partition and groupby+partition)."""
    from mlforecast.pooled import compute_pooled_features

    rng = np.random.default_rng(99)
    n_series, n_times = 6, 10
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    y = rng.standard_normal(n_series * n_times)
    promo = np.tile(np.where(np.arange(n_times) % 3 == 0, "Y", "N"), n_series)
    grp = np.repeat(["A"] * (n_series // 2) + ["B"] * (n_series // 2), n_times)
    df = pd.DataFrame({"unique_id": ids, "ds": times, "y": y, "promo": promo, "grp": grp})

    # --- global + partition_by: fit path ---
    tfm_gp = tfm_factory({"global_": True, "partition_by": ["promo"]})
    ts = TimeSeries(freq=1, lag_transforms={lag: [tfm_gp]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["promo"],
    )
    key = ("nonlocal", (), ("promo",))
    state = ts._pooled_states[key]
    col = tfm_gp._get_name(lag)
    fitted = ts.transforms[col]

    fast = compute_pooled_features(state, {col: fitted})
    saved_aggs = state._ts_aggs
    state._ts_aggs = {}
    slow = compute_pooled_features(state, {col: fitted})
    state._ts_aggs = saved_aggs

    np.testing.assert_allclose(
        fast[col], slow[col], atol=1e-10, equal_nan=True,
        err_msg=f"fit global+partition fast vs slow for {col}",
    )

    # --- global + partition_by: preprocess path ---
    result_fast = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["promo"],
    )
    fast_pre = result_fast[col].values

    ts_slow = TimeSeries(
        freq=1,
        lag_transforms={lag: [tfm_factory({"global_": True, "partition_by": ["promo"]})]},
    )
    ts_slow._fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=["promo"],
    )
    for st in ts_slow._pooled_states.values():
        st._ts_aggs = {}
        st._idsorted_to_bucket_pos = None
    slow_pre = ts_slow._transform(df=df, dropna=False)[col].values
    np.testing.assert_allclose(
        fast_pre, slow_pre, atol=1e-10, equal_nan=True,
        err_msg=f"preprocess global+partition fast vs slow for {col}",
    )

    # --- groupby + partition_by: preprocess path ---
    tfm_grp = tfm_factory({"groupby": ["grp"], "partition_by": ["promo"]})
    ts_grp = TimeSeries(freq=1, lag_transforms={lag: [tfm_grp]})
    col_grp = tfm_grp._get_name(lag)
    fast_grp = ts_grp.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=["grp"],
    )[col_grp].values

    ts_grp_slow = TimeSeries(
        freq=1,
        lag_transforms={lag: [tfm_factory({"groupby": ["grp"], "partition_by": ["promo"]})]},
    )
    ts_grp_slow._fit(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=["grp"],
    )
    for st in ts_grp_slow._pooled_states.values():
        st._ts_aggs = {}
        st._idsorted_to_bucket_pos = None
    slow_grp = ts_grp_slow._transform(df=df, dropna=False)[col_grp].values
    np.testing.assert_allclose(
        fast_grp, slow_grp, atol=1e-10, equal_nan=True,
        err_msg=f"preprocess groupby+partition fast vs slow for {col_grp}",
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_prediction_fast_path_partition(engine):
    """Multi-step predict with fast path + partition_by produces finite values."""
    n_series, n_times = 4, 8
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    times = np.tile(range(n_times), n_series)
    y = np.random.default_rng(7).standard_normal(n_series * n_times)
    promo = np.tile([0, 0, 1, 1, 0, 1, 0, 1][:n_times], n_series)
    rows = {
        "unique_id": ids.tolist(),
        "ds": times.tolist(),
        "y": y.tolist(),
        "promo": promo.tolist(),
    }
    df = _make_df(engine, rows)

    tfm = RollingMean(window_size=3, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        static_features=[],
    )
    ts._predict_setup()
    for step in range(3):
        features = ts._update_features()
        col = tfm._get_name(1)
        vals = features[col].to_numpy()
        assert not np.all(np.isnan(vals)), f"step {step}: all NaN for {col}"
        ts._update_y(vals)


def test_partition_ewm_skips_missing_parent_ordinals():
    """EWM on a partition bucket with gapped parent ordinals [0,1,4,5]
    decays only across observed bucket timestamps, not across missing
    parent ordinals 2 and 3."""
    df = pd.DataFrame({
        "unique_id": ["a"] * 8 + ["b"] * 8,
        "ds": list(range(8)) * 2,
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
              12.0, 22.0, 32.0, 42.0, 52.0, 62.0, 72.0, 82.0],
        "promo": [0, 0, 1, 1, 0, 0, 1, 1] * 2,
    })

    alpha = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tfm = ExponentiallyWeightedMean(alpha=alpha, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    col = tfm._get_name(1)
    vals = result[col].values

    # promo=0 bucket: parent ordinals [0,1,4,5], aggregate means [11,21,51,61]
    # Two-pointer EWM (lag=1, alpha=0.5):
    #   ord 0: upper=-1 → NaN
    #   ord 1: consume ord 0 (mean=11) → 11.0
    #   ord 4: consume ord 1 (mean=21) → 0.5*21 + 0.5*11 = 16.0
    #          ords 2,3 are missing — NOT consumed, no extra decay
    #   ord 5: consume ord 4 (mean=51) → 0.5*51 + 0.5*16 = 33.5
    expected_p0 = [np.nan, 11.0, 16.0, 33.5]

    # promo=1 bucket: parent ordinals [2,3,6,7], aggregate means [31,41,71,81]
    #   ord 2: upper=1, nothing observed ≤ 1 in this bucket → NaN
    #   ord 3: consume ord 2 (mean=31) → 31.0
    #   ord 6: consume ord 3 (mean=41) → 0.5*41 + 0.5*31 = 36.0
    #   ord 7: consume ord 6 (mean=71) → 0.5*71 + 0.5*36 = 53.5
    expected_p1 = [np.nan, 31.0, 36.0, 53.5]

    promo = df["promo"].values
    for start in range(0, len(df), 8):
        chunk = vals[start : start + 8]
        p = promo[start : start + 8]
        np.testing.assert_allclose(
            chunk[p == 0], expected_p0, atol=1e-10, equal_nan=True,
        )
        np.testing.assert_allclose(
            chunk[p == 1], expected_p1, atol=1e-10, equal_nan=True,
        )


def test_global_partition_ewm_uses_timestamp_mean_once():
    """Multiple series in the same partition bucket at the same timestamp
    contribute their aggregate mean once to EWM, not once per row."""
    df = pd.DataFrame({
        "unique_id": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
        "ds": list(range(5)) * 3,
        "y": [10.0, 20.0, 30.0, 40.0, 50.0,
              12.0, 22.0, 32.0, 42.0, 52.0,
              14.0, 24.0, 34.0, 44.0, 54.0],
        "promo": [0] * 15,
    })

    alpha = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tfm = ExponentiallyWeightedMean(alpha=alpha, global_=True, partition_by=["promo"])
    ts = TimeSeries(freq=1, lag_transforms={1: [tfm]})
    result = ts.fit_transform(
        df, id_col="unique_id", time_col="ds", target_col="y",
        dropna=False, static_features=[],
    )
    col = tfm._get_name(1)
    vals = result[col].values

    # Per-timestamp means: [12, 22, 32, 42, 52]
    # EWM (lag=1, alpha=0.5):
    #   ord 0: NaN
    #   ord 1: 12.0
    #   ord 2: 0.5*22 + 0.5*12 = 17.0
    #   ord 3: 0.5*32 + 0.5*17 = 24.5
    #   ord 4: 0.5*42 + 0.5*24.5 = 33.25
    expected = [np.nan, 12.0, 17.0, 24.5, 33.25]

    # If each row contributed individually (3 rows at each timestamp):
    # ord 0 would consume 10→12→14 with three EWM steps, giving a different
    # final ewm at ord 0 that propagates differently. The expected values
    # above only hold when each timestamp contributes its mean once.
    for i in range(3):
        np.testing.assert_allclose(
            vals[i * 5 : (i + 1) * 5], expected, atol=1e-10, equal_nan=True,
        )


def test_partition_ewm_warning():
    """ExponentiallyWeightedMean emits a warning when partition_by is set,
    and does not warn without partition_by."""
    with pytest.warns(UserWarning, match="Partitioned EWM"):
        ExponentiallyWeightedMean(alpha=0.3, partition_by=["promo"])
    with pytest.warns(UserWarning, match="Partitioned EWM"):
        ExponentiallyWeightedMean(alpha=0.3, global_=True, partition_by=["promo"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ExponentiallyWeightedMean(alpha=0.3)
        ExponentiallyWeightedMean(alpha=0.3, global_=True)
        ExponentiallyWeightedMean(alpha=0.3, groupby=["grp"])
