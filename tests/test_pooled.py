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
    state = ts._pooled_states[("global", (), ())]
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
    with pytest.raises(ValueError, match="Partition/group key column"):
        fcst.predict(h=1, X_df=future_df)


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
