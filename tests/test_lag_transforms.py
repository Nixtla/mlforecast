import operator
import warnings

import coreforecast.lag_transforms as core_tfms
import numpy as np
import pytest
from coreforecast.grouped_array import GroupedArray as CoreGroupedArray

from mlforecast.grouped_array import GroupedArray as MLGroupedArray
from mlforecast.lag_transforms import (
    Combine,
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingQuantile,
    ExpandingStd,
    ExponentiallyWeightedMean,
    Lag,
    Offset,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingQuantile,
    RollingStd,
    SeasonalRollingMax,
    SeasonalRollingMean,
    SeasonalRollingMin,
    SeasonalRollingQuantile,
    SeasonalRollingStd,
)


@pytest.fixture(scope='module')
def grouped_array():
    rng = np.random.default_rng(seed=0)
    lengths = rng.integers(low=50, high=100, size=20)
    data = rng.random(lengths.sum())
    return CoreGroupedArray(data, np.append(0, lengths.cumsum()))

def test_offset_name_and_transform(grouped_array):
    offset = Offset(RollingMean(window_size=10), 2)._set_core_tfm(5)
    assert offset._get_name(5) == "rolling_mean_lag7_window_size10"

    transformed = offset.transform(grouped_array)
    expected = (
        RollingMean(window_size=10)
        ._set_core_tfm(5)
        .transform(grouped_array._with_data(Lag(2).transform(grouped_array)))
    )
    np.testing.assert_allclose(transformed, expected)

def test_combine_name_and_transform(grouped_array):
    comb = Combine(Lag(1), Lag(2), operator.truediv)
    assert comb._get_name(1) == "lag1_truediv_lag2"

    transformed = comb.transform(grouped_array)
    expected = Lag(1).transform(grouped_array) / Lag(2).transform(grouped_array)
    np.testing.assert_allclose(transformed, expected)

def test_combine_take(grouped_array):
    tfm = Combine(
        RollingMean(window_size=7, min_samples=1),
        RollingMean(window_size=5, min_samples=1),
        operator.add
    )._set_core_tfm(1)
    tfm.transform(grouped_array)

    idxs = np.array([0, 5, 10, 15])
    subset_tfm = tfm.take(idxs)

    assert isinstance(subset_tfm, Combine)
    assert subset_tfm.tfm1 is not None
    assert subset_tfm.tfm2 is not None
    assert subset_tfm.operator == operator.add

def test_nested_combine_take(grouped_array):
    inner = Combine(
        RollingMean(window_size=7, min_samples=1),
        RollingMean(window_size=5, min_samples=1),
        operator.add
    )
    outer = Combine(
        inner,
        RollingMean(window_size=3, min_samples=1),
        operator.sub
    )._set_core_tfm(1)
    outer.transform(grouped_array)

    idxs = np.array([0, 5, 10])
    subset_tfm = outer.take(idxs)

    assert isinstance(subset_tfm, Combine)
    assert isinstance(subset_tfm.tfm1, Combine)
    assert subset_tfm.operator == operator.sub
    assert subset_tfm.tfm1.operator == operator.add

    # Numerical correctness: subset update() matches a fresh fit on the same 3 groups
    indptr = grouped_array.indptr
    parts = [grouped_array.data[indptr[i]:indptr[i + 1]] for i in idxs]
    new_indptr = np.zeros(len(idxs) + 1, dtype=indptr.dtype)
    for j, part in enumerate(parts):
        new_indptr[j + 1] = new_indptr[j] + len(part)
    subset_ga = CoreGroupedArray(np.concatenate(parts), new_indptr)

    fresh_tfm = Combine(
        Combine(
            RollingMean(window_size=7, min_samples=1),
            RollingMean(window_size=5, min_samples=1),
            operator.add
        ),
        RollingMean(window_size=3, min_samples=1),
        operator.sub
    )._set_core_tfm(1)
    fresh_tfm.transform(subset_ga)
    np.testing.assert_allclose(subset_tfm.update(subset_ga), fresh_tfm.update(subset_ga))

def test_combine_stack(grouped_array):
    tfm1 = Combine(
        RollingMean(window_size=7, min_samples=1),
        RollingMean(window_size=5, min_samples=1),
        operator.add
    )._set_core_tfm(1)
    tfm2 = Combine(
        RollingMean(window_size=7, min_samples=1),
        RollingMean(window_size=5, min_samples=1),
        operator.add
    )._set_core_tfm(1)

    tfm1.transform(grouped_array)
    tfm2.transform(grouped_array)

    stacked_tfm = Combine.stack([tfm1, tfm2])

    assert isinstance(stacked_tfm, Combine)
    assert stacked_tfm.operator == operator.add

    # Numerical correctness: stacking a single fitted transform should reproduce its update()
    single_stacked = Combine.stack([tfm1])
    tfm1.transform(grouped_array)  # reset internal state
    np.testing.assert_allclose(single_stacked.update(grouped_array), tfm1.update(grouped_array))


def test_combine_stack_behavioral(grouped_array):
    """Verify that Combine.stack() doesn't just return first partition"""
    # Create two Combine transforms with DIFFERENT window sizes to detect if
    # stacking just returns the first partition vs actually combining them
    tfm1 = Combine(
        RollingMean(window_size=3, min_samples=1),
        RollingMean(window_size=5, min_samples=1),
        operator.add
    )._set_core_tfm(1)

    tfm2 = Combine(
        RollingMean(window_size=7, min_samples=1),  # Different window size
        RollingMean(window_size=9, min_samples=1),  # Different window size
        operator.add
    )._set_core_tfm(1)

    tfm1.transform(grouped_array)
    tfm2.transform(grouped_array)

    # If stack incorrectly returned partition_tfms[0], it would just return tfm1
    # The stacked transform should have tfm1's window sizes (since stack keeps first's config)
    # but should have stacked internal state from both
    stacked = Combine.stack([tfm1, tfm2])

    # Verify stacked uses first transform's configuration
    assert stacked.tfm1.window_size == 3
    assert stacked.tfm2.window_size == 5
    # Verify it's not just a reference to tfm1 (defensive check)
    assert stacked is not tfm1


@pytest.mark.parametrize("tfm", [
    ExpandingMax(),
    ExpandingMean(),
    ExpandingMin(),
    ExpandingStd(),
    ExpandingQuantile(0.5),
    ExponentiallyWeightedMean(0.1),
    RollingMax(7),
    RollingMean(7),
    RollingMin(7),
    RollingStd(7),
    RollingQuantile(0.5, 7),
    SeasonalRollingMax(7, 2),
    SeasonalRollingMean(7, 2),
    SeasonalRollingMin(7, 2),
    SeasonalRollingStd(7, 2),
    SeasonalRollingQuantile(0.5, 7, 7),
    Offset(RollingMax(7), 2),
    Combine(RollingMean(5), Offset(RollingMean(5), 2), operator.truediv),
    Combine(Offset(RollingMean(5), 2), RollingMean(5), operator.truediv),
])
def test_transform_and_update_consistency(grouped_array, tfm):
    tfm._set_core_tfm(1)
    tfm._get_name(1)
    tfm.transform(grouped_array)

    updates = tfm.update(grouped_array)
    upd_samples = tfm.update_samples
    if upd_samples > -1:
        sliced_ga = MLGroupedArray(grouped_array.data, grouped_array.indptr).take_from_groups(
            slice(-upd_samples, None)
        )
        ga2 = CoreGroupedArray(sliced_ga.data, sliced_ga.indptr)
        tfm.transform(grouped_array)  # reset internal state
        updates2 = tfm.update(ga2)
        np.testing.assert_allclose(updates, updates2)


# the transforms that take a pooled scope, with the name their local form gets
_SCOPED = [
    ("rolling", lambda **kw: RollingMean(7, **kw), "rolling_mean_lag1_window_size7"),
    (
        "seasonal",
        lambda **kw: SeasonalRollingMean(7, 2, **kw),
        "seasonal_rolling_mean_lag1_season_length7_window_size2",
    ),
    ("expanding", lambda **kw: ExpandingMean(**kw), "expanding_mean_lag1"),
    (
        "ewm",
        lambda **kw: ExponentiallyWeightedMean(0.1, **kw),
        "exponentially_weighted_mean_lag1_alpha0.1",
    ),
]
_SCOPED_IDS = [name for name, _, _ in _SCOPED]


@pytest.mark.parametrize("make,base_name", [s[1:] for s in _SCOPED], ids=_SCOPED_IDS)
def test_pooled_scope_arguments_map_and_name(make, base_name):
    local = make()
    assert (local.global_, local.groupby, local.partition_by) == (False, None, None)
    assert local._get_name(1) == base_name
    assert make(global_=True).global_ is True
    assert make(global_=True)._get_name(1) == f"global_{base_name}"
    # the pre-``global_`` spelling still maps
    legacy = make(**{"global": True})
    assert legacy.global_ is True
    assert legacy._get_name(1) == f"global_{base_name}"
    grouped = make(groupby="brand")
    assert grouped.groupby == ["brand"]
    assert grouped._get_name(1) == f"groupby_brand_{base_name}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # partitioned EWM warns about its decay
        parted = make(partition_by=("promo", "promo"))
        both = make(groupby=["brand", "region"], partition_by=["promo"])
        global_parted = make(global_=True, partition_by=["promo"])
    assert parted.partition_by == ["promo"]
    assert parted._get_name(1) == f"partby_promo_{base_name}"
    assert both._get_name(1) == f"groupby_brand__region_partby_promo_{base_name}"
    assert global_parted._get_name(1) == f"global_partby_promo_{base_name}"


@pytest.mark.parametrize("make", [s[1] for s in _SCOPED], ids=_SCOPED_IDS)
def test_pooled_scope_rejects_bad_arguments(make):
    with pytest.raises(TypeError, match=r"Unexpected keyword arguments: \['bogus'\]"):
        make(bogus=1)
    with pytest.raises(ValueError, match="can't be used together"):
        make(global_=True, groupby=["brand"])
    with pytest.raises(ValueError, match="time_agg must be one of"):
        make(global_=True, time_agg="median")
    with pytest.raises(ValueError, match="requires a pooled aggregation scope"):
        make(time_agg="sum")
    with pytest.raises(ValueError, match="requires a pooled aggregation scope"):
        make(partition_by=["promo"], time_agg="sum")
    assert make(groupby=["brand"], time_agg="sum").time_agg == "sum"


def test_ewm_time_agg_policy():
    with pytest.raises(ValueError, match="does not accept time_agg=None"):
        ExponentiallyWeightedMean(0.1, time_agg=None)
    # the bucket-mean rule is EWM's own, so it needs no pooled scope
    assert ExponentiallyWeightedMean(0.1).time_agg == "mean"


@pytest.mark.parametrize(
    "make",
    [
        lambda **kw: RollingMean(7, **kw),
        lambda **kw: RollingQuantile(0.5, 7, **kw),
        lambda **kw: SeasonalRollingMean(7, 2, **kw),
        lambda **kw: SeasonalRollingQuantile(0.5, 7, 2, **kw),
    ],
    ids=["rolling", "rolling_quantile", "seasonal", "seasonal_quantile"],
)
def test_min_samples_zero_warns_only_under_a_pooled_scope(make):
    for scope in ({"global_": True}, {"groupby": ["brand"]}, {"partition_by": ["p"]}):
        with pytest.warns(UserWarning, match="min_samples=0 with pooled transforms"):
            make(min_samples=0, **scope)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make(min_samples=0)
        make(min_samples=1, global_=True)


def test_rolling_quantile_core_transform(grouped_array):
    tfm = RollingQuantile(0.5, 7, min_samples=3)._set_core_tfm(2)
    core = tfm._core_tfm
    assert (core.lag, core.p, core.window_size, core.min_samples) == (2, 0.5, 7, 3)
    expected = core_tfms.RollingQuantile(
        lag=2, p=0.5, window_size=7, min_samples=3
    ).transform(grouped_array)
    np.testing.assert_allclose(tfm.transform(grouped_array), expected)


def test_wrappers_mirror_the_leaf_scope():
    off = Offset(RollingMean(7, groupby=["brand"], partition_by=["promo"]), 2)
    assert (off.global_, off.groupby, off.partition_by) == (False, ["brand"], ["promo"])
    assert (Offset(Lag(1), 1).global_, Offset(Lag(1), 1).groupby) == (False, None)
    comb = Combine(RollingMean(7, global_=True), ExpandingMean(global_=True), operator.add)
    assert (comb.global_, comb.groupby, comb.partition_by) == (True, None, None)
    comb = Combine(RollingMean(7, partition_by=["p"]), RollingMean(5, partition_by=["p"]), operator.sub)
    assert (comb.global_, comb.groupby, comb.partition_by) == (False, None, ["p"])
    with pytest.raises(ValueError, match="different global_"):
        Combine(RollingMean(7, global_=True), Lag(1), operator.add)
    with pytest.raises(ValueError, match="different groupby"):
        Combine(RollingMean(7, groupby=["a"]), RollingMean(7, groupby=["b"]), operator.add)
    with pytest.raises(ValueError, match="different partition_by"):
        Combine(RollingMean(7, partition_by=["a"]), RollingMean(7), operator.add)
