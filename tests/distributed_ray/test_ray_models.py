import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import ray
from sklearn.base import clone

from mlforecast.distributed.models.ray.lgb import RayLGBMForecast
from mlforecast.distributed.models.ray.xgb import RayXGBForecast


@pytest.mark.ray
@pytest.mark.parametrize("model_cls", [RayLGBMForecast, RayXGBForecast])
def test_clone_preserves_booster_params(model_cls):
    """sklearn's introspection ignores **kwargs, so get_params is overridden.

    Without that override `clone` in DistributedMLForecast._fit would silently
    drop every parameter the user passed.
    """
    model = model_cls(num_workers=2, random_state=0, learning_rate=0.05)
    params = model.get_params()
    assert params["random_state"] == 0
    assert params["learning_rate"] == 0.05
    assert params["num_workers"] == 2

    cloned = clone(model)
    assert cloned.params == {"random_state": 0, "learning_rate": 0.05}
    assert cloned.num_workers == 2


@pytest.mark.ray
def test_lgb_param_translation():
    params, num_boost_round = RayLGBMForecast(
        n_estimators=7, verbosity=-1, random_state=0
    )._translate_params()
    assert num_boost_round == 7
    # lightgbm takes random_state as an alias of seed, so it's passed through
    assert params == {"verbosity": -1, "random_state": 0, "objective": "regression"}


@pytest.mark.ray
def test_xgb_param_translation():
    params, num_boost_round = RayXGBForecast(
        n_estimators=7, random_state=0, max_depth=3
    )._translate_params()
    assert num_boost_round == 7
    # the native API doesn't know random_state
    assert params == {
        "seed": 0,
        "max_depth": 3,
        "objective": "reg:squarederror",
    }


@pytest.mark.ray
def test_default_num_boost_round_matches_sklearn():
    """The estimators defaulted to n_estimators=100; keep that."""
    for model_cls in (RayLGBMForecast, RayXGBForecast):
        _, num_boost_round = model_cls()._translate_params()
        assert num_boost_round == 100


@pytest.mark.ray
@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Distributed tests are not supported on Python < 3.10",
)
def test_local_model_uses_the_trained_booster():
    """model_ grafts the distributed booster onto a local estimator.

    This is the fragile part of the lightgbm path (it copies private fitted
    state, as lightgbm_ray did), so pin the estimator's predictions to the
    booster's.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "lag1": rng.normal(size=200),
            "lag2": rng.normal(size=200),
            "y": rng.normal(size=200),
        }
    )
    dataset = ray.data.from_pandas(df)

    model = RayLGBMForecast(verbosity=-1, random_state=0, n_estimators=5)
    model.fit(dataset, target_col="y")

    assert isinstance(model.model_, lgb.LGBMRegressor)
    X = df[["lag1", "lag2"]].head(10)
    np.testing.assert_allclose(
        model.model_.predict(X), model.model_.booster_.predict(X)
    )
    assert model.model_.booster_.num_trees() == 5


@pytest.mark.ray
def test_reclaim_placement_groups_frees_a_leaked_group():
    """The conftest safety net that keeps a leak from hanging the next test."""
    from ray.util.placement_group import placement_group, placement_group_table

    from .conftest import _reclaim_placement_groups

    pg = placement_group([{"CPU": 1}])
    ray.get(pg.ready())
    assert any(info["state"] == "CREATED" for info in placement_group_table().values())

    _reclaim_placement_groups()

    assert all(info["state"] == "REMOVED" for info in placement_group_table().values())
