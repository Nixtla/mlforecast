__all__ = ["RayLGBMForecast"]


import warnings
from typing import Any, Dict

import lightgbm as lgb
from lightgbm.basic import _choose_param_value, _ConfigAliases

from ._base import _RAY_PARAMS, RayForecastBase, report_fitted_model, worker_n_jobs

# feature parallel needs the full data on every worker and serial trains on the shard
_TREE_LEARNERS = {"data", "data_parallel", "voting", "voting_parallel"}
_NETWORK_KEYS = _ConfigAliases.get("num_machines", "machines", "local_listen_port")
_THREAD_KEYS = _ConfigAliases.get("num_threads") - {"n_jobs"}


def _lgb_train_loop(config: Dict[str, Any]) -> None:
    import ray.train
    from ray.train.lightgbm import (
        RayTrainReportCallback,
        get_network_params,
        normalize_pandas_for_lightgbm,
    )

    shard = ray.train.get_dataset_shard("train")
    # since ray 2.56 to_pandas yields pd.ArrowDtype columns, which lightgbm's
    # input validation rejects, so they're mapped back to numpy dtypes here.
    df = normalize_pandas_for_lightgbm(shard.materialize().to_pandas())
    label = df.pop(config["target_col"])
    n_jobs = config["params"].get("n_jobs")
    params = {**config["params"], "n_jobs": worker_n_jobs(n_jobs)}
    params = _choose_param_value("tree_learner", params, "data_parallel")
    if params["tree_learner"] not in _TREE_LEARNERS:
        warnings.warn(
            f"Parameter tree_learner set to {params['tree_learner']}, which is not "
            'allowed. Using "data_parallel" as default'
        )
        params["tree_learner"] = "data_parallel"
    for alias in _NETWORK_KEYS | _THREAD_KEYS:
        if alias in params:
            warnings.warn(f"Parameter {alias} will be ignored.")
            params.pop(alias)
    # each worker only sees its own shard. ray's LightGBMConfig stashes the
    # network params in a per worker global rather than injecting them, so
    # without these every worker trains an independent model on 1/N of the data
    # and rank 0's is the one that gets checkpointed. Plain kwargs, as in
    # lightgbm.dask's _train_part.
    model = lgb.LGBMRegressor(**params, **get_network_params())
    model.fit(df, label, eval_set=[(df, label)], eval_names=["train"])
    # a local refit with them would wait for the training workers
    for key in _NETWORK_KEYS:
        model._other_params.pop(key, None)
    # the clamp is for this worker's thread pool; model_ is shipped to the
    # forecasting workers and returned by to_local, so it keeps what was asked for
    model.set_params(n_jobs=n_jobs)
    report_fitted_model(model, model.booster_, RayTrainReportCallback.CHECKPOINT_NAME)


class RayLGBMForecast(RayForecastBase, lgb.LGBMRegressor):
    """LightGBM forecaster trained with `ray.train.lightgbm.LightGBMTrainer`.

    The booster's parameters are taken as ``**kwargs`` and handled by
    ``LGBMRegressor`` itself; the ray arguments are keyword only
    so that they can't collide with them.

    ``num_workers`` sets the number of ray train workers. The previous
    ``lightgbm_ray`` based implementation derived that from ``n_jobs``
    (``RayParams(num_actors=n_jobs)``); ``n_jobs`` is now the per worker thread
    count, as it is for the local estimator.

    ``resources_per_worker`` is the CPU knob: it decides how many CPUs each
    worker is given and therefore how many threads the booster can use. It
    defaults to the cluster's CPUs split evenly across the workers and bounded by
    the smallest node, as ``xgboost_ray._autodetect_resources`` did, and ``n_jobs``
    can only lower it below that share. ``model_`` keeps the requested ``n_jobs``
    rather than the clamp.

    ``storage_path`` is where ray train writes the run. It defaults to a
    temporary directory that is discarded once the fitted model has been read
    back, so that ``~/ray_results`` doesn't grow by a run per model per fit; as
    with ray's own default, a local path only works on a single node, so point
    it at shared storage for a multi node cluster.

    ``fit`` takes a ray ``Dataset`` and a target column rather than the sklearn
    ``(X, y)`` pair, as the previous implementation did, so the inherited
    ``predict``/``score`` can't be used before fitting. The fitted estimator is
    exposed as ``model_``, a local ``lightgbm.LGBMRegressor`` that is sent to the
    workers in the forecasting step.
    """

    @classmethod
    def _get_param_names(cls):
        # sklearn reads the subclass' signature, which doesn't name the booster params
        return sorted([*lgb.LGBMRegressor._get_param_names(), *_RAY_PARAMS])

    def fit(self, dataset: Any, target_col: str) -> "RayLGBMForecast":  # type: ignore[override]
        from ray.train.lightgbm import LightGBMTrainer

        return self._train(LightGBMTrainer, _lgb_train_loop, dataset, target_col)  # type: ignore[return-value]
