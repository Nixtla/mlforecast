__all__ = ["RayXGBForecast"]


from typing import Any, Dict

import xgboost as xgb

from ._base import RayForecastBase, report_fitted_model, worker_n_jobs


def _xgb_train_loop(config: Dict[str, Any]) -> None:
    import ray.train
    from ray.train.xgboost import RayTrainReportCallback

    shard = ray.train.get_dataset_shard("train")
    # unlike lightgbm, xgboost accepts arrow backed pandas columns.
    df = shard.materialize().to_pandas()
    label = df.pop(config["target_col"])
    params = {
        **config["params"],
        "n_jobs": worker_n_jobs(config["params"].get("n_jobs")),
    }
    # XGBoostConfig wraps the loop in a CommunicatorContext, so unlike lightgbm
    # there are no network params to pass: training is distributed already.
    model = xgb.XGBRegressor(**params)
    model.fit(df, label, eval_set=[(df, label)])
    # the n_jobs clamp is for this worker's thread pool; model_ is shipped to the
    # forecasting workers and returned by to_local, so it keeps what was asked for
    # set_params would push every param into the booster, and a list eval_metric
    # breaks save_model once it's been set that way
    n_jobs = config["params"].get("n_jobs")
    model.n_jobs = n_jobs
    if n_jobs is not None:
        model.get_booster().set_param("nthread", n_jobs)
    report_fitted_model(
        model, model.get_booster(), RayTrainReportCallback.CHECKPOINT_NAME
    )


class RayXGBForecast(RayForecastBase, xgb.XGBRegressor):
    """XGBoost forecaster trained with `ray.train.xgboost.XGBoostTrainer`.

    The booster's parameters are taken as ``**kwargs`` and handled by
    ``XGBRegressor`` itself; the ray arguments are keyword only
    so that they can't collide with them.

    ``num_workers`` sets the number of ray train workers. The previous
    ``xgboost_ray`` based implementation derived that from ``n_jobs``
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
    exposed as ``model_``, a local ``xgboost.XGBRegressor`` that is sent to the
    workers in the forecasting step.
    """

    def fit(self, dataset: Any, target_col: str) -> "RayXGBForecast":  # type: ignore[override]
        from ray.train.xgboost import XGBoostTrainer

        return self._train(XGBoostTrainer, _xgb_train_loop, dataset, target_col)  # type: ignore[return-value]
