__all__ = ["RayForecastBase"]


import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_RAY_PARAMS = ("num_workers", "resources_per_worker")
_MODEL_FILE = "model.pkl"


def worker_n_jobs(requested: Any) -> int:
    """Threads for the booster, from the CPUs this train worker was actually given.

    Both ``lightgbm_ray`` and ``xgboost_ray`` sized the thread pool from the
    actor's CPU share; without it N workers landing on one node each spawn
    threads for every core on the box. An explicit, smaller value is honoured.
    """
    import ray

    assigned = ray.get_runtime_context().get_assigned_resources().get("CPU", 1)
    assigned = max(1, int(assigned))
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        return assigned
    return assigned if requested <= 0 else min(requested, assigned)


class KeepLastMetrics:
    """Remember the last iteration's metrics so the final report can carry them."""

    last_metrics: Dict[str, Any] = {}

    def _report_metrics(self, report_dict: Dict[str, Any]) -> None:
        self.last_metrics = report_dict
        super()._report_metrics(report_dict)  # type: ignore[misc]


def report_fitted_model(
    model: Any, booster: Any, booster_file: str, metrics: Dict[str, Any]
) -> None:
    """Report ray's standard booster artifact along with the fitted estimator.

    The estimator is what becomes ``model_``, which is why it's checkpointed;
    the booster is kept next to it so that ``RayTrainReportCallback.get_model``
    still works on the result.

    ``ray.train.report`` is collective, so every worker has to call it;
    reporting from rank 0 only deadlocks.
    """
    import ray.train
    from ray.train import Checkpoint

    if ray.train.get_context().get_world_rank() != 0:
        ray.train.report(metrics)
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        booster.save_model(Path(tmp_dir, booster_file).as_posix())
        with open(Path(tmp_dir, _MODEL_FILE), "wb") as f:
            pickle.dump(model, f)
        ray.train.report(metrics, checkpoint=Checkpoint.from_directory(tmp_dir))


class RayForecastBase:
    """Mixin holding the ray.train plumbing; subclasses are real sklearn estimators.

    The training loop builds and fits the library's own estimator in the worker,
    as ``lightgbm.dask._train_part`` does, and checkpoints it. That keeps all of
    the parameter handling in the library instead of here.

    The ray specific arguments are keyword only so that they can't collide with
    the booster's parameters, which are taken as ``**kwargs``.
    """

    num_workers: int
    resources_per_worker: Optional[Dict[str, float]]

    def __init__(
        self,
        *,
        num_workers: int = 1,
        resources_per_worker: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ):
        # cooperative: goes on to LGBMRegressor / XGBRegressor
        super().__init__(**kwargs)
        self.num_workers = num_workers
        self.resources_per_worker = resources_per_worker

    def _train(
        self,
        trainer_cls: Any,
        train_loop: Callable[[Dict[str, Any]], None],
        dataset: Any,
        target_col: str,
    ) -> "RayForecastBase":
        from ray.train import ScalingConfig

        params = self.get_params()  # type: ignore[attr-defined]
        for name in _RAY_PARAMS:
            params.pop(name, None)
        trainer = trainer_cls(
            train_loop,
            train_loop_config={"params": params, "target_col": target_col},
            scaling_config=ScalingConfig(
                num_workers=self.num_workers,
                resources_per_worker=self.resources_per_worker,
            ),
            datasets={"train": dataset},
        )
        with trainer.fit().checkpoint.as_directory() as ckpt_dir:
            with open(Path(ckpt_dir, _MODEL_FILE), "rb") as f:
                self.model_ = pickle.load(f)
        return self
