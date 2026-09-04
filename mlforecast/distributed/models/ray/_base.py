__all__ = ["RayForecastBase"]


from typing import Any, Callable, Dict, Optional, Tuple

from sklearn.base import BaseEstimator


class RayForecastBase(BaseEstimator):
    """Base for the ray models, trained through ray.train's Trainer API.

    Subclasses provide ``_trainer_cls``, ``_train_loop`` (a module level function,
    so that it can be shipped to the workers), ``_translate_params`` and
    ``_local_model``.

    The ray specific arguments are keyword only so that they can't collide with
    the booster's parameters, which are taken as ``**params``.
    """

    _trainer_cls: Any
    _train_loop: Callable[[Dict[str, Any]], None]

    def __init__(
        self,
        *,
        num_workers: int = 1,
        resources_per_worker: Optional[Dict[str, float]] = None,
        **params: Any,
    ):
        self.num_workers = num_workers
        self.resources_per_worker = resources_per_worker
        self.params = params

    # sklearn's introspection ignores **kwargs, which would make `clone` drop
    # every booster parameter, so the params are handled explicitly here.
    def get_params(self, deep: bool = True) -> Dict[str, Any]:  # noqa: ARG002 - sklearn's signature
        return {
            "num_workers": self.num_workers,
            "resources_per_worker": self.resources_per_worker,
            **self.params,
        }

    def set_params(self, **params: Any) -> "RayForecastBase":
        for name in ("num_workers", "resources_per_worker"):
            if name in params:
                setattr(self, name, params.pop(name))
        self.params.update(params)
        return self

    def _translate_params(self) -> Tuple[Dict[str, Any], int]:
        """Return the booster params and the number of boosting rounds."""
        raise NotImplementedError

    def _local_model(self, checkpoint: Any, params: Dict[str, Any]) -> Any:
        """Rebuild a local (non-distributed) model from the training checkpoint."""
        raise NotImplementedError

    def fit(self, dataset: Any, target_col: str) -> "RayForecastBase":
        from ray.train import ScalingConfig

        params, num_boost_round = self._translate_params()
        trainer = self._trainer_cls(
            self._train_loop,
            train_loop_config={
                "params": params,
                "num_boost_round": num_boost_round,
                "target_col": target_col,
            },
            scaling_config=ScalingConfig(
                num_workers=self.num_workers,
                resources_per_worker=self.resources_per_worker,
            ),
            datasets={"train": dataset},
        )
        result = trainer.fit()
        self.model_ = self._local_model(result.checkpoint, params)
        return self

    def _pop_num_boost_round(self, params: Dict[str, Any], *aliases: str) -> int:
        for alias in aliases:
            if alias in params:
                num_boost_round = params.pop(alias)
                # drop any remaining aliases so they don't reach the booster
                for other in aliases:
                    params.pop(other, None)
                return int(num_boost_round)
        return 100
