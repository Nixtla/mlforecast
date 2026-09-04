__all__ = ["RayXGBForecast"]


from typing import Any, Dict, Tuple

import xgboost as xgb

from ._base import RayForecastBase


def _xgb_train_loop(config: Dict[str, Any]) -> None:
    import ray.train
    from ray.train.xgboost import RayTrainReportCallback

    shard = ray.train.get_dataset_shard("train")
    # unlike lightgbm, xgboost accepts arrow backed pandas columns.
    df = shard.materialize().to_pandas()
    label = df.pop(config["target_col"])
    dtrain = xgb.DMatrix(df, label=label)
    xgb.train(
        config["params"],
        dtrain,
        num_boost_round=config["num_boost_round"],
        evals=[(dtrain, "train")],
        callbacks=[RayTrainReportCallback()],
    )


class RayXGBForecast(RayForecastBase):
    _train_loop = staticmethod(_xgb_train_loop)

    @property
    def _trainer_cls(self):
        from ray.train.xgboost import XGBoostTrainer

        return XGBoostTrainer

    def _translate_params(self) -> Tuple[Dict[str, Any], int]:
        params = dict(self.params)
        num_boost_round = self._pop_num_boost_round(
            params, "n_estimators", "num_boost_round"
        )
        # the native API doesn't know about the sklearn parameter names
        if "random_state" in params:
            params["seed"] = params.pop("random_state")
        params.setdefault("objective", "reg:squarederror")
        return params, num_boost_round

    def _local_model(
        self,
        checkpoint: Any,
        params: Dict[str, Any],  # noqa: ARG002 - the base class' signature
    ) -> xgb.XGBRegressor:
        from ray.train.xgboost import RayTrainReportCallback

        booster = RayTrainReportCallback.get_model(checkpoint)
        # load_model restores the params from the booster itself, so unlike
        # lightgbm there's nothing to graft on.
        model = xgb.XGBRegressor()
        model.load_model(booster.save_raw("ubj"))
        return model
