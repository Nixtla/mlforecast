__all__ = ["RayLGBMForecast"]


from typing import Any, Dict, Tuple

import lightgbm as lgb

from ._base import RayForecastBase


def _lgb_train_loop(config: Dict[str, Any]) -> None:
    import ray.train
    from ray.train.lightgbm import (
        RayTrainReportCallback,
        normalize_pandas_for_lightgbm,
    )

    shard = ray.train.get_dataset_shard("train")
    # since ray 2.56 to_pandas yields pd.ArrowDtype columns, which lightgbm's
    # input validation rejects, so they're mapped back to numpy dtypes here.
    df = normalize_pandas_for_lightgbm(shard.materialize().to_pandas())
    label = df.pop(config["target_col"])
    train_set = lgb.Dataset(df, label=label)
    lgb.train(
        config["params"],
        train_set,
        num_boost_round=config["num_boost_round"],
        valid_sets=[train_set],
        valid_names=["train"],
        callbacks=[RayTrainReportCallback()],
    )


class RayLGBMForecast(RayForecastBase):
    _train_loop = staticmethod(_lgb_train_loop)

    @property
    def _trainer_cls(self):
        from ray.train.lightgbm import LightGBMTrainer

        return LightGBMTrainer

    def _translate_params(self) -> Tuple[Dict[str, Any], int]:
        params = dict(self.params)
        # lightgbm takes random_state as an alias of seed, so only the number of
        # rounds has to be lifted out of the params.
        num_boost_round = self._pop_num_boost_round(
            params, "n_estimators", "num_iterations", "num_boost_round"
        )
        params.setdefault("objective", "regression")
        return params, num_boost_round

    def _local_model(
        self, checkpoint: Any, params: Dict[str, Any]
    ) -> lgb.LGBMRegressor:
        from ray.train.lightgbm import RayTrainReportCallback

        booster = RayTrainReportCallback.get_model(checkpoint)
        model = lgb.LGBMRegressor(**params)
        # mirrors what lightgbm_ray's _lgb_ray_to_local did: build the local
        # estimator from the params and graft the trained booster onto it.
        model._Booster = booster
        model.fitted_ = True
        model._n_features = booster.num_feature()
        model._n_features_in = booster.num_feature()
        model._best_iteration = booster.best_iteration
        model._best_score = {}
        model._evals_result = {}
        model._objective = params["objective"]
        model._le = None
        model._class_map = None
        model._classes = None
        model._n_classes = -1
        return model
