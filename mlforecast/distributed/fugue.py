# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/distributed.fugue.ipynb.

# %% auto 0
__all__ = ['FugueMLForecast']

# %% ../../nbs/distributed.fugue.ipynb 2
import copy
from typing import Any, Callable, Iterable, List, Optional

import cloudpickle

try:
    import dask.dataframe as dd

    DASK_INSTALLED = True
except ModuleNotFoundError:
    DASK_INSTALLED = False
import fugue.api as fa
import pandas as pd

try:
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import DataFrame as SparkDataFrame

    SPARK_INSTALLED = True
except ModuleNotFoundError:
    SPARK_INSTALLED = False
from sklearn.base import clone

from mlforecast.core import (
    DateFeature,
    Differences,
    Freq,
    LagTransforms,
    Lags,
    TimeSeries,
    _name_models,
)

# %% ../../nbs/distributed.fugue.ipynb 3
class FugueMLForecast:
    def __init__(
        self,
        models,
        freq: Optional[Freq] = None,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        differences: Optional[Differences] = None,
        num_threads: int = 1,
        engine=None,
    ):
        if not isinstance(models, dict) and not isinstance(models, list):
            models = [models]
        if isinstance(models, list):
            model_names = _name_models([m.__class__.__name__ for m in models])
            models_with_names = dict(zip(model_names, models))
        else:
            models_with_names = models
        self.models = models_with_names
        self._base_ts = TimeSeries(
            freq, lags, lag_transforms, date_features, differences, num_threads
        )
        self.engine = engine

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(models=[{", ".join(self.models.keys())}], '
            f"freq={self._base_ts.freq}, "
            f"lag_features={list(self._base_ts.transforms.keys())}, "
            f"date_features={self._base_ts.date_features}, "
            f"num_threads={self._base_ts.num_threads}, "
            f"engine={self.engine})"
        )

    @staticmethod
    def _preprocess_partition(
        part: pd.DataFrame,
        base_ts: TimeSeries,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> List[List[Any]]:
        ts = copy.deepcopy(base_ts)
        transformed = ts.fit_transform(
            part,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
        )
        return [[cloudpickle.dumps(ts), cloudpickle.dumps(transformed)]]

    @staticmethod
    def _retrieve_df(items: List[List[Any]]) -> Iterable[pd.DataFrame]:
        for _, serialized_df in items:
            yield cloudpickle.loads(serialized_df)

    def preprocess(
        self,
        data,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ):
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.partition_results = fa.transform(
            data,
            FugueMLForecast._preprocess_partition,
            params={
                "base_ts": self._base_ts,
                "id_col": id_col,
                "time_col": time_col,
                "target_col": target_col,
                "static_features": static_features,
                "dropna": dropna,
                "keep_last_n": keep_last_n,
            },
            schema="ts:binary,df:binary",
            engine=self.engine,
            as_fugue=True,
        )
        base_schema = fa.get_schema(data[[id_col, time_col, target_col]])
        features_dtypes = [f"{feat}:double" for feat in self._base_ts.features]
        schema = str(base_schema) + "," + ",".join(features_dtypes)
        res = fa.transform(
            self.partition_results,
            FugueMLForecast._retrieve_df,
            schema=schema,
            engine=self.engine,
        )
        return fa.get_native_as_df(res)

    def fit(
        self,
        data,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ):
        prep = self.preprocess(
            data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
        )
        features = [x for x in prep.columns if x not in {id_col, time_col, target_col}]
        self.models_ = {}
        if SPARK_INSTALLED and isinstance(data, SparkDataFrame):
            try:
                import lightgbm as lgb
                from synapse.ml.lightgbm import (
                    LightGBMRegressor as SynapseLGBMRegressor,
                )

                LGBM_INSTALLED = True
            except ModuleNotFoundError:
                LGBM_INSTALLED = False
            try:
                import xgboost as xgb
                from xgboost.spark import SparkXGBRegressor  # type: ignore

                XGB_INSTALLED = True
            except ModuleNotFoundError:
                XGB_INSTALLED = False

            featurizer = VectorAssembler(inputCols=features, outputCol="features")
            train_data = featurizer.transform(prep)[target_col, "features"]
            for name, model in self.models.items():
                if LGBM_INSTALLED and isinstance(model, SynapseLGBMRegressor):
                    trained_model = model.setLabelCol(target_col).fit(train_data)
                    model_str = trained_model.getNativeModel()
                    local_model = lgb.Booster(model_str=model_str)
                elif XGB_INSTALLED and isinstance(model, SparkXGBRegressor):
                    model.setParams(label_col=target_col)
                    trained_model = model.fit(train_data)
                    model_str = trained_model.get_booster().save_raw("ubj")
                    local_model = xgb.XGBRegressor()
                    local_model.load_model(model_str)
                else:
                    raise ValueError(
                        "Only LightGBMRegressor from SynapseML and SparkXGBRegressor are supported in spark."
                    )
                self.models_[name] = local_model
        elif DASK_INSTALLED and isinstance(data, dd.DataFrame):
            try:
                from mlforecast.distributed.models.lgb import LGBMForecast

                LGBM_INSTALLED = True
            except ModuleNotFoundError:
                LGBM_INSTALLED = False
            try:
                from mlforecast.distributed.models.xgb import XGBForecast

                XGB_INSTALLED = True
            except ModuleNotFoundError:
                XGB_INSTALLED = False
            X, y = prep[features], prep[target_col]
            for name, model in self.models.items():
                if not (LGBM_INSTALLED and isinstance(model, LGBMForecast)) or (
                    XGB_INSTALLED and isinstance(model, XGBForecast)
                ):
                    raise ValueError(
                        "Models must be either LGBMForecast or XGBForecast with dask backend."
                    )
                self.models_[name] = clone(model).fit(X, y).model_
        else:
            raise NotImplementedError("Only spark and dask engines are supported.")
        return self

    @staticmethod
    def _predict(
        items: List[List[Any]],
        models,
        horizon,
        dynamic_dfs,
        before_predict_callback,
        after_predict_callback,
    ) -> Iterable[pd.DataFrame]:
        for serialized_ts, _ in items:
            ts = cloudpickle.loads(serialized_ts)
            res = ts.predict(
                models=models,
                horizon=horizon,
                dynamic_dfs=dynamic_dfs,
                before_predict_callback=before_predict_callback,
                after_predict_callback=after_predict_callback,
            )
            yield res.reset_index()

    def predict(
        self,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ):
        model_names = self.models.keys()
        models_schema = ",".join(f"{model_name}:double" for model_name in model_names)
        schema = f"{self.id_col}:string,{self.time_col}:datetime," + models_schema
        return fa.transform(
            self.partition_results,
            FugueMLForecast._predict,
            params={
                "models": self.models_,
                "horizon": horizon,
                "dynamic_dfs": dynamic_dfs,
                "before_predict_callback": before_predict_callback,
                "after_predict_callback": after_predict_callback,
            },
            schema=schema,
            engine=self.engine,
        )
