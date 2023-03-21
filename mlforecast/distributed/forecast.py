# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/distributed.forecast.ipynb.

# %% auto 0
__all__ = ['DistributedMLForecast']

# %% ../../nbs/distributed.forecast.ipynb 5
import copy
from collections import namedtuple
from typing import Any, Callable, Iterable, List, Optional

import cloudpickle

try:
    import dask.dataframe as dd

    DASK_INSTALLED = True
except ModuleNotFoundError:
    DASK_INSTALLED = False
import fugue
import fugue.api as fa
import pandas as pd

try:
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import DataFrame as SparkDataFrame

    SPARK_INSTALLED = True
except ModuleNotFoundError:
    SPARK_INSTALLED = False
try:
    from ray.air.checkpoint import Checkpoint as RayCheckpoint
    from ray.data import Dataset as RayDataset
    from ray.train.sklearn import SklearnTrainer, SklearnPredictor

    RAY_INSTALLED = True
except ModuleNotFoundError:
    RAY_INSTALLED = False
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

# %% ../../nbs/distributed.forecast.ipynb 6
WindowInfo = namedtuple(
    "WindowInfo", ["n_windows", "window_size", "step_size", "i_window"]
)

# %% ../../nbs/distributed.forecast.ipynb 7
class DistributedMLForecast:
    """Multi backend distributed pipeline"""

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
        """Create distributed forecast object

        Parameters
        ----------
        models : regressor or list of regressors
            Models that will be trained and used to compute the forecasts.
        freq : str or int, optional (default=None)
            Pandas offset alias, e.g. 'D', 'W-THU' or integer denoting the frequency of the series.
        lags : list of int, optional (default=None)
            Lags of the target to use as features.
        lag_transforms : dict of int to list of functions, optional (default=None)
            Mapping of target lags to their transformations.
        date_features : list of str or callable, optional (default=None)
            Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input.
        differences : list of int, optional (default=None)
            Differences to take of the target before computing the features. These are restored at the forecasting step.
        num_threads : int (default=1)
            Number of threads to use when computing the features.
        engine : fugue execution engine, optional (default=None)
            Dask Client, Spark Session, etc to use for the distributed computation.
            If None will infer depending on the input type.
        """
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
        window_info: Optional[WindowInfo] = None,
        fit_ts_only: bool = False,
    ) -> List[List[Any]]:
        ts = copy.deepcopy(base_ts)
        if fit_ts_only:
            ts._fit(
                part,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static_features=static_features,
                keep_last_n=keep_last_n,
            )
            return [
                [
                    cloudpickle.dumps(ts),
                    cloudpickle.dumps(None),
                    cloudpickle.dumps(None),
                ]
            ]
        if window_info is None:
            train = part
            valid = None
        else:
            n_windows, window_size, step_size, i_window = window_info
            if step_size is None:
                step_size = window_size
            test_size = window_size + step_size * (n_windows - 1)
            offset = test_size - i_window * step_size
            max_dates = part.groupby(id_col)[time_col].transform("max")
            train_ends = max_dates - offset * base_ts.freq
            valid_ends = train_ends + window_size * base_ts.freq
            train_mask = part[time_col].le(train_ends)
            valid_mask = part[time_col].gt(train_ends) & part[time_col].le(valid_ends)
            train = part[train_mask]
            valid_keep_cols = part.columns
            if static_features is not None:
                valid_keep_cols.drop(static_features)
            cutoffs = (
                part.groupby(id_col)[time_col].max().rename("cutoff")
                - offset * base_ts.freq
            )
            valid = part.loc[valid_mask, valid_keep_cols].merge(cutoffs, on=id_col)
        transformed = ts.fit_transform(
            train,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
        )
        return [
            [
                cloudpickle.dumps(ts),
                cloudpickle.dumps(transformed),
                cloudpickle.dumps(valid),
            ]
        ]

    @staticmethod
    def _retrieve_df(items: List[List[Any]]) -> Iterable[pd.DataFrame]:
        for _, serialized_train, _ in items:
            yield cloudpickle.loads(serialized_train)

    def _preprocess_partitions(
        self,
        data: fugue.AnyDataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        window_info: Optional[WindowInfo] = None,
        fit_ts_only: bool = False,
    ) -> List[Any]:
        return fa.transform(
            data,
            DistributedMLForecast._preprocess_partition,
            params={
                "base_ts": self._base_ts,
                "id_col": id_col,
                "time_col": time_col,
                "target_col": target_col,
                "static_features": static_features,
                "dropna": dropna,
                "keep_last_n": keep_last_n,
                "window_info": window_info,
                "fit_ts_only": fit_ts_only,
            },
            schema="ts:binary,train:binary,valid:binary",
            engine=self.engine,
            as_fugue=True,
            partition=id_col,
        )

    def _preprocess(
        self,
        data: fugue.AnyDataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        window_info: Optional[WindowInfo] = None,
    ) -> fugue.AnyDataFrame:
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.static_features = static_features
        self.dropna = dropna
        self.keep_last_n = keep_last_n
        self.partition_results = self._preprocess_partitions(
            data=data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            window_info=window_info,
        )
        base_schema = str(fa.get_schema(data))
        features_schema = ",".join(f"{feat}:double" for feat in self._base_ts.features)
        res = fa.transform(
            self.partition_results,
            DistributedMLForecast._retrieve_df,
            schema=f"{base_schema},{features_schema}",
            engine=self.engine,
        )
        return fa.get_native_as_df(res)

    def preprocess(
        self,
        data: fugue.AnyDataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> fugue.AnyDataFrame:
        """Add the features to `data`.

        Parameters
        ----------
        data : dask or spark DataFrame.
            Series data in long format.
        id_col : str
            Column that identifies each serie. If 'index' then the index is used.
        time_col : str
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str
            Column that contains the target.
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.

        Returns
        -------
        result : same type as input
            data with added features.
        """
        return self._preprocess(
            data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
        )

    def _fit(
        self,
        data: fugue.AnyDataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        window_info: Optional[WindowInfo] = None,
    ) -> "DistributedMLForecast":
        prep = self._preprocess(
            data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            window_info=window_info,
        )
        features = [
            x
            for x in fa.get_column_names(prep)
            if x not in {id_col, time_col, target_col}
        ]
        self.models_ = {}
        if SPARK_INSTALLED and isinstance(data, SparkDataFrame):
            featurizer = VectorAssembler(inputCols=features, outputCol="features")
            train_data = featurizer.transform(prep)[target_col, "features"]
            for name, model in self.models.items():
                trained_model = model._pre_fit(target_col).fit(train_data)
                self.models_[name] = model.extract_local_model(trained_model)
        elif DASK_INSTALLED and isinstance(data, dd.DataFrame):
            X, y = prep[features], prep[target_col]
            for name, model in self.models.items():
                trained_model = clone(model).fit(X, y)
                self.models_[name] = trained_model.model_
        elif RAY_INSTALLED and isinstance(data, RayDataset):
            for name, model in self.models.items():
                trainer = SklearnTrainer(
                    estimator=clone(model),
                    label_column=target_col,
                    datasets={
                        "train": prep.select_columns(cols=features + [target_col])
                    },
                )
                trained_model = trainer.fit()
                self.models_[name] = trained_model.checkpoint
        else:
            raise NotImplementedError(
                "Only spark, dask, and ray engines are supported."
            )
        return self

    def fit(
        self,
        data: fugue.AnyDataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> "DistributedMLForecast":
        """Apply the feature engineering and train the models.

        Parameters
        ----------
        data : dask or spark DataFrame
            Series data in long format.
        id_col : str
            Column that identifies each serie. If 'index' then the index is used.
        time_col : str
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str
            Column that contains the target.
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.

        Returns
        -------
        self : DistributedMLForecast
            Forecast object with series values and trained models.
        """
        return self._fit(
            data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
        )

    @staticmethod
    def _predict(
        items: List[List[Any]],
        models,
        horizon,
        dynamic_dfs=None,
        before_predict_callback=None,
        after_predict_callback=None,
    ) -> Iterable[pd.DataFrame]:
        for serialized_ts, _, serialized_valid in items:
            valid = cloudpickle.loads(serialized_valid)
            ts = cloudpickle.loads(serialized_ts)
            if valid is not None:
                dynamic_features = valid.columns.drop(
                    [ts.id_col, ts.time_col, ts.target_col]
                )
                if not dynamic_features.empty:
                    dynamic_dfs = [valid.drop(columns=ts.target_col)]
            res = ts.predict(
                models={
                    name: SklearnPredictor.from_checkpoint(model)
                    if isinstance(model, RayCheckpoint)
                    else model
                    for name, model in models.items()
                },
                horizon=horizon,
                dynamic_dfs=dynamic_dfs,
                before_predict_callback=before_predict_callback,
                after_predict_callback=after_predict_callback,
            ).reset_index()
            if valid is not None:
                res = res.merge(valid, how="left")
            yield res

    def _get_predict_schema(self) -> str:
        model_names = self.models.keys()
        models_schema = ",".join(f"{model_name}:double" for model_name in model_names)
        schema = f"{self.id_col}:string,{self.time_col}:datetime," + models_schema
        return schema

    def predict(
        self,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        new_data: Optional[fugue.AnyDataFrame] = None,
    ) -> fugue.AnyDataFrame:
        """Compute the predictions for the next `horizon` steps.

        Parameters
        ----------
        horizon : int
            Number of periods to predict.
        dynamic_dfs : list of pandas DataFrame, optional (default=None)
            Future values of the dynamic features, e.g. prices.
        before_predict_callback : callable, optional (default=None)
            Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index.
        after_predict_callback : callable, optional (default=None)
            Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index.
        new_data : dask or spark DataFrame, optional (default=None)
            Series data of new observations for which forecasts are to be generated.
                This dataframe should have the same structure as the one used to fit the model, including any features and time series data.
                If `new_data` is not None, the method will generate forecasts for the new observations.

        Returns
        -------
        result : dask, spark or ray DataFrame
            Predictions for each serie and timestep, with one column per model.
        """
        if new_data is not None:
            partition_results = self._preprocess_partitions(
                data=new_data,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
                static_features=self.static_features,
                dropna=self.dropna,
                keep_last_n=self.keep_last_n,
                fit_ts_only=True,
            )
        else:
            partition_results = self.partition_results
        schema = self._get_predict_schema()
        res = fa.transform(
            partition_results,
            DistributedMLForecast._predict,
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
        return fa.get_native_as_df(res)

    def cross_validation(
        self,
        data: fugue.AnyDataFrame,
        n_windows: int,
        window_size: int,
        id_col: str,
        time_col: str,
        target_col: str,
        step_size: Optional[int] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        refit: bool = True,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ) -> fugue.AnyDataFrame:
        """Perform time series cross validation.
        Creates `n_windows` splits where each window has `window_size` test periods,
        trains the models, computes the predictions and merges the actuals.

        Parameters
        ----------
        data : dask, spark or ray DataFrame
            Series data in long format.
        n_windows : int
            Number of windows to evaluate.
        window_size : int
            Number of test periods in each window.
        id_col : str
            Column that identifies each serie. If 'index' then the index is used.
        time_col : str
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str
            Column that contains the target.
        step_size : int, optional (default=None)
            Step size between each cross validation window. If None it will be equal to `window_size`.
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
        refit : bool (default=True)
            Retrain model for each cross validation window.
            If False, the models are trained at the beginning and then used to predict each window.
        before_predict_callback : callable, optional (default=None)
            Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index.
        after_predict_callback : callable, optional (default=None)
            Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index.

        Returns
        -------
        result : dask, spark or ray DataFrame
            Predictions for each window with the series id, timestamp, target value and predictions from each model.
        """
        self.cv_models_ = []
        results = []
        for i in range(n_windows):
            window_info = WindowInfo(n_windows, window_size, step_size, i)
            if refit or i == 0:
                self._fit(
                    data,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    static_features=static_features,
                    dropna=dropna,
                    keep_last_n=keep_last_n,
                    window_info=window_info,
                )
                self.cv_models_.append(self.models_)
                partition_results = self.partition_results
            elif not refit:
                partition_results = self._preprocess_partitions(
                    data=data,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    static_features=static_features,
                    dropna=dropna,
                    keep_last_n=keep_last_n,
                    window_info=window_info,
                )
            schema = (
                self._get_predict_schema()
                + f",cutoff:datetime,{self.target_col}:double"
            )
            preds = fa.transform(
                partition_results,
                DistributedMLForecast._predict,
                params={
                    "models": self.models_,
                    "horizon": window_size,
                    "before_predict_callback": before_predict_callback,
                    "after_predict_callback": after_predict_callback,
                },
                schema=schema,
                engine=self.engine,
            )
            results.append(fa.get_native_as_df(preds))
        if len(results) == 1:
            return results[0]
        if len(results) == 2:
            return fa.union(results[0], results[1])
        return fa.union(results[0], results[1], results[2:])
