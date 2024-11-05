# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/distributed.forecast.ipynb.

# %% auto 0
__all__ = ['DistributedMLForecast']

# %% ../../nbs/distributed.forecast.ipynb 5
import copy
from collections import namedtuple
from typing import Any, Callable, Iterable, List, Optional

import cloudpickle
import fsspec

try:
    import dask.dataframe as dd

    DASK_INSTALLED = True
except ModuleNotFoundError:
    DASK_INSTALLED = False
import fugue
import fugue.api as fa
import numpy as np
import pandas as pd
import utilsforecast.processing as ufp

try:
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import DataFrame as SparkDataFrame

    SPARK_INSTALLED = True
except ModuleNotFoundError:
    SPARK_INSTALLED = False
try:
    from lightgbm_ray import RayDMatrix
    from ray.data import Dataset as RayDataset

    RAY_INSTALLED = True
except ModuleNotFoundError:
    RAY_INSTALLED = False
from sklearn.base import clone

from mlforecast.core import (
    DateFeature,
    Freq,
    LagTransforms,
    Lags,
    TargetTransform,
    TimeSeries,
    _build_transform_name,
    _name_models,
)
from ..forecast import MLForecast
from ..grouped_array import GroupedArray

# %% ../../nbs/distributed.forecast.ipynb 6
WindowInfo = namedtuple(
    "WindowInfo", ["n_windows", "window_size", "step_size", "i_window", "input_size"]
)

# %% ../../nbs/distributed.forecast.ipynb 7
class DistributedMLForecast:
    """Multi backend distributed pipeline"""

    def __init__(
        self,
        models,
        freq: Freq,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        num_threads: int = 1,
        target_transforms: Optional[List[TargetTransform]] = None,
        engine=None,
        num_partitions: Optional[int] = None,
        lag_transforms_namer: Optional[Callable] = None,
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
        num_threads : int (default=1)
            Number of threads to use when computing the features.
        target_transforms : list of transformers, optional(default=None)
            Transformations that will be applied to the target before computing the features and restored after the forecasting step.
        engine : fugue execution engine, optional (default=None)
            Dask Client, Spark Session, etc to use for the distributed computation.
            If None will infer depending on the input type.
        num_partitions: number of data partitions to use, optional (default=None)
            If None, the default partitions provided by the AnyDataFrame used
            by the `fit` and `cross_validation` methods will be used. If a Ray
            Dataset is provided and `num_partitions` is None, the partitioning
            will be done by the `id_col`.
        lag_transforms_namer : callable, optional(default=None)
            Function that takes a transformation (either function or class), a lag and extra arguments and produces a name
        """
        if not isinstance(models, dict) and not isinstance(models, list):
            models = [models]
        if isinstance(models, list):
            model_names = _name_models([m.__class__.__name__ for m in models])
            models_with_names = dict(zip(model_names, models))
        else:
            models_with_names = models
        self.models = models_with_names
        if lag_transforms_namer is None:

            def name_without_dots(tfm, lag, *args):
                name = _build_transform_name(tfm, lag, args)
                return name.replace(".", "_")

            lag_transforms_namer = name_without_dots
        self._base_ts = TimeSeries(
            freq=freq,
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
            num_threads=num_threads,
            target_transforms=target_transforms,
            lag_transforms_namer=lag_transforms_namer,
        )
        self.engine = engine
        self.num_partitions = num_partitions

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
            core_tfms = ts._get_core_lag_tfms()
            if core_tfms:
                # populate the stats needed for the updates
                ts._compute_transforms(core_tfms, updates_only=False)
            ts.as_numpy = False
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
            max_dates = part.groupby(id_col, observed=True)[time_col].transform("max")
            cutoffs, train_mask, valid_mask = ufp._single_split(
                part,
                i_window=window_info.i_window,
                n_windows=window_info.n_windows,
                h=window_info.window_size,
                id_col=id_col,
                time_col=time_col,
                freq=ts.freq,
                max_dates=max_dates,
                step_size=window_info.step_size,
                input_size=window_info.input_size,
            )
            train = part[train_mask]
            valid_keep_cols = part.columns
            if static_features is not None:
                valid_keep_cols = valid_keep_cols.drop(static_features)
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
        if self.num_partitions:
            partition = dict(by=id_col, num=self.num_partitions, algo="coarse")
        elif RAY_INSTALLED and isinstance(
            data, RayDataset
        ):  # num partitions is None but data is a RayDataset
            # We need to add this because
            # currently ray doesnt support partitioning a Dataset
            # based on a column.
            # If a Dataset is partitioned using `.repartition(num_partitions)`
            # we will have akward results.
            partition = dict(by=id_col)
        else:
            partition = None
        res = fa.transform(
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
            partition=partition,
        )
        # so that we don't need to recompute this on predict
        return fa.persist(res, lazy=False, engine=self.engine, as_fugue=True)

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
        self._base_ts.id_col = id_col
        self._base_ts.time_col = time_col
        self._base_ts.target_col = target_col
        self._base_ts.static_features = static_features
        self._base_ts.dropna = dropna
        self._base_ts.keep_last_n = keep_last_n
        self._partition_results = self._preprocess_partitions(
            data=data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            window_info=window_info,
        )
        base_schema = fa.get_schema(data)
        features_schema = {
            f: "double" for f in self._base_ts.features if f not in base_schema
        }
        res = fa.transform(
            self._partition_results,
            DistributedMLForecast._retrieve_df,
            schema=base_schema + features_schema,
            engine=self.engine,
        )
        return fa.get_native_as_df(res)

    def preprocess(
        self,
        df: fugue.AnyDataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> fugue.AnyDataFrame:
        """Add the features to `data`.

        Parameters
        ----------
        df : dask, spark or ray DataFrame.
            Series data in long format.
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.

        Returns
        -------
        result : same type as df
            `df` with added features.
        """
        return self._preprocess(
            df,
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
            X = RayDMatrix(
                prep.select_columns(cols=features + [target_col]),
                label=target_col,
            )
            for name, model in self.models.items():
                trained_model = clone(model).fit(X, y=None)
                self.models_[name] = trained_model.model_
        else:
            raise NotImplementedError(
                "Only spark, dask, and ray engines are supported."
            )
        return self

    def fit(
        self,
        df: fugue.AnyDataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> "DistributedMLForecast":
        """Apply the feature engineering and train the models.

        Parameters
        ----------
        df : dask, spark or ray DataFrame
            Series data in long format.
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
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
            df,
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
        before_predict_callback=None,
        after_predict_callback=None,
        X_df=None,
    ) -> Iterable[pd.DataFrame]:
        for serialized_ts, _, serialized_valid in items:
            valid = cloudpickle.loads(serialized_valid)
            if valid is not None:
                X_df = valid
            ts = cloudpickle.loads(serialized_ts)
            res = ts.predict(
                models=models,
                horizon=horizon,
                before_predict_callback=before_predict_callback,
                after_predict_callback=after_predict_callback,
                X_df=X_df,
            )
            if valid is not None:
                res = res.merge(valid, how="left")
            yield res

    def _get_predict_schema(self) -> str:
        model_names = self.models.keys()
        models_schema = ",".join(f"{model_name}:double" for model_name in model_names)
        schema = (
            f"{self._base_ts.id_col}:string,{self._base_ts.time_col}:datetime,"
            + models_schema
        )
        return schema

    def predict(
        self,
        h: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        X_df: Optional[pd.DataFrame] = None,
        new_df: Optional[fugue.AnyDataFrame] = None,
    ) -> fugue.AnyDataFrame:
        """Compute the predictions for the next `horizon` steps.

        Parameters
        ----------
        h : int
            Forecast horizon.
        before_predict_callback : callable, optional (default=None)
            Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index.
        after_predict_callback : callable, optional (default=None)
            Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index.
        X_df : pandas DataFrame, optional (default=None)
            Dataframe with the future exogenous features. Should have the id column and the time column.
        new_df : dask or spark DataFrame, optional (default=None)
            Series data of new observations for which forecasts are to be generated.
                This dataframe should have the same structure as the one used to fit the model, including any features and time series data.
                If `new_df` is not None, the method will generate forecasts for the new observations.

        Returns
        -------
        result : dask, spark or ray DataFrame
            Predictions for each serie and timestep, with one column per model.
        """
        if new_df is not None:
            partition_results = self._preprocess_partitions(
                new_df,
                id_col=self._base_ts.id_col,
                time_col=self._base_ts.time_col,
                target_col=self._base_ts.target_col,
                static_features=self._base_ts.static_features,
                dropna=self._base_ts.dropna,
                keep_last_n=self._base_ts.keep_last_n,
                fit_ts_only=True,
            )
        else:
            partition_results = self._partition_results
        schema = self._get_predict_schema()
        if X_df is not None and not isinstance(X_df, pd.DataFrame):
            raise ValueError("`X_df` should be a pandas DataFrame")
        res = fa.transform(
            partition_results,
            DistributedMLForecast._predict,
            params={
                "models": self.models_,
                "horizon": h,
                "before_predict_callback": before_predict_callback,
                "after_predict_callback": after_predict_callback,
                "X_df": X_df,
            },
            schema=schema,
            engine=self.engine,
        )
        return fa.get_native_as_df(res)

    def cross_validation(
        self,
        df: fugue.AnyDataFrame,
        n_windows: int,
        h: int,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        step_size: Optional[int] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        refit: bool = True,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        input_size: Optional[int] = None,
    ) -> fugue.AnyDataFrame:
        """Perform time series cross validation.
        Creates `n_windows` splits where each window has `h` test periods,
        trains the models, computes the predictions and merges the actuals.

        Parameters
        ----------
        df : dask, spark or ray DataFrame
            Series data in long format.
        n_windows : int
            Number of windows to evaluate.
        h : int
            Number of test periods in each window.
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        step_size : int, optional (default=None)
            Step size between each cross validation window. If None it will be equal to `h`.
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
        input_size : int, optional (default=None)
            Maximum training samples per serie in each window. If None, will use an expanding window.

        Returns
        -------
        result : dask, spark or ray DataFrame
            Predictions for each window with the series id, timestamp, target value and predictions from each model.
        """
        self.cv_models_ = []
        results = []
        for i in range(n_windows):
            window_info = WindowInfo(n_windows, h, step_size, i, input_size)
            if refit or i == 0:
                self._fit(
                    df,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    static_features=static_features,
                    dropna=dropna,
                    keep_last_n=keep_last_n,
                    window_info=window_info,
                )
                self.cv_models_.append(self.models_)
                partition_results = self._partition_results
            elif not refit:
                partition_results = self._preprocess_partitions(
                    df,
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
                + f",cutoff:datetime,{self._base_ts.target_col}:double"
            )
            preds = fa.transform(
                partition_results,
                DistributedMLForecast._predict,
                params={
                    "models": self.models_,
                    "horizon": h,
                    "before_predict_callback": before_predict_callback,
                    "after_predict_callback": after_predict_callback,
                },
                schema=schema,
                engine=self.engine,
            )
            results.append(fa.get_native_as_df(preds))
        if len(results) == 1:
            res = results[0]
        else:
            res = fa.union(*results)
        return res

    @staticmethod
    def _save_ts(items: List[List[Any]], path: str) -> Iterable[pd.DataFrame]:
        for serialized_ts, _, _ in items:
            ts = cloudpickle.loads(serialized_ts)
            first_uid = ts.uids[0]
            last_uid = ts.uids[-1]
            ts.save(f"{path}/ts_{first_uid}-{last_uid}.pkl")
            yield pd.DataFrame({"x": [True]})

    def save(self, path: str) -> None:
        """Save forecast object

        Parameters
        ----------
        path : str
            Directory where artifacts will be stored."""
        dummy_df = fa.transform(
            self._partition_results,
            DistributedMLForecast._save_ts,
            schema="x:bool",
            params={"path": path},
            engine=self.engine,
        )
        # trigger computation
        dummy_df.as_pandas()
        with fsspec.open(f"{path}/models.pkl", "wb") as f:
            cloudpickle.dump(self.models_, f)
        self._base_ts.save(f"{path}/_base_ts.pkl")

    @staticmethod
    def _load_ts(paths: List[List[Any]], protocol: str) -> Iterable[pd.DataFrame]:
        for [path] in paths:
            ts = TimeSeries.load(path, protocol=protocol)
            yield pd.DataFrame(
                {
                    "ts": [cloudpickle.dumps(ts)],
                    "train": [cloudpickle.dumps(None)],
                    "valid": [cloudpickle.dumps(None)],
                }
            )

    @staticmethod
    def load(path: str, engine) -> "DistributedMLForecast":
        """Load forecast object

        Parameters
        ----------
        path : str
            Directory with saved artifacts.
        engine : fugue execution engine
            Dask Client, Spark Session, etc to use for the distributed computation.
        """
        fs, _, paths = fsspec.get_fs_token_paths(f"{path}/ts*")
        protocol = fs.protocol
        if isinstance(protocol, tuple):
            protocol = protocol[0]
        names_df = pd.DataFrame({"path": paths})
        partition_results = fa.transform(
            names_df,
            DistributedMLForecast._load_ts,
            schema="ts:binary,train:binary,valid:binary",
            partition="per_row",
            params={"protocol": protocol},
            engine=engine,
            as_fugue=True,
        )
        with fsspec.open(f"{path}/models.pkl", "rb") as f:
            models = cloudpickle.load(f)
        base_ts = TimeSeries.load(f"{path}/_base_ts.pkl")
        fcst = DistributedMLForecast(models=models, freq=base_ts.freq)
        fcst._base_ts = base_ts
        fcst._partition_results = fa.persist(
            partition_results, lazy=False, engine=engine, as_fugue=True
        )
        fcst.models_ = models
        fcst.engine = engine
        fcst.num_partitions = len(paths)
        return fcst

    @staticmethod
    def _update(items: List[List[Any]], new_df) -> Iterable[List[Any]]:
        for serialized_ts, serialized_transformed, serialized_valid in items:
            ts = cloudpickle.loads(serialized_ts)
            partition_mask = ufp.is_in(new_df[ts.id_col], ts.uids)
            partition_df = ufp.filter_with_mask(new_df, partition_mask)
            ts.update(partition_df)
            yield [cloudpickle.dumps(ts), serialized_transformed, serialized_valid]

    def update(self, df: pd.DataFrame) -> None:
        """Update the values of the stored series.

        Parameters
        ----------
        df : pandas DataFrame
            Dataframe with new observations."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("`df` must be a pandas DataFrame.")
        res = fa.transform(
            self._partition_results,
            DistributedMLForecast._update,
            params={"new_df": df},
            schema="ts:binary,train:binary,valid:binary",
            engine=self.engine,
            as_fugue=True,
        )
        self._partition_results = fa.persist(res)

    def to_local(self) -> MLForecast:
        """Convert this distributed forecast object into a local one

        This pulls all the data from the remote machines, so you have to be sure that
        it fits in the scheduler/driver. If you're not sure use the save method instead.

        Returns
        -------
        MLForecast
            Local forecast object."""
        serialized_ts = (
            fa.select_columns(
                self._partition_results,
                columns=["ts"],
                as_fugue=True,
            )
            .as_pandas()["ts"]
            .tolist()
        )
        all_ts = [cloudpickle.loads(ts) for ts in serialized_ts]
        # sort by ids (these should already be sorted within each partition)
        all_ts = sorted(all_ts, key=lambda ts: ts.uids[0])

        # combine attributes. since fugue works on pandas these are all pandas.
        # we're using utilsforecast here in case we add support for polars
        def possibly_concat_indices(collection):
            items_are_indices = isinstance(collection[0], pd.Index)
            if items_are_indices:
                collection = [pd.Series(item) for item in collection]
            combined = ufp.vertical_concat(collection)
            if items_are_indices:
                combined = pd.Index(combined)
            return combined

        def combine_target_tfms(by_partition):
            if by_partition[0] is None:
                return None
            by_transform = [
                [part[i] for part in by_partition] for i in range(len(by_partition[0]))
            ]
            out = []
            for tfms in by_transform:
                out.append(tfms[0].stack(tfms))
            return out

        def combine_core_lag_tfms(by_partition):
            by_transform = [
                (name, [part[name] for part in by_partition])
                for name in by_partition[0].keys()
            ]
            out = {}
            for name, partition_tfms in by_transform:
                out[name] = partition_tfms[0].stack(partition_tfms)
            return out

        uids = possibly_concat_indices([ts.uids for ts in all_ts])
        last_dates = possibly_concat_indices([ts.last_dates for ts in all_ts])
        statics = ufp.vertical_concat([ts.static_features_ for ts in all_ts])
        combined_target_tfms = combine_target_tfms(
            [ts.target_transforms for ts in all_ts]
        )
        combined_core_lag_tfms = combine_core_lag_tfms(
            [ts._get_core_lag_tfms() for ts in all_ts]
        )
        sizes = np.hstack([np.diff(ts.ga.indptr) for ts in all_ts])
        data = np.hstack([ts.ga.data for ts in all_ts])
        indptr = np.append(0, sizes).cumsum()
        if isinstance(uids, pd.Index):
            uids_idx = uids
        else:
            # uids is polars series
            uids_idx = pd.Index(uids)
        if not uids_idx.is_monotonic_increasing:
            # this seems to happen only with ray
            # we have to sort all data related to the series
            sort_idxs = uids_idx.argsort()
            uids = uids[sort_idxs]
            last_dates = last_dates[sort_idxs]
            statics = ufp.take_rows(statics, sort_idxs)
            statics = ufp.drop_index_if_pandas(statics)
            for tfm in combined_core_lag_tfms.values():
                tfm._core_tfm = tfm._core_tfm.take(sort_idxs)
            if combined_target_tfms is not None:
                combined_target_tfms = [
                    tfm.take(sort_idxs) for tfm in combined_target_tfms
                ]
            old_data = data.copy()
            old_indptr = indptr.copy()
            indptr = np.append(0, sizes[sort_idxs]).cumsum()
            # this loop takes 500ms for 100,000 series of sizes between 500 and 2,000
            # so it may not be that much of a bottleneck, but try to implement in core
            for i, sort_idx in enumerate(sort_idxs):
                old_slice = slice(old_indptr[sort_idx], old_indptr[sort_idx + 1])
                new_slice = slice(indptr[i], indptr[i + 1])
                data[new_slice] = old_data[old_slice]
        ga = GroupedArray(data, indptr)

        # all other attributes should be the same, so we just override the first serie
        ts = all_ts[0]
        ts.uids = uids
        ts.last_dates = last_dates
        ts.ga = ga
        ts.static_features_ = statics
        ts.transforms.update(combined_core_lag_tfms)
        ts.target_transforms = combined_target_tfms
        fcst = MLForecast(models=self.models_, freq=ts.freq)
        fcst.ts = ts
        fcst.models_ = self.models_
        return fcst
