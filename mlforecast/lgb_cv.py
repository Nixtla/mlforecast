# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/lgb_cv.ipynb.

# %% auto 0
__all__ = ['LightGBMCV']

# %% ../nbs/lgb_cv.ipynb 3
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from mlforecast.core import (
    DateFeature,
    Freq,
    LagTransforms,
    Lags,
    TimeSeries,
)
from .utils import backtest_splits
from .target_transforms import BaseTargetTransform, Differences

# %% ../nbs/lgb_cv.ipynb 5
def _mape(y_true, y_pred):
    abs_pct_err = abs(y_true - y_pred) / y_true
    return (
        abs_pct_err.groupby(y_true.index.get_level_values(0), observed=True)
        .mean()
        .mean()
    )


def _rmse(y_true, y_pred):
    sq_err = (y_true - y_pred) ** 2
    return (
        sq_err.groupby(y_true.index.get_level_values(0), observed=True)
        .mean()
        .pow(0.5)
        .mean()
    )


_metric2fn = {"mape": _mape, "rmse": _rmse}


def _update(bst, n):
    for _ in range(n):
        bst.update()


def _predict(
    ts,
    bst,
    valid,
    h,
    id_col,
    time_col,
    dynamic_dfs,
    before_predict_callback,
    after_predict_callback,
):
    preds = ts.predict(
        {"Booster": bst},
        h,
        dynamic_dfs,
        before_predict_callback,
        after_predict_callback,
    )
    return valid.merge(preds, on=[id_col, time_col], how="left")


def _update_and_predict(
    ts,
    bst,
    valid,
    n,
    h,
    id_col,
    time_col,
    dynamic_dfs,
    before_predict_callback,
    after_predict_callback,
):
    _update(bst, n)
    return _predict(
        ts,
        bst,
        valid,
        h,
        id_col,
        time_col,
        dynamic_dfs,
        before_predict_callback,
        after_predict_callback,
    )

# %% ../nbs/lgb_cv.ipynb 6
CVResult = Tuple[int, float]

# %% ../nbs/lgb_cv.ipynb 7
class LightGBMCV:
    def __init__(
        self,
        freq: Optional[Freq] = None,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        differences: Optional[Iterable[int]] = None,
        num_threads: int = 1,
        target_transforms: Optional[List[BaseTargetTransform]] = None,
    ):
        """Create LightGBM CV object.

        Parameters
        ----------
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
        target_transforms : list of transformers, optional(default=None)
            Transformations that will be applied to the target before computing the features and restored after the forecasting step.
        """
        self.num_threads = num_threads
        cpu_count = os.cpu_count()
        if cpu_count is None:
            num_cpus = 1
        else:
            num_cpus = cpu_count
        self.bst_threads = max(num_cpus // num_threads, 1)
        self.ts = TimeSeries(
            freq=freq,
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
            differences=differences,
            num_threads=self.bst_threads,
            target_transforms=target_transforms,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"freq={self.ts.freq}, "
            f"lag_features={list(self.ts.transforms.keys())}, "
            f"date_features={self.ts.date_features}, "
            f"num_threads={self.num_threads}, "
            f"bst_threads={self.bst_threads})"
        )

    def setup(
        self,
        data: pd.DataFrame,
        n_windows: int,
        window_size: int,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        step_size: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
        metric: Union[str, Callable] = "mape",
        input_size: Optional[int] = None,
    ):
        """Initialize internal data structures to iteratively train the boosters. Use this before calling partial_fit.

        Parameters
        ----------
        data : pandas DataFrame
            Series data in long format.
        n_windows : int
            Number of windows to evaluate.
        window_size : int
            Number of test periods in each window.
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        step_size : int, optional (default=None)
            Step size between each cross validation window. If None it will be equal to `window_size`.
        params : dict, optional(default=None)
            Parameters to be passed to the LightGBM Boosters.
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
        weights : sequence of float, optional (default=None)
            Weights to multiply the metric of each window. If None, all windows have the same weight.
        metric : str or callable, default='mape'
            Metric used to assess the performance of the models and perform early stopping.
        input_size : int, optional (default=None)
            Maximum training samples per serie in each window. If None, will use an expanding window.

        Returns
        -------
        self : LightGBMCV
            CV object with internal data structures for partial_fit.
        """
        if weights is None:
            self.weights = np.full(n_windows, 1 / n_windows)
        elif len(weights) != n_windows:
            raise ValueError("Must specify as many weights as the number of windows")
        else:
            self.weights = np.asarray(weights)
        if callable(metric):
            self.metric_fn = metric
            self.metric_name = "custom_metric"
        else:
            if metric not in _metric2fn:
                raise ValueError(
                    f'{metric} is not one of the implemented metrics: ({", ".join(_metric2fn.keys())})'
                )
            self.metric_fn = _metric2fn[metric]
            self.metric_name = metric
        if np.issubdtype(data[time_col].dtype.type, np.integer):
            freq = 1
        else:
            freq = self.ts.freq
        self.items = []
        self.window_size = window_size
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.params = {} if params is None else params
        splits = backtest_splits(
            data,
            n_windows=n_windows,
            window_size=window_size,
            id_col=id_col,
            time_col=time_col,
            freq=freq,
            step_size=step_size,
            input_size=input_size,
        )
        for _, train, valid in splits:
            ts = copy.deepcopy(self.ts)
            prep = ts.fit_transform(
                train,
                id_col,
                time_col,
                target_col,
                static_features,
                dropna,
                keep_last_n,
            )
            ds = lgb.Dataset(
                prep.drop(columns=[id_col, time_col, target_col]), prep[target_col]
            ).construct()
            bst = lgb.Booster({**self.params, "num_threads": self.bst_threads}, ds)
            bst.predict = partial(bst.predict, num_threads=self.bst_threads)
            valid = valid.set_index(time_col, append=True)
            self.items.append((ts, bst, valid))
        return self

    def _single_threaded_partial_fit(
        self,
        metric_values,
        num_iterations,
        dynamic_dfs,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ):
        for j, (ts, bst, valid) in enumerate(self.items):
            preds = _update_and_predict(
                ts=ts,
                bst=bst,
                valid=valid,
                n=num_iterations,
                h=self.window_size,
                id_col=self.id_col,
                time_col=self.time_col,
                dynamic_dfs=dynamic_dfs,
                before_predict_callback=before_predict_callback,
                after_predict_callback=after_predict_callback,
            )
            metric_values[j] = self.metric_fn(preds[self.target_col], preds["Booster"])

    def _multithreaded_partial_fit(
        self,
        metric_values,
        num_iterations,
        dynamic_dfs,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ):
        with ThreadPoolExecutor(self.num_threads) as executor:
            futures = []
            for ts, bst, valid in self.items:
                _update(bst, num_iterations)
                future = executor.submit(
                    _predict,
                    ts=ts,
                    bst=bst,
                    valid=valid,
                    h=self.window_size,
                    id_col=self.id_col,
                    time_col=self.time_col,
                    dynamic_dfs=dynamic_dfs,
                    before_predict_callback=before_predict_callback,
                    after_predict_callback=after_predict_callback,
                )
                futures.append(future)
            cv_preds = [f.result() for f in futures]
        metric_values[:] = [
            self.metric_fn(preds[self.target_col], preds["Booster"])
            for preds in cv_preds
        ]

    def partial_fit(
        self,
        num_iterations: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ) -> float:
        """Train the boosters for some iterations.

        Parameters
        ----------
        num_iterations : int
            Number of boosting iterations to run
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

        Returns
        -------
        metric_value : float
            Weighted metric after training for num_iterations.
        """
        metric_values = np.empty(len(self.items))
        if self.num_threads == 1:
            self._single_threaded_partial_fit(
                metric_values,
                num_iterations,
                dynamic_dfs,
                before_predict_callback,
                after_predict_callback,
            )
        else:
            self._multithreaded_partial_fit(
                metric_values,
                num_iterations,
                dynamic_dfs,
                before_predict_callback,
                after_predict_callback,
            )
        return metric_values @ self.weights

    def should_stop(self, hist, early_stopping_evals, early_stopping_pct) -> bool:
        if len(hist) < early_stopping_evals + 1:
            return False
        improvement_pct = 1 - hist[-1][1] / hist[-(early_stopping_evals + 1)][1]
        return improvement_pct < early_stopping_pct

    def find_best_iter(self, hist, early_stopping_evals) -> int:
        best_iter, best_score = hist[-1]
        for r, m in hist[-(early_stopping_evals + 1) : -1]:
            if m < best_score:
                best_score = m
                best_iter = r
        return best_iter

    def fit(
        self,
        data: pd.DataFrame,
        n_windows: int,
        window_size: int,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        step_size: Optional[int] = None,
        num_iterations: int = 100,
        params: Optional[Dict[str, Any]] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        eval_every: int = 10,
        weights: Optional[Sequence[float]] = None,
        metric: Union[str, Callable] = "mape",
        verbose_eval: bool = True,
        early_stopping_evals: int = 2,
        early_stopping_pct: float = 0.01,
        compute_cv_preds: bool = False,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        input_size: Optional[int] = None,
    ) -> List[CVResult]:
        """Train boosters simultaneously and assess their performance on the complete forecasting window.

        Parameters
        ----------
        data : pandas DataFrame
            Series data in long format.
        n_windows : int
            Number of windows to evaluate.
        window_size : int
            Number of test periods in each window.
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        step_size : int, optional (default=None)
            Step size between each cross validation window. If None it will be equal to `window_size`.
        num_iterations : int (default=100)
            Maximum number of boosting iterations to run.
        params : dict, optional(default=None)
            Parameters to be passed to the LightGBM Boosters.
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
        dynamic_dfs : list of pandas DataFrame, optional (default=None)
            Future values of the dynamic features, e.g. prices.
        eval_every : int (default=10)
            Number of boosting iterations to train before evaluating on the whole forecast window.
        weights : sequence of float, optional (default=None)
            Weights to multiply the metric of each window. If None, all windows have the same weight.
        metric : str or callable, default='mape'
            Metric used to assess the performance of the models and perform early stopping.
        verbose_eval : bool
            Print the metrics of each evaluation.
        early_stopping_evals : int (default=2)
            Maximum number of evaluations to run without improvement.
        early_stopping_pct : float (default=0.01)
            Minimum percentage improvement in metric value in `early_stopping_evals` evaluations.
        compute_cv_preds : bool (default=True)
            Compute predictions for each window after finding the best iteration.
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
        cv_result : list of tuple.
            List of (boosting rounds, metric value) tuples.
        """
        self.setup(
            data=data,
            n_windows=n_windows,
            window_size=window_size,
            params=params,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            input_size=input_size,
            step_size=step_size,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            weights=weights,
            metric=metric,
        )
        hist = []
        for i in range(0, num_iterations, eval_every):
            metric_value = self.partial_fit(
                eval_every, dynamic_dfs, before_predict_callback, after_predict_callback
            )
            rounds = eval_every + i
            hist.append((rounds, metric_value))
            if verbose_eval:
                print(f"[{rounds:,d}] {self.metric_name}: {metric_value:,f}")
            if self.should_stop(hist, early_stopping_evals, early_stopping_pct):
                print(f"Early stopping at round {rounds:,}")
                break
        self.best_iteration_ = self.find_best_iter(hist, early_stopping_evals)
        print(f"Using best iteration: {self.best_iteration_:,}")
        hist = hist[: self.best_iteration_ // eval_every]
        for _, bst, _ in self.items:
            bst.best_iteration = self.best_iteration_

        self.cv_models_ = {f"Booster{i}": item[1] for i, item in enumerate(self.items)}
        if compute_cv_preds:
            with ThreadPoolExecutor(self.num_threads) as executor:
                futures = []
                for ts, bst, valid in self.items:
                    future = executor.submit(
                        _predict,
                        ts=ts,
                        bst=bst,
                        valid=valid,
                        h=self.window_size,
                        id_col=self.id_col,
                        time_col=self.time_col,
                        dynamic_dfs=dynamic_dfs,
                        before_predict_callback=before_predict_callback,
                        after_predict_callback=after_predict_callback,
                    )
                    futures.append(future)
                self.cv_preds_ = pd.concat(
                    [f.result().assign(window=i) for i, f in enumerate(futures)]
                )
        self.ts._fit(data, id_col, time_col, target_col, static_features, keep_last_n)
        return hist

    def predict(
        self,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """Compute predictions with each of the trained boosters.

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

        Returns
        -------
        result : pandas DataFrame
            Predictions for each serie and timestep, with one column per window.
        """
        return self.ts.predict(
            self.cv_models_,
            horizon,
            dynamic_dfs,
            before_predict_callback,
            after_predict_callback,
        )
