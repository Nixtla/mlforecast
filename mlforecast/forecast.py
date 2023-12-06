# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/forecast.ipynb.

# %% auto 0
__all__ = ['MLForecast']

# %% ../nbs/forecast.ipynb 3
import copy
import re
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from utilsforecast.compat import DataFrame
from utilsforecast.processing import (
    assign_columns,
    backtest_splits,
    copy_if_pandas,
    counts_by_id,
    drop_index_if_pandas,
    filter_with_mask,
    is_in,
    is_nan,
    join,
    maybe_compute_sort_indices,
    take_rows,
    to_numpy,
    vertical_concat,
)

from mlforecast.core import (
    DateFeature,
    Freq,
    LagTransforms,
    Lags,
    Models,
    TargetTransform,
    TimeSeries,
    _name_models,
)
from .grouped_array import GroupedArray

if TYPE_CHECKING:
    from mlforecast.lgb_cv import LightGBMCV
from .target_transforms import BaseGroupedArrayTargetTransform
from .utils import PredictionIntervals

# %% ../nbs/forecast.ipynb 6
def _add_conformal_distribution_intervals(
    fcst_df: DataFrame,
    cs_df: DataFrame,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
) -> DataFrame:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This strategy creates forecasts paths
    based on errors and calculate quantiles using those paths.
    """
    fcst_df = copy_if_pandas(fcst_df, deep=False)
    alphas = [100 - lv for lv in level]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    for model in model_names:
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        # restrict scores to horizon
        scores = scores[:, :, :horizon]
        mean = fcst_df[model].to_numpy().reshape(1, n_series, -1)
        scores = np.vstack([mean - scores, mean + scores])
        quantiles = np.quantile(
            scores,
            cuts,
            axis=0,
        )
        quantiles = quantiles.reshape(len(cuts), -1).T
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        out_cols = lo_cols + hi_cols
        fcst_df = assign_columns(fcst_df, out_cols, quantiles)
    return fcst_df

# %% ../nbs/forecast.ipynb 7
def _add_conformal_error_intervals(
    fcst_df: DataFrame,
    cs_df: DataFrame,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
) -> DataFrame:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This startegy creates prediction intervals
    based on the absolute errors.
    """
    fcst_df = copy_if_pandas(fcst_df, deep=False)
    cuts = [lv / 100 for lv in level]
    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        # restrict scores to horizon
        scores = scores[:, :, :horizon]
        quantiles = np.quantile(
            scores,
            cuts,
            axis=0,
        )
        quantiles = quantiles.reshape(len(cuts), -1)
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        quantiles = np.vstack([mean - quantiles[::-1], mean + quantiles]).T
        columns = lo_cols + hi_cols
        fcst_df = assign_columns(fcst_df, columns, quantiles)
    return fcst_df

# %% ../nbs/forecast.ipynb 8
def _get_conformal_method(method: str):
    available_methods = {
        "conformal_distribution": _add_conformal_distribution_intervals,
        "conformal_error": _add_conformal_error_intervals,
    }
    if method not in available_methods.keys():
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f'please choose one of {", ".join(available_methods.keys())}'
        )
    return available_methods[method]

# %% ../nbs/forecast.ipynb 10
class MLForecast:
    def __init__(
        self,
        models: Models,
        freq: Freq,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        num_threads: int = 1,
        target_transforms: Optional[List[TargetTransform]] = None,
        lag_transforms_namer: Optional[Callable] = None,
    ):
        """Forecasting pipeline

        Parameters
        ----------
        models : regressor or list of regressors
            Models that will be trained and used to compute the forecasts.
        freq : str or int or pd.offsets.BaseOffset
            Pandas offset, pandas offset alias, e.g. 'D', 'W-THU' or integer denoting the frequency of the series.
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
        lag_transforms_namer : callable, optional(default=None)
            Function that takes a transformation (either function or class), a lag and extra arguments and produces a name.
        """
        if not isinstance(models, dict) and not isinstance(models, list):
            models = [models]
        if isinstance(models, list):
            model_names = _name_models([m.__class__.__name__ for m in models])
            models_with_names = dict(zip(model_names, models))
        else:
            models_with_names = models
        self.models = models_with_names
        self.ts = TimeSeries(
            freq=freq,
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
            num_threads=num_threads,
            target_transforms=target_transforms,
            lag_transforms_namer=lag_transforms_namer,
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(models=[{", ".join(self.models.keys())}], '
            f"freq={self.freq}, "
            f"lag_features={list(self.ts.transforms.keys())}, "
            f"date_features={self.ts.date_features}, "
            f"num_threads={self.ts.num_threads})"
        )

    @property
    def freq(self):
        return self.ts.freq

    @classmethod
    def from_cv(cls, cv: "LightGBMCV") -> "MLForecast":
        if not hasattr(cv, "best_iteration_"):
            raise ValueError("LightGBMCV object must be fitted first.")
        import lightgbm as lgb

        fcst = cls(
            models=lgb.LGBMRegressor(
                **{**cv.params, "n_estimators": cv.best_iteration_}
            ),
            freq=cv.ts.freq,
        )
        fcst.ts = copy.deepcopy(cv.ts)
        return fcst

    def preprocess(
        self,
        df: DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        return_X_y: bool = False,
        as_numpy: bool = False,
    ) -> Union[DataFrame, Tuple[DataFrame, np.ndarray]]:
        """Add the features to `data`.

        Parameters
        ----------
        df : pandas DataFrame
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
        max_horizon : int, optional (default=None)
            Train this many models, where each model will predict a specific horizon.
        return_X_y : bool (default=False)
            Return a tuple with the features and the target. If False will return a single dataframe.
        as_numpy : bool (default = False)
            Cast features to numpy array. Only works for `return_X_y=True`.

        Returns
        -------
        result : DataFrame or tuple of pandas Dataframe and a numpy array.
            `df` plus added features and target(s).
        """
        return self.ts.fit_transform(
            df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            max_horizon=max_horizon,
            return_X_y=return_X_y,
            as_numpy=as_numpy,
        )

    def fit_models(
        self,
        X: Union[DataFrame, np.ndarray],
        y: np.ndarray,
    ) -> "MLForecast":
        """Manually train models. Use this if you called `MLForecast.preprocess` beforehand.

        Parameters
        ----------
        X : pandas or polars DataFrame or numpy array
            Features.
        y : numpy array.
            Target.

        Returns
        -------
        self : MLForecast
            Forecast object with trained models.
        """
        self.models_: Dict[str, Union[BaseEstimator, List[BaseEstimator]]] = {}
        for name, model in self.models.items():
            if y.ndim == 2 and y.shape[1] > 1:
                self.models_[name] = []
                for col in range(y.shape[1]):
                    keep = ~np.isnan(y[:, col])
                    if isinstance(X, np.ndarray):
                        # TODO: migrate to utils
                        Xh = X[keep]
                    else:
                        Xh = filter_with_mask(X, keep)
                    yh = y[keep, col]
                    self.models_[name].append(clone(model).fit(Xh, yh))
            else:
                self.models_[name] = clone(model).fit(X, y)
        return self

    def _conformity_scores(
        self,
        df: DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        n_windows: int = 2,
        h: int = 1,
        as_numpy: bool = False,
    ):
        """Compute conformity scores.

        We need at least two cross validation errors to compute
        quantiles for prediction intervals (`n_windows=2`).

        The exception is raised by the PredictionIntervals data class.

        In this simplest case, we assume the width of the interval
        is the same for all the forecasting horizon (`h=1`).
        """
        cv_results = self.cross_validation(
            df=df,
            n_windows=n_windows,
            h=h,
            refit=False,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            max_horizon=max_horizon,
            prediction_intervals=None,
            as_numpy=as_numpy,
        )
        # conformity score for each model
        for model in self.models.keys():
            # compute absolute error for each model
            abs_err = abs(cv_results[model] - cv_results[target_col])
            cv_results = assign_columns(cv_results, model, abs_err)
        return cv_results.drop(columns=target_col)

    def _invert_transforms_fitted(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.ts.target_transforms is None:
            return df
        if any(
            isinstance(tfm, BaseGroupedArrayTargetTransform)
            for tfm in self.ts.target_transforms
        ):
            model_cols = [
                c for c in df.columns if c not in (self.ts.id_col, self.ts.time_col)
            ]
            id_counts = counts_by_id(df, self.ts.id_col)
            sizes = id_counts["counts"].to_numpy()
            indptr = np.append(0, sizes.cumsum())
        for tfm in self.ts.target_transforms[::-1]:
            if isinstance(tfm, BaseGroupedArrayTargetTransform):
                if self.ts._dropped_series is not None:
                    tfm.idxs = np.delete(
                        np.arange(self.ts.ga.n_groups), self.ts._dropped_series
                    )
                for col in model_cols:
                    ga = GroupedArray(df[col].to_numpy(), indptr)
                    ga = tfm.inverse_transform_fitted(ga)
                    df = assign_columns(df, col, ga.data)
                tfm.idxs = None
            else:
                df = tfm.inverse_transform(df)
        return df

    def _extract_X_y(
        self,
        prep: DataFrame,
        target_col: str,
    ) -> Tuple[Union[DataFrame, np.ndarray], np.ndarray]:
        X = prep[self.ts.features_order_]
        targets = [c for c in prep.columns if re.match(rf"^{target_col}\d?$", c)]
        if len(targets) == 1:
            targets = targets[0]
        y = prep[targets].to_numpy()
        return X, y

    def _compute_fitted_values(
        self,
        base: DataFrame,
        X: Union[DataFrame, np.ndarray],
        y: np.ndarray,
        id_col: str,
        time_col: str,
        target_col: str,
        max_horizon: Optional[int],
    ) -> DataFrame:
        base = copy_if_pandas(base, deep=False)
        sort_idxs = maybe_compute_sort_indices(base, id_col, time_col)
        if sort_idxs is not None:
            base = take_rows(base, sort_idxs)
            X = take_rows(X, sort_idxs)
            y = y[sort_idxs]
        if max_horizon is None:
            fitted_values = assign_columns(base, target_col, y)
            for name, model in self.models_.items():
                assert not isinstance(model, list)  # mypy
                preds = model.predict(X)
                fitted_values = assign_columns(fitted_values, name, preds)
            fitted_values = self._invert_transforms_fitted(fitted_values)
        else:
            horizon_fitted_values = []
            for horizon in range(max_horizon):
                horizon_base = copy_if_pandas(base, deep=True)
                horizon_base = assign_columns(horizon_base, target_col, y[:, horizon])
                horizon_fitted_values.append(horizon_base)
            for name, horizon_models in self.models_.items():
                for horizon, model in enumerate(horizon_models):
                    preds = model.predict(X)
                    horizon_fitted_values[horizon] = assign_columns(
                        horizon_fitted_values[horizon], name, preds
                    )
            for horizon, horizon_df in enumerate(horizon_fitted_values):
                keep_mask = ~is_nan(horizon_df[target_col])
                horizon_df = filter_with_mask(horizon_df, keep_mask)
                horizon_df = copy_if_pandas(horizon_df, deep=True)
                horizon_df = self._invert_transforms_fitted(horizon_df)
                horizon_df = assign_columns(horizon_df, "h", horizon + 1)
                horizon_fitted_values[horizon] = horizon_df
            fitted_values = vertical_concat(
                horizon_fitted_values, match_categories=False
            )
        if self.ts.target_transforms is not None:
            for tfm in self.ts.target_transforms[::-1]:
                if hasattr(tfm, "store_fitted"):
                    tfm.store_fitted = False
                if hasattr(tfm, "fitted_"):
                    tfm.fitted_ = []
        return fitted_values

    def fit(
        self,
        df: DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        prediction_intervals: Optional[PredictionIntervals] = None,
        fitted: bool = False,
        as_numpy: bool = False,
    ) -> "MLForecast":
        """Apply the feature engineering and train the models.

        Parameters
        ----------
        df : pandas or polars DataFrame
            Series data in long format.
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
                If `None`, will consider all columns (except id_col and time_col) as static.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
        max_horizon : int, optional (default=None)
            Train this many models, where each model will predict a specific horizon.
        prediction_intervals : PredictionIntervals, optional (default=None)
            Configuration to calibrate prediction intervals (Conformal Prediction).
        fitted : bool (default=False)
            Save in-sample predictions.
        as_numpy : bool (default = False)
            Cast features to numpy array.

        Returns
        -------
        self : MLForecast
            Forecast object with series values and trained models.
        """
        if fitted and self.ts.target_transforms is not None:
            for tfm in self.ts.target_transforms:
                if hasattr(tfm, "store_fitted"):
                    tfm.store_fitted = True
        self._cs_df: Optional[DataFrame] = None
        if prediction_intervals is not None:
            self.prediction_intervals = prediction_intervals
            self._cs_df = self._conformity_scores(
                df=df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static_features=static_features,
                dropna=dropna,
                keep_last_n=keep_last_n,
                n_windows=prediction_intervals.n_windows,
                h=prediction_intervals.h,
                as_numpy=as_numpy,
            )
        prep = self.preprocess(
            df=df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            max_horizon=max_horizon,
            return_X_y=not fitted,
            as_numpy=as_numpy,
        )
        if isinstance(prep, tuple):
            X, y = prep
        else:
            base = prep[[id_col, time_col]]
            X, y = self._extract_X_y(prep, target_col)
            if as_numpy:
                X = to_numpy(X)
            del prep
        self.fit_models(X, y)
        if fitted:
            fitted_values = self._compute_fitted_values(
                base=base,
                X=X,
                y=y,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                max_horizon=max_horizon,
            )
            fitted_values = drop_index_if_pandas(fitted_values)
            self.fcst_fitted_values_ = fitted_values
        return self

    def forecast_fitted_values(self):
        """Access in-sample predictions."""
        if not hasattr(self, "fcst_fitted_values_"):
            raise Exception("Please run the `fit` method using `fitted=True`")
        return self.fcst_fitted_values_

    def predict(
        self,
        h: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        new_df: Optional[DataFrame] = None,
        level: Optional[List[Union[int, float]]] = None,
        X_df: Optional[DataFrame] = None,
        ids: Optional[List[str]] = None,
    ) -> DataFrame:
        """Compute the predictions for the next `h` steps.

        Parameters
        ----------
        h : int
            Number of periods to predict.
        before_predict_callback : callable, optional (default=None)
            Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index.
        after_predict_callback : callable, optional (default=None)
            Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index.
        new_df : pandas or polars DataFrame, optional (default=None)
            Series data of new observations for which forecasts are to be generated.
                This dataframe should have the same structure as the one used to fit the model, including any features and time series data.
                If `new_df` is not None, the method will generate forecasts for the new observations.
        level : list of ints or floats, optional (default=None)
            Confidence levels between 0 and 100 for prediction intervals.
        X_df : pandas or polars DataFrame, optional (default=None)
            Dataframe with the future exogenous features. Should have the id column and the time column.
        ids : list of str, optional (default=None)
            List with subset of ids seen during training for which the forecasts should be computed.

        Returns
        -------
        result : pandas or polars DataFrame
            Predictions for each serie and timestep, with one column per model.
        """
        if not hasattr(self, "models_"):
            raise ValueError(
                "No fitted models found. You have to call fit or preprocess + fit_models. "
                "If you used cross_validation before please fit again."
            )
        first_model_is_list = isinstance(next(iter(self.models_.values())), list)
        max_horizon = self.ts.max_horizon
        if first_model_is_list and max_horizon is None:
            raise ValueError(
                "Found one model per horizon but `max_horizon` is None. "
                "If you ran preprocess after fit please run fit again."
            )
        elif not first_model_is_list and max_horizon is not None:
            raise ValueError(
                "Found a single model for all horizons "
                f"but `max_horizon` is {max_horizon}. "
                "If you ran preprocess after fit please run fit again."
            )

        if new_df is not None:
            new_ts = TimeSeries(
                freq=self.ts.freq,
                lags=self.ts.lags,
                lag_transforms=self.ts.lag_transforms,
                date_features=self.ts.date_features,
                num_threads=self.ts.num_threads,
                target_transforms=self.ts.target_transforms,
            )
            new_ts._fit(
                new_df,
                id_col=self.ts.id_col,
                time_col=self.ts.time_col,
                target_col=self.ts.target_col,
                static_features=self.ts.static_features,
                keep_last_n=self.ts.keep_last_n,
            )
            new_ts.max_horizon = self.ts.max_horizon
            new_ts.as_numpy = self.ts.as_numpy
            ts = new_ts
        else:
            ts = self.ts

        forecasts = ts.predict(
            models=self.models_,
            horizon=h,
            before_predict_callback=before_predict_callback,
            after_predict_callback=after_predict_callback,
            X_df=X_df,
            ids=ids,
        )
        if level is not None:
            if self._cs_df is None:
                warn_msg = (
                    "Please rerun the `fit` method passing a proper value "
                    "to prediction intervals to compute them."
                )
                warnings.warn(warn_msg, UserWarning)
            else:
                if (self.prediction_intervals.h != 1) and (
                    self.prediction_intervals.h < h
                ):
                    raise ValueError(
                        "The `h` argument of PredictionIntervals "
                        "should be equal to one or greater or equal to `h`. "
                        "Please rerun the `fit` method passing a proper value "
                        "to prediction intervals."
                    )
                if self.prediction_intervals.h == 1 and h > 1:
                    warn_msg = (
                        "Prediction intervals are calculated using 1-step ahead cross-validation, "
                        "with a constant width for all horizons. To vary the error by horizon, "
                        "pass PredictionIntervals(h=h) to the `prediction_intervals` "
                        "argument when refitting the model."
                    )
                    warnings.warn(warn_msg, UserWarning)
                level_ = sorted(level)
                model_names = self.models.keys()
                conformal_method = _get_conformal_method(
                    self.prediction_intervals.method
                )
                if ids is not None:
                    ids_mask = is_in(self._cs_df[self.ts.id_col], ids)
                    cs_df = filter_with_mask(self._cs_df, ids_mask)
                    n_series = len(ids)
                else:
                    cs_df = self._cs_df
                    n_series = self.ts.ga.n_groups
                forecasts = conformal_method(
                    forecasts,
                    cs_df,
                    model_names=list(model_names),
                    level=level_,
                    cs_h=self.prediction_intervals.h,
                    cs_n_windows=self.prediction_intervals.n_windows,
                    n_series=n_series,
                    horizon=h,
                )
        return forecasts

    def cross_validation(
        self,
        df: DataFrame,
        n_windows: int,
        h: int,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        step_size: Optional[int] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        refit: Union[bool, int] = True,
        max_horizon: Optional[int] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        prediction_intervals: Optional[PredictionIntervals] = None,
        level: Optional[List[Union[int, float]]] = None,
        input_size: Optional[int] = None,
        fitted: bool = False,
        as_numpy: bool = False,
    ) -> DataFrame:
        """Perform time series cross validation.
        Creates `n_windows` splits where each window has `h` test periods,
        trains the models, computes the predictions and merges the actuals.

        Parameters
        ----------
        df : pandas or polars DataFrame
            Series data in long format.
        n_windows : int
            Number of windows to evaluate.
        h : int
            Forecast horizon.
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
        max_horizon: int, optional (default=None)
            Train this many models, where each model will predict a specific horizon.
        refit : bool or int (default=True)
            Retrain model for each cross validation window.
            If False, the models are trained at the beginning and then used to predict each window.
            If positive int, the models are retrained every `refit` windows.
        before_predict_callback : callable, optional (default=None)
            Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index.
        after_predict_callback : callable, optional (default=None)
            Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index.
        prediction_intervals : PredictionIntervals, optional (default=None)
            Configuration to calibrate prediction intervals (Conformal Prediction).
        level : list of ints or floats, optional (default=None)
            Confidence levels between 0 and 100 for prediction intervals.
        input_size : int, optional (default=None)
            Maximum training samples per serie in each window. If None, will use an expanding window.
        fitted : bool (default=False)
            Store the in-sample predictions.
        as_numpy : bool (default = False)
            Cast features to numpy array.

        Returns
        -------
        result : pandas or polars DataFrame
            Predictions for each window with the series id, timestamp, last train date, target value and predictions from each model.
        """
        results = []
        self.cv_models_ = []
        self.ts._validate_freq(df, time_col)
        splits = backtest_splits(
            df,
            n_windows=n_windows,
            h=h,
            id_col=id_col,
            time_col=time_col,
            freq=self.freq,
            step_size=step_size,
            input_size=input_size,
        )
        self.cv_fitted_values_ = []
        for i_window, (cutoffs, train, valid) in enumerate(splits):
            should_fit = i_window == 0 or (refit > 0 and i_window % refit == 0)
            if should_fit:
                self.fit(
                    train,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    static_features=static_features,
                    dropna=dropna,
                    keep_last_n=keep_last_n,
                    max_horizon=max_horizon,
                    prediction_intervals=prediction_intervals,
                    fitted=fitted,
                    as_numpy=as_numpy,
                )
                self.cv_models_.append(self.models_)
                if fitted:
                    self.cv_fitted_values_.append(
                        assign_columns(self.fcst_fitted_values_, "fold", i_window)
                    )
            if fitted and not should_fit:
                if self.ts.target_transforms is not None:
                    for tfm in self.ts.target_transforms:
                        if hasattr(tfm, "store_fitted"):
                            tfm.store_fitted = True
                prep = self.preprocess(
                    train,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    static_features=static_features,
                    dropna=dropna,
                    keep_last_n=keep_last_n,
                    max_horizon=max_horizon,
                    return_X_y=False,
                )
                assert not isinstance(prep, tuple)
                base = prep[[id_col, time_col]]
                train_X, train_y = self._extract_X_y(prep, target_col)
                if as_numpy:
                    train_X = to_numpy(train_X)
                del prep
                fitted_values = self._compute_fitted_values(
                    base=base,
                    X=train_X,
                    y=train_y,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    max_horizon=max_horizon,
                )
                fitted_values = assign_columns(fitted_values, "fold", i_window)
                self.cv_fitted_values_.append(fitted_values)
            static = [c for c in self.ts.static_features_.columns if c != id_col]
            dynamic = [
                c
                for c in valid.columns
                if c not in static + [id_col, time_col, target_col]
            ]
            if dynamic:
                X_df: Optional[DataFrame] = valid.drop(columns=static + [target_col])
            else:
                X_df = None
            y_pred = self.predict(
                h=h,
                before_predict_callback=before_predict_callback,
                after_predict_callback=after_predict_callback,
                new_df=train if not should_fit else None,
                level=level,
                X_df=X_df,
            )
            y_pred = join(y_pred, cutoffs, on=id_col, how="left")
            result = join(
                valid[[id_col, time_col, target_col]],
                y_pred,
                on=[id_col, time_col],
            )
            sort_idxs = maybe_compute_sort_indices(result, id_col, time_col)
            if sort_idxs is not None:
                result = take_rows(result, sort_idxs)
            if result.shape[0] < valid.shape[0]:
                raise ValueError(
                    "Cross validation result produced less results than expected. "
                    "Please verify that the frequency set on the MLForecast constructor matches your series' "
                    "and that there aren't any missing periods."
                )
            results.append(result)
        del self.models_
        out = vertical_concat(results, match_categories=False)
        out = drop_index_if_pandas(out)
        first_out_cols = [id_col, time_col, "cutoff", target_col]
        remaining_cols = [c for c in out.columns if c not in first_out_cols]
        return out[first_out_cols + remaining_cols]

    def cross_validation_fitted_values(self):
        if not getattr(self, "cv_fitted_values_", []):
            raise ValueError("Please run cross_validation with fitted=True first.")
        out = vertical_concat(self.cv_fitted_values_, match_categories=False)
        first_out_cols = [self.ts.id_col, self.ts.time_col, "fold", self.ts.target_col]
        remaining_cols = [c for c in out.columns if c not in first_out_cols]
        out = drop_index_if_pandas(out)
        return out[first_out_cols + remaining_cols]
