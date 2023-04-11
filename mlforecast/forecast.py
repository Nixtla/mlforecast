# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/forecast.ipynb.

# %% auto 0
__all__ = ['MLForecast']

# %% ../nbs/forecast.ipynb 3
import copy
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from mlforecast.core import (
    DateFeature,
    Freq,
    LagTransforms,
    Lags,
    Models,
    TimeSeries,
    _name_models,
)

if TYPE_CHECKING:
    from mlforecast.lgb_cv import LightGBMCV
from .target_transforms import BaseTargetTransform
from .utils import backtest_splits, PredictionIntervals

# %% ../nbs/forecast.ipynb 6
def _add_conformal_distribution_intervals(
    fcst_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_window_size: int,
    n_series: int,
) -> pd.DataFrame:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This strategy creates forecasts paths
    based on errors and calculate quantiles using those paths.
    """
    fcst_df = fcst_df.copy()
    alphas = [100 - lv for lv in level]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    for model in model_names:
        scores = cs_df[model].values.reshape(cs_n_windows, n_series, cs_window_size)
        mean = fcst_df[model].values.reshape(1, n_series, -1)
        scores = np.vstack([mean - scores, mean + scores])
        quantiles = np.quantile(
            scores,
            cuts,
            axis=0,
        )
        quantiles = quantiles.reshape(len(cuts), -1)
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        out_cols = lo_cols + hi_cols
        for i, col in enumerate(out_cols):
            fcst_df[col] = quantiles[i]
    return fcst_df

# %% ../nbs/forecast.ipynb 7
def _add_conformal_error_intervals(
    fcst_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_window_size: int,
    n_series: int,
) -> pd.DataFrame:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This startegy creates prediction intervals
    based on the absolute errors.
    """
    fcst_df = fcst_df.copy()
    cuts = [lv / 100 for lv in level]
    for model in model_names:
        mean = fcst_df[model].values.ravel()
        quantiles = np.quantile(
            cs_df[model].values.reshape(cs_n_windows, n_series, cs_window_size),
            cuts,
            axis=0,
        )
        quantiles = quantiles.reshape(len(cuts), -1)
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        for i, col in enumerate(lo_cols):
            fcst_df[col] = mean - quantiles[len(level) - 1 - i]
        for i, col in enumerate(hi_cols):
            fcst_df[col] = mean + quantiles[i]
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
        freq: Optional[Freq] = None,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        differences: Optional[Iterable[int]] = None,
        num_threads: int = 1,
        target_transforms: Optional[List[BaseTargetTransform]] = None,
    ):
        """Create forecast object

        Parameters
        ----------
        models : regressor or list of regressors
            Models that will be trained and used to compute the forecasts.
        freq : str or int or pd.offsets.BaseOffset, optional (default=None)
            Pandas offset, pandas offset alias, e.g. 'D', 'W-THU' or integer denoting the frequency of the series.
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
            differences=differences,
            num_threads=num_threads,
            target_transforms=target_transforms,
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
            lgb.LGBMRegressor(**{**cv.params, "n_estimators": cv.best_iteration_})
        )
        fcst.ts = copy.deepcopy(cv.ts)
        return fcst

    def preprocess(
        self,
        data: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        return_X_y: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]]:
        """Add the features to `data`.

        Parameters
        ----------
        data : pandas DataFrame
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
        max_horizon: int, optional (default=None)
            Train this many models, where each model will predict a specific horizon.
        return_X_y: bool (default=False)
            Return a tuple with the features and the target. If False will return a single dataframe.

        Returns
        -------
        result : pandas DataFrame or tuple of pandas Dataframe and either a pandas Series or a pandas Dataframe (for multi-output regression).
            `data` plus added features and target(s).
        """
        return self.ts.fit_transform(
            data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            max_horizon=max_horizon,
            return_X_y=return_X_y,
        )

    def fit_models(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
    ) -> "MLForecast":
        """Manually train models. Use this if you called `Forecast.preprocess` beforehand.

        Parameters
        ----------
        X : pandas DataFrame
            Features.
        y : pandas Series or pandas DataFrame (multi-output).
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
                for col in y:
                    keep = y[col].notnull()
                    self.models_[name].append(
                        clone(model).fit(X.loc[keep], y.loc[keep, col])
                    )
            else:
                self.models_[name] = clone(model).fit(X, y)
        return self

    def _conformity_scores(
        self,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        n_windows: int = 2,
        window_size: int = 1,
    ):
        """Compute conformity scores.

        We need at least two cross validation errors to compute
        quantiles for prediction intervals (`n_windows=2`).

        The exception is raised by the PredictionIntervals data class.

        In this simplest case, we assume the width of the interval
        is the same for all the forecasting horizon (`window_size=1`).
        """
        cv_results = self.cross_validation(
            data=data,
            n_windows=n_windows,
            window_size=window_size,
            refit=False,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            max_horizon=max_horizon,
            prediction_intervals=None,
        )
        # conformity score for each model
        for model in self.models.keys():
            # compute absolute error for each model
            cv_results[model] = np.abs(cv_results[model] - cv_results[target_col])
        return cv_results.drop("y", axis=1)

    def fit(
        self,
        data: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        prediction_intervals: Optional[PredictionIntervals] = None,
    ) -> "MLForecast":
        """Apply the feature engineering and train the models.

        Parameters
        ----------
        data : pandas DataFrame
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
        max_horizon: int, optional (default=None)
            Train this many models, where each model will predict a specific horizon.
        prediction_intervals : PredictionIntervals, optional (default=None)
            Configuration to calibrate prediction intervals (Conformal Prediction).

        Returns
        -------
        self : MLForecast
            Forecast object with series values and trained models.
        """
        self._cs_df: Optional[pd.DataFrame] = None
        if prediction_intervals is not None:
            self.prediction_intervals = prediction_intervals
            self._cs_df = self._conformity_scores(
                data=data,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static_features=static_features,
                dropna=dropna,
                keep_last_n=keep_last_n,
                n_windows=prediction_intervals.n_windows,
                window_size=prediction_intervals.window_size,
            )
        X, y = self.preprocess(
            data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            max_horizon=max_horizon,
            return_X_y=True,
        )
        X = X[self.ts.features_order_]
        return self.fit_models(X, y)

    def predict(
        self,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        new_data: Optional[pd.DataFrame] = None,
        level: Optional[List[Union[int, float]]] = None,
    ) -> pd.DataFrame:
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
        new_data : pandas DataFrame, optional (default=None)
            Series data of new observations for which forecasts are to be generated.
                This dataframe should have the same structure as the one used to fit the model, including any features and time series data.
                If `new_data` is not None, the method will generate forecasts for the new observations.
        level : list of ints or floats, optional (default=None)
            Confidence levels between 0 and 100 for prediction intervals.

        Returns
        -------
        result : pandas DataFrame
            Predictions for each serie and timestep, with one column per model.
        """
        if not hasattr(self, "models_"):
            raise ValueError(
                "No fitted models found. You have to call fit or preprocess + fit_models."
            )

        if new_data is not None:
            new_ts = TimeSeries(
                freq=self.ts.freq,
                lags=self.ts.lags,
                lag_transforms=self.ts.lag_transforms,
                date_features=self.ts.date_features,
                num_threads=self.ts.num_threads,
                target_transforms=self.ts.target_transforms,
            )
            new_ts._fit(
                new_data,
                id_col=self.ts.id_col,
                time_col=self.ts.time_col,
                target_col=self.ts.target_col,
                static_features=self.ts.static_features.columns,
                keep_last_n=self.ts.keep_last_n,
            )
            new_ts.max_horizon = self.ts.max_horizon
            ts = new_ts
        else:
            ts = self.ts

        forecasts = ts.predict(
            self.models_,
            horizon,
            dynamic_dfs,
            before_predict_callback,
            after_predict_callback,
        )
        if level is not None:
            if self._cs_df is None:
                warn_msg = (
                    "Please rerun the `fit` method passing a proper value "
                    "to prediction intervals to compute them."
                )
                warnings.warn(warn_msg, UserWarning)
            else:
                if self.prediction_intervals.window_size not in [1, horizon]:
                    raise ValueError(
                        "The `window_size` argument of PredictionIntervals "
                        "should be equal to one or `horizon`. "
                        "Please rerun the `fit` method passing a proper value "
                        "to prediction intervals."
                    )
                if self.prediction_intervals.window_size != horizon:
                    warn_msg = (
                        "Prediction intervals are calculated using 1-step ahead cross-validation, "
                        "with a constant width for all horizons. To vary the error by horizon, "
                        "pass PredictionIntervals(window_size=horizon) to the `prediction_intervals` "
                        "argument when refitting the model."
                    )
                    warnings.warn(warn_msg, UserWarning)
                level_ = sorted(level)
                model_names = self.models.keys()
                conformal_method = _get_conformal_method(
                    self.prediction_intervals.method
                )
                forecasts = conformal_method(
                    forecasts,
                    self._cs_df,
                    model_names=list(model_names),
                    level=level_,
                    cs_window_size=self.prediction_intervals.window_size,
                    cs_n_windows=self.prediction_intervals.n_windows,
                    n_series=self.ts.ga.ngroups,
                )
        return forecasts

    def cross_validation(
        self,
        data: pd.DataFrame,
        n_windows: int,
        window_size: int,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        step_size: Optional[int] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        refit: bool = True,
        max_horizon: Optional[int] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        prediction_intervals: Optional[PredictionIntervals] = None,
        level: Optional[List[Union[int, float]]] = None,
        input_size: Optional[int] = None,
    ):
        """Perform time series cross validation.
        Creates `n_windows` splits where each window has `window_size` test periods,
        trains the models, computes the predictions and merges the actuals.

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
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
        max_horizon: int, optional (default=None)
            Train this many models, where each model will predict a specific horizon.
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
        prediction_intervals : PredictionIntervals, optional (default=None)
            Configuration to calibrate prediction intervals (Conformal Prediction).
        level : list of ints or floats, optional (default=None)
            Confidence levels between 0 and 100 for prediction intervals.
        input_size : int, optional (default=None)
            Maximum training samples per serie in each window. If None, will use an expanding window.

        Returns
        -------
        result : pandas DataFrame
            Predictions for each window with the series id, timestamp, last train date, target value and predictions from each model.
        """
        if hasattr(self, "models_"):
            warnings.warn(
                "Excuting `cross_validation` after `fit` can produce unexpected errors"
            )
        results = []
        self.cv_models_ = []
        if np.issubdtype(data[time_col].dtype.type, np.integer):
            freq = 1
        else:
            freq = self.freq

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
        ex_cols_to_drop = [id_col, time_col, target_col]
        if static_features is not None:
            ex_cols_to_drop.extend(static_features)
        has_ex = not data.columns.drop(ex_cols_to_drop).empty
        for i_window, (cutoffs, train, valid) in enumerate(splits):
            if refit or i_window == 0:
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
                )
            self.cv_models_.append(self.models_)
            dynamic_dfs = [valid.drop(columns=[target_col])] if has_ex else None
            y_pred = self.predict(
                window_size,
                dynamic_dfs,
                before_predict_callback,
                after_predict_callback,
                new_data=train if not refit else None,
                level=level,
            )
            y_pred = y_pred.merge(cutoffs, on=id_col, how="left")
            result = valid[[id_col, time_col, target_col]].merge(
                y_pred, on=[id_col, time_col]
            )
            results.append(result)
        out = pd.concat(results)
        cols_order = [id_col, time_col, "cutoff", target_col]
        return out[cols_order + out.columns.drop(cols_order).tolist()]
