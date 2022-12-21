# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/forecast.ipynb.

# %% auto 0
__all__ = ['MLForecast', 'Forecast']

# %% ../nbs/forecast.ipynb 3
import copy
import warnings
from typing import Callable, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone

from mlforecast.core import (
    DateFeature,
    Differences,
    Freq,
    LagTransforms,
    Lags,
    Models,
    TimeSeries,
    _name_models,
)
from .lgb_cv import LightGBMCV
from .utils import backtest_splits

# %% ../nbs/forecast.ipynb 6
class MLForecast:
    def __init__(
        self,
        models: Models,
        freq: Optional[Freq] = None,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        differences: Optional[Differences] = None,
        num_threads: int = 1,
    ):
        """Create forecast object

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
            freq, lags, lag_transforms, date_features, differences, num_threads
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
    def from_cv(cls, cv: LightGBMCV) -> "MLForecast":
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
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Add the features to `data`.

        Parameters
        ----------
        data : pandas DataFrame
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
        result : pandas DataFrame.
            `data` plus added features.
        """
        return self.ts.fit_transform(
            data, id_col, time_col, target_col, static_features, dropna, keep_last_n
        )

    def fit_models(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
    ) -> "MLForecast":
        """Manually train models. Use this if you called `Forecast.preprocess` beforehand.

        Parameters
        ----------
        X : pandas DataFrame
            Features.
        y : numpy array or pandas Series.
            Target.

        Returns
        -------
        self : Forecast
            Forecast object with trained models.
        """
        self.models_ = {}
        for name, model in self.models.items():
            self.models_[name] = clone(model).fit(X, y)
        return self

    def fit(
        self,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> "MLForecast":
        """Apply the feature engineering and train the models.

        Parameters
        ----------
        data : pandas DataFrame
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
        self : Forecast
            Forecast object with series values and trained models.
        """
        series_df = self.preprocess(
            data, id_col, time_col, target_col, static_features, dropna, keep_last_n
        )
        cols_to_drop = [time_col, target_col]
        if id_col != "index" and id_col not in self.ts.static_features:
            cols_to_drop.append(id_col)
        X, y = series_df.drop(columns=cols_to_drop), series_df[target_col].values
        del series_df
        return self.fit_models(X, y)

    def predict(
        self,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
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


        Returns
        -------
        result : pandas DataFrame
            Predictions for each serie and timestep, with one column per model.
        """
        if not hasattr(self, "models_"):
            raise ValueError(
                "No fitted models found. You have to call fit or preprocess + fit_models."
            )
        return self.ts.predict(
            self.models_,
            horizon,
            dynamic_dfs,
            before_predict_callback,
            after_predict_callback,
        )

    def cross_validation(
        self,
        data: pd.DataFrame,
        n_windows: int,
        window_size: int,
        id_col: str,
        time_col: str,
        target_col: str,
        step_size: int = 1,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
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
        id_col : str
            Column that identifies each serie. If 'index' then the index is used.
        time_col : str
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str
            Column that contains the target.
        step_size : int, optional (default=None)
            Step size between each cross validation window.
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
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
            Predictions for each window with the series id, timestamp, last train date, target value and predictions from each model.
        """
        if window_size != step_size:
            warning_msg = (
                "The gap between each cross validation window (controled by `step_size`) "
                "changed from being equal to the `window_size` to being equal to 1. "
                "Please set `step_size` equal to the value of `window_size` to mantain "
                "the old behavior."
            )
            warnings.warn(warning_msg, RuntimeWarning)

        results = []
        self.cv_models_ = []
        if id_col != "index":
            data = data.set_index(id_col)

        if np.issubdtype(data[time_col].dtype.type, np.integer):
            freq = 1
        else:
            freq = self.freq

        for train_end, train, valid in backtest_splits(
            data, n_windows, window_size, freq, step_size, time_col
        ):
            self.fit(
                train,
                "index",
                time_col,
                target_col,
                static_features,
                dropna,
                keep_last_n,
            )
            self.cv_models_.append(self.models_)
            y_pred = self.predict(
                window_size,
                dynamic_dfs,
                before_predict_callback,
                after_predict_callback,
            )
            y_pred = y_pred.set_index(time_col, append=True)
            result = valid.set_index(time_col, append=True)[[target_col]].copy()
            result = result.join(y_pred).reset_index(time_col)
            result["cutoff"] = train_end
            results.append(result)

        out = pd.concat(results)
        out = out[[time_col, "cutoff", target_col, *y_pred.columns]]
        if id_col != "index":
            out = out.reset_index()
        return out

# %% ../nbs/forecast.ipynb 9
class Forecast(MLForecast):
    def __init__(
        self,
        models: Models,
        freq: Optional[Freq] = None,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        differences: Optional[Differences] = None,
        num_threads: int = 1,
    ):
        warning_msg = (
            "The Forecast class is deprecated and will be removed in a future version, "
            "please use the MLForecast class instead."
        )
        warnings.warn(warning_msg, DeprecationWarning)
        super().__init__(
            models, freq, lags, lag_transforms, date_features, differences, num_threads
        )
