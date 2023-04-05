# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/core.ipynb.

# %% auto 0
__all__ = ['TimeSeries']

# %% ../nbs/core.ipynb 3
import concurrent.futures
import inspect
import reprlib
import warnings
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit
from sklearn.base import BaseEstimator
from window_ops.shift import shift_array

from .grouped_array import GroupedArray
from .target_transforms import BaseTargetTransform, Differences

# %% ../nbs/core.ipynb 10
date_features_dtypes = {
    "year": np.uint16,
    "month": np.uint8,
    "day": np.uint8,
    "hour": np.uint8,
    "minute": np.uint8,
    "second": np.uint8,
    "dayofyear": np.uint16,
    "day_of_year": np.uint16,
    "weekofyear": np.uint8,
    "week": np.uint8,
    "dayofweek": np.uint8,
    "day_of_week": np.uint8,
    "weekday": np.uint8,
    "quarter": np.uint8,
    "daysinmonth": np.uint8,
    "is_month_start": np.uint8,
    "is_month_end": np.uint8,
    "is_quarter_start": np.uint8,
    "is_quarter_end": np.uint8,
    "is_year_start": np.uint8,
    "is_year_end": np.uint8,
}

# %% ../nbs/core.ipynb 11
def _build_transform_name(lag, tfm, *args) -> str:
    """Creates a name for a transformation based on `lag`, the name of the function and its arguments."""
    tfm_name = f"{tfm.__name__}_lag{lag}"
    func_params = inspect.signature(tfm).parameters
    func_args = list(func_params.items())[1:]  # remove input array argument
    changed_params = [
        f"{name}{value}"
        for value, (name, arg) in zip(args, func_args)
        if arg.default != value
    ]
    if changed_params:
        tfm_name += "_" + "_".join(changed_params)
    return tfm_name

# %% ../nbs/core.ipynb 13
def _name_models(current_names):
    ctr = Counter(current_names)
    if not ctr:
        return []
    if max(ctr.values()) < 2:
        return current_names
    names = current_names.copy()
    for i, x in enumerate(reversed(current_names), start=1):
        count = ctr[x]
        if count > 1:
            name = f"{x}{count}"
            ctr[x] -= 1
        else:
            name = x
        names[-i] = name
    return names

# %% ../nbs/core.ipynb 15
@njit
def _identity(x: np.ndarray) -> np.ndarray:
    """Do nothing to the input."""
    return x


def _as_tuple(x):
    """Return a tuple from the input."""
    if isinstance(x, tuple):
        return x
    return (x,)


@njit
def _expand_target(data, indptr, max_horizon):
    out = np.empty((data.size, max_horizon), dtype=data.dtype)
    n_series = len(indptr) - 1
    n = 0
    for i in range(n_series):
        serie = data[indptr[i] : indptr[i + 1]]
        for j in range(serie.size):
            upper = min(serie.size - j, max_horizon)
            for k in range(upper):
                out[n, k] = serie[j + k]
            for k in range(upper, max_horizon):
                out[n, k] = np.nan
            n += 1
    return out

# %% ../nbs/core.ipynb 16
Freq = Union[int, str, pd.offsets.BaseOffset]
Lags = Iterable[int]
LagTransform = Union[Callable, Tuple[Callable, Any]]
LagTransforms = Dict[int, List[LagTransform]]
DateFeature = Union[str, Callable]
Models = Union[BaseEstimator, List[BaseEstimator], Dict[str, BaseEstimator]]

# %% ../nbs/core.ipynb 17
class TimeSeries:
    """Utility class for storing and transforming time series data."""

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
        if isinstance(freq, str):
            self.freq = pd.tseries.frequencies.to_offset(freq)
        elif isinstance(freq, pd.offsets.BaseOffset):
            self.freq = freq
        elif isinstance(freq, int):
            self.freq = freq
        elif freq is None:
            self.freq = 1
        else:
            raise ValueError(
                "Unknown frequency type "
                "Please use a str, int or offset frequency type."
            )
        if not isinstance(num_threads, int) or num_threads < 1:
            warnings.warn("Setting num_threads to 1.")
            num_threads = 1
        self.lags = [] if lags is None else list(lags)
        self.lag_transforms = {} if lag_transforms is None else lag_transforms
        self.date_features = [] if date_features is None else list(date_features)
        if differences is not None:
            warnings.warn(
                "The differences argument is deprecated and will be removed in a future version.\n"
                "Please use pass an `mlforecast.target_transforms.Differences` instance to the `target_transforms` argument instead."
                ""
            )
            if target_transforms is None:
                target_transforms = [Differences(differences)]
            else:
                target_transforms = target_transforms + [Differences(differences)]
        self.num_threads = num_threads
        self.target_transforms = target_transforms
        for feature in self.date_features:
            if callable(feature) and feature.__name__ == "<lambda>":
                raise ValueError(
                    "Can't use a lambda as a date feature because the function name gets used as the feature name."
                )

        self.transforms: Dict[str, Tuple[Any, ...]] = OrderedDict()
        for lag in self.lags:
            self.transforms[f"lag{lag}"] = (lag, _identity)
        for lag in self.lag_transforms.keys():
            for tfm_args in self.lag_transforms[lag]:
                tfm, *args = _as_tuple(tfm_args)
                tfm_name = _build_transform_name(lag, tfm, *args)
                self.transforms[tfm_name] = (lag, tfm, *args)

        self.ga: GroupedArray

    @property
    def _date_feature_names(self):
        return [f.__name__ if callable(f) else f for f in self.date_features]

    @property
    def features(self) -> List[str]:
        """Names of all computed features."""
        return list(self.transforms.keys()) + self._date_feature_names

    def __repr__(self):
        return (
            f"TimeSeries(freq={self.freq}, "
            f"transforms={list(self.transforms.keys())}, "
            f"date_features={self._date_feature_names}, "
            f"num_threads={self.num_threads})"
        )

    def _fit(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        keep_last_n: Optional[int] = None,
    ) -> "TimeSeries":
        """Save the series values, ids and last dates."""
        for col in (id_col, time_col, target_col):
            if col not in df:
                raise ValueError(f"Data doesn't contain {col} column")
        if df[target_col].isnull().any():
            raise ValueError(f"{target_col} column contains null values.")
        if pd.api.types.is_datetime64_dtype(df[time_col]):
            if self.freq == 1:
                raise ValueError(
                    "Must set frequency when using a timestamp type column."
                )
        elif np.issubdtype(df[time_col].dtype.type, np.integer):
            if self.freq != 1:
                warnings.warn("Setting `freq=1` since time col is int.")
                self.freq = 1
        else:
            raise ValueError(f"{time_col} must be either timestamp or integer.")
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col
        if static_features is None:
            static_features = df.columns.drop([id_col, time_col, target_col])
        elif id_col in static_features:
            raise ValueError(
                "Cannot use the id_col as a static feature. Please create a separate column."
            )
        self.static_features = (
            df.set_index(id_col)[static_features].groupby(id_col, observed=True).head(1)
        )
        sort_idxs = pd.core.sorting.lexsort_indexer([df[id_col], df[time_col]])
        self.restore_idxs = np.empty(df.shape[0], dtype=np.int32)
        self.restore_idxs[sort_idxs] = np.arange(df.shape[0])
        sorted_df = df[[id_col, time_col, target_col]].iloc[sort_idxs]
        if self.target_transforms is not None:
            for tfm in self.target_transforms:
                tfm.set_column_names(id_col, time_col, target_col)
                sorted_df = tfm.fit_transform(sorted_df)
        sorted_df = sorted_df.set_index([id_col, time_col])
        self.uids = sorted_df.index.unique(level=0)
        self.ga = GroupedArray.from_sorted_df(sorted_df, id_col, target_col)
        self.features_ = self._compute_transforms()
        if keep_last_n is not None:
            self.ga = self.ga.take_from_groups(slice(-keep_last_n, None))
        self._ga = GroupedArray(self.ga.data, self.ga.indptr)
        self.last_dates = sorted_df.index.get_level_values(self.time_col)[
            self.ga.indptr[1:] - 1
        ]
        self.features_order_ = (
            df.columns.drop([id_col, time_col, target_col]).tolist() + self.features
        )
        return self

    def _apply_transforms(self, updates_only: bool = False) -> Dict[str, np.ndarray]:
        """Apply the transformations using the main process.

        If `updates_only` then only the updates are returned.
        """
        results = {}
        offset = 1 if updates_only else 0
        for tfm_name, (lag, tfm, *args) in self.transforms.items():
            results[tfm_name] = self.ga.transform_series(
                updates_only, lag - offset, tfm, *args
            )
        return results

    def _apply_multithreaded_transforms(
        self, updates_only: bool = False
    ) -> Dict[str, np.ndarray]:
        """Apply the transformations using multithreading.

        If `updates_only` then only the updates are returned.
        """
        future_to_result = {}
        results = {}
        offset = 1 if updates_only else 0
        with concurrent.futures.ThreadPoolExecutor(self.num_threads) as executor:
            for tfm_name, (lag, tfm, *args) in self.transforms.items():
                future = executor.submit(
                    self.ga.transform_series,
                    updates_only,
                    lag - offset,
                    tfm,
                    *args,
                )
                future_to_result[future] = tfm_name
            for future in concurrent.futures.as_completed(future_to_result):
                tfm_name = future_to_result[future]
                results[tfm_name] = future.result()
        return results

    def _compute_transforms(self) -> Dict[str, np.ndarray]:
        """Compute the transformations defined in the constructor.

        If `self.num_threads > 1` these are computed using multithreading."""
        if self.num_threads == 1 or len(self.transforms) == 1:
            return self._apply_transforms()
        return self._apply_multithreaded_transforms()

    def _compute_date_feature(self, dates, feature):
        if callable(feature):
            feat_name = feature.__name__
            feat_vals = feature(dates)
        else:
            feat_name = feature
            if feature in ("week", "weekofyear"):
                dates = dates.isocalendar()
            feat_vals = getattr(dates, feature)
        vals = np.asarray(feat_vals)
        feat_dtype = date_features_dtypes.get(feature)
        if feat_dtype is not None:
            vals = vals.astype(feat_dtype)
        return feat_name, vals

    def _transform(
        self,
        df: pd.DataFrame,
        dropna: bool = True,
        max_horizon: Optional[int] = None,
        return_X_y: bool = False,
    ) -> pd.DataFrame:
        """Add the features to `df`.

        if `dropna=True` then all the null rows are dropped."""
        modifies_target = bool(self.target_transforms)
        df = df.copy(deep=modifies_target and not return_X_y)

        # lag transforms
        for feat in self.transforms.keys():
            df[feat] = self.features_[feat][self.restore_idxs]

        # date features
        dates = df[self.time_col]
        if not np.issubdtype(dates.dtype.type, np.integer):
            dates = pd.DatetimeIndex(dates)
        for feature in self.date_features:
            feat_name, feat_vals = self._compute_date_feature(dates, feature)
            df[feat_name] = feat_vals

        # target
        self.max_horizon = max_horizon
        if max_horizon is None:
            if modifies_target:
                target = pd.Series(self.ga.data[self.restore_idxs], index=df.index)
            else:
                target = df[self.target_col]
        else:
            target = pd.DataFrame(
                self.ga.expand_target(max_horizon)[self.restore_idxs],
                index=df.index,
                columns=[f"{self.target_col}{i}" for i in range(max_horizon)],
            )

        # determine rows to keep
        if dropna:
            feature_nulls = df[self.features].isnull().any(axis=1)
            target_nulls = target.isnull()
            if target_nulls.ndim == 2:
                # target nulls for each horizon are dropped in MLForecast.fit_models
                # we just drop rows here for which all the target values are null
                target_nulls = target_nulls.all(axis=1)
            keep_rows = ~(feature_nulls | target_nulls).values
        else:
            keep_rows = np.full(df.shape[0], True)

        # assemble return
        xs = df.columns.drop(self.target_col)
        if return_X_y:
            return df.loc[keep_rows, xs], target.loc[keep_rows]
        if max_horizon is None:
            if modifies_target:
                df[self.target_col] = target
        else:
            df = pd.concat([df[xs], target], axis=1)
        return df.loc[keep_rows]

    def fit_transform(
        self,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        return_X_y: bool = False,
    ) -> pd.DataFrame:
        """Add the features to `data` and save the required information for the predictions step.

        If not all features are static, specify which ones are in `static_features`.
        If you don't want to drop rows with null values after the transformations set `dropna=False`
        If `keep_last_n` is not None then that number of observations is kept across all series for updates.
        """
        self.dropna = dropna
        self.keep_last_n = keep_last_n
        self._fit(data, id_col, time_col, target_col, static_features, keep_last_n)
        return self._transform(
            data, dropna=dropna, max_horizon=max_horizon, return_X_y=return_X_y
        )

    def _update_y(self, new: np.ndarray) -> None:
        """Appends the elements of `new` to every time serie.

        These values are used to update the transformations and are stored as predictions."""
        if not hasattr(self, "y_pred"):
            self.y_pred = []
        self.y_pred.append(new)
        new_arr = np.asarray(new)
        self.ga = self.ga.append(new_arr)

    def _update_features(self) -> pd.DataFrame:
        """Compute the current values of all the features using the latest values of the time series."""
        if not hasattr(self, "curr_dates"):
            self.curr_dates = self.last_dates.copy()
            self.test_dates = []
        self.curr_dates += self.freq
        self.test_dates.append(self.curr_dates)

        if self.num_threads == 1 or len(self.transforms) == 1:
            features = self._apply_transforms(updates_only=True)
        else:
            features = self._apply_multithreaded_transforms(updates_only=True)

        for feature in self.date_features:
            feat_name, feat_vals = self._compute_date_feature(self.curr_dates, feature)
            features[feat_name] = feat_vals

        features_df = pd.DataFrame(features, columns=self.features)
        features_df[self.id_col] = self.uids
        features_df[self.time_col] = self.curr_dates
        return self.static_features.merge(features_df, on=self.id_col)

    def _get_raw_predictions(self) -> np.ndarray:
        return np.array(self.y_pred).ravel("F")

    def _get_predictions(self) -> pd.DataFrame:
        """Get all the predicted values with their corresponding ids and datestamps."""
        n_preds = len(self.y_pred)
        uids = pd.Series(
            np.repeat(self.uids, n_preds), name=self.id_col, dtype=self.uids.dtype
        )
        df = pd.DataFrame(
            {
                self.id_col: uids,
                self.time_col: np.array(self.test_dates).ravel("F"),
                f"{self.target_col}_pred": self._get_raw_predictions(),
            },
        )
        return df

    def _predict_setup(self) -> None:
        self.curr_dates = self.last_dates.copy()
        self.test_dates = []
        self.y_pred = []
        self.ga = GroupedArray(self._ga.data, self._ga.indptr)

    def _get_features_for_next_step(self, dynamic_dfs):
        new_x = self._update_features()
        if dynamic_dfs:
            for df in dynamic_dfs:
                new_x = new_x.merge(df, how="left")
            new_x = new_x.sort_values(self.id_col)
        nulls = new_x.isnull().any()
        if any(nulls):
            warnings.warn(f'Found null values in {", ".join(nulls[nulls].index)}.')
        return new_x[self.features_order_]

    def _predict_recursive(
        self,
        models: Dict[str, BaseEstimator],
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """Use `model` to predict the next `horizon` timesteps."""
        if dynamic_dfs is None:
            dynamic_dfs = []
        for i, (name, model) in enumerate(models.items()):
            self._predict_setup()
            for _ in range(horizon):
                new_x = self._get_features_for_next_step(dynamic_dfs)
                if before_predict_callback is not None:
                    new_x = before_predict_callback(new_x)
                predictions = model.predict(new_x)
                if after_predict_callback is not None:
                    predictions_serie = pd.Series(predictions, index=self.uids)
                    predictions = after_predict_callback(predictions_serie).values
                self._update_y(predictions)
            if i == 0:
                preds = self._get_predictions()
                preds = preds.rename(
                    columns={f"{self.target_col}_pred": name}, copy=False
                )
            else:
                preds[name] = self._get_raw_predictions()
        return preds

    def _predict_multi(
        self,
        models: Dict[str, BaseEstimator],
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        assert self.max_horizon is not None
        if horizon > self.max_horizon:
            raise ValueError(
                f"horizon must be at most max_horizon ({self.max_horizon})"
            )
        if dynamic_dfs is None:
            dynamic_dfs = []
        uids = np.repeat(self.uids, horizon)
        dates = np.hstack(
            [
                date + (i + 1) * self.freq
                for date in self.last_dates
                for i in range(horizon)
            ]
        )
        result = pd.DataFrame({self.id_col: uids, self.time_col: dates})
        for name, model in models.items():
            self._predict_setup()
            new_x = self._get_features_for_next_step(dynamic_dfs)
            if before_predict_callback is not None:
                new_x = before_predict_callback(new_x)
            predictions = np.empty((new_x.shape[0], horizon))
            for i in range(horizon):
                predictions[:, i] = model[i].predict(new_x)
            raw_preds = predictions.ravel()
            result[name] = raw_preds
        return result

    def predict(
        self,
        models: Dict[str, Union[BaseEstimator, List[BaseEstimator]]],
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        if getattr(self, "max_horizon", None) is None:
            preds = self._predict_recursive(
                models,
                horizon,
                dynamic_dfs,
                before_predict_callback,
                after_predict_callback,
            )
        else:
            preds = self._predict_multi(
                models,
                horizon,
                dynamic_dfs,
                before_predict_callback,
            )
        if self.target_transforms is not None:
            for tfm in self.target_transforms[::-1]:
                preds = tfm.inverse_transform(preds)
        return preds
