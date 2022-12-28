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


@njit
def _append_new(data, indptr, new):
    """Append each value of new to each group in data formed by indptr."""
    n_series = len(indptr) - 1
    new_data = np.empty(data.size + new.size, dtype=data.dtype)
    new_indptr = indptr.copy()
    new_indptr[1:] += np.arange(1, n_series + 1)
    for i in range(n_series):
        prev_slice = slice(indptr[i], indptr[i + 1])
        new_slice = slice(new_indptr[i], new_indptr[i + 1] - 1)
        new_data[new_slice] = data[prev_slice]
        new_data[new_indptr[i + 1] - 1] = new[i]
    return new_data, new_indptr

# %% ../nbs/core.ipynb 11
@njit
def _identity(x: np.ndarray) -> np.ndarray:
    """Do nothing to the input."""
    return x


def _as_tuple(x):
    """Return a tuple from the input."""
    if isinstance(x, tuple):
        return x
    return (x,)


@njit(nogil=True)
def _transform_series(data, indptr, updates_only, lag, func, *args) -> np.ndarray:
    """Shifts every group in `data` by `lag` and computes `func(shifted, *args)`.

    If `updates_only=True` only last value of the transformation for each group is returned,
    otherwise the full transformation is returned"""
    n_series = len(indptr) - 1
    if updates_only:
        out = np.empty_like(data[:n_series])
        for i in range(n_series):
            lagged = shift_array(data[indptr[i] : indptr[i + 1]], lag)
            out[i] = func(lagged, *args)[-1]
    else:
        out = np.empty_like(data)
        for i in range(n_series):
            lagged = shift_array(data[indptr[i] : indptr[i + 1]], lag)
            out[indptr[i] : indptr[i + 1]] = func(lagged, *args)
    return out


@njit
def _diff(x, lag):
    y = x.copy()
    for i in range(lag):
        y[i] = np.nan
    for i in range(lag, x.size):
        y[i] = x[i] - x[i - lag]
    return y


@njit
def _apply_difference(data, indptr, new_data, new_indptr, d):
    n_series = len(indptr) - 1
    for i in range(n_series):
        new_data[new_indptr[i] : new_indptr[i + 1]] = data[
            indptr[i + 1] - d : indptr[i + 1]
        ]
        sl = slice(indptr[i], indptr[i + 1])
        data[sl] = _diff(data[sl], d)


@njit
def _restore_difference(preds, data, indptr, d):
    n_series = len(indptr) - 1
    h = len(preds) // n_series
    for i in range(n_series):
        s = data[indptr[i] : indptr[i + 1]]
        for j in range(min(h, d)):
            preds[i * h + j] += s[j]
        for j in range(d, h):
            preds[i * h + j] += preds[i * h + j - d]

# %% ../nbs/core.ipynb 12
class GroupedArray:
    """Array made up of different groups. Can be thought of (and iterated) as a list of arrays.

    All the data is stored in a single 1d array `data`.
    The indices for the group boundaries are stored in another 1d array `indptr`."""

    def __init__(self, data: np.ndarray, indptr: np.ndarray):
        self.data = data
        self.indptr = indptr
        self.ngroups = len(indptr) - 1

    def __len__(self) -> int:
        return self.ngroups

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[self.indptr[idx] : self.indptr[idx + 1]]

    def __setitem__(self, idx: int, vals: np.ndarray):
        if self[idx].size != vals.size:
            raise ValueError(f"vals must be of size {self[idx].size}")
        self[idx][:] = vals

    @classmethod
    def from_sorted_df(cls, df: pd.DataFrame, target_col: str) -> "GroupedArray":
        grouped = df.groupby(level=0, observed=True)
        sizes = grouped.size().values
        indptr = np.append(0, sizes.cumsum())
        data = df[target_col].values
        if data.dtype not in (np.float32, np.float64):
            # since all transformations generate nulls, we need a float dtype
            data = data.astype(np.float32)
        return cls(data, indptr)

    def transform_series(
        self, updates_only: bool, lag: int, func: Callable, *args
    ) -> np.ndarray:
        return _transform_series(self.data, self.indptr, updates_only, lag, func, *args)

    def restore_difference(self, preds: np.ndarray, d: int):
        _restore_difference(preds, self.data, self.indptr, d)

    def take_from_groups(self, idx: Union[int, slice]) -> "GroupedArray":
        """Takes `idx` from each group in the array."""
        ranges = [
            range(self.indptr[i], self.indptr[i + 1])[idx] for i in range(self.ngroups)
        ]
        items = [self.data[rng] for rng in ranges]
        sizes = np.array([item.size for item in items])
        data = np.hstack(items)
        indptr = np.append(0, sizes.cumsum())
        return GroupedArray(data, indptr)

    def append(self, new: np.ndarray) -> "GroupedArray":
        """Appends each element of `new` to each existing group. Returns a copy."""
        if new.size != self.ngroups:
            raise ValueError(f"new must be of size {self.ngroups}")
        new_data, new_indptr = _append_new(self.data, self.indptr, new)
        return GroupedArray(new_data, new_indptr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(ndata={self.data.size}, ngroups={self.ngroups})"
        )

# %% ../nbs/core.ipynb 19
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

# %% ../nbs/core.ipynb 21
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

# %% ../nbs/core.ipynb 23
Freq = Union[int, str, pd.offsets.BaseOffset]
Lags = Iterable[int]
LagTransform = Union[Callable, Tuple[Callable, Any]]
LagTransforms = Dict[int, List[LagTransform]]
DateFeature = Union[str, Callable]
Differences = Iterable[int]
Models = Union[BaseEstimator, List[BaseEstimator], Dict[str, BaseEstimator]]

# %% ../nbs/core.ipynb 24
class TimeSeries:
    """Utility class for storing and transforming time series data."""

    def __init__(
        self,
        freq: Optional[Freq] = None,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        differences: Optional[Differences] = None,
        num_threads: int = 1,
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
        self.differences = [] if differences is None else list(differences)
        self.num_threads = num_threads
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
        if id_col not in df and id_col != "index":
            raise ValueError(f"Couldn't find {id_col} column.")
        for col in (time_col, target_col):
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
        if id_col != "index":
            df = df.set_index(id_col)
        if static_features is None:
            static_features = df.columns.drop([time_col, target_col])
        self.static_features = (
            df[static_features].groupby(level=0, observed=True).head(1)
        )
        sort_idxs = pd.core.sorting.lexsort_indexer([df.index, df[time_col]])
        sorted_df = (
            df[[time_col, target_col]].set_index(time_col, append=True).iloc[sort_idxs]
        )
        self.restore_idxs = np.empty(df.shape[0], dtype=np.int32)
        self.restore_idxs[sort_idxs] = np.arange(df.shape[0])
        self.uids = sorted_df.index.unique(level=0)
        self.ga = GroupedArray.from_sorted_df(sorted_df, target_col)
        if self.differences:
            original_sizes = self.ga.indptr[1:].cumsum()
            total_diffs = sum(self.differences)
            small_series = self.uids[original_sizes < total_diffs]
            if small_series.size:
                msg = reprlib.repr(small_series.tolist())
                raise ValueError(
                    f"The following series are too short for the differences: {msg}"
                )
            self.original_values_ = []
            n_series = len(self.ga.indptr) - 1
            for d in self.differences:
                new_data = np.empty_like(self.ga.data, shape=n_series * d)
                new_indptr = d * np.arange(n_series + 1, dtype=np.int32)
                _apply_difference(self.ga.data, self.ga.indptr, new_data, new_indptr, d)
                self.original_values_.append(GroupedArray(new_data, new_indptr))
        self.features_ = self._compute_transforms()
        if keep_last_n is not None:
            self.ga = self.ga.take_from_groups(slice(-keep_last_n, None))
        self._ga = GroupedArray(self.ga.data, self.ga.indptr)
        self.last_dates = sorted_df.index.get_level_values(self.time_col)[
            self.ga.indptr[1:] - 1
        ]
        self.features_order_ = (
            df.columns.drop([self.time_col, self.target_col]).tolist() + self.features
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
    ) -> pd.DataFrame:
        """Add the features to `df`.

        if `dropna=True` then all the null rows are dropped."""
        df = df.copy(deep=bool(self.differences))
        for feat in self.transforms.keys():
            df[feat] = self.features_[feat][self.restore_idxs]

        if self.differences:
            df[self.target_col] = self.ga.data[self.restore_idxs]

        if dropna:
            df.dropna(inplace=True)

        dates = df[self.time_col]
        if not np.issubdtype(dates.dtype.type, np.integer):
            dates = pd.DatetimeIndex(dates)
        for feature in self.date_features:
            feat_name, feat_vals = self._compute_date_feature(dates, feature)
            df[feat_name] = feat_vals
        return df

    def fit_transform(
        self,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Add the features to `data` and save the required information for the predictions step.

        If not all features are static, specify which ones are in `static_features`.
        If you don't want to drop rows with null values after the transformations set `dropna=False`
        If `keep_last_n` is not None then that number of observations is kept across all series for updates.
        """
        self.dropna = dropna
        self.keep_last_n = keep_last_n
        self._fit(data, id_col, time_col, target_col, static_features, keep_last_n)
        return self._transform(data, dropna)

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

        features_df = pd.DataFrame(features, columns=self.features, index=self.uids)
        results_df = self.static_features.join(features_df)
        results_df[self.time_col] = self.curr_dates
        return results_df

    def _get_raw_predictions(self) -> np.ndarray:
        preds = np.array(self.y_pred).ravel("F")
        if not self.differences:
            return preds
        for d, ga in zip(reversed(self.differences), reversed(self.original_values_)):
            ga.restore_difference(preds, d)
        return preds

    def _get_predictions(self) -> pd.DataFrame:
        """Get all the predicted values with their corresponding ids and datestamps."""
        n_preds = len(self.y_pred)
        idx = pd.Index(
            np.repeat(self.uids, n_preds),
            name=self.id_col if self.id_col != "index" else None,
            dtype=self.uids.dtype,
        )
        df = pd.DataFrame(
            {
                self.time_col: np.array(self.test_dates).ravel("F"),
                f"{self.target_col}_pred": self._get_raw_predictions(),
            },
            index=idx,
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
            idx_name = new_x.index.name
            new_x = new_x.reset_index()
            for df in dynamic_dfs:
                new_x = new_x.merge(df, how="left")
            new_x = new_x.sort_values(idx_name)
        nulls = new_x.isnull().any()
        if any(nulls):
            warnings.warn(f'Found null values in {", ".join(nulls[nulls].index)}.')
        return new_x[self.features_order_]

    def predict(
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
                    predictions_serie = pd.Series(predictions, index=new_x.index)
                    predictions = after_predict_callback(predictions_serie).values
                self._update_y(predictions)
            if i == 0:
                preds = self._get_predictions()
                preds = preds.rename(
                    columns={f"{self.target_col}_pred": name}, copy=False
                )
            else:
                preds[name] = self._get_raw_predictions()
        if self.id_col != "index":
            preds = preds.reset_index()
        return preds
