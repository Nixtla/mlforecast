# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/core.ipynb.

# %% auto 0
__all__ = ['simple_predict', 'merge_predict', 'TimeSeries']

# %% ../nbs/core.ipynb 3
import concurrent.futures
import inspect
import warnings
from collections import Counter, OrderedDict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit
from window_ops.shift import shift_array

from .utils import data_indptr_from_sorted_df

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
        return f"GroupedArray(ndata={self.data.size}, ngroups={self.ngroups})"

# %% ../nbs/core.ipynb 17
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

# %% ../nbs/core.ipynb 18
def _build_transform_name(lag, tfm, *args) -> str:
    """Creates a name for a transformation based on `lag`, the name of the function and its arguments."""
    tfm_name = f"{tfm.__name__}_lag-{lag}"
    func_params = inspect.signature(tfm).parameters
    func_args = list(func_params.items())[1:]  # remove input array argument
    changed_params = [
        f"{name}-{value}"
        for value, (name, arg) in zip(args, func_args)
        if arg.default != value
    ]
    if changed_params:
        tfm_name += "_" + "_".join(changed_params)
    return tfm_name

# %% ../nbs/core.ipynb 20
def simple_predict(
    model,
    new_x: pd.DataFrame,
    dynamic_dfs: List[pd.DataFrame],
    features_order: List[str],
    **kwargs,
) -> np.ndarray:
    """Drop the ds column from `new_x` and call `model.predict` on it."""
    new_x = new_x[features_order]
    return model.predict(new_x)


def merge_predict(
    model,
    new_x: pd.DataFrame,
    dynamic_dfs: List[pd.DataFrame],
    features_order: List[str],
    **kwargs,
) -> np.ndarray:
    """Perform left join on each of `dynamic_dfs` and call model.predict."""
    idx = new_x.index.name
    new_x = new_x.reset_index()
    for df in dynamic_dfs:
        new_x = new_x.merge(df, how="left")
    new_x = new_x.sort_values(idx)
    new_x = new_x[features_order]
    return model.predict(new_x)

# %% ../nbs/core.ipynb 21
def _name_models(current_names):
    ctr = Counter(current_names)
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
class TimeSeries:
    """Utility class for storing and transforming time series data."""

    def __init__(
        self,
        freq: Optional[str] = None,
        lags: List[int] = [],
        lag_transforms: Dict[int, List[Tuple]] = {},
        date_features: List[str] = [],
        num_threads: int = 1,
    ):
        if freq is not None:
            self.freq = pd.tseries.frequencies.to_offset(freq)
        else:
            self.freq = 1
        if not isinstance(num_threads, int) or num_threads < 1:
            warnings.warn("Setting num_threads to 1.")
            num_threads = 1
        self.num_threads = num_threads
        self.date_features = list(date_features)

        self.transforms: Dict[str, Tuple[Any, ...]] = OrderedDict()
        for lag in lags:
            self.transforms[f"lag-{lag}"] = (lag, _identity)
        for lag in lag_transforms.keys():
            for tfm_args in lag_transforms[lag]:
                tfm, *args = _as_tuple(tfm_args)
                tfm_name = _build_transform_name(lag, tfm, *args)
                self.transforms[tfm_name] = (lag, tfm, *args)

        self.ga: GroupedArray

    @property
    def features(self) -> List[str]:
        """Names of all computed features."""
        return list(self.transforms.keys()) + self.date_features

    def __repr__(self):
        return (
            f"TimeSeries(freq={self.freq}, "
            f"transforms={list(self.transforms.keys())}, "
            f"date_features={self.date_features}, "
            f"num_threads={self.num_threads})"
        )

    def _apply_transforms(self, updates_only: bool = False) -> Dict[str, np.ndarray]:
        """Apply the transformations using the main process.

        If `updates_only` then only the updates are returned.
        """
        results = {}
        offset = 1 if updates_only else 0
        for tfm_name, (lag, tfm, *args) in self.transforms.items():
            results[tfm_name] = _transform_series(
                self.ga.data, self.ga.indptr, updates_only, lag - offset, tfm, *args
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
                    _transform_series,
                    self.ga.data,
                    self.ga.indptr,
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

        if not isinstance(self.freq, int):
            for feature in self.date_features:
                feat_vals = getattr(self.curr_dates, feature).values
                features[feature] = feat_vals.astype(date_features_dtypes[feature])

        features_df = pd.DataFrame(features, columns=self.features, index=self.uids)
        nulls_in_cols = features_df.isnull().any()
        if any(nulls_in_cols):
            warnings.warn(
                f'Found null values in {", ".join(nulls_in_cols[nulls_in_cols].index)}.'
            )
        results_df = self.static_features.join(features_df)
        results_df[self.time_col] = self.curr_dates
        return results_df

    def _get_raw_predictions(self) -> np.ndarray:
        return np.array(self.y_pred).ravel("F")

    def _get_predictions(self) -> pd.DataFrame:
        """Get all the predicted values with their corresponding ids and datestamps."""
        n_preds = len(self.y_pred)
        idx = pd.Index(
            chain.from_iterable([uid] * n_preds for uid in self.uids),
            name=self.id_col,
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

    def _fit(self, df: pd.DataFrame) -> "TimeSeries":
        """Save the series values, ids and last dates."""
        data, indptr = data_indptr_from_sorted_df(df)
        if data.dtype not in (np.float32, np.float64):
            # since all transformations generate nulls, we need a float dtype
            data = data.astype(np.float32)
        self.ga = GroupedArray(data, indptr)
        self.uids = df.index.unique(level="unique_id")
        self.last_dates = df.index.get_level_values("ds")[indptr[1:] - 1]
        return self

    def _transform(
        self,
        df: pd.DataFrame,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Add the features to `df`.

        if `dropna=True` then all the null rows are dropped.
        if `keep_last_n` is not None then that number of observations is kept across all series."""
        df = df.copy()
        features = self._compute_transforms()
        for feat in self.transforms.keys():
            df[feat] = features[feat][self.restore_idxs]

        if dropna:
            df.dropna(inplace=True)

        if not isinstance(self.freq, int):
            for feature in self.date_features:
                feat_vals = getattr(df.ds.dt, feature).values
                df[feature] = feat_vals.astype(date_features_dtypes[feature])

        if keep_last_n is not None:
            self.ga = self.ga.take_from_groups(slice(-keep_last_n, None))

        self._ga = GroupedArray(self.ga.data, self.ga.indptr)
        self.features_order_ = df.columns.drop(["ds", "y"])
        return df

    def fit_transform(
        self,
        data: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Add the features to `data` and save the required information for the predictions step.

        If not all features are static, specify which ones are in `static_features`.
        If you don't want to drop rows with null values after the transformations set `dropna=False`.
        If you want to keep only the last `n` values of each time serie set `keep_last_n=n`.
        """
        if id_col not in data and data.index.name != id_col:
            raise ValueError(f"Couldn't find {id_col} as a column nor as the index.")
        for col in (time_col, target_col):
            if col not in data:
                raise ValueError(f"Data doesn't contain {col} column")
        if pd.api.types.is_datetime64_dtype(data[time_col]):
            if self.freq == 1:
                raise ValueError(
                    "Must set frequency when using a timestamp type column."
                )
        elif np.issubdtype(data[time_col].dtype.type, np.integer):
            if self.date_features:
                warnings.warn("Ignoring date_features since time column is integer.")
        else:
            raise ValueError(f"{time_col} must be either timestamp or integer.")
        if data[target_col].isnull().any():
            raise ValueError(f"{target_col} column contains null values.")
        if id_col in data:
            data = data.set_index(id_col)
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col
        data.index.name = "unique_id"
        data = data.rename(columns={time_col: "ds", target_col: "y"}, copy=False)
        if static_features is None:
            static_features = data.columns.drop(["ds", "y"])
        self.static_features = (
            data[static_features].reset_index().drop_duplicates().set_index("unique_id")
        )
        sort_idxs = pd.core.sorting.lexsort_indexer([data.index, data["ds"]])
        sorted_data = data[["ds", "y"]].set_index("ds", append=True).iloc[sort_idxs]
        self.restore_idxs = np.empty(data.shape[0], dtype=np.int32)
        self.restore_idxs[sort_idxs] = np.arange(data.shape[0])
        self._fit(sorted_data)
        transformed = self._transform(data, dropna, keep_last_n)
        transformed.index.name = id_col
        return transformed.rename(columns={"y": target_col, "ds": time_col}, copy=False)

    def _predict_setup(self) -> None:
        self.curr_dates = self.last_dates.copy()
        self.test_dates = []
        self.y_pred = []
        self.ga = GroupedArray(self._ga.data, self._ga.indptr)

    def _define_predict_fn(self, predict_fn, dynamic_dfs) -> Callable:
        if predict_fn is not None:
            return predict_fn
        if dynamic_dfs is None:
            return simple_predict
        return merge_predict

    def predict(
        self,
        models,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        predict_fn: Optional[Callable] = None,
        **predict_fn_kwargs,
    ) -> pd.DataFrame:
        """Use `model` to predict the next `horizon` timesteps."""
        if not isinstance(models, list):
            models = [models]
        model_names = _name_models([m.__class__.__name__ for m in models])
        for i, model in enumerate(models):
            self._predict_setup()
            predict_fn = self._define_predict_fn(predict_fn, dynamic_dfs)
            for _ in range(horizon):
                new_x = self._update_features()
                predictions = predict_fn(
                    model,
                    new_x,
                    dynamic_dfs=dynamic_dfs,
                    features_order=self.features_order_,
                    **predict_fn_kwargs,
                )
                self._update_y(predictions)
            if i == 0:
                preds = self._get_predictions()
                preds = preds.rename(
                    columns={f"{self.target_col}_pred": model_names[i]}, copy=False
                )
            else:
                preds[model_names[i]] = self._get_raw_predictions()
        return preds
