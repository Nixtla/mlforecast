# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/core.ipynb.

# %% auto 0
__all__ = ['TimeSeries']

# %% ../nbs/core.ipynb 3
import concurrent.futures
import inspect
import warnings
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit
from sklearn.base import BaseEstimator
from utilsforecast.compat import (
    DataFrame,
    pl,
    pl_DataFrame,
    pl_Series,
)
from utilsforecast.processing import (
    DataFrameProcessor,
    assign_columns,
    between,
    copy_if_pandas,
    counts_by_id,
    drop_index_if_pandas,
    fill_null,
    filter_with_mask,
    group_by_agg,
    horizontal_concat,
    is_in,
    is_nan_or_none,
    join,
    match_if_categorical,
    offset_dates,
    rename,
    sort,
    take_rows,
    to_numpy,
    vertical_concat,
)
from utilsforecast.validation import validate_format

from .grouped_array import GroupedArray
from mlforecast.target_transforms import (
    BaseGroupedArrayTargetTransform,
    BaseTargetTransform,
)
from .utils import _ensure_shallow_copy

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
TargetTransform = Union[BaseTargetTransform, BaseGroupedArrayTargetTransform]

# %% ../nbs/core.ipynb 17
class TimeSeries:
    """Utility class for storing and transforming time series data."""

    def __init__(
        self,
        freq: Optional[Freq] = None,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        num_threads: int = 1,
        target_transforms: Optional[List[TargetTransform]] = None,
    ):
        self.freq = freq
        if not isinstance(num_threads, int) or num_threads < 1:
            warnings.warn("Setting num_threads to 1.")
            num_threads = 1
        self.lags = [] if lags is None else list(lags)
        for lag in self.lags:
            if lag <= 0 or not isinstance(lag, int):
                raise ValueError("lags must be positive integers.")
        self.lag_transforms = {} if lag_transforms is None else lag_transforms
        for lag in self.lag_transforms.keys():
            if lag <= 0 or not isinstance(lag, int):
                raise ValueError("keys of lag_transforms must be positive integers.")
        self.date_features = [] if date_features is None else list(date_features)
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

    def _validate_freq(self, df: DataFrame, time_col) -> None:
        if isinstance(df, pd.DataFrame):
            time_col_is_datetime = pd.api.types.is_datetime64_dtype(df[time_col])
        else:
            time_col_is_datetime = isinstance(df[time_col].dtype, pl.Datetime)
        if isinstance(self.freq, str):
            if isinstance(df, pd.DataFrame):
                self.freq = pd.tseries.frequencies.to_offset(self.freq)
                if not time_col_is_datetime:
                    raise ValueError(
                        f"Time col ({time_col}) has integers "
                        "but specified frequency implies datetime."
                    )
        elif isinstance(self.freq, pd.tseries.offsets.BaseOffset):
            if not isinstance(df, pd.DataFrame):
                raise ValueError(
                    f"Inferred frequency for pandas dataframe, but got {type(df)}."
                )
        elif isinstance(self.freq, int):
            if time_col_is_datetime:
                raise ValueError(
                    "Must set frequency when using a datetime type column."
                )
        elif self.freq is None:
            warnings.warn(
                "Setting `freq=1` since it wasn't provided. "
                "The `freq` argument will become required in a future version."
            )
            self.freq = 1
        else:
            raise ValueError(
                "Unknown frequency type "
                "Please use a str, int or offset frequency type."
            )

    def _fit(
        self,
        df: DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        keep_last_n: Optional[int] = None,
    ) -> "TimeSeries":
        """Save the series values, ids and last dates."""
        validate_format(df, id_col, time_col, target_col)
        if is_nan_or_none(df[target_col]).any():
            raise ValueError(f"{target_col} column contains null values.")
        self._validate_freq(df, time_col)
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col
        self.keep_last_n = keep_last_n
        self.static_features = static_features
        proc = DataFrameProcessor(id_col, time_col, target_col)
        sorted_df = df[[id_col, time_col, target_col]]
        sorted_df = copy_if_pandas(sorted_df, deep=False)
        uids, times, data, indptr, sort_idxs = proc.process(sorted_df)
        if data.ndim == 2:
            data = data[:, 0]
        ga = GroupedArray(data, indptr)
        if isinstance(df, pd.DataFrame):
            self.uids = pd.Index(uids)
            self.last_dates = pd.Index(times)
        else:
            self.uids = uids
            self.last_dates = pl_Series(times)
        if sort_idxs is not None:
            self.restore_idxs: Optional[np.ndarray] = np.empty(
                df.shape[0], dtype=np.int32
            )
            self.restore_idxs[sort_idxs] = np.arange(df.shape[0])
            sorted_df = take_rows(sorted_df, sort_idxs)
        else:
            self.restore_idxs = None
        if self.target_transforms is not None:
            for tfm in self.target_transforms:
                if isinstance(tfm, BaseGroupedArrayTargetTransform):
                    ga = tfm.fit_transform(ga)
                    sorted_df = assign_columns(sorted_df, target_col, ga.data)
                else:
                    tfm.set_column_names(id_col, time_col, target_col)
                    sorted_df = tfm.fit_transform(sorted_df)
                    ga.data = sorted_df[target_col].to_numpy()
        self.ga = ga
        self._ga = GroupedArray(self.ga.data, self.ga.indptr)
        last_idxs_per_serie = self.ga.indptr[1:] - 1
        to_drop = [id_col, time_col, target_col]
        if static_features is None:
            static_features = [c for c in df.columns if c not in [time_col, target_col]]
        elif id_col not in static_features:
            static_features = [id_col] + static_features
        else:  # static_features defined and contain id_col
            to_drop = [time_col, target_col]
        if sort_idxs is not None:
            last_idxs_per_serie = sort_idxs[last_idxs_per_serie]
        self.static_features_ = take_rows(df, last_idxs_per_serie)[static_features]
        self.static_features_ = drop_index_if_pandas(self.static_features_)
        self.features_order_ = [
            c for c in df.columns if c not in to_drop
        ] + self.features
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
            if isinstance(dates, pd.DatetimeIndex):
                if feature in ("week", "weekofyear"):
                    dates = dates.isocalendar()
                feat_vals = getattr(dates, feature)
            else:
                feat_vals = getattr(dates.dt, feature)()
        if not isinstance(feat_vals, pl_Series):
            feat_vals = np.asarray(feat_vals)
            feat_dtype = date_features_dtypes.get(feature)
            if feat_dtype is not None:
                feat_vals = feat_vals.astype(feat_dtype)
        return feat_name, feat_vals

    def _transform(
        self,
        df: DataFrame,
        dropna: bool = True,
        max_horizon: Optional[int] = None,
        return_X_y: bool = False,
        as_numpy: bool = False,
    ) -> pd.DataFrame:
        """Add the features to `df`.

        if `dropna=True` then all the null rows are dropped."""
        features = self._compute_transforms()
        if self.restore_idxs is not None:
            for k, v in features.items():
                features[k] = v[self.restore_idxs]

        # target
        self.max_horizon = max_horizon
        if max_horizon is None:
            target = self.ga.data
        else:
            target = self.ga.expand_target(max_horizon)
        if self.restore_idxs is not None:
            target = target[self.restore_idxs]
        del self.restore_idxs

        # determine rows to keep
        if dropna:
            feature_nulls = np.full(df.shape[0], False)
            for feature_vals in features.values():
                feature_nulls |= np.isnan(feature_vals)
            target_nulls = np.isnan(target)
            if target_nulls.ndim == 2:
                # target nulls for each horizon are dropped in MLForecast.fit_models
                # we just drop rows here for which all the target values are null
                target_nulls = target_nulls.all(axis=1)
            keep_rows = ~(feature_nulls | target_nulls)
            for k, v in features.items():
                features[k] = v[keep_rows]
            df = filter_with_mask(df, keep_rows)
            df = copy_if_pandas(df, deep=False)
            target = target[keep_rows]
        elif isinstance(df, pd.DataFrame):
            df = df.copy(deep=False)

        # lag transforms
        for feat in self.transforms.keys():
            df = assign_columns(df, feat, features[feat])

        # date features
        if self.date_features:
            dates = df[self.time_col]
            if isinstance(dates, pd.Series) and not np.issubdtype(
                dates.dtype.type, np.integer
            ):
                dates = pd.DatetimeIndex(dates)
            for feature in self.date_features:
                feat_name, feat_vals = self._compute_date_feature(dates, feature)
                df = assign_columns(df, feat_name, feat_vals)

        # assemble return
        if return_X_y:
            X = df[self.features_order_]
            if as_numpy:
                X = to_numpy(X)
            return X, target
        if max_horizon is not None:
            # remove original target
            out_cols = [c for c in df.columns if c != self.target_col]
            df = df[out_cols]
            target_names = [f"{self.target_col}{i}" for i in range(max_horizon)]
            df = assign_columns(df, target_names, target)
        else:
            if isinstance(df, pd.DataFrame):
                df = _ensure_shallow_copy(df)
            df = assign_columns(df, self.target_col, target)
        return df

    def fit_transform(
        self,
        data: DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        return_X_y: bool = False,
        as_numpy: bool = False,
    ) -> Union[DataFrame, Tuple[DataFrame, np.ndarray]]:
        """Add the features to `data` and save the required information for the predictions step.

        If not all features are static, specify which ones are in `static_features`.
        If you don't want to drop rows with null values after the transformations set `dropna=False`
        If `keep_last_n` is not None then that number of observations is kept across all series for updates.
        """
        self.dropna = dropna
        self.as_numpy = as_numpy
        self._fit(
            df=data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            keep_last_n=keep_last_n,
        )
        return self._transform(
            df=data,
            dropna=dropna,
            max_horizon=max_horizon,
            return_X_y=return_X_y,
            as_numpy=as_numpy,
        )

    def _update_y(self, new: np.ndarray) -> None:
        """Appends the elements of `new` to every time serie.

        These values are used to update the transformations and are stored as predictions.
        """
        if not hasattr(self, "y_pred"):
            self.y_pred = []
        self.y_pred.append(new)
        new_arr = np.asarray(new)
        self.ga = self.ga.append(new_arr)

    def _update_features(self) -> DataFrame:
        """Compute the current values of all the features using the latest values of the time series."""
        self.curr_dates: Union[pd.Index, pl_Series] = offset_dates(
            self.curr_dates, self.freq, 1
        )
        self.test_dates.append(self.curr_dates)

        if self.num_threads == 1 or len(self.transforms) == 1:
            features = self._apply_transforms(updates_only=True)
        else:
            features = self._apply_multithreaded_transforms(updates_only=True)

        for feature in self.date_features:
            feat_name, feat_vals = self._compute_date_feature(self.curr_dates, feature)
            features[feat_name] = feat_vals

        if isinstance(self.last_dates, pl_Series):
            features_df = pl_DataFrame(features, schema=self.features)
        else:
            features_df = pd.DataFrame(features, columns=self.features)
        return horizontal_concat([self._static_features, features_df])

    def _get_raw_predictions(self) -> np.ndarray:
        return np.array(self.y_pred).ravel("F")

    def _get_future_ids(self, h: int):
        if isinstance(self._uids, pl_Series):
            uids = pl.concat([self._uids for _ in range(h)]).sort()
        else:
            uids = pd.Series(
                np.repeat(self._uids, h), name=self.id_col, dtype=self.uids.dtype
            )
        return uids

    def _get_predictions(self) -> DataFrame:
        """Get all the predicted values with their corresponding ids and datestamps."""
        h = len(self.y_pred)
        if isinstance(self._uids, pl_Series):
            df_constructor = pl_DataFrame
        else:
            df_constructor = pd.DataFrame
        uids = self._get_future_ids(h)
        df = df_constructor(
            {
                self.id_col: uids,
                self.time_col: np.array(self.test_dates).ravel("F"),
                f"{self.target_col}_pred": self._get_raw_predictions(),
            },
        )
        return df

    def _predict_setup(self) -> None:
        self.ga = GroupedArray(self._ga.data, self._ga.indptr)
        if isinstance(self.last_dates, pl_Series):
            self.curr_dates = self.last_dates.clone()
        else:
            self.curr_dates = self.last_dates.copy()
        if self._idxs is not None:
            self.ga = self.ga.take(self._idxs)
            self.curr_dates = self.curr_dates[self._idxs]
        self.test_dates: List[Union[pd.Index, pl_Series]] = []
        self.y_pred = []
        if self.keep_last_n is not None:
            self.ga = self.ga.take_from_groups(slice(-self.keep_last_n, None))
        self._h = 0

    def _get_features_for_next_step(self, X_df=None):
        new_x = self._update_features()
        if X_df is not None:
            n_series = len(self._uids)
            h = X_df.shape[0] // n_series
            rows = np.arange(self._h, X_df.shape[0], h)
            X = take_rows(X_df, rows)
            X = drop_index_if_pandas(X)
            new_x = horizontal_concat([new_x, X])
        if isinstance(new_x, pd.DataFrame):
            nulls = new_x.isnull().any()
            cols_with_nulls = nulls[nulls].index.tolist()
        else:
            nulls = new_x.select(pl.all().is_null().any())
            cols_with_nulls = [k for k, v in nulls.to_dicts()[0].items() if v]
        if cols_with_nulls:
            warnings.warn(f'Found null values in {", ".join(cols_with_nulls)}.')
        self._h += 1
        new_x = new_x[self.features_order_]
        if self.as_numpy:
            new_x = to_numpy(new_x)
        return new_x

    def _predict_recursive(
        self,
        models: Dict[str, BaseEstimator],
        horizon: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        X_df: Optional[DataFrame] = None,
    ) -> DataFrame:
        """Use `model` to predict the next `horizon` timesteps."""
        for i, (name, model) in enumerate(models.items()):
            self._predict_setup()
            for _ in range(horizon):
                new_x = self._get_features_for_next_step(X_df)
                if before_predict_callback is not None:
                    new_x = before_predict_callback(new_x)
                predictions = model.predict(new_x)
                if after_predict_callback is not None:
                    predictions = after_predict_callback(predictions)
                self._update_y(predictions)
            if i == 0:
                preds = self._get_predictions()
                rename_dict = {f"{self.target_col}_pred": name}
                preds = rename(preds, rename_dict)
            else:
                raw_preds = self._get_raw_predictions()
                preds = assign_columns(preds, name, raw_preds)
        return preds

    def _predict_multi(
        self,
        models: Dict[str, BaseEstimator],
        horizon: int,
        before_predict_callback: Optional[Callable] = None,
        X_df: Optional[DataFrame] = None,
    ) -> DataFrame:
        assert self.max_horizon is not None
        if horizon > self.max_horizon:
            raise ValueError(
                f"horizon must be at most max_horizon ({self.max_horizon})"
            )
        self._predict_setup()
        uids = self._get_future_ids(horizon)
        if isinstance(self.curr_dates, pl_Series):
            starts = offset_dates(self.curr_dates, self.freq, 1)
            ends = offset_dates(self.curr_dates, self.freq, horizon)
            dates = pl.date_ranges(
                starts, ends, interval=self.freq, eager=True
            ).explode()
            df_constructor = pl_DataFrame
        else:
            assert isinstance(self.freq, (pd.offsets.BaseOffset, int))
            dates = np.hstack(
                [
                    date + (i + 1) * self.freq
                    for date in self.curr_dates
                    for i in range(horizon)
                ]
            )
            df_constructor = pd.DataFrame
        result = df_constructor({self.id_col: uids, self.time_col: dates})
        for name, model in models.items():
            self._predict_setup()
            new_x = self._get_features_for_next_step(X_df)
            if before_predict_callback is not None:
                new_x = before_predict_callback(new_x)
            predictions = np.empty((new_x.shape[0], horizon))
            for i in range(horizon):
                predictions[:, i] = model[i].predict(new_x)
            raw_preds = predictions.ravel()
            result = assign_columns(result, name, raw_preds)
        return result

    def predict(
        self,
        models: Dict[str, Union[BaseEstimator, List[BaseEstimator]]],
        horizon: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        X_df: Optional[DataFrame] = None,
        ids: Optional[List[str]] = None,
    ) -> DataFrame:
        if ids is not None:
            unseen = set(ids) - set(self.uids)
            if unseen:
                raise ValueError(
                    f"The following ids weren't seen during training and thus can't be forecasted: {unseen}"
                )
            self._idxs: Optional[np.ndarray] = np.where(is_in(self.uids, ids))[0]
            self._uids = self.uids[self._idxs]
            self._static_features = take_rows(self.static_features_, self._idxs)
            self._static_features = drop_index_if_pandas(self._static_features)
            last_dates = self.last_dates[self._idxs]
        else:
            self._idxs = None
            self._uids = self.uids
            self._static_features = self.static_features_
            last_dates = self.last_dates
        if X_df is not None:
            if self.id_col not in X_df or self.time_col not in X_df:
                raise ValueError(
                    f"X_df must have '{self.id_col}' and '{self.time_col}' columns."
                )
            if X_df.shape[1] < 3:
                raise ValueError("Found no exogenous features in `X_df`.")
            statics = [c for c in self.static_features_.columns if c != self.id_col]
            dynamics = [
                c for c in X_df.columns if c not in [self.id_col, self.time_col]
            ]
            common = [c for c in dynamics if c in statics]
            if common:
                raise ValueError(
                    f"The following features were provided through `X_df` but were considered as static during fit: {common}.\n"
                    "Please re-run the fit step using the `static_features` argument to indicate which features are static. "
                    "If all your features are dynamic please pass an empty list (static_features=[])."
                )
            starts = offset_dates(last_dates, self.freq, 1)
            ends = offset_dates(last_dates, self.freq, horizon)
            df_constructor = type(X_df)
            dates_validation = df_constructor(
                {
                    self.id_col: self._uids,
                    "_start": starts,
                    "_end": ends,
                }
            )
            X_df = join(X_df, dates_validation, on=self.id_col)
            between_attr = "between" if isinstance(X_df, pd.DataFrame) else "is_between"
            mask = getattr(X_df[self.time_col], between_attr)(
                X_df["_start"], X_df["_end"]
            )
            X_df = X_df[mask]
            if X_df.shape[0] != len(self._uids) * horizon:
                raise ValueError(
                    "Found missing inputs in X_df. "
                    "It should have one row per id and date for the complete forecasting horizon"
                )
            drop_cols = [self.id_col, self.time_col, "_start", "_end"]
            X_df = sort(X_df, [self.id_col, self.time_col]).drop(columns=drop_cols)
        if getattr(self, "max_horizon", None) is None:
            preds = self._predict_recursive(
                models=models,
                horizon=horizon,
                before_predict_callback=before_predict_callback,
                after_predict_callback=after_predict_callback,
                X_df=X_df,
            )
        else:
            preds = self._predict_multi(
                models=models,
                horizon=horizon,
                before_predict_callback=before_predict_callback,
                X_df=X_df,
            )
        if self.target_transforms is not None:
            if any(
                isinstance(tfm, BaseGroupedArrayTargetTransform)
                for tfm in self.target_transforms
            ):
                model_cols = [
                    c for c in preds.columns if c not in (self.id_col, self.time_col)
                ]
                indptr = np.arange(0, horizon * (len(self._uids) + 1), horizon)
            for tfm in self.target_transforms[::-1]:
                if isinstance(tfm, BaseGroupedArrayTargetTransform):
                    tfm.idxs = self._idxs
                    for col in model_cols:
                        ga = GroupedArray(preds[col].to_numpy(), indptr)
                        ga = tfm.inverse_transform(ga)
                        preds = assign_columns(preds, col, ga.data)
                    tfm.idxs = None
                else:
                    preds = tfm.inverse_transform(preds)
        del self._uids, self._idxs, self._static_features
        return preds

    def update(self, df: DataFrame) -> None:
        """Update the values of the stored series."""
        validate_format(df, self.id_col, self.time_col, self.target_col)
        self.uids, new_ids = match_if_categorical(self.uids, df[self.id_col])
        df = assign_columns(df, self.id_col, new_ids)
        df = sort(df, by=[self.id_col, self.time_col])
        values = df[self.target_col].to_numpy()
        new_id_counts = counts_by_id(df, self.id_col)
        sizes = join(self.uids, new_id_counts, on=self.id_col, how="outer")
        sizes = fill_null(sizes, {"counts": 0})
        sizes = sort(sizes, by=self.id_col)
        new_groups = ~is_in(sizes[self.id_col], self.uids)
        last_dates = group_by_agg(df, self.id_col, {self.time_col: "max"})
        last_dates = join(sizes, last_dates, on=self.id_col, how="left")
        curr_last_dates = type(df)({self.id_col: self.uids, "_curr": self.last_dates})
        last_dates = join(last_dates, curr_last_dates, on=self.id_col, how="left")
        last_dates = fill_null(last_dates, {self.time_col: last_dates["_curr"]})
        last_dates = sort(last_dates, by=self.id_col)
        self.last_dates = last_dates[self.time_col]
        self.uids = sort(sizes[self.id_col])
        if new_groups.any():
            unseen_ids = filter_with_mask(sizes[self.id_col], new_groups)
            unseen_statics = filter_with_mask(df, is_in(df[self.id_col], unseen_ids))
            unseen_ids_sizes = filter_with_mask(
                new_id_counts, is_in(new_id_counts[self.id_col], unseen_ids)
            )
            new_statics = take_rows(df, unseen_ids_sizes["counts"].cumsum() - 1)[
                self.static_features_.columns
            ]
            self.static_features_ = vertical_concat(
                [self.static_features_, new_statics]
            )
            self.static_features_ = sort(self.static_features_, self.id_col)
        self._ga = self._ga.append_several(
            new_sizes=sizes["counts"].to_numpy().astype(np.int32),
            new_values=values,
            new_groups=new_groups.to_numpy(),
        )
