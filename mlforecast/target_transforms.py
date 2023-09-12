# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/target_transforms.ipynb.

# %% auto 0
__all__ = ['BaseTargetTransform', 'Differences', 'LocalStandardScaler', 'GlobalSklearnTransformer']

# %% ../nbs/target_transforms.ipynb 3
import abc
import reprlib
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, clone
from numba import njit

from .grouped_array import GroupedArray, _apply_difference
from .utils import _ensure_shallow_copy

# %% ../nbs/target_transforms.ipynb 5
class BaseTargetTransform(abc.ABC):
    """Base class used for target transformations."""

    idxs: Optional[np.ndarray] = None

    def set_column_names(self, id_col: str, time_col: str, target_col: str):
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

    @abc.abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform_fitted(
        self, df: pd.DataFrame, sizes: np.ndarray
    ) -> pd.DataFrame:
        return self.inverse_transform(df)

# %% ../nbs/target_transforms.ipynb 6
class Differences(BaseTargetTransform):
    """Subtracts previous values of the serie. Can be used to remove trend or seasonalities."""

    store_fitted = False

    def __init__(self, differences: Iterable[int]):
        self.differences = list(differences)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fitted_: List[GroupedArray] = []
        ga = GroupedArray.from_sorted_df(df, self.id_col, self.target_col)
        original_sizes = np.diff(ga.indptr)
        total_diffs = sum(self.differences)
        small_series = original_sizes < total_diffs
        if small_series.any():
            uids = df[self.id_col].unique()
            msg = reprlib.repr(uids[small_series].tolist())
            raise ValueError(
                f"The following series are too short for the differences: {msg}"
            )
        self.original_values_ = []
        n_series = len(ga.indptr) - 1
        for d in self.differences:
            if self.store_fitted:
                # these are saved in order to be able to perform a correct
                # inverse transform when trying to retrieve the fitted values.
                self.fitted_.append(GroupedArray(ga.data.copy(), ga.indptr.copy()))
            new_data = np.empty_like(ga.data, shape=n_series * d)
            new_indptr = d * np.arange(n_series + 1, dtype=np.int32)
            _apply_difference(ga.data, ga.indptr, new_data, new_indptr, d)
            self.original_values_.append(GroupedArray(new_data, new_indptr))
        df = df.copy(deep=False)
        df = _ensure_shallow_copy(df)
        df[self.target_col] = ga.data
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        model_cols = df.columns.drop([self.id_col, self.time_col])
        df = df.copy(deep=False)
        df = _ensure_shallow_copy(df)
        for model in model_cols:
            model_preds = df[model].values.copy()
            for d, ga in zip(
                reversed(self.differences), reversed(self.original_values_)
            ):
                if self.idxs is not None:
                    ga = ga.take(self.idxs)
                ga.restore_difference(model_preds, d)
            df[model] = model_preds
        return df

    def inverse_transform_fitted(
        self, df: pd.DataFrame, sizes: np.ndarray
    ) -> pd.DataFrame:
        model_cols = df.columns.drop([self.id_col, self.time_col])
        df = df.copy(deep=False)
        df = _ensure_shallow_copy(df)
        indptr = np.append(0, sizes.cumsum())
        for model in model_cols:
            model_preds = df[model].values.copy()
            for d, ga in zip(reversed(self.differences), reversed(self.fitted_)):
                ga.restore_fitted_difference(model_preds, indptr, d)
            df[model] = model_preds
        return df

# %% ../nbs/target_transforms.ipynb 9
@njit
def _standard_scaler_transform(data, indptr, stats, out):
    n_series = len(indptr) - 1
    for i in range(n_series):
        sl = slice(indptr[i], indptr[i + 1])
        subs = data[sl]
        mean_ = np.nanmean(subs)
        std_ = np.nanstd(subs)
        stats[i] = mean_, std_
        out[sl] = (data[sl] - mean_) / std_

# %% ../nbs/target_transforms.ipynb 10
class LocalStandardScaler(BaseTargetTransform):
    """Standardizes each serie by subtracting its mean and dividing by its standard deviation."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ga = GroupedArray.from_sorted_df(df, self.id_col, self.target_col)
        self.stats_ = np.empty((len(ga.indptr) - 1, 2))
        out = np.empty_like(ga.data)
        _standard_scaler_transform(ga.data, ga.indptr, self.stats_, out)
        df = df.copy(deep=False)
        df = _ensure_shallow_copy(df)
        df[self.target_col] = out
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=False)
        df = _ensure_shallow_copy(df)
        stats = self.stats_
        if self.idxs is not None:
            stats = stats[self.idxs]
        h = df.shape[0] // stats.shape[0]
        means = np.repeat(stats[:, 0], h)
        stds = np.repeat(stats[:, 1], h)
        model_cols = df.columns.drop([self.id_col, self.time_col])
        for model in model_cols:
            df[model] = df[model].values * stds + means
        return df

    def inverse_transform_fitted(
        self, df: pd.DataFrame, sizes: np.ndarray
    ) -> pd.DataFrame:
        df = df.copy(deep=False)
        df = _ensure_shallow_copy(df)
        means = np.repeat(self.stats_[:, 0], sizes)
        stds = np.repeat(self.stats_[:, 1], sizes)
        model_cols = df.columns.drop([self.id_col, self.time_col])
        for model in model_cols:
            df[model] = df[model].values * stds + means
        return df

# %% ../nbs/target_transforms.ipynb 12
class GlobalSklearnTransformer(BaseTargetTransform):
    """Applies the same scikit-learn transformer to all series."""

    def __init__(self, transformer: TransformerMixin):
        self.transformer = transformer

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=False)
        self.transformer_ = clone(self.transformer)
        df[self.target_col] = self.transformer_.fit_transform(
            df[[self.target_col]].values
        )
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=False)
        cols_to_transform = df.columns.drop([self.id_col, self.time_col])
        for col in cols_to_transform:
            df[col] = self.transformer_.inverse_transform(df[[col]].values)
        return df
