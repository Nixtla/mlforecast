# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/target_transforms.ipynb.

# %% auto 0
__all__ = ['BaseTargetTransform', 'BaseGroupedArrayTargetTransform', 'Differences', 'LocalStandardScaler', 'LocalMinMaxScaler',
           'LocalRobustScaler', 'LocalBoxCox', 'GlobalSklearnTransformer']

# %% ../nbs/target_transforms.ipynb 3
import abc
import copy
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, clone
from utilsforecast.compat import DataFrame
from utilsforecast.target_transforms import (
    LocalBoxCox as BoxCox,
    LocalMinMaxScaler as MinMaxScaler,
    LocalRobustScaler as RobustScaler,
    LocalStandardScaler as StandardScaler,
    _common_scaler_inverse_transform,
    _transform,
)

from .grouped_array import GroupedArray, _apply_difference
from .utils import _ShortSeriesException

# %% ../nbs/target_transforms.ipynb 5
class BaseTargetTransform(abc.ABC):
    """Base class used for target transformations."""

    def set_column_names(self, id_col: str, time_col: str, target_col: str):
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

    @abc.abstractmethod
    def fit_transform(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError

# %% ../nbs/target_transforms.ipynb 6
class BaseGroupedArrayTargetTransform(abc.ABC):
    """Base class used for target transformations that operate on grouped arrays."""

    idxs: Optional[np.ndarray] = None

    @abc.abstractmethod
    def fit_transform(self, ga: GroupedArray) -> GroupedArray:
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, ga: GroupedArray) -> GroupedArray:
        raise NotImplementedError

    def inverse_transform_fitted(self, ga: GroupedArray) -> GroupedArray:
        return self.inverse_transform(ga)

# %% ../nbs/target_transforms.ipynb 7
class Differences(BaseGroupedArrayTargetTransform):
    """Subtracts previous values of the serie. Can be used to remove trend or seasonalities."""

    store_fitted = False

    def __init__(self, differences: Iterable[int]):
        self.differences = list(differences)

    def fit_transform(self, ga: GroupedArray) -> GroupedArray:
        ga = copy.copy(ga)
        self.fitted_: List[GroupedArray] = []
        original_sizes = np.diff(ga.indptr)
        total_diffs = sum(self.differences)
        small_series = original_sizes < total_diffs
        if small_series.any():
            raise _ShortSeriesException(np.arange(ga.n_groups)[small_series])
        self.original_values_ = []
        n_series = len(ga.indptr) - 1
        for d in self.differences:
            if self.store_fitted:
                # these are saved in order to be able to perform a correct
                # inverse transform when trying to retrieve the fitted values.
                self.fitted_.append(copy.copy(ga))
            new_data = np.empty_like(ga.data, shape=n_series * d)
            new_indptr = d * np.arange(n_series + 1, dtype=np.int32)
            _apply_difference(ga.data, ga.indptr, new_data, new_indptr, d)
            self.original_values_.append(GroupedArray(new_data, new_indptr))
        return ga

    def inverse_transform(self, ga: GroupedArray) -> GroupedArray:
        ga = copy.copy(ga)
        for d, orig_vals_ga in zip(
            reversed(self.differences), reversed(self.original_values_)
        ):
            if self.idxs is not None:
                orig_vals_ga = orig_vals_ga.take(self.idxs)
            orig_vals_ga.restore_difference(ga.data, d)
        return ga

    def inverse_transform_fitted(self, ga: GroupedArray) -> pd.DataFrame:
        ga = copy.copy(ga)
        for d, orig_vals_ga in zip(reversed(self.differences), reversed(self.fitted_)):
            orig_vals_ga.restore_fitted_difference(ga.data, ga.indptr, d)
        return ga

# %% ../nbs/target_transforms.ipynb 10
class BaseLocalScaler(BaseGroupedArrayTargetTransform):
    """Standardizes each serie by subtracting its mean and dividing by its standard deviation."""

    scaler_factory: type

    def fit_transform(self, ga: GroupedArray) -> GroupedArray:
        self.scaler_ = self.scaler_factory()
        transformed = self.scaler_.fit_transform(ga)
        return GroupedArray(transformed, ga.indptr)

    def inverse_transform(self, ga: GroupedArray) -> GroupedArray:
        stats = self.scaler_.stats_
        if self.idxs is not None:
            stats = stats[self.idxs]
        transformed = _transform(
            ga.data, ga.indptr, stats, _common_scaler_inverse_transform
        )
        return GroupedArray(transformed, ga.indptr)

    def inverse_transform_fitted(self, ga: GroupedArray) -> GroupedArray:
        return self.inverse_transform(ga)

# %% ../nbs/target_transforms.ipynb 12
class LocalStandardScaler(BaseLocalScaler):
    """Standardizes each serie by subtracting its mean and dividing by its standard deviation."""

    scaler_factory = StandardScaler

# %% ../nbs/target_transforms.ipynb 14
class LocalMinMaxScaler(BaseLocalScaler):
    """Scales each serie to be in the [0, 1] interval."""

    scaler_factory = MinMaxScaler

# %% ../nbs/target_transforms.ipynb 16
class LocalRobustScaler(BaseLocalScaler):
    """Scaler robust to outliers.

    Parameters
    ----------
    scale : str (default='iqr')
        Statistic to use for scaling. Can be either 'iqr' (Inter Quartile Range) or 'mad' (Median Asbolute Deviation)
    """

    def __init__(self, scale: str):
        self.scaler_factory = lambda: RobustScaler(scale)  # type: ignore

# %% ../nbs/target_transforms.ipynb 19
class LocalBoxCox(BaseLocalScaler):
    """Finds the optimum lambda for each serie and applies the Box-Cox transformation"""

    def __init__(self):
        self.scaler = BoxCox()

    def fit_transform(self, ga: GroupedArray) -> GroupedArray:
        return GroupedArray(self.scaler.fit_transform(ga), ga.indptr)

    def inverse_transform(self, ga: GroupedArray) -> GroupedArray:
        from scipy.special import inv_boxcox1p

        sizes = np.diff(ga.indptr)
        lmbdas = self.scaler.lmbdas_
        if self.idxs is not None:
            lmbdas = lmbdas[self.idxs]
        lmbdas = np.repeat(lmbdas, sizes, axis=0)
        return GroupedArray(inv_boxcox1p(ga.data, lmbdas), ga.indptr)

# %% ../nbs/target_transforms.ipynb 21
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
