# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/lag_transforms.ipynb.

# %% auto 0
__all__ = ['RollingMean', 'RollingStd', 'RollingMin', 'RollingMax', 'RollingQuantile', 'SeasonalRollingMean',
           'SeasonalRollingStd', 'SeasonalRollingMin', 'SeasonalRollingMax', 'SeasonalRollingQuantile', 'ExpandingMean',
           'ExpandingStd', 'ExpandingMin', 'ExpandingMax', 'ExpandingQuantile', 'ExponentiallyWeightedMean']

# %% ../nbs/lag_transforms.ipynb 3
import inspect
from typing import Optional

import numpy as np

try:
    import coreforecast.lag_transforms as core_tfms
    from coreforecast.grouped_array import GroupedArray as CoreGroupedArray
except ImportError:
    raise ImportError(
        "The lag_transforms module requires the coreforecast package. "
        "Please install it with `pip install coreforecast`.\n"
        'You can also install mlforecast with the lag_transforms extra: `pip install "mlforecast[lag_transforms]"`'
    ) from None
from sklearn.base import BaseEstimator

# %% ../nbs/lag_transforms.ipynb 4
class BaseLagTransform(BaseEstimator):
    def _set_core_tfm(self, lag: int) -> "BaseLagTransform":
        init_args = {
            k: getattr(self, k) for k in inspect.signature(self.__class__).parameters
        }
        self._core_tfm = getattr(core_tfms, self.__class__.__name__)(
            lag=lag, **init_args
        )
        return self

    def transform(self, ga: CoreGroupedArray) -> np.ndarray:
        return self._core_tfm.transform(ga)

    def update(self, ga: CoreGroupedArray) -> np.ndarray:
        return self._core_tfm.update(ga)

# %% ../nbs/lag_transforms.ipynb 5
class Lag(BaseLagTransform):
    def __init__(self, lag: int):
        self.lag = lag
        self._core_tfm = core_tfms.Lag(lag=lag)

    def __eq__(self, other):
        return isinstance(other, Lag) and self.lag == other.lag

# %% ../nbs/lag_transforms.ipynb 6
class _RollingBase(BaseLagTransform):
    "Rolling statistic"

    def __init__(self, window_size: int, min_samples: Optional[int] = None):
        """
        Parameters
        ----------
        window_size : int
            Number of samples in the window.
        min_samples: int
            Minimum samples required to output the statistic.
            If `None`, will be set to `window_size`.
        """
        self.window_size = window_size
        self.min_samples = min_samples

# %% ../nbs/lag_transforms.ipynb 7
class RollingMean(_RollingBase):
    ...


class RollingStd(_RollingBase):
    ...


class RollingMin(_RollingBase):
    ...


class RollingMax(_RollingBase):
    ...


class RollingQuantile(_RollingBase):
    def __init__(self, p: float, window_size: int, min_samples: Optional[int] = None):
        super().__init__(window_size=window_size, min_samples=min_samples)
        self.p = p

    def _set_core_tfm(self, lag: int):
        self._core_tfm = core_tfms.RollingQuantile(
            lag=lag,
            p=self.p,
            window_size=self.window_size,
            min_samples=self.min_samples,
        )
        return self

# %% ../nbs/lag_transforms.ipynb 9
class _Seasonal_RollingBase(BaseLagTransform):
    """Rolling statistic over seasonal periods"""

    def __init__(
        self, season_length: int, window_size: int, min_samples: Optional[int] = None
    ):
        """
        Parameters
        ----------
        season_length : int
            Periodicity of the seasonal period.
        window_size : int
            Number of samples in the window.
        min_samples: int
            Minimum samples required to output the statistic.
            If `None`, will be set to `window_size`.
        """
        self.season_length = season_length
        self.window_size = window_size
        self.min_samples = min_samples

# %% ../nbs/lag_transforms.ipynb 10
class SeasonalRollingMean(_Seasonal_RollingBase):
    ...


class SeasonalRollingStd(_Seasonal_RollingBase):
    ...


class SeasonalRollingMin(_Seasonal_RollingBase):
    ...


class SeasonalRollingMax(_Seasonal_RollingBase):
    ...


class SeasonalRollingQuantile(_Seasonal_RollingBase):
    def __init__(
        self,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
    ):
        super().__init__(
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
        )
        self.p = p

# %% ../nbs/lag_transforms.ipynb 12
class _ExpandingBase(BaseLagTransform):
    """Expanding statistic"""

    def __init__(self):
        ...

# %% ../nbs/lag_transforms.ipynb 13
class ExpandingMean(_ExpandingBase):
    ...


class ExpandingStd(_ExpandingBase):
    ...


class ExpandingMin(_ExpandingBase):
    ...


class ExpandingMax(_ExpandingBase):
    ...


class ExpandingQuantile(_ExpandingBase):
    def __init__(self, p: float):
        self.p = p

# %% ../nbs/lag_transforms.ipynb 15
class ExponentiallyWeightedMean(BaseLagTransform):
    """Exponentially weighted average

    Parameters
    ----------
    alpha : float
        Smoothing factor."""

    def __init__(self, alpha: float):
        self.alpha = alpha
