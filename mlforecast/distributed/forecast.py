# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/distributed.forecast.ipynb.

# %% auto 0
__all__ = ['DistributedForecast']

# %% ../../nbs/distributed.forecast.ipynb 6
import reprlib
import typing
from typing import Callable, Dict, List, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, default_client

from ..core import TimeSeries
from ..forecast import Forecast
from .core import DistributedTimeSeries

# %% ../../nbs/distributed.forecast.ipynb 7
_DIST_FCST = Union["LGBMForecast", "XGBForecast"]

# %% ../../nbs/distributed.forecast.ipynb 8
class DistributedForecast(Forecast):
    """Distributed pipeline encapsulation."""

    def __init__(
        self,
        models: Union[
            _DIST_FCST, List[_DIST_FCST]
        ],  # model or list of mlforecast.distributed.models
        freq: str,  # pandas offset alias, e.g. D, W, M
        lags: List[int] = [],  # list of lags to use as features
        lag_transforms: Dict[
            int, List[Tuple]
        ] = {},  # list of transformations to apply to each lag
        date_features: List[
            str
        ] = [],  # list of names of pandas date attributes to use as features, e.g. dayofweek
        num_threads: int = 1,  # number of threads to use when computing lag features
        client: Optional[Client] = None,  # dask client to use for computations
    ):
        if not isinstance(models, list):
            models = [models]
        self.models = models
        self.client = client or default_client()
        self.dts = DistributedTimeSeries(
            TimeSeries(freq, lags, lag_transforms, date_features, num_threads),
            self.client,
        )
        for model in self.models:
            model.client = self.client

    def __repr__(self) -> str:
        return (
            f'DistributedForecast(models=[{", ".join(m.__class__.__name__ for m in self.models)}], '
            f"freq={self.freq}, "
            f"lag_features={list(self.dts._base_ts.transforms.keys())}, "
            f"date_features={self.dts._base_ts.date_features}, "
            f"num_threads={self.dts._base_ts.num_threads}, "
            f"client={self.client})"
        )

    @property
    def freq(self):
        return self.dts._base_ts.freq

    def preprocess(
        self,
        data: dd.DataFrame,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> dd.DataFrame:
        """Computes the transformations on each partition of `data` and
        saves the required information for the forecasting step.
        Returns a dask dataframe with the computed features."""
        return self.dts.fit_transform(data, static_features, dropna, keep_last_n)

    def fit(
        self,
        data: dd.DataFrame,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> "DistributedForecast":
        """Perform the preprocessing and fit the model."""
        train_ddf = self.preprocess(data, static_features, dropna, keep_last_n)
        X, y = train_ddf.drop(columns=["ds", "y"]), train_ddf.y
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(
        self,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        predict_fn: Optional[Callable] = None,
        **predict_fn_kwargs,
    ) -> dd.DataFrame:
        return self.dts.predict(
            [m.model_ for m in self.models],
            horizon,
            dynamic_dfs,
            predict_fn,
            **predict_fn_kwargs,
        )

    predict.__doc__ = Forecast.predict.__doc__
