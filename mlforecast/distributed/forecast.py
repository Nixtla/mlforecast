# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/distributed.forecast.ipynb.

# %% auto 0
__all__ = ['DistributedForecast']

# %% ../../nbs/distributed.forecast.ipynb 5
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, default_client
from sklearn.base import clone

from ..core import TimeSeries
from ..forecast import Forecast
from .core import DistributedTimeSeries

# %% ../../nbs/distributed.forecast.ipynb 7
class DistributedForecast(Forecast):
    """Distributed pipeline encapsulation."""

    def __init__(
        self,
        models,  # model or list of mlforecast.distributed.models
        freq: Optional[
            str
        ] = None,  # pandas offset alias, e.g. D, W, M. Don't set if you're using integer times.
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
            models = [clone(models)]
        self.models = [clone(m) for m in models]
        self.client = client or default_client()
        self.dts = DistributedTimeSeries(
            TimeSeries(freq, lags, lag_transforms, date_features, num_threads),
            self.client,
        )

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
        id_col: str = "unique_id",  # column that identifies each serie, it's recommended to have this as the index.
        time_col: str = "ds",  # column with the timestamps
        target_col: str = "y",  # column with the series values
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> dd.DataFrame:
        """Computes the transformations on each partition of `data` and
        saves the required information for the forecasting step.
        Returns a dask dataframe with the computed features."""
        if id_col in data:
            warnings.warn(
                "It is recommended to have id_col as the index, since setting the index is a slow operation."
            )
            data = data.set_index(id_col)
        return self.dts.fit_transform(
            data, id_col, time_col, target_col, static_features, dropna, keep_last_n
        )

    def fit(
        self,
        data: dd.DataFrame,
        id_col: str = "unique_id",  # column that identifies each serie, it's recommended to have this as the index.
        time_col: str = "ds",  # column with the timestamps
        target_col: str = "y",  # column with the series values
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> "DistributedForecast":
        """Perform the preprocessing and fit the model."""
        train_ddf = self.preprocess(
            data, id_col, time_col, target_col, static_features, dropna, keep_last_n
        )
        X, y = train_ddf.drop(columns=[time_col, target_col]), train_ddf[target_col]
        self.fitted_models = []
        for i, model in enumerate(self.models):
            model = clone(model)
            model.client = self.client
            self.fitted_models.append(model.fit(X, y))
        return self

    def predict(
        self,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        predict_fn: Optional[Callable] = None,
        **predict_fn_kwargs,
    ) -> dd.DataFrame:
        return self.dts.predict(
            [m.model_ for m in self.fitted_models],
            horizon,
            dynamic_dfs,
            predict_fn,
            **predict_fn_kwargs,
        )

    predict.__doc__ = Forecast.predict.__doc__
