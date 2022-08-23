# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/distributed.core.ipynb.

# %% auto 0
__all__ = ['DistributedTimeSeries']

# %% ../../nbs/distributed.core.ipynb 3
import operator
from typing import Callable, List, Optional

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, default_client, futures_of, wait

from ..core import TimeSeries

# %% ../../nbs/distributed.core.ipynb 6
def _fit_transform(ts, data, **kwargs):
    df = ts.fit_transform(data, **kwargs)
    return ts, df


def _predict(ts, model, horizon, dynamic_dfs, predict_fn, **predict_fn_kwargs):
    return ts.predict(model, horizon, dynamic_dfs, predict_fn, **predict_fn_kwargs)

# %% ../../nbs/distributed.core.ipynb 7
class DistributedTimeSeries:
    """TimeSeries for distributed forecasting."""

    def __init__(
        self,
        ts: TimeSeries,
        client: Optional[Client] = None,
    ):
        self._base_ts = ts
        self.client = client or default_client()

    def fit_transform(
        self,
        data: dd.DataFrame,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> dd.DataFrame:
        """Applies the transformations to each partition of `data`."""
        self.data_divisions = data.divisions
        data = self.client.persist(data)
        wait(data)
        partition_futures = futures_of(data)
        self.ts = []
        df_futures = []
        for part_future in partition_futures:
            future = self.client.submit(
                _fit_transform,
                self._base_ts,
                part_future,
                static_features=static_features,
                dropna=dropna,
                keep_last_n=keep_last_n,
                pure=False,
            )
            ts_future = self.client.submit(operator.itemgetter(0), future)
            df_future = self.client.submit(operator.itemgetter(1), future)
            self.ts.append(ts_future)
            df_futures.append(df_future)
        meta = self.client.submit(lambda x: x.head(0), df_futures[0]).result()
        ret = dd.from_delayed(df_futures, meta=meta)
        assert not isinstance(ret, dd.Series)  # mypy
        return ret

    def predict(
        self,
        models,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        predict_fn: Optional[Callable] = None,
        **predict_fn_kwargs,
    ) -> dd.DataFrame:
        """Broadcasts `models` across all workers and computes the next `horizon` timesteps.

        `predict_fn(model, new_x, features_order, **predict_fn_kwargs)` is called on each timestep.
        """
        if not isinstance(models, list):
            models = [models]
        models_future = self.client.scatter(models, broadcast=True)
        if dynamic_dfs is not None:
            dynamic_dfs_futures = self.client.scatter(dynamic_dfs, broadcast=True)
        else:
            dynamic_dfs_futures = None
        predictions_futures = [
            self.client.submit(
                _predict,
                ts_future,
                models_future,
                horizon,
                dynamic_dfs=dynamic_dfs_futures,
                predict_fn=predict_fn,
                **predict_fn_kwargs,
            )
            for ts_future in self.ts
        ]
        meta = self.client.submit(lambda x: x.head(), predictions_futures[0]).result()
        ret = dd.from_delayed(
            predictions_futures, meta=meta, divisions=self.data_divisions
        )
        assert not isinstance(ret, dd.Series)  # mypy
        return ret

    def __repr__(self):
        ts_repr = repr(self._base_ts)
        return f"Distributed{ts_repr}"
