# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/forecast.ipynb (unless otherwise specified).

__all__ = ['Forecast']

# Cell
from typing import Callable, Generator, List, Optional

import pandas as pd

from .core import TimeSeries
from .utils import backtest_splits


# Cell
class Forecast:
    """Full pipeline encapsulation.

    Takes a model (scikit-learn compatible regressor) and TimeSeries
    and runs all the forecasting pipeline."""

    def __init__(self, model, ts: TimeSeries):
        self.model = model
        self.ts = ts

    def __repr__(self):
        return f'Forecast(model={self.model}, ts={self.ts})'

    def preprocess(
        self,
        data: pd.DataFrame,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.ts.fit_transform(data, static_features, dropna, keep_last_n)

    def fit(
        self,
        data: pd.DataFrame,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        **fit_kwargs,
    ) -> 'Forecast':
        """Preprocesses `data` and fits `model` to it."""
        series_df = self.preprocess(data, static_features, dropna, keep_last_n)
        X, y = series_df.drop(columns=['ds', 'y']), series_df.y.values
        del series_df
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(
        self,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        predict_fn: Optional[Callable] = None,
        **predict_fn_kwargs,
    ) -> pd.DataFrame:
        """Compute the predictions for the next `horizon` steps.

        `predict_fn(model, new_x, features_order, **predict_fn_kwargs)` is called in every timestep, where:
        `model` is the trained model.
        `new_x` is a dataframe with the same format as the input plus the computed features.
        `features_order` is the list of column names that were used in the training step.
        """
        return self.ts.predict(
            self.model, horizon, dynamic_dfs, predict_fn, **predict_fn_kwargs
        )

    def backtest(
        self,
        data: pd.DataFrame,
        n_windows: int,
        window_size: int,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        predict_fn: Optional[Callable] = None,
        **predict_fn_kwargs,
    ) -> Generator[pd.DataFrame, None, None]:
        """Creates `n_windows` splits of `window_size` from `data`, trains the model
        on the training set, predicts the window and merges the actuals and the predictions
        in a dataframe.

        Returns a generator to the dataframes containing the datestamps, actual values
        and predictions."""
        for train, valid in backtest_splits(data, n_windows, window_size):
            self.fit(train, static_features, dropna, keep_last_n)
            y_pred = self.predict(
                window_size, dynamic_dfs, predict_fn, **predict_fn_kwargs
            )
            y_valid = valid[['ds', 'y']]
            result = y_valid.merge(y_pred, on=['unique_id', 'ds'], how='left')
            yield result
