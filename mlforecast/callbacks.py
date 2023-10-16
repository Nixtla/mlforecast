# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/callbacks.ipynb.

# %% auto 0
__all__ = ['SaveFeatures']

# %% ../nbs/callbacks.ipynb 3
import pandas as pd

from utilsforecast.compat import DataFrame
from utilsforecast.processing import (
    assign_columns,
    drop_index_if_pandas,
    vertical_concat,
)

# %% ../nbs/callbacks.ipynb 4
class SaveFeatures:
    """Saves the features in every timestamp."""

    def __init__(self):
        self._inputs = []

    def __call__(self, new_x):
        self._inputs.append(new_x)
        return new_x

    def get_features(self, with_step: bool = False) -> DataFrame:
        """Retrieves the input features for every timestep

        Parameters
        ----------
        with_step : bool
            Add a column indicating the step

        Returns
        -------
        pandas or polars DataFrame
            DataFrame with input features
        """
        if not self._inputs:
            raise ValueError(
                "Inputs list is empty. "
                "Call `predict` using this callback as before_predict_callback"
            )
        if with_step:
            dfs = [assign_columns(df, "step", i) for i, df in enumerate(self._inputs)]
        else:
            dfs = self._inputs
        res = vertical_concat(dfs)
        res = drop_index_if_pandas(res)
        return res
