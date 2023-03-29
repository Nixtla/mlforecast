# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/target_transforms.ipynb.

# %% auto 0
__all__ = ['BaseTargetTransform', 'Differences']

# %% ../nbs/target_transforms.ipynb 2
import reprlib
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import pandas as pd
import numpy as np

from .grouped_array import GroupedArray, _apply_difference

# %% ../nbs/target_transforms.ipynb 3
class BaseTargetTransform:
    def set_column_names(self, id_col: str, time_col: str, target_col: str):
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

# %% ../nbs/target_transforms.ipynb 4
class Differences(BaseTargetTransform):
    def __init__(self, differences: List[int]):
        self.differences = differences

    def fit_transform(self, df: "pd.DataFrame") -> "pd.DataFrame":
        ga = GroupedArray.from_sorted_df(df, self.target_col)
        uids = df.index.unique(level=0)
        original_sizes = ga.indptr[1:].cumsum()
        total_diffs = sum(self.differences)
        small_series = uids[original_sizes < total_diffs]
        if small_series.size:
            msg = reprlib.repr(small_series.tolist())
            raise ValueError(
                f"The following series are too short for the differences: {msg}"
            )
        self.original_values_ = []
        n_series = len(ga.indptr) - 1
        for d in self.differences:
            new_data = np.empty_like(ga.data, shape=n_series * d)
            new_indptr = d * np.arange(n_series + 1, dtype=np.int32)
            _apply_difference(ga.data, ga.indptr, new_data, new_indptr, d)
            self.original_values_.append(GroupedArray(new_data, new_indptr))
        df = df.copy()
        df[self.target_col] = ga.data
        return df

    def inverse_transform(self, df: "pd.DataFrame") -> "pd.DataFrame":
        model_cols = df.columns.drop([self.id_col, self.time_col])
        df = df.copy()
        for model in model_cols:
            model_preds = df[model].values.copy()
            for d, ga in zip(
                reversed(self.differences), reversed(self.original_values_)
            ):
                ga.restore_difference(model_preds, d)
            df[model] = model_preds
        return df
