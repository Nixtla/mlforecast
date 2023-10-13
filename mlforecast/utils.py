# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['generate_daily_series', 'generate_prices_for_series', 'backtest_splits', 'PredictionIntervals']

# %% ../nbs/utils.ipynb 2
import reprlib
from math import ceil, log10
from typing import Optional, Union

import numpy as np
import pandas as pd
from utilsforecast.data import generate_series

# %% ../nbs/utils.ipynb 4
def generate_daily_series(
    n_series: int,
    min_length: int = 50,
    max_length: int = 500,
    n_static_features: int = 0,
    equal_ends: bool = False,
    static_as_categorical: bool = True,
    with_trend: bool = False,
    seed: int = 0,
    engine: str = "pandas",
) -> pd.DataFrame:
    """Generate Synthetic Panel Series.

    Parameters
    ----------
    n_series : int
        Number of series for synthetic panel.
    min_length : int (default=50)
        Minimum length of synthetic panel's series.
    max_length : int (default=500)
        Maximum length of synthetic panel's series.
    n_static_features : int (default=0)
        Number of static exogenous variables for synthetic panel's series.
    equal_ends : bool (default=False)
        Series should end in the same date stamp `ds`.
    static_as_categorical : bool (default=True)
        Static features should have a categorical data type.
    with_trend : bool (default=False)
        Series should have a (positive) trend.
    seed : int (default=0)
        Random seed used for generating the data.
    engine : str (default='pandas')
        Output Dataframe type.

    Returns
    -------
    pandas or polars DataFrame
        Synthetic panel with columns [`unique_id`, `ds`, `y`] and exogenous features.
    """
    series = generate_series(
        n_series=n_series,
        freq="D",
        min_length=min_length,
        max_length=max_length,
        n_static_features=n_static_features,
        equal_ends=equal_ends,
        static_as_categorical=static_as_categorical,
        with_trend=with_trend,
        seed=seed,
        engine=engine,
    )
    n_digits = ceil(log10(n_series))

    def int_id_to_str(uid):
        return f"id_{uid:0{n_digits}}"

    if engine == "pandas":
        series["unique_id"] = series["unique_id"].map(int_id_to_str).astype("category")
    else:
        import polars as pl

        series = series.with_columns(
            pl.col("unique_id").map_elements(int_id_to_str).cast(pl.Categorical)
        )
    return series

# %% ../nbs/utils.ipynb 15
def generate_prices_for_series(
    series: pd.DataFrame, horizon: int = 7, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    unique_last_dates = series.groupby("unique_id")["ds"].max().nunique()
    if unique_last_dates > 1:
        raise ValueError("series must have equal ends.")
    day_offset = pd.tseries.frequencies.Day()
    starts_ends = series.groupby("unique_id")["ds"].agg([min, max])
    dfs = []
    for idx, (start, end) in starts_ends.iterrows():
        product_df = pd.DataFrame(
            {
                "unique_id": idx,
                "price": rng.rand((end - start).days + 1 + horizon),
            },
            index=pd.date_range(start, end + horizon * day_offset, name="ds"),
        )
        dfs.append(product_df)
    prices_catalog = pd.concat(dfs).reset_index()
    return prices_catalog

# %% ../nbs/utils.ipynb 18
def single_split(
    df: pd.DataFrame,
    i_window: int,
    n_windows: int,
    h: int,
    id_col: str,
    time_col: str,
    freq: Union[pd.offsets.BaseOffset, int],
    max_dates: pd.Series,
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
):
    if step_size is None:
        step_size = h
    test_size = h + step_size * (n_windows - 1)
    offset = test_size - i_window * step_size
    train_ends = max_dates - offset * freq
    valid_ends = train_ends + h * freq
    train_mask = df[time_col].le(train_ends)
    if input_size is not None:
        train_mask &= df[time_col].gt(train_ends - input_size * freq)
    train_sizes = train_mask.groupby(df[id_col], observed=True).sum()
    if train_sizes.eq(0).any():
        ids = reprlib.repr(train_sizes[train_sizes.eq(0)].index.tolist())
        raise ValueError(f"The following series are too short for the window: {ids}")
    valid_mask = df[time_col].gt(train_ends) & df[time_col].le(valid_ends)
    cutoffs = (
        train_ends.set_axis(df[id_col])
        .groupby(id_col, observed=True)
        .head(1)
        .rename("cutoff")
    )
    return cutoffs, train_mask, valid_mask

# %% ../nbs/utils.ipynb 19
def backtest_splits(
    df: pd.DataFrame,
    n_windows: int,
    h: int,
    id_col: str,
    time_col: str,
    freq: Union[pd.offsets.BaseOffset, int],
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
):
    max_dates = df.groupby(id_col, observed=True)[time_col].transform("max")
    for i in range(n_windows):
        cutoffs, train_mask, valid_mask = single_split(
            df,
            i_window=i,
            n_windows=n_windows,
            h=h,
            id_col=id_col,
            time_col=time_col,
            freq=freq,
            max_dates=max_dates,
            step_size=step_size,
            input_size=input_size,
        )
        train, valid = df[train_mask], df[valid_mask]
        yield cutoffs, train, valid

# %% ../nbs/utils.ipynb 21
class PredictionIntervals:
    """Class for storing prediction intervals metadata information."""

    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = "conformal_distribution",
    ):
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed_methods = ["conformal_error", "conformal_distribution"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.n_windows = n_windows
        self.h = h
        self.method = method

    def __repr__(self):
        return f"PredictionIntervals(n_windows={self.n_windows}, h={self.h}, method='{self.method}')"

# %% ../nbs/utils.ipynb 22
def _ensure_shallow_copy(df: pd.DataFrame) -> pd.DataFrame:
    from packaging.version import Version

    if Version(pd.__version__) < Version("1.4"):
        # https://github.com/pandas-dev/pandas/pull/43406
        df = df.copy()
    return df

# %% ../nbs/utils.ipynb 26
class _ShortSeriesException(Exception):
    def __init__(self, idxs):
        self.idxs = idxs
