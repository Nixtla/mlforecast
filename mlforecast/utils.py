# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['generate_daily_series', 'generate_prices_for_series', 'backtest_splits', 'PredictionIntervals']

# %% ../nbs/utils.ipynb 3
import reprlib
import warnings
from math import ceil, log10
from typing import Generator, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utilsforecast.compat import DataFrame, Series, pl
from utilsforecast.data import generate_series
from utilsforecast.processing import (
    counts_by_id,
    filter_with_mask,
    group_by,
    is_in,
    offset_dates,
    take_rows,
)

# %% ../nbs/utils.ipynb 5
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
) -> DataFrame:
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

    if engine == "pandas":
        series["unique_id"] = (
            "id_" + series["unique_id"].astype(str).str.rjust(n_digits, "0")
        ).astype("category")
    else:
        series = series.with_columns(
            ("id_" + pl.col("unique_id").cast(pl.Utf8).str.rjust(n_digits, "0"))
            .alias("unique_id")
            .cast(pl.Categorical)
        )
    return series

# %% ../nbs/utils.ipynb 16
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

# %% ../nbs/utils.ipynb 19
def single_split(
    df: DataFrame,
    i_window: int,
    n_windows: int,
    h: int,
    id_col: str,
    time_col: str,
    freq: Union[pd.offsets.BaseOffset, int],
    max_dates: Series,
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
) -> Tuple[DataFrame, Series, Series]:
    if step_size is None:
        step_size = h
    test_size = h + step_size * (n_windows - 1)
    offset = test_size - i_window * step_size
    train_ends = offset_dates(max_dates, freq, -offset)
    valid_ends = offset_dates(train_ends, freq, h)
    train_mask = df[time_col].le(train_ends)
    valid_mask = df[time_col].gt(train_ends) & df[time_col].le(valid_ends)
    if input_size is not None:
        train_starts = offset_dates(train_ends, freq, -input_size)
        train_mask &= df[time_col].gt(train_starts)
    train_sizes = group_by(train_mask, df[id_col], maintain_order=True).sum()
    if isinstance(train_sizes, pd.Series):
        train_sizes = train_sizes.reset_index()
    zeros_mask = train_sizes[time_col].eq(0)
    if zeros_mask.all():
        raise ValueError(
            "All series are too short for the cross validation settings, "
            f"at least {offset + 1} samples are required.\n"
            "Please reduce `n_windows` or `h`."
        )
    elif zeros_mask.any():
        ids = filter_with_mask(train_sizes[id_col], zeros_mask)
        warnings.warn(
            "The following series are too short for the window "
            f"and will be dropped: {reprlib.repr(list(ids))}"
        )
        dropped_ids = is_in(df[id_col], ids)
        valid_mask &= ~dropped_ids
    if isinstance(train_ends, pd.Series):
        cutoffs: DataFrame = (
            train_ends.set_axis(df[id_col])
            .groupby(id_col, observed=True)
            .head(1)
            .rename("cutoff")
            .reset_index()
        )
    else:
        cutoffs = train_ends.to_frame().with_columns(df[id_col])
        cutoffs = (
            group_by(cutoffs, id_col)
            .agg(pl.col(time_col).head(1))
            .explode(pl.col(time_col))
            .rename({time_col: "cutoff"})
        )
    return cutoffs, train_mask, valid_mask

# %% ../nbs/utils.ipynb 20
def backtest_splits(
    df: DataFrame,
    n_windows: int,
    h: int,
    id_col: str,
    time_col: str,
    freq: Union[pd.offsets.BaseOffset, int],
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
) -> Generator[Tuple[DataFrame, DataFrame, DataFrame], None, None]:
    if isinstance(df, pd.DataFrame):
        max_dates = df.groupby(id_col, observed=True)[time_col].transform("max")
    else:
        max_dates = df.select(pl.col(time_col).max().over(id_col))[time_col]
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
        train = filter_with_mask(df, train_mask)
        valid = filter_with_mask(df, valid_mask)
        yield cutoffs, train, valid

# %% ../nbs/utils.ipynb 23
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

# %% ../nbs/utils.ipynb 24
def _ensure_shallow_copy(df: pd.DataFrame) -> pd.DataFrame:
    from packaging.version import Version

    if Version(pd.__version__) < Version("1.4"):
        # https://github.com/pandas-dev/pandas/pull/43406
        df = df.copy()
    return df

# %% ../nbs/utils.ipynb 25
class _ShortSeriesException(Exception):
    def __init__(self, idxs):
        self.idxs = idxs
