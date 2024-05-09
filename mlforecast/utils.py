# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['generate_daily_series', 'generate_prices_for_series', 'PredictionIntervals']

# %% ../nbs/utils.ipynb 3
from math import ceil, log10

import numpy as np
import pandas as pd

from utilsforecast.compat import DataFrame, pl
from utilsforecast.data import generate_series

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
    unique_last_dates = series.groupby("unique_id", observed=True)["ds"].max().nunique()
    if unique_last_dates > 1:
        raise ValueError("series must have equal ends.")
    day_offset = pd.tseries.frequencies.Day()
    starts_ends = series.groupby("unique_id", observed=True)["ds"].agg(["min", "max"])
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

# %% ../nbs/utils.ipynb 20
def _ensure_shallow_copy(df: pd.DataFrame) -> pd.DataFrame:
    from packaging.version import Version

    if Version(pd.__version__) < Version("1.4"):
        # https://github.com/pandas-dev/pandas/pull/43406
        df = df.copy()
    return df

# %% ../nbs/utils.ipynb 21
class _ShortSeriesException(Exception):
    def __init__(self, idxs):
        self.idxs = idxs
