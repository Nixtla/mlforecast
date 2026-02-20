__all__ = [
    'validate_update_start_dates',
    'validate_continuity',
    'validate_update_df',
    'validate_df',
]


import reprlib
import warnings
from typing import Tuple, Union

import pandas as pd
from utilsforecast.compat import DFType, pl
from utilsforecast.processing import offset_times


def validate_update_start_dates(
    df: DFType,
    id_col: str,
    time_col: str,
    last_dates_df: DFType,
    freq: Union[str, int],
) -> Tuple[bool, DFType]:
    """Check that each series in df starts at the expected timestamp (last_date + freq).

    Args:
        df: New observations DataFrame (pandas or polars)
        id_col: Name of the ID column
        time_col: Name of the timestamp column
        last_dates_df: DataFrame with id_col and '_last' columns containing the last
            known timestamp for each series
        freq: Frequency specification. For pandas DataFrames use pandas offset
            strings (e.g. 'D', 'h'). For polars DataFrames use polars duration
            strings (e.g. '1d', '1h'). Pass an integer for integer time columns.

    Returns:
        Tuple of (severity, bad_df):
            - has_issues: False if all series start at the expected timestamp, True otherwise
            - bad_df: DataFrame with id_col for series with invalid start dates (empty if False)
    """
    if df.shape[0] == 0:
        return (False, df[[id_col]].head(0))

    if isinstance(df, pd.DataFrame):
        df_sorted = df.sort_values([id_col, time_col])
        stats = (
            df_sorted.groupby(id_col, observed=True)[time_col]
            .min()
            .rename('_min')
            .reset_index()
        )
        expected_start = offset_times(last_dates_df['_last'], freq, 1)
        expected_df = pd.DataFrame(
            {id_col: last_dates_df[id_col].astype(str), '_expected_start': expected_start}
        )
        stats[id_col] = stats[id_col].astype(str)
        stats = stats.merge(expected_df, on=id_col, how='left')
        start_mismatch = stats['_expected_start'].notna() & (
            stats['_min'] != stats['_expected_start']
        )
        bad = stats.loc[start_mismatch, [id_col]]
        if bad.shape[0] > 0:
            return (True, bad)
        return (False, bad)
    else:
        df_sorted = df.sort([id_col, time_col])
        stats = (
            df_sorted.group_by(id_col)
            .agg(pl.col(time_col).min().alias('_min'))
            .sort(id_col)
        )
        expected_start = offset_times(last_dates_df['_last'], freq, 1)
        expected_df = last_dates_df.with_columns(
            pl.Series(name='_expected_start', values=expected_start),
            pl.col(id_col).cast(pl.Utf8),
        ).select([id_col, '_expected_start'])
        stats = stats.with_columns(pl.col(id_col).cast(pl.Utf8))
        stats = stats.join(expected_df, on=id_col, how='left')
        bad = stats.filter(
            pl.col('_expected_start').is_not_null()
            & (pl.col('_min') != pl.col('_expected_start'))
        ).select([id_col])
        if bad.height > 0:
            return (True, bad)
        return (False, bad)


def validate_continuity(
    df: DFType,
    id_col: str,
    time_col: str,
    freq: Union[str, int],
) -> Tuple[bool, DFType]:
    """Check for gaps or duplicate timestamps within time series data.

    For each series, checks that consecutive timestamps are exactly one frequency
    apart. Detects both missing periods and duplicate timestamps.

    Args:
        df: Input DataFrame (pandas or polars)
        id_col: Name of the ID column
        time_col: Name of the timestamp column
        freq: Frequency specification. For pandas DataFrames use pandas offset
            strings (e.g. 'D', 'h'). For polars DataFrames use polars duration
            strings (e.g. '1d', '1h'). Pass an integer for integer time columns.

    Returns:
        Tuple of (has_issues, bad_df):
            - has_issues: False if no gaps or duplicates found, True otherwise
            - bad_df: DataFrame with id_col for affected series (empty if False)
    """
    if df.shape[0] == 0:
        return (False, df[[id_col]].head(0))

    if isinstance(df, pd.DataFrame):
        df_sorted = df.sort_values([id_col, time_col])
        expected_next = offset_times(df_sorted[time_col], freq, 1)
        next_time = df_sorted.groupby(id_col, observed=True)[time_col].shift(-1)
        gaps = next_time.notna() & (expected_next != next_time)
        bad = df_sorted.loc[gaps, [id_col]].drop_duplicates()
        if bad.shape[0] > 0:
            return (True, bad)
        return (False, bad)
    else:
        df_sorted = df.sort([id_col, time_col])
        expected_next = offset_times(df_sorted[time_col], freq, 1)
        df_check = df_sorted.with_columns(
            pl.Series(name='_expected_next', values=expected_next)
        ).with_columns(
            pl.col(time_col).shift(-1).over(id_col).alias('_next')
        )
        bad = df_check.filter(
            pl.col('_next').is_not_null()
            & (pl.col('_expected_next') != pl.col('_next'))
        ).select([id_col]).unique()
        if bad.height > 0:
            return (True, bad)
        return (False, bad)


def validate_update_df(
    df: DFType,
    id_col: str,
    time_col: str,
    uids,
    last_dates,
    freq: Union[str, int],
) -> None:
    """Validate that new data is continuous with existing series.

    Checks that every series starts at last_date + freq and contains no internal
    gaps or duplicate timestamps. Raises ValueError on the first violation found.

    Args:
        df: New observations DataFrame (pandas or polars)
        id_col: Name of the ID column
        time_col: Name of the timestamp column
        uids: Known series identifiers (pd.Index or polars Series), parallel to last_dates
        last_dates: Last known timestamp per series (pd.Index or polars Series)
        freq: Frequency specification. For pandas DataFrames use pandas offset
            strings (e.g. 'D', 'h'). For polars DataFrames use polars duration
            strings (e.g. '1d', '1h'). Pass an integer for integer time columns.

    Raises:
        ValueError: If any series has an invalid start date or internal gaps/duplicates.
    """
    if isinstance(df, pd.DataFrame):
        last_dates_df = pd.DataFrame({id_col: uids, '_last': last_dates})
    else:
        last_dates_df = pl.DataFrame({id_col: uids, '_last': last_dates})

    has_issues, bad = validate_update_start_dates(df, id_col, time_col, last_dates_df, freq)
    if has_issues:
        bad_ids = bad[id_col].tolist() if isinstance(bad, pd.DataFrame) else bad[id_col].to_list()
        raise ValueError(
            "Series have invalid start dates. "
            f"Expected start at last_date + freq for: {bad_ids}."
        )

    has_issues, bad = validate_continuity(df, id_col, time_col, freq)
    if has_issues:
        bad_ids = bad[id_col].tolist() if isinstance(bad, pd.DataFrame) else bad[id_col].to_list()
        raise ValueError(
            "Found gaps or duplicate timestamps in the update for: "
            f"{bad_ids}."
        )


def validate_df(
    df: DFType,
    id_col: str,
    time_col: str,
    freq: Union[str, int],
) -> None:
    """Run data quality validations and issue warnings if problems are found.

    Checks for gaps or duplicate timestamps using a fast consecutive-comparison
    approach. Issues a warning if problems are found without raising exceptions.

    Args:
        df: Input DataFrame (pandas or polars)
        id_col: Name of the ID column
        time_col: Name of the timestamp column
        freq: Frequency specification. For pandas DataFrames use pandas offset
            strings (e.g. 'D', 'h'). For polars DataFrames use polars duration
            strings (e.g. '1d', '1h'). Pass an integer for integer time columns.
    """
    has_issues, bad = validate_continuity(df, id_col, time_col, freq)
    if has_issues:
        bad_ids = bad[id_col].tolist() if isinstance(bad, pd.DataFrame) else bad[id_col].to_list()
        sample_ids = reprlib.repr(bad_ids[:10])
        warnings.warn(
            f"Found gaps or duplicate timestamps in time series.\n"
            f"This may lead to incorrect lag features.\n"
            f"Affected series: {sample_ids}\n"
            f"Consider using the fill_gaps parameter or preprocessing your data."
        )
