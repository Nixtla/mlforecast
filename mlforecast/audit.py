__all__ = ['AuditDataSeverity', 'audit_missing_dates', 'audit_duplicate_rows']


from enum import Enum
from typing import Tuple, Union

import pandas as pd
from utilsforecast.compat import DFType, pl


class AuditDataSeverity(Enum):
    """Severity levels for data audit results"""
    PASS = "pass"
    FAIL = "fail"


def audit_duplicate_rows(
    df: DFType,
    id_col: str,
    time_col: str,
) -> Tuple[AuditDataSeverity, DFType]:
    """Check for duplicate (id, timestamp) pairs in the data.

    Args:
        df: Input DataFrame (pandas or polars)
        id_col: Name of the ID column
        time_col: Name of the timestamp column

    Returns:
        Tuple of (severity, duplicates_df):
            - severity: PASS if no duplicates found, FAIL otherwise
            - duplicates_df: DataFrame containing duplicate rows (empty if PASS)
    """
    if df.shape[0] == 0:
        # Empty DataFrame - return PASS with empty result
        return (AuditDataSeverity.PASS, df.head(0))

    if isinstance(df, pd.DataFrame):
        # Pandas implementation
        duplicates = df[df.duplicated(subset=[id_col, time_col], keep=False)]
        if len(duplicates) > 0:
            return (AuditDataSeverity.FAIL, duplicates)
        return (AuditDataSeverity.PASS, duplicates)
    else:
        # Polars implementation
        duplicates = df.filter(
            pl.struct([id_col, time_col]).is_duplicated()
        )
        if duplicates.shape[0] > 0:
            return (AuditDataSeverity.FAIL, duplicates)
        return (AuditDataSeverity.PASS, duplicates)


def audit_missing_dates(
    df: DFType,
    freq: Union[str, int],
    id_col: str,
    time_col: str,
) -> Tuple[AuditDataSeverity, DFType]:
    """Check for missing dates/gaps in time series data.

    For each series (id_col), checks if there are gaps between the series'
    minimum and maximum timestamps at the specified frequency.

    Args:
        df: Input DataFrame (pandas or polars)
        freq: Frequency specification (e.g., 'D' for daily, 'H' for hourly, or integer)
        id_col: Name of the ID column
        time_col: Name of the timestamp column

    Returns:
        Tuple of (severity, missing_dates_df):
            - severity: PASS if no gaps found, FAIL otherwise
            - missing_dates_df: DataFrame with missing (id, timestamp) pairs (empty if PASS)
    """
    if df.shape[0] == 0:
        # Empty DataFrame - return PASS with empty result
        return (AuditDataSeverity.PASS, df.head(0))

    if isinstance(df, pd.DataFrame):
        # Pandas implementation
        return _audit_missing_dates_pandas(df, freq, id_col, time_col)
    else:
        # Polars implementation
        return _audit_missing_dates_polars(df, freq, id_col, time_col)


def _audit_missing_dates_pandas(
    df: pd.DataFrame,
    freq: Union[str, int],
    id_col: str,
    time_col: str,
) -> Tuple[AuditDataSeverity, pd.DataFrame]:
    """Pandas implementation of missing dates audit."""
    # Get per-series bounds (min and max for each series)
    series_bounds = df.groupby(id_col, observed=True)[time_col].agg(['min', 'max']).reset_index()
    series_bounds.columns = [id_col, 'start', 'end']

    # Check if time column is datetime or integer
    is_datetime = pd.api.types.is_datetime64_any_dtype(df[time_col])

    all_missing = []

    for _, row in series_bounds.iterrows():
        series_id = row[id_col]
        series_start = row['start']
        series_end = row['end']

        # Generate complete date range for this series (from its own start to its own end)
        if is_datetime:
            # For datetime columns
            if isinstance(freq, int):
                # Integer freq with datetime column - use period
                complete_range = pd.date_range(
                    start=series_start,
                    end=series_end,
                    periods=None,
                    freq=pd.tseries.frequencies.to_offset(freq)
                )
            else:
                # String freq (e.g., 'D', 'H')
                complete_range = pd.date_range(
                    start=series_start,
                    end=series_end,
                    freq=freq
                )
        else:
            # For integer columns
            if not isinstance(freq, int):
                raise ValueError(
                    f"For integer time columns, freq must be an integer, got {type(freq)}"
                )
            complete_range = range(int(series_start), int(series_end) + 1, freq)

        # Get actual dates for this series
        actual_dates = set(df[df[id_col] == series_id][time_col].values)

        # Find missing dates (anti-join)
        missing_dates = [d for d in complete_range if d not in actual_dates]

        if missing_dates:
            # Create DataFrame for missing dates
            missing_df = pd.DataFrame({
                id_col: [series_id] * len(missing_dates),
                time_col: missing_dates
            })
            all_missing.append(missing_df)

    if all_missing:
        result = pd.concat(all_missing, ignore_index=True)
        return (AuditDataSeverity.FAIL, result)

    # No missing dates - return empty DataFrame with correct schema
    empty_result = pd.DataFrame(columns=[id_col, time_col])
    # Preserve dtypes
    if is_datetime:
        empty_result[time_col] = pd.to_datetime(empty_result[time_col])
    return (AuditDataSeverity.PASS, empty_result)


def _convert_freq_to_polars(freq: Union[str, int]) -> str:
    """Convert pandas-style frequency to polars duration string."""
    if isinstance(freq, int):
        return f"{freq}i"

    # Map common pandas freq strings to polars duration strings
    freq_map = {
        'D': '1d',
        'H': '1h',
        'h': '1h',
        'T': '1m',
        'min': '1m',
        'S': '1s',
        's': '1s',
        'W': '1w',
        'M': '1mo',
        'Q': '1q',
        'Y': '1y',
    }

    # Check if it's already a polars-style duration (contains 'd', 'h', etc.)
    if any(c in freq for c in ['d', 'h', 'm', 's', 'w', 'mo', 'q', 'y']):
        return freq

    # Try to map it
    return freq_map.get(freq, freq)


def _audit_missing_dates_polars(
    df: 'pl.DataFrame',
    freq: Union[str, int],
    id_col: str,
    time_col: str,
) -> Tuple[AuditDataSeverity, 'pl.DataFrame']:
    """Polars implementation of missing dates audit."""
    # Get per-series bounds
    series_bounds = df.group_by(id_col).agg([
        pl.col(time_col).min().alias('start'),
        pl.col(time_col).max().alias('end')
    ])

    # Check if time column is datetime or integer
    is_datetime = df[time_col].dtype in [pl.Datetime, pl.Date]

    all_missing = []

    for row in series_bounds.iter_rows(named=True):
        series_id = row[id_col]
        series_start = row['start']
        series_end = row['end']

        # Generate complete date range for this series (from its own start to its own end)
        if is_datetime:
            # For datetime columns - convert freq to polars duration string
            freq_str = _convert_freq_to_polars(freq)

            # Use datetime_range for datetime types, date_range for date types
            if df[time_col].dtype == pl.Date:
                complete_range = pl.date_range(
                    start=series_start,
                    end=series_end,
                    interval=freq_str,
                    eager=True
                )
            else:
                # For Datetime types
                complete_range = pl.datetime_range(
                    start=series_start,
                    end=series_end,
                    interval=freq_str,
                    eager=True
                )
        else:
            # For integer columns
            if not isinstance(freq, int):
                raise ValueError(
                    f"For integer time columns, freq must be an integer, got {type(freq)}"
                )
            complete_range = list(range(int(series_start), int(series_end) + 1, freq))

        # Get actual dates for this series
        actual_dates_list = df.filter(pl.col(id_col) == series_id)[time_col].to_list()

        # Find missing dates
        if is_datetime:
            # Convert complete_range Series to list for comparison
            complete_range_list = complete_range.to_list()
            actual_dates_set = set(actual_dates_list)
            missing_dates = [d for d in complete_range_list if d not in actual_dates_set]
        else:
            actual_dates_set = set(actual_dates_list)
            missing_dates = [d for d in complete_range if d not in actual_dates_set]

        if missing_dates:
            # Create DataFrame for missing dates
            missing_df = pl.DataFrame({
                id_col: [series_id] * len(missing_dates),
                time_col: missing_dates
            })
            all_missing.append(missing_df)

    if all_missing:
        result = pl.concat(all_missing)
        return (AuditDataSeverity.FAIL, result)

    # No missing dates - return empty DataFrame with correct schema
    schema = {id_col: df[id_col].dtype, time_col: df[time_col].dtype}
    empty_result = pl.DataFrame(schema=schema)
    return (AuditDataSeverity.PASS, empty_result)
