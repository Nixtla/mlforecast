__all__ = ['AuditDataSeverity', 'audit_missing_dates', 'audit_duplicate_rows']


from enum import Enum
from typing import Tuple, Union

import pandas as pd
from utilsforecast.compat import DFType, pl
from utilsforecast.preprocessing import fill_gaps
from utilsforecast.processing import anti_join


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
        return (AuditDataSeverity.PASS, df.head(0))

    if isinstance(df, pd.DataFrame):
        duplicates = df[df.duplicated(subset=[id_col, time_col], keep=False)]
        if len(duplicates) > 0:
            return (AuditDataSeverity.FAIL, duplicates)
        return (AuditDataSeverity.PASS, duplicates)
    else:
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
        freq: Frequency specification. For pandas DataFrames use pandas offset
            strings (e.g. 'D', 'h'). For polars DataFrames use polars duration
            strings (e.g. '1d', '1h'). Pass an integer for integer time columns.
        id_col: Name of the ID column
        time_col: Name of the timestamp column

    Returns:
        Tuple of (severity, missing_dates_df):
            - severity: PASS if no gaps found, FAIL otherwise
            - missing_dates_df: DataFrame with missing (id, timestamp) pairs (empty if PASS)
    """
    if df.shape[0] == 0:
        return (AuditDataSeverity.PASS, df[[id_col, time_col]].head(0))

    filled = fill_gaps(
        df, freq=freq, start='per_serie', end='per_serie',
        id_col=id_col, time_col=time_col,
    )
    missing = anti_join(
        filled[[id_col, time_col]],
        df[[id_col, time_col]],
        on=[id_col, time_col],
    )
    if missing.shape[0] == 0:
        return (AuditDataSeverity.PASS, missing)
    return (AuditDataSeverity.FAIL, missing)
