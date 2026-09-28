__all__ = [
    "CalendarFeature",
    "available",
    "year",
    "quarter",
    "month",
    "week_of_year",
    "day",
    "day_of_week",
    "day_of_year",
    "days_in_month",
    "hour",
    "minute",
    "second",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
]


from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import narwhals as nw
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalendarFeature:
    """Date feature computed the same way for pandas and polars inputs.

    Values follow the pandas conventions (e.g. ``day_of_week`` is 0 for Monday)
    regardless of the dataframe backend. Use the instances exported by this
    module (see `available`) instead of creating new ones.

    Args:
        name (str): Name of the feature, used as the column name.
        description (str): Description of the values.
        dtype (type): numpy dtype of the computed values.
        values (range, optional): Possible values of the feature. Features that
            define them are one-hot encoded when `date_features_as_dummies=True`.
    """

    name: str
    description: str = field(repr=False)
    dtype: type = field(repr=False)
    values: Optional[range] = field(repr=False)
    _compute: Callable[[nw.Series], np.ndarray] = field(repr=False, compare=False)

    def compute(self, dates) -> np.ndarray:
        """Compute the feature values.

        Args:
            dates (pandas or polars Series, or pandas DatetimeIndex): Dates to compute the feature from.

        Returns:
            numpy.ndarray: Feature values, one per date.
        """
        if isinstance(dates, pd.Index):
            dates = pd.Series(dates)
        return self._compute(nw.from_native(dates, series_only=True)).astype(self.dtype)

    def __reduce__(self):
        return _from_name, (self.name,)


def _dt(dates: nw.Series, attr: str) -> np.ndarray:
    return getattr(dates.dt, attr)().to_numpy().astype(np.int64)


def _is_leap(year: np.ndarray) -> np.ndarray:
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_DAYS_BEFORE_MONTH = np.concatenate([[0], np.cumsum(_DAYS_IN_MONTH)[:-1]])


def _days_in_month(dates: nw.Series) -> np.ndarray:
    month = _dt(dates, "month")
    return _DAYS_IN_MONTH[month - 1] + ((month == 2) & _is_leap(_dt(dates, "year")))


def _day_of_year(dates: nw.Series) -> np.ndarray:
    # narwhals' ordinal_day converts tz-aware pandas dates to UTC first
    month = _dt(dates, "month")
    leap_day = (month > 2) & _is_leap(_dt(dates, "year"))
    return _DAYS_BEFORE_MONTH[month - 1] + _dt(dates, "day") + leap_day


def _iso_weeks_in_year(year: np.ndarray) -> np.ndarray:
    def jan1_offset(y):
        return (y + y // 4 - y // 100 + y // 400) % 7

    return 52 + ((jan1_offset(year) == 4) | (jan1_offset(year - 1) == 3))


def _week_of_year(dates: nw.Series) -> np.ndarray:
    year = _dt(dates, "year")
    week = (_day_of_year(dates) - _dt(dates, "weekday") + 10) // 7
    return np.where(
        week < 1,
        _iso_weeks_in_year(year - 1),
        np.where(week > _iso_weeks_in_year(year), 1, week),
    )


def _is_month_end(dates: nw.Series) -> np.ndarray:
    return _dt(dates, "day") == _days_in_month(dates)


year = CalendarFeature("year", "Year.", np.uint16, None, lambda d: _dt(d, "year"))
quarter = CalendarFeature(
    "quarter",
    "Quarter of the year, from 1 to 4.",
    np.uint8,
    range(1, 5),
    lambda d: (_dt(d, "month") - 1) // 3 + 1,
)
month = CalendarFeature(
    "month",
    "Month of the year, from 1 to 12.",
    np.uint8,
    range(1, 13),
    lambda d: _dt(d, "month"),
)
week_of_year = CalendarFeature(
    "week_of_year",
    "ISO week of the year, from 1 to 53.",
    np.uint8,
    range(1, 54),
    _week_of_year,
)
day = CalendarFeature(
    "day",
    "Day of the month, from 1 to 31.",
    np.uint8,
    range(1, 32),
    lambda d: _dt(d, "day"),
)
day_of_week = CalendarFeature(
    "day_of_week",
    "Day of the week, from 0 (Monday) to 6 (Sunday).",
    np.uint8,
    range(7),
    lambda d: _dt(d, "weekday") - 1,
)
day_of_year = CalendarFeature(
    "day_of_year",
    "Day of the year, from 1 to 366.",
    np.uint16,
    range(1, 367),
    _day_of_year,
)
days_in_month = CalendarFeature(
    "days_in_month",
    "Number of days in the month, from 28 to 31.",
    np.uint8,
    None,
    _days_in_month,
)
hour = CalendarFeature(
    "hour",
    "Hour of the day, from 0 to 23.",
    np.uint8,
    range(24),
    lambda d: _dt(d, "hour"),
)
minute = CalendarFeature(
    "minute",
    "Minute of the hour, from 0 to 59.",
    np.uint8,
    range(60),
    lambda d: _dt(d, "minute"),
)
second = CalendarFeature(
    "second",
    "Second of the minute, from 0 to 59.",
    np.uint8,
    range(60),
    lambda d: _dt(d, "second"),
)
is_month_start = CalendarFeature(
    "is_month_start",
    "1 on the first day of the month, 0 otherwise.",
    np.uint8,
    None,
    lambda d: _dt(d, "day") == 1,
)
is_month_end = CalendarFeature(
    "is_month_end",
    "1 on the last day of the month, 0 otherwise.",
    np.uint8,
    None,
    _is_month_end,
)
is_quarter_start = CalendarFeature(
    "is_quarter_start",
    "1 on the first day of the quarter, 0 otherwise.",
    np.uint8,
    None,
    lambda d: (_dt(d, "day") == 1) & (_dt(d, "month") % 3 == 1),
)
is_quarter_end = CalendarFeature(
    "is_quarter_end",
    "1 on the last day of the quarter, 0 otherwise.",
    np.uint8,
    None,
    lambda d: _is_month_end(d) & (_dt(d, "month") % 3 == 0),
)
is_year_start = CalendarFeature(
    "is_year_start",
    "1 on January 1st, 0 otherwise.",
    np.uint8,
    None,
    lambda d: (_dt(d, "day") == 1) & (_dt(d, "month") == 1),
)
is_year_end = CalendarFeature(
    "is_year_end",
    "1 on December 31st, 0 otherwise.",
    np.uint8,
    None,
    lambda d: (_dt(d, "day") == 31) & (_dt(d, "month") == 12),
)

_FEATURES: Dict[str, CalendarFeature] = {
    f.name: f
    for f in [
        year,
        quarter,
        month,
        week_of_year,
        day,
        day_of_week,
        day_of_year,
        days_in_month,
        hour,
        minute,
        second,
        is_month_start,
        is_month_end,
        is_quarter_start,
        is_quarter_end,
        is_year_start,
        is_year_end,
    ]
}

# string date features that are one-hot encoded when date_features_as_dummies=True
_DUMMY_ALIASES: Dict[str, CalendarFeature] = {
    "dayofweek": day_of_week,
    "day_of_week": day_of_week,
    "weekday": day_of_week,
    "month": month,
    "quarter": quarter,
    "day": day,
    "hour": hour,
    "minute": minute,
    "second": second,
    "dayofyear": day_of_year,
    "day_of_year": day_of_year,
    "week": week_of_year,
    "weekofyear": week_of_year,
}


def _from_name(name: str) -> CalendarFeature:
    return _FEATURES[name]


def available() -> List[CalendarFeature]:
    """List the calendar features that can be used as `date_features`.

    Returns:
        list of CalendarFeature: Available features.
    """
    return list(_FEATURES.values())
