import warnings
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd
from joblib import cpu_count

from mlforecast.utils import generate_daily_series, generate_prices_for_series, _resolve_num_threads

from .conftest import assert_raises_with_message


@pytest.fixture
def setup():
    n_series = 20
    min_length = 100
    max_length = 1000
    n_static_features = 2

    return n_series, min_length, max_length, n_static_features


def test_generate_daily_series(setup):
    n_series, min_length, max_length, n_static_features = setup

    series_with_statics = generate_daily_series(
        n_series, min_length, max_length, n_static_features
    )

    for i in range(n_static_features):
        assert all(
            series_with_statics.groupby("unique_id")[f"static_{i}"].nunique() == 1
        )
    assert series_with_statics.groupby("unique_id")["ds"].max().nunique() > 1


def test_generate_daily_series_equal_ends(setup):
    n_series, min_length, max_length, n_static_features = setup

    series = generate_daily_series(n_series, min_length, max_length)

    series_equal_ends = generate_daily_series(
        n_series, min_length, max_length, n_static_features, equal_ends=True
    )
    assert series_equal_ends.groupby("unique_id")["ds"].max().nunique() == 1
    series_for_prices = generate_daily_series(20, n_static_features=2, equal_ends=True)
    series_for_prices.rename(columns={"static_1": "product_id"}, inplace=True)
    prices_catalog = generate_prices_for_series(series_for_prices, horizon=7)

    assert set(prices_catalog["unique_id"]) == set(series_for_prices["unique_id"])
    assert_raises_with_message(lambda: generate_prices_for_series(series), "equal ends")


@pytest.mark.parametrize("horizon", [0, 1, 7])
@pytest.mark.parametrize(
    "start, n_days, timezone",
    [
        ("2026-03-06", 5, "America/New_York"),
        ("2026-03-05", 3, "America/New_York"),
        ("2026-10-30", 5, "America/New_York"),
        ("2026-10-29", 3, "America/New_York"),
        ("2026-03-27", 5, "Europe/London"),
        ("2026-10-23", 5, "Europe/London"),
        ("2026-03-06 12:00", 5, "America/New_York"),
        ("2026-10-29 12:00", 3, "America/New_York"),
        ("2026-03-06", 5, None),
        ("2026-03-06", 5, "UTC"),
        ("2026-03-06", 5, "Asia/Tokyo"),
    ],
)
def test_generate_prices_for_series_daily_dates(start, n_days, timezone, horizon):
    dates = pd.date_range(start, periods=n_days, freq="D", tz=timezone)
    series = pd.concat(
        [
            pd.DataFrame({"unique_id": "a", "ds": dates}),
            pd.DataFrame({"unique_id": "b", "ds": dates[1:]}),
        ],
        ignore_index=True,
    )
    series["unique_id"] = pd.Categorical(
        series["unique_id"], categories=["a", "b", "unused"]
    )
    prices = generate_prices_for_series(series, horizon=horizon, seed=13)
    assert set(prices["unique_id"]) == {"a", "b"}
    rng = np.random.RandomState(13)
    for uid, start_idx in [("a", 0), ("b", 1)]:
        actual = prices.loc[prices["unique_id"] == uid]
        expected = pd.date_range(
            dates[start_idx], periods=n_days - start_idx + horizon, freq="D"
        )
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(actual["ds"]), expected, check_names=False
        )
        np.testing.assert_array_equal(actual["price"], rng.rand(len(expected)))


def test_resolve_num_threads_minus_one():
    """Test that num_threads=-1 converts to actual CPU count."""
    resolved = _resolve_num_threads(-1)
    expected = cpu_count()
    assert resolved == expected
    assert isinstance(resolved, int)
    assert resolved >= 1


def test_resolve_num_threads_cpu_count_returns_none():
    """Test fallback when joblib.cpu_count() returns None."""
    with patch('mlforecast.utils.cpu_count', return_value=None):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_num_threads(-1)
            assert result == 1
            assert len(w) == 1
            assert "Could not determine CPU count" in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)


def test_resolve_num_threads_cpu_count_raises_exception():
    """Test fallback when joblib.cpu_count() raises an exception."""
    with patch('mlforecast.utils.cpu_count', side_effect=RuntimeError("Test error")):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_num_threads(-1)
            assert result == 1
            assert len(w) == 1
            assert "Error determining CPU count" in str(w[0].message)
            assert "Test error" in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)


def test_resolve_num_threads_zero_raises():
    """Regression test: num_threads=0 must raise ValueError, not ZeroDivisionError."""
    with pytest.raises(ValueError, match="num_threads must be -1 or a positive integer"):
        _resolve_num_threads(0)


def test_resolve_num_threads_negative_raises():
    """Any negative value other than -1 should raise ValueError."""
    with pytest.raises(ValueError, match="num_threads must be -1 or a positive integer"):
        _resolve_num_threads(-2)
