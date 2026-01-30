import pandas as pd
import polars as pl
import pytest
from mlforecast.audit import (
    AuditDataSeverity,
    audit_duplicate_rows,
    audit_missing_dates,
)


class TestAuditDuplicateRows:
    """Test suite for audit_duplicate_rows function."""

    def test_audit_duplicate_rows_pass_pandas(self):
        """Test clean data with no duplicates (pandas)."""
        df = pd.DataFrame({
            'unique_id': ['A', 'A', 'B', 'B'],
            'ds': pd.date_range('2020-01-01', periods=4, freq='D')[:4],
            'y': [1, 2, 3, 4]
        })
        severity, duplicates = audit_duplicate_rows(df, 'unique_id', 'ds')
        assert severity == AuditDataSeverity.PASS
        assert len(duplicates) == 0

    def test_audit_duplicate_rows_fail_pandas(self):
        """Test data with duplicates (pandas)."""
        df = pd.DataFrame({
            'unique_id': ['A', 'A', 'A', 'B', 'B'],
            'ds': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-01', '2020-01-01', '2020-01-01']),
            'y': [1, 2, 3, 4, 5]
        })
        severity, duplicates = audit_duplicate_rows(df, 'unique_id', 'ds')
        assert severity == AuditDataSeverity.FAIL
        # Should find 2 duplicate rows for A (both instances of 2020-01-01)
        # and 2 duplicate rows for B (both instances of 2020-01-01)
        assert len(duplicates) == 4

    def test_audit_duplicate_rows_pass_polars(self):
        """Test clean data with no duplicates (polars)."""
        df = pl.DataFrame({
            'unique_id': ['A', 'A', 'B', 'B'],
            'ds': pd.date_range('2020-01-01', periods=4, freq='D'),
            'y': [1, 2, 3, 4]
        })
        severity, duplicates = audit_duplicate_rows(df, 'unique_id', 'ds')
        assert severity == AuditDataSeverity.PASS
        assert duplicates.shape[0] == 0

    def test_audit_duplicate_rows_fail_polars(self):
        """Test data with duplicates (polars)."""
        df = pl.DataFrame({
            'unique_id': ['A', 'A', 'A', 'B', 'B'],
            'ds': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-01', '2020-01-01', '2020-01-01']),
            'y': [1, 2, 3, 4, 5]
        })
        severity, duplicates = audit_duplicate_rows(df, 'unique_id', 'ds')
        assert severity == AuditDataSeverity.FAIL
        assert duplicates.shape[0] == 4

    def test_audit_duplicate_rows_empty_dataframe(self):
        """Test empty DataFrame (edge case)."""
        df = pd.DataFrame(columns=['unique_id', 'ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'])
        severity, duplicates = audit_duplicate_rows(df, 'unique_id', 'ds')
        assert severity == AuditDataSeverity.PASS
        assert len(duplicates) == 0


class TestAuditMissingDates:
    """Test suite for audit_missing_dates function."""

    def test_audit_missing_dates_pass_pandas(self):
        """Test data with no gaps (pandas)."""
        df = pd.DataFrame({
            'unique_id': ['A'] * 5 + ['B'] * 5,
            'ds': pd.date_range('2020-01-01', periods=5, freq='D').tolist() +
                 pd.date_range('2020-01-01', periods=5, freq='D').tolist(),
            'y': list(range(10))
        })
        severity, missing = audit_missing_dates(df, 'D', 'unique_id', 'ds')
        assert severity == AuditDataSeverity.PASS
        assert len(missing) == 0

    def test_audit_missing_dates_fail_pandas_daily(self):
        """Test data with gaps - daily frequency (pandas)."""
        # Series A: complete, Series B: has gap on 2020-01-03
        df = pd.DataFrame({
            'unique_id': ['A'] * 5 + ['B'] * 4,
            'ds': pd.date_range('2020-01-01', periods=5, freq='D').tolist() +
                 [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'),
                  pd.Timestamp('2020-01-04'), pd.Timestamp('2020-01-05')],
            'y': list(range(9))
        })
        severity, missing = audit_missing_dates(df, 'D', 'unique_id', 'ds')
        assert severity == AuditDataSeverity.FAIL
        assert len(missing) > 0
        # Should have one missing date for series B
        assert 'B' in missing['unique_id'].values
        # Check that the missing date is 2020-01-03
        missing_b = missing[missing['unique_id'] == 'B']
        assert pd.Timestamp('2020-01-03') in missing_b['ds'].values

    def test_audit_missing_dates_fail_pandas_hourly(self):
        """Test data with gaps - hourly frequency (pandas)."""
        # Create hourly data with a gap
        dates_a = pd.date_range('2020-01-01', periods=5, freq='H')
        # B has gap at hour 2
        dates_b = pd.to_datetime([
            '2020-01-01 00:00', '2020-01-01 01:00',
            '2020-01-01 03:00', '2020-01-01 04:00'
        ])
        df = pd.DataFrame({
            'unique_id': ['A'] * 5 + ['B'] * 4,
            'ds': dates_a.tolist() + dates_b.tolist(),
            'y': list(range(9))
        })
        severity, missing = audit_missing_dates(df, 'H', 'unique_id', 'ds')
        assert severity == AuditDataSeverity.FAIL
        assert len(missing) > 0
        # Should have one missing hour for series B
        assert 'B' in missing['unique_id'].values

    def test_audit_missing_dates_pass_polars(self):
        """Test data with no gaps (polars)."""
        df = pl.DataFrame({
            'unique_id': ['A'] * 5 + ['B'] * 5,
            'ds': pd.date_range('2020-01-01', periods=5, freq='D').tolist() +
                 pd.date_range('2020-01-01', periods=5, freq='D').tolist(),
            'y': list(range(10))
        })
        severity, missing = audit_missing_dates(df, 'D', 'unique_id', 'ds')
        assert severity == AuditDataSeverity.PASS
        assert missing.shape[0] == 0

    def test_audit_missing_dates_fail_polars(self):
        """Test data with gaps (polars)."""
        df = pl.DataFrame({
            'unique_id': ['A'] * 5 + ['B'] * 4,
            'ds': pd.date_range('2020-01-01', periods=5, freq='D').tolist() +
                 [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'),
                  pd.Timestamp('2020-01-04'), pd.Timestamp('2020-01-05')],
            'y': list(range(9))
        })
        severity, missing = audit_missing_dates(df, 'D', 'unique_id', 'ds')
        assert severity == AuditDataSeverity.FAIL
        assert missing.shape[0] > 0
        # Should have one missing date for series B
        assert 'B' in missing['unique_id'].to_list()

    def test_audit_missing_dates_integer_freq_pandas(self):
        """Test with integer time columns (pandas)."""
        df = pd.DataFrame({
            'unique_id': ['A'] * 5 + ['B'] * 4,
            'ds': [0, 1, 2, 3, 4] + [0, 1, 3, 4],  # B missing 2
            'y': list(range(9))
        })
        severity, missing = audit_missing_dates(df, 1, 'unique_id', 'ds')
        assert severity == AuditDataSeverity.FAIL
        assert len(missing) > 0
        # Should have one missing date for series B
        assert 'B' in missing['unique_id'].values
        missing_b = missing[missing['unique_id'] == 'B']
        assert 2 in missing_b['ds'].values

    def test_audit_missing_dates_integer_freq_polars(self):
        """Test with integer time columns (polars)."""
        df = pl.DataFrame({
            'unique_id': ['A'] * 5 + ['B'] * 4,
            'ds': [0, 1, 2, 3, 4] + [0, 1, 3, 4],
            'y': list(range(9))
        })
        severity, missing = audit_missing_dates(df, 1, 'unique_id', 'ds')
        assert severity == AuditDataSeverity.FAIL
        assert missing.shape[0] > 0
        # Should have one missing date for series B
        assert 'B' in missing['unique_id'].to_list()

    def test_audit_missing_dates_empty_dataframe(self):
        """Test empty DataFrame (edge case)."""
        df = pd.DataFrame(columns=['unique_id', 'ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'])
        severity, missing = audit_missing_dates(df, 'D', 'unique_id', 'ds')
        assert severity == AuditDataSeverity.PASS
        assert len(missing) == 0

    def test_audit_missing_dates_single_row_per_series(self):
        """Test with single row per series (no room for gaps)."""
        df = pd.DataFrame({
            'unique_id': ['A', 'B', 'C'],
            'ds': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'y': [1, 2, 3]
        })
        severity, missing = audit_missing_dates(df, 'D', 'unique_id', 'ds')
        # Each series has only one row, so start == end for each
        # No gaps possible within a single-row series
        assert severity == AuditDataSeverity.PASS
        assert len(missing) == 0

    def test_audit_missing_dates_different_series_lengths(self):
        """Test with series of different lengths."""
        # A: 2020-01-01 to 2020-01-05 (complete)
        # B: 2020-01-03 to 2020-01-05 (starts later, complete from start)
        df = pd.DataFrame({
            'unique_id': ['A'] * 5 + ['B'] * 3,
            'ds': pd.date_range('2020-01-01', periods=5, freq='D').tolist() +
                 pd.date_range('2020-01-03', periods=3, freq='D').tolist(),
            'y': list(range(8))
        })
        severity, missing = audit_missing_dates(df, 'D', 'unique_id', 'ds')
        # Both series are complete from their own start to their own end
        # So no missing dates
        assert severity == AuditDataSeverity.PASS
        assert len(missing) == 0

    def test_audit_missing_dates_per_series_end_no_false_positives(self):
        """Test that different end dates don't trigger false positives (per-series end behavior)."""
        # Scenario: Station A operates through Jan 10, Station B closed on Jan 5
        # This should NOT warn about B missing Jan 6-10
        df = pd.DataFrame({
            'unique_id': ['Station_A'] * 10 + ['Station_B'] * 5,
            'ds': pd.date_range('2020-01-01', periods=10, freq='D').tolist() +
                 pd.date_range('2020-01-01', periods=5, freq='D').tolist(),
            'y': list(range(15))
        })
        severity, missing = audit_missing_dates(df, 'D', 'unique_id', 'ds')
        # Each series is complete within its own timespan
        assert severity == AuditDataSeverity.PASS
        assert len(missing) == 0

    def test_audit_missing_dates_gap_within_shorter_series(self):
        """Test that gaps WITHIN a shorter series are still detected."""
        # Station A: complete through Jan 10
        # Station B: has gap on Jan 3, ends Jan 5
        df = pd.DataFrame({
            'unique_id': ['Station_A'] * 10 + ['Station_B'] * 4,
            'ds': pd.date_range('2020-01-01', periods=10, freq='D').tolist() +
                 [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'),
                  pd.Timestamp('2020-01-04'), pd.Timestamp('2020-01-05')],  # Missing Jan 3
            'y': list(range(14))
        })
        severity, missing = audit_missing_dates(df, 'D', 'unique_id', 'ds')
        # Should detect gap on Jan 3 in Station B
        assert severity == AuditDataSeverity.FAIL
        assert len(missing) > 0
        # Verify it's Station_B that has the issue
        assert 'Station_B' in missing['unique_id'].values
        # Verify Jan 3 is the missing date
        missing_b = missing[missing['unique_id'] == 'Station_B']
        assert pd.Timestamp('2020-01-03') in missing_b['ds'].values
        # Station A should not have any missing dates
        assert 'Station_A' not in missing['unique_id'].values

    def test_audit_missing_dates_per_series_end_polars(self):
        """Test per-series end behavior with polars."""
        # Same scenario as pandas test - different end dates should be OK
        df = pl.DataFrame({
            'unique_id': ['Station_A'] * 10 + ['Station_B'] * 5,
            'ds': pd.date_range('2020-01-01', periods=10, freq='D').tolist() +
                 pd.date_range('2020-01-01', periods=5, freq='D').tolist(),
            'y': list(range(15))
        })
        severity, missing = audit_missing_dates(df, 'D', 'unique_id', 'ds')
        assert severity == AuditDataSeverity.PASS
        assert missing.shape[0] == 0
