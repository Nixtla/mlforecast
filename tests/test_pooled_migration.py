# tests/test_pooled_migration.py
"""A model saved by the numpy engine must be migratable without shipping any
legacy classes in the library."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean

from ._pooled_engine_env import pooled_engine


def _panel(n_series=20, n_times=60, n_groups=4, seed=0, backend="polars"):
    rng = np.random.default_rng(seed)
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1),
        pl.datetime(2020, 1, 1) + pl.duration(days=n_times - 1),
        interval="1d",
        eager=True,
    )
    df = pl.DataFrame(
        {
            "unique_id": np.repeat([f"id_{i}" for i in range(n_series)], n_times),
            "ds": np.tile(dates.to_numpy(), n_series),
            "y": rng.normal(10, 2, n_series * n_times),
            "store": np.repeat([i % n_groups for i in range(n_series)], n_times),
        }
    ).sort("unique_id", "ds")
    return df if backend == "polars" else df.to_pandas()


def test_migrated_model_predicts_identically(tmp_path):
    """Save under the numpy engine, migrate, load and predict under narwhals.

    Both engine switches go through ``pooled_engine``, which restores every
    reloaded module object on exit. The earlier form set the env var with
    ``monkeypatch.setenv`` and reloaded, but never reloaded BACK -- the env
    var was restored while the modules stayed on the narwhals engine for the
    remainder of the pytest session. That is what turned the full-suite run
    red at ``ce7dbab``
    (``test_pooled_narwhals.py::test_history_warmup_partition_by_densifies_before_first_predict``
    and ``test_pooled_state_cleanup.py::test_g1_pooled_predictions_byte_identical``
    both failed in the suite and passed in isolation).
    """
    df = _panel()
    tfms = {1: [RollingMean(7, groupby=["store"]), ExpandingMean(groupby=["store"])]}

    with pooled_engine("numpy"):
        old = MLForecast(
            models=[LinearRegression()], freq="1d", lags=[1], lag_transforms=tfms
        )
        old.fit(df, static_features=["store"])
        expected = old.predict(7)
        old.save(str(tmp_path / "old"))

        from mlforecast._pooled_migrate import migrate_saved_model

        migrate_saved_model(tmp_path / "old", tmp_path / "new")

    with pooled_engine("narwhals"):
        from mlforecast import MLForecast as MF

        migrated = MF.load(str(tmp_path / "new"))
        got = migrated.predict(7)

    np.testing.assert_allclose(
        expected.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        got.sort("unique_id", "ds")["LinearRegression"].to_numpy(),
        atol=1e-9,
    )


def test_legacy_load_raises_actionable_error(tmp_path):
    df = _panel()
    with pooled_engine("numpy"):
        f = MLForecast(
            models=[LinearRegression()],
            freq="1d",
            lags=[1],
            lag_transforms={1: [RollingMean(7, groupby=["store"])]},
        )
        f.fit(df, static_features=["store"])
        f.save(str(tmp_path / "m"))

        from mlforecast._pooled_migrate import (
            LegacyPickleError,
            _simulate_missing_legacy,
        )

        with _simulate_missing_legacy():
            with pytest.raises(LegacyPickleError, match="migrate_saved_model"):
                MLForecast.load(str(tmp_path / "m"))


# ---------------------------------------------------------------------------
# Extra coverage beyond the brief's two required tests.
# ---------------------------------------------------------------------------

BACKENDS = ["polars", "pandas"]
# Exercises global, groupby, groupby+partition_by ("nonlocal") and pure
# partition_by ("local") -- every pooled mode Task 11's ``_migrate_one_state``
# branches on. Kept in sync with ``_pooled_migration_worker.py``'s own
# ``_mode_cases``.
MODES = ["global", "groupby", "groupby_partition_by", "local_partition_by"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", MODES)
def test_migration_predicts_identically_across_backends_and_modes(
    tmp_path, backend, mode
):
    """Revert-proof driver for the hard requirement: 'migrate and verify a
    model saved from BOTH a polars-backed and a pandas-backed fit', across
    every pooled mode Task 11's ``_migrate_one_state`` branches on.

    Runs in a subprocess, one per (backend, mode): chaining multiple
    ``importlib.reload`` + ``cloudpickle`` save/load round trips within a
    single process is independently fragile (cloudpickle falls back to
    pickling ``TimeSeries`` by value once any reload has happened, and its
    dynamic-class tracking can then resurrect an EARLIER reload generation's
    class for a LATER cycle) -- see ``_pooled_migration_worker.py``'s
    docstring. A fresh interpreter per combination sidesteps that harness
    artifact entirely rather than risk it masking a real migration bug.
    """
    worker = Path(__file__).parent / "_pooled_migration_worker.py"
    result = subprocess.run(
        [sys.executable, str(worker), backend, mode, str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"[{backend}/{mode}] worker failed:\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_rebuild_agg_from_legacy_empty_ts_aggs_fallback():
    """The slow-path fallback (``_ts_aggs`` cleared) must reconstruct the
    SAME aggregate table as the fast path that reads it directly -- this is
    the regression test for the "Empty _ts_aggs" case flagged in the brief.
    """
    import copy

    with pooled_engine("numpy"):
        _rebuild_agg_from_legacy_empty_ts_aggs_body(copy)


def _rebuild_agg_from_legacy_empty_ts_aggs_body(copy):
    from mlforecast.pooled import PooledState

    df = _panel()
    f = MLForecast(
        models=[LinearRegression()],
        freq="1d",
        lags=[1],
        lag_transforms={1: [RollingMean(7, groupby=["store"])]},
    )
    f.fit(df, static_features=["store"])
    key = next(iter(f.ts._pooled_states))
    legacy_state = f.ts._pooled_states[key]
    assert isinstance(legacy_state, PooledState)
    assert legacy_state._ts_aggs, "fixture must actually reach the fast-path cache"

    from mlforecast._pooled_migrate import _rebuild_agg_from_legacy

    fast = _rebuild_agg_from_legacy(legacy_state, f.ts.time_col)

    cleared_state = copy.copy(legacy_state)
    cleared_state._ts_aggs = {}
    assert not cleared_state._ts_aggs, "precondition: fallback path must actually run"
    slow = _rebuild_agg_from_legacy(cleared_state, f.ts.time_col)

    import narwhals as nw

    fast_nw = nw.from_native(fast, eager_only=True).sort(["_bucket_id", f.ts.time_col])
    slow_nw = nw.from_native(slow, eager_only=True).sort(["_bucket_id", f.ts.time_col])
    assert fast_nw.columns == slow_nw.columns
    assert len(fast_nw) == len(slow_nw)
    for col in ("s", "c", "q", "mn", "mx", "ord", "Es", "Ec", "Eq", "ewm"):
        a = fast_nw.get_column(col).to_numpy()
        b = slow_nw.get_column(col).to_numpy()
        np.testing.assert_allclose(a, b, atol=0, equal_nan=True)
