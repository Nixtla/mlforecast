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
from mlforecast.lag_transforms import (
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingStd,
    ExponentiallyWeightedMean,
    RollingMean,
)

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
    # ExpandingMin/ExpandingMax/EWM are here deliberately: they are the only
    # families whose predict seed reads an `A<col>` accumulate column, and a
    # migrated state reaches `_make_seeds` without ever passing through
    # `feature_frame`. With `RollingMean`/`ExpandingMean` alone (the shape
    # this test had) it could not notice `_migrate_one_state` failing to
    # settle them.
    tfms = {
        1: [
            RollingMean(7, groupby=["store"]),
            ExpandingMean(groupby=["store"]),
            ExpandingMin(groupby=["store"]),
            ExpandingMax(groupby=["store"]),
            ExponentiallyWeightedMean(alpha=0.3, groupby=["store"]),
        ]
    }

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
MODES = [
    "global",
    "groupby",
    "groupby_partition_by",
    "local_partition_by",
    # The accumulate families (ExpandingMin/ExpandingMax/EWM) -- the only ones
    # whose predict seed reads an `A<col>` accumulate column. Added in the
    # final fix wave: `_migrate_one_state` never settled
    # `ensure_time_aggs`/`ensure_accumulates`, so migrating such a model and
    # predicting raised `RuntimeError: _make_seeds: ... missing accumulate
    # column(s)` (and, before the seed guard existed, returned a wrong number
    # silently). Invisible to the four modes above, which use only
    # `RollingMean`/`ExpandingMean` -- the same two blind control families
    # that hid the original Critical.
    "global_accumulate",
    "groupby_accumulate",
]


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

    fast, fast_kref = _rebuild_agg_from_legacy(legacy_state, f.ts.time_col)

    cleared_state = copy.copy(legacy_state)
    cleared_state._ts_aggs = {}
    assert not cleared_state._ts_aggs, "precondition: fallback path must actually run"
    slow, slow_kref = _rebuild_agg_from_legacy(cleared_state, f.ts.time_col)

    import narwhals as nw

    fast_nw = nw.from_native(fast, eager_only=True).sort(["_bucket_id", f.ts.time_col])
    slow_nw = nw.from_native(slow, eager_only=True).sort(["_bucket_id", f.ts.time_col])
    assert fast_nw.columns == slow_nw.columns
    assert len(fast_nw) == len(slow_nw)
    for col in (
        "s",
        "c",
        "sK",
        "qK",
        "mn",
        "mx",
        "ord",
        "Es",
        "Ec",
        "EsK",
        "EqK",
        "ewm",
    ):
        a = fast_nw.get_column(col).to_numpy()
        b = slow_nw.get_column(col).to_numpy()
        np.testing.assert_allclose(a, b, atol=0, equal_nan=True)

    # The frozen centring reference travels with the table (it is what makes
    # every `qK` above summable), and both paths must recover the same one:
    # it is recomputed from the raw `y` either way, never from `_ts_aggs`'
    # zero-centred `sum_sq`, which cannot be re-centred after the fact.
    fast_k = nw.from_native(fast_kref, eager_only=True).sort("_bucket_id")
    slow_k = nw.from_native(slow_kref, eager_only=True).sort("_bucket_id")
    assert fast_k.columns == slow_k.columns
    for col in fast_k.columns:
        np.testing.assert_allclose(
            fast_k.get_column(col).to_numpy().astype(float),
            slow_k.get_column(col).to_numpy().astype(float),
            atol=0,
        )
    assert (fast_k.get_column("K").to_numpy() != 0).all(), (
        "the migrated reference must be the bucket's real mean, not a zero placeholder"
    )


def _large_magnitude_panel(n_series=6, n_times=30, mag=1e11, seed=0):
    """Near-constant values around ``mag``: the regime where the variance is
    lost entirely unless the moments are centred on a per-bucket reference.

    Same construction as
    ``tests/test_pooled_differential.py::_large_magnitude_panel``; duplicated
    rather than imported so this file keeps its own fixture surface.
    """
    rng = np.random.default_rng(seed)
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1),
        pl.datetime(2020, 1, 1) + pl.duration(days=n_times - 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame(
        {
            "unique_id": np.repeat([f"id_{s}" for s in range(n_series)], n_times),
            "ds": np.tile(dates.to_numpy(), n_series),
            "y": mag * (1.0 + rng.normal(0, 1e-15, n_series * n_times)),
            "store": np.zeros(n_series * n_times, dtype=np.int64),
        }
    ).sort("unique_id", "ds")


def test_migrated_state_carries_the_centring_reference(tmp_path):
    """A migrated model must own a frozen centring reference, like a fresh fit.

    Legacy stores ``sum_sq`` -- the zero-centred second moment -- which cannot
    be re-centred after the fact, so ``_rebuild_agg_from_legacy`` recomputes
    ``sK``/``qK`` from the raw ``y`` the legacy state also carries, and must
    hand the reference it used back to the state. Without that, the state has
    no reference at all: ``update()`` then derives a fresh one from the new
    rows while every migrated ``qK`` stays centred on the old one, and the
    expanding variance becomes a sum of terms taken about two different
    centres.

    ``ExpandingStd`` specifically: its window reaches ordinal 0 through the
    cumulative columns, so it is the family that actually reads the migrated
    moments end to end.
    """
    from decimal import Decimal, localcontext
    from fractions import Fraction

    df = _large_magnitude_panel(n_times=30)
    split = sorted(set(df["ds"].to_list()))[-4]
    head, tail = df.filter(pl.col("ds") <= split), df.filter(pl.col("ds") > split)
    assert len(tail) > 0
    tfms = {1: [ExpandingStd(groupby=["store"])]}

    with pooled_engine("numpy"):
        old = MLForecast(
            models=[LinearRegression()], freq="1d", lags=[1], lag_transforms=tfms
        )
        old.fit(head, static_features=["store"])
        old.save(str(tmp_path / "old"))

        from mlforecast._pooled_migrate import migrate_saved_model

        migrate_saved_model(tmp_path / "old", tmp_path / "new")

    with pooled_engine("narwhals"):
        import narwhals as nw

        from mlforecast import MLForecast as MF

        migrated = MF.load(str(tmp_path / "new"))
        key = next(iter(migrated.ts._pooled_states))
        state = migrated.ts._pooled_states[key]
        assert state._kref is not None, "the migrated state carries no reference"
        k = nw.from_native(state._kref, eager_only=True).get_column("K").to_numpy()
        assert (np.abs(k) > 1e10).all(), (
            f"the migrated reference is not the bucket's own mean: {k}"
        )
        migrated.update(tail)
        ts = migrated.ts
        feats = state.latest_features(ts._get_pooled_tfms()[key], len(ts.uids))
        got = float(np.asarray(next(iter(feats.values())), dtype=float)[0])

    # exact expanding std over the whole (head + tail) history
    vals = [Fraction(v) for v in df["y"].to_list()]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    with localcontext() as ctx:
        ctx.prec = 60
        want = (Decimal(var.numerator) / Decimal(var.denominator)).sqrt()
        err = float(abs(Decimal(got) - want) / want)
    assert err < 5e-15, (
        f"expanding std after migrate + update is {err:g} off the exact value "
        f"{float(want)!r} (got {got!r})"
    )
