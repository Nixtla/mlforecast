"""The engine-switching harness must leave the process exactly as it found it.

At ``ce7dbab`` the full suite (``uv run pytest --ignore=tests/distributed_ray
-q -p no:randomly``) was **2 failed, 1665 passed**::

    FAILED tests/test_pooled_narwhals.py::test_history_warmup_partition_by_densifies_before_first_predict
    FAILED tests/test_pooled_state_cleanup.py::test_g1_pooled_predictions_byte_identical

Both passed in isolation. Neither is a regression in the engine: they were
collateral damage from ``tests/test_pooled_migration.py``, which set
``MLFORECAST_POOLED_ENGINE`` with ``monkeypatch.setenv`` and reloaded
``mlforecast.pooled``/``mlforecast.core``, but never reloaded back.

Two distinct defects were behind it, and each gets an assertion here:

1. **Engine left switched.** The env var was restored, the module constants
   were not, so every later test in the session ran on the other engine.
2. **The live ``TimeSeries`` class silently rewritten.** Reloading
   ``mlforecast.core`` while leaving ``mlforecast.forecast`` stale makes
   ``MLForecast`` construct instances of the PRE-reload ``TimeSeries`` while
   ``sys.modules["mlforecast.core"].TimeSeries`` is the POST-reload one.
   cloudpickle reads that identity mismatch as "not importable" and pickles
   ``TimeSeries`` by value; loading such a pickle reuses the tracked class and
   applies ``_class_setstate`` to it, rebinding EVERY method on the original,
   still-referenced class object to a copy carrying frozen module globals.
   Measured: 19 methods of the live ``TimeSeries`` (``predict``, ``update``,
   ``save``, ``_predict_recursive``, ...) were replaced by one
   ``_MigrationUnpickler(...).load()`` call. No module-dict restoration can
   undo that -- the damage is inside the class object.

The last test here is the end-to-end check, in the shape that originally
caught the problem: run the polluting file sequence in one interpreter, under
the DEFAULT engine, and require it green.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from ._pooled_engine_env import ENGINE_MODULES, module_state, pooled_engine

ROOT = Path(__file__).resolve().parents[1]


def _panel(n_series=8, n_times=30):
    rng = np.random.default_rng(0)
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1),
        pl.datetime(2020, 1, 1) + pl.duration(days=n_times - 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame(
        {
            "unique_id": np.repeat([f"id_{i}" for i in range(n_series)], n_times),
            "ds": np.tile(dates.to_numpy(), n_series),
            "y": rng.normal(10, 2, n_series * n_times),
            "store": np.repeat([i % 3 for i in range(n_series)], n_times),
        }
    ).sort("unique_id", "ds")


@pytest.mark.parametrize("engine", ["numpy", "narwhals"])
def test_pooled_engine_switch_is_an_exact_round_trip(engine):
    """Every observable module object is the SAME object after the switch.

    ``is``, not ``==``: reloading restores equal-looking values while handing
    out fresh class objects, which is precisely what breaks
    ``isinstance``-based dispatch elsewhere.
    """
    before = module_state()
    with pooled_engine(engine):
        pass
    after = module_state()
    changed = [k for k in before if before[k] is not after[k]]
    assert not changed, (
        f"pooled_engine({engine!r}) did not restore: {changed}. Restoring the "
        "environment variable is not enough -- every reloaded module's "
        "__dict__ must be put back as it was."
    )


def test_pooled_engine_switch_actually_switches():
    """Guards the round-trip test above from passing vacuously.

    If ``pooled_engine`` silently stopped reloading, the round-trip assertion
    would pass trivially. This pins that the switch is real: the engine
    constant changes inside the block, and the reload does produce a new
    class object while it is in effect.
    """
    import mlforecast.pooled as pooled

    outside_engine = pooled.POOLED_ENGINE
    other = "narwhals" if outside_engine == "numpy" else "numpy"
    outside_cls = pooled.NarwhalsPooledState
    with pooled_engine(other):
        assert pooled.POOLED_ENGINE == other
        import mlforecast.core as core

        assert core.POOLED_ENGINE == other
        assert pooled.NarwhalsPooledState is not outside_cls, (
            "the module was not re-executed, so this harness would not be "
            "isolating anything"
        )
    assert pooled.POOLED_ENGINE == outside_engine
    assert pooled.NarwhalsPooledState is outside_cls


def test_engine_modules_cover_every_pooled_engine_reader():
    """``ENGINE_MODULES`` must name every module that reads the engine.

    A module that reads ``POOLED_ENGINE`` but is missing from the reload set
    keeps the old engine for the duration of the switch -- a silently
    half-applied switch. Derived from the source rather than restated, so
    adding a reader to a new module fails here instead of going unnoticed.
    """
    readers = set()
    for path in (ROOT / "mlforecast").rglob("*.py"):
        text = path.read_text()
        # the constant IMPORT/definition, not a mention in a docstring
        if "POOLED_ENGINE," in text or "POOLED_ENGINE =" in text:
            readers.add(f"mlforecast.{path.stem}")
    missing = readers - set(ENGINE_MODULES)
    assert not missing, (
        f"these modules read POOLED_ENGINE but are not reloaded: {sorted(missing)}"
    )


def test_cloudpickle_round_trip_under_a_switch_does_not_rewrite_timeseries(tmp_path):
    """The cloudpickle identity trap, asserted directly.

    ``MLForecast.save`` cloudpickles the ``TimeSeries``. If the reload set
    omits ``mlforecast.forecast``, the saved instance's class is not the one
    bound in ``sys.modules``, cloudpickle serializes the class BY VALUE, and
    loading it rebinds every method on the live class object.

    Reverting the fix (dropping ``"mlforecast.forecast"`` from
    ``ENGINE_MODULES``) makes this fail with 19 rewritten methods.
    """
    import mlforecast.core as core

    ts_cls = core.TimeSeries
    before = {k: v for k, v in vars(ts_cls).items() if not k.startswith("__")}

    df = _panel()
    with pooled_engine("numpy"):
        fcst = MLForecast(
            models=[LinearRegression()],
            freq="1d",
            lags=[1],
            lag_transforms={1: [RollingMean(7, groupby=["store"])]},
        )
        fcst.fit(df, static_features=["store"])
        fcst.save(str(tmp_path / "m"))

        from mlforecast._pooled_migrate import _MigrationUnpickler

        with open(tmp_path / "m" / "ts.pkl", "rb") as fh:
            _MigrationUnpickler(fh).load()

    assert core.TimeSeries is ts_cls
    rewritten = sorted(k for k, v in before.items() if vars(ts_cls).get(k) is not v)
    assert not rewritten, (
        f"{len(rewritten)} TimeSeries attributes were rebound by a cloudpickle "
        f"round trip inside an engine switch: {rewritten}. The unpickler reused "
        "and mutated the live class because it had been pickled by value."
    )


# The alphabetical file order that reproduced the full-suite failure in ~10s.
# `test_pooled_migration.py` is the polluter; the other two hold the tests that
# went red because of it.
POLLUTION_SEQUENCE = [
    "tests/test_pooled_migration.py",
    "tests/test_pooled_narwhals.py",
    "tests/test_pooled_state_cleanup.py",
]


def test_polluting_file_sequence_is_green_in_one_interpreter():
    """End-to-end, in the shape that caught this: one process, several files.

    A per-file run does NOT reproduce it -- that is exactly why fourteen
    task-level gates missed it. Uses ``sys.executable -m pytest`` (already the
    resolved venv interpreter under ``uv run``) with the engine variable
    cleared, so the child runs on the shipped default.
    """
    env = dict(os.environ)
    env.pop("MLFORECAST_POOLED_ENGINE", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *POLLUTION_SEQUENCE,
            "-q",
            "--no-cov",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        "the pooled file sequence is not green in a single interpreter -- "
        "cross-test pollution from the engine-switching harness.\n"
        f"{proc.stdout[-6000:]}\n{proc.stderr[-2000:]}"
    )
    assert " failed" not in proc.stdout, proc.stdout[-4000:]
