"""In-process pooled-engine switching that leaves NO residue.

``mlforecast.pooled`` reads ``MLFORECAST_POOLED_ENGINE`` exactly once, into a
module constant, at import time. Switching engines inside a live process
therefore needs ``importlib.reload``. The naive form of that -- set the env
var, reload, run, restore the env var, reload again -- is what this module
exists to replace, because **it does not restore the process**:

* ``importlib.reload`` re-executes a module in its existing ``__dict__``, so
  every class it defines becomes a NEW class object. ``mlforecast.core``'s
  ``isinstance(state, NarwhalsPooledState)``, and anything else comparing
  class identity, is then comparing against a different object than the one
  that was there before.
* Reloading again in a ``finally`` does not undo that -- it makes a THIRD
  generation of class objects. The env var is back; the process is not.
* And a test file that simply forgot the ``finally`` (``test_pooled_migration``
  used ``monkeypatch.setenv``, which restores the variable but never reloads)
  leaves the modules stuck on the OTHER engine for the rest of the session.

That last case is not hypothetical: at ``ce7dbab`` the full suite was 2 failed
/ 1665 passed, and both failures
(``test_pooled_narwhals.py::test_history_warmup_partition_by_densifies_before_first_predict``
and ``test_pooled_state_cleanup.py::test_g1_pooled_predictions_byte_identical``)
passed in isolation. They were later tests running under the narwhals engine
because ``tests/test_pooled_migration.py`` had left it selected.

The fix here is to restore by SNAPSHOT rather than by re-execution: keep a copy
of each module's ``__dict__`` before reloading and put the original entries
back afterwards. Every class object, function and constant is then the
identical object it was before the ``with`` block, which
``tests/test_pooled_engine_isolation.py`` asserts directly.

Only ``mlforecast.pooled`` and ``mlforecast.core`` read the engine (verified:
``grep -rn POOLED_ENGINE mlforecast/`` names no other module), and they are
reloaded in dependency order -- ``pooled`` first, since ``core`` imports
``POOLED_ENGINE``/``PooledState``/``NarwhalsPooledState`` from it.
``mlforecast.forecast`` is reloaded as well, even though it never reads the
engine: see ``ENGINE_MODULES`` for the cloudpickle identity trap that made
leaving it stale corrupt the live ``TimeSeries`` class.

A subprocess is still the better tool where the run is coarse-grained enough
to afford one (``tests/test_pooled_acceptance_narwhals.py`` and
``tests/test_pooled_migration.py``'s cross-backend driver both do that). This
module is for the fine-grained differential comparisons, where a subprocess
per assertion would cost ~300 interpreter startups.
"""

import contextlib
import importlib
import os
import sys

#: Reloaded in this order: each imports from the one before it, so a later
#: module must be re-executed after the one it imports from.
#:
#: ``mlforecast.forecast`` does not itself read the engine -- ``grep -rn
#: POOLED_ENGINE mlforecast/`` names only ``pooled`` and ``core``. It is in
#: the list because it holds ``TimeSeries`` by name. Leaving it stale makes
#: ``MLForecast`` build instances of the PRE-reload ``TimeSeries`` class while
#: ``sys.modules["mlforecast.core"].TimeSeries`` is the post-reload one, and
#: cloudpickle takes that identity mismatch as "this class is not importable"
#: and pickles ``TimeSeries`` BY VALUE. Loading such a pickle then reuses and
#: MUTATES the live class through cloudpickle's dynamic-class tracker, giving
#: its methods a frozen copy of the module globals -- pollution that survives
#: any amount of module-dict restoration, because the damage is inside the
#: class object rather than in a module namespace. Observed exactly this way:
#: ``tests/test_pooled_migration.py::test_migrated_model_predicts_identically``
#: made ``test_pooled_state_cleanup.py::test_g1_pooled_predictions_byte_identical``
#: fail from the ``_MigrationUnpickler(...).load()`` call alone.
ENGINE_MODULES = ("mlforecast.pooled", "mlforecast.core", "mlforecast.forecast")

VALID_ENGINES = ("numpy", "narwhals")


def _engine_modules():
    """The live module objects, imported if this is the first touch."""
    return [importlib.import_module(name) for name in ENGINE_MODULES]


def module_state():
    """Identity fingerprint of everything a caller could observe as "changed".

    Used by ``tests/test_pooled_engine_isolation.py`` to prove a switch is a
    true round trip. Values are the objects themselves, compared with ``is``.
    """
    import mlforecast.core as core
    import mlforecast.forecast as forecast
    import mlforecast.pooled as pooled

    return {
        "forecast.MLForecast": forecast.MLForecast,
        "forecast.TimeSeries": forecast.TimeSeries,
        "sys.modules[mlforecast.forecast]": sys.modules["mlforecast.forecast"],
        "pooled.POOLED_ENGINE": pooled.POOLED_ENGINE,
        "pooled.PooledState": pooled.PooledState,
        "pooled.NarwhalsPooledState": pooled.NarwhalsPooledState,
        "core.POOLED_ENGINE": core.POOLED_ENGINE,
        "core.TimeSeries": core.TimeSeries,
        "core.PooledState": core.PooledState,
        "core.NarwhalsPooledState": core.NarwhalsPooledState,
        "sys.modules[mlforecast.pooled]": sys.modules["mlforecast.pooled"],
        "sys.modules[mlforecast.core]": sys.modules["mlforecast.core"],
    }


@contextlib.contextmanager
def pooled_engine(engine):
    """Run the block with ``engine`` selected, then restore the process exactly.

    On exit both the environment variable and the contents of every reloaded
    module's ``__dict__`` are restored to the objects they held on entry --
    not merely to equal values.
    """
    if engine not in VALID_ENGINES:
        raise ValueError(f"engine must be one of {VALID_ENGINES}; got {engine!r}")
    modules = _engine_modules()
    saved_env = os.environ.get("MLFORECAST_POOLED_ENGINE")
    # Shallow copies: reload REBINDS names in the same dict rather than
    # mutating the objects they point at, so putting the original mapping
    # back restores the original objects.
    saved_dicts = [(m, dict(m.__dict__)) for m in modules]
    os.environ["MLFORECAST_POOLED_ENGINE"] = engine
    try:
        for m in modules:
            importlib.reload(m)
        yield
    finally:
        if saved_env is None:
            os.environ.pop("MLFORECAST_POOLED_ENGINE", None)
        else:
            os.environ["MLFORECAST_POOLED_ENGINE"] = saved_env
        for m, saved in saved_dicts:
            m.__dict__.clear()
            m.__dict__.update(saved)
