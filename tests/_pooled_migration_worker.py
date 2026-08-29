"""Worker invoked as a subprocess by test_pooled_migration.py's
across-backend/mode matrix.

Not a test module (no ``test_*`` name, not collected by pytest) -- a plain
script run via ``subprocess.run([sys.executable, __file__, backend, mode])``.

Why a subprocess at all: ``importlib.reload``-based engine switching plus a
full ``cloudpickle`` save/load round trip is fragile when MULTIPLE such
reload+save/load cycles happen within one process -- cloudpickle falls back to
pickling ``TimeSeries`` BY VALUE once ``mlforecast.core``/``mlforecast.pooled``
have been reloaded at all (verified: even a single prior reload is enough),
and its "already seen this dynamic class" tracking can then hand back a
class synthesized from an EARLIER reload generation, whose methods carry a
globals dict frozen to THAT generation's ``NarwhalsPooledState`` -- not the
one live at call time. A single reload cycle within one process (exactly what
the required ``test_migrated_model_predicts_identically`` does) is provably
fine; chaining several in one process to sweep a (backend x mode) matrix is
not. Running each combination in a fresh interpreter sidesteps the whole
class of bugs (and matches how a real user would actually use the migration:
one process running the numpy engine to save, a *different* process running
the narwhals engine to load) instead of masking it.
"""

import importlib
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from mlforecast.lag_transforms import ExpandingMean, RollingMean

N_SERIES = 20
N_TIMES = 60
N_GROUPS = 4


def _panel(backend, seed=0):
    rng = np.random.default_rng(seed)
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1),
        pl.datetime(2020, 1, 1) + pl.duration(days=N_TIMES - 1),
        interval="1d",
        eager=True,
    )
    df = pl.DataFrame(
        {
            "unique_id": np.repeat([f"id_{i}" for i in range(N_SERIES)], N_TIMES),
            "ds": np.tile(dates.to_numpy(), N_SERIES),
            "y": rng.normal(10, 2, N_SERIES * N_TIMES),
            # constant per series -> auto-detected as a static feature.
            "store": np.repeat([i % N_GROUPS for i in range(N_SERIES)], N_TIMES),
            # varies by date (same pattern for every series) -> a genuinely
            # DYNAMIC column, needed for the partition_by modes (a partition
            # column is never auto-filled from history the way a static
            # feature is -- it always needs future values via `X_df`).
            "promo": np.tile(np.arange(N_TIMES) % 3, N_SERIES),
        }
    ).sort("unique_id", "ds")
    return df if backend == "polars" else df.to_pandas()


def _future_x_df(backend, horizon=7):
    """Future ``promo`` values for the horizon, continuing ``_panel``'s
    ``arange(N_TIMES) % 3`` pattern -- needed whenever ``promo`` is used as
    ``partition_by``.
    """
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1) + pl.duration(days=N_TIMES),
        pl.datetime(2020, 1, 1) + pl.duration(days=N_TIMES + horizon - 1),
        interval="1d",
        eager=True,
    )
    promo = (np.arange(N_TIMES, N_TIMES + horizon) % 3).astype(np.int64)
    df = pl.DataFrame(
        {
            "unique_id": np.repeat([f"id_{i}" for i in range(N_SERIES)], horizon),
            "ds": np.tile(dates.to_numpy(), N_SERIES),
            "promo": np.tile(promo, N_SERIES),
        }
    )
    return df if backend == "polars" else df.to_pandas()


def _mode_cases():
    return {
        "global": (
            ["store"],
            lambda: {1: [RollingMean(7), ExpandingMean()]},
        ),
        "groupby": (
            ["store"],
            lambda: {
                1: [
                    RollingMean(7, groupby=["store"]),
                    ExpandingMean(groupby=["store"]),
                ]
            },
        ),
        "groupby_partition_by": (
            ["store"],
            lambda: {
                1: [
                    RollingMean(7, groupby=["store"], partition_by=["promo"]),
                    ExpandingMean(groupby=["store"], partition_by=["promo"]),
                ]
            },
        ),
        "local_partition_by": (
            ["store"],
            lambda: {
                1: [
                    RollingMean(7, partition_by=["promo"]),
                    ExpandingMean(partition_by=["promo"]),
                ]
            },
        ),
    }


def main(backend, mode, out_dir):
    import os

    statics, tfms_factory = _mode_cases()[mode]
    needs_x_df = "partition_by" in mode
    # A partition_by state's recursive (h > 1) predict path re-derives bucket
    # assignments from `X_df` every step (`_update_partition_assignments`),
    # which -- independently of this migration -- was found (via a direct
    # numpy-vs-narwhals fresh-fit comparison, no save/load involved) to
    # already disagree between the two engines for a DYNAMIC partition_by
    # column at h > 1, while h == 1 (no recursion, straight off the fit-time
    # state) matches exactly. That divergence predates Task 11 and is out of
    # its scope (migrating already-fitted state, not the predict recursion);
    # using h=1 here still fully exercises the migrated state's densification/
    # `ensure_*`/tail machinery without tripping over it.
    horizon = 1 if needs_x_df else 7
    df = _panel(backend)
    if not needs_x_df:
        # `promo` is a dynamic (non-static) column: leaving it in for a mode
        # that never references it would still force it into
        # `features_order_`, requiring `X_df` for a column irrelevant to
        # this test.
        df = df.drop(columns=["promo"]) if backend == "pandas" else df.drop("promo")

    os.environ["MLFORECAST_POOLED_ENGINE"] = "numpy"
    import mlforecast.core
    import mlforecast.pooled

    importlib.reload(mlforecast.pooled)
    importlib.reload(mlforecast.core)
    from mlforecast import MLForecast

    if needs_x_df:
        x_df_full = _future_x_df(backend, horizon=7)
        min_ds = x_df_full["ds"].min()
        if backend == "polars":
            x_df = x_df_full.filter(pl.col("ds") == min_ds)
        else:
            x_df = x_df_full.loc[x_df_full["ds"] == min_ds].reset_index(drop=True)
    else:
        x_df = None

    old = MLForecast(
        models=[LinearRegression()], freq="1d", lags=[1], lag_transforms=tfms_factory()
    )
    old.fit(df, static_features=statics)
    expected = old.predict(horizon, X_df=x_df) if needs_x_df else old.predict(horizon)
    old.save(f"{out_dir}/old")

    from mlforecast._pooled_migrate import migrate_saved_model

    migrate_saved_model(f"{out_dir}/old", f"{out_dir}/new")

    os.environ["MLFORECAST_POOLED_ENGINE"] = "narwhals"
    importlib.reload(mlforecast.pooled)
    importlib.reload(mlforecast.core)
    from mlforecast import MLForecast as MF

    migrated = MF.load(f"{out_dir}/new")
    got = (
        migrated.predict(horizon, X_df=x_df)
        if needs_x_df
        else migrated.predict(horizon)
    )

    expected_pd = expected.to_pandas() if hasattr(expected, "to_pandas") else expected
    got_pd = got.to_pandas() if hasattr(got, "to_pandas") else got
    expected_pd = expected_pd.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    got_pd = got_pd.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    np.testing.assert_allclose(
        expected_pd["LinearRegression"].to_numpy(),
        got_pd["LinearRegression"].to_numpy(),
        atol=1e-9,
    )
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
