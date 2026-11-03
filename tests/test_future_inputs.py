import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast


@pytest.fixture(params=["pandas", "polars"])
def future_inputs(request):
    return request.param


def _to_native(df, engine):
    return pl.from_pandas(df) if engine == "polars" else df


def _to_pandas(df):
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def _fit_forecaster(engine, time_kind, strategy):
    times = np.tile(np.arange(10) * 2, 2) + np.repeat([0, 4], 10)
    if time_kind == "datetime":
        times = pd.Timestamp("2020-01-01") + pd.to_timedelta(times, unit="D")
    df = pd.DataFrame(
        {
            "series": np.repeat(["a", "b"], 10),
            "timestamp": times,
            "x": np.random.default_rng(0).random(20),
        }
    )
    df["target"] = 3 * df["x"]
    fit_kwargs = {}
    if strategy == "direct":
        fit_kwargs["max_horizon"] = 3
    elif strategy == "sparse":
        fit_kwargs["horizons"] = [1, 3]
    freq = ("2d" if engine == "polars" else "2D") if time_kind == "datetime" else 2
    fcst = MLForecast(LinearRegression(), freq=freq, lags=[1])
    fcst.fit(
        _to_native(df, engine),
        id_col="series",
        time_col="timestamp",
        target_col="target",
        static_features=[],
        **fit_kwargs,
    )
    future = _to_pandas(fcst.make_future_dataframe(3))
    future["x"] = np.arange(1, len(future) + 1, dtype=float)
    return fcst, future


@pytest.mark.parametrize("time_kind", ["integer", "datetime"])
@pytest.mark.parametrize("strategy", ["recursive", "direct", "sparse"])
@pytest.mark.parametrize(
    "invalid", ["duplicate", "extra_duplicate", "missing", "off_grid", "extra_off_grid"]
)
def test_predict_rejects_invalid_future_keys(
    future_inputs, time_kind, strategy, invalid
):
    engine = future_inputs
    fcst, future = _fit_forecaster(engine, time_kind, strategy)
    expected = _to_pandas(fcst.predict(3, X_df=_to_native(future, engine)))
    bad_future = future.copy()
    if invalid == "duplicate":
        bad_future.iloc[1] = bad_future.iloc[0]
    elif invalid == "extra_duplicate":
        bad_future = pd.concat([bad_future, bad_future.iloc[[0]]], ignore_index=True)
    elif invalid == "missing":
        bad_future = bad_future.drop(index=1)
    else:
        offset = pd.Timedelta(days=1) if time_kind == "datetime" else 1
        if invalid == "extra_off_grid":
            extra = bad_future.iloc[[1]].copy()
            extra["timestamp"] -= offset
            bad_future = pd.concat([bad_future, extra], ignore_index=True)
        else:
            bad_future.loc[1, "timestamp"] -= offset
    message = "duplicate" if "duplicate" in invalid else "inputs in X_df"
    with pytest.raises(ValueError, match=message):
        fcst.predict(3, X_df=_to_native(bad_future, engine))
    pd.testing.assert_frame_equal(
        _to_pandas(fcst.predict(3, X_df=_to_native(future, engine))), expected
    )


@pytest.mark.parametrize("time_kind", ["integer", "datetime"])
@pytest.mark.parametrize("strategy", ["recursive", "direct", "sparse"])
@pytest.mark.parametrize("ids", [None, ["b"]])
def test_predict_future_keys_allow_shuffling_and_unused_rows(
    future_inputs, time_kind, strategy, ids
):
    engine = future_inputs
    fcst, future = _fit_forecaster(engine, time_kind, strategy)
    expected = _to_pandas(fcst.predict(3, X_df=_to_native(future, engine), ids=ids))
    extra = future.iloc[[0, 0]].copy()
    extra["series"] = "unused"
    past = future.iloc[[0, 0]].copy()
    past["timestamp"] -= pd.Timedelta(days=10) if time_kind == "datetime" else 10
    longer_future = pd.concat([future, extra, past], ignore_index=True)
    longer_future = longer_future.sample(frac=1, random_state=42)
    actual = fcst.predict(3, X_df=_to_native(longer_future, engine), ids=ids)
    pd.testing.assert_frame_equal(_to_pandas(actual), expected)


@pytest.mark.parametrize("time_kind", ["integer", "datetime"])
def test_predict_future_keys_validate_selected_ids_only(future_inputs, time_kind):
    engine = future_inputs
    fcst, future = _fit_forecaster(engine, time_kind, "recursive")
    expected = _to_pandas(fcst.predict(3, X_df=_to_native(future, engine), ids=["b"]))
    future.iloc[1] = future.iloc[0]
    actual = fcst.predict(3, X_df=_to_native(future, engine), ids=["b"])
    pd.testing.assert_frame_equal(_to_pandas(actual), expected)
