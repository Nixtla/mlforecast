import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast


def _frames(engine, train_categories, future_categories, feature="x"):
    train = pd.DataFrame(
        {
            "unique_id": np.repeat(["A", "B", "C"], 12),
            "ds": np.tile(np.arange(12), 3),
            "y": np.tile(1 + 2 * np.arange(12), 3),
            "constant": np.ones(36),
            feature: np.tile(np.arange(12), 3),
        }
    )
    future = (
        pd.DataFrame(
            {
                "unique_id": np.repeat(["A", "B", "C"], 3),
                "ds": np.tile([12, 13, 14], 3),
                feature: [20, 21, 22, 50, 51, 52, 80, 81, 82],
            }
        )
        .sample(frac=1, random_state=0)
        .reset_index(drop=True)
    )
    frames = []
    for frame, categories in [(train, train_categories), (future, future_categories)]:
        if engine == "pandas":
            if categories is not None:
                frame["unique_id"] = pd.Categorical(
                    frame["unique_id"], categories=categories
                )
        else:
            if categories is not None:
                # local categorical (no StringCache): physical order is first-seen order
                rank = [categories.index(uid) for uid in frame["unique_id"]]
                frame = frame.iloc[np.argsort(rank, kind="stable")]
            frame = pl.from_pandas(frame)
            if categories is not None:
                frame = frame.with_columns(pl.col("unique_id").cast(pl.Categorical))
        frames.append(frame)
    return frames


def _check_alignment(
    engine, mode, train_categories, future_categories, ids, feature="x"
):
    train, future = _frames(engine, train_categories, future_categories, feature)
    original = future.copy(deep=True) if engine == "pandas" else future.clone()
    fcst = MLForecast(models={"lr": LinearRegression()}, freq=1, lags=[])
    fit_kwargs = {"max_horizon": 3} if mode == "direct" else {}
    if mode == "sparse":
        fit_kwargs = {"horizons": [1, 3]}
    fcst.fit(train, static_features=["constant"], **fit_kwargs)
    predictions = fcst.predict(3, X_df=future, ids=ids)
    if engine == "polars":
        assert future.equals(original)
        predictions = predictions.to_pandas()
        future = future.to_pandas()
    else:
        pd.testing.assert_frame_equal(future, original)
    expected = {
        (str(uid), date): 1 + 2 * value
        for uid, date, value in future.itertuples(index=False, name=None)
    }
    np.testing.assert_allclose(
        predictions["lr"],
        [
            expected[(str(uid), date)]
            for uid, date in predictions[["unique_id", "ds"]].itertuples(
                index=False, name=None
            )
        ],
    )
    assert set(predictions["unique_id"].astype(str)) == set(ids or ["A", "B", "C"])
    assert len(predictions) == (2 if mode == "sparse" else 3) * (
        len(ids) if ids else 3
    )


@pytest.mark.parametrize("mode", ["recursive", "direct", "sparse"])
@pytest.mark.parametrize("ids", [None, ["A", "C"]])
@pytest.mark.parametrize(
    "engine, train_categories, future_categories",
    [
        ("pandas", None, None),
        ("pandas", ["C", "B", "A"], None),
        ("pandas", ["A", "B", "C"], ["C", "B", "A"]),
        ("pandas", ["C", "B", "A"], ["A", "B", "C"]),
        ("pandas", ["C", "B", "A"], ["C", "B", "A"]),
        ("pandas", ["unused", "C", "B", "A"], ["A", "B", "C", "unused"]),
        ("polars", None, None),
        ("polars", ["C", "B", "A"], ["A", "B", "C"]),
    ],
)
def test_predict_future_inputs_follow_fitted_id_order(
    engine, train_categories, future_categories, mode, ids
):
    _check_alignment(engine, mode, train_categories, future_categories, ids)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["recursive", "direct", "sparse"])
def test_predict_preserves_future_feature_named_like_order_column(engine, mode):
    _check_alignment(engine, mode, None, None, None, feature="_mlforecast_id_order")


def _to_pandas(df):
    df = df if isinstance(df, pd.DataFrame) else df.to_pandas()
    df = df.assign(unique_id=df["unique_id"].astype(str))
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_transfer_predict_future_inputs_follow_fitted_id_order(engine):
    train, future = _frames(engine, None, None)
    new_df, new_future = _frames(engine, ["C", "B", "A"], ["A", "B", "C"])
    fcst = MLForecast(models={"lr": LinearRegression()}, freq=1, lags=[])
    fcst.fit(train, static_features=["constant"])
    expected = fcst.transfer(train).predict(3, X_df=future)
    predictions = fcst.transfer(new_df).predict(3, X_df=new_future)
    pd.testing.assert_frame_equal(_to_pandas(predictions), _to_pandas(expected))
