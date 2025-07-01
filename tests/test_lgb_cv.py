import random

import pytest
from datasetsforecast.m4 import M4, M4Info
from fastcore.test import test_eq as _test_eq

from mlforecast.lag_transforms import SeasonalRollingMean
from mlforecast.lgb_cv import LightGBMCV
from mlforecast.target_transforms import Differences
from mlforecast.utils import generate_daily_series, generate_prices_for_series


@pytest.fixture(scope="module")
def m4_data():
    group = "Hourly"
    M4.async_download("data", group=group)
    df, *_ = M4.load(directory="data", group=group)
    df["ds"] = df["ds"].astype("int")
    ids = df["unique_id"].unique()
    random.seed(0)
    sample_ids = random.choices(ids, k=4)
    sample_df = df[df["unique_id"].isin(sample_ids)]
    info = M4Info[group]
    horizon = info.horizon
    valid = sample_df.groupby("unique_id").tail(horizon)
    train = sample_df.drop(valid.index)
    return train, valid, horizon


def evaluate_on_valid(preds, valid):
    preds = preds.copy()
    preds["final_prediction"] = preds.drop(columns=["unique_id", "ds"]).mean(1)
    merged = preds.merge(valid, on=["unique_id", "ds"])
    merged["abs_err"] = abs(merged["final_prediction"] - merged["y"]) / merged["y"]
    return merged.groupby("unique_id")["abs_err"].mean().mean()


def test_lightgbm_cv_pipeline(m4_data):
    train, valid, horizon = m4_data
    static_fit_config = dict(
        n_windows=2,
        h=horizon,
        params={"verbose": -1},
        compute_cv_preds=True,
    )
    cv = LightGBMCV(
        freq=1,
        lags=[24 * (i + 1) for i in range(7)],
    )
    hist = cv.fit(train, **static_fit_config)
    preds = cv.predict(horizon)
    eval1 = evaluate_on_valid(preds, valid)

    cv2 = LightGBMCV(
        freq=1,
        target_transforms=[Differences([24 * 7])],
        lags=[24 * (i + 1) for i in range(7)],
    )
    hist2 = cv2.fit(train, **static_fit_config)
    assert hist2[-1][1] < hist[-1][1]
    preds2 = cv2.predict(horizon)
    eval2 = evaluate_on_valid(preds2, valid)
    assert eval2 < eval1

    cv3 = LightGBMCV(
        freq=1,
        target_transforms=[Differences([24 * 7])],
        lags=[24 * (i + 1) for i in range(7)],
        lag_transforms={
            48: [SeasonalRollingMean(season_length=24, window_size=7)],
        },
    )
    hist3 = cv3.fit(train, **static_fit_config)
    assert hist3[-1][1] < hist2[-1][1]
    # preds3 = cv3.predict(horizon)
    # eval3 = evaluate_on_valid(preds3, valid)

    _test_eq(cv.find_best_iter([(0, 1), (1, 0.5)], 1), 1)
    _test_eq(cv.find_best_iter([(0, 1), (1, 0.5), (2, 0.6)], 1), 1)
    _test_eq(cv.find_best_iter([(0, 1), (1, 0.5), (2, 0.6), (3, 0.4)], 2), 3)

    cv4 = LightGBMCV(
        freq=1,
        lags=[24 * (i + 1) for i in range(7)],
    )
    cv4.setup(
        train,
        n_windows=2,
        h=horizon,
        params={"verbose": -1},
    )
    score = cv4.partial_fit(10)
    assert hist[0][1] == score
    score2 = cv4.partial_fit(20)
    assert hist[2][1] == score2


def test_lightgbmcv_callback():
    def before_predict_callback(df):
        assert not df["price"].isnull().any()
        return df

    dynamic_series = generate_daily_series(
        100, equal_ends=True, n_static_features=2, static_as_categorical=False
    )
    dynamic_series = dynamic_series.rename(columns={"static_1": "product_id"})
    prices_catalog = generate_prices_for_series(dynamic_series)
    series_with_prices = dynamic_series.merge(prices_catalog, how="left")
    cv = LightGBMCV(freq="D", lags=[24])
    _ = cv.fit(
        series_with_prices,
        n_windows=2,
        h=5,
        params={"verbosity": -1},
        static_features=["static_0", "product_id"],
        verbose_eval=False,
        before_predict_callback=before_predict_callback,
    )


def test_lightgbmcv_custom_metric(m4_data):
    train, _, horizon = m4_data

    def weighted_mape(y_true, y_pred, ids, dates):
        abs_pct_err = abs(y_true - y_pred) / abs(y_true)
        mape_by_serie = abs_pct_err.groupby(ids).mean()
        totals_per_serie = y_pred.groupby(ids).sum()
        series_weights = totals_per_serie / totals_per_serie.sum()
        return (mape_by_serie * series_weights).sum()

    _ = LightGBMCV(
        freq=1,
        lags=[24 * (i + 1) for i in range(7)],
    ).fit(
        train,
        n_windows=2,
        h=horizon,
        params={"verbose": -1},
        metric=weighted_mape,
    )
