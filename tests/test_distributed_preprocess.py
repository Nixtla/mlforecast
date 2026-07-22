import pandas as pd

from mlforecast.distributed import DistributedMLForecast


def test_distributed_preprocess_weight_col_is_dynamic():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        [
            (uid, date, float(i), float(i % 2 + 1))
            for uid in ("a", "b")
            for i, date in enumerate(dates)
        ],
        columns=["unique_id", "ds", "y", "weight"],
    )
    forecast = DistributedMLForecast(models=[], freq="D", lags=[1])

    result = forecast.preprocess(df, weight_col="weight")

    assert "weight" in result.columns
