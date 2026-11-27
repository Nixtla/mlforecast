import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals, generate_daily_series


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_conformal_error_single_step_broadcasts_multi_series(backend):
    df = generate_daily_series(
        3, min_length=30, max_length=30, engine=backend, equal_ends=True
    )
    fcst = MLForecast(
        models=[LinearRegression()],
        freq="D" if backend == "pandas" else "1d",
        lags=[1],
    )
    pi = PredictionIntervals(n_windows=3, h=1, method="conformal_error")
    fcst.fit(df, prediction_intervals=pi)

    with pytest.warns(UserWarning, match="1-step ahead cross-validation"):
        preds = fcst.predict(h=4, level=[80, 95])

    assert len(preds) == 3 * 4
    pdf = preds.to_pandas() if isinstance(preds, pl.DataFrame) else preds
    scores = fcst._calibration_.scores
    scores_pd = scores.to_pandas() if isinstance(scores, pl.DataFrame) else scores

    for uid, grp in pdf.groupby("unique_id", observed=True):
        s_vals = scores_pd.loc[
            scores_pd["unique_id"] == uid, "LinearRegression"
        ].to_numpy()
        for lv in (80, 95):
            q = np.quantile(s_vals, lv / 100)
            np.testing.assert_allclose(
                grp[f"LinearRegression-hi-{lv}"].to_numpy()
                - grp["LinearRegression"].to_numpy(),
                np.full(4, q),
            )
            np.testing.assert_allclose(
                grp["LinearRegression"].to_numpy()
                - grp[f"LinearRegression-lo-{lv}"].to_numpy(),
                np.full(4, q),
            )
