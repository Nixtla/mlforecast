"""Guards for the pooled-state cleanup + cheaper-_backup PR.

G1 (no-trim byte-identical): a deterministic multi-mode, multi-model,
multi-horizon forecast must reproduce golden predictions captured on the
pre-cleanup baseline. Because two models are fitted, ``TimeSeries._backup``
runs between them, so this also guards that the cheaper snapshot/restore
backup is behavior-identical to the original deepcopy.

The fit spans all five pooled modes (global / groupby / local-partition /
global+partition / groupby+partition) and every aggregate field
(mean/std/min/max/expanding/EWM), so a regression in any pooled append or
aggregate path moves a prediction and fails here.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlforecast.forecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMean,
    ExponentiallyWeightedMean,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingStd,
)

# Predictions captured on baseline c64a1d5 (pre-cleanup). 12 rows x 2 models,
# row-major (ravel) after sorting by (unique_id, ds). Must stay identical
# through Part A (cleanup) and Part B (cheaper _backup).
_GOLDEN = np.array(
    [
        42.93514960996524,
        46.64935233901883,
        35.51788717586902,
        51.493699673639185,
        48.510361576704526,
        65.11025040404411,
        44.064850390032774,
        47.77905311908637,
        34.194088764839364,
        50.16990126260953,
        48.21717186661735,
        64.81706069395693,
        50.69786936779505,
        54.41207209684865,
        42.872199398702776,
        58.848011896472826,
        55.0232447832594,
        71.62313361059955,
        50.30213063220296,
        54.01633336125667,
        40.83977654200561,
        56.81558903977566,
        55.70428866006259,
        72.30417748740263,
    ]
)


def _make_panel():
    ids, ds, y, brand, promo = [], [], [], [], []
    series = {"a": ("x", 1.0), "b": ("x", 3.0), "c": ("y", 7.0), "d": ("y", 11.0)}
    for sid, (br, base) in series.items():
        for t in range(1, 17):
            ids.append(sid)
            ds.append(t)
            y.append(base + 2.0 * t + 5.0 * ((t * (1 if sid in ("a", "c") else 2)) % 4))
            brand.append(br)
            promo.append(t % 2)
    return pd.DataFrame(
        {"unique_id": ids, "ds": ds, "y": y, "brand": brand, "promo": promo}
    )


def _build_fcst():
    return MLForecast(
        models=[LinearRegression(), LinearRegression(fit_intercept=False)],
        freq=1,
        lags=[1],
        lag_transforms={
            1: [
                RollingMean(2, global_=True),
                RollingMean(2, groupby=["brand"]),
                RollingMean(2, min_samples=1, partition_by=["promo"]),
                RollingMean(2, min_samples=1, global_=True, partition_by=["promo"]),
                RollingMean(
                    2, min_samples=1, groupby=["brand"], partition_by=["promo"]
                ),
                RollingStd(3, min_samples=2, global_=True),
                RollingMin(3, global_=True),
                RollingMax(3, global_=True),
                ExpandingMean(global_=True),
                ExponentiallyWeightedMean(alpha=0.5, global_=True),
            ],
        },
    )


def test_g1_pooled_predictions_byte_identical():
    df = _make_panel()
    fcst = _build_fcst()
    fcst.fit(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=["brand"],
    )
    h = 3
    future = []
    for sid in ["a", "b", "c", "d"]:
        for t in range(17, 17 + h):
            future.append({"unique_id": sid, "ds": t, "promo": t % 2})
    X_df = pd.DataFrame(future)
    preds = fcst.predict(h=h, X_df=X_df)
    preds = preds.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    model_cols = [c for c in preds.columns if c not in ("unique_id", "ds")]
    got = preds[model_cols].to_numpy().ravel()
    # exact: the cleanup must not change any prediction (rtol covers only
    # float-repr round-trip of the embedded golden literals).
    np.testing.assert_allclose(got, _GOLDEN, rtol=1e-12, atol=0)
