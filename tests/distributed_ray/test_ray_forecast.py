import sys
import warnings

import pandas as pd
import pytest
import ray

from mlforecast.distributed import DistributedMLForecast
from mlforecast.distributed.models.ray.lgb import RayLGBMForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.utils import generate_daily_series

warnings.simplefilter("ignore", FutureWarning)


@pytest.mark.ray
@pytest.mark.skipif(sys.platform == "win32", reason="Distributed tests are not supported on Windows")
@pytest.mark.skipif(sys.version_info <= (3, 9), reason="Distributed tests are not supported on Python < 3.10")
def test_ray_distributed_forecast():
    series = generate_daily_series(
        100, equal_ends=True, min_length=500, max_length=1_000
    )

    # Create Ray dataset from pandas
    ray_dataset = ray.data.from_pandas(series)
    ray_dataset = ray_dataset.map_batches(
        lambda batch: batch.assign(unique_id=batch['unique_id'].astype(str)),
        batch_format="pandas"
    )

    # test existing features provide the same result
    fcst = DistributedMLForecast(
        models=[RayLGBMForecast(verbosity=-1, random_state=0)],
        freq="D",
        lags=[1, 2, 3, 4, 5, 6, 7],
        lag_transforms={
            1: [RollingMean(7), RollingMean(30), ExpandingMean()],
            335: [RollingMean(30)],
        },
        date_features=["dayofweek"],
    )
    # df with features
    training_df_featured = fcst.preprocess(
        ray_dataset, static_features=[], dropna=False
    )
    fcst.fit(training_df_featured, static_features=[], dropna=False)
    preds1 = fcst.predict(10).materialize()

    # df without features
    fcst.preprocess(ray_dataset, static_features=[], dropna=False)
    preds2 = fcst.predict(10).materialize()
    pd.testing.assert_frame_equal(preds1, preds2)
