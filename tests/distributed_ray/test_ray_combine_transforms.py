import operator
import sys

import pytest
import ray

from mlforecast.distributed import DistributedMLForecast
from mlforecast.distributed.models.ray.lgb import RayLGBMForecast
from mlforecast.lag_transforms import Combine, ExpandingMean, RollingMean
from mlforecast.utils import generate_daily_series


@pytest.mark.ray
@pytest.mark.skipif(sys.platform == "win32", reason="Distributed tests are not supported on Windows")
@pytest.mark.skipif(sys.version_info <= (3, 9), reason="Distributed tests are not supported on Python < 3.10")
def test_ray_combine_transforms():
    """Test Combine transform with Ray distributed forecast"""
    series = generate_daily_series(
        n_series=20,
        max_length=100,
        static_as_categorical=False,
        with_trend=True
    )

    ray_dataset = ray.data.from_pandas(series)
    ray_dataset = ray_dataset.map_batches(
        lambda batch: batch.assign(unique_id=batch['unique_id'].astype(str)),
        batch_format="pandas"
    )

    dmlf = DistributedMLForecast(
        models=[RayLGBMForecast()],
        freq="D",
        lag_transforms={
            1: [
                Combine(
                    RollingMean(window_size=7, min_samples=1),
                    ExpandingMean(),
                    operator.sub,
                ),
                Combine(
                    RollingMean(window_size=7, min_samples=1),
                    ExpandingMean(),
                    operator.add
                ),
                RollingMean(window_size=7, min_samples=1)
            ]
        },
    )

    # Test preprocessing creates expected features
    transformed_df = dmlf.preprocess(
        ray_dataset,
        id_col='unique_id',
        time_col='ds',
        target_col='y'
    )

    # Verify features were created
    df_pandas = transformed_df.to_pandas()
    assert 'rolling_mean_lag1_window_size7_min_samples1' in df_pandas.columns
    assert not df_pandas.empty

    # Test fitting works
    dmlf.fit(ray_dataset)

    # Test can convert to local
    local_model = dmlf.to_local()
    assert local_model is not None
