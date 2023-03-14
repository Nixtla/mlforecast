# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../../nbs/distributed.models.dask.lgb.ipynb.

# %% auto 0
__all__ = ['LGBMForecast']

# %% ../../../../nbs/distributed.models.dask.lgb.ipynb 3
import warnings

import lightgbm as lgb

# %% ../../../../nbs/distributed.models.dask.lgb.ipynb 4
class LGBMForecast(lgb.dask.DaskLGBMRegressor):
    if lgb.__version__ < "3.3.0":
        warnings.warn(
            "It is recommended to install LightGBM version >= 3.3.0, since "
            "the current LightGBM version might be affected by https://github.com/microsoft/LightGBM/issues/4026, "
            "which was fixed in 3.3.0"
        )

    @property
    def model_(self):
        return self.to_local()
