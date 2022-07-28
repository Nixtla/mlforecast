# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/distributed.models.lgb.ipynb.

# %% auto 0
__all__ = ['LGBMForecast']

# %% ../nbs/distributed.models.lgb.ipynb 3
import warnings

import lightgbm as lgb

# %% ../nbs/distributed.models.lgb.ipynb 4
class LGBMForecast(lgb.dask.DaskLGBMRegressor):
    if lgb.__version__ <= "3.2.1":
        warnings.warn(
            "It is recommended to build LightGBM from source following the instructions here: "
            "https://github.com/microsoft/LightGBM/tree/master/python-package#install-from-github, since "
            "the current LightGBM version might be affected by https://github.com/microsoft/LightGBM/issues/4026, "
            "which was fixed after 3.2.1."
        )

    @property
    def model_(self):
        return self.booster_
