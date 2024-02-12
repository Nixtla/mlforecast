# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/compat.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/compat.ipynb 1
try:
    import coreforecast.lag_transforms as core_tfms
    import coreforecast.scalers as core_scalers
    from coreforecast.grouped_array import GroupedArray as CoreGroupedArray

    from mlforecast.lag_transforms import _BaseLagTransform, Lag

    CORE_INSTALLED = True
except ImportError:
    core_tfms = None
    core_scalers = None
    CoreGroupedArray = None

    class _BaseLagTransform:
        ...

    Lag = None

    CORE_INSTALLED = False
