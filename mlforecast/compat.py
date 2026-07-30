__all__ = []

import copy
import functools
import inspect

import numpy as np
import coreforecast.lag_transforms as _core_tfms
import coreforecast.scalers as _core_scalers
from coreforecast.grouped_array import GroupedArray as _CoreGroupedArray


@functools.lru_cache(maxsize=1)
def core_supports_skipna() -> bool:
    """Whether the installed coreforecast accepts ``skipna`` on its lag transforms.

    ``skipna`` was added after 0.0.16, so it can be missing on an otherwise
    supported coreforecast (we only require ``>=0.0.15``).
    """
    params = inspect.signature(_core_tfms.RollingMean.__init__).parameters
    return "skipna" in params


@functools.lru_cache(maxsize=1)
def core_scalers_support_skipna() -> bool:
    """Whether the installed coreforecast accepts ``skipna`` on its local scalers."""
    params = inspect.signature(_core_scalers.LocalStandardScaler.__init__).parameters
    return "skipna" in params


# Transforms whose coreforecast ``update`` is implemented in Python instead of
# delegating to ``_lib``, and which (as of coreforecast 0.0.18) ignore ``skipna``
# there: the accumulator is poisoned by a single NaN even with ``skipna=True``,
# so ``transform`` and ``update`` disagree. ``transform`` is correct for all of
# them. Membership here only marks a transform as *worth probing* -- the probe
# below is what decides -- so entries can stay after upstream fixes them.
#
# All of these take neither ``window_size`` nor ``min_samples``, which is what
# makes the short probe series below valid for every one of them.
_SKIPNA_UPDATE_SUSPECTS = frozenset(
    {
        "ExpandingMean",
        "ExpandingStd",
        "ExpandingMin",
        "ExpandingMax",
        "ExponentiallyWeightedMean",
    }
)


_skipna_update_probe_cache: dict = {}


def _probe_core_update_skipna(core_tfm) -> bool:
    probe = copy.deepcopy(core_tfm)
    hist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indptr = np.array([0, hist.size], dtype=np.int32)
    ga_hist = _CoreGroupedArray(hist, indptr)
    ga_nan = _CoreGroupedArray(
        np.append(hist, np.nan), np.array([0, hist.size + 1], dtype=np.int32)
    )
    try:
        probe.transform(ga_hist)
        probe.update(ga_hist)  # incorporate the last real observation
        out = probe.update(ga_nan)  # a NaN observation arrives
    except Exception:  # pragma: no cover - defensive, treat as unsupported
        return False
    return not bool(np.isnan(np.asarray(out)).any())


def core_update_honors_skipna(core_tfm) -> bool:
    """Whether ``core_tfm.update`` excludes NaN when built with ``skipna=True``.

    Probes the actual transform rather than hard-coding a version check, so the
    guard disappears on its own once coreforecast fixes its Python-side
    accumulators. Only called when the user asked for ``skipna=True``.
    """
    name = type(core_tfm).__name__
    if name not in _SKIPNA_UPDATE_SUSPECTS:
        return True
    # Which branch ``update`` takes is structural (per class), not
    # parameter-dependent, so caching on the class name is enough.
    if name not in _skipna_update_probe_cache:
        _skipna_update_probe_cache[name] = _probe_core_update_skipna(core_tfm)
    return _skipna_update_probe_cache[name]


try:
    from catboost import CatBoostRegressor
except ImportError:

    class CatBoostRegressor:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise ImportError("Please install catboost to use this model.")


try:
    from lightgbm import LGBMRegressor
except ImportError:

    class LGBMRegressor:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise ImportError("Please install lightgbm to use this model.")


try:
    from xgboost import XGBRegressor
except ImportError:

    class XGBRegressor:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise ImportError("Please install xgboost to use this model.")


try:
    from window_ops.shift import shift_array
except ImportError:
    import numpy as np
    from utilsforecast.compat import njit

    @njit
    def shift_array(x, offset):
        if offset >= x.size or offset < 0:
            return np.full_like(x, np.nan)
        if offset == 0:
            return x.copy()
        out = np.empty_like(x)
        out[:offset] = np.nan
        out[offset:] = x[:-offset]
        return out
