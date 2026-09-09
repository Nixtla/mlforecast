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


def _probe_core_update_skipna(core_tfm) -> bool:
    probe = copy.deepcopy(core_tfm)
    # Test the requested capability even when the caller has not yet resolved
    # its deferred skipna setting. Never mutate the caller's transform/state.
    probe.skipna = True
    hist = np.arange(1.0, probe.lag + 6.0)

    def grouped(values):
        return _CoreGroupedArray(values, np.array([0, values.size], dtype=np.int32))

    try:
        probe.transform(grouped(hist))
        # update predicts the next row, whereas transform includes that row.
        # Advance far enough for the NaN and subsequent observations to reach
        # the accumulator even at lags larger than the initial sample series.
        for value in [np.nan, *range(probe.lag + 2)]:
            expected = copy.deepcopy(core_tfm)
            expected.skipna = True
            want = expected.transform(grouped(np.append(hist, value)))[-1:]
            out = probe.update(grouped(hist))
            if not np.isfinite(want).all() or not np.allclose(out, want):
                return False
            hist = np.append(hist, value)
    except Exception:  # pragma: no cover - defensive, treat as unsupported
        return False
    return True


def core_update_honors_skipna(core_tfm) -> bool:
    """Whether ``core_tfm.update`` excludes NaN when built with ``skipna=True``.

    Probes the actual transform rather than hard-coding a version check, so the
    guard disappears on its own once coreforecast fixes its Python-side
    accumulators. Only called when the user asked for ``skipna=True``.
    """
    name = type(core_tfm).__name__
    if name not in _SKIPNA_UPDATE_SUSPECTS:
        return True
    # The probe depends on the instance's lag and other parameters. It is
    # small enough to run directly, avoiding process-order-dependent caching.
    return _probe_core_update_skipna(core_tfm)


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
