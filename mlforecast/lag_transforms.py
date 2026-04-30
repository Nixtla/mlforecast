__all__ = [
    "RollingMean",
    "RollingStd",
    "RollingMin",
    "RollingMax",
    "RollingQuantile",
    "SeasonalRollingMean",
    "SeasonalRollingStd",
    "SeasonalRollingMin",
    "SeasonalRollingMax",
    "SeasonalRollingQuantile",
    "ExpandingMean",
    "ExpandingStd",
    "ExpandingMin",
    "ExpandingMax",
    "ExpandingQuantile",
    "ExponentiallyWeightedMean",
    "Offset",
    "Combine",
]


import copy
import inspect
import re
from typing import Callable, Optional, Sequence

import coreforecast.lag_transforms as core_tfms
import numpy as np
from coreforecast.grouped_array import GroupedArray as CoreGroupedArray
from sklearn.base import BaseEstimator


def _pascal2camel(pascal_str: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", pascal_str).lower()

def _normalize_columns(columns):
    if columns is None:
        return None
    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)
    if not columns:
        return None
    return list(dict.fromkeys(columns))


class _BaseLagTransform(BaseEstimator):
    def _get_init_signature(self):
        return {
            k: v
            for k, v in inspect.signature(self.__class__.__init__).parameters.items()
            if k != "self" and v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }

    def _set_core_tfm(self, lag: int) -> "_BaseLagTransform":
        init_args = {k: getattr(self, k) for k in self._get_init_signature()}
        init_args.pop("global_", None)
        init_args.pop("global", None)
        init_args.pop("groupby", None)
        init_args.pop("partition_by", None)
        self._core_tfm = getattr(core_tfms, self.__class__.__name__)(
            lag=lag, **init_args
        )
        return self

    def _get_name(self, lag: int) -> str:
        init_params = self._get_init_signature()
        prefix_parts = []
        groupby = getattr(self, "groupby", None)
        partition_by = getattr(self, "partition_by", None)
        if getattr(self, "global_", False):
            prefix_parts.append("global")
        if groupby:
            group_str = "__".join(groupby)
            prefix_parts.append(f"groupby_{group_str}")
        if partition_by:
            partition_str = "__".join(partition_by)
            prefix_parts.append(f"partition_by_{partition_str}")
        prefix = "_".join(prefix_parts)
        if prefix:
            prefix += "_"
        result = f"{prefix}{_pascal2camel(self.__class__.__name__)}_lag{lag}"
        changed_params = [
            f"{name}{getattr(self, name)}"
            for name, arg in init_params.items()
            if arg.default != getattr(self, name)
            and name not in {"global_", "groupby", "partition_by"}
        ]
        if changed_params:
            result += "_" + "_".join(changed_params)
        return result

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,  # noqa: ARG002
        ts_arr: np.ndarray, # noqa: ARG002
        y_arr: np.ndarray, # noqa: ARG002
    ) -> Optional[np.ndarray]:
        """Compute the feature for a non-local (group/global) partition bucket.

        Called during ``_transform`` when ``partition_by`` transforms run in group
        or global mode.  ``bid_arr``, ``ts_arr``, ``y_arr`` are aligned numpy arrays
        over the sorted bucket DataFrame (ordered by bucket_id, timestamp, id).

        Returns a feature array of length ``len(bid_arr)``, or ``None`` to fall back
        to the default: position-based GroupedArray transform followed by
        same-timestamp correction (all observations sharing a ``(bucket_id, ts)``
        receive the feature value of the first observation in that group, which was
        computed from strictly earlier timestamps only).

        The default ``None`` is correct for unbounded (expanding) transforms because
        position-based expanding over observations sorted by timestamp is equivalent
        to a timestamp-based expanding window.  It is also used as a temporary
        fallback for transforms whose RANGE semantics have not yet been implemented
        (e.g. :class:`_Seasonal_RollingBase`, :class:`ExponentiallyWeightedMean`).

        Subclasses override this method when the position-based fallback produces
        semantically wrong results — primarily bounded rolling windows
        (:class:`_RollingBase`) where multiple series can share a timestamp.
        """
        return None

    def transform(self, ga: CoreGroupedArray) -> np.ndarray:
        return self._core_tfm.transform(ga)

    def update(self, ga: CoreGroupedArray) -> np.ndarray:
        return self._core_tfm.update(ga)

    def take(self, idxs: np.ndarray) -> "_BaseLagTransform":
        out = copy.deepcopy(self)
        out._core_tfm = self._core_tfm.take(idxs)
        return out

    @staticmethod
    def stack(transforms: Sequence["_BaseLagTransform"]) -> "_BaseLagTransform":
        out = copy.deepcopy(transforms[0])
        out._core_tfm = transforms[0]._core_tfm.stack(
            [tfm._core_tfm for tfm in transforms]
        )
        return out

    def _get_configured_lag(self) -> int:
        return self._core_tfm.lag

    @property
    def _lag(self):
        return self._core_tfm.lag - 1

    @property
    def update_samples(self) -> int:
        return -1


class Lag(_BaseLagTransform):
    def __init__(self, lag: int):
        self.lag = lag
        self._core_tfm = core_tfms.Lag(lag=lag)

    def _set_core_tfm(self, _lag: int) -> "Lag":
        return self

    def _get_name(self, lag: int) -> str:
        return f"lag{lag}"

    def __eq__(self, other):
        return isinstance(other, Lag) and self.lag == other.lag

    @property
    def update_samples(self) -> int:
        return self.lag


class _RollingBase(_BaseLagTransform):
    "Rolling statistic"

    def __init__(
        self,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """
        Args:
            window_size (int): Number of samples in the window.
            min_samples (int, optional): Minimum samples required to output the statistic.
                If `None`, will be set to `window_size`. Defaults to None.
            global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
                Requires all series to end at the same timestamp. Defaults to False.
            groupby (Sequence[str], optional): Column names to group by before computing the statistic.
                Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
            partition_by (Sequence[str], optional): Column names used to partition observations
                before computing the statistic. Defaults to None.
        """
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if "partition_by" in kwargs:
            partition_by = kwargs.pop("partition_by")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.window_size = window_size
        self.min_samples = min_samples
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        self.partition_by = _normalize_columns(partition_by)
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")

    @property
    def update_samples(self) -> int:
        return self._lag + self.window_size


    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        ts_arr: np.ndarray,
        y_arr: np.ndarray,
    ) -> np.ndarray:
        """RANGE-based rolling for non-local partition modes.

        For each row at ``(bucket_id, ds=T)`` collects all observations in the same
        bucket with ``ts ∈ [T - lag - window_size + 1, T - lag]`` and applies
        :meth:`_window_stat`.  This matches SQL
        ``RANGE BETWEEN (lag + window_size - 1) PRECEDING AND lag PRECEDING``.

        The default loop is O(n × w).  Subclasses may override this method with a
        more efficient algorithm when the statistic supports it (e.g. ``RollingMean``
        uses cumulative sums for O(n log n) performance).
        """
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        n = len(bid_arr)
        result = np.empty(n)
        result[:] = np.nan
        for bid in np.unique(bid_arr):
            idxs = np.where(bid_arr == bid)[0]
            ts_b = ts_arr[idxs]
            y_b = y_arr[idxs]
            unique_ts, inv = np.unique(ts_b, return_inverse=True)
            feat_u = np.full(len(unique_ts), np.nan)
            for k, T in enumerate(unique_ts):
                lower, upper = T - lag - w + 1, T - lag
                mask = (ts_b >= lower) & (ts_b <= upper)
                vals = y_b[mask]
                if len(vals) >= min_samples:
                    feat_u[k] = self._window_stat(vals)
            result[idxs] = feat_u[inv]
        return result

    def _window_stat(self, vals: np.ndarray) -> float:
        """Compute the statistic over ``vals``, the individual observations in the window.

        Subclasses must implement this; it is called by :meth:`_compute_bucket_feature`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement `_window_stat` to support "
            "RANGE-based rolling in non-local partition modes."
        )


class RollingMean(_RollingBase):
    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        ts_arr: np.ndarray,
        y_arr: np.ndarray,
    ) -> np.ndarray:
        """O(m log m) override using cumulative per-bucket sums, where m = unique timestamps.

        Computes each feature value once per unique timestamp in the bucket, then
        broadcasts back to all rows sharing that timestamp via the ``inv`` index
        from ``np.unique``.
        """
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        n = len(bid_arr)
        result = np.empty(n)
        result[:] = np.nan
        for bid in np.unique(bid_arr):
            idxs = np.where(bid_arr == bid)[0]
            ts = ts_arr[idxs]
            y = y_arr[idxs]
            unique_ts, inv, counts = np.unique(
                ts, return_inverse=True, return_counts=True
            )
            ts_sums = np.bincount(inv, weights=y, minlength=len(unique_ts))
            cum_sum = np.cumsum(ts_sums)
            cum_cnt = np.cumsum(counts).astype(float)
            upper_ts_u = unique_ts - lag
            lower_ts_u = unique_ts - lag - w
            upper_idxs = np.searchsorted(unique_ts, upper_ts_u, side="right") - 1
            lower_idxs = np.searchsorted(unique_ts, lower_ts_u, side="right") - 1
            upper_sum = np.where(upper_idxs >= 0, cum_sum[upper_idxs], 0.0)
            upper_cnt = np.where(upper_idxs >= 0, cum_cnt[upper_idxs], 0.0)
            lower_sum = np.where(lower_idxs >= 0, cum_sum[lower_idxs], 0.0)
            lower_cnt = np.where(lower_idxs >= 0, cum_cnt[lower_idxs], 0.0)
            win_sum = upper_sum - lower_sum
            win_cnt = upper_cnt - lower_cnt
            feat_u = np.where(win_cnt >= min_samples, win_sum / win_cnt, np.nan)
            result[idxs] = feat_u[inv]
        return result


class RollingStd(_RollingBase):
    def _window_stat(self, vals: np.ndarray) -> float:
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan


class RollingMin(_RollingBase):
    def _window_stat(self, vals: np.ndarray) -> float:
        return float(np.min(vals))


class RollingMax(_RollingBase):
    def _window_stat(self, vals: np.ndarray) -> float:
        return float(np.max(vals))


class RollingQuantile(_RollingBase):
    def __init__(
        self,
        p: float,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            min_samples=min_samples,
            global_=global_,
            groupby=groupby,
            partition_by=partition_by,
            **kwargs,
        )
        self.p = p

    def _set_core_tfm(self, lag: int):
        self._core_tfm = core_tfms.RollingQuantile(
            lag=lag,
            p=self.p,
            window_size=self.window_size,
            min_samples=self.min_samples,
        )
        return self

    def _window_stat(self, vals: np.ndarray) -> float:
        return float(np.quantile(vals, self.p))


class _Seasonal_RollingBase(_BaseLagTransform):
    """Rolling statistic over seasonal periods"""

    def __init__(
        self,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """
        Args:
            season_length (int): Periodicity of the seasonal period.
            window_size (int): Number of samples in the window.
            min_samples (int, optional): Minimum samples required to output the statistic.
                If `None`, will be set to `window_size`. Defaults to None.
            global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
                Requires all series to end at the same timestamp. Defaults to False.
            groupby (Sequence[str], optional): Column names to group by before computing the statistic.
                Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
            partition_by (Sequence[str], optional): Column names used to partition observations
                before computing the statistic. Defaults to None.
        """
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if "partition_by" in kwargs:
            partition_by = kwargs.pop("partition_by")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.season_length = season_length
        self.window_size = window_size
        self.min_samples = min_samples
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        self.partition_by = _normalize_columns(partition_by)
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")

    @property
    def update_samples(self) -> int:
        return self._lag + self.season_length * self.window_size


class SeasonalRollingMean(_Seasonal_RollingBase): ...


class SeasonalRollingStd(_Seasonal_RollingBase): ...


class SeasonalRollingMin(_Seasonal_RollingBase): ...


class SeasonalRollingMax(_Seasonal_RollingBase): ...


class SeasonalRollingQuantile(_Seasonal_RollingBase):
    def __init__(
        self,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
            global_=global_,
            groupby=groupby,
            partition_by=partition_by,
            **kwargs,
        )
        self.p = p


class _ExpandingBase(_BaseLagTransform):
    """Expanding statistic

    Args:
        global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
            Requires all series to end at the same timestamp. Defaults to False.
        groupby (Sequence[str], optional): Column names to group by before computing the statistic.
            Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
    """

    def __init__(
        self,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if "partition_by" in kwargs:
            partition_by = kwargs.pop("partition_by")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        self.partition_by = _normalize_columns(partition_by)
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")

    @property
    def update_samples(self) -> int:
        return 1


class ExpandingMean(_ExpandingBase): ...


class ExpandingStd(_ExpandingBase): ...


class ExpandingMin(_ExpandingBase): ...


class ExpandingMax(_ExpandingBase): ...


class ExpandingQuantile(_ExpandingBase):
    def __init__(
        self,
        p: float,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(
            global_=global_,
            groupby=groupby,
            partition_by=partition_by,
            **kwargs,
        )
        self.p = p

    @property
    def update_samples(self) -> int:
        return -1


class ExponentiallyWeightedMean(_BaseLagTransform):
    """Exponentially weighted average

    Args:
        alpha (float): Smoothing factor.
        global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
            Requires all series to end at the same timestamp. Defaults to False.
        groupby (Sequence[str], optional): Column names to group by before computing the statistic.
            Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
    """

    def __init__(
        self,
        alpha: float,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if "partition_by" in kwargs:
            partition_by = kwargs.pop("partition_by")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.alpha = alpha
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        self.partition_by = _normalize_columns(partition_by)
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")

    @property
    def update_samples(self) -> int:
        return 1


class Offset(_BaseLagTransform):
    """Shift series before computing transformation

    Args:
        tfm (LagTransform): Transformation to be applied
        n (int): Number of positions to shift (lag) series before applying the transformation
    """

    def __init__(self, tfm: _BaseLagTransform, n: int):
        self.tfm = tfm
        self.n = n
        self.global_ = getattr(tfm, "global_", False)
        self.groupby = getattr(tfm, "groupby", None)
        self.partition_by = getattr(tfm, "partition_by", None)

    def _get_name(self, lag: int) -> str:
        return self.tfm._get_name(lag + self.n)

    def _set_core_tfm(self, lag: int) -> "Offset":
        self.tfm = copy.deepcopy(self.tfm)._set_core_tfm(lag + self.n)
        self._core_tfm = self.tfm._core_tfm
        return self

    def _get_configured_lag(self) -> int:
        return self.tfm._get_configured_lag() - self.n

    @property
    def update_samples(self) -> int:
        return self.tfm.update_samples + self.n


class Combine(_BaseLagTransform):
    """Combine two lag transformations using an operator

    Args:
        tfm1 (LagTransform): First transformation.
        tfm2 (LagTransform): Second transformation.
        operator (callable): Binary operator that defines how to combine the two transformations.
    """

    def __init__(
        self, tfm1: _BaseLagTransform, tfm2: _BaseLagTransform, operator: Callable
    ):
        self.tfm1 = tfm1
        self.tfm2 = tfm2
        self.operator = operator
        global_1 = getattr(tfm1, "global_", False)
        global_2 = getattr(tfm2, "global_", False)
        groupby_1 = getattr(tfm1, "groupby", None)
        groupby_2 = getattr(tfm2, "groupby", None)
        partition_by_1 = getattr(tfm1, "partition_by", None)
        partition_by_2 = getattr(tfm2, "partition_by", None)
        if global_1 != global_2:
            raise ValueError("Can't combine transforms with different global_ settings.")
        if (groupby_1 or groupby_2) and groupby_1 != groupby_2:
            raise ValueError("Can't combine transforms with different groupby settings.")
        if (partition_by_1 or partition_by_2) and partition_by_1 != partition_by_2:
            raise ValueError("Can't combine transforms with different partition_by settings.")
        self.global_ = global_1
        self.groupby = groupby_1
        self.partition_by = partition_by_1

    def _set_core_tfm(self, lag: int) -> "Combine":
        self.tfm1 = copy.deepcopy(self.tfm1)._set_core_tfm(lag)
        self.tfm2 = copy.deepcopy(self.tfm2)._set_core_tfm(lag)
        return self

    def _get_name(self, lag: int) -> str:
        lag1 = getattr(self.tfm1, "lag", lag)
        lag2 = getattr(self.tfm2, "lag", lag)
        return f"{self.tfm1._get_name(lag1)}_{self.operator.__name__}_{self.tfm2._get_name(lag2)}"

    def transform(self, ga: CoreGroupedArray) -> np.ndarray:
        return self.operator(self.tfm1.transform(ga), self.tfm2.transform(ga))

    def update(self, ga: CoreGroupedArray) -> np.ndarray:
        return self.operator(self.tfm1.update(ga), self.tfm2.update(ga))

    def take(self, idxs: np.ndarray) -> "Combine":
        out = copy.deepcopy(self)
        out.tfm1 = self.tfm1.take(idxs)
        out.tfm2 = self.tfm2.take(idxs)
        return out

    @staticmethod
    def stack(transforms: Sequence["Combine"]) -> "Combine":
        out = copy.copy(transforms[0])
        out.tfm1 = transforms[0].tfm1.stack([tfm.tfm1 for tfm in transforms])
        out.tfm2 = transforms[0].tfm2.stack([tfm.tfm2 for tfm in transforms])
        return out

    def _get_configured_lag(self) -> int:
        lag1 = self.tfm1._get_configured_lag()
        lag2 = self.tfm2._get_configured_lag()
        if lag1 != lag2:
            raise ValueError("Combined transforms must share the same configured lag.")
        return lag1

    @property
    def update_samples(self):
        return max(self.tfm1.update_samples, self.tfm2.update_samples)
