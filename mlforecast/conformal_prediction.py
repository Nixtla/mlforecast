__all__ = ["PredictionIntervals", "TransferConformal"]

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import narwhals as nw
import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
from utilsforecast.compat import DFType


_VALID_TRANSFER_METHODS = ("recalibrate", "scale_aligned", "error_scaled")


@dataclass
class TransferConformal:
    """Configuration for adapting the prediction intervals to transferred series.

    Pass to ``MLForecast.transfer``. A plain string is shorthand for
    ``TransferConformal(method=<str>)``.

    Args:
        method (str): ``'recalibrate'``, ``'error_scaled'`` or ``'scale_aligned'``. Defaults to 'recalibrate'.
        h (int, optional): Forecast horizon of the conformity scores. Defaults to the ``h`` of ``PredictionIntervals``. Required for ``'recalibrate'`` on a model fit without prediction intervals.
        n_windows (int, optional): Number of backtest windows on the transferred series. Only for ``'recalibrate'`` and ``'error_scaled'``. Defaults to the ``n_windows`` of ``PredictionIntervals``, or 2 if the model was fit without them.
        step_size (int, optional): Step size between the backtest windows. Only for ``'recalibrate'`` and ``'error_scaled'``. Defaults to 1.
        scale_estimator (str): Per-series scale estimator, ``'mad'`` or ``'std'``. Only for ``'scale_aligned'``. Defaults to 'mad'.
    """

    method: str = "recalibrate"
    h: Optional[int] = None
    n_windows: Optional[int] = None
    step_size: Optional[int] = None
    scale_estimator: Literal["mad", "std"] = "mad"

    def __post_init__(self) -> None:
        if self.method not in _VALID_TRANSFER_METHODS:
            raise ValueError(
                f"TransferConformal.method must be one of {_VALID_TRANSFER_METHODS}, "
                f"got '{self.method}'"
            )
        if self.scale_estimator not in ("mad", "std"):
            raise ValueError(
                "TransferConformal.scale_estimator must be 'mad' or 'std', "
                f"got '{self.scale_estimator}'"
            )
        if self.h is not None and self.h < 1:
            raise ValueError(f"TransferConformal.h must be >= 1, got {self.h}")
        if self.n_windows is not None and self.n_windows < 1:
            raise ValueError(
                f"TransferConformal.n_windows must be >= 1, got {self.n_windows}"
            )
        if self.step_size is not None and self.step_size < 1:
            raise ValueError(
                f"TransferConformal.step_size must be >= 1, got {self.step_size}"
            )
        if self.method == "scale_aligned" and (
            self.n_windows is not None or self.step_size is not None
        ):
            raise ValueError(
                "TransferConformal.n_windows and step_size are only valid with "
                f"method='recalibrate' or 'error_scaled', got method='{self.method}'"
            )
        if self.method == "recalibrate" and self.n_windows == 1:
            raise ValueError(
                "TransferConformal(method='recalibrate') requires at least 2 "
                "windows, got n_windows=1"
            )
        if self.method != "scale_aligned" and self.scale_estimator != "mad":
            raise ValueError(
                "TransferConformal.scale_estimator is only valid with "
                f"method='scale_aligned', got method='{self.method}'"
            )


class PredictionIntervals:
    """Class for storing prediction intervals metadata information."""

    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = "conformal_distribution",
    ):
        """
        Args:
            n_windows (int): Number of cross validation windows used to compute the conformity scores. Defaults to 2.
            h (int): Forecast horizon of each cross validation window. Defaults to 1.
            method (str): ``'conformal_distribution'`` or ``'conformal_error'``. Defaults to 'conformal_distribution'.
        """
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed_methods = ["conformal_error", "conformal_distribution"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.n_windows = n_windows
        self.h = h
        self.method = method

    def __repr__(self):
        return (
            f"PredictionIntervals(n_windows={self.n_windows}, h={self.h}, "
            f"method='{self.method}')"
        )


def _compute_series_scales(
    df: DFType,
    id_col: str,
    time_col: str,
    target_col: str,
    method: str,
    floor_factor: float = 1e-3,
) -> Dict:
    """Compute per-series scale estimates on first differences for trend invariance.

    Uses MAD(Δy) or std(Δy). A data-relative floor (floor_factor × global median)
    with an absolute backstop of 1e-8 prevents zero-scale collapse for flat or very
    short series.
    """
    raw = _raw_series_scales(df, id_col, time_col, target_col, (method,))
    return _floor_series_scales(raw[method], floor_factor, stacklevel=3)


def _raw_series_scales(
    df: DFType,
    id_col: str,
    time_col: str,
    target_col: str,
    methods: Sequence[str],
) -> Dict[str, Dict]:
    """MAD(Δy) and/or std(Δy) per series, before the floor of `_floor_series_scales`."""
    ndf = nw.from_native(df, eager_only=True)
    ndf = ndf.select([id_col, time_col, target_col]).with_columns(
        nw.col(target_col)
        .cast(nw.Float64)
        .diff()
        .over(id_col, order_by=time_col)
        .alias("_dy")
    )
    # Precompute abs columns so every agg below is a plain column aggregation
    # (avoids narwhals' "complex group-by expression" pandas warning/fallback).
    ndf = ndf.with_columns(
        nw.col(target_col).abs().alias("_abs_y"),
        nw.col("_dy").abs().alias("_abs_dy_col"),
    )
    scale_exprs = []
    if "mad" in methods:
        ndf = ndf.with_columns(
            (nw.col("_dy") - nw.col("_dy").median().over(id_col)).abs().alias("_ady")
        )
        scale_exprs.append(nw.col("_ady").median().alias("_mad"))
    if "std" in methods:
        scale_exprs.append(nw.col("_dy").std(ddof=1).alias("_std"))
    stats = ndf.group_by(id_col).agg(
        *scale_exprs,
        nw.col("_dy").count().alias("_n_dy"),
        nw.col("_abs_y").mean().alias("_abs_mean"),
        nw.col("_abs_dy_col").max().alias("_max_abs_dy"),
    )
    uids = stats[id_col].to_list()
    n_dys = stats["_n_dy"].to_list()
    abs_means = stats["_abs_mean"].to_list()
    max_abs_dys = stats["_max_abs_dy"].to_list()
    raw: Dict[str, Dict] = {}
    for method in methods:
        raw[method] = {}
        for uid, scale, n_dy, abs_mean, max_abs_dy in zip(
            uids, stats[f"_{method}"].to_list(), n_dys, abs_means, max_abs_dys
        ):
            if n_dy == 0:
                # single-observation series: no diffs available
                raw[method][uid] = float(abs_mean) if abs_mean is not None else 1.0
            elif method == "std" and n_dy == 1:
                # std(ddof=1) of a single diff is NaN; use |dy[0]| = max(|dy|) as before
                raw[method][uid] = float(max_abs_dy)
            else:
                raw[method][uid] = float(scale)
    return raw


def _floor_series_scales(
    raw: Dict, floor_factor: float, stacklevel: int, kind: str = "Series"
) -> Dict:
    vals = np.array(list(raw.values()), dtype=float)
    global_median = float(np.median(vals)) if len(vals) > 0 else 1.0
    floor = max(floor_factor * global_median, 1e-8)
    floored: Dict = {}
    for uid, v in raw.items():
        effective = max(v, floor)
        if effective > v:
            warnings.warn(
                f"{kind} '{uid}' has near-zero scale estimate ({v:.2e}); "
                f"applying floor {floor:.2e}. Check for flat or constant series.",
                UserWarning,
                stacklevel=stacklevel,
            )
        floored[uid] = effective
    return floored


def _apply_scale_alignment(
    cs_df: DFType,
    model_names: List[str],
    id_col: str,
    source_scales: Dict,
) -> DFType:
    """Normalize source residuals by per-series source scale.

    For each source row: normalized_residual = residual / σ̂_src(uid).
    The per-target-series σ̂_tgt multiplication is applied post-hoc in
    ``predict()`` after the quantile step.  Returns a copy; does not mutate.
    """
    cs_df = ufp.copy_if_pandas(cs_df, deep=False)
    uid_arr = cs_df[id_col].to_numpy()
    # hash-based factorization: avoids np.unique's O(n log n) sort on string ids
    codes, uniques = pd.factorize(uid_arr)
    unique_norm = np.array([1.0 / source_scales[uid] for uid in uniques], dtype=float)
    norm_scales = unique_norm[codes]
    for model in model_names:
        vals = cs_df[model].to_numpy().astype(float) * norm_scales
        cs_df = ufp.assign_columns(cs_df, model, vals)
    return cs_df


def _rescale_interval_columns(
    fcst_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    sigma_tgt: np.ndarray,
) -> DFType:
    """Multiply interval half-widths by per-row σ̂_tgt, one assignment per column."""
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    for model_name in model_names:
        mean_arr = fcst_df[model_name].to_numpy().astype(float)
        for lv in level:
            for direction in ("lo", "hi"):
                col = f"{model_name}-{direction}-{lv}"
                offset = fcst_df[col].to_numpy().astype(float) - mean_arr
                fcst_df = ufp.assign_columns(
                    fcst_df, col, mean_arr + offset * sigma_tgt
                )
    return fcst_df


def _add_conformal_distribution_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
    is_transfer: bool = False,
) -> DFType:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This strategy creates forecasts paths
    based on errors and calculate quantiles using those paths.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    alphas = [100 - lv for lv in level]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    for model in model_names:
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        # restrict scores to horizon
        scores = scores[:, :, :horizon]
        mean_flat = fcst_df[model].to_numpy().ravel()
        n_target = len(mean_flat) // horizon
        if is_transfer:
            # Transfer scenario: pool all source calibration points globally.
            # quantile(mean_t + {-s_i, +s_i}) = mean_t + quantile({-s_i, +s_i})
            scores_pooled = scores.reshape(cs_n_windows * n_series, horizon)
            sym_scores = np.vstack([-scores_pooled, scores_pooled])
            global_q = np.quantile(sym_scores, cuts, axis=0)  # (n_cuts, horizon)
            mean_2d = mean_flat.reshape(n_target, horizon)
            quantiles = (
                (global_q[:, np.newaxis, :] + mean_2d[np.newaxis, :, :])
                .reshape(len(cuts), -1)
                .T
            )  # (n_target * horizon, n_cuts)
        else:
            mean = mean_flat.reshape(1, n_series, -1)
            paths = np.vstack([mean - scores, mean + scores])
            quantiles = np.quantile(paths, cuts, axis=0)
            quantiles = quantiles.reshape(len(cuts), -1).T
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        out_cols = lo_cols + hi_cols
        fcst_df = ufp.assign_columns(fcst_df, out_cols, quantiles)
    return fcst_df


def _add_conformal_error_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
    is_transfer: bool = False,
) -> DFType:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This startegy creates prediction intervals
    based on the absolute errors.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    cuts = [lv / 100 for lv in level]
    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        # restrict scores to horizon
        scores = scores[:, :, :horizon]
        n_target = len(mean) // horizon
        if is_transfer:
            # Transfer scenario: pool all source calibration points globally.
            # Compute a single quantile per horizon step, then tile to all target series.
            scores_pooled = scores.reshape(cs_n_windows * n_series, horizon)
            quantiles = np.quantile(scores_pooled, cuts, axis=0)  # (n_levels, horizon)
            quantiles = np.tile(
                quantiles, (1, n_target)
            )  # (n_levels, n_target * horizon)
        else:
            quantiles = np.quantile(scores, cuts, axis=0)
            quantiles = quantiles.reshape(len(cuts), -1)
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        quantiles = np.vstack([mean - quantiles[::-1], mean + quantiles]).T
        columns = lo_cols + hi_cols
        fcst_df = ufp.assign_columns(fcst_df, columns, quantiles)
    return fcst_df


def _add_signed_transfer_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    horizon: int,
    cs_h: int = 1,
) -> DFType:
    """Add asymmetric prediction intervals from signed conformity scores (y − pred).

    Unlike the symmetric conformal methods, uses separate lower and upper quantiles
    so that a systematically biased transferred model shifts the interval rather than
    merely widening it.

    When ``cs_h == 1`` but ``horizon > 1`` (calibration was done with 1-step ahead
    cross-validation), the same constant-width interval is broadcast across all
    horizon steps, matching the behaviour of the regular conformal methods in that
    case.

    Emits UserWarning when the interval lies entirely above or below the point forecast
    (both quantiles have the same sign), indicating severe source-model bias on the
    target domain.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    # level is pre-sorted ascending (e.g. [80, 90, 95])
    # lo cols: reversed levels (widest first) → [lo-95, lo-90, lo-80]
    # hi cols: ascending levels (narrowest first) → [hi-80, hi-90, hi-95]
    lo_cuts = [((100 - lv) / 100) / 2 for lv in reversed(level)]
    hi_cuts = [1 - ((100 - lv) / 100) / 2 for lv in level]
    all_cuts = lo_cuts + hi_cuts

    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().astype(float).reshape(-1, cs_h)
        # (n_cuts, horizon), or (n_cuts, 1) broadcast to every step when cs_h=1
        q_per_horizon = np.quantile(scores[:, :horizon], all_cuts, axis=0)

        # Bias warnings
        n_lo = len(lo_cuts)
        q_hi_vals = q_per_horizon[n_lo:]  # (n_hi_cuts, horizon)
        q_lo_vals = q_per_horizon[:n_lo]  # (n_lo_cuts, horizon)
        for i, lv in enumerate(level):
            q_hi_lv = q_hi_vals[i]  # upper quantile for this level
            q_lo_lv = q_lo_vals[-(i + 1)]  # matching lower quantile
            if np.all(q_hi_lv < 0):
                warnings.warn(
                    f"Transfer recalibrate level={lv}: upper quantile is negative across "
                    "all horizon steps — interval lies entirely below point forecasts. "
                    "The transferred model systematically over-predicts on the target domain.",
                    UserWarning,
                    stacklevel=3,
                )
            elif np.all(q_lo_lv > 0):
                warnings.warn(
                    f"Transfer recalibrate level={lv}: lower quantile is positive across "
                    "all horizon steps — interval lies entirely above point forecasts. "
                    "The transferred model systematically under-predicts on the target domain.",
                    UserWarning,
                    stacklevel=3,
                )

        # Apply: lo = pred + q_lo, hi = pred + q_hi, broadcast per horizon step
        n_target = len(mean) // horizon
        mean_2d = mean.reshape(n_target, horizon)
        quantiles = (
            (q_per_horizon[:, np.newaxis, :] + mean_2d[np.newaxis, :, :])
            .reshape(len(all_cuts), -1)
            .T
        )  # (n_target * horizon, n_cuts)

        out_cols = [f"{model}-lo-{lv}" for lv in reversed(level)] + [
            f"{model}-hi-{lv}" for lv in level
        ]
        fcst_df = ufp.assign_columns(fcst_df, out_cols, quantiles)

    return fcst_df


_INTERVAL_METHODS: Dict[str, Callable[..., Any]] = {
    "conformal_distribution": _add_conformal_distribution_intervals,
    "conformal_error": _add_conformal_error_intervals,
}


def get_conformal_method(method: str) -> Callable[..., Any]:
    if method not in _INTERVAL_METHODS:
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f"please choose one of {', '.join(_INTERVAL_METHODS.keys())}"
        )
    return _INTERVAL_METHODS[method]


def compute_conformity_scores(
    cv_results: DFType,
    model_names: List[str],
    target_col: str,
    signed: bool = False,
) -> DFType:
    """Compute conformity scores over CV folds.

    Args:
        cv_results: Cross-validation output with actual and predicted columns.
        model_names: Names of model columns to convert to errors.
        target_col: Name of the target column (dropped from output).
        signed: If True, compute signed errors (y - pred). If False (default),
            compute absolute errors |pred - target|.
    """
    for model in model_names:
        if signed:
            err = cv_results[target_col] - cv_results[model]
        else:
            err = abs(cv_results[model] - cv_results[target_col])
        cv_results = ufp.assign_columns(cv_results, model, err)
    return ufp.drop_columns(cv_results, target_col)


@dataclass
class TransferResult:
    """Return value from transfer conformal functions."""

    cs_df: DFType
    target_scales: Optional[Dict[str, float]] = None  # uid -> sigma_tgt
    signed: bool = False


@dataclass
class Calibration:
    """Conformity scores and per-series scales of the training series, computed at fit."""

    scores: DFType
    source_scales: Optional[Dict[str, Dict]] = None  # estimator -> uid -> raw scale


@dataclass
class TransferState:
    """Transfer method chosen in ``MLForecast.transfer`` and its adaptation to `df`.

    `result` is computed from `df` on the first ``predict(level=...)``.
    """

    method: Optional[TransferConformal] = None
    df: Optional[DFType] = None
    train_ends: Optional[DFType] = None
    result: Optional[TransferResult] = None


def _robust_scale_ratio(src: np.ndarray, tgt: np.ndarray) -> float:
    """IQR(|tgt_errors|) / IQR(|src_errors|) with std and constant fallbacks."""
    src_abs = np.abs(src)
    tgt_abs = np.abs(tgt)
    iqr_src = float(np.percentile(src_abs, 75) - np.percentile(src_abs, 25))
    iqr_tgt = float(np.percentile(tgt_abs, 75) - np.percentile(tgt_abs, 25))
    if iqr_src >= 1e-10 and iqr_tgt >= 1e-10:
        return iqr_tgt / iqr_src
    # Fallback 1: std ratio
    std_src = float(np.std(src)) if len(src) > 1 else 0.0
    std_tgt = float(np.std(tgt)) if len(tgt) > 1 else 0.0
    if std_src >= 1e-10:
        warnings.warn(
            "IQR of residuals near zero; falling back to std ratio for scale estimation.",
            UserWarning,
            stacklevel=4,
        )
        return std_tgt / max(std_src, 1e-10)
    # Fallback 2: constant
    warnings.warn(
        "Both IQR and std of residuals near zero; scale ratio defaulting to 1.0.",
        UserWarning,
        stacklevel=4,
    )
    return 1.0


def _recalibrate_transfer(
    backtest_results: DFType,
    model_names: List[str],
    target_col: str,
) -> TransferResult:
    """Recompute conformity scores from frozen-model backtest on new_df.

    Uses signed residuals (y − pred) so systematic bias shifts the interval
    instead of merely widening it.
    """
    return TransferResult(
        cs_df=compute_conformity_scores(
            backtest_results, model_names, target_col, signed=True
        ),
        signed=True,
    )


def _scale_aligned_transfer(
    new_df: DFType,
    model_names: List[str],
    id_col: str,
    time_col: str,
    target_col: str,
    scale_estimator: str,
    source_cs_df: DFType,
    source_scales: Dict,
) -> TransferResult:
    """Zero-shot scale alignment: compute σ̂_target per series from new_df y history.

    Returns the source conformity scores normalized by σ̂_source together with
    per-series target scales in ``TransferResult.target_scales``.
    """
    # called from MLForecast.predict through _adapt, warn at the user's call
    raw_target = _raw_series_scales(
        new_df, id_col, time_col, target_col, (scale_estimator,)
    )
    target_scale_dict = _floor_series_scales(
        raw_target[scale_estimator], floor_factor=1e-3, stacklevel=5
    )
    source_scales = _floor_series_scales(
        source_scales, floor_factor=1e-3, stacklevel=5, kind="Training series"
    )
    return TransferResult(
        cs_df=_apply_scale_alignment(source_cs_df, model_names, id_col, source_scales),
        target_scales=target_scale_dict,
    )


def _error_scaled_transfer(
    backtest_results: DFType,
    model_names: List[str],
    target_col: str,
    source_cs_df: DFType,
) -> TransferResult:
    """Scale source conformity scores by a single global IQR ratio (target / source).

    The ratio is computed from the pooled residuals of all series combined — not
    per-series. This corrects for domain-level scale differences between source and
    target but does not account for series-level heterogeneity within the target domain.
    For per-series scale correction use ``scale_aligned`` instead.
    """
    target_cs_df = compute_conformity_scores(backtest_results, model_names, target_col)

    scaled = ufp.copy_if_pandas(source_cs_df, deep=False)
    for model in model_names:
        src = source_cs_df[model].to_numpy().astype(float)
        tgt = target_cs_df[model].to_numpy().astype(float)
        scale = _robust_scale_ratio(src, tgt)
        scaled = ufp.assign_columns(scaled, model, src * scale)

    return TransferResult(cs_df=scaled)
