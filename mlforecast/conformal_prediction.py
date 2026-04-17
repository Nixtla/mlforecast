__all__ = ["PredictionIntervals"]

from typing import Callable, List, Union

import numpy as np
import utilsforecast.processing as ufp
from utilsforecast.compat import DFType


class PredictionIntervals:
    """Class for storing prediction intervals metadata information."""

    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = "conformal_distribution",
    ):
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
        return f"PredictionIntervals(n_windows={self.n_windows}, h={self.h}, method='{self.method}')"


def _add_conformal_distribution_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
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
        mean = fcst_df[model].to_numpy().reshape(1, n_series, -1)
        scores = np.vstack([mean - scores, mean + scores])
        quantiles = np.quantile(
            scores,
            cuts,
            axis=0,
        )
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
        quantiles = np.quantile(
            scores,
            cuts,
            axis=0,
        )
        quantiles = quantiles.reshape(len(cuts), -1)
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        quantiles = np.vstack([mean - quantiles[::-1], mean + quantiles]).T
        columns = lo_cols + hi_cols
        fcst_df = ufp.assign_columns(fcst_df, columns, quantiles)
    return fcst_df


_INTERVAL_METHODS = {
    "conformal_distribution": _add_conformal_distribution_intervals,
    "conformal_error": _add_conformal_error_intervals,
}


def get_conformal_method(method: str) -> Callable:
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
) -> DFType:
    """Compute absolute-error conformity scores over CV folds."""
    for model in model_names:
        abs_err = abs(cv_results[model] - cv_results[target_col])
        cv_results = ufp.assign_columns(cv_results, model, abs_err)
    return ufp.drop_columns(cv_results, target_col)


def _recalibrate_transfer(
    new_df: DFType,
    prediction_intervals: PredictionIntervals,
    cv_fn: Callable,
    model_names: List[str],
    target_col: str,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> DFType:
    """Recompute conformity scores via cross_validation on new_df."""
    cv_results = cv_fn(
        df=new_df,
        n_windows=prediction_intervals.n_windows,
        h=prediction_intervals.h,
        refit=False,
        prediction_intervals=None,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )
    return compute_conformity_scores(cv_results, model_names, target_col)


_TRANSFER_METHODS = {
    "recalibrate": _recalibrate_transfer,
}


def get_transfer_conformal_method(method: str) -> Callable:
    if method not in _TRANSFER_METHODS:
        raise ValueError(
            f"transfer conformal method {method} not supported "
            f"please choose one of {', '.join(_TRANSFER_METHODS.keys())}"
        )
    return _TRANSFER_METHODS[method]
