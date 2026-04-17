__all__ = ["PredictionIntervals", "estimate_density_ratio"]

from typing import Callable, List, Optional, Union

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
        allowed_methods = [
            "conformal_error",
            "conformal_distribution",
            "weighted_conformal_error",
            "weighted_conformal_distribution",
        ]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.n_windows = n_windows
        self.h = h
        self.method = method
        self._cs_weights: Optional[np.ndarray] = None

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
    weights: Optional[np.ndarray] = None,
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
    weights: Optional[np.ndarray] = None,
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


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    w_test: float = 1.0,
) -> float:
    """Tibshirani et al. (2019) Eq. (1): weighted (1-alpha) quantile.

    Appends a virtual ∞ residual with weight w_test, then returns the
    leftmost residual value whose cumulative weight meets (1-alpha).
    """
    total = weights.sum() + w_test
    sort_idx = np.argsort(values)
    sorted_vals = np.append(values[sort_idx], np.inf)
    sorted_w = np.append(weights[sort_idx] / total, w_test / total)
    cum_w = np.cumsum(sorted_w)
    idx = np.searchsorted(cum_w, 1.0 - alpha, side="left")
    return float(sorted_vals[idx])


def _add_weighted_conformal_error_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
    weights: Optional[np.ndarray] = None,
) -> DFType:
    """Weighted conformal error intervals per Tibshirani et al. (2019).

    When weights is None falls back to np.quantile (identical to unweighted).
    When weights are provided computes a global weighted quantile per horizon
    step across all n_windows * n_series calibration points, applied uniformly
    to all target-domain series.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    cuts = [lv / 100 for lv in level]
    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        scores = scores[:, :, :horizon]
        if weights is None:
            quantiles = np.quantile(scores, cuts, axis=0)
            quantiles = quantiles.reshape(len(cuts), -1)
        else:
            w = weights.reshape(cs_n_windows, n_series, cs_h)[:, :, :horizon]
            w_test = float(w.mean())
            quantiles = np.empty((len(cuts), n_series * horizon))
            for h_i in range(horizon):
                s_h = scores[:, :, h_i].ravel()
                w_h = w[:, :, h_i].ravel()
                for li, alpha in enumerate([1 - c for c in cuts]):
                    q = _weighted_quantile(s_h, w_h, alpha, w_test)
                    quantiles[li, h_i::horizon] = q
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        quantiles_out = np.vstack([mean - quantiles[::-1], mean + quantiles]).T
        fcst_df = ufp.assign_columns(fcst_df, lo_cols + hi_cols, quantiles_out)
    return fcst_df


def _add_weighted_conformal_distribution_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
    weights: Optional[np.ndarray] = None,
) -> DFType:
    """Weighted conformal distribution intervals per Tibshirani et al. (2019).

    When weights is None falls back to np.quantile (identical to unweighted).
    When weights are provided applies _weighted_quantile over synthetic paths.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    alphas = [100 - lv for lv in level]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    for model in model_names:
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        scores = scores[:, :, :horizon]
        mean = fcst_df[model].to_numpy().reshape(1, n_series, -1)
        paths = np.vstack([mean - scores, mean + scores])  # (2*n_windows, n_series, horizon)
        if weights is None:
            quantiles = np.quantile(paths, cuts, axis=0)
            quantiles = quantiles.reshape(len(cuts), -1).T
        else:
            w = weights.reshape(cs_n_windows, n_series, cs_h)[:, :, :horizon]
            w_double = np.vstack([w, w])  # replicate for both path directions
            w_test = float(w.mean())
            quantiles = np.empty((n_series * horizon, len(cuts)))
            for h_i in range(horizon):
                p_h = paths[:, :, h_i].ravel()
                w_h = w_double[:, :, h_i].ravel()
                for li, cut in enumerate(cuts):
                    q = _weighted_quantile(p_h, w_h, 1.0 - cut, w_test)
                    quantiles[h_i::horizon, li] = q
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        fcst_df = ufp.assign_columns(fcst_df, lo_cols + hi_cols, quantiles)
    return fcst_df


def estimate_density_ratio(
    source_features: np.ndarray,
    target_features: np.ndarray,
    estimator: str = "logistic",
) -> np.ndarray:
    """Estimate w(x) = p_target(x) / p_source(x) for source domain points.

    Trains a binary classifier (source=0, target=1) on StandardScaler-
    normalised features and returns the odds ratio p(1|x) / p(0|x) for
    each source point. Weights are clipped at 1e-10 to avoid division by zero.

    Args:
        source_features: Feature matrix for source-domain calibration points,
            shape (n_source, n_features). Obtain via ``fcst.preprocess(source_df)``.
        target_features: Feature matrix for target-domain points,
            shape (n_target, n_features). Obtain via ``fcst.preprocess(target_df)``.
        estimator: Sklearn estimator type. ``"logistic"`` (default) uses
            ``LogisticRegression``; ``"gradient_boosting"`` uses
            ``GradientBoostingClassifier``.

    Returns:
        np.ndarray of shape (n_source,) with likelihood-ratio weights >= 1e-10.
    """
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([source_features, target_features])
    y = np.array([0] * len(source_features) + [1] * len(target_features))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if estimator == "logistic":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000)
    elif estimator == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        clf = GradientBoostingClassifier()
    else:
        raise ValueError(
            f"estimator must be 'logistic' or 'gradient_boosting', got '{estimator}'"
        )

    clf.fit(X_scaled, y)
    proba = clf.predict_proba(X_scaled[: len(source_features)])
    weights = proba[:, 1] / np.clip(proba[:, 0], 1e-10, None)
    return weights


_INTERVAL_METHODS = {
    "conformal_distribution": _add_conformal_distribution_intervals,
    "conformal_error": _add_conformal_error_intervals,
    "weighted_conformal_distribution": _add_weighted_conformal_distribution_intervals,
    "weighted_conformal_error": _add_weighted_conformal_error_intervals,
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
    feature_cols: Optional[List[str]] = None,
) -> DFType:
    """Compute absolute-error conformity scores over CV folds.

    Args:
        cv_results: Cross-validation output with actual and predicted columns.
        model_names: Names of model columns to convert to absolute errors.
        target_col: Name of the target column (dropped from output).
        feature_cols: Optional list of extra feature columns to keep alongside
            the error columns. Used by the weighted conformal DRE logic.
    """
    for model in model_names:
        abs_err = abs(cv_results[model] - cv_results[target_col])
        cv_results = ufp.assign_columns(cv_results, model, abs_err)
    result = ufp.drop_columns(cv_results, target_col)
    if feature_cols is not None:
        keep = [c for c in result.columns if c not in feature_cols] + feature_cols
        result = result[keep]
    return result


def _recalibrate_transfer(
    new_df: DFType,
    prediction_intervals: PredictionIntervals,
    cv_fn: Callable,
    model_names: List[str],
    target_col: str,
    id_col: str = "unique_id",
    time_col: str = "ds",
    preprocess_fn: Optional[Callable] = None,
    dre_estimator: str = "logistic",
    source_cs_df: Optional[DFType] = None,
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


def _weighted_conformal_transfer(
    new_df: DFType,
    prediction_intervals: PredictionIntervals,
    cv_fn: Callable,
    model_names: List[str],
    target_col: str,
    id_col: str = "unique_id",
    time_col: str = "ds",
    preprocess_fn: Optional[Callable] = None,
    dre_estimator: str = "logistic",
    source_cs_df: Optional[DFType] = None,
) -> DFType:
    """Compute DRE weights for source conformity scores under covariate shift.

    Unlike ``_recalibrate_transfer``, this method does NOT replace the source
    conformity scores. Instead it trains a logistic classifier to distinguish
    source from target features, computes likelihood-ratio weights for each
    source calibration point, and attaches them to
    ``prediction_intervals._cs_weights``. The original ``source_cs_df`` is
    returned unchanged so the caller can continue using source residuals with
    weighted quantiles.

    Requires ``preprocess_fn`` (``MLForecast.preprocess``) and
    ``source_cs_df`` (the existing ``_cs_df``) to be provided.
    """
    if preprocess_fn is None or source_cs_df is None:
        raise ValueError(
            "transfer_conformal_method='weighted_conformal' requires the model "
            "to have been fit with a weighted_conformal method so that source "
            "features are stored, and preprocess_fn must be supplied."
        )

    non_feature_cols = set(list(model_names) + [id_col, time_col, "cutoff"])
    feature_cols = [c for c in source_cs_df.columns if c not in non_feature_cols]

    if not feature_cols:
        raise ValueError(
            "No feature columns found in source conformity scores. "
            "Refit the model with PredictionIntervals(method='weighted_conformal_error') "
            "or 'weighted_conformal_distribution' so that source features are stored."
        )

    src_np = np.column_stack(
        [source_cs_df[c].to_numpy() for c in feature_cols]
    ).astype(float)

    tgt_preprocessed = preprocess_fn(new_df, validate_data=False)
    tgt_feature_cols = [c for c in feature_cols if c in tgt_preprocessed.columns]
    tgt_np = np.column_stack(
        [tgt_preprocessed[c].to_numpy() for c in tgt_feature_cols]
    ).astype(float)

    weights = estimate_density_ratio(src_np, tgt_np, estimator=dre_estimator)
    prediction_intervals._cs_weights = weights
    return source_cs_df


_TRANSFER_METHODS = {
    "recalibrate": _recalibrate_transfer,
    "weighted_conformal": _weighted_conformal_transfer,
}


def get_transfer_conformal_method(method: str) -> Callable:
    if method not in _TRANSFER_METHODS:
        raise ValueError(
            f"transfer conformal method {method} not supported "
            f"please choose one of {', '.join(_TRANSFER_METHODS.keys())}"
        )
    return _TRANSFER_METHODS[method]
