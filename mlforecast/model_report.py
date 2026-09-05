"""Reporting utilities for MLForecast-owned feature preparation."""

from dataclasses import asdict, dataclass, fields, is_dataclass
from math import isfinite
from numbers import Real
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import narwhals as nw
import numpy as np
import pandas as pd
from utilsforecast.compat import pl_DataFrame


@dataclass
class MetricComparison:
    """Comparison of one numeric report field."""

    baseline: Optional[float]
    candidate: Optional[float]
    delta: Optional[float]
    delta_percent: Optional[float]
    change_kind: Literal["compared", "added", "removed"]
    exceeds_threshold: bool


@dataclass
class ReportComparison:
    """Structured numeric comparison of two reports."""

    metrics: Dict[str, MetricComparison]


class _ComparableReport:
    def diff(
        self,
        other: Any,
        *,
        relative_threshold: float = 0.05,
        absolute_threshold: float = 0.0,
    ) -> ReportComparison:
        """Compare this report to a baseline report of the same type."""
        if type(self) is not type(other):
            raise TypeError("Reports can only be compared with the same report type.")
        if relative_threshold < 0 or absolute_threshold < 0:
            raise ValueError("Comparison thresholds must be non-negative.")
        baseline = _numeric_fields(other)
        candidate = _numeric_fields(self)
        metrics = {}
        for name in sorted(baseline.keys() | candidate.keys()):
            baseline_value = baseline.get(name)
            candidate_value = candidate.get(name)
            if baseline_value is None:
                metrics[name] = MetricComparison(
                    baseline=None,
                    candidate=candidate_value,
                    delta=None,
                    delta_percent=None,
                    change_kind="added",
                    exceeds_threshold=True,
                )
                continue
            if candidate_value is None:
                metrics[name] = MetricComparison(
                    baseline=baseline_value,
                    candidate=None,
                    delta=None,
                    delta_percent=None,
                    change_kind="removed",
                    exceeds_threshold=True,
                )
                continue
            delta = candidate_value - baseline_value
            delta_percent = (
                None if baseline_value == 0 else delta / abs(baseline_value) * 100
            )
            metrics[name] = MetricComparison(
                baseline=baseline_value,
                candidate=candidate_value,
                delta=delta,
                delta_percent=delta_percent,
                change_kind="compared",
                exceeds_threshold=(
                    delta != 0
                    and abs(delta) >= absolute_threshold
                    and (
                        delta_percent is None
                        or abs(delta_percent) >= relative_threshold * 100
                    )
                ),
            )
        return ReportComparison(metrics=metrics)


def _numeric_fields(value: Any, prefix: str = "") -> Dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return {prefix: number} if isfinite(number) else {}
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            key = f"{prefix}.{name}" if prefix else str(name)
            result.update(_numeric_fields(item, key))
        return result
    if is_dataclass(value):
        result = {}
        for field in fields(value):
            if field.name in {"rss_start_bytes", "rss_end_bytes"}:
                continue
            key = f"{prefix}.{field.name}" if prefix else field.name
            result.update(_numeric_fields(getattr(value, field.name), key))
        if hasattr(value, "rss_delta_bytes"):
            key = f"{prefix}.rss_delta_bytes" if prefix else "rss_delta_bytes"
            result.update(_numeric_fields(value.rss_delta_bytes, key))
        return result
    return {}


@dataclass
class FeaturePreparationReport(_ComparableReport):
    """Observed input and output characteristics of a preprocessing call.

    The report describes only operations performed by MLForecast.
    """

    input_backend: str
    output_backend: str
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    input_dtypes: Dict[str, str]
    output_dtypes: Dict[str, str]
    operations: List[str]
    input_memory_bytes: Optional[int]
    output_memory_bytes: Optional[int]
    elapsed_seconds: float

    @property
    def input_memory_mb(self) -> Optional[float]:
        return _bytes_to_mb(self.input_memory_bytes)

    @property
    def output_memory_mb(self) -> Optional[float]:
        return _bytes_to_mb(self.output_memory_bytes)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the report."""
        result = asdict(self)
        result["input_memory_mb"] = self.input_memory_mb
        result["output_memory_mb"] = self.output_memory_mb
        return result

    def __repr__(self) -> str:
        operations = "\n".join(f"  - {operation}" for operation in self.operations)
        return (
            "FeaturePreparationReport(\n"
            f"  input: {self.input_backend} {self.input_shape}\n"
            f"  output: {self.output_backend} {self.output_shape}\n"
            f"  memory: {self.input_memory_mb!s} MiB -> {self.output_memory_mb!s} MiB\n"
            f"  elapsed: {self.elapsed_seconds:.3f} s\n"
            "  operations:\n"
            f"{operations}\n"
            ")"
        )


def _backend(data: Any) -> str:
    if isinstance(data, np.ndarray):
        return "numpy"
    if isinstance(data, pd.DataFrame):
        return "pandas"
    if isinstance(data, pl_DataFrame):
        return "polars"
    return type(data).__name__


def _dtypes(data: Any) -> Dict[str, str]:
    if isinstance(data, np.ndarray):
        return {f"feature_{i}": str(data.dtype) for i in range(data.shape[1])}
    if isinstance(data, pd.DataFrame):
        return {column: str(dtype) for column, dtype in data.dtypes.items()}
    if isinstance(data, pl_DataFrame):
        return {column: str(dtype) for column, dtype in data.schema.items()}
    return {}


def _memory_bytes(data: Any) -> int:
    if isinstance(data, np.ndarray):
        return int(data.nbytes)
    if isinstance(data, pd.DataFrame):
        return int(data.memory_usage(index=True, deep=True).sum())
    if isinstance(data, pl_DataFrame):
        return int(data.estimated_size())
    return 0


def get_process_rss_bytes() -> Optional[int]:
    """Get current process RSS when psutil is available, otherwise return None."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _bytes_to_mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / 2**20


@dataclass
class ModelFitReport(_ComparableReport):
    """Observed cost of calls to estimator ``fit`` methods."""

    elapsed_seconds: float
    fit_calls: int
    model_seconds: Dict[str, float]
    rss_start_bytes: Optional[int]
    rss_end_bytes: Optional[int]

    @property
    def rss_delta_bytes(self) -> Optional[int]:
        if self.rss_start_bytes is None or self.rss_end_bytes is None:
            return None
        return self.rss_end_bytes - self.rss_start_bytes

    @property
    def rss_delta_mb(self) -> Optional[float]:
        return _bytes_to_mb(self.rss_delta_bytes)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["rss_delta_bytes"] = self.rss_delta_bytes
        result["rss_delta_mb"] = self.rss_delta_mb
        return result


@dataclass
class FitReport(_ComparableReport):
    """Observed end-to-end cost of a successful ``MLForecast.fit`` call."""

    elapsed_seconds: float
    rss_start_bytes: Optional[int]
    rss_end_bytes: Optional[int]
    model_fit_report: Optional[ModelFitReport]
    calibration_model_fit_report: Optional[ModelFitReport]
    final_model_fit_report: Optional[ModelFitReport]

    @property
    def rss_delta_bytes(self) -> Optional[int]:
        if self.rss_start_bytes is None or self.rss_end_bytes is None:
            return None
        return self.rss_end_bytes - self.rss_start_bytes

    @property
    def rss_delta_mb(self) -> Optional[float]:
        return _bytes_to_mb(self.rss_delta_bytes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "rss_start_bytes": self.rss_start_bytes,
            "rss_end_bytes": self.rss_end_bytes,
            "rss_delta_bytes": self.rss_delta_bytes,
            "rss_delta_mb": self.rss_delta_mb,
            "model_fit_report": (
                None
                if self.model_fit_report is None
                else self.model_fit_report.to_dict()
            ),
            "calibration_model_fit_report": (
                None
                if self.calibration_model_fit_report is None
                else self.calibration_model_fit_report.to_dict()
            ),
            "final_model_fit_report": (
                None
                if self.final_model_fit_report is None
                else self.final_model_fit_report.to_dict()
            ),
        }


def aggregate_model_fit_reports(
    reports: List[ModelFitReport],
) -> Optional[ModelFitReport]:
    """Combine sequential model-fit reports into one aggregate report."""
    if not reports:
        return None
    model_seconds: Dict[str, float] = {}
    for report in reports:
        for name, seconds in report.model_seconds.items():
            model_seconds[name] = model_seconds.get(name, 0.0) + seconds
    return ModelFitReport(
        elapsed_seconds=sum(report.elapsed_seconds for report in reports),
        fit_calls=sum(report.fit_calls for report in reports),
        model_seconds=model_seconds,
        rss_start_bytes=reports[0].rss_start_bytes,
        rss_end_bytes=reports[-1].rss_end_bytes,
    )


@dataclass
class PredictReport(_ComparableReport):
    """Observed end-to-end cost of a successful ``MLForecast.predict`` call."""

    elapsed_seconds: float
    rss_start_bytes: Optional[int]
    rss_end_bytes: Optional[int]
    horizon: int

    @property
    def rss_delta_bytes(self) -> Optional[int]:
        if self.rss_start_bytes is None or self.rss_end_bytes is None:
            return None
        return self.rss_end_bytes - self.rss_start_bytes

    @property
    def rss_delta_mb(self) -> Optional[float]:
        return _bytes_to_mb(self.rss_delta_bytes)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["rss_delta_bytes"] = self.rss_delta_bytes
        result["rss_delta_mb"] = self.rss_delta_mb
        return result


def make_feature_preparation_report(
    input_df: Any,
    output: Any,
    *,
    as_numpy: bool,
    elapsed_seconds: float,
    detailed: bool,
) -> FeaturePreparationReport:
    """Describe the observed result of MLForecast feature preparation."""
    input_backend = _backend(input_df)
    output_backend = _backend(output)
    operations = [f"received {input_backend} input"]
    if output.shape[0] < input_df.shape[0]:
        operations.append(
            f"dropped {input_df.shape[0] - output.shape[0]} rows with unavailable features or targets"
        )
    if input_backend != output_backend:
        reason = "because as_numpy=True" if as_numpy else "during preprocessing"
        operations.append(f"converted {input_backend} -> {output_backend} {reason}")
    else:
        operations.append(f"kept {output_backend} representation")
    return FeaturePreparationReport(
        input_backend=input_backend,
        output_backend=output_backend,
        input_shape=tuple(input_df.shape),
        output_shape=tuple(output.shape),
        input_dtypes=_dtypes(input_df),
        output_dtypes=_dtypes(output),
        operations=operations,
        input_memory_bytes=_memory_bytes(input_df) if detailed else None,
        output_memory_bytes=_memory_bytes(output) if detailed else None,
        elapsed_seconds=elapsed_seconds,
    )


def summarize_model_metrics(
    df: Any,
    *,
    models: Sequence[str],
    metrics: Sequence[Callable],
    id_col: str,
    target_col: str,
) -> Dict[str, Dict[str, float]]:
    """Compute per-model mean loss values from utilsforecast-style metrics."""
    summary = {model: {} for model in models}
    for metric in metrics:
        name = metric_name(metric)
        evaluated = metric(
            df,
            models=list(models),
            id_col=id_col,
            target_col=target_col,
        )
        evaluated_nw = nw.from_native(evaluated, eager_only=True)
        for model in models:
            summary[model][name] = float(evaluated_nw[model].mean())
    return summary


def metric_name(metric: Callable) -> str:
    """Return the stable display name used for a metric callable."""
    return getattr(metric, "__name__", type(metric).__name__)
