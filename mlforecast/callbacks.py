__all__ = ["Callback", "Profiler", "SaveFeatures"]


import functools
import inspect
from dataclasses import dataclass, field
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
)

import pandas as pd
from utilsforecast.compat import DataFrame
from utilsforecast.processing import (
    assign_columns,
    drop_index_if_pandas,
    vertical_concat,
)

if TYPE_CHECKING:
    from mlforecast.forecast import MLForecast

P = ParamSpec("P")
R = TypeVar("R")


class Callback:
    """Observe the stages of an :class:`mlforecast.MLForecast` object.

    Subclass this, override the hooks you need and pass instances through
    ``MLForecast(callbacks=[...])``. Stages nest: a ``fit`` with prediction
    intervals runs a ``cross_validation``, which runs a ``fit`` and a ``predict``
    per window. The hooks are called for every level, so a callback can build a
    tree of what happened, react only to top-level calls or filter by stage.

    Stages and the keyword arguments they carry:

    - ``preprocess``, ``fit``, ``fit_models``, ``predict``, ``cross_validation``: the bound arguments of the method call.
    - ``fit_model``: ``name``, ``model``, ``X``, ``y``, ``fit_kwargs`` and ``h`` (0-indexed horizon for direct models, None otherwise). Emitted once per estimator ``fit`` call.
    - ``cv_window``: ``i_window`` and ``cutoffs``.
    """

    def on_start(self, stage: str, fcst: "MLForecast", **kwargs: Any) -> None:
        """Called right before ``stage`` runs."""

    def on_end(
        self, stage: str, fcst: "MLForecast", result: Any, **kwargs: Any
    ) -> None:
        """Called right after ``stage`` returns ``result``."""

    def on_error(
        self, stage: str, fcst: "MLForecast", error: BaseException, **kwargs: Any
    ) -> None:
        """Called when ``stage`` raises ``error``, right before it propagates."""


def _emit(obj: Any, hook: str, stage: str, **kwargs: Any) -> None:
    for callback in obj.callbacks:
        getattr(callback, hook)(stage, obj, **kwargs)


def _stage(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Notify ``self.callbacks`` when the decorated method starts, ends or fails."""

    def decorator(method: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self: Any = args[0]
            if not self.callbacks:
                return method(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            info = dict(bound.arguments)
            del info["self"]
            _emit(self, "on_start", name, **info)
            try:
                result = method(*args, **kwargs)
            except BaseException as error:
                _emit(self, "on_error", name, error=error, **info)
                raise
            _emit(self, "on_end", name, result=result, **info)
            return result

        return wrapper

    return decorator


@dataclass
class Span:
    """Cost of one stage. ``children`` holds the stages that ran inside it."""

    stage: str
    seconds: float
    details: Dict[str, Any]
    rss_delta_bytes: Optional[int] = None
    children: List["Span"] = field(default_factory=list)


def _span_details(stage: str, kwargs: Dict[str, Any], result: Any) -> Dict[str, Any]:
    if stage == "preprocess":
        features = result[0] if isinstance(result, tuple) else result
        return {"input_rows": kwargs["df"].shape[0], "output_rows": features.shape[0]}
    if stage == "fit":
        return {"input_rows": kwargs["df"].shape[0]}
    if stage == "fit_model":
        X = kwargs["X"]
        return {
            "model": kwargs["name"],
            "h": kwargs["h"],
            "n_rows": X.shape[0],
            "n_features": X.shape[1],
        }
    if stage == "predict":
        return {"h": kwargs["h"]}
    if stage == "cross_validation":
        return {"n_windows": kwargs["n_windows"], "h": kwargs["h"]}
    if stage == "cv_window":
        return {"i_window": kwargs["i_window"]}
    return {}


class Profiler(Callback):
    """Record how long each stage takes and, optionally, its memory delta.

    Args:
        memory (bool): Record the change in the process' resident memory across each stage. Requires psutil. Defaults to False.

    Top-level calls end up in ``spans`` with everything that ran inside them as
    ``children``; ``to_df`` flattens them. For example, the time spent fitting
    each model across a whole cross validation is
    ``profiler.to_df().query("stage == 'fit_model'").groupby("model")["seconds"].sum()``.
    """

    def __init__(self, memory: bool = False):
        self.memory = memory
        self.spans: List[Span] = []
        self._stack: List[Tuple[str, float, Optional[int], List[Span]]] = []

    def _rss(self) -> Optional[int]:
        if not self.memory:
            return None
        import psutil

        return psutil.Process().memory_info().rss

    def on_start(self, stage: str, fcst: "MLForecast", **kwargs: Any) -> None:  # noqa: ARG002
        self._stack.append((stage, perf_counter(), self._rss(), []))

    def on_end(
        self,
        stage: str,
        fcst: "MLForecast",  # noqa: ARG002
        result: Any,
        **kwargs: Any,
    ) -> None:
        started, start, rss_start, children = self._stack.pop()
        assert started == stage
        seconds = perf_counter() - start
        rss_delta = None if rss_start is None else self._rss() - rss_start  # type: ignore[operator]
        span = Span(
            stage=stage,
            seconds=seconds,
            details=_span_details(stage, kwargs, result),
            rss_delta_bytes=rss_delta,
            children=children,
        )
        parent = self._stack[-1][3] if self._stack else self.spans
        parent.append(span)

    def on_error(
        self,
        stage: str,
        fcst: "MLForecast",  # noqa: ARG002
        error: BaseException,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        # stages without an error hook (cv_window) can sit above the failing one
        while self._stack.pop()[0] != stage:
            pass

    def reset(self) -> None:
        self.spans = []
        self._stack = []

    def to_df(self) -> pd.DataFrame:
        """Flatten the recorded spans into one row per stage, in execution order."""
        rows: List[Dict[str, Any]] = []

        def visit(span: Span, depth: int) -> None:
            rows.append(
                {
                    "stage": span.stage,
                    "depth": depth,
                    "seconds": span.seconds,
                    "rss_delta_bytes": span.rss_delta_bytes,
                    **span.details,
                }
            )
            for child in span.children:
                visit(child, depth + 1)

        for span in self.spans:
            visit(span, 0)
        return pd.DataFrame(rows)


class SaveFeatures:
    """Saves the features in every timestamp."""

    def __init__(self):
        self._inputs = []

    def __call__(self, new_x):
        self._inputs.append(new_x)
        return new_x

    def get_features(self, with_step: bool = False) -> DataFrame:
        """Retrieves the input features for every timestep

        Args:
            with_step (bool): Add a column indicating the step. Defaults to False.

        Returns:
            (pandas or polars DataFrame): DataFrame with input features
        """
        if not self._inputs:
            raise ValueError(
                "Inputs list is empty. "
                "Call `predict` using this callback as before_predict_callback"
            )
        if with_step:
            dfs = [assign_columns(df, "step", i) for i, df in enumerate(self._inputs)]
        else:
            dfs = self._inputs
        res = vertical_concat(dfs, match_categories=False)
        res = drop_index_if_pandas(res)
        return res
