"""Backend-preserving feature encoders used between feature generation and models."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.base import clone
from utilsforecast.compat import pl, pl_DataFrame

__all__ = [
    "PolarsTargetEncoder",
    "PolarsCountEncoder",
    "PolarsOneHotEncoder",
    "PolarsOrdinalEncoder",
]


def _clone_encoder(encoder: Any) -> Any:
    try:
        return clone(encoder)
    except (TypeError, AttributeError):
        return copy.deepcopy(encoder)


def _accepts_context(method: Any) -> bool:
    parameters = inspect.signature(method).parameters.values()
    return any(
        parameter.name == "context" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _fit_transform(encoder: Any, X: Any, y: np.ndarray, context: Optional[Any]) -> Any:
    method = encoder.fit_transform
    if context is not None and _accepts_context(method):
        return method(X, y, context=context)
    return method(X, y)


class _EncodedModel:
    """Fitted model plus the fitted encoders required at prediction time."""

    def __init__(self, model: Any, encoders: Iterable[Any]):
        self.model = model
        self.encoders = list(encoders)

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        *,
        encoder_context: Optional[Any] = None,
        **fit_kwargs: Any,
    ) -> "_EncodedModel":
        for encoder in self.encoders:
            X = _fit_transform(encoder, X, y, encoder_context)
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
        for encoder in self.encoders:
            X = encoder.transform(X)
        return self.model.predict(X, **predict_kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


class _PolarsEncoder:
    """Shared validation and output handling for Polars feature encoders."""

    def __init__(self, columns: Iterable[str], drop_original: bool = False):
        self.columns = list(columns)
        self.drop_original = drop_original

    def _validate_X(self, X: pl_DataFrame) -> None:
        if pl is None or not isinstance(X, pl_DataFrame):
            raise TypeError(f"{type(self).__name__} requires a polars DataFrame.")

    def _drop_sources(self, X: pl_DataFrame) -> pl_DataFrame:
        return X.drop(self.columns) if self.drop_original else X


class PolarsOrdinalEncoder(_PolarsEncoder):
    """Ordinal encoder for Polars dataframes; unseen categories map to ``-1``."""

    def fit_transform(self, X: pl_DataFrame, _y: np.ndarray) -> pl_DataFrame:
        self._validate_X(X)
        self.mappings_ = {}
        for column in self.columns:
            encoded_name = f"{column}__ordinal"
            values = X.get_column(column).unique(maintain_order=True)
            self.mappings_[column] = pl.DataFrame(
                {column: values, encoded_name: np.arange(len(values), dtype=np.int32)}
            )
        return self.transform(X)

    def transform(self, X: pl_DataFrame) -> pl_DataFrame:
        self._validate_X(X)
        out = X
        for column, mapping in self.mappings_.items():
            encoded_name = mapping.columns[-1]
            out = out.join(mapping, on=column, how="left").with_columns(
                pl.col(encoded_name).fill_null(-1).cast(pl.Int32)
            )
        return self._drop_sources(out)


class PolarsCountEncoder(_PolarsEncoder):
    """Count encoder for Polars dataframes; unseen categories map to ``0``.

    Set ``normalize=True`` to encode category frequencies instead of raw counts.
    """

    def __init__(
        self,
        columns: Iterable[str],
        normalize: bool = False,
        drop_original: bool = False,
    ):
        super().__init__(columns=columns, drop_original=drop_original)
        self.normalize = normalize

    def fit_transform(self, X: pl_DataFrame, _y: np.ndarray) -> pl_DataFrame:
        self._validate_X(X)
        self.mappings_ = {}
        for column in self.columns:
            encoded_name = (
                f"{column}__frequency" if self.normalize else f"{column}__count"
            )
            expression = pl.len() / len(X) if self.normalize else pl.len()
            dtype = pl.Float32 if self.normalize else pl.UInt32
            self.mappings_[column] = X.group_by(column).agg(
                expression.cast(dtype).alias(encoded_name)
            )
        return self.transform(X)

    def transform(self, X: pl_DataFrame) -> pl_DataFrame:
        self._validate_X(X)
        out = X
        for column, mapping in self.mappings_.items():
            encoded_name = mapping.columns[-1]
            dtype = pl.Float32 if self.normalize else pl.UInt32
            out = out.join(mapping, on=column, how="left").with_columns(
                pl.col(encoded_name).fill_null(0).cast(dtype)
            )
        return self._drop_sources(out)


class PolarsOneHotEncoder(_PolarsEncoder):
    """Dense one-hot encoder for Polars dataframes.

    Use only for low-cardinality columns. It creates one UInt8 feature per seen
    category; unseen categories produce zeros in every feature for that column.
    """

    def fit_transform(self, X: pl_DataFrame, _y: np.ndarray) -> pl_DataFrame:
        self._validate_X(X)
        self.categories_ = {
            column: X.get_column(column).unique(maintain_order=True).to_list()
            for column in self.columns
        }
        return self.transform(X)

    def transform(self, X: pl_DataFrame) -> pl_DataFrame:
        self._validate_X(X)
        expressions = []
        for column, categories in self.categories_.items():
            for index, category in enumerate(categories):
                expressions.append(
                    (pl.col(column) == category)
                    .fill_null(False)
                    .cast(pl.UInt8)
                    .alias(f"{column}__onehot_{index}")
                )
        out = X.with_columns(expressions)
        return self._drop_sources(out)


class PolarsTargetEncoder:
    """Causal, smoothed target encoding for Polars dataframes.

    Training rows at timestamp ``t`` are encoded only with observations from
    timestamps strictly before ``t``. Future rows use statistics accumulated on
    the complete training data. The encoder adds ``{column}__mean`` columns and
    leaves source columns unchanged by default.
    """

    def __init__(
        self,
        columns: Iterable[str],
        smoothing: float = 20.0,
        drop_original: bool = False,
        prior: float = 0.0,
    ):
        self.columns = list(columns)
        self.smoothing = smoothing
        self.drop_original = drop_original
        self.prior = prior

    def fit_transform(
        self, X: pl_DataFrame, y: np.ndarray, *, context: Any
    ) -> pl_DataFrame:
        if pl is None or not isinstance(X, pl_DataFrame):
            raise TypeError("PolarsTargetEncoder requires a polars DataFrame.")
        if context is None or "times" not in context:
            raise ValueError("PolarsTargetEncoder requires timestamp context.")
        if len(X) != len(y) or len(X) != len(context["times"]):
            raise ValueError("X, y and timestamp context must have the same length.")

        frame = X.with_row_index("__encoder_row").with_columns(
            pl.Series("__encoder_time", context["times"]),
            pl.Series("__encoder_target", y),
        )
        global_by_time = (
            frame.group_by("__encoder_time")
            .agg(
                pl.col("__encoder_target").sum().alias("__sum"),
                pl.len().alias("__count"),
            )
            .sort("__encoder_time")
            .with_columns(
                pl.col("__sum").cum_sum().shift(1).alias("__prior_sum"),
                pl.col("__count").cum_sum().shift(1).alias("__prior_count"),
            )
            .select("__encoder_time", "__prior_sum", "__prior_count")
        )
        self.global_mean_ = float(np.mean(y))
        self.mappings_ = {}
        encoded = frame
        for column in self.columns:
            stats = (
                frame.group_by([column, "__encoder_time"])
                .agg(
                    pl.col("__encoder_target").sum().alias("__sum"),
                    pl.len().alias("__count"),
                )
                .sort([column, "__encoder_time"])
                .with_columns(
                    pl.col("__sum")
                    .cum_sum()
                    .shift(1)
                    .over(column)
                    .alias("__prior_sum"),
                    pl.col("__count")
                    .cum_sum()
                    .shift(1)
                    .over(column)
                    .alias("__prior_count"),
                )
                .join(
                    global_by_time, on="__encoder_time", how="left", suffix="__global"
                )
                .with_columns(
                    (
                        (
                            pl.col("__prior_sum").fill_null(0.0)
                            + self.smoothing
                            * (
                                pl.col("__prior_sum__global")
                                / pl.col("__prior_count__global")
                            )
                            .fill_nan(None)
                            .fill_null(self.prior)
                        )
                        / (pl.col("__prior_count").fill_null(0) + self.smoothing)
                    )
                    .fill_nan(None)
                    .fill_null(self.prior)
                    .cast(pl.Float32)
                    .alias(f"{column}__mean")
                )
                .select(column, "__encoder_time", f"{column}__mean")
            )
            encoded = encoded.join(stats, on=[column, "__encoder_time"], how="left")
            mapping = (
                frame.group_by(column)
                .agg(
                    pl.col("__encoder_target").sum().alias("__sum"),
                    pl.len().alias("__count"),
                )
                .with_columns(
                    (
                        (pl.col("__sum") + self.smoothing * self.global_mean_)
                        / (pl.col("__count") + self.smoothing)
                    )
                    .cast(pl.Float32)
                    .alias(f"{column}__mean")
                )
                .select(column, f"{column}__mean")
            )
            self.mappings_[column] = mapping
        out = encoded.sort("__encoder_row").drop(
            "__encoder_row", "__encoder_time", "__encoder_target"
        )
        if self.drop_original:
            out = out.drop(self.columns)
        return out

    def transform(self, X: pl_DataFrame) -> pl_DataFrame:
        if pl is None or not isinstance(X, pl_DataFrame):
            raise TypeError("PolarsTargetEncoder requires a polars DataFrame.")
        out = X.with_row_index("__encoder_row")
        for column, mapping in self.mappings_.items():
            out = out.join(mapping, on=column, how="left")
            out = out.with_columns(
                pl.col(f"{column}__mean").fill_null(self.global_mean_)
            )
        out = out.sort("__encoder_row").drop("__encoder_row")
        if self.drop_original:
            out = out.drop(self.columns)
        return out
