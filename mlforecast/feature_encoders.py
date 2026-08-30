"""Backend-preserving feature encoders used between feature generation and models."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.base import clone
from utilsforecast.compat import pl, pl_DataFrame

from .compat import CatBoostRegressor

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
    return "context" in inspect.signature(method).parameters


def _fit_transform(encoder: Any, X: Any, y: np.ndarray, context: Optional[Any]) -> Any:
    method = encoder.fit_transform
    if context is not None and _accepts_context(method):
        return method(X, y, context=context)
    return method(X, y)


class _EncodedModel:
    """Fitted model plus the fitted encoders required at prediction time."""

    def __init__(self, model: Any, encoders: Iterable[Any], as_numpy: bool = False):
        self.model = model
        self.encoders = list(encoders)
        self.as_numpy = as_numpy

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        *,
        encoder_context: Optional[Any] = None,
        store_fitted_predictions: bool = False,
        **fit_kwargs: Any,
    ) -> "_EncodedModel":
        for encoder in self.encoders:
            X = _fit_transform(encoder, X, y, encoder_context)
        self.model.fit(self._prepare_for_model(X), y, **fit_kwargs)
        if store_fitted_predictions:
            self.fitted_predictions_ = self.model.predict(self._prepare_for_model(X))
        return self

    def predict(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
        for encoder in self.encoders:
            X = encoder.transform(X)
        return self.model.predict(self._prepare_for_model(X), **predict_kwargs)

    def predict_fitted(self, order: Optional[np.ndarray] = None) -> np.ndarray:
        if order is None:
            return self.fitted_predictions_
        return self.fitted_predictions_[order]

    def _prepare_for_model(self, X: Any) -> Any:
        if isinstance(self.model, CatBoostRegressor) and isinstance(X, pl_DataFrame):
            X = X.to_pandas()
        if self.as_numpy and hasattr(X, "to_numpy"):
            X = X.to_numpy()
        return X

    def __getattr__(self, name: str) -> Any:
        model = self.__dict__.get("model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    def __getstate__(self) -> dict[str, Any]:
        """Keep wrapper attributes intact across cloudpickle versions."""
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


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
            values = X.get_column(column).drop_nulls().unique(maintain_order=True)
            self.mappings_[column] = pl.DataFrame(
                {column: values, encoded_name: np.arange(len(values), dtype=np.int32)}
            )
        return self.transform(X)

    def transform(self, X: pl_DataFrame) -> pl_DataFrame:
        self._validate_X(X)
        out = X
        for column, mapping in self.mappings_.items():
            encoded_name = mapping.columns[-1]
            out = out.join(
                mapping, on=column, how="left", maintain_order="left"
            ).with_columns(pl.col(encoded_name).fill_null(-1).cast(pl.Int32))
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
            out = out.join(
                mapping, on=column, how="left", maintain_order="left"
            ).with_columns(pl.col(encoded_name).fill_null(0).cast(dtype))
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
    """Smoothed target encoding for Polars dataframes.

    Training and future rows are encoded with category statistics fit on the
    complete training data. The encoder adds ``{column}__mean`` columns and
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

    def _fit_transform_same_times(self, frame: pl_DataFrame) -> pl_DataFrame:
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
                    global_by_time,
                    on="__encoder_time",
                    how="left",
                    suffix="__global",
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
            encoded = encoded.join(
                stats,
                on=[column, "__encoder_time"],
                how="left",
                nulls_equal=True,
            )
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
            "__encoder_row",
            "__encoder_time",
            "__encoder_target_time",
            "__encoder_target",
        )
        return out.drop(self.columns) if self.drop_original else out

    def fit_transform(
        self, X: pl_DataFrame, y: np.ndarray, *, context: Optional[Any] = None
    ) -> pl_DataFrame:
        if pl is None or not isinstance(X, pl_DataFrame):
            raise TypeError("PolarsTargetEncoder requires a polars DataFrame.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        self.global_mean_ = float(np.mean(y))
        frame = X.with_columns(pl.Series("__encoder_target", y))
        self.mappings_ = {}
        for column in self.columns:
            self.mappings_[column] = (
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
        return self.transform(X)

        # Retained below temporarily while the old ordered implementation is
        # removed in the same change.
        target_times = context.get("target_times", context["times"])
        if len(target_times) != len(X):
            raise ValueError(
                "X, y and target timestamp context must have the same length."
            )
        frame = X.with_row_index("__encoder_row").with_columns(
            pl.Series("__encoder_time", context["times"]),
            pl.Series("__encoder_target_time", target_times),
            pl.Series("__encoder_target", y),
        )
        self.global_mean_ = float(np.mean(y))
        self.mappings_ = {}
        if np.array_equal(target_times, context["times"]):
            return self._fit_transform_same_times(frame)
        global_by_target_time = (
            frame.group_by("__encoder_target_time")
            .agg(
                pl.col("__encoder_target").sum().alias("__sum"),
                pl.len().alias("__count"),
            )
            .sort("__encoder_target_time")
            .with_columns(
                pl.col("__sum").cum_sum().alias("__prior_sum"),
                pl.col("__count").cum_sum().alias("__prior_count"),
            )
            .select("__encoder_target_time", "__prior_sum", "__prior_count")
        )
        global_priors = (
            frame.select("__encoder_row", "__encoder_time")
            .sort("__encoder_time")
            .join_asof(
                global_by_target_time,
                left_on="__encoder_time",
                right_on="__encoder_target_time",
                strategy="backward",
                allow_exact_matches=False,
            )
            .select(
                "__encoder_row",
                pl.col("__prior_sum").alias("__prior_sum__global"),
                pl.col("__prior_count").alias("__prior_count__global"),
            )
        )
        encoded = frame.join(global_priors, on="__encoder_row", how="left")
        for column_index, column in enumerate(self.columns):
            if frame[column].null_count():
                category_key = f"__encoder_category_{column_index}"
                column_frame = encoded.with_columns(
                    pl.when(pl.col(column).is_null())
                    .then(pl.lit("1:"))
                    .otherwise(pl.lit("0:") + pl.col(column).cast(pl.String))
                    .alias(category_key)
                )
            else:
                category_key = column
                column_frame = encoded
            label_stats = (
                column_frame.group_by([category_key, "__encoder_target_time"])
                .agg(
                    pl.col("__encoder_target").sum().alias("__sum"),
                    pl.len().alias("__count"),
                )
                .sort([category_key, "__encoder_target_time"])
                .with_columns(
                    pl.col("__sum").cum_sum().over(category_key).alias("__prior_sum"),
                    pl.col("__count")
                    .cum_sum()
                    .over(category_key)
                    .alias("__prior_count"),
                )
            )
            category_priors = (
                column_frame.select(
                    "__encoder_row",
                    category_key,
                    "__encoder_time",
                    "__prior_sum__global",
                    "__prior_count__global",
                )
                .sort([category_key, "__encoder_time"])
                .join_asof(
                    label_stats,
                    left_on="__encoder_time",
                    right_on="__encoder_target_time",
                    by=category_key,
                    strategy="backward",
                    allow_exact_matches=False,
                    check_sortedness=False,
                )
            )
            stats = (
                category_priors.with_columns(
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
                .select("__encoder_row", f"{column}__mean")
                .sort("__encoder_row")
            )
            encoded = encoded.with_columns(stats[f"{column}__mean"])
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
            "__encoder_row",
            "__encoder_time",
            "__encoder_target_time",
            "__encoder_target",
        )
        if self.drop_original:
            out = out.drop(self.columns)
        return out

    def transform(self, X: pl_DataFrame) -> pl_DataFrame:
        if pl is None or not isinstance(X, pl_DataFrame):
            raise TypeError("PolarsTargetEncoder requires a polars DataFrame.")
        out = X.with_row_index("__encoder_row")
        for column, mapping in self.mappings_.items():
            out = out.join(mapping, on=column, how="left", nulls_equal=True)
            out = out.with_columns(
                pl.col(f"{column}__mean").fill_null(self.global_mean_)
            )
        out = out.sort("__encoder_row").drop("__encoder_row")
        if self.drop_original:
            out = out.drop(self.columns)
        return out
