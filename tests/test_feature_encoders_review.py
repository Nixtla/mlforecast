"""Regression tests for defects found while reviewing the feature-encoder work.

Every test here asserts the behavior the encoder API promises, and derives its
expectation from an independent source -- the same data fed in a different row
order, an earlier cross-validation fold, or hand arithmetic -- rather than from
a constant copied out of the current implementation. They therefore stay valid
whichever way each defect is fixed.
"""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

from mlforecast import MLForecast
from mlforecast.feature_encoders import (
    _EncodedModel,
    PolarsOrdinalEncoder,
    PolarsTargetEncoder,
)


class _EchoEncodedColumn(BaseEstimator, RegressorMixin):
    """Predicts the encoded column verbatim, exposing the encodings a model saw."""

    def fit(self, _X, _y):
        return self

    def predict(self, X):
        return np.asarray(X["category__mean"])


def _causal_target_encoder():
    # smoothing=0, prior=0 makes each encoding the plain mean of prior targets.
    return PolarsTargetEncoder(
        ["category"], smoothing=0.0, prior=0.0, drop_original=True
    )


def _two_series(order="sorted"):
    rows = pl.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 5),
            "ds": list(range(5)) * 2,
            "y": np.arange(10, dtype=float),
            "category": np.repeat(["first", "second"], 5),
        }
    ).with_columns(pl.col("category").cast(pl.Categorical))
    if order == "sorted":
        return rows
    # Interleave the two ids: b, a, b, a, ... Same rows, different arrival order.
    return (
        rows.with_row_index("__i")
        .sort((pl.col("__i") % 5) * 2 + (pl.col("unique_id") == "a").cast(pl.Int64))
        .drop("__i")
    )


def test_stock_sklearn_transformer_is_accepted():
    """A plain sklearn transformer must work as a feature encoder.

    `_accepts_context` treats a `**kwargs` in the signature as "accepts
    context", and `TransformerMixin.fit_transform` is `(X, y=None,
    **fit_params)`, so every stock sklearn transformer is called with a
    `context=` keyword it forwards into `fit()`. This is the exact interface
    the `feature_encoders` docstring advertises, so `OneHotEncoder`,
    `ColumnTransformer` and `category_encoders.*` are all affected.
    """
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 5),
            "ds": list(range(5)) * 2,
            "y": np.arange(10, dtype=float),
        }
    )
    fcst = MLForecast(
        models=DummyRegressor(),
        freq=1,
        lags=[1],
        feature_encoders=[StandardScaler()],
    )

    fcst.fit(df, static_features=[])

    assert fcst.predict(1).shape == (2, 3)


def test_fitted_values_follow_input_row_order():
    """Fitted values must not depend on the row order of the input frame.

    `_compute_fitted_values` reorders `base`/`X`/`y` by `sort_idxs`, but
    `fitted_X_` was stored during `fit_models` in the original input order, so
    `predict_fitted()` output is attributed to the wrong rows. Nothing raises;
    the numbers are just wrong.
    """
    kwargs = dict(fitted=True, static_features=["category"])

    sorted_fcst = MLForecast(
        models=_EchoEncodedColumn(),
        freq=1,
        lags=[1],
        feature_encoders=[_causal_target_encoder()],
    )
    sorted_fcst.fit(_two_series("sorted"), **kwargs)
    expected = sorted_fcst.forecast_fitted_values().sort(["unique_id", "ds"])

    shuffled_fcst = MLForecast(
        models=_EchoEncodedColumn(),
        freq=1,
        lags=[1],
        feature_encoders=[_causal_target_encoder()],
    )
    shuffled_fcst.fit(_two_series("interleaved"), **kwargs)
    actual = shuffled_fcst.forecast_fitted_values().sort(["unique_id", "ds"])

    np.testing.assert_allclose(
        actual["_EchoEncodedColumn"], expected["_EchoEncodedColumn"]
    )


def test_direct_fitted_values_follow_input_row_order():
    """Same invariant for the direct (`max_horizon`) path.

    `preds[valid] = model.predict_fitted()` assumes `fitted_X_` is aligned with
    the possibly re-sorted `X_h`. `valid.sum()` is order-invariant, so lengths
    still match and the misalignment is silent.
    """
    kwargs = dict(fitted=True, max_horizon=2, static_features=["category"])

    sorted_fcst = MLForecast(
        models=_EchoEncodedColumn(),
        freq=1,
        lags=[1],
        feature_encoders=[_causal_target_encoder()],
    )
    sorted_fcst.fit(_two_series("sorted"), **kwargs)
    expected = sorted_fcst.forecast_fitted_values().sort(["unique_id", "ds", "h"])

    shuffled_fcst = MLForecast(
        models=_EchoEncodedColumn(),
        freq=1,
        lags=[1],
        feature_encoders=[_causal_target_encoder()],
    )
    shuffled_fcst.fit(_two_series("interleaved"), **kwargs)
    actual = shuffled_fcst.forecast_fitted_values().sort(["unique_id", "ds", "h"])

    np.testing.assert_allclose(
        actual["_EchoEncodedColumn"], expected["_EchoEncodedColumn"]
    )


def test_cross_validation_fitted_values_are_causal_in_every_fold():
    """Every CV fold's fitted values must be causal, not just the first.

    `fitted_X_` is deleted after its first use, so from the second window on the
    `else: model.predict(X)` fallback re-encodes with full-training statistics.
    Those in-sample values have seen their own target. Both folds encode a given
    timestamp from strictly prior targets only, so on the timestamps the two
    folds share the encodings must agree.
    """
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 8,
            "ds": range(8),
            "y": np.arange(8, dtype=float),
            "category": ["x"] * 8,
        }
    ).with_columns(pl.col("category").cast(pl.Categorical))
    fcst = MLForecast(
        models=_EchoEncodedColumn(),
        freq=1,
        lags=[1],
        feature_encoders=[_causal_target_encoder()],
    )

    fcst.cross_validation(
        df,
        n_windows=2,
        h=1,
        step_size=1,
        fitted=True,
        refit=False,
        static_features=["category"],
    )
    fitted = fcst.cross_validation_fitted_values()

    first, second = (
        fitted.filter(pl.col("fold") == fold).sort("ds") for fold in (0, 1)
    )
    shared = first["ds"]
    np.testing.assert_allclose(
        second.filter(pl.col("ds").is_in(shared))["_EchoEncodedColumn"],
        first["_EchoEncodedColumn"],
    )


def test_target_encoder_null_category_is_encoded_at_fit_time():
    """Null categories must get a real encoding during `fit_transform`.

    The fit-time join is `how="left"` with polars' default `nulls_equal=False`
    and, unlike `transform`, has no `fill_null` afterwards, so null categories
    come out null. Models that reject NaN then fail during `fit`, while tree
    models train on NaN and meet a finite `global_mean_` at predict time.
    """
    X = pl.DataFrame({"category": ["a", None, "a", None]})
    y = np.array([1.0, 2.0, 3.0, 4.0])
    context = {"times": np.array([1, 2, 3, 4])}
    encoder = PolarsTargetEncoder(["category"])

    encoded = encoder.fit_transform(X, y, context=context)["category__mean"]

    assert encoded.is_null().sum() == 0
    assert np.isfinite(encoded.to_numpy()).all()


def test_save_load_refit_keeps_encoders(tmp_path):
    """Refitting a loaded forecaster must keep encoding its features.

    `load` rebuilds `MLForecast(models=models, freq=ts.freq)` without
    `feature_encoders`, and `_EncodedModel.__getattr__` forwards
    `__sklearn_clone__` to the wrapped estimator, so `clone` silently unwraps
    it. The refit then trains on raw, unencoded features and raises nothing.
    """
    df = _two_series("sorted")
    fcst = MLForecast(
        models=DummyRegressor(),
        freq=1,
        lags=[1],
        feature_encoders=[_causal_target_encoder()],
    )
    fcst.fit(df, static_features=["category"])

    path = tmp_path / "model"
    fcst.save(path)
    loaded = MLForecast.load(path)
    loaded.fit(df, static_features=["category"])

    assert isinstance(loaded.models_["DummyRegressor"], _EncodedModel)


def test_ordinal_encoder_does_not_spend_a_code_on_null():
    """Ordinal codes must be contiguous over the categories actually seen.

    `unique(maintain_order=True)` includes null, so it consumes a code from
    `np.arange(len(values))` that the `nulls_equal=False` join can never match.
    Nulls fall through to the "unseen" sentinel anyway and the code space is
    left with a hole.
    """
    X = pl.DataFrame({"category": ["a", None, "b"]})
    encoder = PolarsOrdinalEncoder(["category"], drop_original=True)

    encoded = encoder.fit_transform(X, np.zeros(3))["category__ordinal"]

    seen = sorted(code for code in encoded.to_list() if code != -1)
    assert seen == list(range(len(seen)))
