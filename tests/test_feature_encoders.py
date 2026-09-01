from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlforecast import MLForecast
from mlforecast.feature_encoders import (
    _EncodedModel,
    PolarsTargetEncoder,
    PolarsCountEncoder,
    PolarsOneHotEncoder,
    PolarsOrdinalEncoder,
)
from mlforecast.auto import AutoMLForecast, AutoModel


class _EncodedColumnRegressor(BaseEstimator, RegressorMixin):
    def fit(self, _X, _y):
        return self

    def predict(self, X):
        return X["category__mean"].to_numpy()


class _EchoEncodedColumn(BaseEstimator, RegressorMixin):
    """Predicts the encoded column verbatim, exposing the encodings a model saw."""

    def fit(self, _X, _y):
        return self

    def predict(self, X):
        return np.asarray(X["category__mean"])


class _PolarsOnlyCatBoost(BaseEstimator, RegressorMixin):
    """Stand-in that proves conversion happens after Polars encoding."""

    inputs: list[type[Any]] = []

    def fit(self, X, _y):
        type(self).inputs.append(type(X))
        assert isinstance(X, pd.DataFrame)
        return self

    def predict(self, X):
        type(self).inputs.append(type(X))
        assert isinstance(X, pd.DataFrame)
        return np.zeros(len(X))


def _target_encoder():
    # smoothing=0 makes each encoding the plain full-training category mean.
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


def test_polars_ordinal_encoder_handles_unseen_categories():
    encoder = PolarsOrdinalEncoder(["category"], drop_original=True)
    train = encoder.fit_transform(
        pl.DataFrame({"category": ["b", "a", "b"]}), np.zeros(3)
    )
    future = encoder.transform(pl.DataFrame({"category": ["a", "missing"]}))

    assert train.columns == ["category__ordinal"]
    assert train["category__ordinal"].to_list() == [0, 1, 0]
    assert future["category__ordinal"].to_list() == [1, -1]


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


def test_polars_count_frequency_and_one_hot_encoders():
    X = pl.DataFrame({"category": ["a", "b", "a"]})
    count = PolarsCountEncoder(["category"], drop_original=True)
    frequency = PolarsCountEncoder(["category"], normalize=True, drop_original=True)
    one_hot = PolarsOneHotEncoder(["category"], drop_original=True)

    encoded_count = count.fit_transform(X, np.zeros(3))
    encoded_frequency = frequency.fit_transform(X, np.zeros(3))
    encoded_one_hot = one_hot.fit_transform(X, np.zeros(3))
    unknown_count = count.transform(pl.DataFrame({"category": ["missing"]}))
    unknown_frequency = frequency.transform(pl.DataFrame({"category": ["missing"]}))
    unknown_one_hot = one_hot.transform(pl.DataFrame({"category": ["missing"]}))

    np.testing.assert_allclose(
        encoded_frequency["category__frequency"], [2 / 3, 1 / 3, 2 / 3]
    )
    assert encoded_count["category__count"].to_list() == [2, 1, 2]
    assert encoded_one_hot.columns == ["category__onehot_0", "category__onehot_1"]
    assert unknown_count["category__count"].to_list() == [0]
    assert unknown_frequency["category__frequency"].to_list() == [0.0]
    assert unknown_one_hot.row(0) == (0, 0)


def test_ordinal_and_count_encoders_request_left_join_order(monkeypatch):
    join = pl.DataFrame.join
    join_orders = []

    def audited_join(*args, **kwargs):
        join_orders.append(kwargs.get("maintain_order"))
        return join(*args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "join", audited_join)
    X = pl.DataFrame({"category": ["a", "b", "a"]})
    PolarsOrdinalEncoder(["category"]).fit_transform(X, np.zeros(len(X)))
    PolarsCountEncoder(["category"]).fit_transform(X, np.zeros(len(X)))

    assert join_orders == ["left", "left"]


def test_polars_target_encoder_uses_full_training_category_means():
    X = pl.DataFrame({"category": ["a", "b", "a"], "lag1": [1.0, 2.0, 3.0]})
    encoder = PolarsTargetEncoder(["category"], smoothing=0.0, prior=0.0)

    encoded = encoder.fit_transform(X, np.array([10.0, 100.0, 30.0]))
    future = encoder.transform(pl.DataFrame({"category": ["a", "unknown"]}))

    np.testing.assert_allclose(encoded["category__mean"], [20.0, 100.0, 20.0])
    np.testing.assert_allclose(future["category__mean"], [20.0, 140 / 3])


def test_polars_target_encoder_defaults_to_dropping_source_columns():
    encoded = PolarsTargetEncoder(["category"], smoothing=0.0).fit_transform(
        pl.DataFrame({"category": ["a", "a"]}), np.array([1.0, 3.0])
    )

    assert encoded.columns == ["category__mean"]


def test_polars_target_encoder_uses_category_encoders_sigmoid_regularization():
    X = pl.DataFrame({"category": ["rare", "common", "common", "common"]})
    y = np.array([0.0, 8.0, 10.0, 12.0])
    encoder = PolarsTargetEncoder(
        ["category"], min_samples_leaf=2, smoothing=1.0, drop_original=False
    )

    encoded = encoder.fit_transform(X, y)["category__mean"]
    global_mean = y.mean()
    rare_weight = 1 / (1 + np.exp(1))
    common_weight = 1 / (1 + np.exp(-1))
    expected = [
        global_mean * (1 - rare_weight),
        global_mean * (1 - common_weight) + 10 * common_weight,
    ]

    np.testing.assert_allclose(
        encoded, [expected[0], expected[1]] * 1 + [expected[1]] * 2
    )


@pytest.mark.parametrize(
    ("handle_missing", "handle_unknown", "expected"),
    [
        ("value", "value", [2.0, 2.0]),
        ("return_nan", "return_nan", [np.nan, np.nan]),
    ],
)
def test_polars_target_encoder_controls_missing_and_unknown_values(
    handle_missing, handle_unknown, expected
):
    encoder = PolarsTargetEncoder(
        ["category"],
        smoothing=0.0,
        handle_missing=handle_missing,
        handle_unknown=handle_unknown,
        drop_original=False,
    )
    encoder.fit_transform(pl.DataFrame({"category": ["a", "a"]}), np.array([1.0, 3.0]))

    encoded = encoder.transform(pl.DataFrame({"category": [None, "new"]}))

    np.testing.assert_allclose(encoded["category__mean"], expected, equal_nan=True)


def test_polars_target_encoder_errors_for_missing_and_unknown_values():
    missing = PolarsTargetEncoder(["category"], handle_missing="error")
    with pytest.raises(ValueError, match="Missing values"):
        missing.fit_transform(pl.DataFrame({"category": [None]}), np.array([1.0]))

    unknown = PolarsTargetEncoder(["category"], handle_unknown="error")
    unknown.fit_transform(pl.DataFrame({"category": ["a"]}), np.array([1.0]))
    with pytest.raises(ValueError, match="Unknown categories"):
        unknown.transform(pl.DataFrame({"category": ["new"]}))


def test_polars_target_encoder_smooths_towards_global_training_mean():
    X = pl.DataFrame(
        {
            "category": ["a", "a", "b", "b"],
            "division": ["x", "x", "x", "x"],
        }
    )
    encoded = PolarsTargetEncoder(
        ["category", "division"], min_samples_leaf=2, smoothing=2.0, prior=0.0
    ).fit_transform(
        X,
        np.array([10.0, 20.0, 30.0, 40.0]),
    )

    np.testing.assert_allclose(encoded["category__mean"], [20.0, 20.0, 30.0, 30.0])
    np.testing.assert_allclose(encoded["division__mean"], [25.0] * 4)


def test_direct_models_pass_context_to_context_aware_encoder():
    class ContextRecordingEncoder:
        contexts = []

        def fit_transform(self, X, _y, *, context):
            type(self).contexts.append(
                (np.asarray(context["times"]), np.asarray(context["target_times"]))
            )
            return X

        def transform(self, X):
            return X

    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 8,
            "ds": range(8),
            "y": np.arange(8, dtype=float),
            "category": ["x"] * 8,
        }
    )
    ContextRecordingEncoder.contexts = []
    fcst = MLForecast(
        models=DummyRegressor(),
        freq=1,
        lags=[1],
        feature_encoders=[ContextRecordingEncoder()],
    )

    fcst.fit(df, max_horizon=2, static_features=["category"])

    origin_times, target_times = ContextRecordingEncoder.contexts[1]
    np.testing.assert_array_equal(target_times, origin_times + 1)


def test_polars_target_encoder_supports_as_numpy():
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 6,
            "ds": range(6),
            "y": np.arange(6, dtype=float),
            "category": ["x"] * 6,
        }
    )
    fcst = MLForecast(
        models=DummyRegressor(),
        freq=1,
        lags=[1],
        feature_encoders=[PolarsTargetEncoder(["category"], drop_original=True)],
    )

    fcst.fit(df, static_features=["category"], as_numpy=True)

    assert fcst.predict(1).shape == (1, 3)


def test_polars_target_encoder_fills_missing_categories_with_global_mean():
    encoder = PolarsTargetEncoder(["category"], smoothing=0.0, prior=0.0)
    encoded = encoder.fit_transform(
        pl.DataFrame({"category": [None, None, "a"]}),
        np.array([1.0, 2.0, 3.0]),
    )

    assert encoded["category__mean"].null_count() == 0
    np.testing.assert_allclose(encoded["category__mean"], [2.0, 2.0, 3.0])


def test_target_encoder_null_category_is_encoded_at_fit_time():
    """Null categories are regular categories in the fitted mapping."""
    X = pl.DataFrame({"category": ["a", None, "a", None]})
    y = np.array([1.0, 2.0, 3.0, 4.0])
    encoder = PolarsTargetEncoder(["category"])

    encoded = encoder.fit_transform(X, y)["category__mean"]

    assert encoded.is_null().sum() == 0
    assert np.isfinite(encoded.to_numpy()).all()


def test_mlforecast_feature_encoder_hook_fits_and_transforms():
    class RecordingEncoder:
        def fit_transform(self, X, _y):
            self.fit_called = True
            return X.assign(encoded_feature=1.0)

        def transform(self, X):
            self.transform_called = True
            return X.assign(encoded_feature=1.0)

    df = pd.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 5),
            "ds": list(range(5)) * 2,
            "y": np.arange(10, dtype=float),
        }
    )
    encoder = RecordingEncoder()
    fcst = MLForecast(
        models=DummyRegressor(), freq=1, lags=[1], feature_encoders=[encoder]
    )

    fcst.fit(df)
    fcst.predict(1)

    fitted_encoder = fcst.models_["DummyRegressor"].encoders[0]
    assert fitted_encoder.fit_called
    assert fitted_encoder.transform_called


def test_existing_sklearn_category_encoder_pipeline_is_compatible():
    category_encoders = pytest.importorskip("category_encoders")
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 6),
            "ds": list(range(6)) * 2,
            "y": np.arange(12, dtype=float),
            "category": np.repeat(["first", "second"], 6),
        }
    )
    model = make_pipeline(
        category_encoders.TargetEncoder(cols=["category"]), DummyRegressor()
    )
    fcst = MLForecast(models=model, freq=1, lags=[1])

    fcst.fit(df, static_features=["category"])
    forecast = fcst.predict(1)

    assert forecast.shape == (2, 3)


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


@pytest.mark.parametrize(("max_horizon", "h"), [(None, 1), (2, 2)])
def test_polars_encoder_converts_for_catboost_at_the_model_boundary(
    monkeypatch, max_horizon, h
):
    import mlforecast.feature_encoders as feature_encoders

    monkeypatch.setattr(feature_encoders, "CatBoostRegressor", _PolarsOnlyCatBoost)
    _PolarsOnlyCatBoost.inputs = []
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 8,
            "ds": range(8),
            "y": np.arange(8, dtype=float),
            "category": ["x"] * 8,
        }
    )
    fcst = MLForecast(
        models=_PolarsOnlyCatBoost(),
        freq=1,
        lags=[1],
        feature_encoders=[PolarsOrdinalEncoder(["category"], drop_original=True)],
    )

    fcst.fit(df, max_horizon=max_horizon, static_features=["category"])
    fcst.predict(h)

    expected_inputs = 2 if max_horizon is None else 4
    assert _PolarsOnlyCatBoost.inputs == [pd.DataFrame] * expected_inputs


def test_mlforecast_polars_target_encoder_fits_and_predicts():
    df = pl.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 5),
            "ds": list(range(5)) * 2,
            "y": np.arange(10, dtype=float),
            "category": np.repeat(["first", "second"], 5),
        }
    ).with_columns(pl.col("category").cast(pl.Categorical))
    fcst = MLForecast(
        models=DummyRegressor(),
        freq=1,
        lags=[1],
        feature_encoders=[PolarsTargetEncoder(["category"], drop_original=True)],
    )

    fcst.fit(df, static_features=["category"])
    forecast = fcst.predict(1)

    assert forecast.shape == (2, 3)


def test_polars_target_encoder_supports_direct_models_and_save_load(tmp_path):
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 10,
            "ds": range(10),
            "y": np.arange(10, dtype=float),
            "category": ["x"] * 10,
        }
    ).with_columns(pl.col("category").cast(pl.Categorical))
    fcst = MLForecast(
        models=DummyRegressor(),
        freq=1,
        lags=[1],
        feature_encoders=[PolarsTargetEncoder(["category"], drop_original=True)],
    )

    fcst.fit(df, max_horizon=2, static_features=["category"])
    assert fcst.predict(2).shape == (2, 3)
    path = tmp_path / "model"
    fcst.save(path)
    assert MLForecast.load(path).predict(2).shape == (2, 3)


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
        feature_encoders=[_target_encoder()],
    )
    fcst.fit(df, static_features=["category"])

    path = tmp_path / "model"
    fcst.save(path)
    loaded = MLForecast.load(path)
    loaded.fit(df, static_features=["category"])

    assert isinstance(loaded.models_["DummyRegressor"], _EncodedModel)


def test_fitted_values_reuse_target_encoder_training_features():
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 6,
            "ds": range(6),
            "y": np.arange(6, dtype=float),
            "category": ["x"] * 6,
        }
    ).with_columns(pl.col("category").cast(pl.Categorical))
    fcst = MLForecast(
        models=_EncodedColumnRegressor(),
        freq=1,
        lags=[1],
        feature_encoders=[
            PolarsTargetEncoder(
                ["category"], smoothing=0.0, prior=0.0, drop_original=True
            )
        ],
    )

    fcst.fit(df, fitted=True, static_features=["category"])
    fitted = fcst.forecast_fitted_values()

    np.testing.assert_allclose(fitted["_EncodedColumnRegressor"], [3.0] * 5)


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
        feature_encoders=[_target_encoder()],
    )
    sorted_fcst.fit(_two_series("sorted"), **kwargs)
    expected = sorted_fcst.forecast_fitted_values().sort(["unique_id", "ds"])

    shuffled_fcst = MLForecast(
        models=_EchoEncodedColumn(),
        freq=1,
        lags=[1],
        feature_encoders=[_target_encoder()],
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
        feature_encoders=[_target_encoder()],
    )
    sorted_fcst.fit(_two_series("sorted"), **kwargs)
    expected = sorted_fcst.forecast_fitted_values().sort(["unique_id", "ds", "h"])

    shuffled_fcst = MLForecast(
        models=_EchoEncodedColumn(),
        freq=1,
        lags=[1],
        feature_encoders=[_target_encoder()],
    )
    shuffled_fcst.fit(_two_series("interleaved"), **kwargs)
    actual = shuffled_fcst.forecast_fitted_values().sort(["unique_id", "ds", "h"])

    np.testing.assert_allclose(
        actual["_EchoEncodedColumn"], expected["_EchoEncodedColumn"]
    )


def test_fitted_values_cache_predictions_not_encoded_feature_frames():
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 8,
            "ds": range(8),
            "y": np.arange(8, dtype=float),
            "category": ["x"] * 8,
        }
    )
    fcst = MLForecast(
        models=DummyRegressor(),
        freq=1,
        lags=[1],
        feature_encoders=[PolarsOrdinalEncoder(["category"], drop_original=True)],
    )

    fcst.fit(df, max_horizon=2, fitted=True, static_features=["category"])

    for model in fcst.models_["DummyRegressor"].values():
        assert hasattr(model, "fitted_predictions_")
        assert not hasattr(model, "fitted_X_")


def test_direct_fitted_values_handle_lexically_sorted_series_ids():
    n_series, n_days = 12, 8
    df = pl.DataFrame(
        {
            "unique_id": np.repeat(np.arange(n_series).astype(str), n_days),
            "ds": np.tile(np.arange(n_days), n_series),
            "y": np.arange(n_series * n_days, dtype=float),
            "category": np.repeat(["x", "y"], n_series * n_days // 2),
        }
    )
    fcst = MLForecast(
        models=DummyRegressor(),
        freq=1,
        lags=[1],
        feature_encoders=[PolarsOrdinalEncoder(["category"], drop_original=True)],
    )

    fcst.fit(df, max_horizon=2, fitted=True, static_features=["category"])

    assert fcst.forecast_fitted_values().shape[0] > 0


def test_polars_encoder_works_with_cross_validation_and_automl():
    df = pl.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 8),
            "ds": list(range(8)) * 2,
            "y": np.arange(16, dtype=float),
            "category": np.repeat(["first", "second"], 8),
        }
    ).with_columns(pl.col("category").cast(pl.Categorical))
    encoder = PolarsTargetEncoder(["category"], drop_original=True)
    fcst = MLForecast(
        models=DummyRegressor(), freq=1, lags=[1], feature_encoders=[encoder]
    )

    cv_results = fcst.cross_validation(
        df, n_windows=2, h=1, step_size=1, static_features=["category"]
    )
    assert cv_results.shape == (4, 5)

    auto = AutoMLForecast(
        models={"dummy": AutoModel(DummyRegressor(), lambda _trial: {})},
        freq=1,
        init_config=lambda _trial: {
            "lags": [1],
            "feature_encoders": [PolarsTargetEncoder(["category"], drop_original=True)],
        },
        fit_config=lambda _trial: {"static_features": ["category"]},
    )
    auto.fit(df, n_windows=2, h=1, num_samples=1)

    assert auto.predict(1).shape == (2, 3)


def test_target_encoder_uses_each_cross_validation_fold_training_mean():
    class AuditedTargetEncoder(PolarsTargetEncoder):
        batches = []

        def fit_transform(self, X, y):
            out = super().fit_transform(X, y)
            type(self).batches.append(out["category__mean"].to_numpy())
            return out

    # The last target is intentionally enormous. Neither CV fold may use it
    # when fitting its encoder or creating its training encodings.
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 8,
            "ds": range(8),
            "y": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10_000.0],
            "category": ["x"] * 8,
        }
    ).with_columns(pl.col("category").cast(pl.Categorical))
    AuditedTargetEncoder.batches = []
    fcst = MLForecast(
        DummyRegressor(),
        freq=1,
        lags=[1],
        feature_encoders=[
            AuditedTargetEncoder(
                ["category"], smoothing=0.0, prior=0.0, drop_original=True
            )
        ],
    )

    fcst.cross_validation(
        df, n_windows=2, h=1, step_size=1, static_features=["category"]
    )

    # Each fold uses only its own training partition. Its training rows use the
    # conventional in-sample category mapping fit on that complete partition.
    expected = [
        np.full(5, 3.0),
        np.full(6, 3.5),
    ]
    for encoded, expected_encoded in zip(AuditedTargetEncoder.batches, expected):
        np.testing.assert_allclose(encoded, expected_encoded)

    # `cv_models_` retains one model per fold. Their final category mappings
    # equal the training-only means (3.0 and 3.5), not the held-out 10,000.
    mappings = [
        model["DummyRegressor"].encoders[0].mappings_["category"]["category__mean"][0]
        for model in fcst.cv_models_
    ]
    np.testing.assert_allclose(mappings, [3.0, 3.5])


def test_cross_validation_fitted_values_reuse_initial_mapping_when_refit_is_false():
    """`refit=False` must reuse the model and its encoder from the first fold."""
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
        feature_encoders=[_target_encoder()],
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

    first, second = (fitted.filter(pl.col("fold") == fold) for fold in (0, 1))
    np.testing.assert_allclose(first["_EchoEncodedColumn"], 3.0)
    np.testing.assert_allclose(second["_EchoEncodedColumn"], 3.0)


def test_transform_per_horizon_skips_context_without_feature_encoders():
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 8,
            "ds": range(8),
            "y": np.arange(8, dtype=float),
        }
    )
    fcst = MLForecast(models=DummyRegressor(), freq=1, lags=[1])
    prep = fcst.preprocess(df, max_horizon=2, return_X_y=False)

    batches = list(
        fcst.ts._transform_per_horizon(
            prep,
            df,
            horizons=[0, 1],
            target_col="y",
            with_encoder_context=False,
        )
    )

    assert all(len(batch) == 3 for batch in batches)


def test_fit_models_accepts_legacy_three_value_generator_factory():
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 6,
            "ds": range(6),
            "y": np.arange(6, dtype=float),
        }
    )
    fcst = MLForecast(models=DummyRegressor(), freq=1, lags=[1])
    X, y = fcst.preprocess(df, return_X_y=True)

    def generator_factory():
        yield 0, X, y

    fcst.fit_models(generator_factory=generator_factory)

    assert 0 in fcst.models_["DummyRegressor"]
