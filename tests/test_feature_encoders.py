import numpy as np
import pandas as pd
import polars as pl
from sklearn.dummy import DummyRegressor

from mlforecast import (
    MLForecast,
    PolarsTargetEncoder,
    PolarsCountEncoder,
    PolarsOneHotEncoder,
    PolarsOrdinalEncoder,
)
from mlforecast.auto import AutoMLForecast, AutoModel


def test_polars_ordinal_encoder_handles_unseen_categories():
    encoder = PolarsOrdinalEncoder(["category"], drop_original=True)
    train = encoder.fit_transform(pl.DataFrame({"category": ["b", "a", "b"]}), np.zeros(3))
    future = encoder.transform(pl.DataFrame({"category": ["a", "missing"]}))

    assert train.columns == ["category__ordinal"]
    assert train["category__ordinal"].to_list() == [0, 1, 0]
    assert future["category__ordinal"].to_list() == [1, -1]


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


def test_polars_target_encoder_uses_only_prior_timestamps():
    X = pl.DataFrame({"category": ["a", "a"], "lag1": [1.0, 2.0]})
    context = {"times": np.array([1, 2])}
    encoder = PolarsTargetEncoder(["category"], smoothing=0.0, prior=0.0)

    encoded = encoder.fit_transform(X, np.array([1.0, 2.0]), context=context)
    encoded_with_changed_future = PolarsTargetEncoder(
        ["category"], smoothing=0.0, prior=0.0
    ).fit_transform(X, np.array([1.0, 999.0]), context=context)

    np.testing.assert_allclose(encoded["category__mean"], [0.0, 1.0])
    np.testing.assert_allclose(
        encoded["category__mean"], encoded_with_changed_future["category__mean"]
    )


def test_mlforecast_feature_encoder_hook_fits_and_transforms():
    class RecordingEncoder:
        def fit_transform(self, X, y):
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
        models={"dummy": AutoModel(DummyRegressor(), lambda trial: {})},
        freq=1,
        init_config=lambda trial: {
            "lags": [1],
            "feature_encoders": [
                PolarsTargetEncoder(["category"], drop_original=True)
            ],
        },
        fit_config=lambda trial: {"static_features": ["category"]},
    )
    auto.fit(df, n_windows=2, h=1, num_samples=1)

    assert auto.predict(1).shape == (2, 3)


def test_target_encoder_cross_validation_is_leakage_free():
    class AuditedTargetEncoder(PolarsTargetEncoder):
        batches = []

        def fit_transform(self, X, y, *, context):
            out = super().fit_transform(X, y, context=context)
            type(self).batches.append(
                (np.asarray(context["times"]), out["category__mean"].to_numpy())
            )
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
            AuditedTargetEncoder(["category"], smoothing=0.0, prior=0.0, drop_original=True)
        ],
    )

    fcst.cross_validation(
        df, n_windows=2, h=1, step_size=1, static_features=["category"]
    )

    # Hand calculation: at each timestamp the encoding is the mean of the
    # preceding targets only; the first feature row has no prior target.
    expected = [
        np.array([0.0, 1.0, 1.5, 2.0, 2.5]),
        np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0]),
    ]
    for (_, encoded), expected_encoded in zip(AuditedTargetEncoder.batches, expected):
        np.testing.assert_allclose(encoded, expected_encoded)

    # `cv_models_` retains one model per fold. Their final category mappings
    # equal the training-only means (3.0 and 3.5), not the held-out 10,000.
    mappings = [
        model["DummyRegressor"].encoders[0].mappings_["category"]["category__mean"][0]
        for model in fcst.cv_models_
    ]
    np.testing.assert_allclose(mappings, [3.0, 3.5])
