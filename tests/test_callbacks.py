import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from utilsforecast.losses import smape

from mlforecast import MLForecast
from mlforecast.callbacks import Callback, Profiler
from mlforecast.utils import PredictionIntervals, generate_daily_series


class Recorder(Callback):
    def __init__(self):
        self.events = []

    def on_start(self, stage, fcst, **kwargs):  # noqa: ARG002
        self.events.append(("start", stage, kwargs))

    def on_end(self, stage, fcst, result, **kwargs):  # noqa: ARG002
        self.events.append(("end", stage, kwargs))

    def on_error(self, stage, fcst, error, **kwargs):  # noqa: ARG002
        self.events.append(("error", stage, kwargs))

    def stages(self, hook):
        return [stage for h, stage, _ in self.events if h == hook]


class Boom(BaseEstimator):
    def fit(self, X, y):  # noqa: ARG002
        raise RuntimeError("boom")


@pytest.fixture
def series():
    return generate_daily_series(4, min_length=20, max_length=30, equal_ends=True)


def _stages(span):
    return [child.stage for child in span.children]


def _count(span, stage):
    return (span.stage == stage) + sum(_count(child, stage) for child in span.children)


def test_callbacks_are_off_by_default_and_methods_keep_their_identity():
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1])
    assert fcst.callbacks == []
    assert MLForecast.fit.__name__ == "fit"
    assert MLForecast.predict.__doc__.startswith("Compute the predictions")


def test_stages_receive_bound_arguments(series):
    recorder = Recorder()
    fcst = MLForecast(
        models=LinearRegression(), freq="D", lags=[1], callbacks=[recorder]
    )
    fcst.fit(series).predict(3)

    assert recorder.stages("start") == [
        "fit",
        "preprocess",
        "fit_models",
        "fit_model",
        "predict",
    ]
    assert recorder.stages("end") == [
        "preprocess",
        "fit_model",
        "fit_models",
        "fit",
        "predict",
    ]
    (fit_model,) = [kw for h, s, kw in recorder.events if s == "fit_model"][:1]
    assert fit_model["name"] == "LinearRegression"
    assert fit_model["h"] is None
    assert fit_model["X"].shape[0] == len(fit_model["y"])
    (predict,) = [kw for h, s, kw in recorder.events if s == "predict"][:1]
    assert predict["h"] == 3
    # defaults are bound too
    assert predict["level"] is None
    (fit,) = [kw for h, s, kw in recorder.events if s == "fit"][:1]
    assert fit["df"] is series
    assert fit["id_col"] == "unique_id"


def test_direct_models_emit_one_fit_model_per_horizon(series):
    recorder = Recorder()
    fcst = MLForecast(
        models=[LinearRegression(), LinearRegression()],
        freq="D",
        lags=[1],
        callbacks=[recorder],
    )
    fcst.fit(series, max_horizon=2)

    fit_models = [kw for h, s, kw in recorder.events if h == "end" and s == "fit_model"]
    assert [(kw["name"], kw["h"]) for kw in fit_models] == [
        ("LinearRegression", 0),
        ("LinearRegression", 1),
        ("LinearRegression2", 0),
        ("LinearRegression2", 1),
    ]


def test_profiler_nests_calibration_inside_fit(series):
    profiler = Profiler()
    fcst = MLForecast(
        models=LinearRegression(), freq="D", lags=[1], callbacks=[profiler]
    )
    fcst.fit(series, prediction_intervals=PredictionIntervals(n_windows=2, h=1))

    assert [span.stage for span in profiler.spans] == ["fit"]
    fit = profiler.spans[0]
    assert _stages(fit) == ["cross_validation", "preprocess", "fit_models"]
    cv = fit.children[0]
    assert cv.details == {"n_windows": 2, "h": 1}
    assert _stages(cv) == ["cv_window", "cv_window"]
    # refit=False during calibration: only the first window fits
    assert _stages(cv.children[0]) == ["fit", "predict"]
    assert _stages(cv.children[1]) == ["predict"]
    assert _count(fit, "fit_model") == 2
    assert fit.seconds >= sum(child.seconds for child in fit.children)


def test_profiler_keeps_every_cross_validation_window(series):
    profiler = Profiler()
    fcst = MLForecast(
        models=[LinearRegression(), LinearRegression()],
        freq="D",
        lags=[1],
        callbacks=[profiler],
    )
    fcst.cross_validation(series, n_windows=3, h=2, refit=True)
    df = profiler.to_df()

    assert df["stage"].tolist()[:2] == ["cross_validation", "cv_window"]
    assert (df["stage"] == "cv_window").sum() == 3
    assert (df["stage"] == "fit").sum() == 3
    assert df.loc[df["stage"] == "cv_window", "i_window"].tolist() == [0, 1, 2]
    assert df.loc[df["stage"] == "cross_validation", "depth"].tolist() == [0]
    assert df.loc[df["stage"] == "fit_model", "depth"].tolist() == [4] * 6
    per_model = df.query("stage == 'fit_model'").groupby("model")["seconds"].sum()
    assert per_model.index.tolist() == ["LinearRegression", "LinearRegression2"]
    assert (per_model > 0).all()


def test_profiler_can_record_memory(series):
    pytest.importorskip("psutil")
    profiler = Profiler(memory=True)
    fcst = MLForecast(
        models=LinearRegression(), freq="D", lags=[1], callbacks=[profiler]
    )
    fcst.fit(series)

    assert isinstance(profiler.spans[0].rss_delta_bytes, int)
    assert Profiler().on_start("fit", fcst) is None


def test_errors_are_reported_and_profiler_recovers(series):
    recorder = Recorder()
    profiler = Profiler()
    fcst = MLForecast(models=Boom(), freq="D", lags=[1], callbacks=[recorder, profiler])
    with pytest.raises(RuntimeError, match="boom"):
        fcst.cross_validation(series, n_windows=2, h=1)

    # cv_window has no error hook, the profiler unwinds past it
    assert recorder.stages("error") == [
        "fit_model",
        "fit_models",
        "fit",
        "cross_validation",
    ]
    assert profiler._stack == []
    assert profiler.spans == []

    fcst.models = {"LinearRegression": LinearRegression()}
    fcst.fit(series)
    assert [span.stage for span in profiler.spans] == ["fit"]


def test_custom_callback_scores_every_window(series):
    class WindowScores(Callback):
        def __init__(self):
            self.scores = []

        def on_end(self, stage, fcst, result, **kwargs):
            if stage == "cv_window":
                score = smape(result, models=list(fcst.models))
                self.scores.append((kwargs["i_window"], score))

    scores = WindowScores()
    fcst = MLForecast(models=LinearRegression(), freq="D", lags=[1], callbacks=[scores])
    fcst.cross_validation(series, n_windows=3, h=2)

    assert [i for i, _ in scores.scores] == [0, 1, 2]
    assert all(score.shape[0] == 4 for _, score in scores.scores)
