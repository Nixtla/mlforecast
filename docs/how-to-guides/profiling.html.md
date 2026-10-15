---
title: Profile forecasting stages
description: Find where MLForecast spends time and memory during fit and cross validation.
---

Pass a `Profiler` through `hooks` to record nested stages:

```python
from sklearn.linear_model import LinearRegression
from mlforecast import MLForecast
from mlforecast.hooks import Profiler

profiler = Profiler(memory=True)
fcst = MLForecast(
    models=LinearRegression(), freq="D", lags=[1, 7], hooks=[profiler]
)
cv = fcst.cross_validation(df, n_windows=3, h=7)
spans = profiler.to_df()
print(spans[["stage", "depth", "seconds", "rss_delta_bytes"]])
```

To compare model fit costs, select only the `fit_model` rows:

```python
print(spans.query("stage == 'fit_model'").groupby("model")["seconds"].sum())
```

Each stage's duration includes its child stages. Failed stages remain in the
table with an `error` value. To profile another run, call `profiler.reset()`
after the first run finishes.
