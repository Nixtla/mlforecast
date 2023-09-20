Nixtla  
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/statsforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting)
 [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white.png)](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

<div align="center">

<center>
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/logo_mid.png">
</center>
<h1 align="center">
Machine Learning 🤖 Forecast
</h1>
<h3 align="center">
Scalable machine learning for time series forecasting
</h3>

[![CI](https://github.com/Nixtla/mlforecast/actions/workflows/ci.yaml/badge.svg)](https://github.com/Nixtla/mlforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/mlforecast.png)](https://pypi.org/project/mlforecast/)
[![PyPi](https://img.shields.io/pypi/v/mlforecast?color=blue.png)](https://pypi.org/project/mlforecast/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/mlforecast?color=blue.png)](https://anaconda.org/conda-forge/mlforecast)
[![License](https://img.shields.io/github/license/Nixtla/mlforecast.png)](https://github.com/Nixtla/mlforecast/blob/main/LICENSE)

**mlforecast** is a framework to perform time series forecasting using
machine learning models, with the option to scale to massive amounts of
data using remote clusters.

</div>

## Install

### PyPI

`pip install mlforecast`

If you want to perform distributed training, you can instead use
`pip install "mlforecast[distributed]"`, which will also install
[dask](https://dask.org/). Note that you’ll also need to install either
[LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package)
or
[XGBoost](https://xgboost.readthedocs.io/en/latest/install.html#python).

### conda-forge

`conda install -c conda-forge mlforecast`

Note that this installation comes with the required dependencies for the
local interface. If you want to perform distributed training, you must
install dask (`conda install -c conda-forge dask`) and either
[LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package)
or
[XGBoost](https://xgboost.readthedocs.io/en/latest/install.html#python).

## Quick Start

**Minimal Example**

``` python
import lightgbm as lgb

from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression

mlf = MLForecast(
    models = [LinearRegression(), lgb.LGBMRegressor()],
    lags=[1, 12],
    freq = 'M'
)
mlf.fit(df)
mlf.predict(12)
```

**Get Started with this [quick
guide](https://nixtla.github.io/mlforecast/docs/quick_start_local.html).**

**Follow this [end-to-end
walkthrough](https://nixtla.github.io/mlforecast/docs/end_to_end_walkthrough.html)
for best practices.**

## Why?

Current Python alternatives for machine learning models are slow,
inaccurate and don’t scale well. So we created a library that can be
used to forecast in production environments. `MLForecast` includes
efficient feature engineering to train any machine learning model (with
`fit` and `predict` methods such as
[`sklearn`](https://scikit-learn.org/stable/)) to fit millions of time
series.

## Features

- Fastest implementations of feature engineering for time series
  forecasting in Python.
- Out-of-the-box compatibility with Spark, Dask, and Ray.
- Probabilistic Forecasting with Conformal Prediction.
- Support for exogenous variables and static covariates.
- Familiar `sklearn` syntax: `.fit` and `.predict`.

Missing something? Please open an issue or write us in
[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white.png)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## Examples and Guides

📚 [End to End
Walkthrough](https://nixtla.github.io/mlforecast/docs/end_to_end_walkthrough.html):
model training, evaluation and selection for multiple time series.

🔎 [Probabilistic
Forecasting](https://nixtla.github.io/mlforecast/docs/prediction_intervals.html):
use Conformal Prediction to produce prediciton intervals.

👩‍🔬 [Cross
Validation](https://nixtla.github.io/mlforecast/docs/cross_validation.html):
robust model’s performance evaluation.

🔌 [Predict Demand
Peaks](https://nixtla.github.io/mlforecast/docs/electricity_peak_forecasting.html):
electricity load forecasting for detecting daily peaks and reducing
electric bills.

📈 [Transfer
Learning](https://nixtla.github.io/mlforecast/docs/transfer_learning.html):
pretrain a model using a set of time series and then predict another one
using that pretrained model.

🌡️ [Distributed
Training](https://nixtla.github.io/mlforecast/docs/quick_start_distributed.html):
use a Dask cluster to train models at scale.

## How to use

The following provides a very basic overview, for a more detailed
description see the
[documentation](https://nixtla.github.io/mlforecast/).

### Data setup

Store your time series in a pandas dataframe in long format, that is,
each row represents an observation for a specific serie and timestamp.

``` python
from mlforecast.utils import generate_daily_series

series = generate_daily_series(
    n_series=20,
    max_length=100,
    n_static_features=1,
    static_as_categorical=False,
    with_trend=True
)
series.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unique_id</th>
      <th>ds</th>
      <th>y</th>
      <th>static_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_00</td>
      <td>2000-01-01</td>
      <td>17.519167</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_00</td>
      <td>2000-01-02</td>
      <td>87.799695</td>
      <td>72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_00</td>
      <td>2000-01-03</td>
      <td>177.442975</td>
      <td>72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_00</td>
      <td>2000-01-04</td>
      <td>232.704110</td>
      <td>72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_00</td>
      <td>2000-01-05</td>
      <td>317.510474</td>
      <td>72</td>
    </tr>
  </tbody>
</table>
</div>

### Models

Next define your models. If you want to use the local interface this can
be any regressor that follows the scikit-learn API. For distributed
training there are `LGBMForecast` and `XGBForecast`.

``` python
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

models = [
    lgb.LGBMRegressor(verbosity=-1),
    xgb.XGBRegressor(),
    RandomForestRegressor(random_state=0),
]
```

### Forecast object

Now instantiate a `MLForecast` object with the models and the features
that you want to use. The features can be lags, transformations on the
lags and date features. The lag transformations are defined as
[numba](http://numba.pydata.org/) *jitted* functions that transform an
array, if they have additional arguments you can either supply a tuple
(`transform_func`, `arg1`, `arg2`, …) or define new functions fixing the
arguments. You can also define differences to apply to the series before
fitting that will be restored when predicting.

``` python
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean


@njit
def rolling_mean_28(x):
    return rolling_mean(x, window_size=28)


fcst = MLForecast(
    models=models,
    freq='D',
    lags=[7, 14],
    lag_transforms={
        1: [expanding_mean],
        7: [rolling_mean_28]
    },
    date_features=['dayofweek'],
    target_transforms=[Differences([1])],
)
```

### Training

To compute the features and train the models call `fit` on your
`Forecast` object.

``` python
fcst.fit(series)
```

    MLForecast(models=[LGBMRegressor, XGBRegressor, RandomForestRegressor], freq=<Day>, lag_features=['lag7', 'lag14', 'expanding_mean_lag1', 'rolling_mean_28_lag7'], date_features=['dayofweek'], num_threads=1)

### Predicting

To get the forecasts for the next `n` days call `predict(n)` on the
forecast object. This will automatically handle the updates required by
the features using a recursive strategy.

``` python
predictions = fcst.predict(14)
predictions
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unique_id</th>
      <th>ds</th>
      <th>LGBMRegressor</th>
      <th>XGBRegressor</th>
      <th>RandomForestRegressor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_00</td>
      <td>2000-04-04</td>
      <td>299.923771</td>
      <td>309.664124</td>
      <td>298.424164</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_00</td>
      <td>2000-04-05</td>
      <td>365.424147</td>
      <td>382.150085</td>
      <td>365.816014</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_00</td>
      <td>2000-04-06</td>
      <td>432.562441</td>
      <td>453.373779</td>
      <td>436.360620</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_00</td>
      <td>2000-04-07</td>
      <td>495.628000</td>
      <td>527.965149</td>
      <td>503.670100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_00</td>
      <td>2000-04-08</td>
      <td>60.786223</td>
      <td>75.762299</td>
      <td>62.176080</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>275</th>
      <td>id_19</td>
      <td>2000-03-23</td>
      <td>36.266780</td>
      <td>29.889120</td>
      <td>34.799780</td>
    </tr>
    <tr>
      <th>276</th>
      <td>id_19</td>
      <td>2000-03-24</td>
      <td>44.370984</td>
      <td>34.968884</td>
      <td>39.920982</td>
    </tr>
    <tr>
      <th>277</th>
      <td>id_19</td>
      <td>2000-03-25</td>
      <td>50.746222</td>
      <td>39.970238</td>
      <td>46.196266</td>
    </tr>
    <tr>
      <th>278</th>
      <td>id_19</td>
      <td>2000-03-26</td>
      <td>58.906524</td>
      <td>45.125305</td>
      <td>51.653060</td>
    </tr>
    <tr>
      <th>279</th>
      <td>id_19</td>
      <td>2000-03-27</td>
      <td>63.073949</td>
      <td>50.682716</td>
      <td>56.845384</td>
    </tr>
  </tbody>
</table>
<p>280 rows × 5 columns</p>
</div>

### Visualize results

``` python
from utilsforecast.plotting import plot_series
```

``` python
fig = plot_series(series, predictions, max_ids=4, plot_random=False)
fig.savefig('figs/index.png', bbox_inches='tight')
```

![](https://raw.githubusercontent.com/Nixtla/mlforecast/main/nbs/figs/index.png)

## Sample notebooks

- [m5](https://www.kaggle.com/code/lemuz90/m5-mlforecast-eval)
- [m4](https://www.kaggle.com/code/lemuz90/m4-competition)
- [m4-cv](https://www.kaggle.com/code/lemuz90/m4-competition-cv)

## How to contribute

See
[CONTRIBUTING.md](https://github.com/Nixtla/mlforecast/blob/main/CONTRIBUTING.md).
