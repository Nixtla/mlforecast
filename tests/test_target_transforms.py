import numpy as np
import pandas as pd
from fastcore.test import test_fail
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from utilsforecast.processing import counts_by_id

from mlforecast import MLForecast
from mlforecast.grouped_array import GroupedArray
from mlforecast.target_transforms import (
    AutoDifferences,
    AutoSeasonalDifferences,
    AutoSeasonalityAndDifferences,
    Differences,
    GlobalSklearnTransformer,
    LocalBoxCox,
    LocalMinMaxScaler,
    LocalRobustScaler,
    LocalStandardScaler,
)
from mlforecast.utils import generate_daily_series

series = generate_daily_series(10, min_length=50, max_length=100)
diffs = Differences([1, 2, 5])
id_counts = counts_by_id(series, 'unique_id')
indptr = np.append(0, id_counts['counts'].cumsum())
ga = GroupedArray(series['y'].values, indptr)

# differences are applied correctly
transformed = diffs.fit_transform(ga)
assert diffs.fitted_ == []
expected = series.copy()
for d in diffs.differences:
    expected['y'] -= expected.groupby('unique_id', observed=True)['y'].shift(d)
np.testing.assert_allclose(transformed.data, expected['y'].values)

# fitted differences are restored correctly
diffs.store_fitted = True
transformed = diffs.fit_transform(ga)
keep_mask = ~np.isnan(transformed.data)
restored = diffs.inverse_transform_fitted(transformed)
np.testing.assert_allclose(ga.data[keep_mask], restored.data[keep_mask])

# test transform
new_ga = GroupedArray(np.random.rand(10), np.arange(11))
prev_orig = [diffs.scalers_[i].tails_[::d].copy() for i, d in enumerate(diffs.differences)]
expected = new_ga.data - np.add.reduce(prev_orig)
updates = diffs.update(new_ga)
np.testing.assert_allclose(expected, updates.data)
np.testing.assert_allclose(diffs.scalers_[0].tails_, new_ga.data)
np.testing.assert_allclose(diffs.scalers_[1].tails_[1::2], new_ga.data - prev_orig[0])
np.testing.assert_allclose(diffs.scalers_[2].tails_[4::5], new_ga.data - np.add.reduce(prev_orig[:2]))
# variable sizes
diff1 = Differences([1])
ga = GroupedArray(np.arange(10), np.array([0, 3, 10]))
diff1.fit_transform(ga)
new_ga = GroupedArray(np.arange(4), np.array([0, 1, 4]))
updates = diff1.update(new_ga)
np.testing.assert_allclose(updates.data, np.array([0 - 2, 1 - 9, 2 - 1, 3 - 2]))
np.testing.assert_allclose(diff1.scalers_[0].tails_, np.array([0, 3]))

# short series
ga = GroupedArray(np.arange(20), np.array([0, 2, 20]))
test_fail(lambda: diffs.fit_transform(ga), contains="[0]")

# stack
diffs = Differences([1, 2, 5])
ga = GroupedArray(series['y'].values, indptr)
diffs.fit_transform(ga)
stacked = Differences.stack([diffs, diffs])
for i in range(len(diffs.differences)):
    np.testing.assert_allclose(
        stacked.scalers_[i].tails_,
        np.tile(diffs.scalers_[i].tails_, 2)
    )
sc = AutoDifferences(1)
ga = GroupedArray(np.arange(10), np.array([0, 10]))
transformed = sc.fit_transform(ga)
np.testing.assert_allclose(transformed.data, np.append(np.nan, np.ones(9)))
np.testing.assert_equal(sc.scaler_.diffs_, np.array([1.0], dtype=np.float32))
inv = sc.inverse_transform(ga).data
np.testing.assert_allclose(9 + np.arange(10).cumsum(), inv)
upd = sc.update(GroupedArray(np.array([10]), np.array([0, 1])))
np.testing.assert_equal(np.array([1.0]), upd.data)
np.testing.assert_equal(sc.scaler_.tails_[0], np.array([10]))

# stack
stacked = AutoDifferences.stack([sc, sc])
np.testing.assert_allclose(
    stacked.scaler_.diffs_,
    np.tile(sc.scaler_.diffs_, 2),
)
np.testing.assert_allclose(
    stacked.scaler_.tails_,
    np.tile(sc.scaler_.tails_, 2),
)
sc = AutoSeasonalDifferences(season_length=5, max_diffs=1)
ga = GroupedArray(np.arange(5)[np.arange(10) % 5], np.array([0, 10]))
transformed = sc.fit_transform(ga)
sc.inverse_transform(ga)
sc.update(ga)

# stack
stacked = AutoDifferences.stack([sc, sc])
np.testing.assert_allclose(
    stacked.scaler_.diffs_,
    np.tile(sc.scaler_.diffs_, 2),
)
np.testing.assert_allclose(
    stacked.scaler_.tails_,
    np.tile(sc.scaler_.tails_, 2),
)
sc = AutoSeasonalityAndDifferences(max_season_length=5, max_diffs=1)
ga = GroupedArray(np.arange(5)[np.arange(10) % 5], np.array([0, 10]))
transformed = sc.fit_transform(ga)
sc.inverse_transform(ga)
sc.update(ga)

# stack
stacked = AutoDifferences.stack([sc, sc])
np.testing.assert_allclose(
    stacked.scaler_.diffs_,
    np.tile(sc.scaler_.diffs_, 2),
)
np.testing.assert_allclose(
    stacked.scaler_.tails_,
    np.tile(sc.scaler_.tails_, 2),
)
def test_scaler(sc, series):
    id_counts = counts_by_id(series, 'unique_id')
    indptr = np.append(0, id_counts['counts'].cumsum())
    ga = GroupedArray(series['y'].values, indptr)
    transformed = sc.fit_transform(ga)
    np.testing.assert_allclose(
        sc.inverse_transform(transformed).data,
        ga.data,
    )
    transformed2 = sc.update(ga)
    np.testing.assert_allclose(transformed.data, transformed2.data)

    idxs = [0, 7]
    subset = ga.take(idxs)
    transformed_subset = transformed.take(idxs)
    subsc = sc.take(idxs)
    np.testing.assert_allclose(
        subsc.inverse_transform(transformed_subset).data,
        subset.data,
    )

    stacked = sc.stack([sc, sc])
    stacked_stats = stacked.scaler_.stats_
    np.testing.assert_allclose(
        stacked_stats,
        np.tile(sc.scaler_.stats_, (2, 1)),
    )
test_scaler(LocalStandardScaler(), series)
test_scaler(LocalMinMaxScaler(), series)
test_scaler(LocalRobustScaler(scale='iqr'), series)
test_scaler(LocalRobustScaler(scale='mad'), series)
test_scaler(LocalBoxCox(), series)
# need this import in order for isinstance to work
from mlforecast.target_transforms import Differences as ExportedDifferences

sk_boxcox = PowerTransformer(method='box-cox', standardize=False)
boxcox_global = GlobalSklearnTransformer(sk_boxcox)
single_difference = ExportedDifferences([1])
series = generate_daily_series(10)
fcst = MLForecast(
    models=[LinearRegression(), HistGradientBoostingRegressor()],
    freq='D',
    lags=[1, 2],
    target_transforms=[boxcox_global, single_difference]
)
prep = fcst.preprocess(series, dropna=False)
expected = (
    pd.Series(
        sk_boxcox.fit_transform(series[['y']])[:, 0], index=series['unique_id']
    ).groupby('unique_id', observed=True)
    .diff()
    .dropna()
    .values
)
np.testing.assert_allclose(prep['y'].values, expected)
preds = fcst.fit(series).predict(5)
series_pl = generate_daily_series(10, engine='polars')
fcst_pl = MLForecast(
    models=[LinearRegression(), HistGradientBoostingRegressor()],
    freq='1d',
    lags=[1, 2],
    target_transforms=[boxcox_global, single_difference]
)
prep_pl = fcst_pl.preprocess(series_pl, dropna=False)
pd.testing.assert_frame_equal(prep.reset_index(drop=True), prep_pl.to_pandas())
pl_preds = fcst_pl.fit(series_pl).predict(5)
pd.testing.assert_frame_equal(preds, pl_preds.to_pandas())
