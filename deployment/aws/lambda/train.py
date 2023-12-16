import pickle

from mlforecast import MLForecast
from mlforecast.utils import generate_series
from sklearn.linear_model import LinearRegression

series = generate_series(10)
fcst = MLForecast(
    models=[LinearRegression()],
    freq='D',
    lags=[1],
    date_features=['dayofweek', 'month'],
)
fcst.fit(series)
with open('app/fcst.pkl', 'wb') as f:
    pickle.dump(fcst, f)
