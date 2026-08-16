from importlib.metadata import version

__version__ = version("mlforecast")
__all__ = [
    "MLForecast",
    "PolarsTargetEncoder",
    "PolarsCountEncoder",
    "PolarsOneHotEncoder",
    "PolarsOrdinalEncoder",
]
from mlforecast.forecast import MLForecast
from mlforecast.feature_encoders import (
    PolarsTargetEncoder,
    PolarsCountEncoder,
    PolarsOneHotEncoder,
    PolarsOrdinalEncoder,
)
