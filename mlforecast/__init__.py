from importlib.metadata import version

__version__ = version("mlforecast")
__all__ = [
    "MLForecast",
    "PolarsTargetEncoder",
    "PolarsFrequencyEncoder",
    "PolarsOneHotEncoder",
    "PolarsOrdinalEncoder",
]
from mlforecast.forecast import MLForecast
from mlforecast.feature_encoders import (
    PolarsTargetEncoder,
    PolarsFrequencyEncoder,
    PolarsOneHotEncoder,
    PolarsOrdinalEncoder,
)
