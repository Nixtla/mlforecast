from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mlforecast")
except PackageNotFoundError:
    __version__ = "unknown"
__all__ = ["MLForecast", "PerformanceEvaluator", "ParetoFrontier"]
from mlforecast.forecast import MLForecast
from mlforecast.evaluation import PerformanceEvaluator, ParetoFrontier
