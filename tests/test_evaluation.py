import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from utilsforecast.losses import mae, rmse

from mlforecast import MLForecast
from mlforecast.evaluation import ParetoFrontier, PerformanceEvaluator
from mlforecast.utils import generate_daily_series

def test_pareto_frontier_logic():
    # Performance data: lower is better for both metrics
    data = {
        'rmse': [10, 12, 8, 11, 9],
        'mae':  [5,  4,  6,  4.5, 5.5]
    }
    df = pd.DataFrame(data, index=['M1', 'M2', 'M3', 'M4', 'M5'])
    
    # M1: (10, 5)
    # M2: (12, 4) -> Non-dominated (best MAE)
    # M3: (8, 6) -> Non-dominated (best RMSE)
    # M4: (11, 4.5) -> Dominated by M2 (M2 has better RMSE 12 vs 11? No wait, M2 has 12, M4 has 11. M4 better RMSE, M2 better MAE. both non-dominated?)
    # Let's re-evaluate:
    # M1 (10, 5) vs M2 (12, 4): No domination.
    # M1 (10, 5) vs M3 (8, 6): No domination.
    # M1 (10, 5) vs M5 (9, 5.5): No domination.
    
    # Let's use simpler points:
    # A: (1, 5)
    # B: (2, 4)
    # C: (3, 3)
    # D: (2, 5) -> Dominated by A (1 < 2, 5=5)
    # E: (4, 2)
    # F: (5, 5) -> Dominated by all
    
    data = {
        'm1': [1, 2, 3, 2, 4, 5],
        'm2': [5, 4, 3, 5, 2, 5]
    }
    df = pd.DataFrame(data, index=['A', 'B', 'C', 'D', 'E', 'F'])
    
    pareto = ParetoFrontier.find_non_dominated(df)
    
    # Non-dominated should be A, B, C, E
    assert set(pareto.index) == {'A', 'B', 'C', 'E'}
    assert 'D' not in pareto.index
    assert 'F' not in pareto.index

def test_pareto_frontier_maximization():
    # m1: lower is better, m2: higher is better
    data = {
        'm1': [10, 10, 11],
        'm2': [100, 90, 100]
    }
    df = pd.DataFrame(data, index=['A', 'B', 'C'])
    # A: (10, 100)
    # B: (10, 90) -> Dominated by A (10=10, 90 < 100)
    # C: (11, 100) -> Dominated by A (11 > 10, 100=100)
    
    pareto = ParetoFrontier.find_non_dominated(df, maximization=['m2'])
    assert set(pareto.index) == {'A'}

def test_performance_evaluator_integration():
    series = generate_daily_series(5, n_static_features=2, equal_ends=True)
    
    models = {
        'm1': LinearRegression(),
        'm2': LinearRegression(), # Same model just to check behavior
    }
    
    fcst = MLForecast(
        models=models,
        freq='D',
        lags=[1, 7],
    )
    
    cv_results = fcst.cross_validation(
        series,
        n_windows=2,
        h=7,
        static_features=['static_0', 'static_1']
    )
    
    # Test evaluate method on MLForecast
    performance = fcst.evaluate(cv_results, metrics=[rmse, mae])
    
    assert isinstance(performance, pd.DataFrame)
    assert 'rmse' in performance.columns
    assert 'mae' in performance.columns
    assert 'm1' in performance.index
    assert 'm2' in performance.index
    
    # Since m1 and m2 are identical LinearRegressions, their performance should be very similar/same
    np.testing.assert_allclose(performance.loc['m1', 'rmse'], performance.loc['m2', 'rmse'])

def test_pareto_plot_smoke(tmp_path):
    # Just check it doesn't crash
    data = {'rmse': [1, 2, 3], 'mae': [3, 2, 1]}
    df = pd.DataFrame(data, index=['M1', 'M2', 'M3'])
    
    plt = ParetoFrontier.plot_pareto_2d(df, 'rmse', 'mae')
    if plt:
        # Save to temp file to verify it works
        plt.savefig(tmp_path / "pareto.png")
        plt.close()
