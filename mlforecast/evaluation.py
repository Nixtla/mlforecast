import pandas as pd
import numpy as np
from typing import List, Callable, Dict, Optional, Union
import utilsforecast.processing as ufp
from utilsforecast.compat import DataFrame

class PerformanceEvaluator:
    """Evaluates the performance of multiple models on cross-validation results."""
    
    def __init__(
        self,
        metrics: List[Callable],
        id_col: str = "unique_id",
        target_col: str = "y",
    ):
        """
        Args:
            metrics (List[Callable]): List of metric functions from utilsforecast.losses or similar.
            id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
            target_col (str): Column that contains the target. Defaults to 'y'.
        """
        self.metrics = metrics
        self.id_col = id_col
        self.target_col = target_col

    def evaluate(self, cv_results: DataFrame) -> pd.DataFrame:
        """Computes all metrics for each model in cv_results.
        
        Args:
            cv_results (DataFrame): Output from MLForecast.cross_validation.
            
        Returns:
            pd.DataFrame: Performance metrics per model, indexed by model name.
        """
        # Identify model columns (those not in the metadata)
        metadata_cols = {self.id_col, "ds", "cutoff", self.target_col}
        model_names = [c for c in cv_results.columns if c not in metadata_cols]
        
        results = {}
        for metric in self.metrics:
            metric_name = metric.__name__
            # Most utilsforecast.losses return a DataFrame with id_col and one column per model
            # We average across series. 
            # Note: if there are multiple cutoffs, we should ideally handle them.
            # utilsforecast.losses usually handle the full DataFrame if (id, ds) is unique.
            # In CV results, (id, ds, cutoff) is unique. 
            # We can group by cutoff to ensure (id, ds) uniqueness for the metric call if needed, 
            # but many metrics work fine on the flat results if we don't care about the temporal structure.
            
            # For simplicity and consistency with existing patterns in mlforecast:
            # We average the metric over all series and cutoffs.
            try:
                m_res = metric(
                    cv_results,
                    models=model_names,
                    id_col=self.id_col,
                    target_col=self.target_col,
                )
                # m_res is (n_series * n_windows, n_models + 1)
                # We want mean per model
                results[metric_name] = m_res[model_names].mean()
            except Exception as e:
                # Handle cases where the metric might fail or have a different API
                # Some metrics might not support the 'models' argument or might return a scalar
                results[metric_name] = pd.Series(
                    [np.nan] * len(model_names), index=model_names
                )
                print(f"Error computing metric {metric_name}: {e}")

        performance_df = pd.DataFrame(results)
        performance_df.index.name = "model"
        return performance_df

class ParetoFrontier:
    """Utilities for Pareto frontier analysis."""
    
    @staticmethod
    def is_dominated(candidate: np.ndarray, others: np.ndarray, directions: np.ndarray) -> bool:
        """Checks if a candidate solution is dominated by any of the others.
        
        Args:
            candidate (np.ndarray): Metric values for the candidate.
            others (np.ndarray): Metric values for other models.
            directions (np.ndarray): 1 for minimization, -1 for maximization.
        """
        # A solution B dominates A if B is at least as good as A in all objectives
        # AND strictly better in at least one objective.
        
        # Adjust for direction (multiply by 1 for min, -1 for max so it's always 'less is better')
        c = candidate * directions
        o = others * directions
        
        # d is True if others[i] <= candidate in all metrics
        better_or_equal = np.all(o <= c, axis=1)
        # s is True if others[i] < candidate in at least one metric
        strictly_better = np.any(o < c, axis=1)
        
        return np.any(better_or_equal & strictly_better)

    @classmethod
    def find_non_dominated(
        self, 
        performance_df: pd.DataFrame, 
        metrics: Optional[List[str]] = None,
        maximization: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Returns the non-dominated models (Pareto frontier).
        
        Args:
            performance_df (pd.DataFrame): Output from PerformanceEvaluator.evaluate.
            metrics (List[str], optional): Metrics to consider. Defaults to all.
            maximization (List[str], optional): Metrics where 'more is better'.
        """
        if metrics is None:
            metrics = performance_df.columns.tolist()
        
        data = performance_df[metrics].values
        directions = np.ones(len(metrics))
        if maximization:
            for i, m in enumerate(metrics):
                if m in maximization:
                    directions[i] = -1
        
        is_paretto = []
        for i in range(len(data)):
            others = np.delete(data, i, axis=0)
            if len(others) == 0:
                is_paretto.append(True)
                continue
            dominated = self.is_dominated(data[i], others, directions)
            is_paretto.append(not dominated)
            
        return performance_df[is_paretto]

    @staticmethod
    def plot_pareto_2d(
        performance_df: pd.DataFrame,
        metric_x: str,
        metric_y: str,
        maximize_x: bool = False,
        maximize_y: bool = False,
        show_dominated: bool = True,
        title: str = "Pareto Frontier"
    ):
        """Plots the 2D Pareto frontier."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting.")
            return

        pareto_df = ParetoFrontier.find_non_dominated(
            performance_df, 
            metrics=[metric_x, metric_y],
            maximization=([metric_x] if maximize_x else []) + ([metric_y] if maximize_y else [])
        )
        
        plt.figure(figsize=(10, 6))
        
        if show_dominated:
            plt.scatter(
                performance_df[metric_x], 
                performance_df[metric_y], 
                color='grey', alpha=0.5, label='Dominated'
            )
            for idx, row in performance_df.iterrows():
                plt.annotate(idx, (row[metric_x], row[metric_y]), alpha=0.7)
                
        plt.scatter(
            pareto_df[metric_x], 
            pareto_df[metric_y], 
            color='red', s=100, label='Pareto Optimal'
        )
        
        # Sort pareto points for a nice line
        pareto_sorted = pareto_df.sort_values(metric_x)
        plt.plot(pareto_sorted[metric_x], pareto_sorted[metric_y], 'r--', alpha=0.5)
        
        plt.xlabel(metric_x)
        plt.ylabel(metric_y)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt
