"""
Evaluation Metrics and Evaluator
================================
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)
import pandas as pd


class Evaluator:
    """Comprehensive evaluation for multi-task learning."""
    
    # Task configurations
    TASK_CONFIGS = {
        'binary': {
            'type': 'classification',
            'num_classes': 2,
            'class_names': ['Healthy', 'Pathological']
        },
        'coarse': {
            'type': 'classification',
            'num_classes': 3,
            'class_names': ['Healthy', 'Neuro', 'Ortho']
        },
        'fine': {
            'type': 'classification',
            'num_classes': 8,
            'class_names': ['HS', 'CVA', 'PD', 'CIPN', 'RIL', 'KOA', 'HOA', 'ACL']
        },
        'vga_class': {
            'type': 'classification',
            'num_classes': 5,
            'class_names': ['Normal', 'Mild', 'Moderate', 'Significant', 'Severe']
        },
        'gender': {
            'type': 'classification',
            'num_classes': 2,
            'class_names': ['Male', 'Female']
        },
        'neuro_fine': {
            'type': 'classification',
            'num_classes': 4,
            'class_names': ['CVA', 'PD', 'CIPN', 'RIL']
        },
        'regression': {
            'type': 'regression'
        },
        'vga_regression': {
            'type': 'regression'
        },
        'age': {
            'type': 'regression'
        },
        'tug': {
            'type': 'regression'
        }
    }
    
    def __init__(self):
        self.results = {}
        
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task_name: str = 'classification'
    ) -> Dict[str, float]:
        """Evaluate classification task."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
        }
        
        # Per-class F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, f1 in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1
            
        # AUC (if probabilities available)
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    # Binary classification
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class
                    metrics['auc_ovr'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average='macro'
                    )
            except Exception:
                pass
                
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate regression task."""
        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
        if len(y_true) == 0:
            return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
            
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def evaluate_all_tasks(
        self,
        outputs: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        probas: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all tasks, with automatic masking for missing labels (-1)."""
        results = {}
        
        for task_name, config in self.TASK_CONFIGS.items():
            if task_name not in outputs or task_name not in targets:
                continue
                
            y_pred = outputs[task_name]
            y_true = targets[task_name]
            
            if config['type'] == 'classification':
                # Mask out missing labels (labeled as -1)
                mask = y_true >= 0
                if not mask.any():
                    continue
                y_true_valid = y_true[mask]
                y_pred_valid = y_pred[mask]
                y_proba = None
                if probas and task_name in probas:
                    y_proba = probas[task_name][mask]
                results[task_name] = self.evaluate_classification(
                    y_true_valid, y_pred_valid, y_proba, task_name
                )
            else:
                mask = y_true >= 0  # Valid regression targets
                results[task_name] = self.evaluate_regression(y_true, y_pred, mask)
                
        return results
    
    def format_results(
        self,
        results: Dict[str, Dict[str, float]],
        include_cm: bool = False
    ) -> str:
        """Format results as string."""
        lines = []
        
        for task_name, metrics in results.items():
            lines.append(f"\n{task_name.upper()}")
            lines.append("-" * 40)
            
            for metric_name, value in metrics.items():
                if metric_name == 'confusion_matrix':
                    if include_cm:
                        lines.append(f"Confusion Matrix:\n{value}")
                elif isinstance(value, float):
                    lines.append(f"  {metric_name}: {value:.4f}")
                    
        return "\n".join(lines)
    
    def results_to_dataframe(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        
        for task_name, metrics in results.items():
            row = {'task': task_name}
            for metric_name, value in metrics.items():
                if metric_name != 'confusion_matrix' and isinstance(value, (int, float)):
                    row[metric_name] = value
            rows.append(row)
            
        return pd.DataFrame(rows)


class CrossValidationEvaluator:
    """Evaluate with cross-validation."""
    
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.fold_results = []
        
    def add_fold_result(self, results: Dict[str, Dict[str, float]]):
        """Add results from one fold."""
        self.fold_results.append(results)
        
    def get_summary(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Get mean and std across folds."""
        if not self.fold_results:
            return {}
            
        # Collect all metrics
        all_metrics = {}
        
        for fold_result in self.fold_results:
            for task, metrics in fold_result.items():
                if task not in all_metrics:
                    all_metrics[task] = {}
                    
                for metric, value in metrics.items():
                    if metric == 'confusion_matrix':
                        continue
                    if not isinstance(value, (int, float)):
                        continue
                        
                    if metric not in all_metrics[task]:
                        all_metrics[task][metric] = []
                    all_metrics[task][metric].append(value)
                    
        # Compute mean and std
        summary = {}
        for task, metrics in all_metrics.items():
            summary[task] = {}
            for metric, values in metrics.items():
                summary[task][metric] = (np.mean(values), np.std(values))
                
        return summary
    
    def format_summary(self) -> str:
        """Format summary as string."""
        summary = self.get_summary()
        
        lines = [f"Cross-Validation Results ({self.n_folds} folds)"]
        lines.append("=" * 50)
        
        for task, metrics in summary.items():
            lines.append(f"\n{task.upper()}")
            lines.append("-" * 40)
            
            for metric, (mean, std) in metrics.items():
                lines.append(f"  {metric}: {mean:.4f} +/- {std:.4f}")
                
        return "\n".join(lines)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert summary to DataFrame."""
        summary = self.get_summary()
        
        rows = []
        for task, metrics in summary.items():
            row = {'task': task}
            for metric, (mean, std) in metrics.items():
                row[f'{metric}_mean'] = mean
                row[f'{metric}_std'] = std
            rows.append(row)
            
        return pd.DataFrame(rows)


def compute_statistical_significance(
    results1: List[float],
    results2: List[float],
    test: str = 'ttest'
) -> Tuple[float, float]:
    """
    Compute statistical significance between two methods.
    
    Returns:
        statistic, p_value
    """
    from scipy import stats
    
    results1 = np.array(results1)
    results2 = np.array(results2)
    
    if test == 'ttest':
        # Paired t-test
        statistic, p_value = stats.ttest_rel(results1, results2)
    elif test == 'wilcoxon':
        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(results1, results2)
    else:
        raise ValueError(f"Unknown test: {test}")
        
    return statistic, p_value


if __name__ == "__main__":
    # Test evaluator
    np.random.seed(42)
    
    # Simulate predictions
    n_samples = 100
    
    outputs = {
        'binary': np.random.randint(0, 2, n_samples),
        'coarse': np.random.randint(0, 3, n_samples),
        'fine': np.random.randint(0, 8, n_samples),
        'regression': np.random.rand(n_samples)
    }
    
    targets = {
        'binary': np.random.randint(0, 2, n_samples),
        'coarse': np.random.randint(0, 3, n_samples),
        'fine': np.random.randint(0, 8, n_samples),
        'regression': np.random.rand(n_samples)
    }
    
    evaluator = Evaluator()
    results = evaluator.evaluate_all_tasks(outputs, targets)
    
    print(evaluator.format_results(results))
