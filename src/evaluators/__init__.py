"""
Evaluators Module
=================
"""

from .metrics import (
    Evaluator,
    CrossValidationEvaluator,
    compute_statistical_significance
)

__all__ = [
    'Evaluator',
    'CrossValidationEvaluator',
    'compute_statistical_significance'
]
