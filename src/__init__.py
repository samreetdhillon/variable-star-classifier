"""
Variable Star Classifier Package

A package for classifying variable stars and computing their physical properties.
"""

from .classifier import VariableStarClassifier
from .stellar_properties import (
    compute_stellar_properties,
    print_summary_statistics,
    save_pulsators_data
)
from .visualization import create_all_plots

__all__ = [
    'VariableStarClassifier',
    'compute_stellar_properties',
    'print_summary_statistics',
    'save_pulsators_data',
    'create_all_plots'
]
