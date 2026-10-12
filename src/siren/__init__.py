"""
SIREN - Semantic Item Reduction Engine

Automated psychometric scale reduction using semantic similarity and
optimization algorithms.
"""

from .core import SIREN
from .optimizers import (
    OptimizationMethod,
    AntColonyOptimizer,
    SLSQPOptimizer,
    GeneticAlgorithmOptimizer,
    SimulatedAnnealingOptimizer,
)
from .selection import (
    select_centroid,
    select_greedy_coverage,
    select_random_stratified,
)
from .diagnostics import (
    coverage_score,
    coherence_score,
    between_score,
    coverage_report,
    coverage_percentile,
    redundancy,
    fidelity_report,
)
from .plots import plot_item_map, plot_similarity_heatmap

__version__ = "0.3.1"
__author__ = "Benjamin Listyg, Brennah V. Ross"
__all__ = [
    "SIREN",
    "OptimizationMethod",
    "AntColonyOptimizer",
    "SLSQPOptimizer",
    "GeneticAlgorithmOptimizer",
    "SimulatedAnnealingOptimizer",
    "select_centroid",
    "select_greedy_coverage",
    "select_random_stratified",
    "coverage_score",
    "coherence_score",
    "between_score",
    "coverage_report",
    "coverage_percentile",
    "redundancy",
    "fidelity_report",
    "plot_item_map",
    "plot_similarity_heatmap",
]
