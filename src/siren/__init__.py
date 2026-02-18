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

__version__ = "0.1.0"
__author__ = "SIREN Contributors"
__all__ = [
    "SIREN",
    "OptimizationMethod",
    "AntColonyOptimizer",
    "SLSQPOptimizer",
    "GeneticAlgorithmOptimizer",
    "SimulatedAnnealingOptimizer",
]
