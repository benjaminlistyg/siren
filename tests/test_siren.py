"""
Basic tests for SIREN package.
"""

import pytest
import numpy as np
from siren import SIREN, AntColonyOptimizer, SLSQPOptimizer


def test_siren_initialization():
    """Test SIREN initialization."""
    siren = SIREN()
    assert siren is not None
    assert isinstance(siren.optimizer, AntColonyOptimizer)


def test_siren_with_different_optimizers():
    """Test SIREN with different optimizers."""
    siren_aco = SIREN.with_aco()
    assert isinstance(siren_aco.optimizer, AntColonyOptimizer)

    siren_slsqp = SIREN.with_slsqp()
    assert isinstance(siren_slsqp.optimizer, SLSQPOptimizer)


def test_get_dimension_indices():
    """Test dimension indices extraction."""
    siren = SIREN()
    dimensions = ['A', 'A', 'B', 'B', 'C']
    indices = siren.get_dimension_indices(dimensions)

    assert 'A' in indices
    assert 'B' in indices
    assert 'C' in indices
    assert indices['A'] == [0, 1]
    assert indices['B'] == [2, 3]
    assert indices['C'] == [4]


def test_reduce_scale_basic():
    """Test basic scale reduction."""
    items = [
        "I feel confident.",
        "I trust myself.",
        "I am capable.",
        "I have support.",
        "I can rely on others.",
        "People help me."
    ]
    dimensions = ['A', 'A', 'A', 'B', 'B', 'B']

    siren = SIREN()
    result, sim_matrix = siren.reduce_scale(
        items,
        dimensions,
        items_per_dim=2,
        suppress_details=True,
        n_tries=2
    )

    assert 'A' in result
    assert 'B' in result
    assert len(result['A']['items']) == 2
    assert len(result['B']['items']) == 2
    assert sim_matrix.shape == (6, 6)


def test_reduce_scale_variable_items():
    """Test scale reduction with variable items per dimension."""
    items = [
        "Item A1", "Item A2", "Item A3",
        "Item B1", "Item B2", "Item B3"
    ]
    dimensions = ['A', 'A', 'A', 'B', 'B', 'B']
    items_per_dim = {'A': 2, 'B': 1}

    siren = SIREN()
    result, _ = siren.reduce_scale(
        items,
        dimensions,
        items_per_dim=items_per_dim,
        suppress_details=True,
        n_tries=2
    )

    assert len(result['A']['items']) == 2
    assert len(result['B']['items']) == 1


def test_reduce_scale_invalid_count():
    """Test error handling when requesting too many items."""
    items = ["Item 1", "Item 2"]
    dimensions = ['A', 'A']

    siren = SIREN()

    with pytest.raises(ValueError):
        siren.reduce_scale(
            items,
            dimensions,
            items_per_dim=5,  # More than available
            suppress_details=True,
            n_tries=1
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
