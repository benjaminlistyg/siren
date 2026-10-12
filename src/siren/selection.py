"""
Alternative item selectors.

These are deterministic, single-pass selectors that operate on a precomputed
item-by-item similarity matrix. They are provided (a) as fast baselines
against which the optimizer-based selection in :meth:`siren.SIREN.reduce_scale`
can be compared and (b) because they reproduce the selection logic used by
clustering-based tools (nearest-centroid / greedy-medoid selection).

All functions take a similarity matrix (``numpy.ndarray``, shape n x n),
a mapping of dimension name -> list of item indices, and the target number
of items per dimension (int or dict), and return a mapping of dimension
name -> list of selected item indices.
"""

from typing import Dict, List, Optional, Union

import numpy as np

ItemsPerDim = Union[int, Dict[str, int]]


def _targets(dim_indices: Dict[str, List[int]], items_per_dim: ItemsPerDim) -> Dict[str, int]:
    """Expand ``items_per_dim`` to a per-dimension dict and validate counts."""
    if isinstance(items_per_dim, int):
        targets = {dim: items_per_dim for dim in dim_indices}
    else:
        missing = set(dim_indices) - set(items_per_dim)
        if missing:
            raise ValueError(f"items_per_dim is missing dimensions: {sorted(missing)}")
        targets = dict(items_per_dim)
    for dim, k in targets.items():
        if k < 1:
            raise ValueError(f"Target for dimension {dim!r} must be >= 1, got {k}")
        if k > len(dim_indices[dim]):
            raise ValueError(
                f"Requested {k} items for dimension {dim} "
                f"but only {len(dim_indices[dim])} available"
            )
    return targets


def select_centroid(similarity_matrix: np.ndarray,
                    dim_indices: Dict[str, List[int]],
                    items_per_dim: ItemsPerDim) -> Dict[str, List[int]]:
    """
    Nearest-centroid ("anchor") selection.

    Within each dimension, items are ranked by their mean absolute similarity
    to the other items of that dimension (a leave-one-out proxy for distance
    to the dimension centroid) and the top ``k`` are kept. This reproduces the
    selection rule of centroid-based clustering tools. It tends to retain
    near-duplicate items; see :func:`select_greedy_coverage` for a
    redundancy-aware alternative.

    Args:
        similarity_matrix: Pairwise item similarity matrix (n x n).
        dim_indices: Mapping of dimension name -> item indices.
        items_per_dim: Target number of items per dimension.

    Returns:
        Mapping of dimension name -> selected item indices (sorted by rank).
    """
    S = np.asarray(similarity_matrix)
    targets = _targets(dim_indices, items_per_dim)
    selected = {}
    for dim, indices in dim_indices.items():
        idx = np.array(indices)
        if len(idx) == 1:
            selected[dim] = [int(idx[0])]
            continue
        sub = np.abs(S[np.ix_(idx, idx)])
        np.fill_diagonal(sub, np.nan)
        centrality = np.nanmean(sub, axis=1)
        order = np.argsort(-centrality, kind="stable")
        selected[dim] = [int(idx[i]) for i in order[:targets[dim]]]
    return selected


def select_greedy_coverage(similarity_matrix: np.ndarray,
                           dim_indices: Dict[str, List[int]],
                           items_per_dim: ItemsPerDim) -> Dict[str, List[int]]:
    """
    Greedy facility-location ("medoid") selection.

    Within each dimension, items are added one at a time, each time choosing
    the item that most increases *coverage*: the mean, over all items of the
    dimension, of the maximum absolute similarity to any selected item. This
    is the standard greedy approximation to the facility-location objective
    optimized by ``objective='coverage'`` in :meth:`siren.SIREN.reduce_scale`,
    without the between-dimension penalty and without stochastic search. It
    is deterministic and runs in O(n^2 k) per dimension.

    Args:
        similarity_matrix: Pairwise item similarity matrix (n x n).
        dim_indices: Mapping of dimension name -> item indices.
        items_per_dim: Target number of items per dimension.

    Returns:
        Mapping of dimension name -> selected item indices (in order added).
    """
    S = np.asarray(similarity_matrix)
    targets = _targets(dim_indices, items_per_dim)
    selected = {}
    for dim, indices in dim_indices.items():
        idx = np.array(indices)
        sub = np.abs(S[np.ix_(idx, idx)])
        k = targets[dim]
        chosen: List[int] = []
        best_cover = np.zeros(len(idx))  # current max similarity to chosen set
        for _ in range(k):
            best_gain, best_j = -np.inf, None
            for j in range(len(idx)):
                if j in chosen:
                    continue
                gain = np.mean(np.maximum(best_cover, sub[:, j]))
                if gain > best_gain:
                    best_gain, best_j = gain, j
            chosen.append(best_j)
            best_cover = np.maximum(best_cover, sub[:, best_j])
        selected[dim] = [int(idx[j]) for j in chosen]
    return selected


def select_random_stratified(dim_indices: Dict[str, List[int]],
                             items_per_dim: ItemsPerDim,
                             rng: Optional[np.random.Generator] = None) -> Dict[str, List[int]]:
    """
    Random dimension-stratified selection.

    Draws ``k`` items uniformly at random from each dimension. Used to build
    null distributions for :func:`siren.diagnostics.coverage_percentile`.

    Args:
        dim_indices: Mapping of dimension name -> item indices.
        items_per_dim: Target number of items per dimension.
        rng: Optional ``numpy.random.Generator`` for reproducibility.

    Returns:
        Mapping of dimension name -> selected item indices.
    """
    rng = rng or np.random.default_rng()
    targets = _targets(dim_indices, items_per_dim)
    return {dim: [int(i) for i in rng.choice(indices, size=targets[dim], replace=False)]
            for dim, indices in dim_indices.items()}


def selection_to_vector(selected: Dict[str, List[int]], n_items: int) -> np.ndarray:
    """Convert a dimension -> indices mapping to a binary solution vector."""
    x = np.zeros(n_items)
    for indices in selected.values():
        x[list(indices)] = 1
    return x
