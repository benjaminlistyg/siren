"""
Optional plotting helpers (require ``matplotlib``; install with
``pip install siren-scale[plots]``).

- :func:`plot_item_map` -- 2-D map of items (PCA or MDS) colored by
  dimension, with retained items highlighted.
- :func:`plot_similarity_heatmap` -- item-by-item similarity matrix grouped
  by dimension, with retained items marked.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .diagnostics import _as_selection, _flat, dimension_indices

Selection = Union[Dict[str, List[int]], Dict[str, Dict]]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "Plotting requires matplotlib. Install it with "
            "`pip install matplotlib` or `pip install siren-scale[plots]`."
        ) from exc
    return plt


def _coords(embeddings: Optional[np.ndarray], similarity_matrix: Optional[np.ndarray],
            method: str, seed: Optional[int]) -> np.ndarray:
    if method == "pca":
        if embeddings is None:
            raise ValueError("method='pca' requires embeddings; pass similarity_matrix "
                             "with method='mds' instead")
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(np.asarray(embeddings))
    if method == "mds":
        from sklearn.manifold import MDS
        if similarity_matrix is None:
            if embeddings is None:
                raise ValueError("method='mds' requires embeddings or similarity_matrix")
            E = np.asarray(embeddings)
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
            similarity_matrix = E @ E.T
        D = 1.0 - np.abs(np.asarray(similarity_matrix))
        np.fill_diagonal(D, 0.0)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
                  n_init=4, normalized_stress="auto")
        return mds.fit_transform(D)
    raise ValueError("method must be 'pca' or 'mds'")


def plot_item_map(embeddings: Optional[np.ndarray] = None,
                  similarity_matrix: Optional[np.ndarray] = None,
                  dimension_labels: Optional[Sequence[str]] = None,
                  selected: Optional[Selection] = None,
                  items: Optional[Sequence[str]] = None,
                  method: str = "pca",
                  annotate: bool = False,
                  seed: Optional[int] = 0,
                  ax=None):
    """
    Draw a 2-D semantic map of the items.

    Args:
        embeddings: Item embeddings (n x d). Required for ``method='pca'``.
        similarity_matrix: Item similarity matrix (n x n). Used for
            ``method='mds'`` (distance = 1 - |similarity|).
        dimension_labels: Optional dimension label per item (colors points).
        selected: Optional selection ({dim: [indices]} or reduce_scale()
            output); retained items are drawn larger with a black edge.
        items: Optional item texts; shown as labels when ``annotate=True``.
        method: ``'pca'`` (default) or ``'mds'``.
        annotate: Write the item index (or text) next to each point.
        seed: Random state for the projection.
        ax: Optional matplotlib Axes to draw on.

    Returns:
        The matplotlib Axes.
    """
    plt = _require_matplotlib()
    xy = _coords(embeddings, similarity_matrix, method, seed)
    n = xy.shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    kept = set(_flat(_as_selection(selected))) if selected is not None else set()
    labels = list(dimension_labels) if dimension_labels is not None else ["items"] * n
    for dim, idx in dimension_indices(labels).items():
        idx = np.array(idx)
        drop = np.array([i for i in idx if i not in kept], dtype=int)
        keep = np.array([i for i in idx if i in kept], dtype=int)
        base = ax.scatter(xy[drop, 0], xy[drop, 1], s=30, alpha=0.5, label=str(dim))
        if len(keep):
            ax.scatter(xy[keep, 0], xy[keep, 1], s=110, edgecolors="black", linewidths=1.2,
                       color=base.get_facecolor()[0])
    if annotate:
        for i in range(n):
            txt = (items[i][:40] if items is not None else str(i))
            ax.annotate(txt, (xy[i, 0], xy[i, 1]), fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    title = "Item map"
    if selected is not None:
        title += f" ({len(kept)} retained of {n}, outlined)"
    ax.set_title(title)
    if dimension_labels is not None:
        ax.legend(title="Dimension", fontsize=8, title_fontsize=9)
    return ax


def plot_similarity_heatmap(similarity_matrix: np.ndarray,
                            dimension_labels: Optional[Sequence[str]] = None,
                            selected: Optional[Selection] = None,
                            items: Optional[Sequence[str]] = None,
                            absolute: bool = True,
                            ax=None):
    """
    Draw the item-by-item similarity matrix, grouped by dimension.

    Args:
        similarity_matrix: Item similarity matrix (n x n).
        dimension_labels: Optional labels; items are reordered so dimensions
            are contiguous and separated by lines.
        selected: Optional selection; retained items are marked on the axes.
        items: Optional item texts for tick labels (truncated).
        absolute: Plot |similarity| (default) or signed similarity.
        ax: Optional matplotlib Axes.

    Returns:
        The matplotlib Axes.
    """
    plt = _require_matplotlib()
    S = np.asarray(similarity_matrix, dtype=float)
    if absolute:
        S = np.abs(S)
    n = S.shape[0]
    if dimension_labels is not None:
        groups = dimension_indices(list(dimension_labels))
        order = [i for idx in groups.values() for i in idx]
        bounds = np.cumsum([len(idx) for idx in groups.values()])[:-1]
    else:
        order, bounds = list(range(n)), []
    S = S[np.ix_(order, order)]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(S, cmap="viridis" if absolute else "coolwarm",
                   vmin=0 if absolute else -1, vmax=1)
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=1)
        ax.axvline(b - 0.5, color="white", lw=1)
    kept = set(_flat(_as_selection(selected))) if selected is not None else set()
    ticklabels = []
    for i in order:
        base = (items[i][:25] if items is not None else str(i))
        ticklabels.append(("* " if i in kept else "") + base)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ticklabels, rotation=90, fontsize=6)
    ax.set_yticklabels(ticklabels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="|cosine similarity|" if absolute else "cosine similarity")
    ax.set_title("Item similarity" + (" (* = retained)" if kept else ""))
    return ax
