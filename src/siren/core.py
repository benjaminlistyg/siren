"""
SIREN - Semantic Item Reduction Engine
Main class for psychometric scale reduction.
"""

import itertools
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .optimizers import (
    OptimizationMethod,
    AntColonyOptimizer,
    SLSQPOptimizer,
    GeneticAlgorithmOptimizer,
    SimulatedAnnealingOptimizer
)
from .selection import select_centroid, select_greedy_coverage, selection_to_vector
from . import diagnostics as _diag

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'all-MiniLM-L12-v2'
SELECTION_METHODS = ('optimize', 'centroid', 'greedy')


class SIREN:
    """
    Semantic Item Reduction Engine - Main class for scale reduction.

    SIREN uses semantic similarity and combinatorial optimization to reduce
    psychometric scales while maintaining their structure and content.

    The sentence-transformer model is loaded lazily on first use, so an
    instance created with precomputed ``embeddings`` or a ``similarity_matrix``
    in mind never loads a model.
    """

    def __init__(self, model_name: Optional[str] = DEFAULT_MODEL,
                 optimizer: Optional[OptimizationMethod] = None,
                 seed: Optional[int] = None,
                 optimizer_options: Optional[Dict] = None,
                 model=None):
        """
        Initialize SIREN with a sentence transformer model and optimizer.

        Args:
            model_name: Name of the sentence-transformer model to use. Any
                model loadable by ``sentence_transformers.SentenceTransformer``
                is accepted (e.g. 'all-mpnet-base-v2',
                'Qwen/Qwen3-Embedding-0.6B'). ``None`` disables model loading;
                you must then pass ``embeddings`` or ``similarity_matrix`` to
                :meth:`reduce_scale`.
            optimizer: Optimization method instance (default: AntColonyOptimizer)
            seed: Default random seed for :meth:`reduce_scale` (can be
                overridden per call). ``None`` leaves the RNG unseeded.
            optimizer_options: Default options forwarded to the optimizer on
                every call, e.g. ``{'n_ants': 40, 'maxiter': 200}`` for ACO,
                ``{'popsize': 30}`` for the genetic algorithm, ``{'maxiter': 500}``
                for SLSQP / simulated annealing. Merged with (and overridden
                by) the ``optimizer_options`` argument of :meth:`reduce_scale`.
            model: An already-loaded embedding model exposing ``encode(texts)``.
                Takes precedence over ``model_name``.
        """
        self.model_name = model_name
        self._model = model
        # Default to ACO as it shows superior performance for scale reduction
        self.optimizer = optimizer or AntColonyOptimizer()
        self.seed = seed
        self.optimizer_options = dict(optimizer_options or {})
        # Objective mode used by objective_function; set per-call in reduce_scale
        self._objective_mode = 'coherence'
        # Populated by reduce_scale()
        self.last_similarity_: Optional[np.ndarray] = None
        self.last_embeddings_: Optional[np.ndarray] = None
        self.last_dimension_labels_: Optional[List[str]] = None
        self.last_selection_: Optional[Dict[str, List[int]]] = None
        if model is None and model_name is not None:
            logger.info(f"SIREN will use model: {model_name} (loaded on first use)")

    # ------------------------------------------------------------------
    # Model handling
    # ------------------------------------------------------------------

    @property
    def model(self):
        """The embedding model, loaded on first access."""
        if self._model is None:
            if self.model_name is None:
                raise ValueError(
                    "No embedding model configured (model_name=None). Pass "
                    "`embeddings` or `similarity_matrix` to reduce_scale(), or "
                    "construct SIREN with a model_name."
                )
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def set_optimizer(self, optimizer: OptimizationMethod):
        """
        Change the optimization method.

        Args:
            optimizer: New OptimizationMethod instance
        """
        self.optimizer = optimizer

    @classmethod
    def with_aco(cls, model_name: Optional[str] = DEFAULT_MODEL, **kwargs) -> 'SIREN':
        """Create SIREN with Ant Colony Optimization (default)."""
        return cls(model_name, AntColonyOptimizer(), **kwargs)

    @classmethod
    def with_slsqp(cls, model_name: Optional[str] = DEFAULT_MODEL, **kwargs) -> 'SIREN':
        """Create SIREN with SLSQP optimizer."""
        return cls(model_name, SLSQPOptimizer(), **kwargs)

    @classmethod
    def with_genetic_algorithm(cls, model_name: Optional[str] = DEFAULT_MODEL,
                               **kwargs) -> 'SIREN':
        """Create SIREN with genetic algorithm optimizer."""
        return cls(model_name, GeneticAlgorithmOptimizer(), **kwargs)

    @classmethod
    def with_simulated_annealing(cls, model_name: Optional[str] = DEFAULT_MODEL,
                                 **kwargs) -> 'SIREN':
        """Create SIREN with simulated annealing optimizer."""
        return cls(model_name, SimulatedAnnealingOptimizer(), **kwargs)

    # ------------------------------------------------------------------
    # Embeddings and similarity
    # ------------------------------------------------------------------

    def embed(self, items: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode item texts as unit-length embedding vectors.

        Args:
            items: Item texts.
            show_progress_bar: Show the encoder's progress bar.

        Returns:
            Array of shape (n_items, dim), L2-normalized rows.
        """
        emb = self.model.encode(list(items), convert_to_numpy=True,
                                show_progress_bar=show_progress_bar)
        emb = np.asarray(emb, dtype=float)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    @staticmethod
    def similarity_from_embeddings(embeddings: np.ndarray,
                                   reverse_scored: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        Cosine similarity matrix from embeddings, with optional sign flips.

        Args:
            embeddings: Array (n_items, dim); rows are normalized internally.
            reverse_scored: 0-indexed positions of reverse-keyed items. Their
                rows and columns are sign-flipped so that they align with their
                dimension; the diagonal is flipped twice and stays at 1.

        Returns:
            Similarity matrix clipped to [-1, 1].
        """
        E = np.asarray(embeddings, dtype=float)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        S = np.clip(E @ E.T, -1.0, 1.0)
        return SIREN._apply_reverse(S, reverse_scored)

    @staticmethod
    def _apply_reverse(S: np.ndarray, reverse_scored: Optional[Sequence[int]]) -> np.ndarray:
        S = np.array(S, dtype=float, copy=True)
        if reverse_scored:
            for idx in reverse_scored:
                S[idx, :] *= -1
                S[:, idx] *= -1
        return S

    def compute_similarity(self, items: Optional[Sequence[str]] = None,
                           embeddings: Optional[np.ndarray] = None,
                           reverse_scored: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        Compute (or derive) the item similarity matrix.

        Args:
            items: Item texts (embedded with the configured model) or
            embeddings: Precomputed embeddings (n_items, dim).
            reverse_scored: Positions of reverse-keyed items to sign-flip.

        Returns:
            Similarity matrix (n_items, n_items).
        """
        if embeddings is None:
            if items is None:
                raise ValueError("Provide items or embeddings")
            embeddings = self.embed(items)
        self.last_embeddings_ = np.asarray(embeddings, dtype=float)
        return self.similarity_from_embeddings(embeddings, reverse_scored)

    # ------------------------------------------------------------------
    # Structure discovery
    # ------------------------------------------------------------------

    def infer_dimensions(self, items: Optional[Sequence[str]] = None,
                         embeddings: Optional[np.ndarray] = None,
                         n_dimensions: Optional[int] = None,
                         max_dimensions: int = 10,
                         seed: Optional[int] = None,
                         prefix: str = 'D') -> List[str]:
        """
        Infer a dimensional structure from item text alone.

        Clusters item embeddings with k-means (cosine geometry, since rows are
        unit length). If ``n_dimensions`` is not given, the number of clusters
        is chosen by the silhouette criterion over 2..``max_dimensions``.
        Labels are renumbered in order of first appearance so that results
        are stable for a given seed.

        This is a convenience for scales without a theoretical key. Supplying
        the instrument's own dimension labels to :meth:`reduce_scale` is
        always preferable when they exist.

        Args:
            items: Item texts, or
            embeddings: Precomputed embeddings.
            n_dimensions: Number of clusters; ``None`` selects by silhouette.
            max_dimensions: Upper bound searched when ``n_dimensions`` is None.
            seed: Random state for k-means.
            prefix: Label prefix; labels are ``f"{prefix}{k}"``.

        Returns:
            One label per item.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        if embeddings is None:
            if items is None:
                raise ValueError("Provide items or embeddings")
            embeddings = self.embed(items)
        E = np.asarray(embeddings, dtype=float)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        n = E.shape[0]
        seed = self.seed if seed is None else seed

        def fit(k):
            km = KMeans(n_clusters=k, n_init=10, random_state=seed)
            return km.fit_predict(E)

        if n_dimensions is None:
            upper = min(max_dimensions, n - 1)
            if upper < 2:
                raise ValueError("Need at least 3 items to infer more than one dimension")
            best_k, best_s, best_lab = None, -np.inf, None
            for k in range(2, upper + 1):
                lab = fit(k)
                if len(set(lab)) < 2:
                    continue
                s = silhouette_score(E, lab, metric='cosine')
                if s > best_s:
                    best_k, best_s, best_lab = k, s, lab
            labels_raw = best_lab
            logger.info(f"infer_dimensions: chose k={best_k} (silhouette={best_s:.3f})")
        else:
            if n_dimensions < 1 or n_dimensions > n:
                raise ValueError("n_dimensions must be between 1 and the number of items")
            labels_raw = fit(n_dimensions) if n_dimensions > 1 else np.zeros(n, dtype=int)

        remap: Dict[int, str] = {}
        labels = []
        for lab in labels_raw:
            if lab not in remap:
                remap[lab] = f"{prefix}{len(remap) + 1}"
            labels.append(remap[lab])
        return labels

    # ------------------------------------------------------------------
    # Item vetting
    # ------------------------------------------------------------------

    def item_fit(self, candidates: Sequence[str], items: Sequence[str],
                 dimension_labels: Sequence[str],
                 embeddings: Optional[np.ndarray] = None,
                 candidate_embeddings: Optional[np.ndarray] = None,
                 redundancy_threshold: float = 0.8) -> pd.DataFrame:
        """
        Vet candidate items against an existing scale.

        For each candidate, reports its similarity to every dimension's
        centroid, the best-fitting dimension, the nearest existing item and
        its similarity, and whether the candidate duplicates an existing item.
        High centroid similarity with low nearest-item similarity suggests a
        gap-filler (on-construct, new content); high nearest-item similarity
        suggests redundancy.

        Args:
            candidates: Candidate item texts.
            items: Existing item texts.
            dimension_labels: Dimension label per existing item.
            embeddings: Optional precomputed embeddings of ``items``.
            candidate_embeddings: Optional precomputed embeddings of ``candidates``.
            redundancy_threshold: |similarity| to the nearest existing item at
                or above which the candidate is flagged as redundant.

        Returns:
            DataFrame with one row per candidate: ``candidate``,
            ``best_dimension``, ``best_dimension_similarity``, one
            ``sim_<dimension>`` column per dimension, ``nearest_index``,
            ``nearest_item``, ``nearest_similarity``, ``is_redundant``.
        """
        if len(items) != len(dimension_labels):
            raise ValueError("items and dimension_labels must have the same length")
        E = self.embed(items) if embeddings is None else np.asarray(embeddings, dtype=float)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        C = (self.embed(candidates) if candidate_embeddings is None
             else np.asarray(candidate_embeddings, dtype=float))
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        dim_indices = self.get_dimension_indices(list(dimension_labels))
        centroids = {}
        for dim, idx in dim_indices.items():
            c = E[idx].mean(axis=0)
            centroids[dim] = c / np.linalg.norm(c)
        sims_items = C @ E.T
        rows = []
        for ci, text in enumerate(candidates):
            row = {"candidate": text}
            dim_sims = {dim: float(C[ci] @ c) for dim, c in centroids.items()}
            best = max(dim_sims, key=dim_sims.get)
            row["best_dimension"] = best
            row["best_dimension_similarity"] = dim_sims[best]
            for dim, v in dim_sims.items():
                row[f"sim_{dim}"] = v
            j = int(np.argmax(np.abs(sims_items[ci])))
            row["nearest_index"] = j
            row["nearest_item"] = items[j]
            row["nearest_similarity"] = float(sims_items[ci, j])
            row["is_redundant"] = bool(abs(sims_items[ci, j]) >= redundancy_threshold)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def get_dimension_indices(self, dimension_labels: List[str]) -> Dict[str, List[int]]:
        """
        Get indices for each dimension.

        Args:
            dimension_labels: List of dimension labels for each item

        Returns:
            Dictionary mapping dimension names to lists of item indices
        """
        dim_indices = {}
        for i, dim in enumerate(dimension_labels):
            if dim not in dim_indices:
                dim_indices[dim] = []
            dim_indices[dim].append(i)
        return dim_indices

    def objective_function(self, x: np.ndarray, similarity_matrix: np.ndarray,
                          dim_indices: Dict[str, List[int]],
                          items_per_dim: Union[int, Dict[str, int]]) -> float:
        """
        Objective function for optimization.

        Two objective modes are supported (set via reduce_scale's `objective`
        argument):

        - 'coherence' (default): maximizes the mean pairwise similarity among
          selected items within each dimension. Undefined (contributes 0) when
          only one item is selected per dimension.
        - 'coverage': facility-location style content coverage. For each
          dimension, maximizes the mean over *non-selected* pool items of
          their maximum similarity to any selected item in that dimension.
          This rewards item subsets that are semantically representative of
          the full dimension pool (rather than mutually redundant) and is
          well-defined for one item per dimension.

        Both modes subtract mean between-dimension similarity among selected
        items and apply penalties for item count constraint violations.

        Args:
            x: Binary solution vector (1 = item selected, 0 = not selected)
            similarity_matrix: Pairwise cosine similarity matrix
            dim_indices: Dictionary mapping dimensions to item indices
            items_per_dim: Target number of items per dimension

        Returns:
            Objective function value (lower is better)
        """
        selected_indices = np.where(x > 0.5)[0]

        if len(selected_indices) == 0:
            return np.inf

        total_score = 0

        # For each dimension
        for dim, indices in dim_indices.items():
            dim_selected = np.intersect1d(selected_indices, indices)

            if len(dim_selected) == 0:
                return np.inf

            if self._objective_mode == 'coverage':
                # Within-dimension coverage (maximize): how well do the
                # selected items semantically cover the dropped pool items?
                pool_rest = np.setdiff1d(np.array(indices), dim_selected)
                if len(pool_rest) == 0:
                    within_score = 1.0
                else:
                    within_score = np.mean([
                        np.max([abs(similarity_matrix[p, s]) for s in dim_selected])
                        for p in pool_rest
                    ])
            else:
                # Within-dimension similarity (maximize)
                within_sims = []
                for i, j in itertools.combinations(dim_selected, 2):
                    sim = similarity_matrix[i, j]
                    within_sims.append(abs(sim))
                within_score = np.mean(within_sims) if within_sims else 0

            # Between-dimension similarity (minimize)
            between_sims = []
            other_selected = np.setdiff1d(selected_indices, dim_selected)
            for i in dim_selected:
                for j in other_selected:
                    between_sims.append(abs(similarity_matrix[i, j]))
            between_score = np.mean(between_sims) if between_sims else 0

            # Combine scores
            dim_score = 2.0 * within_score - between_score
            total_score += dim_score

        # Penalties for constraint violations
        penalty = 0
        for dim, indices in dim_indices.items():
            target_count = items_per_dim[dim] if isinstance(items_per_dim, dict) else items_per_dim
            selected_count = len(np.intersect1d(selected_indices, indices))
            if selected_count != target_count:
                penalty += 1000 * abs(selected_count - target_count)

        return -(total_score - penalty)

    def reduce_scale(self, items: List[str], dimension_labels: Optional[List[str]] = None,
                    items_per_dim: Union[int, Dict[str, int]] = 2,
                    suppress_details: bool = False,
                    n_tries: int = 5,
                    reverse_scored: Optional[List[int]] = None,
                    objective: str = 'coherence',
                    embeddings: Optional[np.ndarray] = None,
                    similarity_matrix: Optional[np.ndarray] = None,
                    method: str = 'optimize',
                    seed: Optional[int] = None,
                    optimizer_options: Optional[Dict] = None,
                    n_dimensions: Optional[int] = None) -> Tuple[Dict[str, Dict], pd.DataFrame]:
        """
        Reduce scale using constrained optimization (or a deterministic selector).

        Args:
            items: List of item texts
            dimension_labels: List of dimension labels (same length as items).
                ``None`` infers a structure from item text with
                :meth:`infer_dimensions` (see ``n_dimensions``); the inferred
                labels are stored in ``self.last_dimension_labels_``.
            items_per_dim: Either a single integer for all dimensions or a dictionary
                          mapping dimension names to target counts
            suppress_details: If True, suppress detailed optimization output
            n_tries: Number of random initializations to try (``method='optimize'``)
            reverse_scored: Optional list of 0-indexed item positions that are
                reverse-scored. Their rows and columns in the similarity matrix
                will be sign-flipped so the optimizer treats them correctly.
                Applied whether the matrix is computed here or passed in.
            objective: Objective mode, 'coherence' (default) or 'coverage'.
                See objective_function for definitions. 'coverage' is
                recommended when selecting one item per dimension, where the
                coherence objective is degenerate (no within-dimension pairs).
            embeddings: Optional precomputed item embeddings (n_items, dim);
                skips the encoder.
            similarity_matrix: Optional precomputed item similarity matrix
                (n_items, n_items); skips the encoder and ``embeddings``. Any
                symmetric similarity (e.g. predicted inter-item correlations
                from a fine-tuned model) can be used.
            method: ``'optimize'`` (default; the configured optimizer searches
                the objective), ``'centroid'`` (keep the items most central to
                their dimension; see :func:`siren.selection.select_centroid`)
                or ``'greedy'`` (greedy facility-location coverage; see
                :func:`siren.selection.select_greedy_coverage`).
            seed: Random seed for reproducible selection. Overrides the
                instance-level ``seed``.
            optimizer_options: Options forwarded to the optimizer for this
                call (e.g. ``{'n_ants': 40, 'maxiter': 200}``). Merged over the
                instance-level ``optimizer_options``.
            n_dimensions: Number of dimensions to infer when
                ``dimension_labels`` is None (``None`` = silhouette selection).

        Returns:
            Tuple of (results dictionary, similarity matrix DataFrame)
                - results: Dict with keys as dimension names, values as dicts containing:
                    - 'indices': List of selected item indices
                    - 'items': List of selected item texts
                    - 'metrics': Dict with 'within_dim_scores',
                      'between_dim_scores' and 'coverage'
                - similarity_matrix: DataFrame with pairwise item similarities
        """
        if objective not in ('coherence', 'coverage'):
            raise ValueError(f"Unknown objective: {objective!r}; "
                             "use 'coherence' or 'coverage'")
        if method not in SELECTION_METHODS:
            raise ValueError(f"Unknown method: {method!r}; use one of {SELECTION_METHODS}")
        self._objective_mode = objective
        n_items = len(items)

        seed = self.seed if seed is None else seed
        if seed is not None:
            np.random.seed(seed)

        # --- similarity matrix -------------------------------------------
        if embeddings is not None:
            embeddings = np.asarray(embeddings, dtype=float)
            if embeddings.shape[0] != n_items:
                raise ValueError("embeddings must have one row per item")
        if similarity_matrix is not None:
            S = np.asarray(similarity_matrix, dtype=float)
            if S.shape != (n_items, n_items):
                raise ValueError(f"similarity_matrix must be {n_items}x{n_items}, got {S.shape}")
            S = self._apply_reverse(np.clip(S, -1.0, 1.0), reverse_scored)
        else:
            if embeddings is None:
                if not suppress_details:
                    logger.info("Computing embeddings...")
                embeddings = self.embed(items, show_progress_bar=not suppress_details)
            if not suppress_details:
                logger.info("Computing similarity matrix...")
            S = self.similarity_from_embeddings(embeddings, reverse_scored)
        similarity_matrix = S
        self.last_similarity_ = S
        self.last_embeddings_ = embeddings

        # --- dimension structure ------------------------------------------
        if dimension_labels is None:
            if embeddings is None:
                # Only a similarity matrix was supplied; structure inference
                # needs embeddings, so encode the item texts.
                embeddings = self.embed(items, show_progress_bar=not suppress_details)
                self.last_embeddings_ = embeddings
            dimension_labels = self.infer_dimensions(
                embeddings=embeddings, n_dimensions=n_dimensions, seed=seed)
            if not suppress_details:
                print(f"Inferred {len(set(dimension_labels))} dimensions from item text")
        elif len(dimension_labels) != n_items:
            raise ValueError("dimension_labels must have the same length as items")
        self.last_dimension_labels_ = list(dimension_labels)

        # Get dimension indices
        dim_indices = self.get_dimension_indices(dimension_labels)

        # Convert items_per_dim to dictionary if it's an integer
        if isinstance(items_per_dim, int):
            items_per_dim_dict = {dim: items_per_dim for dim in dim_indices.keys()}
        else:
            items_per_dim_dict = dict(items_per_dim)
        for dim, indices in dim_indices.items():
            if dim not in items_per_dim_dict:
                raise ValueError(f"items_per_dim has no entry for dimension {dim!r}")
            if items_per_dim_dict[dim] > len(indices):
                raise ValueError(
                    f"Requested {items_per_dim_dict[dim]} items for dimension {dim} "
                    f"but only {len(indices)} available"
                )

        # --- selection ------------------------------------------------------
        if method == 'centroid':
            selection = select_centroid(similarity_matrix, dim_indices, items_per_dim_dict)
            selected = selection_to_vector(selection, n_items).astype(bool)
        elif method == 'greedy':
            selection = select_greedy_coverage(similarity_matrix, dim_indices, items_per_dim_dict)
            selected = selection_to_vector(selection, n_items).astype(bool)
        else:
            selected = self._optimize(items, similarity_matrix, dim_indices, items_per_dim_dict,
                                      n_tries, suppress_details, seed, optimizer_options)

        # Format results
        output = {}
        coverage = _diag.coverage_score(similarity_matrix, dim_indices,
                                        {d: list(np.array(i)[selected[i]])
                                         for d, i in dim_indices.items()})
        for dim, indices in dim_indices.items():
            dim_selected = np.array(indices)[selected[indices]]

            # Calculate metrics
            within_dim_scores = [np.mean([abs(similarity_matrix[i, j])
                for j in dim_selected if j != i]) if len(dim_selected) > 1 else float('nan')
                for i in dim_selected]

            selected_all = np.where(selected)[0]
            between_dim_scores = [np.mean([abs(similarity_matrix[i, j])
                for d, idx in dim_indices.items()
                if d != dim
                for j in idx if j in selected_all]) for i in dim_selected]

            output[dim] = {
                'indices': dim_selected.tolist(),
                'items': [items[i] for i in dim_selected],
                'metrics': {
                    'within_dim_scores': within_dim_scores,
                    'between_dim_scores': between_dim_scores,
                    'coverage': coverage[dim],
                }
            }
        self.last_selection_ = {d: v['indices'] for d, v in output.items()}

        # Create similarity matrix DataFrame
        sim_matrix_df = pd.DataFrame(similarity_matrix)

        return output, sim_matrix_df

    def _optimize(self, items, similarity_matrix, dim_indices, items_per_dim_dict,
                  n_tries, suppress_details, seed, optimizer_options) -> np.ndarray:
        """Run the configured optimizer with multiple restarts; return a boolean mask."""
        n_items = len(items)
        if not suppress_details:
            print(f"\nScale reduction using {self.optimizer.__class__.__name__}")
            print(f"Target items per dimension: {items_per_dim_dict}")
            print(f"Number of optimization attempts: {n_tries}")

        options = {
            'maxiter': 100 if isinstance(self.optimizer, AntColonyOptimizer) else 1000,
        }
        options.update(self.optimizer_options)
        options.update(optimizer_options or {})
        options['suppress_output'] = suppress_details
        options['seed'] = seed

        best_score = np.inf
        best_result = None

        for try_idx in range(n_tries):
            if not suppress_details:
                print(f"\n{'='*50}")
                print(f"OPTIMIZATION ATTEMPT {try_idx + 1}/{n_tries}")
                print(f"{'='*50}")

            # Initial solution: randomly select appropriate number of items from each dimension
            x0 = np.zeros(n_items)
            for dim, indices in dim_indices.items():
                target_count = items_per_dim_dict[dim]
                chosen = np.random.choice(indices, size=target_count, replace=False)
                x0[chosen] = 1

            # Constraints for SLSQP
            constraints = []
            for dim, indices in dim_indices.items():
                target_count = items_per_dim_dict[dim]
                constraint = {
                    'type': 'eq',
                    'fun': lambda x, idx=indices, tc=target_count: np.sum(x[idx]) - tc
                }
                constraints.append(constraint)

            # Binary constraints handled by bounds
            bounds = [(0, 1) for _ in range(n_items)]

            # Give scipy-based optimizers a distinct seed per restart
            run_options = dict(options)
            if seed is not None:
                run_options['seed'] = seed + try_idx

            result = self.optimizer.optimize(
                self.objective_function,
                x0,
                args=(similarity_matrix, dim_indices, items_per_dim_dict),
                constraints=constraints,
                bounds=bounds,
                options=run_options
            )

            if result.fun < best_score:
                best_score = result.fun
                best_result = result
                if not suppress_details:
                    print(f"  -> New best solution found! Score: {best_score:.4f}")

        return np.round(best_result.x).astype(bool)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def fidelity_report(self, result: Optional[Dict[str, Dict]] = None,
                        similarity_matrix: Optional[np.ndarray] = None,
                        dimension_labels: Optional[Sequence[str]] = None,
                        n_random: int = 1000,
                        seed: Optional[int] = None) -> Dict[str, object]:
        """
        Fidelity summary of a selection (see :func:`siren.diagnostics.fidelity_report`).

        Defaults to the most recent :meth:`reduce_scale` call's selection,
        similarity matrix and dimension labels.
        """
        result = self.last_selection_ if result is None else result
        S = self.last_similarity_ if similarity_matrix is None else np.asarray(similarity_matrix)
        labels = self.last_dimension_labels_ if dimension_labels is None else dimension_labels
        if result is None or S is None or labels is None:
            raise ValueError("Call reduce_scale() first or pass result, similarity_matrix "
                             "and dimension_labels")
        return _diag.fidelity_report(S, self.get_dimension_indices(list(labels)), result,
                                     n_random=n_random,
                                     seed=self.seed if seed is None else seed)

    def coverage_report(self, result: Optional[Dict[str, Dict]] = None,
                        similarity_matrix: Optional[np.ndarray] = None,
                        dimension_labels: Optional[Sequence[str]] = None,
                        items: Optional[Sequence[str]] = None,
                        gap_threshold: float = 0.5) -> Dict[str, pd.DataFrame]:
        """
        Dropped-item audit of a selection (see :func:`siren.diagnostics.coverage_report`).

        Defaults to the most recent :meth:`reduce_scale` call.
        """
        result = self.last_selection_ if result is None else result
        S = self.last_similarity_ if similarity_matrix is None else np.asarray(similarity_matrix)
        labels = self.last_dimension_labels_ if dimension_labels is None else dimension_labels
        if result is None or S is None or labels is None:
            raise ValueError("Call reduce_scale() first or pass result, similarity_matrix "
                             "and dimension_labels")
        return _diag.coverage_report(S, self.get_dimension_indices(list(labels)), result,
                                     items=items, gap_threshold=gap_threshold)

    def print_comparison(self, items: List[str], dimension_labels: List[str],
                        result: Dict[str, Dict], items_per_dim: Union[int, Dict[str, int]],
                        show_summary: bool = False,
                        reverse_scored: Optional[List[int]] = None):
        """
        Print original and shortened scales side by side.

        Args:
            items: Original list of items
            dimension_labels: Original dimension labels
            result: Results dictionary from reduce_scale()
            items_per_dim: Target items per dimension used
            show_summary: If True, show summary statistics
            reverse_scored: Optional list of 0-indexed item positions that are
                reverse-scored. Items in this list are tagged with (R).
        """
        # Get dimension indices
        dim_indices = self.get_dimension_indices(dimension_labels)

        # Calculate column widths
        max_len_original = min(60, max(len(item) for item in items) + 5)
        max_len_shortened = max_len_original

        # Print header
        print("\n" + "="*120)
        optimizer_name = self.optimizer.__class__.__name__.replace("Optimizer", "").upper()
        print(f"SCALE COMPARISON: ORIGINAL vs SHORTENED ({optimizer_name} OPTIMIZED)")
        print("="*120)
        orig_header = f"{'ORIGINAL SCALE':<{max_len_original}}"
        short_header = f"{'SHORTENED SCALE':<{max_len_shortened}}"
        print(f"\n{orig_header} | {short_header}")
        print("-"*120)

        # Process each dimension
        for dim in sorted(dim_indices.keys()):
            indices = dim_indices[dim]
            original_items = [items[i] for i in indices]
            shortened_items = result[dim]['items']

            print(f"\n{'='*40} DIMENSION {dim} {'='*40}")
            n_orig = len(original_items)
            n_short = len(shortened_items)
            print(f"Original: {n_orig} items | Shortened: {n_short} items")
            print("-"*120)

            max_rows = max(len(original_items), len(shortened_items))

            for i in range(max_rows):
                # Original item
                if i < len(original_items):
                    orig_item = original_items[i]
                    if len(orig_item) > 57:
                        orig_item = orig_item[:54] + "..."
                    orig_text = f"{i+1:2}. {orig_item}"
                else:
                    orig_text = ""

                # Shortened item
                if i < len(shortened_items):
                    short_item = shortened_items[i]
                    if len(short_item) > 57:
                        short_item = short_item[:54] + "..."
                    short_text = f"{i+1:2}. {short_item}"
                    # Tag reverse-scored items
                    if reverse_scored and i < len(result[dim]['indices']):
                        original_index = result[dim]['indices'][i]
                        if original_index in reverse_scored:
                            short_text += " (R)"
                else:
                    short_text = ""

                print(f"{orig_text:<{max_len_original}} | {short_text:<{max_len_shortened}}")

        if show_summary:
            # Print summary statistics
            print("\n" + "="*120)
            print("SUMMARY STATISTICS")
            print("="*120)

            total_original = len(items)
            total_shortened = sum(len(data['items']) for data in result.values())

            print(f"Total items: {total_original} → {total_shortened} "
                  f"(reduction of {total_original - total_shortened} items)")
            print(f"Compression ratio: {total_shortened/total_original:.1%}")

            # Overall metrics
            all_within = [score for data in result.values()
                         for score in data['metrics']['within_dim_scores']
                         if not np.isnan(score)]
            all_between = [score for data in result.values()
                          for score in data['metrics']['between_dim_scores']]
            all_cov = [data['metrics'].get('coverage', np.nan) for data in result.values()]
            overall_within = np.mean(all_within) if all_within else 0
            overall_between = np.mean(all_between) if all_between else 0
            overall_cov = np.nanmean(all_cov) if all_cov else float('nan')
            if overall_between > 0:
                overall_ratio = overall_within / overall_between
            else:
                overall_ratio = float('inf')

            print(f"\nOverall: Within-similarity={overall_within:.3f}, "
                  f"Between-similarity={overall_between:.3f}, Ratio={overall_ratio:.2f}, "
                  f"Coverage={overall_cov:.3f}")
            print("="*120)
