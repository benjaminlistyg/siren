"""
SIREN - Semantic Item Reduction Engine
Main class for psychometric scale reduction.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple, Union

from .optimizers import (
    OptimizationMethod,
    AntColonyOptimizer,
    SLSQPOptimizer,
    GeneticAlgorithmOptimizer,
    SimulatedAnnealingOptimizer
)

logger = logging.getLogger(__name__)


class SIREN:
    """
    Semantic Item Reduction Engine - Main class for scale reduction.

    SIREN uses semantic similarity and advanced optimization algorithms
    to reduce psychometric scales while maintaining their structure and
    psychometric properties.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L12-v2',
                 optimizer: Optional[OptimizationMethod] = None):
        """
        Initialize SIREN with a sentence transformer model and optimizer.

        Args:
            model_name: Name of the sentence-transformer model to use
            optimizer: Optimization method instance (default: AntColonyOptimizer)
        """
        logger.info(f"Initializing SIREN with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        # Default to ACO as it shows superior performance for scale reduction
        self.optimizer = optimizer or AntColonyOptimizer()

    def set_optimizer(self, optimizer: OptimizationMethod):
        """
        Change the optimization method.

        Args:
            optimizer: New OptimizationMethod instance
        """
        self.optimizer = optimizer

    @classmethod
    def with_aco(cls, model_name: str = 'all-MiniLM-L12-v2') -> 'SIREN':
        """
        Create SIREN with Ant Colony Optimization (default).

        Args:
            model_name: Name of the sentence-transformer model

        Returns:
            SIREN instance with ACO optimizer
        """
        return cls(model_name, AntColonyOptimizer())

    @classmethod
    def with_slsqp(cls, model_name: str = 'all-MiniLM-L12-v2') -> 'SIREN':
        """
        Create SIREN with SLSQP optimizer.

        Args:
            model_name: Name of the sentence-transformer model

        Returns:
            SIREN instance with SLSQP optimizer
        """
        return cls(model_name, SLSQPOptimizer())

    @classmethod
    def with_genetic_algorithm(cls, model_name: str = 'all-MiniLM-L12-v2') -> 'SIREN':
        """
        Create SIREN with genetic algorithm optimizer.

        Args:
            model_name: Name of the sentence-transformer model

        Returns:
            SIREN instance with genetic algorithm optimizer
        """
        return cls(model_name, GeneticAlgorithmOptimizer())

    @classmethod
    def with_simulated_annealing(cls, model_name: str = 'all-MiniLM-L12-v2') -> 'SIREN':
        """
        Create SIREN with simulated annealing optimizer.

        Args:
            model_name: Name of the sentence-transformer model

        Returns:
            SIREN instance with simulated annealing optimizer
        """
        return cls(model_name, SimulatedAnnealingOptimizer())

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

        Maximizes within-dimension similarity and minimizes between-dimension
        similarity while satisfying item count constraints.

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

    def reduce_scale(self, items: List[str], dimension_labels: List[str],
                    items_per_dim: Union[int, Dict[str, int]] = 2,
                    suppress_details: bool = False,
                    n_tries: int = 5) -> Tuple[Dict[str, Dict], pd.DataFrame]:
        """
        Reduce scale using constrained optimization.

        Args:
            items: List of item texts
            dimension_labels: List of dimension labels (same length as items)
            items_per_dim: Either a single integer for all dimensions or a dictionary
                          mapping dimension names to target counts
            suppress_details: If True, suppress detailed optimization output
            n_tries: Number of random initializations to try

        Returns:
            Tuple of (results dictionary, similarity matrix DataFrame)
                - results: Dict with keys as dimension names, values as dicts containing:
                    - 'indices': List of selected item indices
                    - 'items': List of selected item texts
                    - 'metrics': Dict with 'within_dim_scores' and 'between_dim_scores'
                - similarity_matrix: DataFrame with pairwise item similarities
        """
        if not suppress_details:
            logger.info("Computing embeddings...")

        embeddings = self.model.encode(items, convert_to_tensor=True,
                                      show_progress_bar=not suppress_details)
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        if not suppress_details:
            logger.info("Computing similarity matrix...")

        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = np.clip(similarity_matrix, -1, 1)

        # Get dimension indices
        dim_indices = self.get_dimension_indices(dimension_labels)
        n_items = len(items)

        # Convert items_per_dim to dictionary if it's an integer
        if isinstance(items_per_dim, int):
            items_per_dim_dict = {dim: items_per_dim for dim in dim_indices.keys()}
        else:
            items_per_dim_dict = items_per_dim

        if not suppress_details:
            print(f"\nScale reduction using {self.optimizer.__class__.__name__}")
            print(f"Target items per dimension: {items_per_dim_dict}")
            print(f"Number of optimization attempts: {n_tries}")

        # Try multiple random initializations
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
                if target_count > len(indices):
                    raise ValueError(
                        f"Requested {target_count} items for dimension {dim} "
                        f"but only {len(indices)} available"
                    )
                selected = np.random.choice(indices, size=target_count, replace=False)
                x0[selected] = 1

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

            # Run optimization
            options = {
                'maxiter': 100 if isinstance(self.optimizer, AntColonyOptimizer) else 1000,
                'suppress_output': suppress_details
            }

            result = self.optimizer.optimize(
                self.objective_function,
                x0,
                args=(similarity_matrix, dim_indices, items_per_dim_dict),
                constraints=constraints,
                bounds=bounds,
                options=options
            )

            if result.fun < best_score:
                best_score = result.fun
                best_result = result
                if not suppress_details:
                    print(f"  -> New best solution found! Score: {best_score:.4f}")

        # Round to binary solution
        selected = np.round(best_result.x).astype(bool)

        # Format results
        output = {}
        for dim, indices in dim_indices.items():
            dim_selected = np.array(indices)[selected[indices]]

            # Calculate metrics
            within_dim_scores = [np.mean([abs(similarity_matrix[i, j])
                for j in dim_selected if j != i]) for i in dim_selected]

            between_dim_scores = [np.mean([abs(similarity_matrix[i, j])
                for d, idx in dim_indices.items()
                if d != dim
                for j in idx if j in np.where(selected)[0]]) for i in dim_selected]

            output[dim] = {
                'indices': dim_selected.tolist(),
                'items': [items[i] for i in dim_selected],
                'metrics': {
                    'within_dim_scores': within_dim_scores,
                    'between_dim_scores': between_dim_scores
                }
            }

        # Create similarity matrix DataFrame
        sim_matrix_df = pd.DataFrame(similarity_matrix)

        return output, sim_matrix_df

    def print_comparison(self, items: List[str], dimension_labels: List[str],
                        result: Dict[str, Dict], items_per_dim: Union[int, Dict[str, int]],
                        show_summary: bool = False):
        """
        Print original and shortened scales side by side.

        Args:
            items: Original list of items
            dimension_labels: Original dimension labels
            result: Results dictionary from reduce_scale()
            items_per_dim: Target items per dimension used
            show_summary: If True, show summary statistics
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
        print(f"\n{'ORIGINAL SCALE':<{max_len_original}} | {'SHORTENED SCALE':<{max_len_shortened}}")
        print("-"*120)

        # Process each dimension
        for dim in sorted(dim_indices.keys()):
            indices = dim_indices[dim]
            original_items = [items[i] for i in indices]
            shortened_items = result[dim]['items']

            print(f"\n{'='*40} DIMENSION {dim} {'='*40}")
            print(f"Original: {len(original_items)} items | Shortened: {len(shortened_items)} items")
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
                    # Add (R) for reverse-scored items
                    reverse_keywords = ["doubt", "rarely", "lack", "struggle", "overwhelm",
                                      "anxious", "panic", "freeze", "hesitate", "isolated",
                                      "disconnected", "distracted"]
                    if any(word in short_item.lower() for word in reverse_keywords):
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
                         for score in data['metrics']['within_dim_scores']]
            all_between = [score for data in result.values()
                          for score in data['metrics']['between_dim_scores']]
            overall_within = np.mean(all_within) if all_within else 0
            overall_between = np.mean(all_between) if all_between else 0
            overall_ratio = overall_within / overall_between if overall_between > 0 else float('inf')

            print(f"\nOverall: Within-similarity={overall_within:.3f}, "
                  f"Between-similarity={overall_between:.3f}, Ratio={overall_ratio:.2f}")
            print("="*120)
