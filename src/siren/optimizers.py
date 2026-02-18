"""
Optimization methods for SIREN scale reduction.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing, OptimizeResult
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Any


class OptimizationMethod(ABC):
    """Abstract base class for optimization methods."""

    @abstractmethod
    def optimize(self, objective_function: Callable, x0: np.ndarray, args: Tuple,
                 constraints: List[Dict], bounds: List[Tuple], **kwargs) -> OptimizeResult:
        """
        Run optimization and return results.

        Args:
            objective_function: Function to minimize
            x0: Initial solution
            args: Additional arguments for objective function
            constraints: List of constraint dictionaries
            bounds: List of (min, max) tuples for each variable
            **kwargs: Additional optimizer-specific parameters

        Returns:
            OptimizeResult object with optimization results
        """
        pass


class AntColonyOptimizer(OptimizationMethod):
    """
    Ant Colony Optimization for scale reduction.

    ACO is particularly effective for combinatorial optimization problems
    like scale reduction, as shown by Schroeders et al. (2016) and
    subsequent applications in psychometrics.
    """

    def optimize(self, objective_function: Callable, x0: np.ndarray, args: Tuple,
                 constraints: List[Dict], bounds: List[Tuple], **kwargs) -> OptimizeResult:
        """
        Optimize using Ant Colony Optimization (ACO).

        Args:
            objective_function: Function to minimize
            x0: Initial solution (not used in ACO, but kept for interface consistency)
            args: (similarity_matrix, dim_indices, items_per_dim)
            constraints: Constraints (not used in ACO)
            bounds: Variable bounds
            **kwargs: Options including n_ants, maxiter, alpha, beta, rho, q0, suppress_output

        Returns:
            OptimizeResult with optimized solution
        """
        options = kwargs.get('options', {})
        n_ants = options.get('n_ants', 20)
        n_iterations = options.get('maxiter', 100)
        alpha = options.get('alpha', 1.0)  # Pheromone importance
        beta = options.get('beta', 2.0)    # Heuristic importance
        rho = options.get('rho', 0.5)      # Evaporation rate
        q0 = options.get('q0', 0.9)        # Exploitation vs exploration
        suppress_output = options.get('suppress_output', False)

        # Extract problem parameters
        similarity_matrix, dim_indices, items_per_dim = args
        n_items = len(bounds)

        # Initialize pheromone matrix
        tau_0 = 1.0 / n_items  # Initial pheromone level
        pheromone = np.ones((n_items,)) * tau_0

        # Calculate heuristic information (item quality based on similarity)
        eta = self._calculate_heuristic(similarity_matrix, dim_indices)

        # Track best solution
        best_solution = None
        best_score = np.inf
        iteration_scores = []

        if not suppress_output:
            print(f"\nStarting Ant Colony Optimization with {n_ants} ants and {n_iterations} iterations")
            print(f"Target items per dimension: {items_per_dim}")
            print(f"Parameters: alpha={alpha}, beta={beta}, rho={rho}, q0={q0}")

        for iteration in range(n_iterations):
            solutions = []
            scores = []

            # Each ant constructs a solution
            for ant in range(n_ants):
                solution = self._construct_solution(
                    pheromone, eta, dim_indices, items_per_dim,
                    alpha, beta, q0
                )

                # Evaluate solution
                score = objective_function(solution, similarity_matrix, dim_indices, items_per_dim)
                solutions.append(solution)
                scores.append(score)

                # Update best solution
                if score < best_score:
                    best_score = score
                    best_solution = solution.copy()

            # Update pheromones
            pheromone = self._update_pheromones(
                pheromone, solutions, scores, rho, best_solution, best_score
            )

            # Track progress
            iteration_scores.append(best_score)

            # Print progress if not suppressed
            if not suppress_output and ((iteration + 1) % 10 == 0 or iteration == 0):
                progress = (iteration + 1) / n_iterations * 100
                avg_score = np.mean(scores)
                print(f"ACO Progress: {progress:.1f}% - Iteration {iteration+1}/{n_iterations}, "
                      f"Best: {best_score:.4f}, Avg: {avg_score:.4f}")

                # Show current best solution details
                selected_indices = np.where(best_solution > 0.5)[0]
                print(f"Current best solution has {len(selected_indices)} items selected")

                # Show constraint satisfaction
                for dim, indices in dim_indices.items():
                    target_count = items_per_dim[dim] if isinstance(items_per_dim, dict) else items_per_dim
                    selected_count = len(np.intersect1d(selected_indices, indices))
                    status = "✓" if selected_count == target_count else "✗"
                    print(f"  Dimension {dim}: {selected_count}/{target_count} items {status}")

            # Early stopping if no improvement
            if len(iteration_scores) > 20:
                recent_scores = iteration_scores[-20:]
                if len(set(recent_scores)) == 1:  # No improvement in 20 iterations
                    if not suppress_output:
                        print(f"Early stopping at iteration {iteration+1} - no improvement detected")
                    break

        if not suppress_output:
            print(f"Ant Colony Optimization completed with score: {best_score:.4f}")

        # Create result object compatible with scipy.optimize
        result = OptimizeResult()
        result.x = best_solution
        result.fun = best_score
        result.success = True
        result.message = f"ACO converged after {iteration+1} iterations"
        result.nit = iteration + 1

        return result

    def _calculate_heuristic(self, similarity_matrix: np.ndarray,
                            dim_indices: Dict[str, List[int]]) -> np.ndarray:
        """Calculate heuristic information for each item."""
        n_items = len(similarity_matrix)
        eta = np.zeros(n_items)

        for dim, indices in dim_indices.items():
            for i in indices:
                # Within-dimension similarity (want high)
                within_sim = np.mean([abs(similarity_matrix[i, j]) for j in indices if i != j])

                # Between-dimension similarity (want low)
                other_indices = [idx for d, idxs in dim_indices.items()
                                if d != dim for idx in idxs]
                if other_indices:
                    between_sim = np.mean([abs(similarity_matrix[i, j]) for j in other_indices])
                else:
                    between_sim = 0

                # Heuristic value: high within-dim similarity, low between-dim similarity
                eta[i] = within_sim / (between_sim + 0.01)  # Add small constant to avoid division by zero

        # Normalize
        eta = eta / np.max(eta) if np.max(eta) > 0 else eta
        return eta

    def _construct_solution(self, pheromone: np.ndarray, eta: np.ndarray,
                           dim_indices: Dict[str, List[int]],
                           items_per_dim: Any, alpha: float, beta: float,
                           q0: float) -> np.ndarray:
        """Construct a solution using ACO probabilistic rules."""
        n_items = len(pheromone)
        solution = np.zeros(n_items)

        # Convert items_per_dim to dict if needed
        if isinstance(items_per_dim, int):
            items_per_dim_dict = {dim: items_per_dim for dim in dim_indices.keys()}
        else:
            items_per_dim_dict = items_per_dim

        # Select items for each dimension
        for dim, indices in dim_indices.items():
            target_count = items_per_dim_dict[dim]
            available_indices = list(indices)
            selected = []

            # Select items one by one
            for _ in range(min(target_count, len(available_indices))):
                if not available_indices:
                    break

                # Calculate probabilities
                probs = []
                for idx in available_indices:
                    prob = (pheromone[idx] ** alpha) * (eta[idx] ** beta)
                    probs.append(prob)

                probs = np.array(probs)
                if np.sum(probs) > 0:
                    probs = probs / np.sum(probs)
                else:
                    probs = np.ones(len(probs)) / len(probs)

                # Select item using pseudo-random proportional rule
                if np.random.random() < q0:
                    # Exploitation: choose best
                    chosen_idx = available_indices[np.argmax(probs)]
                else:
                    # Exploration: probabilistic choice
                    chosen_idx = np.random.choice(available_indices, p=probs)

                selected.append(chosen_idx)
                available_indices.remove(chosen_idx)

            # Mark selected items in solution
            for idx in selected:
                solution[idx] = 1

        return solution

    def _update_pheromones(self, pheromone: np.ndarray, solutions: List[np.ndarray],
                           scores: List[float], rho: float,
                           best_solution: np.ndarray, best_score: float) -> np.ndarray:
        """Update pheromone levels using the ant system approach."""
        # Evaporation
        pheromone = pheromone * (1 - rho)

        # Add pheromone from solutions
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score != min_score else 1

        for solution, score in zip(solutions, scores):
            # Better solutions deposit more pheromone
            quality = 1.0 - (score - min_score) / score_range
            delta_tau = quality / len(solution)

            selected_indices = np.where(solution > 0.5)[0]
            for idx in selected_indices:
                pheromone[idx] += delta_tau

        # Elite ant: best solution deposits extra pheromone
        best_indices = np.where(best_solution > 0.5)[0]
        elite_bonus = 2.0 / (best_score + 1)  # Higher bonus for better scores
        for idx in best_indices:
            pheromone[idx] += elite_bonus

        # Prevent pheromone levels from getting too extreme
        min_pheromone = 0.001
        max_pheromone = 1.0
        pheromone = np.clip(pheromone, min_pheromone, max_pheromone)

        return pheromone


class SLSQPOptimizer(OptimizationMethod):
    """Sequential Least Squares Programming optimizer."""

    def optimize(self, objective_function: Callable, x0: np.ndarray, args: Tuple,
                 constraints: List[Dict], bounds: List[Tuple], **kwargs) -> OptimizeResult:
        """Optimize using SLSQP method."""
        options = kwargs.get('options', {'maxiter': 1000})
        suppress_output = options.get('suppress_output', False)
        maxiter = options.get('maxiter', 1000)

        if not suppress_output:
            print(f"\nStarting SLSQP optimization with max iterations {maxiter}")
            print(f"Target items per dimension: {args[2]}")

        # Create clean options for minimize
        clean_options = {'maxiter': maxiter}

        result = minimize(
            objective_function,
            x0,
            args=args,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options=clean_options
        )

        if not suppress_output:
            print(f"SLSQP completed with score: {result.fun:.4f}, Success: {result.success}")

        return result


class GeneticAlgorithmOptimizer(OptimizationMethod):
    """Genetic Algorithm optimizer using differential evolution."""

    def optimize(self, objective_function: Callable, x0: np.ndarray, args: Tuple,
                 constraints: List[Dict], bounds: List[Tuple], **kwargs) -> OptimizeResult:
        """Optimize using differential evolution (genetic algorithm)."""
        options = kwargs.get('options', {})
        max_iterations = options.get('maxiter', 100)
        popsize = options.get('popsize', 20)
        convergence_threshold = options.get('tol', 0.01)
        suppress_output = options.get('suppress_output', False)

        def constrained_objective(x, *args):
            x_binary = np.round(x).astype(int)
            obj_value = objective_function(x, *args)

            # Apply constraint penalties
            penalty = 0
            similarity_matrix, dim_indices, items_per_dim = args
            for dim, indices in dim_indices.items():
                target_count = items_per_dim[dim] if isinstance(items_per_dim, dict) else items_per_dim
                selected_indices = np.where(x_binary > 0.5)[0]
                selected_count = len(np.intersect1d(selected_indices, indices))
                if selected_count != target_count:
                    penalty += 100 * (abs(selected_count - target_count) ** 2)

            return obj_value + penalty

        if not suppress_output:
            print(f"\nStarting Genetic Algorithm optimization with population size {popsize}")
            print(f"Target items per dimension: {args[2]}")

        result = differential_evolution(
            constrained_objective,
            bounds=bounds,
            args=args,
            maxiter=max_iterations,
            popsize=popsize,
            tol=convergence_threshold,
            atol=convergence_threshold,
            polish=False,
            workers=1,
            disp=False
        )

        if not suppress_output:
            print(f"Genetic Algorithm completed with score: {result.fun:.4f}")

        return result


class SimulatedAnnealingOptimizer(OptimizationMethod):
    """Simulated Annealing optimizer."""

    def optimize(self, objective_function: Callable, x0: np.ndarray, args: Tuple,
                 constraints: List[Dict], bounds: List[Tuple], **kwargs) -> OptimizeResult:
        """Optimize using simulated annealing."""
        options = kwargs.get('options', {})
        maxiter = options.get('maxiter', 1000)
        suppress_output = options.get('suppress_output', False)

        def constrained_objective(x, *args):
            obj_value = objective_function(x, *args)

            # Apply constraint penalties
            penalty = 0
            similarity_matrix, dim_indices, items_per_dim = args
            for dim, indices in dim_indices.items():
                target_count = items_per_dim[dim] if isinstance(items_per_dim, dict) else items_per_dim
                x_binary = np.round(x).astype(int)
                selected_indices = np.where(x_binary > 0.5)[0]
                selected_count = len(np.intersect1d(selected_indices, indices))
                if selected_count != target_count:
                    penalty += 100 * (abs(selected_count - target_count) ** 2)

            return obj_value + penalty

        if not suppress_output:
            print(f"\nStarting Simulated Annealing optimization with max iterations {maxiter}")
            print(f"Target items per dimension: {args[2]}")

        result = dual_annealing(
            constrained_objective,
            bounds=bounds,
            x0=x0,
            args=args,
            maxiter=maxiter,
            no_local_search=True
        )

        if not suppress_output:
            print(f"Simulated Annealing completed with score: {result.fun:.4f}")

        return result
