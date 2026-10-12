# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-12

### Added
- Precomputed inputs: `reduce_scale()` accepts `embeddings=` or
  `similarity_matrix=` and skips the encoder. `SIREN(model_name=None)` never
  loads a model; the model is otherwise loaded lazily on first use.
  `SIREN(model=...)` accepts an already-loaded encoder.
- Reproducibility: `seed=` on `SIREN()` and `reduce_scale()` seeds the
  restarts, ACO, and the scipy-based optimizers.
- Optimizer tuning: `optimizer_options=` on `SIREN()` and `reduce_scale()`
  forwards e.g. `{'n_ants': 40, 'maxiter': 200}` to the optimizer.
- Structure discovery: `SIREN.infer_dimensions()` clusters item embeddings
  (k-means; k by silhouette unless given). `reduce_scale(dimension_labels=None)`
  infers labels automatically.
- Alternative selectors: `reduce_scale(method='centroid' | 'greedy')` and
  `siren.selection.select_centroid` / `select_greedy_coverage` /
  `select_random_stratified` for deterministic baselines.
- Diagnostics (`siren.diagnostics`): `coverage_report` (dropped-item proxy
  audit and content gaps), `coverage_percentile` (null distribution of random
  dimension-stratified subsets), `redundancy` (near-duplicate pairs and
  clusters), `fidelity_report` (one-table summary), plus `coverage_score`,
  `coherence_score`, `between_score`. Convenience wrappers
  `SIREN.fidelity_report()` and `SIREN.coverage_report()`.
- Item vetting: `SIREN.item_fit()` scores candidate items against dimension
  centroids and nearest existing items.
- Plots (`siren.plots`, optional `matplotlib`; `pip install siren-scale[plots]`):
  `plot_item_map` (PCA/MDS) and `plot_similarity_heatmap`.
- `embed()`, `compute_similarity()`, `similarity_from_embeddings()` helpers.
- Per-dimension `'coverage'` added to `result[dim]['metrics']`.

### Changed
- `within_dim_scores` is `nan` (not 0) for a dimension with a single retained item.
- `print_comparison(show_summary=True)` also prints mean coverage.

## [0.2.0] - 2026-08-25

### Added
- `objective` parameter for `reduce_scale()`: `'coherence'` (default, original
  behavior) or `'coverage'` (facility-location content coverage). Coverage
  selects items semantically representative of each dimension's full pool and
  is well-defined for one item per dimension, where coherence is degenerate.
- `reverse_scored` parameter for `reduce_scale()` and `print_comparison()`:
  sign-flips reverse-keyed items in the similarity matrix before optimization.
- Model-free unit tests for both objective modes.

## [0.1.0] - 2024-10-29

### Added
- Initial release of SIREN (Semantic Item Reduction Engine)
- Core SIREN class for psychometric scale reduction
- Multiple optimization algorithms:
  - Ant Colony Optimization (ACO) - default and recommended
  - SLSQP (Sequential Least Squares Programming)
  - Genetic Algorithm (Differential Evolution)
  - Simulated Annealing
- Semantic similarity computation using sentence-transformers
- Support for multi-dimensional scales
- Variable items per dimension configuration
- Comprehensive documentation and examples
- Basic test suite
- MIT License

### Features
- Automatic item selection based on semantic similarity
- Within-dimension similarity maximization
- Between-dimension similarity minimization
- Constraint satisfaction for exact item counts
- Multiple random initialization attempts for better optimization
- Detailed progress reporting (can be suppressed)
- Side-by-side comparison of original and reduced scales
- Summary statistics for reduction quality

### Dependencies
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- sentence-transformers >= 2.2.0
- torch >= 1.9.0

## [Unreleased]

### Planned
- Additional optimization algorithms (Particle Swarm Optimization, etc.)
- Support for factor loadings and psychometric properties preservation
- Cross-validation and stability analysis
- Interactive visualization of results
- Command-line interface (CLI)
- More comprehensive test coverage
- Performance benchmarks
- Multi-language support for item embeddings
