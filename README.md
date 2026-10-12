# SIREN - Semantic Item Reduction Engine

[![PyPI version](https://badge.fury.io/py/siren-scale.svg)](https://badge.fury.io/py/siren-scale)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SIREN is an automated psychometric scale reduction tool that uses semantic similarity and advanced optimization algorithms to create shorter, more efficient psychological and behavioral assessment scales while maintaining their psychometric properties.

## Features

- **Multiple Optimization Algorithms**:
  - **Ant Colony Optimization (ACO)** - Default and recommended method, particularly effective for combinatorial optimization
  - **SLSQP** - Sequential Least Squares Programming
  - **Genetic Algorithm** - Differential Evolution approach
  - **Simulated Annealing** - Probabilistic optimization technique

- **Semantic Analysis**: Uses sentence transformers (any `sentence-transformers` model by name) to compute semantic similarity between items; or bring your own embeddings / similarity matrix
- **Two Objectives**: semantic *coverage* (content-representative items) or *coherence* (mutually similar items), both with a between-dimension penalty
- **Multi-dimensional Support**: Handle scales with multiple dimensions/factors, with exact per-dimension item quotas; reverse-keyed items handled inside the objective
- **Structure Discovery**: No dimension key? `infer_dimensions()` clusters items from text
- **Deterministic Baselines**: nearest-centroid and greedy facility-location selectors (`method='centroid' | 'greedy'`) for comparison with the optimizer
- **Diagnostics**: dropped-item coverage audit and content gaps, percentile of any short form against random dimension-stratified subsets, near-duplicate (redundancy) detection, one-table fidelity report, candidate-item vetting
- **Plots**: 2-D item maps (PCA/MDS) and similarity heatmaps (`pip install siren-scale[plots]`)
- **Reproducible**: `seed=` controls restarts and every optimizer; `optimizer_options=` exposes ACO/GA/SA/SLSQP settings

## Project Structure

```
siren/
├── src/siren/              # Main package source code
│   ├── __init__.py        # Package initialization and exports
│   ├── core.py            # SIREN class: embedding, selection, inference, vetting
│   ├── optimizers.py      # Optimization algorithms (ACO, SLSQP, GA, SA)
│   ├── selection.py       # Deterministic selectors (centroid, greedy coverage, random)
│   ├── diagnostics.py     # Coverage audit, null percentiles, redundancy, fidelity report
│   ├── plots.py           # Optional matplotlib plots
│   └── py.typed           # PEP 561 type marker
├── examples/              # Example scripts and demonstrations
│   ├── example_usage.py   # Comprehensive usage examples
│   ├── diagnostics_example.py  # Diagnostics, selectors, structure discovery, plots
│   └── verify_install.py  # Installation verification script
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_siren.py      # Core tests (load the default model)
│   └── test_selection_diagnostics.py  # Model-free tests
├── .github/workflows/     # CI/CD
│   ├── tests.yml          # Test workflow
│   └── publish.yml        # PyPI publish workflow
├── pyproject.toml         # Package configuration and dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── CHANGELOG.md           # Version history
├── MANIFEST.in            # Package manifest
└── .gitignore             # Git ignore rules
```

## Installation

### From PyPI (recommended)

```bash
pip install siren-scale          # core
pip install siren-scale[plots]   # + matplotlib for plot_item_map / plot_similarity_heatmap
```

### From source

```bash
git clone https://github.com/benjaminlistyg/siren.git
cd siren
pip install -e .
```

### Development installation

```bash
git clone https://github.com/benjaminlistyg/siren.git
cd siren
pip install -e ".[dev]"
```

## Quick Start

```python
from siren import SIREN

# Your scale items
items = [
    "I feel confident in my abilities.",
    "I believe I can achieve my goals.",
    "I trust my decision-making skills.",
    # ... more items
]

# Dimension labels for each item
dimensions = ['confidence', 'confidence', 'confidence', ...]

# Initialize SIREN with default ACO optimizer
siren = SIREN()

# Reduce scale to 3 items per dimension
result, similarity_matrix = siren.reduce_scale(
    items=items,
    dimension_labels=dimensions,
    items_per_dim=3,
    n_tries=5
)

# Print comparison
siren.print_comparison(items, dimensions, result, items_per_dim=3)
```

## Usage

### Basic Usage

```python
from siren import SIREN

# Create SIREN instance with default ACO optimizer
siren = SIREN()

# Perform scale reduction
result, sim_matrix = siren.reduce_scale(
    items=items,
    dimension_labels=dimensions,
    items_per_dim=2
)

# Access results
for dimension, data in result.items():
    print(f"Dimension {dimension}:")
    print(f"  Selected items: {data['items']}")
    print(f"  Item indices: {data['indices']}")
    print(f"  Within-dimension similarity: {data['metrics']['within_dim_scores']}")
    print(f"  Between-dimension similarity: {data['metrics']['between_dim_scores']}")
```

### Using Different Optimizers

```python
from siren import SIREN

# Ant Colony Optimization (recommended, default)
siren_aco = SIREN.with_aco()

# SLSQP Optimizer
siren_slsqp = SIREN.with_slsqp()

# Genetic Algorithm
siren_ga = SIREN.with_genetic_algorithm()

# Simulated Annealing
siren_sa = SIREN.with_simulated_annealing()

# Or switch optimizer on existing instance
siren = SIREN()
from siren.optimizers import GeneticAlgorithmOptimizer
siren.set_optimizer(GeneticAlgorithmOptimizer())
```

### Variable Items Per Dimension

```python
# Specify different target counts for each dimension
items_per_dim = {
    'confidence': 3,
    'social_support': 2,
    'stress_management': 4,
    'goal_orientation': 3
}

result, _ = siren.reduce_scale(
    items=items,
    dimension_labels=dimensions,
    items_per_dim=items_per_dim,
    n_tries=5
)
```

### Using Custom Models

```python
# Use a different sentence transformer model
siren = SIREN(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# For multilingual support
siren = SIREN(model_name='paraphrase-multilingual-mpnet-base-v2')
```

### Suppressing Output

```python
# For use in scripts or production
result, _ = siren.reduce_scale(
    items=items,
    dimension_labels=dimensions,
    items_per_dim=3,
    suppress_details=True,
    n_tries=5
)
```

### Precomputed Embeddings or Similarity Matrix

`reduce_scale()` accepts `embeddings=` (n_items × dim) or `similarity_matrix=` (n_items × n_items) and skips the encoder. Any symmetric similarity works, e.g. predicted inter-item correlations from a fine-tuned model. With `SIREN(model_name=None)` no model is ever loaded.

```python
siren = SIREN(seed=0)
E = siren.embed(items)                                   # encode once
S = siren.similarity_from_embeddings(E, reverse_scored=[5, 6])

result, _ = siren.reduce_scale(items, dimensions, items_per_dim=3,
                               similarity_matrix=S, objective="coverage")

# or, with a matrix from elsewhere and no model at all
siren = SIREN(model_name=None)
result, _ = siren.reduce_scale(items, dimensions, items_per_dim=3, similarity_matrix=my_matrix)
```

### Reproducibility and Optimizer Settings

```python
siren = SIREN(seed=42, optimizer_options={"n_ants": 40, "maxiter": 200})   # defaults for every call
result, _ = siren.reduce_scale(items, dimensions, items_per_dim=3,
                               seed=7,                                      # per-call override
                               optimizer_options={"maxiter": 300})          # merged over instance defaults
```

ACO accepts `n_ants`, `maxiter`, `alpha`, `beta`, `rho`, `q0`; the genetic algorithm `maxiter`, `popsize`, `tol`; SLSQP and simulated annealing `maxiter`. The seed drives the restarts, ACO, and the scipy-based optimizers.

### Deterministic Selectors

Two single-pass selectors are available as baselines for the optimizer (and reproduce the selection rule of clustering-based tools):

```python
result, _ = siren.reduce_scale(items, dimensions, items_per_dim=3, method="centroid")  # most central items
result, _ = siren.reduce_scale(items, dimensions, items_per_dim=3, method="greedy")    # greedy facility-location coverage

from siren import select_centroid, select_greedy_coverage, select_random_stratified
dim_idx = siren.get_dimension_indices(dimensions)
chosen = select_greedy_coverage(S, dim_idx, items_per_dim=3)   # {dimension: [indices]}
```

### Structure Discovery

```python
labels = siren.infer_dimensions(items)                 # k by silhouette over 2..10
labels = siren.infer_dimensions(items, n_dimensions=4)
result, _ = siren.reduce_scale(items, None, items_per_dim=2)   # infers labels; see siren.last_dimension_labels_
```

Prefer the instrument's own dimension key when one exists; inference is for pools without one.

### Diagnostics

```python
from siren import coverage_report, coverage_percentile, redundancy, fidelity_report

# What did the short form give up? Each dropped item, its closest retained proxy, and content gaps
audit = siren.coverage_report(items=items, gap_threshold=0.5)   # uses the last reduce_scale() call
audit["dropped"]; audit["gaps"]; audit["summary"]

# Where does the form fall among random dimension-stratified subsets of the same size?
fid = siren.fidelity_report(n_random=1000)
fid["table"]                      # per-dimension coherence, coverage, between + overall
fid["percentiles"]["coverage"]    # {'observed', 'percentile', 'null_mean', 'null_sd', 'z', ...}

# The same audit for *any* proposed short form (including data-driven ones)
pct = coverage_percentile(S, dim_idx, {"A": [0, 2, 5], "B": [7, 9, 10]}, n_random=1000, seed=0)

# Near-duplicate items in the pool
red = redundancy(S, threshold=0.8, items=items, dimension_labels=dimensions)
red["pairs"]; red["clusters"]; red["suggest_remove"]

# Vet candidate items against the existing scale
fit = siren.item_fit(["I can count on my colleagues."], items, dimensions)
```

### Plots

```python
from siren import plot_item_map, plot_similarity_heatmap   # requires matplotlib
plot_item_map(embeddings=E, dimension_labels=dimensions, selected=result, items=items, annotate=True)
plot_similarity_heatmap(S, dimension_labels=dimensions, selected=result, items=items)
```

## Algorithm Details

### Ant Colony Optimization (ACO)

SIREN's default optimizer uses ACO, which has been shown to be particularly effective for psychometric scale reduction (Schroeders et al., 2016). Key features:

- **Pheromone-based learning**: Iteratively improves solutions based on successful paths
- **Balance of exploration and exploitation**: Controlled by parameters α, β, and q₀
- **Constraint satisfaction**: Naturally handles complex constraints about item selection
- **Parallel solution construction**: Multiple "ants" explore solution space simultaneously

### Objective Functions

SIREN supports two objectives (`objective=` parameter of `reduce_scale()`), both of which also minimize between-dimension similarity and enforce exact item counts per dimension:

1. **`'coherence'`** (default): maximizes the mean pairwise semantic similarity among the *selected* items within each dimension. Selected items are mutually similar in meaning. Note: degenerate when `items_per_dim=1` (no within-dimension pairs exist) — use `'coverage'` there.
2. **`'coverage'`**: facility-location criterion — maximizes, for each *dropped* item, its maximum similarity to a retained item of the same dimension. Selected items are semantically *representative* of the full dimension pool, preserving content breadth instead of rewarding redundancy. Well-defined for any `items_per_dim >= 1`.

The objective function is:
```
Score = Σ(2.0 × within_score - between_similarity) - penalties
```
where `within_score` is mean pairwise similarity (coherence) or mean max-similarity coverage of dropped items (coverage).

## API Reference

### SIREN Class

```python
SIREN(model_name='all-MiniLM-L12-v2', optimizer=None, seed=None, optimizer_options=None, model=None)
```

**Parameters:**
- `model_name` (str | None): Name of the sentence-transformer model to use; loaded lazily on first use. `None` disables model loading (pass `embeddings=` or `similarity_matrix=` to `reduce_scale()`)
- `optimizer` (OptimizationMethod): Optimization method instance (default: AntColonyOptimizer)
- `seed` (int | None): Default random seed for `reduce_scale()`
- `optimizer_options` (dict | None): Default options forwarded to the optimizer
- `model`: An already-loaded encoder exposing `encode(texts)`; takes precedence over `model_name`

**Attributes set by the last `reduce_scale()` call:** `last_similarity_`, `last_embeddings_`, `last_dimension_labels_`, `last_selection_`

**Methods:**

#### `reduce_scale()`
```python
reduce_scale(
    items: List[str],
    dimension_labels: Optional[List[str]] = None,
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
    n_dimensions: Optional[int] = None,
) -> Tuple[Dict[str, Dict], pd.DataFrame]
```

Reduce a psychometric scale to fewer items.

**Parameters:**
- `items`: List of item texts
- `dimension_labels`: List of dimension labels (same length as items); `None` infers them from text
- `items_per_dim`: Target number of items per dimension (int or dict)
- `suppress_details`: If True, suppress optimization output
- `n_tries`: Number of random initializations to try (`method='optimize'`)
- `reverse_scored`: 0-indexed positions of reverse-scored items; their similarity-matrix rows/columns are sign-flipped before selection
- `objective`: `'coherence'` (default) or `'coverage'`; see Objective Functions above
- `embeddings` / `similarity_matrix`: precomputed inputs; skip the encoder
- `method`: `'optimize'` (default), `'centroid'`, or `'greedy'`
- `seed`: random seed for this call
- `optimizer_options`: options forwarded to the optimizer for this call
- `n_dimensions`: number of dimensions to infer when `dimension_labels is None` (default: silhouette)

**Returns:**
- Tuple of (results dictionary, similarity matrix DataFrame). `result[dim]['metrics']` holds `within_dim_scores`, `between_dim_scores`, and `coverage`.

#### Other methods
- `embed(items)`, `similarity_from_embeddings(E, reverse_scored=None)`, `compute_similarity(items=None, embeddings=None, reverse_scored=None)`
- `infer_dimensions(items=None, embeddings=None, n_dimensions=None, max_dimensions=10, seed=None)`
- `item_fit(candidates, items, dimension_labels, embeddings=None, candidate_embeddings=None, redundancy_threshold=0.8)`
- `fidelity_report(result=None, similarity_matrix=None, dimension_labels=None, n_random=1000, seed=None)`
- `coverage_report(result=None, similarity_matrix=None, dimension_labels=None, items=None, gap_threshold=0.5)`

### Module functions

- `siren.selection`: `select_centroid(S, dim_indices, items_per_dim)`, `select_greedy_coverage(...)`, `select_random_stratified(dim_indices, items_per_dim, rng=None)`
- `siren.diagnostics`: `coverage_score`, `coherence_score`, `between_score`, `coverage_report`, `coverage_percentile(S, dim_indices, selected, statistic='coverage', n_random=1000, seed=None)`, `redundancy(S, threshold=0.8, items=None, dimension_labels=None, within_dimension_only=False)`, `fidelity_report`
- `siren.plots`: `plot_item_map(embeddings=None, similarity_matrix=None, dimension_labels=None, selected=None, items=None, method='pca', annotate=False, seed=0, ax=None)`, `plot_similarity_heatmap(S, dimension_labels=None, selected=None, items=None, absolute=True, ax=None)`

#### `print_comparison()`
```python
print_comparison(
    items: List[str],
    dimension_labels: List[str],
    result: Dict[str, Dict],
    items_per_dim: Union[int, Dict[str, int]],
    show_summary: bool = False
)
```

Print original and shortened scales side by side.

### Factory Methods

- `SIREN.with_aco()`: Create SIREN with Ant Colony Optimization
- `SIREN.with_slsqp()`: Create SIREN with SLSQP optimizer
- `SIREN.with_genetic_algorithm()`: Create SIREN with Genetic Algorithm
- `SIREN.with_simulated_annealing()`: Create SIREN with Simulated Annealing

## Examples

See the `examples/` directory for complete working examples:

```bash
python examples/example_usage.py
python examples/diagnostics_example.py
```

## Performance Considerations

- **Model Selection**: Lighter models (MiniLM) are faster; heavier models (MPNet) may be more accurate
- **Number of Tries**: More tries increase quality but take longer
- **Optimizer Selection**: ACO generally provides best results for scale reduction
- **Suppressing Output**: Use `suppress_details=True` for faster execution in production

## References

- Schroeders, U., Wilhelm, O., & Olaru, G. (2016). Meta-heuristics in short scale construction: Ant colony optimization and genetic algorithm. *PLOS ONE*, *11*(11), e0167110.

## Roadmap

Looking to contribute? Here's what's on our to-do list:

### Documentation
- [ ] ReadTheDocs/Sphinx site — API docs auto-generated from docstrings, a tutorial, and algorithm explanation pages
- [ ] Add a `CONTRIBUTING.md` — contribution workflow, dev setup, code style expectations, how to run tests
- [ ] Jupyter notebook tutorials — walkthrough examples with real (or realistic) scale data showing end-to-end usage

### Testing
- [ ] Expand test coverage — 28 tests right now. Good targets: edge cases (single dimension, 1 item per dim, all identical items), each optimizer individually, `print_comparison` output, and metrics correctness
- [x] Model-free tests via `SIREN(model_name=None)` + precomputed similarity matrices (`tests/test_selection_diagnostics.py`)
- [x] Add a test for reproducibility — same seed should give same results

### Benchmarking & Validation
- [ ] Build a benchmark suite — compare optimizers on known scales (speed, solution quality, convergence) and produce a table/chart for the docs
- [ ] Validate against published results — replicate the Schroeders et al. (2016) findings or similar published scale reductions to show SIREN produces comparable output

### Features
- [ ] CLI interface — `siren reduce --items items.csv --dims dims.csv --per-dim 3` using `click` or `argparse`
- [ ] Results export — save results to CSV/JSON, not just print to console
- [x] Visualization — `plot_similarity_heatmap`, `plot_item_map`
- [ ] Cross-validation / stability analysis — run reduction multiple times with different seeds and report how stable the item selection is (seeds are now exposed; a helper that reports Jaccard stability is still to do)

### Code Quality
- [ ] Add `py.typed` + full type annotations — get `mypy --strict` passing
- [ ] Replace `print()` calls with proper logging — the optimizers currently use `print` for progress; should use `logging` or a callback pattern so users can plug in `tqdm`, logging, or silence it cleanly

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: https://github.com/benjaminlistyg/siren/issues
- **Discussions**: https://github.com/benjaminlistyg/siren/discussions
