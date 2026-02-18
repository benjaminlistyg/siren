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

- **Semantic Analysis**: Uses state-of-the-art sentence transformers to compute semantic similarity between items
- **Multi-dimensional Support**: Handle scales with multiple dimensions/factors
- **Flexible Configuration**: Specify different target item counts per dimension
- **Comprehensive Metrics**: Detailed within-dimension and between-dimension similarity scores
- **Easy-to-use API**: Simple, intuitive interface for scale reduction

## Project Structure

```
siren/
├── src/siren/              # Main package source code
│   ├── __init__.py        # Package initialization and exports
│   ├── core.py            # SIREN class and main functionality
│   ├── optimizers.py      # Optimization algorithms (ACO, SLSQP, GA, SA)
│   └── py.typed           # PEP 561 type marker
├── examples/              # Example scripts and demonstrations
│   ├── example_usage.py   # Comprehensive usage examples
│   └── verify_install.py  # Installation verification script
├── tests/                 # Test suite
│   ├── __init__.py
│   └── test_siren.py      # Unit tests
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
pip install siren-scale
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

## Algorithm Details

### Ant Colony Optimization (ACO)

SIREN's default optimizer uses ACO, which has been shown to be particularly effective for psychometric scale reduction (Schroeders et al., 2016). Key features:

- **Pheromone-based learning**: Iteratively improves solutions based on successful paths
- **Balance of exploration and exploitation**: Controlled by parameters α, β, and q₀
- **Constraint satisfaction**: Naturally handles complex constraints about item selection
- **Parallel solution construction**: Multiple "ants" explore solution space simultaneously

### Objective Function

SIREN optimizes for:
1. **High within-dimension similarity**: Items in the same dimension should be semantically similar
2. **Low between-dimension similarity**: Items from different dimensions should be distinct
3. **Constraint satisfaction**: Exact number of items selected per dimension

The objective function is:
```
Score = Σ(2.0 × within_similarity - between_similarity) - penalties
```

## API Reference

### SIREN Class

```python
SIREN(model_name='all-MiniLM-L12-v2', optimizer=None)
```

**Parameters:**
- `model_name` (str): Name of the sentence-transformer model to use
- `optimizer` (OptimizationMethod): Optimization method instance (default: AntColonyOptimizer)

**Methods:**

#### `reduce_scale()`
```python
reduce_scale(
    items: List[str],
    dimension_labels: List[str],
    items_per_dim: Union[int, Dict[str, int]] = 2,
    suppress_details: bool = False,
    n_tries: int = 5
) -> Tuple[Dict[str, Dict], pd.DataFrame]
```

Reduce a psychometric scale to fewer items.

**Parameters:**
- `items`: List of item texts
- `dimension_labels`: List of dimension labels (same length as items)
- `items_per_dim`: Target number of items per dimension (int or dict)
- `suppress_details`: If True, suppress optimization output
- `n_tries`: Number of random initializations to try

**Returns:**
- Tuple of (results dictionary, similarity matrix DataFrame)

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
```

## Performance Considerations

- **Model Selection**: Lighter models (MiniLM) are faster; heavier models (MPNet) may be more accurate
- **Number of Tries**: More tries increase quality but take longer
- **Optimizer Selection**: ACO generally provides best results for scale reduction
- **Suppressing Output**: Use `suppress_details=True` for faster execution in production

## References

- Schroeders, U., Wilhelm, O., & Olaru, G. (2016). Meta-heuristics in short scale construction: Ant colony optimization and genetic algorithm. *PLOS ONE*, *11*(11), e0167110.

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
