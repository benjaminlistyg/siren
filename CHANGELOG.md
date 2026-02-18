# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
