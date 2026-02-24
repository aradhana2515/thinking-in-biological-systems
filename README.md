[![CI](https://github.com/aradhana2515/thinking-in-biological-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/aradhana2515/thinking-in-biological-systems/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
# thinking-in-biological-systems

Computational tools for simulating and analyzing dynamical regimes in
biological systems using interpretable machine learning.

This repository is designed to build mechanistic intuition from
time-series data by combining:

-   Dynamical system simulation
-   Feature extraction from trajectories
-   Regime classification using interpretable ML
-   Mechanistic reasoning about biological constraints

The emphasis is not on black-box prediction, but on understanding how
dynamic behaviors emerge from underlying rules.

------------------------------------------------------------------------

## Motivation

Biological systems are inherently dynamic. They oscillate, bifurcate,
stabilize, collapse, and evolve.

Examples include:

-   Gene regulatory networks
-   Evolutionary selection dynamics
-   Neural activity regimes
-   Protein interaction feedback circuits
-   Automated closed-loop experimentation

This repository provides a structured sandbox for:

1.  Generating synthetic dynamical data
2.  Extracting interpretable features
3.  Classifying regimes (stable, oscillatory, bistable, etc.)
4.  Connecting statistical signatures back to mechanism

All examples use synthetic or public data to develop transferable
intuition.

------------------------------------------------------------------------

## Repository Structure

    src/tibs/
        models/          # Mechanistic biological models
        simulators/      # Trajectory generation and ensemble simulation
        features/        # Time-series feature extraction
        ml/              # Interpretable ML classifiers
    examples/            # Example workflows
    tools/               # Utility scripts
    tests/               # Unit tests

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/aradhana2515/thinking-in-biological-systems.git
cd thinking-in-biological-systems
```

Create a clean environment:

``` bash
conda create -n tibs python=3.10 -y
conda activate tibs
```

Install in editable mode:

``` bash
pip install -e .
```

Verify installation:

``` bash
pytest -q
```

------------------------------------------------------------------------

## Quick Demo

Run a basic simulation:

``` python
from tibs.models.gene_expression import GeneExpressionModel
from tibs.simulators.ensemble import run_ensemble

model = GeneExpressionModel()
results = run_ensemble(model, n_runs=10)

print(results.summary())
```

Or run a basic check:

``` bash
python tools/preflight_check.py
```

------------------------------------------------------------------------

## Core Concepts

### Mechanistic Simulation

Models are defined as explicit dynamical systems (ODE-based or
discrete-time approximations). Parameters can be varied to explore
regime transitions.

### Time-Series Feature Extraction

Trajectories are converted into interpretable descriptors such as:

-   Amplitude\
-   Frequency\
-   Stability metrics\
-   Peak statistics\
-   Convergence behavior

### Interpretable Machine Learning

Regime classification uses transparent models (e.g., logistic
regression, decision trees) to map feature space to dynamical
categories.

The focus is on understanding which features drive classification
decisions.

------------------------------------------------------------------------

## Intended Applications

-   Systems biology\
-   Neuroscience dynamics\
-   Evolutionary modeling\
-   Protein engineering feedback systems\
-   Closed-loop automated experimentation

The framework is designed to be extensible to real experimental
time-series data.

------------------------------------------------------------------------

## Development

Format code:

``` bash
ruff check . --fix
ruff format .
```

Run tests:

``` bash
pytest
```

------------------------------------------------------------------------

## Author

Aradhana\
PhD Student, Biomedical Engineering\
Duke University
