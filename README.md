
# Neural Network Approximation

## Project Goals

- Approximate **1D and 2D functions** $f : [0,1] \rightarrow \mathbb{R}$ and $f : [0,1]^2 \rightarrow \mathbb{R}$ using ReLU networks.
- Focus on both **convex** and **non-convex** functions.
- Compare **traditional ReLU network** with **Axon-style networks** (Greedy layer growing).
- Evaluate performance and convergence relative to **theoretical approximation bounds** through papers [1], [2].
- Understand the practical limitations of classical neural regressors when faced with complex geometries or higher dimensions.

---

## 📁 Project Structure

```
project/
├── data/           # Data generators for 1D and 2D test functions
├── model/          # Network definitions (ReLU, piecewise-linear, Axon-style, etc.)
├── utils/          # Training loop, evaluation, and visualization tools
├── tests/          # Unit tests for core components
├── experiments/    # Jupyter notebooks with results and analysis
├── setup.py        # Project environment setup
├── requirements.txt # Python dependencies
├── .gitlab-ci.yml  # GitLab CI/CD configuration
```

## Getting Started

### Prerequisites

- Python 3.12
- pip package manager

### Installation

Create and activate the Python virtual environment:

```bash
python setup.py
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate.bat     # Windows
```

Then launch Jupyter notebooks for experiments:

```bash
jupyter notebook
```

---

## Running Unit Tests

Ensure everything works correctly with:

```bash
pytest tests/
```

---

## 📌 CI/CD

GitLab CI/CD automatically runs all tests on push and merge requests via `.gitlab-ci.yml`.

---

## 📚 References

[1] Bo Liu, Yi Liang (2021). *Optimal function approximation with ReLU neural networks*, Neurocomputing, Volume 435.

[2] Fokina, Daria and Oseledets, Ivan. (2023). Growing axons: greedy learning of neural networks with application to function approximation. Russian Journal of Numerical Analysis and Mathematical Modelling. 38. 1-12.10.1515/rnam-2023-0001.

[3] Implementation of Axon algorithm for function approximation: https://github.com/dashafok/axon-approximation