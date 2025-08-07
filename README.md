
# Neural Network Approximation

A research project exploring optimal function approximation using ReLU neural networks and comparing traditional approaches with innovative Axon-style architectures.

## Overview

This project investigates neural network approximation capabilities for both convex and non-convex functions, comparing traditional ReLU networks with Axon-style networks that employ greedy layer growing. The implementation focuses on 1D and 2D function approximation with theoretical performance analysis.

### Key Features

- **Function Approximation**: Support for 1D functions $f : [0,1] \rightarrow \mathbb{R}$ and 2D functions $f : [0,1]^2 \rightarrow \mathbb{R}$
- **Architecture Comparison**: Traditional ReLU networks vs. Axon-style networks (greedy layer growing)
- **Function Types**: Both convex and non-convex function approximation
- **Theoretical Analysis**: Performance evaluation against theoretical approximation bounds
- **Comprehensive Testing**: Unit tests and experimental notebooks

## Project Structure

```
nna_24-25_pham/
├── data/               # Data generators for test functions
├── model/              # Network architectures (ReLU, Axon-style)
├── utils/              # Training, evaluation, and visualization
├── tests/              # Unit tests for core components
├── experiments/        # Jupyter notebooks with results
├── setup.py            # Environment setup script
├── requirements.txt    # Python dependencies
└── .gitlab-ci.yml      # CI/CD configuration
```

## Getting Started

### Prerequisites

- Python 3.12
- pip package manager
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd nna_24-25_pham
   ```

2. Run the setup script:
   ```bash
   python setup.py
   ```

3. Activate the virtual environment:
   ```bash
   # Linux/macOS
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate.bat
   ```

### Running Experiments

Launch Jupyter notebooks to explore the experimental results:

```bash
jupyter notebook experiments/
```

### Running Tests

Execute the test suite to verify functionality:

```bash
pytest tests/
```

## Architecture Details

### Traditional ReLU Networks
Standard feedforward neural networks with ReLU activation functions, trained using gradient descent optimization.

### Axon-Style Networks
Greedy layer-growing approach that incrementally builds the network architecture, offering potential advantages in approximation efficiency.

## Continuous Integration

The project uses GitLab CI/CD for automated testing. All tests run automatically on:
- Push to any branch
- Merge requests

Configuration details can be found in `.gitlab-ci.yml`.

## Research Context

This project explores practical implementations of theoretical results in neural network approximation theory, particularly focusing on:
- Optimal approximation rates for different function classes
- Practical limitations of classical neural regressors
- Performance comparison between different architectural approaches

## References

1. Bo Liu, Yi Liang (2021). *Optimal function approximation with ReLU neural networks*, Neurocomputing, Volume 435.

2. Fokina, Daria and Oseledets, Ivan. (2023). Growing axons: greedy learning of neural networks with application to function approximation. Russian Journal of Numerical Analysis and Mathematical Modelling. 38. 1-12.10.1515/rnam-2023-0001.

3. Implementation of Axon algorithm: [github.com/dashafok/axon-approximation](https://github.com/dashafok/axon-approximation)