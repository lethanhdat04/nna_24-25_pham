# Neural-Network-Approximation

This project explores the theoretical and practical limits of approximating univariate convex functions using piecewise-linear neural networks (ReLU-based).

---

## 🧠 Project Goals

- Approximate convex functions using ReLU networks
- Evaluate performance across different architectures
- Compare with theoretical approximation bounds
- Understand the limitations of classical neural regressors

---

## 📁 Structure

```
project/
├── data/           # Data generators
├── model/          # Network definitions (ReLU, piecewise, etc.)
├── utils/          # Training, evaluation, visualizers
├── tests/          # Unit tests for CI/CD
├── experiments/    # Main notebook with theory & results
├── .gitlab-ci.yml  # GitLab Actions CI
├── setup.py        # Python-based environment setup
├── requirements.txt
├── README.md
```

---

## ⚙️ Setup

```bash
python setup.py
```

Then activate virtual environment and launch:

```bash
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate.bat     # Windows
jupyter notebook
```

---

## ✅ Run Unit Tests

```bash
pytest tests/
```

---

## 📌 CI/CD

GitLab Actions automatically runs tests on push/pull. See `.gitlab-ci.yml`.