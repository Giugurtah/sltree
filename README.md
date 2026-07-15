# SL-TREE: Statistical Learning Tree

**SL-TREE** is a comprehensive, production-grade statistical learning ecosystem for tree-based models, implemented in Python with a parallelized C backend. It grounds recursive partitioning in Vapnik's Statistical Learning Theory, prioritising structural honesty, predictability evaluation, and intrinsic explainability over black-box empirical risk minimisation.

## Overview

SL-TREE is organised around five technological pillars:

1. **Predictability-driven tree growth** — splits are selected via the Global Predictability Improvement (GPI) and Predictability Improvement (PI) metrics, with TWO-STAGE, FAST, CART-TWOING and TWO-CLASS algorithms.
2. **Mixture-model partitioning** — Latent Budget Trees (LBT) and Simultaneous Latent Budget Trees (SLBT) handle multi-class, imbalanced, and multilevel/hierarchical data.
3. **Interactive visual pruning** — a Dendrogram-like Tree Graph (DTG) maps information gain to branch height, enabling human-guided pruning of weak branches.
4. **Enhanced ensembles** — Random Forests and AdaBoost with any SL-TREE algorithm as the base learner.
5. **Non-parametric missing data imputation** — BINPI (Boosted Incremental Non-Parametric Imputation), rooted in Statistical Learning Theory.

### Available classes

| Class | Description |
|---|---|
| `SCTree` | Single tree (TWO-STAGE / CART-TWOING / TWO-CLASS) |
| `SCTreeWeighted` | Weighted variant of `SCTree` (AdaBoost weak learner) |
| `SLBT` | Simultaneous Latent Budget Tree for stratified/multilevel data |
| `SLBTWeighted` | Weighted variant of `SLBT` |
| `SCTreeForest` | Random Forest of `SCTree` with PPI feature importance |
| `SLBTForest` | Random Forest of `SLBT` |
| `AdaBoostForest` | AdaBoost.M1 ensemble of `SCTree` |
| `SLBTAdaBoostForest` | AdaBoost.M1 ensemble of `SLBT` |
| `BINPI` | Boosted Incremental Non-Parametric Imputation |
| `Categorizer` | KMeans-based continuous → categorical discretizer |

## Requirements

- Python ≥ 3.10
- `numpy ≥ 1.23`, `pandas ≥ 1.5`
- A C compiler (`gcc` or `cc`) and `make` — used to compile the C backend at install time

On macOS these are provided by Xcode Command Line Tools (`xcode-select --install`). On Linux install `build-essential` (`sudo apt install build-essential`).

## Installation

```bash
pip install git+https://github.com/Giugurtah/lbtree.git
```

Or clone and install locally:

```bash
git clone https://github.com/Giugurtah/lbtree.git
cd lbtree
pip install .
```

For development (editable install):

```bash
pip install -e .
```

> `pip install` automatically compiles the C shared libraries via `make`. No manual build step is needed.

## Quick start

```python
from lbtree import SCTree, Categorizer

# Discretise continuous predictors
cat = Categorizer(method="elbow", k_max=5)
X_cat = cat.fit_transform(X).astype(str)

# Fit a tree
tree = SCTree(model="twoStage", max_depth=4, feats_viewed=10)
tree.fit(X_cat, y)
predictions = tree.predict(X_cat)
```

```python
from lbtree import SLBT, plot_html

# Stratified Latent Budget Tree
tree = SLBT(homogeneity="AB", feats_viewed=5, FAST=True,
            min_pi=0.005, min_gpi=0.005, max_depth=6)
tree.fit(X_cat, y, x_s=stratum)

# Interactive visual pruning
plot_html(tree, "tree.html", visual_pruning=True)
```

```python
from lbtree import BINPI

# Non-parametric missing data imputation
imputer = BINPI(ensemble="forest", n_estimators=50)
X_imputed = imputer.fit_transform(X_cat)
```

## Citation

If you use SL-TREE in your research, please cite us.
