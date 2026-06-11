"""
lbtree/forest.py
================
SCTreeForest — Random Forest of SCTree with PPI-based feature importance.

Algorithm overview
------------------
Each tree is trained on a bootstrap sample of the data with a random
feature subset (feature subsampling).  The final prediction is a
majority vote across all trees.

PPI importance
--------------
For each internal node *t* that splits on feature *x_j*:

    contribution(t, x_j) = (N_t / N_root) * pi_t

where N_t is the number of observations at t and pi_t is the PPI
(Proportional Predictability Improvement).  The importance of x_j
is the sum of contributions across all nodes that use it, averaged
over all trees and normalised to sum to 1.

OOB score
---------
Each bootstrap sample leaves ~36.8 % of observations out-of-bag.
The OOB score is the accuracy of majority-vote predictions among
the trees that did not see each sample — a free estimate of
out-of-sample accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter

from .sctree import SCTree


class SCTreeForest:
    """
    Random Forest of SCTree with PPI-weighted feature importance.

    Parameters
    ----------
    n_estimators : int, default 10
        Number of trees.
    model : {"twoStage", "twoing"}, default "twoStage"
        Split algorithm for each tree.
    max_features : int | float | "sqrt" | "log2" | None, default "sqrt"
        Number of features sampled per tree:
        - int   → exact count
        - float → fraction of total columns
        - "sqrt"→ floor(sqrt(p))
        - "log2"→ floor(log2(p))
        - None  → all features (no subsampling)
    bootstrap : bool, default True
        If True, each tree is trained on a bootstrap sample of rows.
        If False, all rows are used (only feature subsampling applies).
    random_state : int | None
        Seed for reproducibility.
    **tree_kwargs
        Additional keyword arguments forwarded to each SCTree
        (e.g. min_ppi, max_depth, feats_viewed, …).

    Attributes (set after fit)
    --------------------------
    estimators_ : list[SCTree]
    estimators_features_ : list[list[str]]
    ppi_importance_ : dict[str, float]
        Feature importance normalised to sum 1, sorted descending.
    oob_score_ : float | None
        OOB accuracy (only when bootstrap=True).
    classes_ : np.ndarray
    """

    def __init__(
        self,
        n_estimators: int = 10,
        model: str        = "twoStage",
        max_features      = "sqrt",
        bootstrap: bool   = True,
        random_state      = None,
        **tree_kwargs,
    ):
        self.n_estimators = n_estimators
        self.model        = model
        self.max_features = max_features
        self.bootstrap    = bootstrap
        self.random_state = random_state
        self.tree_kwargs  = tree_kwargs

        self.estimators_          : list[SCTree]       = []
        self.estimators_features_ : list[list[str]]    = []
        self.ppi_importance_      : dict               = {}
        self.oob_score_           : float | None       = None
        self.classes_             : np.ndarray | None  = None

        self._rng = np.random.default_rng(random_state)

    # ================================================================
    #  PUBLIC — fit / predict / predict_proba
    # ================================================================

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SCTreeForest":
        """
        Train ``n_estimators`` trees on bootstrap samples of (X, y).

        For each tree *b*:
          1. Row bootstrap sample  → X_boot, y_boot
          2. Feature subsample     → reduced X_boot
          3. Fit SCTree on (X_boot_reduced, y_boot)
          4. Collect OOB predictions

        After all trees:
          5. Compute OOB score
          6. Compute ppi_importance_

        Returns
        -------
        self
        """
        self.classes_             = np.unique(y)
        self.estimators_          = []
        self.estimators_features_ = []

        n_samples, n_feats    = X.shape
        n_features_to_use     = self._resolve_max_features(n_feats)
        all_columns           = list(X.columns)

        # OOB: per-sample list of predictions from trees that did not see that sample
        oob_preds = [[] for _ in range(n_samples)]

        for b in range(self.n_estimators):
            if (b + 1) % 10 == 0 or b == 0:
                print(f"  Forest: {b + 1}/{self.n_estimators} trees fitted")

            # 1. Bootstrap row sampling
            if self.bootstrap:
                boot_idx = self._rng.integers(0, n_samples, size=n_samples)
            else:
                boot_idx = np.arange(n_samples)

            oob_idx = np.setdiff1d(np.arange(n_samples), boot_idx)

            # 2. Feature subsampling
            feat_idx   = self._rng.choice(n_feats, size=n_features_to_use, replace=False)
            feat_names = list(X.columns[feat_idx])

            X_boot = X.iloc[boot_idx][feat_names].reset_index(drop=True)
            y_boot = y.iloc[boot_idx].reset_index(drop=True)

            # 3. Fit single tree
            tree = SCTree(model=self.model, **self.tree_kwargs)
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)
            self.estimators_features_.append(feat_names)

            # 4. OOB predictions
            if self.bootstrap and len(oob_idx) > 0:
                X_oob = X.iloc[oob_idx][feat_names].reset_index(drop=True)
                preds = tree.predict(X_oob)
                for local_i, global_i in enumerate(oob_idx):
                    oob_preds[global_i].append(preds[local_i])

        # 5. OOB score
        if self.bootstrap:
            self.oob_score_ = self._compute_oob_score(y, oob_preds)
            print(f"\nOOB Score: {self.oob_score_:.4f}")

        # 6. PPI importance
        self.ppi_importance_ = self._compute_ppi_importance(all_columns)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Majority-vote prediction across all trees."""
        self._check_fitted()
        votes = [Counter() for _ in range(len(X))]
        for tree, feat_names in zip(self.estimators_, self.estimators_features_):
            available = [f for f in feat_names if f in X.columns]
            preds = tree.predict(X[available].reset_index(drop=True))
            for i, pred in enumerate(preds):
                votes[i][pred] += 1
        return np.array([v.most_common(1)[0][0] for v in votes])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Class probability as vote fraction.

        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
            Columns ordered as self.classes_.
        """
        self._check_fitted()
        n_samples = len(X)
        n_classes = len(self.classes_)
        proba     = np.zeros((n_samples, n_classes))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for tree, feat_names in zip(self.estimators_, self.estimators_features_):
            available = [f for f in feat_names if f in X.columns]
            preds = tree.predict(X[available].reset_index(drop=True))
            for i, pred in enumerate(preds):
                if pred in class_to_idx:
                    proba[i, class_to_idx[pred]] += 1

        proba /= len(self.estimators_)
        return proba

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "model":        self.model,
            "max_features": self.max_features,
            "bootstrap":    self.bootstrap,
            "random_state": self.random_state,
            **self.tree_kwargs,
        }

    def set_params(self, **params) -> "SCTreeForest":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.tree_kwargs[k] = v
        return self

    # ================================================================
    #  PRIVATE — PPI importance
    # ================================================================

    def _compute_ppi_importance(self, all_columns: list) -> dict:
        importances = {col: 0.0 for col in all_columns}
        for tree in self.estimators_:
            if tree.root is None:
                continue
            root_N = tree.root.N
            self._accumulate_ppi_importance(tree.root, importances, root_N)
        # Average over n_estimators
        for col in importances:
            importances[col] /= self.n_estimators
        # Normalise to sum 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}
        return dict(sorted(importances.items(), key=lambda x: -x[1]))

    def _accumulate_ppi_importance(self, node, importances: dict, root_N: int) -> None:
        if node is None or node._is_leaf_node():
            return
        if (node.feature in importances
                and node.pi is not None
                and node.N  is not None):
            importances[node.feature] += (node.N / root_N) * node.pi
        self._accumulate_ppi_importance(node.left,  importances, root_N)
        self._accumulate_ppi_importance(node.right, importances, root_N)

    # ================================================================
    #  PRIVATE — OOB
    # ================================================================

    @staticmethod
    def _compute_oob_score(y: pd.Series, oob_preds: list) -> float:
        correct = counted = 0
        for i, preds in enumerate(oob_preds):
            if preds:
                majority = Counter(preds).most_common(1)[0][0]
                if majority == y.iloc[i]:
                    correct += 1
                counted += 1
        return correct / counted if counted > 0 else float("nan")

    # ================================================================
    #  PRIVATE — helpers
    # ================================================================

    def _resolve_max_features(self, n_feats: int) -> int:
        mf = self.max_features
        if mf is None:            return n_feats
        if isinstance(mf, int):   return min(mf, n_feats)
        if isinstance(mf, float): return max(1, int(mf * n_feats))
        if mf == "sqrt":          return max(1, int(np.sqrt(n_feats)))
        if mf == "log2":          return max(1, int(np.log2(n_feats)))
        raise ValueError(f"max_features='{mf}' is not valid.")

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError("Forest not fitted. Call fit() first.")

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        status = (f"{len(self.estimators_)} trees fitted"
                  if self.estimators_ else "not fitted")
        return (
            f"SCTreeForest("
            f"n_estimators={self.n_estimators}, "
            f"model='{self.model}', "
            f"max_features={self.max_features!r}, "
            f"bootstrap={self.bootstrap}, "
            f"[{status}])"
        )
