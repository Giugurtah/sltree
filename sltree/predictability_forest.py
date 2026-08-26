# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/predictability_forest.py
================
PredictabilityForest — Random Forest of PredictabilityTree with PPI-based feature importance.

Algorithm overview
------------------
Each tree is trained on a bootstrap sample of the data with a random
feature subset (feature subsampling).  The final prediction is a
majority vote across all trees (classification) or an average of leaf
mean predictions (twoClass / regression).

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
- Classification (twoStage / twoing): accuracy of majority-vote predictions.
- Regression (twoClass): R² of averaged leaf-mean predictions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from collections import Counter

from .predictability_tree import PredictabilityTree


class PredictabilityForest:
    """
    Random Forest of PredictabilityTree with PPI-weighted feature importance.

    Parameters
    ----------
    n_estimators : int, default 10
        Number of trees.
    model : {"twoStage", "twoing", "twoClass"}, default "twoStage"
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
        Additional keyword arguments forwarded to each PredictabilityTree
        (e.g. min_ppi, max_depth, feats_viewed, …).

    Attributes (set after fit)
    --------------------------
    estimators_ : list[PredictabilityTree]
    estimators_features_ : list[list[str]]
    ppi_importance_ : dict[str, float]
        Feature importance normalised to sum 1, sorted descending.
    oob_score_ : float | None
        OOB accuracy (classification) or R² (twoClass), only when bootstrap=True.
    classes_ : np.ndarray
        Unique class labels (classification) or unique target values (twoClass).
    """

    def __init__(
        self,
        n_estimators: int    = 10,
        model: str           = "twoStage",
        max_features         = "sqrt",
        feature_sampling: str = "tree",
        bootstrap: bool      = True,
        random_state         = None,
        verbose: int           = 0,
        **tree_kwargs,
    ):
        if feature_sampling not in ("tree", "node"):
            raise ValueError(
                f"feature_sampling='{feature_sampling}' not supported. "
                "Choose 'tree' (random subspace, default) or 'node' (true RF)."
            )
        self.n_estimators     = n_estimators
        self.model            = model
        self.max_features     = max_features
        self.feature_sampling = feature_sampling
        self.bootstrap        = bootstrap
        self.random_state     = random_state
        self.verbose       = verbose
        self.tree_kwargs      = tree_kwargs

        self.estimators_          : list[PredictabilityTree]       = []
        self.estimators_features_ : list[list[str]]    = []
        self.ppi_importance_      : dict               = {}
        self.oob_score_           : float | None       = None
        self.classes_             : np.ndarray | None  = None

        self._rng = np.random.default_rng(random_state)

    # ================================================================
    #  PUBLIC — fit / predict / predict_proba
    # ================================================================

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PredictabilityForest":
        """
        Train ``n_estimators`` trees on bootstrap samples of (X, y).

        For each tree *b*:
          1. Row bootstrap sample  → X_boot, y_boot
          2. Feature subsample     → reduced X_boot
          3. Fit PredictabilityTree on (X_boot_reduced, y_boot)
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
            if self.verbose and ((b + 1) % 10 == 0 or b == 0):
                print(f"  Forest: {b + 1}/{self.n_estimators} trees fitted")

            # 1. Bootstrap row sampling
            if self.bootstrap:
                boot_idx = self._rng.integers(0, n_samples, size=n_samples)
            else:
                boot_idx = np.arange(n_samples)

            oob_idx = np.setdiff1d(np.arange(n_samples), boot_idx)

            # 2. Feature subsampling
            if self.feature_sampling == "tree":
                # random subspace: subsample columns once per tree (existing behaviour)
                feat_idx   = self._rng.choice(n_feats, size=n_features_to_use, replace=False)
                feat_names = list(X.columns[feat_idx])
                X_boot     = X.iloc[boot_idx][feat_names].reset_index(drop=True)
                tree       = PredictabilityTree(model=self.model, **self.tree_kwargs)
            else:
                # true RF: all columns reach the tree; each node subsamples independently
                feat_names = list(X.columns)
                X_boot     = X.iloc[boot_idx].reset_index(drop=True)
                tree_seed  = int(self._rng.integers(0, 2**31))
                tree_rng   = np.random.default_rng(tree_seed)
                tree       = PredictabilityTree(
                    model=self.model,
                    node_max_features=n_features_to_use,
                    rng=tree_rng,
                    **self.tree_kwargs,
                )

            y_boot = y.iloc[boot_idx].reset_index(drop=True)

            # 3. Fit single tree
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
            self.oob_score_ = self._compute_oob_score(y, oob_preds, self.model)
            label = "OOB R²" if self.model == "twoClass" else "OOB Score"
            if self.verbose:
                print(f"\n{label}: {self.oob_score_:.4f}")

        # 6. PPI importance
        self.ppi_importance_ = self._compute_ppi_importance(all_columns)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediction across all trees.

        - Classification (twoStage / twoing): majority vote.
        - Regression (twoClass): average of leaf mean predictions.
        """
        self._check_fitted()
        if self.model == "twoClass":
            return self._predict_regression(X)
        votes = [Counter() for _ in range(len(X))]
        for tree, feat_names in zip(self.estimators_, self.estimators_features_):
            available = [f for f in feat_names if f in X.columns]
            preds = tree.predict(X[available].reset_index(drop=True))
            for i, pred in enumerate(preds):
                votes[i][pred] += 1
        return np.array([v.most_common(1)[0][0] for v in votes])

    def _predict_regression(self, X: pd.DataFrame) -> np.ndarray:
        """Average leaf-mean predictions across all trees (twoClass only)."""
        n_samples = len(X)
        sums      = np.zeros(n_samples, dtype=np.float64)
        for tree, feat_names in zip(self.estimators_, self.estimators_features_):
            available = [f for f in feat_names if f in X.columns]
            preds     = tree.predict(X[available].reset_index(drop=True))
            sums     += preds.astype(np.float64)
        return sums / len(self.estimators_)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Class probability as vote fraction (classification only).

        For ``model="twoClass"`` this method is not meaningful — use
        ``predict()`` which returns the averaged numeric prediction.

        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
            Columns ordered as self.classes_.

        Raises
        ------
        NotImplementedError
            When ``model="twoClass"``.
        """
        self._check_fitted()
        if self.model == "twoClass":
            raise NotImplementedError(
                "predict_proba() is not available for model='twoClass' "
                "(regression mode). Use predict() to get averaged numeric "
                "predictions."
            )
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
            "n_estimators":     self.n_estimators,
            "model":            self.model,
            "max_features":     self.max_features,
            "feature_sampling": self.feature_sampling,
            "bootstrap":        self.bootstrap,
            "random_state":     self.random_state,
            "verbose":       self.verbose,
            **self.tree_kwargs,
        }

    def set_params(self, **params) -> "PredictabilityForest":
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
    def _compute_oob_score(y: pd.Series, oob_preds: list, model: str = "twoStage") -> float:
        """
        Compute out-of-bag score.

        - Classification (twoStage / twoing): accuracy of majority-vote.
        - Regression (twoClass): R² of averaged leaf-mean predictions.
        """
        if model == "twoClass":
            # Regression: average OOB predictions, then compute R²
            y_true = y.to_numpy(dtype=np.float64)
            y_pred = np.full(len(y), np.nan)
            for i, preds in enumerate(oob_preds):
                if preds:
                    y_pred[i] = np.mean([float(p) for p in preds])
            mask = ~np.isnan(y_pred)
            if mask.sum() == 0:
                return float("nan")
            y_true_m = y_true[mask]
            y_pred_m = y_pred[mask]
            ss_res = ((y_true_m - y_pred_m) ** 2).sum()
            ss_tot = ((y_true_m - y_true_m.mean()) ** 2).sum()
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            # Classification: majority-vote accuracy
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
            f"PredictabilityForest("
            f"n_estimators={self.n_estimators}, "
            f"model='{self.model}', "
            f"max_features={self.max_features!r}, "
            f"feature_sampling='{self.feature_sampling}', "
            f"bootstrap={self.bootstrap}, "
            f"[{status}])"
        )
