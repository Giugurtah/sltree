# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/slbt_weighted.py
=======================
SLBTWeighted — SLBT with per-sample weights (for AdaBoost).

Identical in structure to SLBT but every computation that depends on
observation counts is replaced by its weight-sum equivalent:

  - Gini impurity    → weighted Gini
  - GPI ranking      → weighted stratified GPI
  - Contingency mats → weighted stratified contingency
  - Leaf prediction  → argmax of per-class weight sums (= weighted majority)

The C backend functions (slba, gpi from libslbt) are completely agnostic
to weighting: they operate on float matrices and do not know whether
cells contain raw counts or weight sums.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base      import BaseTree, Node
from .reporting import TreeReporter
from ._utils.criteria import (
    _impurity_weighted,
    _distribution_weighted,
)
from ._utils.utils  import _row_conditional
from ._utils.coding import non_constant, NodeCodes, conditional_gini
from ._tree._visual_pruning import visual_pruning_path
from ._tree._coded  import CodedSLBTMixin, resolve_n_jobs
from ._tree.split   import score_slbt, _split, _splitS
from ._tree._homogeneity.base import get_homogeneity_strategy
from ._tree._stratified import (
    strat_distribution, leaf_lcr, seen_modalities, traverse,
)


#: Margin of the FAST early stop (see the FAST parameter).
_FAST_MARGIN = 1e-12


class SLBTWeighted(CodedSLBTMixin, BaseTree):
    """
    SLBT with per-sample weights — weak learner for SLBTAdaBoostForest.

    Parameters are identical to SLBT.  The only difference is that
    ``fit()`` accepts an optional ``sample_weight`` array which is
    propagated through every computation inside the tree.

    Parameters
    ----------
    homogeneity : {"none", "A", "B", "AB"}, default "none"
    min_ppi : float, default 0.001
    min_gpi : float, default 0.001
    min_impurity : float, default 0.001
    min_samples_split : int, default 2
    max_depth : int, default 5
    feats_viewed : int, default 10
    FAST : bool, default True
        Stop the loop over the candidate predictors as soon as the best PI
        found exceeds the GPI of the next predictor.  Since PI <= GPI for
        every split of a predictor (the split coarsens its categories) and
        the predictors are visited by decreasing GPI, no later predictor can
        win: the tree is the same as with FAST=False, only faster.  A margin
        of 1e-12 keeps the guarantee against rounding in the last digit.
    n_init, ls_weights, epsilon, als_max_iter, als_tol, optimizer, tie_break,
    n_jobs, random_state :
        As in ``SLBT``.
    """

    def __init__(
        self,
        homogeneity: str       = "none",
        min_ppi: float         = 0.001,
        min_gpi: float         = 0.001,
        min_impurity: float    = 0.001,
        min_samples_split: int = 2,
        max_depth: int         = 5,
        feats_viewed: int      = 10,
        FAST: bool             = True,
        rng                    = None,
        random_state: int | None = None,
        n_init: int            = 1,
        ls_weights: str        = "budget",
        epsilon: float         = 0.01,
        als_max_iter: int      = 5000,
        als_tol: float         = 1e-12,
        optimizer: str         = "softmax",
        tie_break: str         = "loss",
        n_jobs: int | None     = 1,
    ):
        super().__init__(
            min_ppi=min_ppi, min_gpi=min_gpi,
            min_impurity=min_impurity,
            min_samples_split=min_samples_split,
            max_depth=max_depth, feats_viewed=feats_viewed, FAST=FAST,
        )
        if homogeneity not in ("none", "A", "B", "AB"):
            raise ValueError(
                f"homogeneity='{homogeneity}' not supported. "
                "Choose from: 'none', 'A', 'B', 'AB'."
            )
        if ls_weights not in ("budget", "frequency"):
            raise ValueError(
                f"ls_weights='{ls_weights}' not supported. "
                "Choose from: 'budget', 'frequency'."
            )
        if int(n_init) < 1:
            raise ValueError("n_init must be >= 1.")
        self.homogeneity  = homogeneity
        self._rng         = rng
        self.random_state = random_state
        self.n_init       = int(n_init)
        self.ls_weights   = ls_weights
        self.epsilon      = float(epsilon)
        self.als_max_iter = int(als_max_iter)
        self.als_tol      = float(als_tol)
        if optimizer not in ("softmax", "projected"):
            raise ValueError(
                f"optimizer='{optimizer}' not supported. "
                "Choose from: 'softmax', 'projected'."
            )
        self.optimizer    = optimizer
        if tie_break not in ("loss", "pi"):
            raise ValueError("tie_break must be 'loss' or 'pi'.")
        self.tie_break    = tie_break
        resolve_n_jobs(n_jobs)                 # validates
        self.n_jobs       = n_jobs
        self.strata_      = None


    def get_params(self, deep: bool = True) -> dict:
        """
        Return the estimator's hyperparameters, scikit-learn style.

        Parameters
        ----------
        deep : bool, default True
            Accepted for interface compatibility.  This estimator holds no
            nested estimators, so the flag makes no difference.

        Returns
        -------
        dict
            Parameter names mapped to their current values.
        """
        p = super().get_params(deep)
        p["homogeneity"]  = self.homogeneity
        p["random_state"] = self.random_state
        p["n_init"]       = self.n_init
        p["ls_weights"]   = self.ls_weights
        p["epsilon"]      = self.epsilon
        p["als_max_iter"] = self.als_max_iter
        p["als_tol"]      = self.als_tol
        p["optimizer"]    = self.optimizer
        p["tie_break"]    = self.tie_break
        p["n_jobs"]       = self.n_jobs
        p.pop("model", None)
        return p

    # ================================================================
    #  PUBLIC — fit / predict
    # ================================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        x_s: np.ndarray | None  = None,
        sample_weight: np.ndarray | None = None,
    ) -> "SLBTWeighted":
        """
        Fit a weighted SLBT.

        Parameters
        ----------
        X             : pd.DataFrame — all-categorical predictors.
        y             : pd.Series    — categorical target.
        x_s           : np.ndarray, optional — stratum indicator.
                        If None, all observations share stratum 0 and
                        homogeneity is forced to "AB" (classic LBT).
        sample_weight : np.ndarray, optional — per-sample weights.
                        If None, uniform weights 1/n are used.

        Returns
        -------
        self
        """
        n = len(y)

        if sample_weight is None:
            sample_weight = np.ones(n, dtype=np.float64) / n
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            s = sample_weight.sum()
            if s > 0:
                sample_weight = sample_weight / s   # normalise to sum=1

        if x_s is None:
            x_s              = np.zeros(n, dtype=int)
            self.homogeneity = "AB"
            self.strata_     = None
        else:
            x_s          = np.asarray(x_s)
            self.strata_ = np.unique(x_s)

        strategy = get_homogeneity_strategy(self.homogeneity)

        self.targhet_dist = [
            np.unique(y),
            _distribution_weighted(y, sample_weight),
        ]
        self.classes_ = self.targhet_dist[0]
        self._root_strata, self._root_strat_dist = strat_distribution(
            y, x_s, self.classes_, sample_weight
        )
        self.convergence_ = []
        self.root_N   = n
        self.reporter = TreeReporter(homogeneity=self.homogeneity, decimals=4)

        W_root    = sample_weight.sum()          # 1.0 after normalisation
        _, root_yc = np.unique(np.asarray(y), return_inverse=True)
        _, root_sc = np.unique(np.asarray(x_s), return_inverse=True)
        # impurity of Y within the strata at the root (V_Y, Eq. 6)
        root_imp  = conditional_gini(root_yc.ravel(), int(root_yc.max()) + 1,
                                     root_sc.ravel(), int(root_sc.max()) + 1,
                                     sample_weight)

        self._begin_fit(X, y, x_s, sample_weight)
        try:
            self.root = self._grow_tree(
                strategy, np.arange(n), np.arange(X.shape[1]),
                root_impurity=root_imp, W_root=W_root,
            )
        finally:
            self._end_fit()
        self._calculate_tree_partial_impurity_reduction()
        return self

    def predict(self, X: pd.DataFrame, x_s=None) -> np.ndarray:
        """
        Predict class labels for X.  ``x_s`` (stratum of each row) is
        required when the tree has stratum-specific splits (homogeneity
        "none" or "B" fitted with strata).
        """
        if self.root is None:
            raise RuntimeError("Tree not fitted. Call fit() first.")
        if x_s is None or self.strata_ is None:
            strata = [None] * len(X)
        else:
            strata = list(np.asarray(x_s))
        return np.array([
            traverse(self.root, X.iloc[i], strata[i]).value
            for i in range(len(X))
        ])

    # ================================================================
    #  PRIVATE — tree growth
    # ================================================================

    def _grow_tree(
        self,
        strategy,
        rows: np.ndarray,
        cols,
        root_impurity: float  = 1.0,
        W_root: float         = 1.0,
        depth: int            = 0,
        pos: int              = 1,
    ) -> Node:
        """
        Grow the subtree of the node holding the observations ``rows``
        (positions in the data given to ``fit``), with candidate predictors
        ``cols`` (column indices).
        """
        y   = self._y.iloc[rows]
        x_s = self._xs[rows]
        w   = self._w[rows]

        nonconst = non_constant(self._C[np.ix_(rows, cols)]) if len(cols) else np.zeros(0, bool)
        cols = [c for c, keep in zip(cols, nonconst) if keep]

        n_samples    = len(rows)
        n_feats      = len(cols)
        n_labels     = len(np.unique(y))
        impurity     = _impurity_weighted(y, w)
        distribution = _distribution_weighted(y, w)

        W_node = w.sum()
        # V_Y(t), Eq. 6, with the weighted impurity of Y within the strata
        impurity_z = conditional_gini(self._yc[rows], self._J, self._sc[rows], self._G, w)
        impurity_decrease = (
            (root_impurity - impurity_z) / root_impurity if root_impurity > 0 else 0.0
        )

        node_strata, node_strat_dist = strat_distribution(y, x_s, np.unique(y), w)

        # --- pre-split stopping criteria ---
        leaf = self._check_criteria_before(
            y, w, pos, impurity, distribution, depth,
            n_labels, n_samples, n_feats,
            impurity_decrease, node_strata, node_strat_dist,
        )
        if leaf is not None:
            leaf.mass = W_node / W_root
            return leaf

        # --- weighted stratified GPI ranking (one table per predictor) ---
        nc = NodeCodes(self._yc[rows], self._J, self._sc[rows], self._G, w)
        gpi_vals, gpi_order, tables = self._node_tables(rows, cols, nc)

        # --- best split search ---
        best_feature, best_treshold, best_pi, best_gpi, best_alpha, best_beta, best_diag = (
            self._find_best_predictor(tables, gpi_order, gpi_vals, pos)
        )

        # --- post-split stopping criteria ---
        leaf = self._check_criteria_after(
            y, w, pos, impurity, distribution, depth,
            best_gpi, best_pi,
            impurity_decrease, node_strata, node_strat_dist,
        )
        if leaf is not None:
            leaf.mass = W_node / W_root
            return leaf

        c_best    = {str(k): v[0] for k, v in tables.items()}[best_feature]
        x_best    = self._X_cols[c_best].iloc[rows]
        x_vals    = np.unique(x_best)
        thresholds = strategy.get_treshold_values(best_treshold, x_vals, x_s)
        indexL, indexR = strategy.split(x_best, x_s, thresholds)
        lift1, lift2   = strategy.compute_lift(best_beta, node_strat_dist)
        if best_diag is not None:
            self.convergence_.append({"node": pos, "feature": best_feature, **best_diag})

        left  = self._grow_tree(
            strategy, rows[indexL], cols,
            root_impurity, W_root, depth + 1, 2 * pos,
        )
        right = self._grow_tree(
            strategy, rows[indexR], cols,
            root_impurity, W_root, depth + 1, 2 * pos + 1,
        )

        node = Node(
            gpi=best_gpi, pi=best_pi, position=pos,
            feature=best_feature, treshold=thresholds,
            left=left, right=right,
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=0.0,
            distribution=distribution, N=n_samples, labels=np.unique(y),
            LIFT_1=lift1, LIFT_2=lift2,
            GCR=None,
            strat_labels=node_strata,
            strat_distribution=node_strat_dist,
            seen=seen_modalities(x_best, x_s),
            als_diag=best_diag,
        )
        node.mass = W_node / W_root
        if self.reporter is not None:
            self.reporter.add_node(node, is_leaf=False)
        return node

    # ================================================================
    #  PRIVATE — best predictor search
    # ================================================================

    def _find_best_predictor(self, tables, gpi_order, gpi_vals, pos):
        best = {
            "feature":   None,
            "threshold": None,
            "pi":        -np.inf,
            "gpi":       -np.inf,
            "alpha":     None,
            "beta":      None,
            "diag":      None,
        }

        for rank, col in enumerate(gpi_order[: self.feats_viewed]):
            c, Fs_noN = tables[col]
            Fs        = _row_conditional(Fs_noN)

            pi, S, alpha, beta, diag = score_slbt(
                Fs_noN, Fs, self.homogeneity, self._als_seed(pos, c),
                n_init=self.n_init, ls_weights=self.ls_weights,
                max_iter=self.als_max_iter, tol=self.als_tol,
                optimizer=self.optimizer, tie_break=self.tie_break,
                executor=self._executor,
            )

            if pi > best["pi"]:
                best["pi"]        = pi
                best["feature"]   = str(col)
                best["threshold"] = S
                best["alpha"]     = alpha
                best["beta"]      = beta
                best["gpi"]       = gpi_vals[rank]
                best["diag"]      = diag

            if (self.FAST
                    and rank < len(gpi_order) - 1
                    and best["pi"] > gpi_vals[rank + 1] + _FAST_MARGIN):
                break

        return (
            best["feature"],
            best["threshold"],
            best["pi"],
            best["gpi"],
            best["alpha"],
            best["beta"],
            best["diag"],
        )

    # ================================================================
    #  PRIVATE — stopping criteria
    # ================================================================

    def _check_criteria_before(
        self, y, w, pos, impurity, distribution, depth,
        n_labels, n_samples, n_feats,
        impurity_decrease, node_strata=None, node_strat_dist=None,
    ):
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split
                or n_feats  == 0
                or impurity  < self.min_impurity):
            return self._make_leaf(
                y, w, pos, impurity, distribution,
                impurity_decrease, node_strata, node_strat_dist,
            )
        return None

    def _check_criteria_after(
        self, y, w, pos, impurity, distribution, depth,
        best_gpi, best_pi,
        impurity_decrease, node_strata=None, node_strat_dist=None,
    ):
        if best_gpi < self.min_gpi or best_pi < self.min_ppi or best_pi < 1e-8:
            return self._make_leaf(
                y, w, pos, impurity, distribution,
                impurity_decrease, node_strata, node_strat_dist,
            )
        return None

    def _make_leaf(
        self, y, w, pos, impurity, distribution,
        impurity_decrease, node_strata=None, node_strat_dist=None,
    ) -> Node:
        # Weighted majority: class with highest total weight
        y_arr   = np.asarray(y)
        w_arr   = np.asarray(w, dtype=np.float64)
        classes = np.unique(y_arr)
        class_weights = {c: w_arr[y_arr == c].sum() for c in classes}
        value   = max(class_weights, key=class_weights.get)

        leaf = Node(
            position=pos,
            value=value,
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=0.0,
            distribution=distribution,
            N=len(y),
            labels=classes,
            GCR=self._get_gcr(distribution, classes),
            strat_labels=node_strata,
            strat_distribution=node_strat_dist,
        )
        if node_strat_dist is not None:
            leaf.LCR = leaf_lcr(
                node_strata, classes, node_strat_dist,
                self._root_strata, self.classes_, self._root_strat_dist,
            )
        if self.reporter is not None:
            self.reporter.add_node(leaf, is_leaf=True)
        return leaf

    # ================================================================
    #  PRIVATE — utilities (shared with SLBT)
    # ================================================================

    def _traverse_tree(self, x: pd.Series, node: Node, s=None):
        return traverse(node, x, s).value

    def _drop_constant_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=X.columns[X.nunique(dropna=False) <= 1])

    def _get_gcr(self, distribution, labels):
        gcr = [0.0] * len(labels)
        for i, lbl in enumerate(labels):
            for j, g_lbl in enumerate(self.targhet_dist[0]):
                if g_lbl == lbl:
                    ref = self.targhet_dist[1][j]
                    gcr[i] = distribution[i] / ref if ref > 0 else 0.0
        return gcr

    def _collect_nodes(self, node: Node, nodes_list: list):
        if node is None:
            return
        nodes_list.append(node)
        if not node._is_leaf_node():
            self._collect_nodes(node.left,  nodes_list)
            self._collect_nodes(node.right, nodes_list)

    def _calculate_tree_partial_impurity_reduction(self):
        """V_Y(T) along the visual-pruning sequence and the suggested cut
        (Eqs. 6-8): see ``_tree._visual_pruning``; p(h) is the weight share."""
        visual_pruning_path(self.root, self.epsilon)

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        status = "fitted" if self.root is not None else "not fitted"
        return (
            f"SLBTWeighted(homogeneity='{self.homogeneity}', "
            f"max_depth={self.max_depth}, "
            f"feats_viewed={self.feats_viewed}, "
            f"[{status}])"
        )


# ================================================================
#  Module-level sort helpers (mirrors SLBT private statics)
# ================================================================

def _bubble_sort_nodes(nodes):
    changed, max_iter, iteration = True, len(nodes) * 2, 0
    while changed and iteration < max_iter:
        changed   = False
        iteration += 1
        for i in range(len(nodes) - 1):
            if nodes[i].impurity_decrease > nodes[i + 1].impurity_decrease:
                nodes[i], nodes[i + 1] = nodes[i + 1], nodes[i]
                changed = True

