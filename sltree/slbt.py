# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/slbt.py
==============
SLBT — Simultaneous Latent Budget Tree for stratified categorical predictors.

Homogeneity modes
-----------------
"none" : per-stratum A and B (fully unconstrained)
"A"    : mixing coefficients A shared across strata
"B"    : latent budget profiles B shared across strata
"AB"   : both A and B shared (equivalent to non-stratified LBA)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base       import BaseTree, Node
from .reporting  import TreeReporter
from ._utils.utils    import _row_conditional
from ._utils.coding   import non_constant, NodeCodes, conditional_gini
from ._tree._visual_pruning import visual_pruning_path, cut_at
from ._tree._coded    import CodedSLBTMixin, resolve_n_jobs
from ._tree.split     import score_slbt, _split, _splitS
from ._tree._homogeneity.base import get_homogeneity_strategy
from ._tree._budgets import label_budgets
from ._tree._stratified import (
    strat_distribution, leaf_lcr, seen_modalities, traverse,
)


#: Margin of the FAST early stop (see the FAST parameter).
_FAST_MARGIN = 1e-12


class SLBT(CodedSLBTMixin, BaseTree):
    """
    Simultaneous Latent Budget Tree for stratified categorical data.

    Parameters
    ----------
    homogeneity : {"none", "A", "B", "AB"}, default "none"
        Homogeneity constraint on the LBA parameters across strata.
    min_ppi : float, default 0.0
        Minimum PPI to accept a split.
    min_gpi : float, default 0.0
        Minimum GPI for a feature to be a candidate.
    min_impurity : float, default 0.0
        Nodes below this impurity become leaves.
    min_samples_split : int, default 1
        Minimum samples to attempt a split.
    max_depth : int, default 100
        Maximum tree depth.
    feats_viewed : int, default 10
        Top-GPI features evaluated per node.
    FAST : bool, default True
        Stop the loop over the candidate predictors as soon as the best PI
        found exceeds the GPI of the next predictor.  Since PI <= GPI for
        every split of a predictor (the split coarsens its categories) and
        the predictors are visited by decreasing GPI, no later predictor can
        win: the tree is the same as with FAST=False, only faster.  A margin
        of 1e-12 keeps the guarantee against rounding in the last digit.
    n_init : int, default 1
        Random starts of the least-squares SLBM fit at every candidate
        split; the start with the lowest loss is kept.  Values of 5-10 make
        the split far less dependent on the random start.
    ls_weights : {"budget", "frequency"}, default "budget"
        Row weighting of the least-squares fit.  "budget": every observed
        (stratum, category) budget weighs the same (behaviour of earlier
        versions).  "frequency": rows weighted by their frequency, i.e.
        Eq. 13 of the SLBT paper.
    epsilon : float, default 0.01
        Threshold of Eq. 8 for the suggested tree size in visual pruning
        (``prune_suggested``): the suggested tree stops before the first
        split, in the visual-pruning sequence, that raises V_Y(T) by at most
        epsilon; the split of the root included.  V_Y uses the impurity of Y
        within the strata.  See ``_tree._visual_pruning``.
    als_max_iter : int, default 5000
        Iteration cap per start of the least-squares fit.
    als_tol : float, default 1e-12
        Relative-decrease tolerance of the least-squares fit.
    optimizer : {"softmax", "projected"}, default "softmax"
        "softmax": gradient descent on the softmax parameters, the neural
        network formulation of the paper (Eq. 16-19).  "projected":
        projected gradient on the simplex, same objective and gradients
        (Eq. 14-15) without the reparameterisation; it reaches optima with
        some alpha or beta exactly 0 or 1, which the softmax only
        approaches.
    tie_break : {"loss", "pi"}, default "loss"
        Which start is kept when ``n_init`` > 1: the lowest loss ("loss"),
        or, among the starts with the same lowest loss, the split with the
        largest PI ("pi").  See ``_tree.split.score_slbt``.
    n_jobs : int or None, default 1
        Threads running the ``n_init`` starts of each least-squares fit in
        parallel (the C fit releases the GIL); -1 uses every core.  Useful
        with ``n_init`` > 1 only.  The starts are the same as with one
        thread, so the tree is identical for every value of ``n_jobs``.
        Keep 1 when several fits already run in parallel processes.
    random_state : int, Generator or None, default None
        Seed of the fit.  The starting matrices of the least-squares fit of
        predictor c at node t are drawn from a seed fixed by
        (random_state, t, c), so the tree does not depend on ``FAST``,
        ``n_jobs`` or the order in which candidates are evaluated, and a
        second call of ``fit`` on the same data grows the same tree.

    Attributes (set after fit)
    --------------------------
    root : Node
    reporter : TreeReporter
    targhet_dist : list
    root_N : int
    classes_ : np.ndarray
    strata_ : np.ndarray or None
        Strata seen in fit (None when fit without ``x_s``).
    """

    def __init__(
        self,
        homogeneity: str       = "none",
        min_ppi: float         = 0.0,
        min_gpi: float         = 0.0,
        min_impurity: float    = 0.0,
        min_samples_split: int = 1,
        max_depth: int         = 100,
        feats_viewed: int      = 10,
        FAST: bool             = True,
        node_max_features: int | None = None,
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
        self.homogeneity       = homogeneity
        self.node_max_features = node_max_features
        self._rng              = rng
        self.random_state      = random_state
        self.n_init            = int(n_init)
        self.ls_weights        = ls_weights
        self.epsilon           = float(epsilon)
        self.als_max_iter      = int(als_max_iter)
        self.als_tol           = float(als_tol)
        if optimizer not in ("softmax", "projected"):
            raise ValueError(
                f"optimizer='{optimizer}' not supported. "
                "Choose from: 'softmax', 'projected'."
            )
        self.optimizer         = optimizer
        if tie_break not in ("loss", "pi"):
            raise ValueError("tie_break must be 'loss' or 'pi'.")
        self.tie_break         = tie_break
        resolve_n_jobs(n_jobs)                 # validates
        self.n_jobs            = n_jobs
        self.strata_           = None


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
    #  PUBLIC — fit / predict / prune
    # ================================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        x_s: np.ndarray | None = None,
    ) -> "SLBT":
        """
        Fit the SLBT on stratified categorical data.

        Parameters
        ----------
        X   : pd.DataFrame — all-categorical predictor matrix
        y   : pd.Series    — categorical target
        x_s : np.ndarray, shape (n_samples,), optional
            Stratum indicator (integer per observation).
            If None, all observations belong to stratum 0 and the
            homogeneity is set to "AB" (equivalent to non-stratified LBA).

        Returns
        -------
        self
        """
        if x_s is None:
            x_s              = np.zeros(len(y), dtype=int)
            self.homogeneity = "AB"
            self.strata_     = None
        else:
            x_s          = np.asarray(x_s)
            self.strata_ = np.unique(x_s)

        strategy = get_homogeneity_strategy(self.homogeneity)

        self.targhet_dist = [
            np.unique(y),
            np.unique(y, return_counts=True)[1] / len(y),
        ]
        self.classes_ = self.targhet_dist[0]
        self._root_strata, self._root_strat_dist = strat_distribution(
            y, x_s, self.classes_
        )
        self.root_N  = len(y)
        self.reporter = TreeReporter(homogeneity=self.homogeneity, decimals=4)
        self.convergence_ = []

        self._begin_fit(X, y, x_s)
        try:
            self.root = self._grow_tree(
                strategy, np.arange(len(y)), np.arange(X.shape[1]),
            )
        finally:
            self._end_fit()
        self._calculate_tree_partial_impurity_reduction()
        return self

    def predict(self, X: pd.DataFrame, x_s=None, by_stratum: bool = False) -> np.ndarray:
        """
        Predict class labels for X.

        Parameters
        ----------
        X   : pd.DataFrame
        x_s : array-like, optional
            Stratum of each row.  Required when the tree has
            stratum-specific splits (homogeneity "none" or "B" fitted with
            strata); ignored for a tree fitted without ``x_s``.
        by_stratum : bool, default False
            False: the leaf predicts its pooled majority class.
            True:  the leaf predicts the majority class of the observation's
            own stratum within the leaf (the stratum-specific response
            profile); falls back to the pooled class when that stratum is
            absent from the leaf.
        """
        proba = None
        if by_stratum:
            proba = self.predict_proba(X, x_s=x_s, by_stratum=True)
            return self.classes_[np.argmax(proba, axis=1)]
        leaves = self._leaves_for(X, x_s)
        return np.array([leaf.value for leaf in leaves])

    def predict_proba(self, X: pd.DataFrame, x_s=None, by_stratum: bool = False) -> np.ndarray:
        """
        Class probabilities: the class distribution of the leaf reached by
        each row (pooled, or of the row's own stratum if ``by_stratum``),
        with columns ordered as ``classes_``.
        """
        leaves = self._leaves_for(X, x_s)
        strata = self._strata_for(X, x_s)
        col = {str(c): k for k, c in enumerate(self.classes_)}
        P = np.zeros((len(leaves), len(self.classes_)))
        for r, leaf in enumerate(leaves):
            dist = leaf.distribution
            if by_stratum and strata[r] is not None and leaf.strat_distribution is not None:
                labels_s = [str(z) for z in leaf.strat_labels]
                if str(strata[r]) in labels_s:
                    dist = leaf.strat_distribution[labels_s.index(str(strata[r]))]
            for lab, p in zip(leaf.labels, dist):
                P[r, col[str(lab)]] = p
        return P

    def apply(self, X: pd.DataFrame, x_s=None) -> np.ndarray:
        """Position (node id) of the leaf reached by each row."""
        return np.array([leaf.position for leaf in self._leaves_for(X, x_s)])

    def _strata_for(self, X, x_s):
        if x_s is None or self.strata_ is None:
            return [None] * len(X)
        x_s = np.asarray(x_s)
        if len(x_s) != len(X):
            raise ValueError("x_s must have one entry per row of X.")
        return list(x_s)

    def _leaves_for(self, X, x_s):
        if self.root is None:
            raise RuntimeError("Tree not fitted. Call fit() first.")
        strata = self._strata_for(X, x_s)
        return [traverse(self.root, X.iloc[i], strata[i]) for i in range(len(X))]

    def prune_suggested(self, epsilon: float | None = None) -> int | None:
        """
        Prune at the size suggested by Eq. 8.

        Parameters
        ----------
        epsilon : float, optional
            Threshold of Eq. 8; ``self.epsilon`` when omitted.

        Returns the id of the node at which the tree was cut (1 when only the
        root is left), or None if every split gains more than epsilon
        (nothing pruned).
        """
        if self.root is None:
            raise RuntimeError("Tree not fitted. Call fit() first.")
        if epsilon is not None:
            self.epsilon = float(epsilon)
        self._calculate_tree_partial_impurity_reduction()
        nodes = []
        self._collect_nodes(self.root, nodes)
        flagged = [n for n in nodes if n.suggested_pruning]
        if not flagged:
            return None
        node_id = flagged[0].position
        self.prune_after_vp(node_id)
        return node_id

    def prune_after_vp(self, nodeID: int) -> None:
        """
        Prune the tree at the node identified by *nodeID*.

        The internal nodes are expanded in the order of the visual-pruning
        sequence (increasing canvas height, see ``_tree._visual_pruning``)
        until *nodeID* is reached; every internal node still unexpanded at
        that point, *nodeID* included, becomes a leaf.  This is the
        horizontal cut of the visual-pruning canvas.  With the node flagged
        by Eq. 8 it gives the suggested tree; with the root it leaves the
        root alone.
        """
        if self.root is None:
            return

        virtual_leaves = cut_at(self.root, nodeID)

        for node in virtual_leaves:
            if not node._is_leaf_node():
                node.feature   = None
                node.treshold  = None
                node.left      = None
                node.right     = None
                node.gpi       = None
                node.pi        = None
                node.LIFT_1    = None
                node.LIFT_2    = None
                node.als_diag  = None
                node.alpha     = None
                node.beta_left = node.beta_right = None
                node.seen      = None
                best_idx       = int(np.argmax(node.distribution))
                node.value     = node.labels[best_idx]
                node.GCR       = self._get_gcr(node.distribution, node.labels)
                node.LCR       = self._get_lcr(node)

        self._calculate_tree_partial_impurity_reduction()
        self._rebuild_report()

    # ================================================================
    #  PRIVATE — tree growth
    # ================================================================

    def _grow_tree(
        self,
        strategy,
        rows: np.ndarray,
        cols,
        root_impurity: float  = 1.0,
        root_N: int           = 1,
        depth: int            = 0,
        pos: int              = 1,
    ) -> Node:
        """
        Grow the subtree of the node holding the observations ``rows``
        (positions in the data given to ``fit``), with candidate predictors
        ``cols`` (column indices, in the order the node sees them).
        """
        y   = self._y.iloc[rows]
        x_s = self._xs[rows]
        yc  = self._yc[rows]

        # constant predictors are dropped (as X.drop did, keeping the order)
        nonconst = non_constant(self._C[np.ix_(rows, cols)]) if len(cols) else np.zeros(0, bool)
        cols = [c for c, keep in zip(cols, nonconst) if keep]

        counts = np.bincount(yc, minlength=self._J)
        counts = counts[counts > 0]
        n_samples    = len(rows)
        n_feats      = len(cols)
        n_labels     = len(counts)
        distribution = counts / n_samples
        impurity     = float(1.0 - np.sum(distribution ** 2))

        # V_Y(t), Eq. 6, with the impurity of Y within the strata
        impurity_z = conditional_gini(yc, self._J, self._sc[rows], self._G)
        if root_N == 1:
            root_N        = len(y)
            root_impurity = impurity_z
            impurity_decrease = 0.0
        else:
            impurity_decrease = (
                (root_impurity - impurity_z) / root_impurity if root_impurity > 0 else 0.0
            )
        tree_partial_impurity_reduction = 0.0

        node_strata, node_strat_dist = strat_distribution(y, x_s, np.unique(y))

        # --- pre-split stopping criteria ---
        leaf = self._check_criteria_before(
            y, pos, impurity, distribution, depth,
            n_labels, n_samples, n_feats,
            impurity_decrease, tree_partial_impurity_reduction,
            node_strata, node_strat_dist,
        )
        if leaf is not None:
            return leaf

        # --- per-node random feature subsampling (true RF mode) ---
        # As before, the children only see the predictors sampled here.
        if self.node_max_features is not None and self._sub_rng is not None:
            n = min(self.node_max_features, len(cols))
            by_name = {self._feat_names[c]: c for c in cols}
            sampled = self._sub_rng.choice([self._feat_names[c] for c in cols],
                                       size=n, replace=False)
            cols = [by_name[name] for name in sampled]

        # --- stratified GPI ranking (one table per predictor) ---
        nc = NodeCodes(yc, self._J, self._sc[rows], self._G)
        gpi_vals, gpi_order, tables = self._node_tables(rows, cols, nc)

        # --- best split search ---
        best_feature, best_treshold, best_pi, best_gpi, best_alpha, best_beta, best_diag = (
            self._find_best_predictor(tables, gpi_order, gpi_vals, pos)
        )

        # --- post-split stopping criteria ---
        leaf = self._check_criteria_after(
            y, pos, impurity, distribution, depth,
            best_gpi, best_pi,
            impurity_decrease, tree_partial_impurity_reduction,
            node_strata, node_strat_dist,
        )
        if leaf is not None:
            return leaf

        # --- threshold values and split ---
        c_best    = {str(k): v[0] for k, v in tables.items()}[best_feature]
        x_best    = self._X_cols[c_best].iloc[rows]
        x_vals    = np.unique(x_best)
        thresholds = strategy.get_treshold_values(best_treshold, x_vals, x_s)

        indexL, indexR = strategy.split(x_best, x_s, thresholds)

        # --- LIFT values (Eq. 4, prior = class distribution of each stratum) ---
        lift1, lift2 = strategy.compute_lift(best_beta, node_strat_dist)
        # --- mixing parameters and latent budgets of the split, labelled ---
        F_best = {str(k): v[1] for k, v in tables.items()}[best_feature]
        alpha_l, beta_l, beta_r = label_budgets(
            best_alpha, best_beta, F_best, x_vals, node_strata, self.homogeneity,
        )
        if best_diag is not None:
            self.convergence_.append({"node": pos, "feature": best_feature, **best_diag})

        left  = self._grow_tree(
            strategy, rows[indexL], cols,
            root_impurity, root_N, depth + 1, 2 * pos,
        )
        right = self._grow_tree(
            strategy, rows[indexR], cols,
            root_impurity, root_N, depth + 1, 2 * pos + 1,
        )

        node = Node(
            gpi=best_gpi, pi=best_pi, position=pos,
            feature=best_feature, treshold=thresholds,
            left=left, right=right,
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=tree_partial_impurity_reduction,
            distribution=distribution, N=len(y), labels=np.unique(y),
            LIFT_1=lift1, LIFT_2=lift2,
            GCR=None,
            strat_labels=node_strata,
            strat_distribution=node_strat_dist,
            seen=seen_modalities(x_best, x_s),
            als_diag=best_diag,
            alpha=alpha_l, beta_left=beta_l, beta_right=beta_r,
        )
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
        self, y, pos, impurity, distribution, depth,
        n_labels, n_samples, n_feats,
        impurity_decrease, tree_partial_impurity_reduction,
        node_strata=None, node_strat_dist=None,
    ):
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split
                or n_feats  == 0
                or impurity  < self.min_impurity):
            return self._make_leaf(
                y, pos, impurity, distribution,
                impurity_decrease, tree_partial_impurity_reduction,
                node_strata, node_strat_dist,
            )
        return None

    def _check_criteria_after(
        self, y, pos, impurity, distribution, depth,
        best_gpi, best_pi,
        impurity_decrease, tree_partial_impurity_reduction,
        node_strata=None, node_strat_dist=None,
    ):
        if best_gpi < self.min_gpi or best_pi < self.min_ppi or best_pi < 1e-8:
            return self._make_leaf(
                y, pos, impurity, distribution,
                impurity_decrease, tree_partial_impurity_reduction,
                node_strata, node_strat_dist,
            )
        return None

    def _make_leaf(
        self, y, pos, impurity, distribution,
        impurity_decrease, tree_partial_impurity_reduction,
        node_strata=None, node_strat_dist=None,
    ) -> Node:
        leaf = Node(
            position=pos,
            value=y.mode()[0],
            impurity=impurity,
            impurity_decrease=impurity_decrease,
            tree_partial_impurity_reduction=tree_partial_impurity_reduction,
            distribution=distribution,
            N=len(y),
            labels=np.unique(y),
            GCR=self._get_gcr(distribution, np.unique(y)),
            strat_labels=node_strata,
            strat_distribution=node_strat_dist,
        )
        leaf.LCR = self._get_lcr(leaf)
        if self.reporter is not None:
            self.reporter.add_node(leaf, is_leaf=True)
        return leaf

    # ================================================================
    #  PRIVATE — utilities
    # ================================================================

    def _traverse_tree(self, x: pd.Series, node: Node, s=None):
        return traverse(node, x, s).value

    def _drop_constant_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=X.columns[X.nunique(dropna=False) <= 1])

    def _get_lcr(self, node):
        """Leaf Classification Ratio per stratum (Eq. 5)."""
        if node.strat_distribution is None or node.strat_labels is None:
            return None
        return leaf_lcr(
            node.strat_labels, node.labels, node.strat_distribution,
            self._root_strata, self.classes_, self._root_strat_dist,
        )

    def _get_gcr(self, distribution, labels):
        gcr = [0.0] * len(labels)
        for i, lbl in enumerate(labels):
            for j, g_lbl in enumerate(self.targhet_dist[0]):
                if g_lbl == lbl:
                    gcr[i] = distribution[i] / self.targhet_dist[1][j]
        return gcr

    # ================================================================
    #  PRIVATE — pruning & impurity reduction
    # ================================================================

    def _prune_tree(self, node: Node):
        if node is None:
            return None, False
        if node._is_leaf_node():
            return node, False
        left,  cl = self._prune_tree(node.left)
        right, cr = self._prune_tree(node.right)
        node.left, node.right = left, right
        changed = False
        if (node.left  is not None and node.left._is_leaf_node()
                and node.right is not None and node.right._is_leaf_node()
                and node.left.value == node.right.value):
            node.value    = node.left.value
            node.gpi      = node.pi = node.feature = node.treshold = None
            node.LIFT_1   = None
            node.LIFT_2   = None
            node.alpha    = node.beta_left = node.beta_right = None
            node.GCR      = self._get_gcr(node.distribution, node.labels)
            node.LCR      = self._get_lcr(node)
            node.left     = node.right = None
            changed       = True
        return node, cl or cr or changed

    def _calculate_tree_partial_impurity_reduction(self):
        """V_Y(T) along the visual-pruning sequence and the suggested cut
        (Eqs. 6-8): see ``_tree._visual_pruning``."""
        visual_pruning_path(self.root, self.epsilon)

    @staticmethod
    def _bubble_sort_nodes(nodes):
        changed, max_iter, iteration = True, len(nodes) * 2, 0
        while changed and iteration < max_iter:
            changed   = False
            iteration += 1
            for i in range(len(nodes) - 1):
                if nodes[i].impurity_decrease > nodes[i + 1].impurity_decrease:
                    nodes[i], nodes[i + 1] = nodes[i + 1], nodes[i]
                    changed = True

    def _collect_nodes(self, node: Node, nodes_list: list):
        if node is None:
            return
        nodes_list.append(node)
        if not node._is_leaf_node():
            self._collect_nodes(node.left,  nodes_list)
            self._collect_nodes(node.right, nodes_list)

    def _rebuild_report(self):
        if self.reporter is None:
            return
        new_reporter = TreeReporter(
            homogeneity=self.homogeneity,
            decimals=getattr(self.reporter, "decimals", 4),
        )

        def _traverse(n: Node):
            if n is None:
                return
            new_reporter.add_node(n, is_leaf=n._is_leaf_node())
            if not n._is_leaf_node():
                _traverse(n.left)
                _traverse(n.right)

        _traverse(self.root)
        self.reporter = new_reporter

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        status = "fitted" if self.root is not None else "not fitted"
        return (
            f"SLBT(homogeneity='{self.homogeneity}', "
            f"max_depth={self.max_depth}, "
            f"feats_viewed={self.feats_viewed}, "
            f"[{status}])"
        )
