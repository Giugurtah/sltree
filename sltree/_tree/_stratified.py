# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/_tree/_stratified.py
===========================
Stratum-level helpers shared by SLBT and SLBTWeighted.

strat_distribution()  — class distribution of Y within each stratum at a node
safe_ratio()          — elementwise ratio with NaN where the denominator is 0
leaf_lcr()            — Leaf Classification Ratio per stratum (Eq. 5)
seen_modalities()     — modalities of the split variable observed at a node
traverse()            — route one observation down a fitted tree
"""

from __future__ import annotations

import numpy as np


def strat_distribution(y, x_s, labels, w=None):
    """
    Relative class frequencies of ``y`` within each stratum.

    Parameters
    ----------
    y      : array-like — target values at the node
    x_s    : array-like — stratum of each observation
    labels : array-like — class labels defining the column order
    w      : array-like, optional — sample weights

    Returns
    -------
    strata : np.ndarray — strata present, in ``np.unique`` order
    D      : np.ndarray, shape (len(strata), len(labels)) — row t is
             p_{j|(g_t)} at the node (rows sum to 1)
    """
    y   = np.asarray(y)
    x_s = np.asarray(x_s)
    labels = np.asarray(labels)
    w = np.ones(len(y), dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)

    strata = np.unique(x_s)
    D = np.zeros((len(strata), len(labels)), dtype=np.float64)
    for t, s in enumerate(strata):
        m = (x_s == s)
        tot = w[m].sum()
        if tot <= 0:
            continue
        for j, lab in enumerate(labels):
            D[t, j] = w[m & (y == lab)].sum() / tot
    return strata, D


def safe_ratio(num, den):
    """num / den elementwise, NaN where den == 0."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=(den > 0))
    return out


def leaf_lcr(node_strata, node_labels, node_D, root_strata, root_labels, root_D):
    """
    Leaf Classification Ratio (Eq. 5): p_{j|(g)}(h) / p_{j|(g)}(1), for every
    stratum g present at the leaf and every class j present at the leaf.

    Returns a list (one array per stratum, aligned with ``node_strata`` and
    ``node_labels``).
    """
    root_s = {str(s): t for t, s in enumerate(root_strata)}
    root_l = {str(l): j for j, l in enumerate(root_labels)}
    out = []
    for t, s in enumerate(node_strata):
        prior = np.array([
            root_D[root_s[str(s)], root_l[str(l)]] for l in node_labels
        ])
        out.append(safe_ratio(node_D[t], prior))
    return out


def seen_modalities(X_col, x_s):
    """
    Modalities of the split variable observed at the node, overall and per
    stratum (as strings).  Used by ``traverse`` to route values the split
    has never seen.
    """
    x  = np.asarray(X_col).astype(str)
    xs = np.asarray(x_s)
    by_stratum = {str(s): set(x[xs == s]) for s in np.unique(xs)}
    return {"all": set(x), "by_stratum": by_stratum}


def _larger_child(node):
    left_n  = getattr(node.left,  "N", 0) or 0
    right_n = getattr(node.right, "N", 0) or 0
    return node.left if left_n >= right_n else node.right


def traverse(node, row, s=None):
    """
    Route one observation to its leaf.

    Parameters
    ----------
    node : Node   — current node (start from the root)
    row  : pd.Series — predictor values of the observation
    s    : stratum of the observation, or None

    Rules
    -----
    * Shared threshold (homogeneity "A" / "AB"): left iff the value is in
      the left set.
    * Stratum-specific thresholds ("none" / "B"): the left set of the
      observation's own stratum is used, so ``s`` is required.
    * A value (or a stratum) not observed at the node when the split was
      estimated carries no information on the side it belongs to; it is
      sent to the larger child, as the rule "go where most training
      observations went".
    """
    while not node._is_leaf_node():
        thr  = node.treshold
        seen = getattr(node, "seen", None)
        v    = str(row[node.feature])

        stratified = not (isinstance(thr, np.ndarray) and thr.ndim == 1)

        if stratified:
            if s is None and len(node.strat_labels) == 1:
                # a single stratum (e.g. a forest fitted without x_s): no
                # ambiguity about which threshold applies
                s = node.strat_labels[0]
            if s is None:
                raise ValueError(
                    "This tree has stratum-specific splits (homogeneity "
                    "'none' or 'B'): pass the stratum of each observation, "
                    "e.g. predict(X, x_s=...)."
                )
            strat_labels = [str(z) for z in node.strat_labels]
            key = str(s)
            if key not in strat_labels:
                node = _larger_child(node)
                continue
            if seen is not None and v not in seen["by_stratum"].get(key, set()):
                node = _larger_child(node)
                continue
            left = {str(z) for z in thr[strat_labels.index(key)]}
        else:
            if seen is not None and v not in seen["all"]:
                node = _larger_child(node)
                continue
            left = {str(z) for z in thr}

        node = node.left if v in left else node.right
    return node
