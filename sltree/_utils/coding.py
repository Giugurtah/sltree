# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/_utils/coding.py
=======================
Integer coding of the data, done once per fit, and the contingency tables
built from the codes at every node.

Every modality of a predictor, every class of the target and every stratum
is replaced at the start of ``fit`` by its rank in ``np.unique`` order (the
sorted order).  At a node, the levels present are a subset of the levels of
the whole sample and keep their relative order, so a table built from the
codes and restricted to the levels present at the node has the same rows and
columns, in the same order, as the table built from the original values with
``np.unique`` at that node.  The cell counts are the same integers and the
weighted cells are accumulated by the same ``np.bincount`` in the same order,
so the tables are identical bit for bit.  What changes is the cost: no
sorting of the values at every node and for every predictor, and a single
table per (node, predictor) instead of three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def encode(values) -> tuple[np.ndarray, int]:
    """Codes 0..L-1 of ``values`` in ``np.unique`` order, and L."""
    levels, codes = np.unique(np.asarray(values), return_inverse=True)
    return codes.ravel().astype(np.intp), len(levels)


def encode_frame(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Code every column of ``X``.

    Returns
    -------
    codes : np.ndarray, shape (n, p), dtype intp — column c holds the codes
            of ``X.iloc[:, c]``
    sizes : np.ndarray, shape (p,) — number of levels of each column
    """
    n, p = X.shape
    codes = np.empty((n, p), dtype=np.intp)
    sizes = np.empty(p, dtype=np.intp)
    for c in range(p):
        codes[:, c], sizes[c] = encode(X.iloc[:, c])
    return codes, sizes


def present(codes: np.ndarray, size: int) -> np.ndarray:
    """Boolean mask of the levels that occur in ``codes``."""
    return np.bincount(codes, minlength=size) > 0


def non_constant(codes: np.ndarray) -> np.ndarray:
    """Columns of a code matrix with more than one distinct value."""
    if codes.shape[0] == 0:
        return np.zeros(codes.shape[1], dtype=bool)
    return codes.min(axis=0) != codes.max(axis=0)


class NodeCodes:
    """
    Codes of the target and of the strata at one node, with the levels
    present there.  Built once per node and shared by all its predictors.
    """

    __slots__ = ("yc", "J", "y_pres", "sc", "G", "s_pres", "w")

    def __init__(self, yc, J, sc=None, G=1, w=None):
        self.yc = yc
        self.J = int(J)
        self.y_pres = present(yc, self.J)
        if sc is None:
            sc = np.zeros(len(yc), dtype=np.intp)
            G = 1
        self.sc = sc
        self.G = int(G)
        self.s_pres = present(sc, self.G)
        self.w = None if w is None else np.asarray(w, dtype=np.float64)

    def joint(self, xc: np.ndarray, Ix: int) -> np.ndarray:
        """
        (strata x modalities x classes) table of the predictor with codes
        ``xc`` (``Ix`` levels in the whole sample), restricted to the levels
        present at the node and normalised to sum to 1.  Identical to
        ``_utils.utils._joint_table`` on the original values.
        """
        Ix = int(Ix)
        G, J = self.G, self.J
        cell = (self.sc * Ix + xc) * J + self.yc
        F = np.bincount(cell, weights=self.w, minlength=G * Ix * J)
        F = F.astype(np.float64).reshape(G, Ix, J)
        x_pres = present(xc, Ix)
        F = F[np.ix_(self.s_pres, x_pres, self.y_pres)]
        total = F.sum()
        if total > 0:
            F /= total
        return F

    def counts(self, xc: np.ndarray, Ix: int) -> np.ndarray:
        """
        (modalities x classes) table of counts, levels present at the node
        only; strata ignored.  Same rows and columns as ``pd.crosstab``.
        """
        Ix = int(Ix)
        J = self.J
        F = np.bincount(xc * J + self.yc, minlength=Ix * J).reshape(Ix, J)
        return F[np.ix_(present(xc, Ix), self.y_pres)]


def row_conditional_counts(C: np.ndarray) -> np.ndarray:
    """
    Rows of a count table divided by their totals; empty rows give 0.
    Same values as ``pd.crosstab(x, y).div(row sums).fillna(0)``.
    """
    row = C.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = C / row
    F[np.broadcast_to(row == 0, F.shape)] = 0.0
    return F.astype(np.float64)


def conditional_gini(yc: np.ndarray, J: int, sc: np.ndarray | None = None,
                     G: int = 1, w: np.ndarray | None = None) -> float:
    """
    Gini impurity of the target within the strata,

        i_{Y|Z} = sum_g p(g) (1 - sum_j p(j | g)^2),

    with p from counts, or from the weights ``w`` when given.  Without strata
    (``sc`` None) it is the ordinary Gini impurity.
    """
    if len(yc) == 0:
        return 0.0
    if sc is None:
        sc = np.zeros(len(yc), dtype=np.intp)
        G = 1
    T = np.bincount(sc * J + yc, weights=w, minlength=G * J).astype(np.float64).reshape(G, J)
    tot_g = T.sum(axis=1)
    total = tot_g.sum()
    if total <= 0:
        return 0.0
    out = 0.0
    for g in range(G):
        if tot_g[g] > 0:
            p = T[g] / tot_g[g]
            out += tot_g[g] / total * (1.0 - float(np.sum(p ** 2)))
    return out
