# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/_utils/utils.py
======================
Contingency-matrix utilities used by both PredictabilityTree and SLBT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _contingency_matrix(x: pd.Series, y: pd.Series) -> np.ndarray:
    """
    Build a row-conditional contingency matrix F (shape I × J).

    Each row i is the conditional distribution  P(Y=j | X=x_i).
    Rows with zero marginal are set to 0 (via fillna).

    Parameters
    ----------
    x : pd.Series   — predictor column (categorical)
    y : pd.Series   — target column (categorical)

    Returns
    -------
    F : np.ndarray, shape (I, J), dtype float64
    """
    F = pd.crosstab(x, y, dropna=False)
    F_cond = F.div(F.sum(axis=1), axis=0).fillna(0)
    return F_cond.to_numpy(dtype=np.float64)


def _joint_table(x, y, x_s, w=None) -> np.ndarray:
    """
    K x I x J table of (weighted) counts, normalised to sum to 1.

    Levels are ordered as ``np.unique`` orders them (sorted), for the strata,
    the predictor modalities and the classes alike — the order in which every
    caller pairs thresholds, budgets and strata.

    Built with a single ``np.bincount`` on the combined cell index.  Earlier
    versions called ``pd.crosstab`` once per stratum, which accounted for
    ~97% of the fitting time of an SLBT.
    """
    _, xi = np.unique(np.asarray(x),   return_inverse=True)
    _, yi = np.unique(np.asarray(y),   return_inverse=True)
    _, si = np.unique(np.asarray(x_s), return_inverse=True)
    I = int(xi.max()) + 1 if xi.size else 0
    J = int(yi.max()) + 1 if yi.size else 0
    K = int(si.max()) + 1 if si.size else 0
    cell = (si.ravel() * I + xi.ravel()) * J + yi.ravel()
    weights = None if w is None else np.asarray(w, dtype=np.float64)
    F = np.bincount(cell, weights=weights, minlength=K * I * J).astype(np.float64)
    F = F.reshape(K, I, J)
    total = F.sum()
    if total > 0:
        F /= total
    return F


def _row_conditional(F: np.ndarray) -> np.ndarray:
    """Budgets p_{.|i(k)}: each (stratum, modality) row divided by its sum."""
    row = F.sum(axis=2, keepdims=True)
    return np.divide(F, row, out=np.zeros_like(F), where=row > 0)


def _stratified_contingency_weighted(
    x: pd.Series,
    y: pd.Series,
    x_s: np.ndarray,
    w: np.ndarray,
    norm: bool = False,
) -> np.ndarray:
    """
    Weighted version of ``_stratified_contingency``.

    Each cell (k, i, j) is the sum of sample weights where
    ``x_s == s_k``, ``x == x_i``, ``y == y_j``, normalised by the
    total weight sum.

    Parameters
    ----------
    x    : pd.Series
    y    : pd.Series
    x_s  : np.ndarray  — stratum indicator
    w    : np.ndarray  — per-sample weights
    norm : bool, default False
        Same semantics as in ``_stratified_contingency``:
        False → joint weight proportions (K, I, J), summing to 1
        True  → row-conditional within-stratum budgets (K, I, J)

    Returns
    -------
    Fs : np.ndarray, shape (K, I, J), dtype float64
    """
    F = _joint_table(x, y, x_s, w)
    return _row_conditional(F) if norm else F


def _stratified_contingency(
    x: pd.Series,
    y: pd.Series,
    x_s: np.ndarray,
    norm: bool = False,
) -> np.ndarray:
    """
    Build an array of contingency matrices stratified by ``x_s``.

    Parameters
    ----------
    x    : pd.Series    — predictor column
    y    : pd.Series    — target column
    x_s  : np.ndarray  — stratum indicator (one entry per observation)
    norm : bool, default False
        If False  → joint proportions Fs[k, i, j], summing to 1 over all
                    k, i, j
        If True   → row-conditional within-stratum budgets
                    Fs[k, i, j] = P(Y=j | X=i, stratum k)  (rows of zeros for
                    modalities absent from a stratum)

    Returns
    -------
    Fs : np.ndarray, shape (K, I, J), dtype float64
    """
    F = _joint_table(x, y, x_s)
    return _row_conditional(F) if norm else F
