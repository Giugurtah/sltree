"""
lbtree/_tree/split.py
=====================
Splitting functions for SCTree and SLBT.

score_sctree()  — call the C SCTree backend (twoStage / twoing)
score_slbt()    — call the C SLBT backend (slba)
_split()        — partition a column into left/right boolean masks
_splitS()       — stratified partition (SLBT)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
#  SCTree score
# ============================================================

def score_sctree(F: np.ndarray, model: str = "twoStage"):
    """
    Call the C SCTree splitting algorithm on a contingency matrix F.

    Parameters
    ----------
    F     : np.ndarray, shape (I, J) — row-conditional contingency matrix
    model : str — "twoStage" | "twoing"

    Returns
    -------
    pi : float          — best PPI achieved
    S  : np.ndarray (I,) — binary left-mask (1 = row goes left)
    """
    from lbtree._backend._lbtree import twoStage, twoing

    _MODEL_MAP = {"twoStage": twoStage, "twoing": twoing, "twoClass": twoStage}
    if model not in _MODEL_MAP:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(_MODEL_MAP)}")

    I, J   = F.shape
    F_flat = F.ravel().astype(np.float64)
    return _MODEL_MAP[model](I, J, F_flat)


# ============================================================
#  SCTree weighted score (for AdaBoost)
# ============================================================

def score_sctree_weighted(
    F: np.ndarray,
    W: np.ndarray,
    model: str = "twoStage",
):
    """
    Call the weighted C SCTree splitting algorithm.

    Parameters
    ----------
    F : np.ndarray, shape (I, J)
    W : np.ndarray, shape (I,) — row weight vector
    model : str

    Returns
    -------
    pi : float
    S  : np.ndarray (I,)
    """
    from lbtree._backend._lbtree import twoStage_weighted, twoing_weighted

    _MODEL_MAP = {"twoStage": twoStage_weighted, "twoing": twoing_weighted, "twoClass": twoStage_weighted}
    if model not in _MODEL_MAP:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(_MODEL_MAP)}")

    I, J   = F.shape
    F_flat = F.ravel().astype(np.float64)
    W_arr  = np.asarray(W, dtype=np.float64)
    return _MODEL_MAP[model](I, J, W_arr, F_flat)


# ============================================================
#  SLBT score
# ============================================================

_HOMOGENEITY_MAP = {
    "none": lambda K: (K, K),
    "A":    lambda K: (1, K),
    "B":    lambda K: (K, 1),
    "AB":   lambda K: (1, 1),
}


def score_slbt(Fs_noN: np.ndarray, Fs: np.ndarray, homogeneity: str = "none"):
    """
    Call the C SLBT splitting algorithm (slba).

    Parameters
    ----------
    Fs_noN     : np.ndarray, shape (K, I, J) — raw joint frequencies (not normalised)
    Fs         : np.ndarray, shape (K, I, J) — row-conditional stratified frequencies
    homogeneity : str — "none" | "A" | "B" | "AB"

    Returns
    -------
    pi    : float
    S     : np.ndarray, shape (KA, I)
    alpha : np.ndarray, shape (KA, I, 2)
    beta  : np.ndarray, shape (KB, J, 2)
    """
    from lbtree._backend._slbt import slba

    if homogeneity not in _HOMOGENEITY_MAP:
        raise ValueError(
            f"Unknown homogeneity '{homogeneity}'. "
            f"Choose from: {list(_HOMOGENEITY_MAP)}"
        )

    K, I, J    = Fs.shape
    KA, KB     = _HOMOGENEITY_MAP[homogeneity](K)
    FsNoN_flat = Fs_noN.ravel().astype(np.float64)
    Fs_flat    = Fs.ravel().astype(np.float64)

    return slba(K, KA, KB, I, J, FsNoN_flat, Fs_flat)


# ============================================================
#  Column-splitting helpers
# ============================================================

def _split(X_col: pd.Series, best_treshold) -> tuple:
    """
    Partition a predictor column into left / right boolean masks.

    Rows whose value appears in ``best_treshold`` go left.

    Returns
    -------
    indexL : np.ndarray bool
    indexR : np.ndarray bool
    """
    x_vals      = X_col.astype(str)
    left_values = {str(v) for v in best_treshold}
    indexL      = x_vals.isin(left_values).to_numpy()
    return indexL, ~indexL


def _splitS(X_col: pd.Series, x_s: np.ndarray, best_treshold) -> tuple:
    """
    Stratified split: ``best_treshold`` is a list of arrays, one per stratum.

    Returns
    -------
    indexL : np.ndarray bool
    indexR : np.ndarray bool
    """
    x_vals   = X_col.astype(str)
    s_vals   = x_s.astype(str)
    strata   = np.unique(s_vals)

    left_pairs = set()
    for t, s in enumerate(strata):
        for v in best_treshold[t]:
            left_pairs.add((str(v), s))

    pairs  = list(zip(x_vals, s_vals))
    indexL = np.array([p in left_pairs for p in pairs])
    return indexL, ~indexL
