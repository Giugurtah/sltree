# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/_tree/split.py
=====================
Splitting functions for PredictabilityTree and SLBT.

score_predictability()  — call the C PredictabilityTree backend (twoStage / twoing)
score_slbt()    — call the C SLBT backend (slba); takes an explicit seed
_split()        — partition a column into left/right boolean masks
_splitS()       — stratified partition (SLBT)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
#  PredictabilityTree score
# ============================================================

def score_predictability(F: np.ndarray, model: str = "twoStage"):
    """
    Call the C PredictabilityTree splitting algorithm on a contingency matrix F.

    Parameters
    ----------
    F     : np.ndarray, shape (I, J) — row-conditional contingency matrix
    model : str — "twoStage" | "twoing"

    Returns
    -------
    pi : float          — best PPI achieved
    S  : np.ndarray (I,) — binary left-mask (1 = row goes left)
    """
    from sltree._backend._pitree import twoStage, twoing

    _MODEL_MAP = {"twoStage": twoStage, "twoing": twoing, "twoClass": twoStage}
    if model not in _MODEL_MAP:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(_MODEL_MAP)}")

    I, J   = F.shape
    F_flat = F.ravel().astype(np.float64)
    return _MODEL_MAP[model](I, J, F_flat)


# ============================================================
#  PredictabilityTree weighted score (for AdaBoost)
# ============================================================

def score_predictability_weighted(
    F: np.ndarray,
    W: np.ndarray,
    model: str = "twoStage",
):
    """
    Call the weighted C PredictabilityTree splitting algorithm.

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
    from sltree._backend._pitree import twoStage_weighted, twoing_weighted

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


#: Row weighting of the least-squares SLBM fit.
LS_WEIGHTS = ("budget", "frequency")


def _ls_table(Fs_noN: np.ndarray, Fs: np.ndarray, ls_weights: str) -> np.ndarray:
    """
    Table handed to the least-squares fit.

    "budget"    : the observed budgets p_{.|i(t)} themselves, so every
                  observed (stratum, category) row weighs the same in the
                  loss.  This is what the library did before this revision.
    "frequency" : budgets multiplied by the row frequency, i.e. Eq. 13 of
                  the SLBT paper with D_t = diag of the row counts.  Rows
                  are rescaled so that the mean weight of the observed rows
                  is 1: the minimiser is unchanged, and the loss stays on
                  the same scale as in the "budget" case, which keeps the
                  step size of the descent comparable.
    """
    if ls_weights == "budget":
        return Fs
    if ls_weights == "frequency":
        row_w = Fs_noN.sum(axis=2)                 # (K, I) joint row mass
        observed = row_w > 0
        if not observed.any():
            return Fs
        row_w = row_w / row_w[observed].mean()
        return Fs * row_w[:, :, None]
    raise ValueError(
        f"Unknown ls_weights '{ls_weights}'. Choose from: {list(LS_WEIGHTS)}"
    )


#: Increment of the splitmix64 generator of csrc/common/lbrandom.h.
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_U64 = (1 << 64) - 1


def als_seed(base: int, node: int, feature: int) -> int:
    """
    Seed of the least-squares fit of one candidate split.

    It depends only on the seed of the fit (``base``), on the node (its
    position in the tree) and on the predictor (its column index in the data
    given to ``fit``).  Earlier versions drew the seeds one after the other
    from a single generator, so the seed of a predictor depended on how many
    fits had been run before it: skipping a predictor (``FAST``) or running
    two of them at the same time changed every split below.  With the seed
    fixed by (fit, node, predictor) the tree does not depend on the order or
    on the number of fits.
    """
    ss = np.random.SeedSequence([int(base), int(node), int(feature)])
    return int(ss.generate_state(1, dtype=np.uint64)[0]) & (2 ** 63 - 1)


def start_seeds(seed: int, n_init: int, KA: int, KB: int, I: int, J: int) -> list[int]:
    """
    Seeds that make ``n_init`` single-start calls of the C fit draw exactly
    the starting matrices of one call with ``n_init`` starts.

    The C fit draws, for every start, (KA*I + KB*J)*2 uniforms from one
    splitmix64 stream, whose state only grows by a fixed increment per draw:
    start s therefore begins where a stream seeded with
    seed + s*(KA*I + KB*J)*2*increment begins.
    """
    D = (KA * I + KB * J) * 2
    s0 = int(seed) & _U64
    return [(s0 + s * D * _SPLITMIX_GAMMA) & _U64 for s in range(n_init)]


def _same_split(S_best, S_cand, obs_alpha, per_stratum) -> bool:
    """
    Same split on the observed rows, up to the label switches the model
    cannot identify (swap of the two children, per stratum when every stratum
    has its own A and B).  Python twin of ``slba_same_split`` in slbt_als.c.
    """
    global_same = global_swap = True
    for t in range(S_best.shape[0]):
        b = S_best[t][obs_alpha[t]] > 0.5
        c = S_cand[t][obs_alpha[t]] > 0.5
        same = bool(np.all(b == c))
        swap = bool(np.all(b != c))
        if per_stratum:
            if not same and not swap:
                return False
        else:
            global_same &= same
            global_swap &= swap
    return True if per_stratum else (global_same or global_swap)


def score_slbt(Fs_noN: np.ndarray, Fs: np.ndarray, homogeneity: str = "none",
               seed: int = 0, n_init: int = 1, ls_weights: str = "budget",
               max_iter: int = 5000, tol: float = 1e-12,
               optimizer: str = "softmax", tie_break: str = "loss",
               tie_tol: float = 1e-7, executor=None):
    """
    Call the C SLBT splitting algorithm (slba).

    Parameters
    ----------
    Fs_noN     : np.ndarray, shape (K, I, J) — joint proportions at the node
    Fs         : np.ndarray, shape (K, I, J) — row-conditional budgets
    homogeneity : str — "none" | "A" | "B" | "AB"
    seed        : int — 64-bit seed for the random initialisation
    n_init      : int — number of random starts of the least-squares fit
    ls_weights  : {"budget", "frequency"} — row weighting of the fit
    max_iter    : int — iteration cap per start
    tol         : float — relative-decrease tolerance
    optimizer   : {"softmax", "projected"} — see ``_backend._slbt.slba``
    tie_break   : {"loss", "pi"}
        Which of the n_init starts is kept.
        "loss" : the start with the lowest loss (first one on exact ties).
        "pi"   : among the starts whose loss is within ``tie_tol`` (relative)
                 of the lowest, the one whose split has the largest PI.
                 The latent budget model with K = 2 is not identified:
                 different (A, B) reach the same optimum and alpha can cross
                 0.5 along that set, so the split rule alpha >= 0.5 is not
                 fixed by the loss alone.  "pi" resolves the indeterminacy in
                 favour of the most predictive of the equivalent splits.
    tie_tol     : float — relative tolerance defining "same loss" for "pi"
    executor    : concurrent.futures.Executor, optional
        When given and n_init > 1, the starts run in parallel on it (the C
        fit releases the GIL).  The starts are the same as in the sequential
        run (see ``start_seeds``), so the result is identical.

    Both tie-break rules use the same starts: start s of a fit with seed
    ``seed`` begins from the matrices that the C multistart draws for it.

    Returns
    -------
    pi    : float  — tau_{Y|s,Z}
    S     : np.ndarray, shape (KA, I)
    alpha : np.ndarray, shape (KA, I, 2)
    beta  : np.ndarray, shape (KB, J, 2)
    diag  : dict — convergence diagnostics
    """
    from sltree._backend._slbt import slba

    if homogeneity not in _HOMOGENEITY_MAP:
        raise ValueError(
            f"Unknown homogeneity '{homogeneity}'. "
            f"Choose from: {list(_HOMOGENEITY_MAP)}"
        )
    if optimizer not in ("softmax", "projected"):
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Choose from: ['softmax', 'projected']"
        )
    if tie_break not in ("loss", "pi"):
        raise ValueError(f"Unknown tie_break '{tie_break}'. Choose from: ['loss', 'pi']")

    K, I, J    = Fs.shape
    KA, KB     = _HOMOGENEITY_MAP[homogeneity](K)
    FsNoN_flat = np.ascontiguousarray(Fs_noN.ravel(), dtype=np.float64)
    Fs_flat    = np.ascontiguousarray(_ls_table(Fs_noN, Fs, ls_weights).ravel(),
                                      dtype=np.float64)

    parallel = executor is not None and n_init > 1
    if n_init == 1 or (tie_break == "loss" and not parallel):
        # one C call runs all the starts
        return slba(K, KA, KB, I, J, FsNoN_flat, Fs_flat, seed,
                    n_init=n_init, max_iter=max_iter, tol=tol, optimizer=optimizer)

    # one C call per start
    seeds = start_seeds(seed, n_init, KA, KB, I, J)

    def run(sd):
        return slba(K, KA, KB, I, J, FsNoN_flat, Fs_flat, sd,
                    n_init=1, max_iter=max_iter, tol=tol, optimizer=optimizer)

    runs = list(executor.map(run, seeds)) if parallel else [run(sd) for sd in seeds]
    losses = np.array([r[4]["loss"] for r in runs])

    observed  = (Fs_noN.sum(axis=2) > 0)           # (K, I)
    obs_alpha = observed if KA == K else observed.any(axis=0, keepdims=True)
    per_stratum = (KA == K and KB == K and K > 1)

    if tie_break == "loss":
        # same rule as the C multistart: a later start replaces the current
        # best only if its loss is lower by more than 1e-15
        chosen, best = 0, np.inf
        for s, l in enumerate(losses):
            if l < best - 1e-15:
                chosen, best = s, l
        pi, S, alpha, beta, diag = runs[chosen]
        diag = dict(diag)
        diag.update({
            "loss_worst_start":    float(losses.max()),
            "n_converged_starts":  int(sum(r[4]["converged"] for r in runs)),
            "n_same_split_starts": int(sum(_same_split(S, r[1], obs_alpha, per_stratum)
                                           for r in runs)),
            "n_init":              int(n_init),
        })
        return pi, S, alpha, beta, diag

    best_loss = losses.min()
    optimal = np.flatnonzero(losses <= best_loss + tie_tol * (1.0 + abs(best_loss)))
    pis = np.array([runs[k][0] for k in optimal])
    chosen = int(optimal[np.argmax(pis)])          # first one on exact PI ties
    pi, S, alpha, beta, diag = runs[chosen]

    def canon(S_):
        rows = []
        for t in range(S_.shape[0]):
            r = S_[t][obs_alpha[t]].astype(int)
            if per_stratum and r.size and r[0] == 0:
                r = 1 - r
            rows.append(tuple(r))
        flat = tuple(v for r in rows for v in r)
        if not per_stratum and flat and flat[0] == 0:
            flat = tuple(1 - v for v in flat)
        return flat

    diag = dict(diag)
    diag.update({
        "loss_worst_start":    float(losses.max()),
        "n_converged_starts":  int(sum(r[4]["converged"] for r in runs)),
        "n_same_split_starts": int(sum(_same_split(S, r[1], obs_alpha, per_stratum)
                                       for r in runs)),
        "n_init":              int(n_init),
        "n_optimal_starts":    int(optimal.size),
        "n_optimal_splits":    int(len({canon(runs[k][1]) for k in optimal})),
        "pi_range_optimal":    float(pis.max() - pis.min()),
    })
    return pi, S, alpha, beta, diag



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
    Stratified split: ``best_treshold`` is a list of arrays, one per stratum,
    in the order of ``np.unique(x_s)`` — the order in which
    ``_stratified_contingency`` builds the tables.

    Returns
    -------
    indexL : np.ndarray bool
    indexR : np.ndarray bool
    """
    x_s    = np.asarray(x_s)
    strata = np.unique(x_s)          # natural order, NOT string order:
                                     # with integer strata 2 and 10 the
                                     # string order ('10' < '2') would pair
                                     # each stratum with the wrong threshold
    x_vals = X_col.astype(str).to_numpy()
    indexL = np.zeros(len(x_vals), dtype=bool)
    for t, s in enumerate(strata):
        in_s  = (x_s == s)
        left  = {str(v) for v in best_treshold[t]}
        indexL |= in_s & np.isin(x_vals, list(left))
    return indexL, ~indexL
