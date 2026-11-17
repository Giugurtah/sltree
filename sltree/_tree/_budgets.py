# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/_tree/_budgets.py
========================
Mixing parameters and latent budgets of the split chosen at a node, labelled
with the strata, modalities and classes they refer to (SLBT, SLBTWeighted).

The least-squares fit returns them as arrays indexed by position in the joint
table of the node: strata, modalities and classes present there, in sorted
order.  The split rule sends a modality to the left child when its mixing
parameter for the first budget is at least 0.5, so the first budget is the
one of the left child and the second the one of the right child; LIFT_1 and
LIFT_2 are built from the same two columns of beta.
"""

from __future__ import annotations

import numpy as np

# Which homogeneity settings give each stratum its own A (mixing parameters)
# and its own B (latent budgets).
_STRATUM_A = ("none", "B")
_STRATUM_B = ("none", "A")


def label_budgets(alpha, beta, F, modalities, strata, homogeneity):
    """
    Label the mixing parameters and the latent budgets of a split.

    Parameters
    ----------
    alpha       : array, shape (KA, I, 2) — mixing parameters from ``score_slbt``
    beta        : array, shape (KB, J, 2) — latent budgets from ``score_slbt``
    F           : array, shape (K, I, J) — joint table of the split predictor
                  at the node (strata x modalities x classes)
    modalities  : the I modalities of the predictor at the node, sorted
    strata      : the K strata present at the node, sorted
    homogeneity : "none", "A", "B" or "AB"

    Returns
    -------
    alpha_left : dict
        Mixing parameter of the left budget, alpha_{1|i(g)}; the one of the
        right budget is 1 minus it.  ``{stratum: {modality: value}}`` when
        every stratum has its own A ("none", "B"), ``{modality: value}`` when
        A is shared ("A", "AB").  A modality that does not occur in a stratum
        at the node gets None there: its mixing parameter does not enter the
        fit and is not identified.
    beta_left, beta_right : dict or list
        Latent budgets of the left and right child, beta_{j|1} and beta_{j|2},
        for the classes present at the node in sorted order (the ``labels``
        of the node).  ``{stratum: [values]}`` when
        every stratum has its own B ("none", "A"), a single list when B is
        shared ("B", "AB").
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    mods = [str(m) for m in modalities]
    strata = [str(s) for s in strata]
    observed = np.asarray(F).sum(axis=2) > 0                     # (K, I)

    if homogeneity in _STRATUM_A:
        alpha_left = {
            s: {m: (float(alpha[t, i, 0]) if observed[t, i] else None)
                for i, m in enumerate(mods)}
            for t, s in enumerate(strata)
        }
    else:
        alpha_left = {m: float(alpha[0, i, 0]) for i, m in enumerate(mods)}

    if homogeneity in _STRATUM_B:
        beta_left = {s: beta[t, :, 0].tolist() for t, s in enumerate(strata)}
        beta_right = {s: beta[t, :, 1].tolist() for t, s in enumerate(strata)}
    else:
        beta_left, beta_right = beta[0, :, 0].tolist(), beta[0, :, 1].tolist()

    return alpha_left, beta_left, beta_right
