# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""Homogeneity strategy: none — each stratum has its own A and B."""
from __future__ import annotations
import numpy as np
from .base import HomogeneityStrategy
from .._stratified import safe_ratio
from ..split import _splitS


class HomogeneityNone(HomogeneityStrategy):
    """
    No constraint: every stratum gets its own A and B.

    The most permissive of the four.  Because the mixing parameters are free,
    the split direction may differ between strata — a modality can go left in
    one stratum and right in another.  It is the only setting, together with
    ``HomogeneityB``, able to represent a relationship that reverses across
    strata, which is what the London mode-choice application turns on.

    The price is the largest number of free parameters, hence the greatest
    exposure to overfitting on small strata.
    """


    def get_treshold_values(self, best_threshold, x_vals, x_s):
        bt         = np.asarray(best_threshold)   # shape (K, I)
        strat_vals = np.unique(x_s)
        return [x_vals[bt[t] > 0] for t in range(len(strat_vals))]

    def split(self, X_best, x_s, thresholds):
        return _splitS(X_best, x_s, thresholds)

    def compute_lift(self, beta, strat_dist):
        """
        Lift (Eq. 4): beta_{j|k(g)} / p_{j|(g)}, one array per stratum.

        ``strat_dist`` has one row per stratum present at the node: the class
        distribution of that stratum, p_{j|(g)}(t), which is the prior of
        Eq. 4.  Each stratum has its own latent budgets here.
        """
        beta = np.asarray(beta)
        lift1 = [safe_ratio(beta[t][:, 0], strat_dist[t]) for t in range(len(beta))]
        lift2 = [safe_ratio(beta[t][:, 1], strat_dist[t]) for t in range(len(beta))]
        return lift1, lift2
