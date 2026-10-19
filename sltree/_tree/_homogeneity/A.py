# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""Homogeneity strategy: A — mixing parameters (A) shared across strata."""
from __future__ import annotations
import numpy as np
from .base import HomogeneityStrategy
from .._stratified import safe_ratio
from ..split import _split


class HomogeneityA(HomogeneityStrategy):
    """
    A (the mixing parameters) shared across strata; B free.

    One split direction is imposed on every stratum, while the latent budget
    profiles stay stratum-specific.  The tree structure is therefore common and
    only the composition of the classes within each branch varies by stratum.

    A relationship that reverses between strata is invisible here: sharing A
    forces a single threshold direction.
    """


    def get_treshold_values(self, best_threshold, x_vals, x_s):
        bt = np.asarray(best_threshold)
        if bt.ndim == 2:
            bt = bt[0]   # single shared threshold
        return x_vals[bt > 0]

    def split(self, X_best, x_s, threshold):
        return _split(X_best, threshold)

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
