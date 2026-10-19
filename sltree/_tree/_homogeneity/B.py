# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""Homogeneity strategy: B — latent budget profiles (B) shared across strata."""
from __future__ import annotations
import numpy as np
from .base import HomogeneityStrategy
from .._stratified import safe_ratio
from ..split import _splitS


class HomogeneityB(HomogeneityStrategy):
    """
    B (the latent budget profiles) shared across strata; A free.

    The mirror image of ``HomogeneityA``: each stratum may split in its own
    direction, but the profiles the branches point to are common.  Like
    ``HomogeneityNone`` it can represent a reversal, while spending fewer
    parameters on the profiles.
    """


    def get_treshold_values(self, best_threshold, x_vals, x_s):
        bt         = np.asarray(best_threshold)   # shape (K, I)
        strat_vals = np.unique(x_s)
        return [x_vals[bt[t] > 0] for t in range(len(strat_vals))]

    def split(self, X_best, x_s, thresholds):
        return _splitS(X_best, x_s, thresholds)

    def compute_lift(self, beta, strat_dist):
        """
        Lift (Eq. 4): beta_{j|k} / p_{j|(g)}, one array per stratum.

        The latent budgets are shared across strata, but the prior
        p_{j|(g)}(t) is not, so the Lift still differs between strata.  With a
        single stratum (LBT, no stratification) a flat array is returned.
        """
        bt = np.asarray(beta)
        if bt.ndim == 3:
            bt = bt[0]   # single shared B
        strat_dist = np.atleast_2d(strat_dist)
        lift1 = [safe_ratio(bt[:, 0], d) for d in strat_dist]
        lift2 = [safe_ratio(bt[:, 1], d) for d in strat_dist]
        if len(strat_dist) == 1:
            return lift1[0], lift2[0]
        return lift1, lift2
