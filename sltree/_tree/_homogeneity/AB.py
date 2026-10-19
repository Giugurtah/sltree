# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""Homogeneity strategy: AB — both A and B shared across strata."""
from __future__ import annotations
import numpy as np
from .base import HomogeneityStrategy
from .._stratified import safe_ratio
from ..split import _split


class HomogeneityAB(HomogeneityStrategy):
    """
    Both A and B shared across strata: the most constrained setting.

    Equivalent to ignoring the stratification during the split search, and the
    setting SLBT falls back to when ``fit`` is called without ``x_s``.  Useful
    as a baseline: comparing it against ``HomogeneityNone`` measures how much
    the stratification actually buys.
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
