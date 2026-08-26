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
from ..split import _splitS


class HomogeneityNone(HomogeneityStrategy):

    def get_treshold_values(self, best_threshold, x_vals, x_s):
        bt         = np.asarray(best_threshold)   # shape (K, I)
        strat_vals = np.unique(x_s)
        return [x_vals[bt[t] > 0] for t in range(len(strat_vals))]

    def split(self, X_best, x_s, thresholds):
        return _splitS(X_best, x_s, thresholds)

    def compute_lift(self, beta, distribution):
        lift1, lift2 = [], []
        for t in range(len(beta)):
            lift1.append(np.array([b[0] for b in beta[t]]) / distribution)
            lift2.append(np.array([b[1] for b in beta[t]]) / distribution)
        return lift1, lift2
