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
from ..split import _split


class HomogeneityA(HomogeneityStrategy):

    def get_treshold_values(self, best_threshold, x_vals, x_s):
        bt = np.asarray(best_threshold)
        if bt.ndim == 2:
            bt = bt[0]   # single shared threshold
        return x_vals[bt > 0]

    def split(self, X_best, x_s, threshold):
        return _split(X_best, threshold)

    def compute_lift(self, beta, distribution):
        lift1, lift2 = [], []
        for t in range(len(beta)):
            lift1.append(np.array([b[0] for b in beta[t]]) / distribution)
            lift2.append(np.array([b[1] for b in beta[t]]) / distribution)
        return lift1, lift2
