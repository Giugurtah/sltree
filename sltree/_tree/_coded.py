# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/_tree/_coded.py
======================
Machinery shared by SLBT and SLBTWeighted during a fit: the data coded once
(``_utils.coding``), the tables of a node, the seeds of the least-squares
fits.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .._utils.coding import encode, encode_frame
from .._backend._slbt import gpi as _c_gpi_slbt
from .split import als_seed


def resolve_n_jobs(n_jobs) -> int:
    """scikit-learn convention: None or 1 -> 1, -1 -> all cores, -2 -> all but one."""
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer or None.")
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return n_jobs


class CodedSLBTMixin:
    """State of one fit, removed when the fit ends."""

    _executor = None
    _sub_rng = None

    def _begin_fit(self, X, y, x_s, w=None):
        """
        Integer codes of predictors, target and strata, and the original
        columns, which the nodes read by row position; seed of the fit;
        thread pool when ``n_jobs`` > 1.
        """
        # Seed of the fit.  An rng supplied from outside (the ensembles pass
        # one) wins over random_state, so the ensemble's seed governs its
        # trees; one draw per fit, whatever the size of the tree.
        if self._rng is not None:
            self._seed_base = int(self._rng.integers(0, 2 ** 63))
        elif isinstance(self.random_state, (int, np.integer)):
            self._seed_base = int(self.random_state)
        elif self.random_state is None:
            self._seed_base = int(np.random.SeedSequence().entropy)
        else:          # a Generator or a SeedSequence
            self._seed_base = int(np.random.default_rng(self.random_state).integers(0, 2 ** 63))
        # generator of the per-node predictor subsampling (forest "node" mode)
        if getattr(self, "node_max_features", None) is not None:
            self._sub_rng = (self._rng if self._rng is not None
                             else np.random.default_rng(self._seed_base))
        else:
            self._sub_rng = None
        n_jobs = resolve_n_jobs(getattr(self, "n_jobs", 1))
        self._executor = (ThreadPoolExecutor(max_workers=n_jobs)
                          if n_jobs > 1 and self.n_init > 1 else None)

        self._feat_names = list(X.columns)
        self._C, self._C_sizes = encode_frame(X)
        self._yc, self._J = encode(y)
        self._sc, self._G = encode(x_s)
        self._X_cols = [X.iloc[:, c] for c in range(X.shape[1])]
        self._y = y if isinstance(y, pd.Series) else pd.Series(y)
        self._xs = x_s
        self._w = None if w is None else np.asarray(w, dtype=np.float64)

    def _end_fit(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        for a in ("_C", "_C_sizes", "_yc", "_sc", "_X_cols", "_y", "_xs", "_w",
                  "_executor", "_sub_rng"):
            setattr(self, a, None)

    def _node_tables(self, rows, cols, nc):
        """
        Stratified GPI of every candidate predictor, sorted as
        ``_gpi_stratified`` sorts them (descending, ties by name), and its
        joint table, built once and reused by the split search.
        """
        vals, names, tables = [], [], {}
        for c in cols:
            F = nc.joint(self._C[rows, c], self._C_sizes[c])
            K, I, J = F.shape
            vals.append(_c_gpi_slbt(K, I, J, F.ravel().astype(np.float64)))
            names.append(self._feat_names[c])
            tables[self._feat_names[c]] = (c, F)
        vals, names = zip(*sorted(zip(vals, names), reverse=True))
        return vals, names, tables

    def _als_seed(self, pos, c):
        """Seed of the fit of predictor ``c`` at node ``pos`` (see ``split.als_seed``)."""
        return als_seed(self._seed_base, pos, c)
