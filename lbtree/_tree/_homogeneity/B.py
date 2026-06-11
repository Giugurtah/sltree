"""Homogeneity strategy: B — latent budget profiles (B) shared across strata."""
from __future__ import annotations
import numpy as np
from .base import HomogeneityStrategy
from ..split import _splitS


class HomogeneityB(HomogeneityStrategy):

    def get_treshold_values(self, best_threshold, x_vals, x_s):
        bt         = np.asarray(best_threshold)   # shape (K, I)
        strat_vals = np.unique(x_s)
        return [x_vals[bt[t] > 0] for t in range(len(strat_vals))]

    def split(self, X_best, x_s, thresholds):
        return _splitS(X_best, x_s, thresholds)

    def compute_lift(self, beta, distribution):
        bt = np.asarray(beta)
        if bt.ndim == 3:
            bt = bt[0]   # single shared B
        beta_left  = bt[:, 0]
        beta_right = bt[:, 1]
        return beta_left / distribution, beta_right / distribution
