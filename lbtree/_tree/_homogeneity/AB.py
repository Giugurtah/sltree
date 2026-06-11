"""Homogeneity strategy: AB — both A and B shared across strata."""
from __future__ import annotations
import numpy as np
from .base import HomogeneityStrategy
from ..split import _split


class HomogeneityAB(HomogeneityStrategy):

    def get_treshold_values(self, best_threshold, x_vals, x_s):
        bt = np.asarray(best_threshold)
        if bt.ndim == 2:
            bt = bt[0]   # single shared threshold
        return x_vals[bt > 0]

    def split(self, X_best, x_s, threshold):
        return _split(X_best, threshold)

    def compute_lift(self, beta, distribution):
        bt = np.asarray(beta)
        if bt.ndim == 3:
            bt = bt[0]
        beta_left  = bt[:, 0]
        beta_right = bt[:, 1]
        return beta_left / distribution, beta_right / distribution
