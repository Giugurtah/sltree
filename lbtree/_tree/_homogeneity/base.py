"""
lbtree/_tree/_homogeneity/base.py
==================================
Abstract base class for homogeneity strategies (SLBT only).

The four concrete strategies are:
  HomogeneityNone  — no constraint (each stratum has its own A and B)
  HomogeneityA     — A (mixing parameters) shared across strata
  HomogeneityB     — B (latent budget profiles) shared across strata
  HomogeneityAB    — both A and B shared across strata
"""

from __future__ import annotations


class HomogeneityStrategy:
    """
    Abstract interface for stratified-split homogeneity strategies.

    All concrete strategies must implement the three methods below.
    """

    def get_treshold_values(self, best_threshold, x_vals, x_s):
        """
        Convert the raw SLBT output ``S`` array into threshold values
        suitable for storing in a Node and for later splitting.

        Parameters
        ----------
        best_threshold : np.ndarray
            Raw ``S`` output from ``score_slbt``.
        x_vals : np.ndarray
            Sorted unique modality values of the current predictor.
        x_s : np.ndarray
            Stratum array for the current subset.

        Returns
        -------
        thresholds : np.ndarray or list of np.ndarray
        """
        raise NotImplementedError

    def split(self, X_best, x_s, thresholds):
        """
        Split the predictor column into left / right boolean masks.

        Parameters
        ----------
        X_best     : pd.Series  — predictor column
        x_s        : np.ndarray — stratum array
        thresholds  : as returned by ``get_treshold_values``

        Returns
        -------
        (indexL, indexR) : tuple of np.ndarray bool
        """
        raise NotImplementedError

    def compute_lift(self, beta, distribution):
        """
        Compute LIFT_1 and LIFT_2 from the SLBT beta output.

        Parameters
        ----------
        beta         : np.ndarray — beta output from ``score_slbt``
        distribution : np.ndarray — relative class frequencies at the node

        Returns
        -------
        (lift1, lift2) : tuple
        """
        raise NotImplementedError


def get_homogeneity_strategy(homogeneity: str) -> HomogeneityStrategy:
    """Factory: return the correct HomogeneityStrategy instance."""
    if homogeneity == "none":
        from .none import HomogeneityNone
        return HomogeneityNone()
    elif homogeneity == "A":
        from .A import HomogeneityA
        return HomogeneityA()
    elif homogeneity == "B":
        from .B import HomogeneityB
        return HomogeneityB()
    elif homogeneity == "AB":
        from .AB import HomogeneityAB
        return HomogeneityAB()
    else:
        raise ValueError(
            f"Unknown homogeneity '{homogeneity}'. "
            "Choose from: 'none', 'A', 'B', 'AB'."
        )
