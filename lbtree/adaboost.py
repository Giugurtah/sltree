"""
lbtree/adaboost.py
==================
AdaBoostForest — ensemble of SCTreeWeighted with AdaBoost.M1 algorithm.

Note on categorical data
------------------------
AdaBoost with depth-1 stumps performs poorly on multi-class categorical
predictors.  Prefer max_depth >= 2 (recommended: 3–4) or use
SCTreeForest (Random Forest) which is more robust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sctree_weighted import SCTreeWeighted


class AdaBoostForest:
    """
    AdaBoost.M1 ensemble of SCTreeWeighted.

    Parameters
    ----------
    n_estimators : int, default 50
        Maximum number of boosting iterations (weak learners).
    max_depth : int, default 3
        Maximum depth of each weak learner.
        For multi-class categorical data, use >= 2.
    learning_rate : float, default 1.0
        Shrinkage applied to each alpha_t.
    min_error_improvement : float, default 0.01
        Early stopping: if the weighted error does not improve by at
        least this amount for 5 consecutive iterations (after the 10th),
        training stops.
    random_state : int | None
        Seed for reproducibility.
    **tree_kwargs
        Additional keyword arguments forwarded to SCTreeWeighted
        (e.g. min_ppi, min_gpi, feats_viewed, …).

    Attributes (set after fit)
    --------------------------
    estimators_ : list[SCTreeWeighted]
    estimator_weights_ : list[float]
        Alpha_t for each weak learner.
    estimator_errors_ : list[float]
        Weighted error of each weak learner.
    classes_ : np.ndarray
    """

    def __init__(
        self,
        n_estimators: int          = 50,
        max_depth: int             = 3,
        learning_rate: float       = 1.0,
        min_error_improvement: float = 0.01,
        random_state               = None,
        **tree_kwargs,
    ):
        self.n_estimators           = n_estimators
        self.max_depth              = max_depth
        self.learning_rate          = learning_rate
        self.min_error_improvement  = min_error_improvement
        self.random_state           = random_state
        self.tree_kwargs            = tree_kwargs

        self.estimators_        : list[SCTreeWeighted] = []
        self.estimator_weights_ : list[float]          = []
        self.estimator_errors_  : list[float]          = []
        self.classes_           : np.ndarray | None    = None

        self._rng = np.random.default_rng(random_state)

    # ================================================================
    #  PUBLIC — fit / predict / predict_proba / staged_predict
    # ================================================================

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AdaBoostForest":
        """
        Train weak learners with AdaBoost.M1.

        Returns
        -------
        self
        """
        self.classes_           = np.unique(y)
        self.estimators_        = []
        self.estimator_weights_ = []
        self.estimator_errors_  = []

        n_samples     = len(y)
        sample_weight = np.ones(n_samples) / n_samples

        best_err             = 1.0
        no_improvement_count = 0

        for t in range(self.n_estimators):
            # Fit weak learner
            tree = SCTreeWeighted(max_depth=self.max_depth, **self.tree_kwargs)
            tree.fit(X, y, sample_weight=sample_weight)
            preds = tree.predict(X)

            # Weighted error
            incorrect = (preds != y.values)
            err       = (sample_weight * incorrect).sum() / sample_weight.sum()

            # Early stopping: err >= 0.5 (weak learner is no better than chance)
            if err >= 0.5:
                if t == 0:
                    # Save first tree with negligible weight to avoid empty ensemble
                    self.estimators_.append(tree)
                    self.estimator_weights_.append(1e-10)
                    self.estimator_errors_.append(err)
                    print(f"  AdaBoost iter 1: err={err:.4f} >= 0.5, saved with alpha~0")
                break

            # Compute alpha
            err_clip = np.clip(err, 1e-10, 1 - 1e-10)
            alpha    = 0.5 * np.log((1 - err_clip) / err_clip)
            alpha   *= self.learning_rate

            if (t + 1) % 10 == 0 or t == 0 or t == self.n_estimators - 1:
                print(f"  AdaBoost iter {t + 1}/{self.n_estimators}: "
                      f"err={err:.4f}, alpha={alpha:.4f}")

            # Early stopping: no improvement
            if err < best_err - self.min_error_improvement:
                best_err             = err
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= 5 and t >= 10:
                    print("  Early stop: no improvement for 5 consecutive iterations.")
                    break

            # Update sample weights
            sample_weight *= np.exp(alpha * (2 * incorrect - 1))
            sample_weight /= sample_weight.sum()

            self.estimators_.append(tree)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(err)

        print(f"AdaBoost: {len(self.estimators_)} weak learners fitted.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the class with the highest weighted vote."""
        if not self.estimators_:
            raise RuntimeError("AdaBoostForest not fitted. Call fit() first.")
        votes = self._compute_vote_matrix(X)
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Class probability as normalised weighted votes.

        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
        """
        if not self.estimators_:
            raise RuntimeError("AdaBoostForest not fitted.")
        votes    = self._compute_vote_matrix(X)
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return votes / row_sums

    def staged_predict(self, X: pd.DataFrame):
        """
        Generator that yields predictions after each boosting stage.

        Useful for selecting the optimal number of estimators.
        """
        if not self.estimators_:
            raise RuntimeError("AdaBoostForest not fitted.")
        n_samples    = len(X)
        n_classes    = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        votes        = np.zeros((n_samples, n_classes))

        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X)
            for i, pred in enumerate(preds):
                if pred in class_to_idx:
                    votes[i, class_to_idx[pred]] += alpha
            yield self.classes_[np.argmax(votes, axis=1)]

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_estimators":          self.n_estimators,
            "max_depth":             self.max_depth,
            "learning_rate":         self.learning_rate,
            "min_error_improvement": self.min_error_improvement,
            "random_state":          self.random_state,
            **self.tree_kwargs,
        }

    def set_params(self, **params) -> "AdaBoostForest":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.tree_kwargs[k] = v
        return self

    # ================================================================
    #  PRIVATE
    # ================================================================

    def _compute_vote_matrix(self, X: pd.DataFrame) -> np.ndarray:
        n_samples    = len(X)
        n_classes    = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        votes        = np.zeros((n_samples, n_classes))
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X)
            for i, pred in enumerate(preds):
                if pred in class_to_idx:
                    votes[i, class_to_idx[pred]] += alpha
        return votes

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        status = (f"{len(self.estimators_)} weak learners fitted"
                  if self.estimators_ else "not fitted")
        return (
            f"AdaBoostForest("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate}, "
            f"[{status}])"
        )
