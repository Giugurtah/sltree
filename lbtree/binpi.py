"""
lbtree/binpi.py
===============
BINPI — Boosted Incremental Non-Parametric Imputation
for missing data in categorical predictor matrices.

Reference
---------
D'Ambrosio, Aria, Siciliano (2012).
"Accurate Tree-based Missing Data Imputation and Data Fusion
within the Statistical Learning Paradigm."
Journal of Classification 29.

Algorithm (BINPI-MISSING, Table 1 of the paper)
------------------------------------------------
Input: matrix Q (N × K) with missing values in some columns.

1. LEXICOGRAPHIC COLUMN ORDERING
   Columns are sorted by ascending number of missing values:
     - First:  the k₀ completely observed columns  → block X
     - Then:   columns with missing, ascending order → block Y

2. INCREMENTAL LOOP  for l = k₀, ..., K-1:
   a. Target     = (l+1)-th column of Z (first remaining Y column)
   b. Predictors = the l columns of X (observed or already imputed)
   c. Training   = only rows where the target is OBSERVED (not missing)
   d. Fitting    = ensemble on (predictors_train, target_train)
   e. Imputation = predict missing target values with the ensemble
   f. Update     = add the imputed column to X, remove from Y

3. Output: fully imputed matrix Q.

Available ensembles
-------------------
- "forest"   : SCTreeForest (Random Forest with bagging)
- "adaboost" : AdaBoostForest (boosting with re-weighting)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forest   import SCTreeForest
from .adaboost import AdaBoostForest


class BINPI:
    """
    BINPI: Boosted Incremental Non-Parametric Imputation.

    Imputes missing values in a categorical DataFrame using a sequence
    of ensemble models fitted incrementally.

    Parameters
    ----------
    ensemble : {"forest", "adaboost"}, default "forest"
        Type of ensemble used at each imputation step.
    n_estimators : int, default 20
        Number of trees / boosting iterations per ensemble.
    max_depth : int, default 1
        Maximum depth of each individual tree.
        For AdaBoost on categorical data, use >= 2 (recommend 3).
    max_features : int | float | "sqrt" | "log2" | None, default "sqrt"
        (forest only) Feature subsampling per tree.
    bootstrap : bool, default True
        (forest only) Row bootstrap sampling.
    learning_rate : float, default 1.0
        (adaboost only) Shrinkage on alpha.
    random_state : int | None
        Seed for reproducibility.
    **ensemble_kwargs
        Additional keyword arguments forwarded to the ensemble
        (e.g. min_ppi, min_gpi, …).

    Attributes (set after fit_transform)
    -------------------------------------
    imputed_matrix_ : pd.DataFrame
        Fully imputed matrix with the same columns as the input.
    ensembles_ : dict[str, SCTreeForest | AdaBoostForest]
        Ensemble fitted for each imputed column.
    imputation_order_ : list[str]
        Order in which columns were imputed.
    missing_counts_ : pd.Series
        Number of missing values per column before imputation.
    """

    def __init__(
        self,
        ensemble: str        = "forest",
        n_estimators: int    = 20,
        max_depth: int       = 1,
        max_features         = "sqrt",
        bootstrap: bool      = True,
        learning_rate: float = 1.0,
        random_state         = None,
        **ensemble_kwargs,
    ):
        if ensemble not in ("forest", "adaboost"):
            raise ValueError(
                f"ensemble='{ensemble}' not valid. Choose 'forest' or 'adaboost'."
            )

        self.ensemble         = ensemble
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.max_features     = max_features
        self.bootstrap        = bootstrap
        self.learning_rate    = learning_rate
        self.random_state     = random_state
        self.ensemble_kwargs  = ensemble_kwargs

        self.imputed_matrix_   : pd.DataFrame | None = None
        self.ensembles_        : dict                 = {}
        self.imputation_order_ : list[str]            = []
        self.missing_counts_   : pd.Series | None     = None

        self._rng = np.random.default_rng(random_state)

    # ================================================================
    #  PUBLIC API — fit_transform / transform
    # ================================================================

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in X and return the completed matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Categorical predictor matrix (dtype object or category).
            May contain NaN in any column.

        Returns
        -------
        pd.DataFrame
            Copy of X with all NaNs replaced by imputed values.
            Column order is identical to the input.
        """
        self.missing_counts_ = X.isnull().sum()

        complete_cols = [c for c in X.columns if self.missing_counts_[c] == 0]
        missing_cols  = [c for c in X.columns if self.missing_counts_[c] >  0]

        if not missing_cols:
            print("No missing values found. Returning original matrix.")
            self.imputed_matrix_ = X.copy()
            return self.imputed_matrix_

        # Sort columns with missing by ascending count
        missing_cols_sorted    = sorted(missing_cols, key=lambda c: self.missing_counts_[c])
        self.imputation_order_ = missing_cols_sorted

        print("=" * 60)
        print(f"BINPI — incremental imputation (ensemble={self.ensemble})")
        print(f"  Complete columns    : {len(complete_cols)}")
        print(f"  Columns with missing: {len(missing_cols_sorted)}")
        print(f"  Imputation order    : {missing_cols_sorted}")
        print("=" * 60)

        # Working copy: imputed values are added incrementally
        Z = X.copy()

        # Current predictor block: starts with completely observed columns
        current_predictors = list(complete_cols)

        for step, target_col in enumerate(missing_cols_sorted):
            n_missing = int(self.missing_counts_[target_col])
            print(f"\n[Step {step + 1}/{len(missing_cols_sorted)}] "
                  f"Target: '{target_col}'  (missing: {n_missing})")

            observed_mask = Z[target_col].notna()
            missing_mask  = Z[target_col].isna()

            n_train = observed_mask.sum()
            if n_train == 0:
                raise ValueError(
                    f"Column '{target_col}': no rows with observed value. "
                    "Cannot train the ensemble."
                )

            print(f"  Training rows: {n_train}  |  Rows to impute: {n_missing}")

            X_train  = Z.loc[observed_mask, current_predictors].reset_index(drop=True)
            y_train  = Z.loc[observed_mask, target_col].reset_index(drop=True)
            X_impute = Z.loc[missing_mask,  current_predictors].reset_index(drop=True)

            # Keep only predictors without any remaining missing values
            pred_complete = [
                c for c in current_predictors
                if X_train[c].notna().all() and X_impute[c].notna().all()
            ]

            if not pred_complete:
                raise ValueError(
                    f"Column '{target_col}': no predictor available without missing. "
                    "Check the column ordering."
                )

            if len(pred_complete) < len(current_predictors):
                dropped = set(current_predictors) - set(pred_complete)
                print(f"  [warn] Predictors excluded (still with missing): {dropped}")

            X_train  = X_train[pred_complete]
            X_impute = X_impute[pred_complete]

            # Fit ensemble and impute
            ens = self._create_ensemble()
            ens.fit(X_train, y_train)
            self.ensembles_[target_col] = ens

            imputed_values  = ens.predict(X_impute)
            missing_indices = Z.index[missing_mask]
            Z.loc[missing_indices, target_col] = imputed_values

            print("  Imputation completed.")

            # Expand predictor block
            current_predictors.append(target_col)

        # Restore original column order
        self.imputed_matrix_ = Z[X.columns]

        print("\n" + "=" * 60)
        print("BINPI — imputation complete.")
        print(f"  Residual missing: {self.imputed_matrix_.isnull().sum().sum()}")
        print("=" * 60)

        return self.imputed_matrix_

    def transform(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in a new DataFrame using the already-fitted ensembles.

        The ensembles are applied in the same incremental order as
        fit_transform().  Columns without missing values are left unchanged.

        Parameters
        ----------
        X_new : pd.DataFrame
            New DataFrame with the same columns as the original X.

        Returns
        -------
        pd.DataFrame
            Copy of X_new with NaNs replaced.
        """
        if not self.ensembles_:
            raise RuntimeError("Call fit_transform() before transform().")

        Z = X_new.copy()

        for target_col in self.imputation_order_:
            if target_col not in Z.columns:
                continue
            missing_mask = Z[target_col].isna()
            if not missing_mask.any():
                continue

            ens = self.ensembles_[target_col]

            if self.ensemble == "forest":
                known_feats = set()
                for feats in ens.estimators_features_:
                    known_feats.update(feats)
                pred_cols = [
                    c for c in Z.columns
                    if c != target_col
                    and Z.loc[missing_mask, c].notna().all()
                    and c in known_feats
                ]
            else:
                pred_cols = [
                    c for c in Z.columns
                    if c != target_col and Z.loc[missing_mask, c].notna().all()
                ]

            if not pred_cols:
                continue

            X_impute     = Z.loc[missing_mask, pred_cols].reset_index(drop=True)
            imputed_vals = ens.predict(X_impute)
            Z.loc[Z.index[missing_mask], target_col] = imputed_vals

        return Z[X_new.columns]

    def get_params(self, deep: bool = True) -> dict:
        return {
            "ensemble":      self.ensemble,
            "n_estimators":  self.n_estimators,
            "max_depth":     self.max_depth,
            "max_features":  self.max_features,
            "bootstrap":     self.bootstrap,
            "learning_rate": self.learning_rate,
            "random_state":  self.random_state,
            **self.ensemble_kwargs,
        }

    def set_params(self, **params) -> "BINPI":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.ensemble_kwargs[k] = v
        return self

    # ================================================================
    #  PUBLIC — summary
    # ================================================================

    def summary(self) -> str:
        """Return a text summary of the imputation."""
        if self.missing_counts_ is None:
            return "BINPI has not been run yet."

        lines = [
            "=" * 60,
            f"BINPI — imputation summary (ensemble={self.ensemble})",
            "=" * 60,
        ]
        missing_feats = self.missing_counts_[self.missing_counts_ > 0]
        if missing_feats.empty:
            lines.append("No missing values found.")
        else:
            lines.append(f"{'Column':<20} {'Missing':>8}  {'Order':>7}  Ensemble fitted")
            lines.append("-" * 60)
            for rank, col in enumerate(self.imputation_order_, 1):
                fitted = "yes" if col in self.ensembles_ else "no"
                lines.append(
                    f"  {col:<18} {int(self.missing_counts_[col]):>8}  "
                    f"{rank:>7}  {fitted}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)

    # ================================================================
    #  PRIVATE
    # ================================================================

    def _create_ensemble(self):
        seed = int(self._rng.integers(0, 2**31))
        if self.ensemble == "forest":
            return SCTreeForest(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=seed,
                max_depth=self.max_depth,
                **self.ensemble_kwargs,
            )
        else:  # adaboost
            # For categorical predictors, depth < 2 is typically too weak
            depth = max(self.max_depth, 3)
            return AdaBoostForest(
                n_estimators=self.n_estimators,
                max_depth=depth,
                learning_rate=self.learning_rate,
                random_state=seed,
                **self.ensemble_kwargs,
            )

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        status = (f"{len(self.ensembles_)} columns imputed"
                  if self.ensembles_ else "not run")
        return (
            f"BINPI("
            f"ensemble='{self.ensemble}', "
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"[{status}])"
        )
