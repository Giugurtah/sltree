"""
lbtree/_preprocessing/categorizer.py
======================================
Categorizer — scikit-learn style discretizer for continuous variables.

Uses KMeans clustering (C backend) to convert numeric columns into
ordered categorical bins.  Three methods are available for choosing
the number of bins:

  - "fixed"      : user-specified k
  - "elbow"      : automatic K via the elbow method
  - "silhouette" : automatic K via the silhouette score
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._backend import (
    categorize_fixed_k,
    categorize_elbow,
    categorize_silhouette,
)


class Categorizer:
    """
    Automatic discretization of continuous variables using KMeans.

    Provides a scikit-learn style interface (fit / transform / fit_transform).

    Parameters
    ----------
    method : {"fixed", "elbow", "silhouette"}, default "elbow"
        Method for determining the number of bins.
    k : int, optional
        Number of bins.  Required when ``method="fixed"``.
    k_max : int, default 10
        Maximum number of bins to test (for "elbow" and "silhouette").
    k_min : int, default 2
        Minimum number of bins to consider (for "elbow" and "silhouette").
    min_size : int, default 1
        Minimum cluster size required for each bin.
    labels : list of str, optional
        Custom category labels (e.g. ["low", "medium", "high"]).
        When provided, takes priority over ``label_style``.
        If None, labels are generated according to ``label_style``.
    label_style : {"int", "interval"}, default "int"
        Controls how bin labels are generated when ``labels`` is None.
        ``"int"``      — integer indices (0, 1, 2, …).
        ``"interval"`` — human-readable range strings derived from the bin
                         edges, e.g. ``"<10"``, ``"10-20"``, ``"20-35"``,
                         ``">35"``.

    Attributes (set after fit)
    --------------------------
    bins_ : dict[str, np.ndarray]
        Bin edges per column.
    labels_ : dict[str, list]
        Bin labels per column.
    k_ : dict[str, int]
        Number of bins found per column.
    centers_ : dict[str, np.ndarray]
        Cluster centres per column.
    columns_ : list[str]
        Columns seen during fit.

    Examples
    --------
    >>> import pandas as pd
    >>> from lbtree import Categorizer
    >>> df = pd.DataFrame({"age": [18, 25, 45, 55, 75]})
    >>> cat = Categorizer(method="elbow", k_max=4)
    >>> df_cat = cat.fit_transform(df)
    """

    def __init__(
        self,
        method: str                 = "elbow",
        k: Optional[int]            = None,
        k_max: int                  = 10,
        k_min: int                  = 2,
        min_size: int               = 1,
        labels: Optional[List[str]] = None,
        label_style: str            = "int",
    ):
        if method not in ("fixed", "elbow", "silhouette"):
            raise ValueError(
                f"method must be 'fixed', 'elbow', or 'silhouette', got '{method}'"
            )
        if method == "fixed" and k is None:
            raise ValueError("k must be specified when method='fixed'")
        if method == "fixed" and k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if label_style not in ("int", "interval"):
            raise ValueError(
                f"label_style must be 'int' or 'interval', got '{label_style}'"
            )

        self.method      = method
        self.k           = k
        self.k_max       = k_max
        self.k_min       = k_min
        self.min_size    = min_size
        self.labels      = labels
        self.label_style = label_style

        self.bins_    : Dict[str, np.ndarray] = {}
        self.labels_  : Dict[str, list]       = {}
        self.k_       : Dict[str, int]        = {}
        self.centers_ : Dict[str, np.ndarray] = {}
        self.columns_ : Optional[List[str]]   = None

    # ================================================================
    #  PUBLIC — fit / transform / fit_transform
    # ================================================================

    def fit(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> "Categorizer":
        """
        Learn bin edges from the data.

        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            Numeric data to discretize.

        Returns
        -------
        self
        """
        X_df = self._validate_input(X)
        self.columns_ = X_df.columns.tolist()
        for col in self.columns_:
            self._fit_column(col, X_df[col].values)
        return self

    def transform(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """
        Discretize continuous data.

        Parameters
        ----------
        X : DataFrame, Series, or ndarray

        Returns
        -------
        pd.DataFrame with the same column names as the fitted data.
        """
        if self.columns_ is None:
            raise RuntimeError("Categorizer must be fitted before transform().")
        X_df = self._validate_input(X)
        if set(X_df.columns) != set(self.columns_):
            raise ValueError(
                f"Columns in X don't match fitted columns. "
                f"Expected {self.columns_}, got {X_df.columns.tolist()}"
            )
        return pd.DataFrame(
            {col: self._transform_column(col, X_df[col].values) for col in self.columns_},
            index=X_df.index,
        )

    def fit_transform(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "method":      self.method,
            "k":           self.k,
            "k_max":       self.k_max,
            "k_min":       self.k_min,
            "min_size":    self.min_size,
            "labels":      self.labels,
            "label_style": self.label_style,
        }

    def set_params(self, **params) -> "Categorizer":
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def get_bin_info(self, column: Optional[str] = None) -> Dict:
        """
        Return information about the learned bins.

        Parameters
        ----------
        column : str, optional.
            If None, returns info for all columns.
        """
        if self.columns_ is None:
            raise RuntimeError("Categorizer must be fitted first.")
        if column is not None:
            if column not in self.columns_:
                raise ValueError(f"Column '{column}' not found.")
            return {
                "k":       self.k_[column],
                "centers": self.centers_[column],
                "bins":    self.bins_[column],
                "labels":  self.labels_[column],
            }
        return {
            col: {
                "k":       self.k_[col],
                "centers": self.centers_[col],
                "bins":    self.bins_[col],
                "labels":  self.labels_[col],
            }
            for col in self.columns_
        }

    # ================================================================
    #  PRIVATE — column-level helpers
    # ================================================================

    def _fit_column(self, column: str, X: np.ndarray) -> None:
        X_clean = X[~np.isnan(X)]
        if len(X_clean) == 0:
            warnings.warn(f"Column '{column}' contains only NaN values.")
            self.k_[column]       = 0
            self.centers_[column] = np.array([])
            self.bins_[column]    = np.array([])
            self.labels_[column]  = []
            return

        X_sorted = np.sort(X_clean)

        if self.method == "fixed":
            cluster_labels, centers = categorize_fixed_k(X_sorted, self.k)
            k_opt = self.k
        elif self.method == "elbow":
            cluster_labels, centers, k_opt = categorize_elbow(
                X_sorted, self.k_max, self.k_min, self.min_size
            )
        else:  # silhouette
            cluster_labels, centers, k_opt = categorize_silhouette(
                X_sorted, self.k_max, self.k_min, self.min_size
            )

        self.k_[column]       = k_opt
        self.centers_[column] = centers
        self.bins_[column]    = self._calculate_bin_edges(
            X_sorted, cluster_labels, centers
        )
        self.labels_[column]  = self._create_labels(k_opt, self.bins_[column])

    def _transform_column(self, column: str, X: np.ndarray) -> pd.Series:
        if self.k_[column] == 0:
            return pd.Series([np.nan] * len(X))

        bins   = self.bins_[column]
        labels = self.labels_[column]

        if len(bins) <= 2:
            return pd.Series([labels[0]] * len(X))

        n_bins     = len(bins) - 1
        cut_labels = labels[:n_bins] if len(labels) >= n_bins else labels

        try:
            return pd.cut(
                X,
                bins=bins,
                labels=cut_labels,
                include_lowest=True,
                duplicates="drop",
            )
        except ValueError as e:
            warnings.warn(
                f"pd.cut failed for column '{column}': {e}. "
                "Falling back to nearest-centre assignment."
            )
            return self._assign_to_nearest_center(X, column)

    def _assign_to_nearest_center(self, X: np.ndarray, column: str) -> pd.Series:
        centers = self.centers_[column]
        labels  = self.labels_[column]
        result  = []
        for x in X:
            if np.isnan(x):
                result.append(np.nan)
            else:
                result.append(labels[int(np.argmin(np.abs(centers - x)))])
        return pd.Series(result)

    def _calculate_bin_edges(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        k = len(centers)
        if k == 1:
            return np.array([X.min(), X.max()])

        unique_labels = np.unique(labels)
        if len(unique_labels) > k:
            warnings.warn(
                f"Got {len(unique_labels)} unique labels but expected {k}. "
                "Reassigning by nearest centre."
            )
            labels        = np.array([np.argmin(np.abs(centers - x)) for x in X])
            unique_labels = np.unique(labels)

        sorted_indices = np.argsort(centers)
        sorted_centers = centers[sorted_indices]

        label_map: dict = {}
        for new_idx, old_idx in enumerate(sorted_indices):
            old_label = (
                unique_labels[old_idx] if old_idx < len(unique_labels) else old_idx
            )
            label_map[old_label] = new_idx
        for ul in unique_labels:
            if ul not in label_map:
                label_map[ul] = (
                    np.argmin(np.abs(sorted_centers - centers[ul]))
                    if ul < k else 0
                )

        try:
            sorted_labels = np.array([label_map.get(int(l), 0) for l in labels])
        except Exception as e:
            warnings.warn(f"Error mapping labels: {e}. Using modulo fallback.")
            sorted_labels = labels % k

        edges = [X.min()]
        for i in range(k - 1):
            pts_i   = X[sorted_labels == i]
            pts_i1  = X[sorted_labels == i + 1]
            if len(pts_i) > 0 and len(pts_i1) > 0:
                edges.append((pts_i.max() + pts_i1.min()) / 2)
            elif i + 1 < len(sorted_centers):
                edges.append((sorted_centers[i] + sorted_centers[i + 1]) / 2)
        edges.append(X.max())
        return np.unique(edges)

    def _create_labels(self, k: int, bins: np.ndarray) -> list:
        # Priority 1: explicit custom labels
        if self.labels is not None:
            if len(self.labels) != k:
                warnings.warn(
                    f"Number of custom labels ({len(self.labels)}) does not match "
                    f"number of bins ({k}). Using integer labels instead."
                )
                return list(range(k))
            return list(self.labels)

        # Priority 2: label_style
        if self.label_style == "interval":
            return self._interval_labels(bins)

        # Default: integer indices
        return list(range(k))

    def _interval_labels(self, bins: np.ndarray) -> list:
        """
        Build human-readable interval labels from bin edges.

        For bins = [min, e1, e2, e3, max] the four labels are:
            "<e1",  "e1-e2",  "e2-e3",  ">e3"
        """
        def fmt(v: float) -> str:
            # Show as integer when the value is whole, else 2 decimal places
            return str(int(v)) if float(v) == int(float(v)) else f"{v:.2f}"

        k      = len(bins) - 1
        edges  = bins           # bins[0]=min … bins[k]=max
        labels = []

        for i in range(k):
            if k == 1:
                # Degenerate: a single bin spanning the whole range
                labels.append(f"{fmt(edges[0])}-{fmt(edges[1])}")
            elif i == 0:
                labels.append(f"<{fmt(edges[1])}")
            elif i == k - 1:
                labels.append(f">{fmt(edges[-2])}")
            else:
                labels.append(f"{fmt(edges[i])}-{fmt(edges[i + 1])}")

        return labels

    def _validate_input(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X_df = pd.DataFrame({"X": X})
            else:
                X_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        elif isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            raise TypeError(f"X must be DataFrame, Series, or ndarray, got {type(X)}")
        non_numeric = X_df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise ValueError(
                f"All columns must be numeric. Non-numeric columns: {non_numeric.tolist()}"
            )
        return X_df

    # ================================================================
    #  REPR
    # ================================================================

    def __repr__(self) -> str:
        if self.columns_ is None:
            return (
                f"Categorizer(method='{self.method}', "
                f"label_style='{self.label_style}', not fitted)"
            )
        k_str = ", ".join(f"{col}={self.k_[col]}" for col in self.columns_)
        return (
            f"Categorizer(method='{self.method}', "
            f"label_style='{self.label_style}', "
            f"fitted on {len(self.columns_)} columns, "
            f"k={{{k_str}}})"
        )
