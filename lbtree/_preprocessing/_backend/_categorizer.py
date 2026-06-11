"""
lbtree/_preprocessing/_backend/_categorizer.py
===============================================
ctypes interface for libcategorizer — the KMeans discretizer C backend.

Exposed functions
-----------------
categorize_fixed_k(X, k)                          → (labels, centers)
categorize_elbow(X, k_max, k_min, min_size)        → (labels, centers, k_opt)
categorize_silhouette(X, k_max, k_min, min_size)   → (labels, centers, k_opt)
get_centers(X, labels, k)                          → centers
get_sizes(labels, k)                               → sizes

The shared library (libcategorizer.so / .dylib / .dll) must have been
compiled from csrc/categorizer/ and copied into this directory.
"""

from __future__ import annotations

import sys
import ctypes
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ #
#  Locate the shared library
# ------------------------------------------------------------------ #

def _find_lib() -> Path:
    here       = Path(__file__).parent
    candidates = {
        "darwin": ["libcategorizer.dylib"],
        "win32":  ["categorizer.dll", "libcategorizer.dll"],
    }
    names = candidates.get(sys.platform, ["libcategorizer.so"])
    for name in names:
        p = here / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"libcategorizer shared library not found in {here}.\n"
        "Run 'pip install .' or 'make install' inside csrc/categorizer/ to compile it."
    )


_lib = ctypes.CDLL(str(_find_lib()))

c_int    = ctypes.c_int
c_double = ctypes.c_double
p_int    = ctypes.POINTER(c_int)
p_double = ctypes.POINTER(c_double)

# ------------------------------------------------------------------ #
#  Function signatures
# ------------------------------------------------------------------ #

_lib.categorize_kmeans.argtypes = [c_int, c_int, p_double, p_int]
_lib.categorize_kmeans.restype  = None

_lib.categorize_kmeans_elbow.argtypes = [c_int, c_int, c_int, c_int, p_double, p_int]
_lib.categorize_kmeans_elbow.restype  = c_int

_lib.categorize_kmeans_silhouette.argtypes = [c_int, c_int, c_int, c_int, p_double, p_int]
_lib.categorize_kmeans_silhouette.restype  = c_int

_lib.get_cluster_centers.argtypes = [c_int, c_int, p_double, p_int, p_double]
_lib.get_cluster_centers.restype  = None

_lib.get_cluster_sizes.argtypes = [c_int, c_int, p_int, p_int]
_lib.get_cluster_sizes.restype  = None


# ------------------------------------------------------------------ #
#  Python wrappers
# ------------------------------------------------------------------ #

def categorize_fixed_k(X: np.ndarray, k: int):
    """
    Categorise data with a fixed number of KMeans clusters.

    Parameters
    ----------
    X : np.ndarray  — 1-D array, sorted
    k : int         — number of clusters

    Returns
    -------
    labels  : np.ndarray (int32), shape (len(X),)
    centers : np.ndarray (float64), shape (k,)
    """
    X      = np.ascontiguousarray(X, dtype=np.float64)
    labels = np.zeros(len(X), dtype=np.int32)
    _lib.categorize_kmeans(
        len(X), k,
        X.ctypes.data_as(p_double),
        labels.ctypes.data_as(p_int),
    )
    centers = get_centers(X, labels, k)
    return labels, centers


def categorize_elbow(
    X: np.ndarray,
    k_max: int   = 10,
    k_min: int   = 2,
    min_size: int = 1,
):
    """
    Categorise data using automatic K selection via the elbow method.

    Returns
    -------
    labels    : np.ndarray (int32)
    centers   : np.ndarray (float64)
    k_optimal : int
    """
    X      = np.ascontiguousarray(X, dtype=np.float64)
    labels = np.zeros(len(X), dtype=np.int32)
    k_opt  = _lib.categorize_kmeans_elbow(
        len(X), k_max, k_min, min_size,
        X.ctypes.data_as(p_double),
        labels.ctypes.data_as(p_int),
    )
    centers = get_centers(X, labels, k_opt)
    return labels, centers, k_opt


def categorize_silhouette(
    X: np.ndarray,
    k_max: int   = 10,
    k_min: int   = 2,
    min_size: int = 1,
):
    """
    Categorise data using automatic K selection via the silhouette method.

    Returns
    -------
    labels    : np.ndarray (int32)
    centers   : np.ndarray (float64)
    k_optimal : int
    """
    X      = np.ascontiguousarray(X, dtype=np.float64)
    labels = np.zeros(len(X), dtype=np.int32)
    k_opt  = _lib.categorize_kmeans_silhouette(
        len(X), k_max, k_min, min_size,
        X.ctypes.data_as(p_double),
        labels.ctypes.data_as(p_int),
    )
    centers = get_centers(X, labels, k_opt)
    return labels, centers, k_opt


def get_centers(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Return the k cluster centres for a given labelling."""
    X       = np.ascontiguousarray(X,      dtype=np.float64)
    labels  = np.ascontiguousarray(labels, dtype=np.int32)
    centers = np.zeros(k, dtype=np.float64)
    _lib.get_cluster_centers(
        len(X), k,
        X.ctypes.data_as(p_double),
        labels.ctypes.data_as(p_int),
        centers.ctypes.data_as(p_double),
    )
    return centers


def get_sizes(labels: np.ndarray, k: int) -> np.ndarray:
    """Return the k cluster sizes for a given labelling."""
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    sizes  = np.zeros(k, dtype=np.int32)
    _lib.get_cluster_sizes(
        len(labels), k,
        labels.ctypes.data_as(p_int),
        sizes.ctypes.data_as(p_int),
    )
    return sizes
