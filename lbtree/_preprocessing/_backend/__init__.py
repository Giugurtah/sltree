"""
lbtree/_preprocessing/_backend
================================
ctypes interface to libcategorizer (KMeans-based discretizer).
"""

from ._categorizer import (
    categorize_fixed_k,
    categorize_elbow,
    categorize_silhouette,
    get_centers,
    get_sizes,
)

__all__ = [
    "categorize_fixed_k",
    "categorize_elbow",
    "categorize_silhouette",
    "get_centers",
    "get_sizes",
]
