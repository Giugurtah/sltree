# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/_preprocessing/_backend
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
