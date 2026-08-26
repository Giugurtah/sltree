# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
lbtree — Latent Budget Tree library
=====================================

Public API
----------
Trees
~~~~~
SCTree            Two-stage Decision Tree for categorical predictors.
SCTreeWeighted    Weighted variant of SCTree (used by AdaBoost).
SLBT              Simultaneous Latent Budget Tree for stratified data.

Ensembles
~~~~~~~~~
SCTreeForest      Random Forest of SCTree with PPI feature importance.
SLBTForest        Random Forest of SLBT with PPI feature importance.
AdaBoostForest    AdaBoost.M1 ensemble of SCTreeWeighted.
SLBTAdaBoostForest  AdaBoost.M1 ensemble of SLBTWeighted.
BINPI             Boosted Incremental Non-Parametric Imputation.

Preprocessing
~~~~~~~~~~~~~
Categorizer       KMeans-based continuous-to-categorical discretizer.

Plotting
~~~~~~~~
plot_html         Interactive HTML tree visualisation.
"""

from .sctree           import SCTree
from .sctree_weighted  import SCTreeWeighted
from .slbt             import SLBT
from .forest           import SCTreeForest
from .slbt_forest      import SLBTForest
from .adaboost         import AdaBoostForest
from .slbt_weighted    import SLBTWeighted
from .adaboost_slbt    import SLBTAdaBoostForest
from .binpi            import BINPI
from ._preprocessing   import Categorizer
from .plotting         import plot_html

__version__ = "0.1.0"

__all__ = [
    "SCTree",
    "SCTreeWeighted",
    "SLBT",
    "SCTreeForest",
    "SLBTForest",
    "AdaBoostForest",
    "SLBTWeighted",
    "SLBTAdaBoostForest",
    "BINPI",
    "Categorizer",
    "plot_html",
]
