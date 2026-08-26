# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree — Latent Budget Tree library
=====================================

Public API
----------
Trees
~~~~~
PredictabilityTree            Two-stage Decision Tree for categorical predictors.
PredictabilityTreeWeighted    Weighted variant of PredictabilityTree (used by AdaBoost).
SLBT              Simultaneous Latent Budget Tree for stratified data.

Ensembles
~~~~~~~~~
PredictabilityForest      Random Forest of PredictabilityTree with PPI feature importance.
SLBTForest        Random Forest of SLBT with PPI feature importance.
PredictabilityAdaBoost    AdaBoost.M1 ensemble of PredictabilityTreeWeighted.
SLBTAdaBoostForest  AdaBoost.M1 ensemble of SLBTWeighted.
BINPI             Boosted Incremental Non-Parametric Imputation.

Preprocessing
~~~~~~~~~~~~~
Categorizer       KMeans-based continuous-to-categorical discretizer.

Plotting
~~~~~~~~
plot_html         Interactive HTML tree visualisation.
"""

from .predictability_tree           import PredictabilityTree
from .predictability_tree_weighted  import PredictabilityTreeWeighted
from .slbt             import SLBT
from .predictability_forest           import PredictabilityForest
from .slbt_forest      import SLBTForest
from .predictability_adaboost         import PredictabilityAdaBoost
from .slbt_weighted    import SLBTWeighted
from .adaboost_slbt    import SLBTAdaBoostForest
from .binpi            import BINPI
from ._preprocessing   import Categorizer
from .plotting         import plot_html

__version__ = "0.1.0"

__all__ = [
    "PredictabilityTree",
    "PredictabilityTreeWeighted",
    "SLBT",
    "PredictabilityForest",
    "SLBTForest",
    "PredictabilityAdaBoost",
    "SLBTWeighted",
    "SLBTAdaBoostForest",
    "BINPI",
    "Categorizer",
    "plot_html",
]
