# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
The four homogeneity settings of SLBT.

A stratified split is governed by two sets of parameters: A, the mixing
parameters that decide the split direction, and B, the latent budget profiles
the branches point to.  Either may be shared across strata or left free, which
gives four combinations:

===============  ========  ========  =========================================
setting          A         B         can represent a reversal between strata?
===============  ========  ========  =========================================
``none``         free      free      yes
``A``            shared    free      no
``B``            free      shared    yes
``AB``           shared    shared    no
===============  ========  ========  =========================================

The last column is the one that matters in practice: only a free A lets a
modality go left in one stratum and right in another.

Internal: selected through the ``homogeneity`` argument of :class:`~sltree.SLBT`.
"""
