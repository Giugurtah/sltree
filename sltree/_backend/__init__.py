# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
ctypes bindings to the compiled C backends.

Two shared libraries live here, built from ``csrc/`` by ``build_c.py``:
``libpitree`` for the predictability splitting kernels, ``libslbt`` for the
stratified latent budget analysis.  The modules in this package wrap them and
are the only place in SL-TREE that talks to C.

Internal: the public API is the estimators in the top-level package.
"""
