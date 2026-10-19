# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/_backend/_slbt.py
=========================
ctypes interface for libslbt — the SLBT (Simultaneous Latent Budget Tree) C backend.

Exposed functions
-----------------
gpi(K, I, J, Fs_flat)                              → float
slba(K, KA, KB, I, J, FsNoN_flat, Fs_flat, seed, n_init, max_iter, tol)
                                                   → (pi, S, alpha, beta, diag)

The shared library (libslbt.so / .dylib / .dll) must have been
compiled from csrc/slbt/ and copied into this directory.
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
        "darwin": ["libslbt.dylib"],
        "win32":  ["slbt.dll", "libslbt.dll"],
    }
    names = candidates.get(sys.platform, ["libslbt.so"])
    for name in names:
        p = here / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"libslbt shared library not found in {here}.\n"
        "Run 'pip install .' or 'make install' inside csrc/slbt/ to compile it."
    )


_lib = ctypes.CDLL(str(_find_lib()))

c_int    = ctypes.c_int
c_uint64 = ctypes.c_uint64
c_double = ctypes.c_double
p_double = ctypes.POINTER(c_double)

# ------------------------------------------------------------------ #
#  gpi_c
# ------------------------------------------------------------------ #

_lib.gpi_c.argtypes = [c_int, c_int, c_int, p_double]
_lib.gpi_c.restype  = c_double


def gpi(K: int, I: int, J: int, Fs_flat: np.ndarray) -> float:
    """
    Stratified GPI.

    Parameters
    ----------
    K       : number of strata
    I       : number of predictor modalities
    J       : number of target classes
    Fs_flat : np.ndarray, shape (K*I*J,) — flattened stratified contingency array
    """
    return _lib.gpi_c(
        K, I, J,
        Fs_flat.ctypes.data_as(p_double),
    )


# ------------------------------------------------------------------ #
#  slba_c
# ------------------------------------------------------------------ #

_lib.slba_c.argtypes = [
    c_int, c_int, c_int,      # K, KA, KB
    c_int, c_int,             # I, J
    p_double,                 # Fs_noN  (K*I*J)
    p_double,                 # Fs      (K*I*J)
    p_double,                 # out_pi
    p_double,                 # out_S   (KA*I)
    p_double,                 # out_alpha (KA*I*2)
    p_double,                 # out_beta  (KB*J*2)
    c_uint64,                 # seed for the random initialisation
    c_int,                    # n_init
    c_int,                    # max_iter
    c_double,                 # tol
    c_int,                    # optimizer: 0 softmax, 1 projected
    p_double,                 # out_diag (SLBA_NDIAG)
]
_lib.slba_c.restype = None

_OPTIMIZERS = {"softmax": 0, "projected": 1}

#: Number of diagnostics returned by the C fit (see csrc/slbt/slbt_als.h).
SLBA_NDIAG = 7
_DIAG_KEYS = (
    "loss", "loss_worst_start", "n_iter", "converged",
    "n_converged_starts", "n_same_split_starts", "n_init",
)


def slba(
    K: int, KA: int, KB: int,
    I: int, J: int,
    FsNoN_flat: np.ndarray,
    Fs_flat: np.ndarray,
    seed: int = 0,
    n_init: int = 1,
    max_iter: int = 5000,
    tol: float = 1e-12,
    optimizer: str = "softmax",
):
    """
    Simultaneous Latent Budget Analysis split.

    Parameters
    ----------
    K, KA, KB   : homogeneity parameters
    I, J        : predictor modalities, target classes
    FsNoN_flat  : joint proportions at the node (summing to 1), shape (K*I*J,)
    Fs_flat     : table for the least-squares fit, shape (K*I*J,): the
                  observed row budgets, optionally multiplied by row weights
    seed        : 64-bit seed for the random initialisation.  Start 0 draws
                  the same matrices as the original single-start code.
    n_init      : number of random starts; the one with the lowest loss is kept
    max_iter    : iteration cap per start
    tol         : relative-decrease tolerance of the descent
    optimizer   : "softmax"   — gradient descent on the softmax parameters
                                (Eq. 16-19 of the SLBT paper);
                  "projected" — projected gradient on the simplex, same
                                objective and gradients (Eq. 14-15) without
                                the reparameterisation; reaches optima on the
                                boundary of the simplex exactly.

    Returns
    -------
    pi    : float  — Predictability Improvement tau_{Y|s,Z}
    S     : np.ndarray, shape (KA, I) — 1 = row goes left
    alpha : np.ndarray, shape (KA, I, 2)
    beta  : np.ndarray, shape (KB, J, 2)
    diag  : dict — convergence diagnostics of the fit
    """
    out_pi    = np.zeros(1,          dtype=np.float64)
    out_S     = np.zeros(KA * I,     dtype=np.float64)
    out_alpha = np.zeros(KA * I * 2, dtype=np.float64)
    out_beta  = np.zeros(KB * J * 2, dtype=np.float64)
    out_diag  = np.zeros(SLBA_NDIAG, dtype=np.float64)

    FsNoN_flat = np.ascontiguousarray(FsNoN_flat, dtype=np.float64)
    Fs_flat    = np.ascontiguousarray(Fs_flat,    dtype=np.float64)

    _lib.slba_c(
        K, KA, KB, I, J,
        FsNoN_flat.ctypes.data_as(p_double),
        Fs_flat.ctypes.data_as(p_double),
        out_pi.ctypes.data_as(p_double),
        out_S.ctypes.data_as(p_double),
        out_alpha.ctypes.data_as(p_double),
        out_beta.ctypes.data_as(p_double),
        c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
        int(n_init), int(max_iter), float(tol),
        _OPTIMIZERS[optimizer],
        out_diag.ctypes.data_as(p_double),
    )

    diag = dict(zip(_DIAG_KEYS, out_diag.tolist()))
    for key in ("n_iter", "converged", "n_converged_starts",
                "n_same_split_starts", "n_init"):
        diag[key] = int(diag[key])

    return (
        float(out_pi[0]),
        out_S.reshape(KA, I),
        out_alpha.reshape(KA, I, 2),
        out_beta.reshape(KB, J, 2),
        diag,
    )
