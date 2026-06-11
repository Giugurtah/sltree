"""
lbtree/_backend/_slbt.py
=========================
ctypes interface for libslbt — the SLBT (Simultaneous Latent Budget Tree) C backend.

Exposed functions
-----------------
gpi(K, I, J, Fs_flat)                              → float
slba(K, KA, KB, I, J, FsNoN_flat, Fs_flat)         → (pi, S, alpha, beta)

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
]
_lib.slba_c.restype = None


def slba(
    K: int, KA: int, KB: int,
    I: int, J: int,
    FsNoN_flat: np.ndarray,
    Fs_flat: np.ndarray,
):
    """
    Simultaneous Latent Budget Analysis split.

    Parameters
    ----------
    K, KA, KB   : homogeneity parameters
    I, J        : predictor modalities, target classes
    FsNoN_flat  : raw joint frequencies (not normalised), shape (K*I*J,)
    Fs_flat     : row-conditional stratified array, shape (K*I*J,)

    Returns
    -------
    pi    : float
    S     : np.ndarray, shape (KA, I)
    alpha : np.ndarray, shape (KA, I, 2)
    beta  : np.ndarray, shape (KB, J, 2)
    """
    out_pi    = np.zeros(1,         dtype=np.float64)
    out_S     = np.zeros(KA * I,    dtype=np.float64)
    out_alpha = np.zeros(KA * I * 2, dtype=np.float64)
    out_beta  = np.zeros(KB * J * 2, dtype=np.float64)

    _lib.slba_c(
        K, KA, KB, I, J,
        FsNoN_flat.ctypes.data_as(p_double),
        Fs_flat.ctypes.data_as(p_double),
        out_pi.ctypes.data_as(p_double),
        out_S.ctypes.data_as(p_double),
        out_alpha.ctypes.data_as(p_double),
        out_beta.ctypes.data_as(p_double),
    )

    return (
        float(out_pi[0]),
        out_S.reshape(KA, I),
        out_alpha.reshape(KA, I, 2),
        out_beta.reshape(KB, J, 2),
    )
