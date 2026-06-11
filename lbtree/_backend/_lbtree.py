"""
lbtree/_backend/_lbtree.py
===========================
ctypes interface for liblbtree — the SCTree / AdaBoost C backend.

Exposed functions
-----------------
gpi(I, J, F_flat)                          → float
twoStage(I, J, F_flat)                     → (pi, S)
twoing(I, J, F_flat)                       → (pi, S)
gpi_weighted(I, J, W, F_flat)              → float
twoStage_weighted(I, J, W, F_flat)         → (pi, S)
twoing_weighted(I, J, W, F_flat)           → (pi, S)
binarize_target(N_unique, y_sorted)        → float  [twoClass]

The shared library (liblbtree.so / .dylib / .dll) must have been
compiled from csrc/lbtree/lbtree_core.c and copied into this
directory before import.  The build step is handled by setup.py.
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
    here = Path(__file__).parent
    candidates = {
        "darwin": ["liblbtree.dylib"],
        "win32":  ["lbtree.dll", "liblbtree.dll"],
    }
    names = candidates.get(sys.platform, ["liblbtree.so"])
    for name in names:
        p = here / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"liblbtree shared library not found in {here}.\n"
        "Run 'pip install .' or 'make install' inside csrc/lbtree/ to compile it."
    )


_lib = ctypes.CDLL(str(_find_lib()))

c_int    = ctypes.c_int
c_double = ctypes.c_double
p_double = ctypes.POINTER(c_double)

# ------------------------------------------------------------------ #
#  gpi_c
# ------------------------------------------------------------------ #

_lib.gpi_c.argtypes = [c_int, c_int, p_double]
_lib.gpi_c.restype  = c_double


def gpi(I: int, J: int, F_flat: np.ndarray) -> float:
    """Return the GPI (Global Predictability Index) for a single feature."""
    return _lib.gpi_c(
        I, J,
        F_flat.ctypes.data_as(p_double),
    )


# ------------------------------------------------------------------ #
#  twoStage_c
# ------------------------------------------------------------------ #

_lib.twoStage_c.argtypes = [c_int, c_int, p_double, p_double, p_double]
_lib.twoStage_c.restype  = None


def twoStage(I: int, J: int, F_flat: np.ndarray):
    """
    Two-stage split: find the optimal bipartition of predictor modalities.

    Returns
    -------
    pi : float            — best PPI
    S  : np.ndarray (I,) — binary left-mask
    """
    out_pi = np.zeros(1,  dtype=np.float64)
    out_S  = np.zeros(I,  dtype=np.float64)
    _lib.twoStage_c(
        I, J,
        F_flat.ctypes.data_as(p_double),
        out_pi.ctypes.data_as(p_double),
        out_S.ctypes.data_as(p_double),
    )
    return float(out_pi[0]), out_S


# ------------------------------------------------------------------ #
#  twoing_c
# ------------------------------------------------------------------ #

_lib.twoing_c.argtypes = [c_int, c_int, p_double, p_double, p_double]
_lib.twoing_c.restype  = None


def twoing(I: int, J: int, F_flat: np.ndarray):
    """
    Twoing split: exhaustive row×column bipartition.

    Returns
    -------
    pi : float
    S  : np.ndarray (I,)
    """
    out_pi = np.zeros(1, dtype=np.float64)
    out_S  = np.zeros(I, dtype=np.float64)
    _lib.twoing_c(
        I, J,
        F_flat.ctypes.data_as(p_double),
        out_pi.ctypes.data_as(p_double),
        out_S.ctypes.data_as(p_double),
    )
    return float(out_pi[0]), out_S


# ------------------------------------------------------------------ #
#  gpi_weighted_c
# ------------------------------------------------------------------ #

_lib.gpi_weighted_c.argtypes = [c_int, c_int, p_double, p_double]
_lib.gpi_weighted_c.restype  = c_double


def gpi_weighted(I: int, J: int, W: np.ndarray, F_flat: np.ndarray) -> float:
    """Weighted GPI for AdaBoost."""
    W_arr = np.asarray(W,      dtype=np.float64)
    F_arr = np.asarray(F_flat, dtype=np.float64)
    return _lib.gpi_weighted_c(
        I, J,
        W_arr.ctypes.data_as(p_double),
        F_arr.ctypes.data_as(p_double),
    )


# ------------------------------------------------------------------ #
#  twoStage_weighted_c
# ------------------------------------------------------------------ #

_lib.twoStage_weighted_c.argtypes = [c_int, c_int, p_double, p_double, p_double, p_double]
_lib.twoStage_weighted_c.restype  = None


def twoStage_weighted(I: int, J: int, W: np.ndarray, F_flat: np.ndarray):
    """Weighted two-stage split (AdaBoost)."""
    W_arr  = np.asarray(W,      dtype=np.float64)
    F_arr  = np.asarray(F_flat, dtype=np.float64)
    out_pi = ctypes.c_double()
    out_S  = (ctypes.c_double * I)()
    _lib.twoStage_weighted_c(
        I, J,
        W_arr.ctypes.data_as(p_double),
        F_arr.ctypes.data_as(p_double),
        ctypes.byref(out_pi),
        out_S,
    )
    return out_pi.value, np.array([out_S[i] for i in range(I)])


# ------------------------------------------------------------------ #
#  twoing_weighted_c
# ------------------------------------------------------------------ #

_lib.twoing_weighted_c.argtypes = [c_int, c_int, p_double, p_double, p_double, p_double]
_lib.twoing_weighted_c.restype  = None


def twoing_weighted(I: int, J: int, W: np.ndarray, F_flat: np.ndarray):
    """Weighted twoing split (AdaBoost)."""
    W_arr  = np.asarray(W,      dtype=np.float64)
    F_arr  = np.asarray(F_flat, dtype=np.float64)
    out_pi = ctypes.c_double()
    out_S  = (ctypes.c_double * I)()
    _lib.twoing_weighted_c(
        I, J,
        W_arr.ctypes.data_as(p_double),
        F_arr.ctypes.data_as(p_double),
        ctypes.byref(out_pi),
        out_S,
    )
    return out_pi.value, np.array([out_S[i] for i in range(I)])


# ------------------------------------------------------------------ #
#  binarize_target_c  — twoClass support
# ------------------------------------------------------------------ #

_lib.binarize_target_c.argtypes = [c_int, p_double, p_double]
_lib.binarize_target_c.restype  = None


def binarize_target(N_unique: int, y_sorted: np.ndarray) -> float:
    """
    Find the optimal split threshold T for a numeric target (twoClass model).

    Values strictly less than T are labelled 'low';
    values >= T are labelled 'high'.

    Parameters
    ----------
    N_unique : int
        Number of distinct sorted values in y_sorted.
    y_sorted : np.ndarray, shape (N_unique,)
        Sorted distinct values of the numeric target (ascending).

    Returns
    -------
    threshold : float
        The split point T chosen by ``binarize_target_c``.
        NOTE: the C function currently returns a placeholder (median).
    """
    y_arr         = np.asarray(y_sorted, dtype=np.float64)
    out_threshold = np.zeros(1, dtype=np.float64)
    _lib.binarize_target_c(
        N_unique,
        y_arr.ctypes.data_as(p_double),
        out_threshold.ctypes.data_as(p_double),
    )
    return float(out_threshold[0])
