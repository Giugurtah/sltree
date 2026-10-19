# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
Memory hygiene of the C backends (rOpenSci G5.9, extended).

Every reproducibility test in this suite until now compared two fits inside
one process.  That is not enough for a library whose numerical core is C: a
routine that reads uninitialised memory returns the same answer twice in a
row, because the stack is in the same state both times.  It differs only when
something else ran first — another test, another platform, another allocation
pattern.

Exactly that happened.  slbt_als.c declared DX as I x I and zeroed it with a
loop bounded by J, the number of response classes.  Where J < I the columns
past J kept whatever the stack held and the fitted tree depended on it; where
J > I the same line wrote past the row.  Two runs of the same fit in separate
processes disagreed, the suite passed anyway, and a benchmark ran on numbers
that could not be repeated.

An honest warning about what is and is not testable from here.  **Python
tests alone cannot reliably detect this class of bug.**  On Linux the kernel
hands out zeroed stack pages, so the garbage read is usually zero, which is
the correct value, and everything below passes even against the broken
library — verified.  The reliable detector is a compiler one:

    python build_c.py --init-pattern && pytest

which fills uninitialised automatic variables with a non-zero pattern.  The
snapshot test in test_cross_platform.py then fails loudly on the broken
library (23 nodes against the 29 recorded) and passes on the fixed one.  That
command belongs in CI, and finding this bug is what it is for.

What the two tests here add is cheaper and partial, and worth having anyway.
The first catches the fault on platforms that do not zero the stack — it is
how the bug surfaced on macOS.  The second sweeps the shape of the
contingency table on both sides of I == J, which is where the two failure
modes live; run under AddressSanitizer it is where the out-of-bounds write
appears.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from sltree import SLBT


# ------------------------------------------------------------------ #
#  Same fit, separate processes
# ------------------------------------------------------------------ #

_CHILD = textwrap.dedent("""
    import contextlib, json, os, sys, warnings
    warnings.simplefilter("ignore")
    import numpy as np, pandas as pd
    from sltree import SLBT

    rng = np.random.default_rng(20250827)
    n = 180
    X = pd.DataFrame({f"v{k}": rng.choice(list("abcd"), size=n) for k in range(3)})
    y = pd.Series(rng.choice(list("AB"), size=n))
    s = rng.integers(0, 2, n)

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        m = SLBT(homogeneity="none", max_depth=3, feats_viewed=3, random_state=17)
        m.fit(X, y, s)

    out = []
    def walk(nd):
        if nd is None:
            return
        out.append([int(nd.position), str(nd.feature), int(nd.N),
                    round(float(nd.impurity), 12),
                    None if nd.pi is None else round(float(nd.pi), 12)])
        walk(nd.left); walk(nd.right)
    walk(m.root)
    print(json.dumps(out))
""")


def _fit_in_a_new_process():
    r = subprocess.run([sys.executable, "-c", _CHILD],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"il processo figlio e' fallito:\n{r.stderr[-2000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_slbt_gives_the_same_tree_in_separate_processes():
    """The same seed and the same data, fitted in four fresh interpreters.

    Passing is necessary, not sufficient: see the module docstring.  On a
    platform that zeroes fresh stack pages this agrees even when the C code
    is reading memory it never wrote.  On one that does not — macOS — it is
    what caught the DX bug.
    """
    runs = [_fit_in_a_new_process() for _ in range(4)]
    first = json.dumps(runs[0], sort_keys=True)

    for i, r in enumerate(runs[1:], start=1):
        if json.dumps(r, sort_keys=True) != first:
            diff = next(
                (f"nodo {a[0]}: {a} != {b}" for a, b in zip(runs[0], r) if a != b),
                f"{len(runs[0])} nodi contro {len(r)}",
            )
            pytest.fail(f"il processo {i} ha prodotto un albero diverso — {diff}")


# ------------------------------------------------------------------ #
#  Both sides of I == J
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("n_modalities,n_classes", [
    (2, 2),   # I == J
    (2, 3),   # J > I  — la scrittura fuori dai limiti
    (2, 5),   # J > I, largamente
    (4, 2),   # J < I  — le colonne non inizializzate
    (5, 2),
    (3, 4),
    (4, 4),
])
@pytest.mark.parametrize("homogeneity", ["none", "AB"])
def test_slbt_over_every_table_shape(n_modalities, n_classes, homogeneity):
    """I is the number of predictor modalities, J the number of response
    classes.  The two enter the C core as separate dimensions, and code that
    confuses them only misbehaves when they differ — so both orderings, and
    the equal case, have to be exercised.
    """
    import contextlib
    import os

    rng = np.random.default_rng(n_modalities * 100 + n_classes)
    n = 200
    mods = [chr(97 + i) for i in range(n_modalities)]
    cls = [chr(65 + i) for i in range(n_classes)]

    X = pd.DataFrame({f"v{k}": rng.choice(mods, size=n) for k in range(3)})
    y = pd.Series(rng.choice(cls, size=n))
    s = rng.integers(0, 2, n)

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        a = SLBT(homogeneity=homogeneity, max_depth=3, feats_viewed=3,
                 random_state=5)
        a.fit(X, y, s)
        b = SLBT(homogeneity=homogeneity, max_depth=3, feats_viewed=3,
                 random_state=5)
        b.fit(X, y, s)

    assert a.root.N == len(y)
    assert 0.0 <= a.root.impurity <= 1.0
    if a.root.pi is not None:
        assert 0.0 <= a.root.pi <= 1.0 + 1e-9
    # e la stessa configurazione due volte deve dare lo stesso albero
    assert a.root.feature == b.root.feature
    assert a.root.pi == b.root.pi
