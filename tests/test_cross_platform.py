"""
Cross-platform agreement.

The claim these tests defend is the one a replication script rests on: the same
data with the same seed gives the same answer on macOS, Linux and Windows.  It
is not free.  Two things could break it.

The random stream.  The C backends carry an explicit splitmix64 generator
(csrc/common/lbrandom.h) rather than libc rand(), precisely because rand() is
implementation-defined and differs between glibc, musl and the Windows CRT.
splitmix64 is specified bit for bit, so the same seed must give the same
sequence everywhere.

The floating-point arithmetic.  IEEE 754 fixes each individual operation, but
not the order a compiler chooses to perform them in.  Fusing a*b + c into a
single fused-multiply-add rounds once instead of twice and lands on a different
last bit; auto-vectorisation reorders sums with the same effect.  build_c.py
passes -ffp-contract=off on every platform to shut the first door.  A last-bit
difference matters here more than in most code, because a flipped
`pi > best_pi` comparison changes every split beneath it.

The snapshots in tests/data/snapshots.json were generated once, on Linux with
gcc.  If a platform disagrees, one of the two mechanisms above has failed, and
the failure message says which node diverged.
"""

import contextlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sltree import PredictabilityTree, SLBT

SNAPSHOTS = json.loads(
    (Path(__file__).parent / "data" / "snapshots.json").read_text(encoding="utf-8")
)


def _quiet():
    return contextlib.redirect_stdout(open(os.devnull, "w"))


def _structure(tree):
    """The same serialisation the snapshots were generated with."""
    out = []

    def walk(node):
        if node is None:
            return
        out.append({
            "position": int(node.position),
            "feature": None if node.feature is None else str(node.feature),
            "N": int(node.N),
            "impurity": round(float(node.impurity), 12),
            "impurity_decrease": round(float(node.impurity_decrease), 12),
            "pi": None if node.pi is None else round(float(node.pi), 12),
            "gpi": None if node.gpi is None else round(float(node.gpi), 12),
        })
        walk(node.left)
        walk(node.right)

    walk(tree.root)
    return out


def _fixture_data():
    """Regenerates exactly the data the snapshots were built on."""
    rng = np.random.default_rng(20250827)
    n = 240
    Z = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    Xc = pd.DataFrame({
        c: pd.cut(Z[c], [-9, -0.7, 0, 0.7, 9], labels=list("abcd")).astype(str)
        for c in Z
    })
    y = pd.Series(np.where(Z.x0 - Z.x1 + rng.normal(0, 0.4, n) > 0, "A", "B"))
    strata = rng.integers(0, 2, n)
    return Xc, y, strata


def _report(where, expected, actual):
    """A difference report that names the first node that diverged."""
    if len(expected) != len(actual):
        return (f"{where}: {len(expected)} nodes expected, {len(actual)} produced "
                f"on {platform.system()} {platform.machine()}")
    for exp, act in zip(expected, actual):
        for key in exp:
            if exp[key] != act[key]:
                return (f"{where}: node {exp['position']}, field '{key}': "
                        f"expected {exp[key]!r}, got {act[key]!r} "
                        f"on {platform.system()} {platform.machine()}")
    return ""


# ------------------------------------------------------------------ #
#  The C random stream
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("seed", ["0", "1", "42"])
def test_the_c_generator_gives_the_same_stream_everywhere(seed):
    """A known-answer test that runs through the real C path.

    categorize_kmeans() seeds a local splitmix64 and hands it to the k-means++
    initialisation, whose first centroid is X[lb_rng_below(rng, I)].  Feeding it
    twenty evenly spaced values makes the resulting labelling a direct function
    of the draws, so a divergence in the generator shows up as a different
    labelling rather than being smoothed away.
    """
    from sltree._preprocessing._backend._categorizer import categorize_fixed_k

    X = np.arange(20, dtype=np.float64)
    with _quiet():
        labels, _ = categorize_fixed_k(X, 4, int(seed))

    expected = SNAPSHOTS["kmeans_labels"][seed]
    assert labels.tolist() == expected, (
        f"seed {seed}: the C generator produced a different stream on "
        f"{platform.system()} {platform.machine()}.\n"
        f"  expected {expected}\n  got      {labels.tolist()}"
    )


def test_different_seeds_reach_the_generator():
    """Guards the test above.

    If the seed were dropped somewhere between Python and C, every seed would
    give the same labelling and the known-answer test would still pass.
    """
    from sltree._preprocessing._backend._categorizer import categorize_fixed_k

    X = np.arange(20, dtype=np.float64)
    labellings = set()
    for seed in range(8):
        with _quiet():
            labels, _ = categorize_fixed_k(X, 4, seed)
        labellings.add(tuple(labels.tolist()))
    assert len(labellings) > 1


# ------------------------------------------------------------------ #
#  Whole-tree snapshots
# ------------------------------------------------------------------ #

def test_predictability_tree_matches_the_stored_structure():
    """Every node: feature, size, impurity, V, PI and GPI.

    This is the strongest statement the suite makes.  It covers the splitting
    kernel, the impurity arithmetic and the traversal order at once — the same
    tree, to twelve decimal places, on any platform.
    """
    Xc, y, _ = _fixture_data()
    tree = PredictabilityTree(max_depth=5, feats_viewed=4)
    tree.fit(Xc, y)

    difference = _report("PredictabilityTree",
                         SNAPSHOTS["predictability_tree"], _structure(tree))
    assert not difference, difference


def test_slbt_matches_the_stored_structure():
    """The same, through the ALS.

    Harder than the tree above: SLBT runs an iterative alternating least squares
    from a random start, so agreement here means both the seeded initialisation
    and the iteration itself land on the same values across platforms.
    """
    Xc, y, strata = _fixture_data()
    with _quiet():
        model = SLBT(homogeneity="none", max_depth=4, feats_viewed=4,
                     random_state=20250827)
        model.fit(Xc, y, strata)

    difference = _report("SLBT(none)", SNAPSHOTS["slbt_none"], _structure(model))
    assert not difference, difference


def test_the_snapshots_are_not_trivially_satisfiable():
    """A tree grown from different data must not match the snapshot.

    Without this, a serialisation bug that emptied every field would let the
    comparison pass.
    """
    rng = np.random.default_rng(999)
    n = 240
    Z = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    Xc = pd.DataFrame({
        c: pd.cut(Z[c], [-9, -0.7, 0, 0.7, 9], labels=list("abcd")).astype(str)
        for c in Z
    })
    y = pd.Series(rng.choice(["A", "B"], size=n))

    tree = PredictabilityTree(max_depth=5, feats_viewed=4)
    tree.fit(Xc, y)

    assert _structure(tree) != SNAPSHOTS["predictability_tree"]


# ------------------------------------------------------------------ #
#  Environment
# ------------------------------------------------------------------ #

def test_the_shared_libraries_carry_the_expected_extension():
    """A wheel built for the wrong platform, or a stale library left behind by
    an earlier build, shows up here rather than as an obscure crash later.
    """
    import sltree
    import sys

    expected = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    backend = Path(sltree.__file__).parent / "_backend"
    found = [p.suffix for p in backend.iterdir()
             if p.suffix in {".so", ".dylib", ".dll"}]

    assert found, f"no compiled library in {backend}"
    assert set(found) == {expected}, \
        f"expected only {expected} on {sys.platform}, found {sorted(set(found))}"
