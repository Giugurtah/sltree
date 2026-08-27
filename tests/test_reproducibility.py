"""
Reproducibility (rOpenSci G5.9: expected stochastic behaviour).

Two random sources sit under the library, both in C: the k-means++ centroid
initialisation behind the Categorizer, and the ALS initialisation of U and Z
inside SLBT.  Both used to run on libc rand(), whose state is global to the
process — so results depended on the clock, could not be seeded, and one
component silently perturbed the other.

These tests defend the fix.  They are not decoration: an unseeded tree
algorithm amplifies tiny differences, because a single flipped `pi > best_pi`
comparison changes every split below it.
"""

import contextlib
import os

import numpy as np
import pytest

from sltree import (Categorizer, PredictabilityForest, SLBT,
                    SLBTAdaBoostForest, SLBTForest, SLBTWeighted)


def _quiet():
    return contextlib.redirect_stdout(open(os.devnull, "w"))


def _signature(tree):
    """A structural fingerprint: position, feature and PI of every node.

    Comparing predictions alone is too weak — two different trees can agree on
    a small sample.  This compares the structure itself.
    """
    out = []

    def walk(node):
        if node is None:
            return
        out.append((node.position, node.feature,
                    None if node.pi is None else round(float(node.pi), 12)))
        walk(node.left)
        walk(node.right)

    walk(tree.root)
    return tuple(out)


# ------------------------------------------------------------------ #
#  A fixed seed must give a fixed answer
# ------------------------------------------------------------------ #

def test_slbt_is_reproducible(X_small, y_small, strata_small):
    """Ten fits, one tree.

    Before the ALS was seeded this produced ten different trees, with accuracy
    spread over five percentage points and leaf counts from 21 to 31.
    """
    signatures = set()
    for _ in range(10):
        model = SLBT(max_depth=3, feats_viewed=2, random_state=0)
        model.fit(X_small, y_small, strata_small)
        signatures.add(_signature(model))
    assert len(signatures) == 1


def test_slbt_weighted_is_reproducible(X_small, y_small, strata_small, weights_small):
    signatures = set()
    for _ in range(5):
        model = SLBTWeighted(max_depth=3, feats_viewed=2, random_state=0)
        model.fit(X_small, y_small, strata_small, weights_small)
        signatures.add(_signature(model))
    assert len(signatures) == 1


@pytest.mark.parametrize("sampling", ["tree", "node"])
def test_slbt_forest_is_reproducible(X_small, y_small, strata_small, sampling):
    """Both feature-sampling branches, because they build trees differently."""
    kwargs = ({"feature_sampling": "tree"} if sampling == "tree"
              else {"node_max_features": 2})
    preds = set()
    for _ in range(5):
        forest = SLBTForest(n_estimators=3, max_depth=2, feats_viewed=2,
                            random_state=0, **kwargs)
        forest.fit(X_small, y_small, strata_small)
        preds.add(tuple(forest.predict(X_small)))
    assert len(preds) == 1


def test_slbt_adaboost_is_reproducible(X_small, y_small, strata_small):
    preds = set()
    for _ in range(4):
        boost = SLBTAdaBoostForest(n_estimators=3, max_depth=2, feats_viewed=2,
                                   random_state=0)
        boost.fit(X_small, y_small, strata_small)
        preds.add(tuple(boost.predict(X_small)))
    assert len(preds) == 1


@pytest.mark.parametrize("method,kwargs", [
    ("fixed",      {"k": 4}),
    ("elbow",      {"k_max": 6}),
    ("silhouette", {"k_max": 5}),
])
def test_categorizer_is_reproducible(numeric, method, kwargs):
    """All three K-selection methods.

    Unseeded, three runs agreed on 99.8% of cells — and that 0.2% was enough
    to change which feature the tree chose at the root.
    """
    outputs = []
    for _ in range(5):
        with _quiet():
            cat = Categorizer(method=method, random_state=0, **kwargs)
            outputs.append(cat.fit_transform(numeric))
    first = outputs[0].values
    assert all((first == out.values).all() for out in outputs[1:])


# ------------------------------------------------------------------ #
#  A different seed must give a different answer
# ------------------------------------------------------------------ #
#
# Without this, a function that ignored random_state entirely would pass every
# test above.

def test_different_seeds_give_different_slbt_trees(X_small, y_small, strata_small):
    a = SLBT(max_depth=4, feats_viewed=3, random_state=0)
    b = SLBT(max_depth=4, feats_viewed=3, random_state=1)
    a.fit(X_small, y_small, strata_small)
    b.fit(X_small, y_small, strata_small)
    assert _signature(a) != _signature(b)


def test_random_state_none_is_not_reproducible(X_small, y_small, strata_small):
    """The scikit-learn convention: None draws fresh entropy every fit.

    Deliberately asserted, so that a future change of default is caught here
    rather than silently altering everyone's results.
    """
    signatures = {_signature(SLBT(max_depth=4, feats_viewed=3).fit(X_small, y_small, strata_small))
                  for _ in range(5)}
    assert len(signatures) > 1


# ------------------------------------------------------------------ #
#  The components must not perturb each other
# ------------------------------------------------------------------ #

def test_categorizer_does_not_disturb_slbt(X_small, y_small, strata_small, numeric):
    """libcategorizer and libslbt are separate libraries but shared libc's
    global rand() state, so fitting one moved the other's results.  Each now
    carries its own generator; this proves the coupling is gone.
    """
    first = SLBT(max_depth=3, feats_viewed=2, random_state=5)
    first.fit(X_small, y_small, strata_small)

    with _quiet():                                    # noise in between
        Categorizer(method="elbow", k_max=6, random_state=123).fit_transform(numeric)

    second = SLBT(max_depth=3, feats_viewed=2, random_state=5)
    second.fit(X_small, y_small, strata_small)

    assert _signature(first) == _signature(second)


def test_slbt_does_not_disturb_categorizer(X_small, y_small, strata_small, numeric):
    """The same coupling, checked in the other direction."""
    with _quiet():
        first = Categorizer(method="fixed", k=4, random_state=7).fit_transform(numeric)

    SLBT(max_depth=3, feats_viewed=2, random_state=99).fit(X_small, y_small, strata_small)

    with _quiet():
        second = Categorizer(method="fixed", k=4, random_state=7).fit_transform(numeric)

    assert (first.values == second.values).all()


# ------------------------------------------------------------------ #
#  The generator itself
# ------------------------------------------------------------------ #

def test_seeds_are_drawn_deterministically(X_small, y_small, strata_small):
    """score_slbt runs once per candidate feature at every node, each with its
    own seed drawn from the estimator's generator.  Fitting twice must consume
    that generator identically, or the trees would diverge deeper down even
    though the root agreed.
    """
    a = SLBT(max_depth=4, feats_viewed=3, random_state=42)
    b = SLBT(max_depth=4, feats_viewed=3, random_state=42)
    a.fit(X_small, y_small, strata_small)
    b.fit(X_small, y_small, strata_small)

    ra, rb = a.reporter.results, b.reporter.results
    assert len(ra) == len(rb)
    assert list(ra["feature"].fillna("-")) == list(rb["feature"].fillna("-"))
    np.testing.assert_allclose(ra["pi"].fillna(-1).to_numpy(),
                               rb["pi"].fillna(-1).to_numpy(), rtol=0, atol=0)
