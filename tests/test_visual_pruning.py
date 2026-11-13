"""
Visual pruning and decision-tree selection, Eqs. 6-8 of the SLBT paper.

* V(t) = (i(1) - i(t)) / i(1), with the impurity of Y within the strata in
  the stratified trees (Eq. 2 of the paper);
* V(T) = sum over the leaves of V(h) p(h);
* the suggested tree is T_{nu-1} for the first nu with
  V(T_nu) - V(T_{nu-1}) <= epsilon, starting from T_1 against the root alone.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from sltree import SLBT, SLBTWeighted, PredictabilityTree
from sltree._tree._visual_pruning import expansion_order


def _leaves(node):
    return [node] if node.left is None else _leaves(node.left) + _leaves(node.right)


def _gini(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return 1.0 - float(np.sum(p ** 2))


def _gini_z(y, z):
    return sum((z == g).mean() * _gini(y[z == g]) for g in np.unique(z))


class _RecSLBT(SLBT):
    """SLBT that remembers the training rows of every node."""

    def fit(self, *a, **k):
        self.rows_ = {}
        return super().fit(*a, **k)

    def _grow_tree(self, strategy, rows, cols, *a, **k):
        node = super()._grow_tree(strategy, rows, cols, *a, **k)
        self.rows_[node.position] = rows.copy()
        return node


def _null_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{c}": rng.choice(list("abcdef")[: 2 + c], n) for c in range(5)})
    y = pd.Series(rng.choice(["A", "B", "C"], n))
    z = rng.integers(0, 2, n)
    return X, y, z


def _inverted_data(n=1500, seed=1):
    """x0 decides y, with the opposite sign in the two strata: marginally,
    x0 says nothing about y."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{c}": rng.choice(list("abcd"), n) for c in range(4)})
    z = rng.integers(0, 2, n)
    good = X.x0.isin(["a", "b"]).to_numpy() ^ (z == 1)
    y = pd.Series(np.where(rng.random(n) < np.where(good, 0.85, 0.15), "A", "B"))
    return X, y, z


# ------------------------------------------------------------------ #
#  Eq. 6 within the strata
# ------------------------------------------------------------------ #

def test_slbt_v_uses_the_impurity_within_the_strata():
    X, y, z = _inverted_data()
    m = _RecSLBT(homogeneity="A", max_depth=3, feats_viewed=4, random_state=0).fit(X, y, z)
    yv = y.to_numpy()
    i1 = _gini_z(yv, z)
    stack = [m.root]
    while stack:
        node = stack.pop()
        r = m.rows_[node.position]
        assert node.impurity_decrease == pytest.approx(1 - _gini_z(yv[r], z[r]) / i1, abs=1e-12)
        stack.extend(c for c in (node.left, node.right) if c is not None)


def test_without_strata_v_is_the_marginal_one():
    X, y, _ = _null_data(n=600)
    m = _RecSLBT(max_depth=3, feats_viewed=5, random_state=0).fit(X, y)
    yv = y.to_numpy()
    for node in _leaves(m.root):
        r = m.rows_[node.position]
        assert node.impurity_decrease == pytest.approx(1 - _gini(yv[r]) / _gini(yv), abs=1e-12)


# ------------------------------------------------------------------ #
#  Eq. 8: which tree is suggested
# ------------------------------------------------------------------ #

def _path(m):
    """(node, gain) along the visual-pruning sequence."""
    out, prev = [], 0.0
    for node in expansion_order(m.root):
        out.append((node, node.tree_partial_impurity_reduction - prev))
        prev = node.tree_partial_impurity_reduction
    return out


@pytest.mark.parametrize("eps", [0.005, 0.01, 0.02])
def test_the_suggested_tree_stops_before_the_first_small_gain(eps):
    X, y, z = _inverted_data(seed=3)
    m = SLBT(homogeneity="A", max_depth=5, feats_viewed=4, random_state=1, epsilon=eps).fit(X, y, z)
    path = _path(m)
    gains = [g for _, g in path]
    assert all(g >= -1e-12 for g in gains)              # V(T) never decreases
    flagged = [n for n, _ in path if n.suggested_pruning]
    assert len(flagged) <= 1
    full = copy.deepcopy(m)
    cut = m.prune_suggested()
    if cut is None:
        assert all(g > eps for g in gains)
        return
    k = [n.position for n, _ in path].index(cut)
    assert gains[k] <= eps and all(g > eps for g in gains[:k])
    # the pruned tree is the one with the first k splits of the sequence
    assert len(_leaves(m.root)) == k + 1
    assert len(_leaves(full.root)) > len(_leaves(m.root))


def test_null_data_are_pruned_to_the_root():
    X, y, z = _null_data()
    m = SLBT(homogeneity="A", max_depth=5, feats_viewed=5, random_state=0).fit(X, y, z)
    assert len(_leaves(m.root)) > 4
    assert m.prune_suggested() == 1
    assert m.root.left is None and m.root.right is None
    assert m.predict(X, x_s=z).shape == (len(y),)
    assert m.predict_proba(X, x_s=z, by_stratum=True).shape == (len(y), 3)


def test_an_inversion_survives_the_pruning():
    """With the impurity within the strata the reversed effect of x0 is a
    gain; with the marginal impurity it would be pruned away."""
    X, y, z = _inverted_data()
    m = SLBT(homogeneity="A", max_depth=5, feats_viewed=4, random_state=0).fit(X, y, z)
    m.prune_suggested()
    assert m.root.feature == "x0"


def test_pruning_twice_changes_nothing():
    X, y, z = _inverted_data(seed=4)
    m = SLBT(homogeneity="A", max_depth=5, feats_viewed=4, random_state=0).fit(X, y, z)
    m.prune_suggested()
    n_leaves = len(_leaves(m.root))
    assert m.prune_suggested() is None
    assert len(_leaves(m.root)) == n_leaves


def test_larger_epsilon_never_gives_a_larger_tree():
    X, y, z = _inverted_data(seed=5)
    m = SLBT(homogeneity="A", max_depth=5, feats_viewed=4, random_state=0).fit(X, y, z)
    sizes = []
    for eps in (0.001, 0.005, 0.01, 0.02, 0.05):
        mm = copy.deepcopy(m)
        mm.prune_suggested(eps)
        sizes.append(len(_leaves(mm.root)))
    assert sizes == sorted(sizes, reverse=True)


def test_parents_come_before_children_in_the_sequence():
    X, y, z = _null_data(n=800)
    m = SLBT(homogeneity="none", max_depth=5, feats_viewed=5, random_state=0).fit(X, y, z)
    seen = set()
    for node in expansion_order(m.root):
        if node is not m.root:
            assert node.position // 2 in seen
        seen.add(node.position)


# ------------------------------------------------------------------ #
#  the other trees
# ------------------------------------------------------------------ #

def test_predictability_tree_prunes_null_data_to_the_root():
    X, y, _ = _null_data()
    t = PredictabilityTree(max_depth=5, feats_viewed=5).fit(X, y)
    assert t.prune_suggested() == 1
    assert t.root.left is None


def test_two_class_pruning_keeps_a_numeric_prediction():
    """Collapsing a subtree of a twoClass tree used to read the mean of the
    subtree after the subtree had been dropped."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame({f"x{c}": rng.choice(list("abc"), 500) for c in range(3)})
    y = pd.Series(rng.normal(size=500) + (X.x0 == "a") * 2.0)
    t = PredictabilityTree(model="twoClass", max_depth=4, feats_viewed=3).fit(X, y)
    internal = [n for n in expansion_order(t.root)]
    t.prune_after_vp(internal[len(internal) // 2].position)
    for leaf in _leaves(t.root):
        assert isinstance(leaf.value, float)


def test_weighted_slbt_uses_the_weight_shares():
    X, y, z = _inverted_data(n=600)
    w = np.random.default_rng(0).gamma(2.0, size=len(y))
    m = SLBTWeighted(homogeneity="A", max_depth=3, feats_viewed=4, random_state=0).fit(
        X, y, z, sample_weight=w)
    total = sum(leaf.mass for leaf in _leaves(m.root))
    assert total == pytest.approx(1.0)
    v = sum(leaf.impurity_decrease * leaf.mass for leaf in _leaves(m.root))
    assert v == pytest.approx(max(n.tree_partial_impurity_reduction
                                  for n in expansion_order(m.root)))
