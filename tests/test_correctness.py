"""
Correctness and parameter recovery (rOpenSci G5.4 and G5.6).

G5.4 asks that "statistical algorithms produce expected results to some fixed
test data sets"; G5.6 that they "produce expected results given data with known
properties".  The data here is built so that the right answer is known in
advance, and the tests assert that answer — not whatever the library happens to
return today.

Two identities from the reference paper are checked directly:
Iorio, Aria, D'Ambrosio and Siciliano (2019), *Informative trees by visual
pruning*, Expert Systems With Applications 127:228-240 — equation 3.1 for the
node impurity proportional reduction, and 3.6 for the tree-level one.
"""

import contextlib
import os

import numpy as np
import pandas as pd
import pytest

from sltree import BINPI, Categorizer, PredictabilityTree, SLBT


def _quiet():
    return contextlib.redirect_stdout(open(os.devnull, "w"))


def _leaves(node):
    if node is None:
        return
    if node.left is None:
        yield node
    else:
        yield from _leaves(node.left)
        yield from _leaves(node.right)


def _internal(node):
    if node is None or node.left is None:
        return
    yield node
    yield from _internal(node.left)
    yield from _internal(node.right)


# ------------------------------------------------------------------ #
#  Recovering a known split
# ------------------------------------------------------------------ #

def test_the_informative_predictor_is_chosen_at_the_root():
    """y is a deterministic function of x0; the other two columns are noise.

    The root must split on x0, reach PI = 1 and classify perfectly.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "signal": rng.choice(list("ab"), size=n),
        "noise1": rng.choice(list("pq"), size=n),
        "noise2": rng.choice(list("rs"), size=n),
    })
    y = pd.Series(np.where(X["signal"] == "a", "A", "B"))

    tree = PredictabilityTree(max_depth=3, feats_viewed=3)
    tree.fit(X, y)

    assert tree.root.feature == "signal"
    assert tree.root.pi == pytest.approx(1.0)
    assert (tree.predict(X) == y.values).mean() == pytest.approx(1.0)


@pytest.mark.parametrize("model", ["twoStage", "twoing"])
def test_both_splitting_criteria_recover_the_same_signal(model):
    """The criteria differ in how they search, not in what a clean signal is."""
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame({
        "signal": rng.choice(list("abc"), size=n),
        "noise":  rng.choice(list("pq"),  size=n),
    })
    y = pd.Series(np.where(X["signal"] == "a", "A", "B"))

    tree = PredictabilityTree(model=model, max_depth=3, feats_viewed=2)
    tree.fit(X, y)

    assert tree.root.feature == "signal"
    assert (tree.predict(X) == y.values).mean() == pytest.approx(1.0)


def test_a_pure_leaf_has_zero_impurity():
    rng = np.random.default_rng(2)
    n = 120
    X = pd.DataFrame({"signal": rng.choice(list("ab"), size=n)})
    y = pd.Series(np.where(X["signal"] == "a", "A", "B"))

    tree = PredictabilityTree(max_depth=2, feats_viewed=1)
    tree.fit(X, y)

    for leaf in _leaves(tree.root):
        assert leaf.impurity == pytest.approx(0.0)


# ------------------------------------------------------------------ #
#  The two identities of Iorio et al. (2019)
# ------------------------------------------------------------------ #

def test_impurity_decrease_is_zero_at_the_root(X, y):
    """Equation 3.1 normalised is V_z = 1 - I_z/I_root, so the root gives 0.

    This is the quantity the DTG uses for the vertical axis; if it drifted from
    zero at the root the whole visual-pruning canvas would be shifted.
    """
    tree = PredictabilityTree(max_depth=5, feats_viewed=3)
    tree.fit(X, y)
    assert tree.root.impurity_decrease == pytest.approx(0.0, abs=1e-12)


def test_impurity_decrease_matches_its_definition(X, y):
    """Recomputed from impurity and N, node by node: V_z = 1 - I_z/I_root."""
    tree = PredictabilityTree(max_depth=5, feats_viewed=3)
    tree.fit(X, y)

    I_root = tree.root.impurity * tree.root.N

    def walk(node):
        if node is None:
            return
        expected = 1.0 - (node.impurity * node.N) / I_root
        assert node.impurity_decrease == pytest.approx(expected, abs=1e-12), \
            f"node {node.position}"
        walk(node.left)
        walk(node.right)

    walk(tree.root)


def test_tree_impurity_proportional_reduction_lies_in_the_unit_interval(X, y):
    """Equation 3.6: V(T) = sum over terminal nodes of V_h(T) p(h).

    It is the quantity the automatic tree-size suggestion is built on, and by
    construction it cannot leave [0, 1].
    """
    tree = PredictabilityTree(max_depth=6, feats_viewed=3)
    tree.fit(X, y)

    N = tree.root.N
    V = sum(leaf.impurity_decrease * leaf.N / N for leaf in _leaves(tree.root))
    assert 0.0 <= V <= 1.0


def test_impurity_decrease_grows_away_from_the_root(X, y):
    """A child always sits higher on the DTG than its parent.

    This monotonicity is what makes the horizontal cut of the visual pruning
    meaningful; prune_after_vp relies on it.
    """
    tree = PredictabilityTree(max_depth=5, feats_viewed=3)
    tree.fit(X, y)

    for node in _internal(tree.root):
        for child in (node.left, node.right):
            assert child.impurity_decrease >= node.impurity_decrease - 1e-12, \
                f"node {node.position} -> {child.position}"


# ------------------------------------------------------------------ #
#  Stratified splits: the case SLBT exists for
# ------------------------------------------------------------------ #

def _inverted_data(n=400, seed=9):
    """A predictor whose relation with y reverses between strata.

    In stratum 0, 'a' means A and 'b' means B; in stratum 1 the mapping is
    exactly the other way round.  Marginally, the predictor is useless: pooling
    the strata leaves the classes evenly mixed.  Only a model that can give each
    stratum its own split direction can see the structure.

    This is the London finding in miniature — there, car ownership sent
    car-free men and car-owning women to the same child.
    """
    rng = np.random.default_rng(seed)
    stratum = rng.integers(0, 2, n)
    signal = rng.choice(list("ab"), size=n)
    y = np.where(stratum == 0,
                 np.where(signal == "a", "A", "B"),
                 np.where(signal == "a", "B", "A"))
    X = pd.DataFrame({"inverted": signal,
                      "noise": rng.choice(list("pq"), size=n)})
    return X, pd.Series(y), stratum


@pytest.mark.parametrize("homogeneity", ["none", "B"])
def test_free_alpha_finds_a_split_that_reverses_between_strata(homogeneity):
    """'none' and 'B' leave alpha free per stratum, so they can represent it."""
    X, y, stratum = _inverted_data()

    with _quiet():
        model = SLBT(homogeneity=homogeneity, max_depth=2, feats_viewed=2,
                     random_state=0)
        model.fit(X, y, stratum)

    assert model.root.feature == "inverted"
    assert model.root.pi == pytest.approx(1.0, abs=1e-6)

    # One threshold per stratum, and they must differ — that is the inversion.
    threshold = model.root.treshold
    assert isinstance(threshold, list) and len(threshold) == 2
    assert set(threshold[0]) != set(threshold[1])


@pytest.mark.parametrize("homogeneity", ["A", "AB"])
def test_shared_alpha_cannot_see_the_inversion(homogeneity):
    """The other half of the argument.

    'A' and 'AB' impose one split direction across strata, so the inverted
    predictor looks like noise to them.  Without this half, the test above would
    only show that SLBT works, not that the free-alpha setting is *needed*.
    """
    X, y, stratum = _inverted_data()

    with _quiet():
        model = SLBT(homogeneity=homogeneity, max_depth=2, feats_viewed=2,
                     random_state=0)
        model.fit(X, y, stratum)

    assert model.root.pi < 0.1, \
        "a constrained model should find almost no signal here"


def test_predict_ignores_the_stratum_and_loses_the_inverted_split():
    """A capability gap, asserted so that it is visible rather than forgotten.

    SLBT.predict(X) takes no stratum.  When thresholds are stratified — the
    'none' and 'B' settings — _traverse_tree falls back to the union of the
    per-stratum modality sets.  For an inversion that union is every modality,
    so the split collapses and prediction drops to chance, even though the tree
    found the structure perfectly (PI = 1 in the test above).

    Feeding each row its own stratum recovers perfect accuracy, which is what
    the second half measures.  The test documents the gap; closing it means
    giving predict() an x_s argument.
    """
    X, y, stratum = _inverted_data()

    with _quiet():
        model = SLBT(homogeneity="none", max_depth=2, feats_viewed=2,
                     random_state=0)
        model.fit(X, y, stratum)

    blind = (model.predict(X) == y.values).mean()

    def traverse_with_stratum(row, s, node):
        if node.left is None:
            return node.value
        threshold = node.treshold
        if isinstance(threshold, np.ndarray) and threshold.ndim == 1:
            goes_left = row[node.feature] in threshold
        else:
            goes_left = row[node.feature] in threshold[s]
        return traverse_with_stratum(row, s, node.left if goes_left else node.right)

    informed = np.mean([
        traverse_with_stratum(X.iloc[i], stratum[i], model.root) == y.iloc[i]
        for i in range(len(y))
    ])

    assert blind < 0.6, "prediction without the stratum should be near chance"
    assert informed == pytest.approx(1.0), \
        "the same tree, given the stratum, classifies perfectly"


# ------------------------------------------------------------------ #
#  Categorizer
# ------------------------------------------------------------------ #

def test_elbow_recovers_three_well_separated_clusters():
    """Three tight clusters at 0, 10 and 20."""
    rng = np.random.default_rng(4)
    values = np.concatenate([rng.normal(0, 0.3, 80),
                             rng.normal(10, 0.3, 80),
                             rng.normal(20, 0.3, 80)])
    with _quiet():
        cat = Categorizer(method="elbow", k_max=5, random_state=0)
        cat.fit(pd.DataFrame({"v": values}))

    assert cat.k_["v"] == 3


def test_the_elbow_result_depends_on_k_max():
    """Asserted deliberately, because it is a property of the heuristic.

    kmeans_elbow derives its slope threshold from SSE at K=1 and K=Kmax, so the
    K it selects moves with Kmax: the same three clusters give 3 at Kmax=5 and 4
    at Kmax=8.  Anyone reading the results needs to know that Kmax is a real
    parameter, not just an upper bound.
    """
    rng = np.random.default_rng(4)
    values = np.concatenate([rng.normal(0, 0.3, 80),
                             rng.normal(10, 0.3, 80),
                             rng.normal(20, 0.3, 80)])
    found = {}
    for k_max in (5, 8):
        with _quiet():
            cat = Categorizer(method="elbow", k_max=k_max, random_state=0)
            cat.fit(pd.DataFrame({"v": values}))
        found[k_max] = cat.k_["v"]

    assert found[5] != found[8], \
        "if this ever passes equal, the heuristic changed and the docs must too"


def test_fixed_k_produces_exactly_k_bins():
    rng = np.random.default_rng(5)
    values = np.concatenate([rng.normal(m, 0.3, 60) for m in (0, 5, 10, 15)])
    with _quiet():
        out = Categorizer(method="fixed", k=4, random_state=0) \
            .fit_transform(pd.DataFrame({"v": values}))

    assert out["v"].nunique() == 4


def test_bins_preserve_order():
    """Discretisation must be monotone: a larger value never lands in a lower
    bin.  Interval labels are ordered, so comparing bin edges is enough."""
    rng = np.random.default_rng(6)
    values = np.sort(rng.normal(size=200))
    with _quiet():
        cat = Categorizer(method="fixed", k=4, random_state=0)
        cat.fit(pd.DataFrame({"v": values}))

    edges = cat.bins_["v"]
    assert list(edges) == sorted(edges)


def test_silhouette_labels_agree_with_the_k_it_reports():
    """A regression guard.

    kmeans_silhouette used to encode the label buffer as optimal_k + 0.5*alt,
    where alt was the parity of the last iteration rather than of the best K, so
    the caller copied labels belonging to a different K.  Three of four columns
    came back inconsistent, and the mismatch overflowed the centers array.
    """
    from sltree._preprocessing._backend._categorizer import categorize_silhouette

    rng = np.random.default_rng(11)
    for seed in range(6):
        values = np.sort(rng.normal(size=300))
        with _quiet():
            labels, centers, k_opt = categorize_silhouette(values, 5, 2, 1, seed)
        assert labels.max() < k_opt, f"seed {seed}: labels reach {labels.max()} but k={k_opt}"
        assert len(centers) == k_opt


# ------------------------------------------------------------------ #
#  BINPI
# ------------------------------------------------------------------ #

def _mirrored_data(n=200, seed=7):
    """A column that is a pure function of another, with a fifth of it removed.

    Three categories map one-to-one onto three others, so the imputation has a
    single right answer and recovery can be measured exactly.
    """
    rng = np.random.default_rng(seed)
    driver = rng.choice(list("abc"), size=n)
    mirror = pd.Series(np.where(driver == "a", "x",
                       np.where(driver == "b", "y", "z")), dtype=object)
    X = pd.DataFrame({"driver": driver, "mirror": mirror.copy()})
    truth = X["mirror"].copy()
    X.loc[::5, "mirror"] = np.nan
    return X, truth


def test_binpi_recovers_a_deterministic_relationship():
    """With weak learners deep enough to represent it, recovery is exact."""
    X, truth = _mirrored_data()

    with _quiet():
        out = BINPI(ensemble="forest", n_estimators=10, max_depth=2,
                    random_state=0).fit_transform(X)

    assert out.isnull().sum().sum() == 0
    agreement = (out["mirror"].values == truth.values).mean()
    assert agreement == pytest.approx(1.0), \
        f"only {agreement:.1%} of the imputed values are right"


def test_the_default_binpi_depth_cannot_represent_a_three_way_mapping():
    """Asserted on purpose: BINPI defaults to max_depth=1.

    A depth-1 weak learner makes a single binary split, so it cannot separate
    three categories however many of them are averaged.  On the mapping above it
    recovers 93% against 100% at depth 2.  Nothing is broken — a stump is a
    legitimate weak learner — but the default is worth knowing about, and the
    docstring should say so.
    """
    X, truth = _mirrored_data()

    scores = {}
    for depth in (1, 2):
        with _quiet():
            out = BINPI(ensemble="forest", n_estimators=10, max_depth=depth,
                        random_state=0).fit_transform(X.copy())
        scores[depth] = (out["mirror"].values == truth.values).mean()

    assert scores[1] < scores[2]
    assert scores[2] == pytest.approx(1.0)
