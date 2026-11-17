"""
Mixing parameters (alpha) and latent budgets (beta) of each SLBT split, kept
on the node and reported by ``reporter.results``.

The checks tie them to quantities the tree already exposes: the split itself
(a modality goes left when its alpha is at least 0.5) and the LIFT values
(beta divided by the class distribution of the stratum, Eq. 4).
"""

import numpy as np
import pandas as pd
import pytest

from sltree import SLBT, SLBTWeighted


def _data(n=1500, seed=0, drop_d_in_stratum_1=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "X1": rng.choice(list("abcd"), n),
        "X2": rng.choice(list("abc"), n),
        "X3": rng.choice(list("ab"), n),
    })
    z = rng.integers(0, 2, n)
    if drop_d_in_stratum_1:
        X.loc[(z == 1) & (X.X1 == "d"), "X1"] = "c"
    left = np.where(z == 0, X.X1.isin(["a", "b"]), X.X1.isin(["b", "c"]))
    y = np.where(left,
                 rng.choice(["p", "q", "r"], n, p=[.7, .2, .1]),
                 rng.choice(["p", "q", "r"], n, p=[.1, .2, .7]))
    return X, pd.Series(y), z


def _internal(node, out):
    if node.left is None:
        return out
    out.append(node)
    _internal(node.left, out)
    _internal(node.right, out)
    return out


def _fit(hom, stratified=True, weighted=False):
    X, y, z = _data()
    kw = dict(max_depth=3, random_state=1, feats_viewed=3)
    if weighted:
        return SLBTWeighted(homogeneity=hom, **kw).fit(X, y, z, np.ones(len(y)))
    m = SLBT(homogeneity=hom, **kw)
    return m.fit(X, y, z) if stratified else m.fit(X, y)


CASES = [("none", True, False), ("A", True, False), ("B", True, False),
         ("AB", True, False), ("AB", False, False), ("none", True, True),
         ("A", True, True)]


@pytest.mark.parametrize("hom, stratified, weighted", CASES)
def test_alpha_gives_the_split(hom, stratified, weighted):
    """alpha >= 0.5 exactly for the modalities sent to the left child."""
    m = _fit(hom, stratified, weighted)
    nodes = _internal(m.root, [])
    assert nodes
    for node in nodes:
        if isinstance(next(iter(node.alpha.values())), dict):        # per stratum
            for s, thr in zip(node.strat_labels, node.treshold):
                left = {str(v) for v in thr}
                for mod, a in node.alpha[str(s)].items():
                    if a is not None:
                        assert (a >= 0.5) == (mod in left)
        else:
            left = {str(v) for v in node.treshold}
            for mod, a in node.alpha.items():
                assert (a >= 0.5) == (mod in left)


@pytest.mark.parametrize("hom, stratified, weighted", CASES)
def test_beta_are_distributions_and_give_the_lift(hom, stratified, weighted):
    """Each latent budget sums to 1, and LIFT = beta / p_{j|(g)} (Eq. 4)."""
    m = _fit(hom, stratified, weighted)
    for node in _internal(m.root, []):
        strata = [str(s) for s in node.strat_labels]
        prior = np.atleast_2d(node.strat_distribution)
        for beta, lift in ((node.beta_left, node.LIFT_1), (node.beta_right, node.LIFT_2)):
            per_stratum = isinstance(beta, dict)
            rows = [np.asarray(beta[s]) for s in strata] if per_stratum else \
                   [np.asarray(beta)] * len(strata)
            lift = np.atleast_2d(np.asarray(lift, dtype=float))
            for t, b in enumerate(rows):
                assert b.shape == (len(node.labels),)
                assert np.all(b >= -1e-12) and np.all(b <= 1 + 1e-12)
                assert b.sum() == pytest.approx(1.0, abs=1e-8)
                ok = prior[t] > 0
                np.testing.assert_allclose(lift[t][ok], b[ok] / prior[t][ok], rtol=1e-9)


def test_unobserved_modality_has_no_alpha():
    """A modality absent from a stratum at the node gets None there."""
    X, y, z = _data(drop_d_in_stratum_1=True)
    m = SLBT(homogeneity="none", max_depth=1, random_state=1, feats_viewed=3).fit(X, y, z)
    assert m.root.feature == "X1"
    assert m.root.alpha["1"]["d"] is None
    assert m.root.alpha["0"]["d"] is not None


def test_reporter_columns_and_pruning():
    """The report carries alpha / beta for internal nodes only, also after pruning."""
    m = _fit("none")
    rep = m.reporter.results
    assert {"alpha", "beta_left", "beta_right"} <= set(rep.columns)
    internal = rep[rep.node_type == "Internal"]
    leaves = rep[rep.node_type == "Leaf"]
    assert internal.alpha.notna().all() and internal.beta_left.notna().all()
    assert leaves.alpha.isna().all() and leaves.beta_right.isna().all()

    m.prune_after_vp(1)                       # the root alone
    assert m.root.alpha is None and m.root.beta_left is None
    rep = m.reporter.results
    assert len(rep) == 1 and rep.alpha.isna().all()


def test_empty_report_has_the_new_columns():
    from sltree.reporting import TreeReporter
    cols = TreeReporter().results.columns
    assert {"alpha", "beta_left", "beta_right"} <= set(cols)
