"""
SLBT against an independent reference implementation.

Every quantity of the SLBT paper that the library computes is recomputed here
from the raw data with plain numpy (and scipy for the least-squares
optimum), without going through the C backend or the library's own helpers:

    Eq. 1        partial tau of Gray and Williams (GPI)
    Sec. 2.3     Predictability Improvement  PI = tau_{Y|s,Z}
    Eq. 4        Lift, prior = class distribution of each stratum at the node
    Eq. 5        Leaf Classification Ratio per stratum
    Eq. 12-13    least-squares objective, "budget" and "frequency" weighting
    Eq. 8        epsilon of the suggested pruning

These tests were added with the revision that aligned the code with the paper;
each docstring says which discrepancy it guards against.
"""

import contextlib
import itertools
import os

import numpy as np
import pandas as pd
import pytest

from sltree import SLBT


def _quiet():
    return contextlib.redirect_stdout(open(os.devnull, "w"))


# ------------------------------------------------------------------ #
#  Reference implementation
# ------------------------------------------------------------------ #

def ref_partial_tau(x, y, z):
    """Eq. 1 written out from its definition, on raw vectors."""
    x, y, z = map(np.asarray, (x, y, z))
    n = len(y)
    num = 0.0
    den = 0.0
    for g in np.unique(z):
        mg = z == g
        p_g = mg.sum() / n
        prof_g = np.array([(y[mg] == j).mean() for j in np.unique(y)])
        den += p_g * (1.0 - (prof_g ** 2).sum())
        for i in np.unique(x[mg]):
            mi = mg & (x == i)
            p_ig = mi.sum() / n
            prof_ig = np.array([(y[mi] == j).mean() for j in np.unique(y)])
            num += p_ig * ((prof_ig - prof_g) ** 2).sum()
    return 0.0 if den <= 1e-12 else num / den


def ref_split_side(node, x, z):
    """Side (True = left) of each observation for the split stored at node."""
    thr = node.treshold
    x = np.asarray(x).astype(str)
    if isinstance(thr, np.ndarray) and thr.ndim == 1:
        left = {str(v) for v in thr}
        return np.array([v in left for v in x])
    labels = [str(s) for s in node.strat_labels]
    out = np.zeros(len(x), dtype=bool)
    for k, (v, s) in enumerate(zip(x, np.asarray(z))):
        out[k] = v in {str(u) for u in thr[labels.index(str(s))]}
    return out


def ref_ls_loss(budgets, weights, alpha, beta):
    """sum_{t,i} w_ti || p_ti - alpha_ti beta_t' ||^2  (Eq. 12-13 up to a constant).

    budgets : (T, I, J); weights : (T, I); alpha : (T, I, 2); beta : (T, J, 2)
    """
    pi = np.einsum("tik,tjk->tij", alpha, beta)
    return float((weights * ((budgets - pi) ** 2).sum(axis=2)).sum())


def ref_ls_optimum(budgets, weights, homogeneity, n_starts=30, seed=0):
    """Global LS optimum by multi-start L-BFGS on the softmax parameters."""
    from scipy.optimize import minimize

    T, I, J = budgets.shape
    TA = 1 if homogeneity in ("A", "AB") else T
    TB = 1 if homogeneity in ("B", "AB") else T

    def unpack(theta):
        U = theta[: TA * I * 2].reshape(TA, I, 2)
        V = theta[TA * I * 2:].reshape(TB, J, 2)
        A = np.exp(U - U.max(axis=2, keepdims=True))
        A /= A.sum(axis=2, keepdims=True)
        B = np.exp(V - V.max(axis=1, keepdims=True))
        B /= B.sum(axis=1, keepdims=True)
        A = np.broadcast_to(A, (T, I, 2)) if TA == 1 else A
        B = np.broadcast_to(B, (T, J, 2)) if TB == 1 else B
        return A, B

    def f(theta):
        A, B = unpack(theta)
        return ref_ls_loss(budgets, weights, A, B)

    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(n_starts):
        theta0 = rng.normal(0, 2, TA * I * 2 + TB * J * 2)
        res = minimize(f, theta0, method="L-BFGS-B")
        best = min(best, res.fun)
    return best


# ------------------------------------------------------------------ #
#  Data
# ------------------------------------------------------------------ #

def _data(n=600, seed=3, G=2, J=3):
    """Categorical data in which the effect of x0 differs between strata."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, G, n)
    X = pd.DataFrame({
        "x0": rng.choice(list("abcd"), n),
        "x1": rng.choice(list("pqr"), n),
        "x2": rng.choice(list("uv"), n),
    })
    classes = np.array(list("ABCDE")[:J])
    logits = np.zeros((n, J))
    eff = {"a": 0, "b": 1, "c": 2, "d": 0}
    for k in range(n):
        j = (eff[X.x0[k]] + z[k]) % J          # effect of x0 shifts with z
        logits[k, j] += 2.0
        logits[k, 0] += 0.7 * (X.x1[k] == "p")
    P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(classes, p=p) for p in P])
    return X, pd.Series(y), z


def _internal(node):
    if node is None or node._is_leaf_node():
        return []
    return [node] + _internal(node.left) + _internal(node.right)


def _leaves(node):
    if node is None:
        return []
    if node._is_leaf_node():
        return [node]
    return _leaves(node.left) + _leaves(node.right)


def _node_rows(model, X, z):
    """Row indices reaching each node, by replaying the stored splits."""
    rows = {}

    def walk(node, idx):
        rows[node.position] = idx
        if node._is_leaf_node():
            return
        side = ref_split_side(node, X[node.feature].iloc[idx], z[idx])
        walk(node.left, idx[side])
        walk(node.right, idx[~side])

    walk(model.root, np.arange(len(X)))
    return rows


HOMOGENEITIES = ["none", "A", "B", "AB"]


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("homogeneity", HOMOGENEITIES)
def test_pi_and_gpi_match_the_paper_definition_at_every_node(homogeneity):
    """PI = tau_{Y|s,Z} and GPI = tau_{Y|X,Z} (Eq. 1), node by node.

    Guards the main discrepancy found in the revision: the C code returned
    tau(parent) - (L tau_left + R tau_right) instead of tau_{Y|s,Z}.
    """
    X, y, z = _data()
    with _quiet():
        model = SLBT(homogeneity=homogeneity, max_depth=3, feats_viewed=3,
                     random_state=1, n_init=3).fit(X, y, z)
    rows = _node_rows(model, X, z)
    nodes = _internal(model.root)
    assert nodes, "the tree did not split"
    for node in nodes:
        idx = rows[node.position]
        xs, ys, zs = X[node.feature].iloc[idx], y.iloc[idx], z[idx]
        side = ref_split_side(node, xs, zs)
        assert node.pi == pytest.approx(ref_partial_tau(side, ys, zs), abs=1e-9), node.position
        assert node.gpi == pytest.approx(ref_partial_tau(xs, ys, zs), abs=1e-9), node.position
        assert node.pi <= node.gpi + 1e-12


@pytest.mark.parametrize("homogeneity", HOMOGENEITIES)
def test_predict_follows_the_partition_grown_in_fit(homogeneity):
    """Routing training rows with predict() reproduces the fitted leaves."""
    X, y, z = _data()
    with _quiet():
        model = SLBT(homogeneity=homogeneity, max_depth=3, feats_viewed=3,
                     random_state=1).fit(X, y, z)
    rows = _node_rows(model, X, z)
    leaf_of_row = model.apply(X, x_s=z)
    for leaf in _leaves(model.root):
        assert set(np.flatnonzero(leaf_of_row == leaf.position)) == set(rows[leaf.position])
        assert len(rows[leaf.position]) == leaf.N


@pytest.mark.parametrize("homogeneity", HOMOGENEITIES)
def test_lift_uses_the_class_distribution_of_each_stratum(homogeneity):
    """Eq. 4: Lift = beta_{j|k(g)} / p_{j|(g)}(t).

    beta is not stored on the node, but each latent budget sums to one, so
    sum_j Lift_j(g) * p_{j|(g)}(t) must equal 1 for every stratum and both
    children.  With the pooled distribution as denominator (earlier versions)
    the identity fails whenever the strata have different class profiles.
    """
    X, y, z = _data()
    with _quiet():
        model = SLBT(homogeneity=homogeneity, max_depth=2, feats_viewed=3,
                     random_state=1).fit(X, y, z)
    rows = _node_rows(model, X, z)
    for node in _internal(model.root):
        idx = rows[node.position]
        ys, zs = y.iloc[idx].to_numpy(), z[idx]
        for lift in (node.LIFT_1, node.LIFT_2):
            lift = lift if isinstance(lift, list) else [lift]
            for t, g in enumerate(np.unique(zs)):
                prior = np.array([(ys[zs == g] == j).mean() for j in node.labels])
                ok = prior > 0
                assert (np.asarray(lift[t])[ok] * prior[ok]).sum() == pytest.approx(1.0, abs=1e-9)


def test_leaf_classification_ratio_is_computed_within_strata():
    """Eq. 5: LCR_{j|(g)}(h) = p_{j|(g)}(h) / p_{j|(g)}(1)."""
    X, y, z = _data()
    with _quiet():
        model = SLBT(homogeneity="A", max_depth=3, feats_viewed=3,
                     random_state=1).fit(X, y, z)
    rows = _node_rows(model, X, z)
    yv = y.to_numpy()
    for leaf in _leaves(model.root):
        idx = rows[leaf.position]
        for t, g in enumerate(leaf.strat_labels):
            in_leaf = idx[z[idx] == g]
            for jj, j in enumerate(leaf.labels):
                root_p = (yv[z == g] == j).mean()
                leaf_p = (yv[in_leaf] == j).mean()
                assert leaf.LCR[t][jj] == pytest.approx(leaf_p / root_p, abs=1e-9)


def _budget_tables(seed, T=2, I=4, J=3, n=500):
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, rng.dirichlet(np.ones(T * I * J) * 0.8)).reshape(T, I, J).astype(float)
    counts[0, 1, :] = 0.0                      # one unobserved row
    joint = counts / counts.sum()
    row = joint.sum(axis=2)
    budgets = np.divide(joint, row[:, :, None], out=np.zeros_like(joint), where=row[:, :, None] > 0)
    return joint, budgets, row


@pytest.mark.parametrize("optimizer", ["projected", "softmax"])
@pytest.mark.parametrize("homogeneity", HOMOGENEITIES)
@pytest.mark.parametrize("ls_weights", ["budget", "frequency"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_least_squares_fit_reaches_the_reference_optimum(homogeneity, ls_weights, seed, optimizer):
    """The C fit with 10 starts reaches the scipy multi-start optimum.

    "projected" reaches it to numerical precision.  "softmax" (the neural
    network parameterisation of the paper, Eq. 16-19) only approaches optima
    lying on the boundary of the simplex, where its gradient vanishes: it is
    held to 1% of the optimum, the gap observed on these tables being < 0.5%.

    "budget"    : sum_{t,i observed} || p_ti - pi_ti ||^2
    "frequency" : sum_{t,i} f_ti || p_ti - pi_ti ||^2, i.e. Eq. 13 with the
                  row frequencies on the diagonal of D (up to a constant
                  factor, which does not move the minimiser).
    """
    pytest.importorskip("scipy")
    from sltree._tree.split import score_slbt, _ls_table

    joint, budgets, row = _budget_tables(seed)
    _, _, alpha, beta, diag = score_slbt(joint, budgets, homogeneity, seed=7,
                                         n_init=10, ls_weights=ls_weights,
                                         optimizer=optimizer)
    T = budgets.shape[0]
    A = np.broadcast_to(alpha, (T,) + alpha.shape[1:]) if alpha.shape[0] == 1 else alpha
    B = np.broadcast_to(beta,  (T,) + beta.shape[1:])  if beta.shape[0] == 1 else beta

    if ls_weights == "budget":
        w = (row > 0).astype(float)
    else:
        w = row / row[row > 0].mean()
    assert np.allclose(_ls_table(joint, budgets, ls_weights).sum(axis=2), w)

    c_loss = ref_ls_loss(budgets, w, A, B)
    assert diag["loss"] == pytest.approx(c_loss, rel=1e-6, abs=1e-9)
    ref = ref_ls_optimum(budgets, w, homogeneity)
    tol = 1e-6 if optimizer == "projected" else 1e-2
    assert c_loss >= ref - 1e-6 * (1 + ref), (c_loss, ref)   # never below the optimum
    assert c_loss <= ref + tol * (1 + ref), (c_loss, ref)


def test_more_starts_never_increase_the_loss():
    """Start 0 is shared, so the best of n starts is at least as good."""
    from sltree._tree.split import score_slbt
    for seed in range(5):
        joint, budgets, _ = _budget_tables(seed, I=5, J=4)
        l1 = score_slbt(joint, budgets, "none", seed=seed, n_init=1)[4]["loss"]
        l8 = score_slbt(joint, budgets, "none", seed=seed, n_init=8)[4]["loss"]
        assert l8 <= l1 + 1e-12


def test_diagnostics_report_convergence_and_split_agreement():
    from sltree._tree.split import score_slbt
    joint, budgets, _ = _budget_tables(0)
    *_, diag = score_slbt(joint, budgets, "A", seed=1, n_init=6, optimizer="projected")
    assert diag["n_init"] == 6
    assert 1 <= diag["n_same_split_starts"] <= 6
    assert 0 <= diag["n_converged_starts"] <= 6
    assert diag["loss"] <= diag["loss_worst_start"] + 1e-12
    assert diag["converged"] == 1


@pytest.mark.parametrize("optimizer", ["projected", "softmax"])
def test_unconstrained_components_are_aligned_across_strata(optimizer):
    """In 'none' nothing ties component 1 of one stratum to component 1 of
    another.  The fit aligns them: every stratum's pair of latent budgets is
    at least as close to the reference stratum's (the heaviest) as the swapped
    pair would be.  Without this the random start decided which observations
    of different strata share a child node."""
    from sltree._tree.split import score_slbt
    for seed in range(6):
        joint, budgets, row = _budget_tables(seed, T=3, I=5, J=3)
        _, _, _, beta, _ = score_slbt(joint, budgets, "none", seed=seed,
                                      n_init=4, optimizer=optimizer)
        ref = int(np.argmax(row.sum(axis=1)))
        for t in range(beta.shape[0]):
            same = ((beta[t] - beta[ref]) ** 2).sum()
            swap = ((beta[t][:, ::-1] - beta[ref]) ** 2).sum()
            assert same <= swap + 1e-12, (seed, t)


def test_equal_loss_can_give_different_splits():
    """Documents a property of the model, not a bug.

    The latent budget model with K = 2 is not identified: distinct (A, B)
    reach the same least-squares optimum, and along that set alpha_{1|i} can
    cross 0.5.  The split rule alpha >= 0.5 is then not determined by the fit
    alone.  n_same_split_starts in the diagnostics measures how often this
    happens; this test pins down that it does happen on a simple table, so
    that a future identification constraint shows up as a change here.
    """
    from sltree._tree.split import score_slbt
    joint, budgets, _ = _budget_tables(1)
    found = {}
    for s in range(40):
        _, S, _, _, d = score_slbt(joint, budgets, "none", seed=s, n_init=1,
                                   optimizer="projected")
        obs = joint.sum(axis=2) > 0
        key = tuple(S[obs].astype(int))
        if key[0] == 0:
            key = tuple(1 - np.array(key))
        found.setdefault(key, set()).add(round(d["loss"], 8))
    best = min(min(v) for v in found.values())
    optimal_splits = [k for k, v in found.items() if best in v]
    assert len(optimal_splits) >= 2


def test_stratum_order_is_numeric_with_ten_or_more_strata():
    """_splitS used string order: stratum 10 sorted before stratum 2."""
    from sltree._tree.split import _splitS
    z = np.array([2, 10, 2, 10])
    x = pd.Series(["a", "a", "b", "b"])
    left, _ = _splitS(x, z, [np.array(["a"]), np.array(["b"])])   # strata 2, 10
    assert left.tolist() == [True, False, False, True]


def test_epsilon_drives_the_suggested_pruning():
    """Eq. 8: a larger epsilon suggests a smaller (or equal) tree."""
    X, y, z = _data(n=900)
    sizes = []
    for eps in (0.001, 0.01, 0.05):
        with _quiet():
            m = SLBT(homogeneity="A", max_depth=5, feats_viewed=3,
                     random_state=2, epsilon=eps).fit(X, y, z)
        cut = m.prune_suggested()
        sizes.append(len(_leaves(m.root)))
        assert cut is None or isinstance(cut, int)
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_report_keeps_shared_thresholds_whole():
    """The CSV report split a shared threshold into its characters."""
    X, y, z = _data()
    with _quiet():
        m = SLBT(homogeneity="A", max_depth=1, feats_viewed=3,
                 random_state=1).fit(X, y, z)
    thr = m.reporter.results.query("node_type == 'Internal'").threshold.iloc[0]
    assert isinstance(thr, list) and all(len(str(v)) == 1 for v in thr)
    assert set(map(str, thr)) <= set("abcdpqruv")


def test_frequency_weighting_changes_only_the_fit():
    """Same data, both weightings: same GPI ranking, possibly different split."""
    X, y, z = _data()
    with _quiet():
        mb = SLBT(homogeneity="none", max_depth=1, feats_viewed=3,
                  random_state=1, ls_weights="budget").fit(X, y, z)
        mf = SLBT(homogeneity="none", max_depth=1, feats_viewed=3,
                  random_state=1, ls_weights="frequency").fit(X, y, z)
    if mb.root.feature == mf.root.feature:
        assert mb.root.gpi == pytest.approx(mf.root.gpi)
    with pytest.raises(ValueError):
        SLBT(ls_weights="counts")
