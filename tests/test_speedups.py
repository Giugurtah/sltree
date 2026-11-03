"""
The speed-ups must not change a single number.

* the integer coding of the data and the single table per (node, predictor)
  (``_utils.coding``) give the same tables as the functions on the original
  values they replace;
* the seed of a least-squares fit depends only on (fit, node, predictor), so
  ``FAST`` and ``n_jobs`` leave the tree unchanged;
* the starts run one per C call (in parallel, or for tie_break="pi") are the
  starts of the C multistart.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from sltree import SLBT, SLBTWeighted, PredictabilityTree, PredictabilityTreeWeighted
from sltree._utils.coding import encode, encode_frame, NodeCodes, row_conditional_counts
from sltree._utils.utils import _joint_table, _contingency_matrix
from sltree._tree.split import score_slbt, start_seeds, als_seed, _HOMOGENEITY_MAP
from sltree._backend._slbt import slba


# ------------------------------------------------------------------ #
#  helpers
# ------------------------------------------------------------------ #

def _signature(model):
    out = []

    def walk(n):
        if n is None:
            return
        out.append((n.position, n.feature, None if n.treshold is None else str(n.treshold),
                    n.pi, n.gpi, n.N, str(n.value), tuple(np.atleast_1d(n.distribution)),
                    n.impurity, n.impurity_decrease))
        walk(n.left)
        walk(n.right)

    walk(model.root)
    return out


def _data(seed, n=400, p=5, G=3, J=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{c}": rng.choice(list("abcdefg")[: 2 + c], size=n) for c in range(p)})
    lin = (X.x0 == "a").astype(float) - (X.x1 == "b") + rng.normal(0, 1, n)
    y = pd.Series(np.digitize(lin, np.quantile(lin, np.linspace(0, 1, J + 1)[1:-1]))).map(
        lambda v: f"c{v}")
    z = rng.integers(0, G, n)
    return X, y, z


# ------------------------------------------------------------------ #
#  tables
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("seed", range(6))
def test_coded_tables_equal_the_tables_on_the_values(seed):
    X, y, z = _data(seed)
    rng = np.random.default_rng(seed)
    w = rng.gamma(2.0, size=len(y))
    w[::7] = 0.0
    C, sizes = encode_frame(X)
    yc, J = encode(y)
    sc, G = encode(z)
    for _ in range(5):                          # random "nodes"
        rows = np.flatnonzero(rng.random(len(y)) < rng.uniform(0.2, 1.0))
        nc = NodeCodes(yc[rows], J, sc[rows], G)
        ncw = NodeCodes(yc[rows], J, sc[rows], G, w[rows])
        for c, col in enumerate(X.columns):
            x = X[col].iloc[rows]
            assert np.array_equal(nc.joint(C[rows, c], sizes[c]),
                                  _joint_table(x, y.iloc[rows], z[rows]))
            assert np.array_equal(ncw.joint(C[rows, c], sizes[c]),
                                  _joint_table(x, y.iloc[rows], z[rows], w[rows]))
            assert np.array_equal(row_conditional_counts(nc.counts(C[rows, c], sizes[c])),
                                  _contingency_matrix(x, y.iloc[rows]))


# ------------------------------------------------------------------ #
#  seeds, FAST, n_jobs
# ------------------------------------------------------------------ #

def test_als_seed_depends_only_on_fit_node_feature():
    assert als_seed(7, 5, 2) == als_seed(7, 5, 2)
    seeds = {als_seed(7, t, c) for t in (1, 2, 3, 2 ** 70) for c in range(4)}
    assert len(seeds) == 16
    assert als_seed(7, 5, 2) != als_seed(8, 5, 2)


@pytest.mark.parametrize("homogeneity", ["none", "A", "B", "AB"])
@pytest.mark.parametrize("cfg", [dict(), dict(n_init=4, optimizer="projected", tie_break="pi")])
def test_fast_does_not_change_the_slbt(homogeneity, cfg):
    X, y, z = _data(3, n=300)
    kw = dict(homogeneity=homogeneity, max_depth=3, feats_viewed=5, random_state=11, **cfg)
    slow = SLBT(FAST=False, **kw).fit(X, y, z)
    fast = SLBT(FAST=True, **kw).fit(X, y, z)
    assert _signature(slow) == _signature(fast)
    assert slow.convergence_ == fast.convergence_


@pytest.mark.parametrize("model", ["twoStage", "twoing"])
def test_fast_does_not_change_the_predictability_tree(model):
    X, y, _ = _data(4)
    slow = PredictabilityTree(model=model, max_depth=5, FAST=False, feats_viewed=5).fit(X, y)
    fast = PredictabilityTree(model=model, max_depth=5, FAST=True, feats_viewed=5).fit(X, y)
    assert _signature(slow) == _signature(fast)
    w = np.random.default_rng(1).gamma(2.0, size=len(y))
    slow = PredictabilityTreeWeighted(model=model, max_depth=4, FAST=False,
                                      feats_viewed=5).fit(X, y, sample_weight=w)
    fast = PredictabilityTreeWeighted(model=model, max_depth=4, FAST=True,
                                      feats_viewed=5).fit(X, y, sample_weight=w)
    assert _signature(slow) == _signature(fast)


@pytest.mark.parametrize("tie_break", ["loss", "pi"])
@pytest.mark.parametrize("homogeneity", ["none", "A"])
def test_n_jobs_does_not_change_the_slbt(tie_break, homogeneity):
    X, y, z = _data(5, n=300)
    kw = dict(homogeneity=homogeneity, max_depth=3, feats_viewed=5, random_state=2,
              n_init=5, optimizer="projected", tie_break=tie_break)
    one = SLBT(n_jobs=1, **kw).fit(X, y, z)
    many = SLBT(n_jobs=3, **kw).fit(X, y, z)
    assert _signature(one) == _signature(many)
    assert one.convergence_ == many.convergence_
    assert many._executor is None                    # the pool is closed after fit

    w = np.random.default_rng(2).gamma(2.0, size=len(y))
    one = SLBTWeighted(n_jobs=1, **kw).fit(X, y, z, sample_weight=w)
    many = SLBTWeighted(n_jobs=3, **kw).fit(X, y, z, sample_weight=w)
    assert _signature(one) == _signature(many)


def test_refitting_grows_the_same_tree():
    X, y, z = _data(6, n=300)
    m = SLBT(homogeneity="A", max_depth=3, feats_viewed=5, random_state=9)
    first = _signature(m.fit(X, y, z))
    assert _signature(m.fit(X, y, z)) == first


def test_bad_n_jobs_is_rejected():
    with pytest.raises(ValueError):
        SLBT(n_jobs=0)


# ------------------------------------------------------------------ #
#  starts
# ------------------------------------------------------------------ #

def _random_tables(rng):
    K, I, J = int(rng.integers(1, 4)), int(rng.integers(2, 6)), int(rng.integers(2, 5))
    C = rng.integers(0, 25, size=(K, I, J)).astype(float)
    C[rng.random((K, I)) < 0.15] = 0
    C[0, 0, 0] += 1
    F = C / C.sum()
    row = F.sum(axis=2, keepdims=True)
    return F, np.divide(F, row, out=np.zeros_like(F), where=row > 0)


@pytest.mark.parametrize("seed", range(4))
def test_start_seeds_reproduce_the_c_multistart(seed):
    rng = np.random.default_rng(seed)
    for _ in range(10):
        F, Fs = _random_tables(rng)
        K, I, J = F.shape
        for hom in ("none", "A", "B", "AB"):
            KA, KB = _HOMOGENEITY_MAP[hom](K)
            s0 = int(rng.integers(0, 2 ** 63))
            full = slba(K, KA, KB, I, J, F.ravel(), Fs.ravel(), s0, n_init=5, max_iter=300)
            runs = [slba(K, KA, KB, I, J, F.ravel(), Fs.ravel(), sd, n_init=1, max_iter=300)
                    for sd in start_seeds(s0, 5, KA, KB, I, J)]
            losses = [r[4]["loss"] for r in runs]
            assert max(losses) == full[4]["loss_worst_start"]
            # the C rule: a later start wins only if lower by more than 1e-15
            best = 0
            for s, l in enumerate(losses):
                if l < losses[best] - 1e-15:
                    best = s
            assert losses[best] == full[4]["loss"]
            assert np.array_equal(runs[best][1], full[1])
            assert np.array_equal(runs[best][2], full[2])


@pytest.mark.parametrize("tie_break", ["loss", "pi"])
def test_parallel_starts_equal_sequential_starts(tie_break):
    rng = np.random.default_rng(10)
    with ThreadPoolExecutor(3) as ex:
        for _ in range(15):
            F, Fs = _random_tables(rng)
            for hom in ("none", "A", "B", "AB"):
                kw = dict(seed=int(rng.integers(0, 2 ** 63)), n_init=6, max_iter=300,
                          optimizer="projected", tie_break=tie_break, ls_weights="frequency")
                a = score_slbt(F, Fs, hom, **kw)
                b = score_slbt(F, Fs, hom, executor=ex, **kw)
                assert a[0] == b[0] and a[4] == b[4]
                for u, v in zip(a[1:4], b[1:4]):
                    assert np.array_equal(u, v)
