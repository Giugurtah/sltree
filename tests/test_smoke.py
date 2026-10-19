"""
Smoke tests: every public class fits, predicts and reports.

The floor of the suite. These do not check that answers are *right* — that is
test_correctness.py — only that nothing is broken end to end.  A failure here
usually means the C backends did not build or an import path is wrong.
"""

import numpy as np
import pandas as pd
import pytest

from sltree import (BINPI, Categorizer, PredictabilityAdaBoost,
                    PredictabilityForest, PredictabilityTree,
                    PredictabilityTreeWeighted, SLBT, SLBTAdaBoostForest,
                    SLBTForest, SLBTWeighted, plot_html)


# ------------------------------------------------------------------ #
#  Single trees
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("model", ["twoStage", "twoing"])
def test_predictability_tree_fits_and_predicts(X, y, model):
    tree = PredictabilityTree(model=model, max_depth=4, feats_viewed=3)
    tree.fit(X, y)
    preds = tree.predict(X)

    assert len(preds) == len(y)
    assert set(preds) <= set(y.unique())
    assert tree.root is not None


def test_two_class_accepts_a_numeric_target(X, y_numeric):
    """twoClass binarises y locally at every node; the leaves hold means."""
    tree = PredictabilityTree(model="twoClass", max_depth=4, feats_viewed=3)
    tree.fit(X, y_numeric)
    preds = tree.predict(X)

    assert len(preds) == len(y_numeric)
    assert np.isfinite(np.asarray(preds, dtype=float)).all()


def test_weighted_tree_fits(X, y, weights):
    tree = PredictabilityTreeWeighted(max_depth=4, feats_viewed=3)
    tree.fit(X, y, weights)
    assert len(tree.predict(X)) == len(y)


@pytest.mark.parametrize("homogeneity", ["none", "A", "B", "AB"])
def test_slbt_fits_under_every_homogeneity(X, y, strata, homogeneity):
    tree = SLBT(homogeneity=homogeneity, max_depth=4, feats_viewed=3,
                random_state=0)
    tree.fit(X, y, strata)
    assert len(tree.predict(X, x_s=strata)) == len(y)
    assert tree.predict_proba(X, x_s=strata).shape == (len(y), len(tree.classes_))


def test_slbt_weighted_fits(X, y, strata, weights):
    tree = SLBTWeighted(max_depth=4, feats_viewed=3, random_state=0)
    tree.fit(X, y, strata, weights)
    assert len(tree.predict(X, x_s=strata)) == len(y)


# ------------------------------------------------------------------ #
#  Ensembles
# ------------------------------------------------------------------ #

def test_predictability_forest_fits(X, y):
    forest = PredictabilityForest(n_estimators=4, max_depth=3, feats_viewed=2,
                                  random_state=0)
    forest.fit(X, y)
    assert len(forest.predict(X)) == len(y)
    assert len(forest.estimators_) == 4


def test_predictability_adaboost_fits(X, y):
    boost = PredictabilityAdaBoost(n_estimators=4, max_depth=3, feats_viewed=2,
                                   random_state=0)
    boost.fit(X, y)
    assert len(boost.predict(X)) == len(y)
    assert len(boost.estimators_) <= 4      # may stop early


@pytest.mark.parametrize("sampling", ["tree", "node"])
def test_slbt_forest_fits_on_both_branches(X, y, strata, sampling):
    """The two feature-sampling branches take different code paths.

    Only the 'node' branch used to receive a seeded generator; the 'tree'
    branch built its trees without one and was irreproducible.
    """
    kwargs = ({"feature_sampling": "tree"} if sampling == "tree"
              else {"node_max_features": 2})
    forest = SLBTForest(n_estimators=4, max_depth=3, feats_viewed=2,
                        random_state=0, **kwargs)
    forest.fit(X, y, strata)
    assert len(forest.predict(X, x_s=strata)) == len(y)


def test_slbt_adaboost_fits(X, y, strata):
    boost = SLBTAdaBoostForest(n_estimators=4, max_depth=3, feats_viewed=2,
                               random_state=0)
    boost.fit(X, y, strata)
    assert len(boost.predict(X, x_s=strata)) == len(y)


# ------------------------------------------------------------------ #
#  Preprocessing and imputation
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("method,kwargs", [
    ("fixed",      {"k": 3}),
    ("elbow",      {"k_max": 5}),
    ("silhouette", {"k_max": 5}),
])
def test_categorizer_discretises(numeric, method, kwargs):
    cat = Categorizer(method=method, random_state=0, **kwargs)
    out = cat.fit_transform(numeric)

    assert out.shape == numeric.shape
    assert list(out.columns) == list(numeric.columns)
    for col in out:
        assert out[col].nunique() >= 2


def test_binpi_fills_every_missing_value(X_missing):
    imputed = BINPI(ensemble="forest", n_estimators=4, random_state=0) \
        .fit_transform(X_missing)

    assert imputed.shape == X_missing.shape
    assert imputed.isnull().sum().sum() == 0


# ------------------------------------------------------------------ #
#  Reporting and plotting
# ------------------------------------------------------------------ #

def test_report_is_a_dataframe_with_one_row_per_node(X, y):
    tree = PredictabilityTree(max_depth=4, feats_viewed=3)
    tree.fit(X, y)
    report = tree.reporter.results

    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert {"id", "node_type", "impurity", "N"} <= set(report.columns)
    assert set(report["node_type"]) <= {"Internal", "Leaf"}


@pytest.mark.parametrize("visual_pruning", [False, True])
def test_plot_html_writes_a_file(X, y, tmp_path, visual_pruning):
    tree = PredictabilityTree(max_depth=4, feats_viewed=3)
    tree.fit(X, y)
    out = tmp_path / "tree.html"
    plot_html(tree, str(out), visual_pruning=visual_pruning)

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "treeData" in text and "plotData" in text


def test_prune_after_vp_shrinks_the_tree(X, y):
    tree = PredictabilityTree(max_depth=6, feats_viewed=3)
    tree.fit(X, y)
    before = len(tree.reporter.results)

    internal = [n for n in _walk(tree.root) if n.left is not None]
    target = sorted(internal, key=lambda n: n.impurity_decrease)[len(internal) // 2]
    tree.prune_after_vp(target.position)

    assert len(tree.reporter.results) < before


def _walk(node):
    if node is None:
        return
    yield node
    yield from _walk(node.left)
    yield from _walk(node.right)
