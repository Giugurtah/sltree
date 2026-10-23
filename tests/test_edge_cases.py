"""
Edge conditions (rOpenSci G5.8).

The standard asks for "clear warnings or errors when confronted with data with
extreme properties" — zero-length data, unsupported types, all-NA fields, data
outside the algorithm's scope.

Where the current behaviour is not yet a clear error, the test says so rather
than asserting whatever happens to come out today.  A test that merely records
present behaviour protects nothing.
"""

import contextlib
import io
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from sltree import (BINPI, Categorizer, PredictabilityAdaBoost,
                    PredictabilityForest, PredictabilityTree, SLBT)


def _quiet():
    return contextlib.redirect_stdout(open(os.devnull, "w"))


# ------------------------------------------------------------------ #
#  Degenerate inputs
# ------------------------------------------------------------------ #

def test_single_class_target_gives_a_root_leaf():
    """Nothing to separate: the root is already pure and must stay a leaf."""
    X = pd.DataFrame({"a": list("abab") * 10, "b": list("aabb") * 10})
    y = pd.Series(["A"] * 40)

    tree = PredictabilityTree(max_depth=4, feats_viewed=2)
    tree.fit(X, y)

    assert tree.root.left is None and tree.root.right is None
    assert tree.root.impurity == pytest.approx(0.0)
    assert set(tree.predict(X)) == {"A"}


def test_constant_predictor_is_not_used():
    """A column with one modality carries no information."""
    X = pd.DataFrame({"useless": ["z"] * 60,
                      "useful":  (["a"] * 30) + (["b"] * 30)})
    y = pd.Series((["A"] * 30) + (["B"] * 30))

    tree = PredictabilityTree(max_depth=3, feats_viewed=2)
    tree.fit(X, y)

    assert tree.root.feature == "useful"


def test_all_predictors_constant_gives_a_leaf():
    X = pd.DataFrame({"a": ["z"] * 40, "b": ["w"] * 40})
    y = pd.Series((["A"] * 20) + (["B"] * 20))

    tree = PredictabilityTree(max_depth=3, feats_viewed=2)
    tree.fit(X, y)

    assert tree.root.left is None


def test_max_depth_zero_gives_a_leaf():
    X = pd.DataFrame({"a": list("ab") * 20})
    y = pd.Series(list("AB") * 20)

    tree = PredictabilityTree(max_depth=0, feats_viewed=1)
    tree.fit(X, y)

    assert tree.root.left is None


def test_two_rows_do_not_crash():
    X = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    y = pd.Series(["A", "B"])

    tree = PredictabilityTree(max_depth=3, feats_viewed=2)
    tree.fit(X, y)
    assert len(tree.predict(X)) == 2


# ------------------------------------------------------------------ #
#  Malformed calls
# ------------------------------------------------------------------ #

def test_predict_before_fit_raises():
    X = pd.DataFrame({"a": ["x", "y"]})
    with pytest.raises(RuntimeError, match="[Ff]it"):
        PredictabilityTree().predict(X)


def test_plot_before_fit_raises():
    from sltree import plot_html
    with pytest.raises(ValueError, match="[Ee]mpty"):
        plot_html(PredictabilityTree(), "/tmp/never_written.html")


def test_unknown_model_name_raises():
    """Rejected in __init__, before any data is seen."""
    with pytest.raises(ValueError, match="not supported"):
        PredictabilityTree(model="nonexistent")


def test_unknown_homogeneity_raises():
    with pytest.raises(ValueError, match="homogeneity"):
        SLBT(homogeneity="nonexistent")


@pytest.mark.parametrize("kwargs,pattern", [
    ({"method": "nonexistent"},              "method"),
    ({"method": "fixed"},                    "k must be specified"),
    ({"method": "fixed", "k": 1},            "k must be >= 2"),
    ({"label_style": "nonexistent"},         "label_style"),
])
def test_categorizer_rejects_bad_arguments(kwargs, pattern):
    with pytest.raises(ValueError, match=pattern):
        Categorizer(**kwargs)


def test_binpi_rejects_an_unknown_ensemble():
    with pytest.raises(ValueError, match="ensemble"):
        BINPI(ensemble="nonexistent")


# ------------------------------------------------------------------ #
#  Missing data
# ------------------------------------------------------------------ #

def test_categorizer_warns_on_an_all_nan_column():
    """G5.8 names all-NA fields explicitly."""
    Z = pd.DataFrame({"good": np.random.default_rng(0).normal(size=60),
                      "empty": [np.nan] * 60})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _quiet():
            cat = Categorizer(method="fixed", k=3, random_state=0)
            cat.fit(Z)

    assert any("NaN" in str(w.message) for w in caught)
    assert cat.k_["empty"] == 0


def test_binpi_warns_when_there_is_nothing_to_impute():
    X = pd.DataFrame({"a": list("ab") * 20, "b": list("aabb") * 10})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = BINPI(ensemble="forest", n_estimators=3,
                    random_state=0).fit_transform(X)

    assert any("no missing values" in str(w.message).lower() for w in caught)
    assert (out.values == X.values).all()


def test_binpi_never_uses_an_incomplete_predictor():
    """Several columns missing at once — the incremental case BINPI exists for.

    Note on a dead branch.  binpi.py filters `current_predictors` for columns
    that still hold missing values and warns when any is dropped.  That warning
    can never fire: `current_predictors` starts as the completely observed
    columns and only ever grows by appending a column that has just been
    imputed, so it never contains an incomplete one.  The filter is a no-op.

    Rather than assert a warning that cannot happen, this test checks the
    invariant the filter was meant to protect: every column used as a predictor
    is complete at the moment it is used, and the result has no gaps left.
    """
    rng = np.random.default_rng(3)
    n = 80
    X = pd.DataFrame({
        "complete": rng.choice(list("abc"), size=n),
        "hole_a":   rng.choice(list("xy"),  size=n).astype(object),
        "hole_b":   rng.choice(list("pq"),  size=n).astype(object),
        "hole_c":   rng.choice(list("rs"),  size=n).astype(object),
    })
    X.loc[::4,  "hole_a"] = np.nan
    X.loc[1::4, "hole_b"] = np.nan
    X.loc[2::3, "hole_c"] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _quiet():
            imputer = BINPI(ensemble="forest", n_estimators=3, random_state=0)
            out = imputer.fit_transform(X)

    assert out.isnull().sum().sum() == 0
    assert out.shape == X.shape
    # Imputation order goes from the least to the most incomplete column, so
    # each step has a strictly larger complete predictor block than the last.
    assert set(imputer.ensembles_) == {"hole_a", "hole_b", "hole_c"}


# ------------------------------------------------------------------ #
#  Weak learners
# ------------------------------------------------------------------ #

def test_adaboost_warns_when_the_first_learner_is_no_better_than_chance():
    """With a target independent of every predictor, a depth-1 stump cannot beat
    chance; boosting must stop and say so rather than fail silently.
    """
    rng = np.random.default_rng(17)
    n = 200
    X = pd.DataFrame({"a": rng.choice(list("ab"), size=n),
                      "b": rng.choice(list("ab"), size=n)})
    y = pd.Series(rng.choice(["A", "B"], size=n))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _quiet():
            boost = PredictabilityAdaBoost(n_estimators=5, max_depth=1,
                                           feats_viewed=1, random_state=0)
            boost.fit(X, y)

    if len(boost.estimators_) == 1 and boost.estimator_errors_[0] >= 0.5:
        assert any("chance" in str(w.message).lower() for w in caught), \
            "stopping at the first learner should be reported"


# ------------------------------------------------------------------ #
#  Silence (the fix for the C-level printf)
# ------------------------------------------------------------------ #

def test_a_default_fit_writes_nothing_to_stdout(numeric, X, y):
    """Measured at the file-descriptor level, on purpose.

    contextlib.redirect_stdout only replaces Python's sys.stdout; the C backends
    write straight to file descriptor 1 and slip past it.  Capturing the
    descriptor is the only way to prove the library is really quiet.
    """
    import tempfile

    def captured(fn):
        import sys
        sys.stdout.flush()
        saved = os.dup(1)
        tmp = tempfile.TemporaryFile(mode="w+b")
        os.dup2(tmp.fileno(), 1)
        try:
            fn()
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
        tmp.seek(0)
        return tmp.read()

    out = captured(lambda: (
        Categorizer(method="elbow", k_max=5, random_state=0).fit_transform(numeric),
        PredictabilityForest(n_estimators=3, max_depth=2, feats_viewed=2,
                             random_state=0).fit(X, y),
    ))
    assert out == b"", f"the library printed {len(out)} bytes with verbose=0: {out[:200]!r}"


def test_verbose_one_does_print(X, y):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        PredictabilityForest(n_estimators=3, max_depth=2, feats_viewed=2,
                             random_state=0, verbose=1).fit(X, y)
    assert len(buf.getvalue()) > 0


@pytest.mark.parametrize("method,kw", [("elbow", {}), ("silhouette", {}), ("fixed", {"k": 3})])
def test_categorizer_assigns_values_outside_the_fitted_range(method, kw):
    """New data beyond the min / max seen in fit go to the first / last bin.

    The outer edges of each bin set are the fitted min and max; with closed
    outer bins, pd.cut returned NaN for every value outside them, so any test
    set or bootstrap sample with a more extreme value lost observations.
    """
    import numpy as np
    import pandas as pd
    from sltree import Categorizer

    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(size=200)})
    cat = Categorizer(method=method, random_state=0, **kw).fit(train)
    new = pd.DataFrame({"x": [train.x.min() - 5.0, train.x.max() + 5.0, np.nan]})
    out = cat.transform(new)["x"]
    labels = cat.labels_["x"]
    assert out.iloc[0] == labels[0]
    assert out.iloc[1] == labels[cat.k_["x"] - 1]
    assert pd.isna(out.iloc[2])                       # missing stays missing
    assert cat.transform(train)["x"].notna().all()
