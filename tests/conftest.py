"""
Shared fixtures and the --extended switch.

Two tiers of tests, following rOpenSci standard G5.10: extended tests live in
the same framework as the rest but are off unless asked for.

    pytest                  fast tier only, a few seconds
    pytest --extended       fast tier plus the long-running ones
"""

import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ #
#  The --extended switch (G5.10)
# ------------------------------------------------------------------ #

def pytest_addoption(parser):
    parser.addoption("--extended", action="store_true", default=False,
                     help="also run the long-running tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "extended: long-running; needs --extended")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--extended"):
        return
    skip = pytest.mark.skip(reason="long-running; pass --extended to run")
    for item in items:
        if "extended" in item.keywords:
            item.add_marker(skip)


# ------------------------------------------------------------------ #
#  Data
# ------------------------------------------------------------------ #
#
# Every fixture is built from a fixed seed, so a failure is always
# reproducible from the test name alone.

N = 300


@pytest.fixture(scope="session")
def numeric():
    """Three continuous predictors. Input to the Categorizer."""
    rng = np.random.default_rng(5)
    return pd.DataFrame({f"x{i}": rng.normal(size=N) for i in range(3)})


@pytest.fixture(scope="session")
def X(numeric):
    """The same predictors, discretised deterministically.

    pd.cut with explicit edges rather than the Categorizer: a fixture must not
    depend on the component under test.
    """
    return pd.DataFrame({
        c: pd.cut(numeric[c], [-9, -0.7, 0, 0.7, 9],
                  labels=list("abcd")).astype(str)
        for c in numeric
    })


@pytest.fixture(scope="session")
def y(numeric):
    """Binary target driven mostly by x0 minus x1, with noise."""
    rng = np.random.default_rng(6)
    return pd.Series(np.where(
        numeric.x0 - numeric.x1 + rng.normal(0, 0.4, N) > 0, "A", "B"))


@pytest.fixture(scope="session")
def y_numeric(numeric):
    """Continuous target, for the twoClass model."""
    rng = np.random.default_rng(7)
    return pd.Series(numeric.x0 * 2 + rng.normal(0, 0.3, N))


@pytest.fixture(scope="session")
def strata():
    """Two strata, for the SLBT family."""
    return np.random.default_rng(8).integers(0, 2, N)


@pytest.fixture(scope="session")
def weights():
    """Uniform sample weights, the AdaBoost starting point."""
    return np.ones(N) / N


# A smaller set for the SLBT-heavy tests.  The ALS costs about 3 seconds per
# fit at depth 5 on 300 rows, and the reproducibility tests fit ten times over;
# 120 rows and depth 3 give the same guarantee in a tenth of the time.

M = 120


@pytest.fixture(scope="session")
def X_small():
    rng = np.random.default_rng(11)
    Z = pd.DataFrame({f"x{i}": rng.normal(size=M) for i in range(3)})
    return pd.DataFrame({
        c: pd.cut(Z[c], [-9, -0.7, 0, 0.7, 9], labels=list("abcd")).astype(str)
        for c in Z
    })


@pytest.fixture(scope="session")
def y_small():
    rng = np.random.default_rng(12)
    return pd.Series(rng.choice(["A", "B"], size=M))


@pytest.fixture(scope="session")
def strata_small():
    return np.random.default_rng(13).integers(0, 2, M)


@pytest.fixture(scope="session")
def weights_small():
    return np.ones(M) / M


@pytest.fixture
def X_missing(X):
    """A copy with a tenth of the first column missing, for BINPI."""
    Xm = X.copy()
    Xm.iloc[::10, 0] = np.nan
    return Xm
