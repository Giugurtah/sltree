# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# Sphinx configuration for the SL-TREE documentation.
#
#     cd docs && make html      # build into docs/_build/html
#
# The API reference is generated from the docstrings, which are already in
# numpydoc format — hence the napoleon extension rather than a rewrite.

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "SL-TREE"
author = "Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano"
copyright = f"{date.today().year}, {author}"

try:                                    # keep the docs in step with the package
    from importlib.metadata import version as _version
    release = _version("sltree")
except Exception:                       # not installed: docs still build
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",          # pull docstrings out of the package
    "sphinx.ext.napoleon",         # understand numpydoc sections
    "sphinx.ext.autosummary",      # the class tables on the API page
    "sphinx.ext.viewcode",         # a [source] link beside every object
    "sphinx.ext.intersphinx",      # cross-link numpy, pandas, sklearn
    "sphinx.ext.mathjax",          # the few formulas in the docstrings
]

autosummary_generate = True
autodoc_member_order = "bysource"       # source order, not alphabetical:
                                        # fit before predict reads better
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,          # the homogeneity strategies document
                                        # their methods on the base class
    "show-inheritance": True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python":  ("https://docs.python.org/3", None),
    "numpy":   ("https://numpy.org/doc/stable/", None),
    "pandas":  ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = f"SL-TREE {version}"
html_static_path = []
html_theme_options = {
    "github_url": "https://github.com/Giugurtah/sltree",
    "show_prev_next": False,
    "navigation_with_keys": False,
}

# Warnings are errors on CI: a broken cross-reference should fail the build
# rather than ship a page with a dead link.
nitpicky = False
