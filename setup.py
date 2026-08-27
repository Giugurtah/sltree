"""
setup.py
========
Custom build hooks that compile the three C shared libraries before
installing the Python package.

The actual compilation lives in build_c.py, a plain Python script that works
the same on macOS, Linux and Windows.  It replaced three Makefiles, which
needed `make` and `uname` and therefore could not run on Windows at all.

Usage
-----
pip install .
pip install -e .   (editable / development mode)
"""

import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop  import develop

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))


def _build_libs():
    """Compile the three C backends by delegating to build_c.py."""
    import build_c
    build_c.build_all()


# ------------------------------------------------------------------ #
#  Custom commands
#
#  build_py  — always invoked by `pip install .` (pure-Python packages)
#  develop   — invoked by `pip install -e .`
# ------------------------------------------------------------------ #

class BuildPyWithLibs(build_py):
    """Compile C libraries then run the normal Python build step."""
    def run(self):
        _build_libs()
        super().run()


class DevelopWithLibs(develop):
    """Compile C libraries then run the normal editable-install step."""
    def run(self):
        _build_libs()
        super().run()


# ------------------------------------------------------------------ #
#  Setup call
# ------------------------------------------------------------------ #

setup(
    cmdclass={
        "build_py": BuildPyWithLibs,
        "develop":  DevelopWithLibs,
    },
)
