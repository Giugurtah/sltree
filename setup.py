# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
setup.py
========
Custom build hooks that compile the three C shared libraries before
installing the Python package.

Three libraries are built:
  csrc/pitree/       → sltree/_backend/libpitree.{so|dylib|dll}
  csrc/slbt/         → sltree/_backend/libslbt.{so|dylib|dll}
  csrc/categorizer/  → sltree/_preprocessing/_backend/libcategorizer.{so|dylib|dll}

Each subdirectory has its own Makefile with an `install` target that
compiles and copies the library to the correct Python package directory.

Usage
-----
pip install .
pip install -e .   (editable / development mode)
"""

import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop  import develop


# ------------------------------------------------------------------ #
#  Helper: run make install in each C source directory
# ------------------------------------------------------------------ #

HERE = Path(__file__).parent.resolve()

C_DIRS = [
    HERE / "csrc" / "pitree",
    HERE / "csrc" / "slbt",
    HERE / "csrc" / "categorizer",
]


def _build_libs():
    """Compile all shared libraries by invoking `make install` in each csrc subdirectory."""
    for d in C_DIRS:
        if not d.is_dir():
            print(f"[sltree setup] WARNING: C source directory not found: {d}", file=sys.stderr)
            continue
        print(f"[sltree setup] Building C library in {d} ...")
        result = subprocess.run(
            ["make", "install"],
            cwd=str(d),
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"make install failed in {d} (exit code {result.returncode}).\n"
                "Make sure gcc/cc is available and all C sources are present."
            )
        print(f"[sltree setup] ✓  {d.name} built successfully.")


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
