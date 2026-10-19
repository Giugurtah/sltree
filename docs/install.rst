Installation
============

.. code-block:: bash

   pip install sltree

From source
-----------

.. code-block:: bash

   git clone https://github.com/Giugurtah/sltree.git
   cd sltree
   pip install .

Installing from source compiles three small C libraries, so a C compiler must
be available.  ``pip install`` invokes ``build_c.py``, which picks the compiler
and the platform flags on its own — there is no ``make`` step.

Requirements
------------

Python 3.10 or newer, with NumPy 1.23+ and pandas 1.5+.

A compiler is needed only when building from source:

======================  ==========================================================
macOS                   ``xcode-select --install``
Linux                   ``sudo apt install build-essential``
Windows                 `MSYS2 <https://www.msys2.org>`_, then
                        ``pacman -S mingw-w64-x86_64-gcc``
======================  ==========================================================

.. note::

   On Windows the compiler must be MinGW-w64, not MSVC.  The C sources use
   C99 variable-length arrays, which MSVC has never supported.  This is not a
   limitation for users installing from PyPI: those wheels ship the libraries
   already built.

Verifying the installation
--------------------------

.. code-block:: python

   import sltree
   print(sltree.__version__, len(sltree.__all__), "public classes")

The test suite runs in under a minute:

.. code-block:: bash

   pip install pytest
   pytest                  # the fast tier
   pytest --extended       # also the long-running tests
