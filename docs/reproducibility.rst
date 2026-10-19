Reproducibility
===============

Two components draw random numbers: the k-means++ initialisation inside
:class:`~sltree.Categorizer`, and the alternating least squares that
:class:`~sltree.SLBT` starts from.  Both accept ``random_state``.

.. code-block:: python

   Categorizer(method="elbow", k_max=5, random_state=0)
   SLBT(max_depth=6, random_state=0)
   PredictabilityForest(n_estimators=100, random_state=0)

Following the scikit-learn convention, ``random_state=None`` — the default —
draws fresh entropy at every fit.  Pass an integer whenever a result has to be
repeatable.

Why it matters more here than elsewhere
---------------------------------------

A tree amplifies small differences.  A single ``pi > best_pi`` comparison that
flips changes the split, and every split below it.  Unseeded, ten fits of the
same SLBT on the same data produced ten different trees, with accuracy spread
over five percentage points and leaf counts from 21 to 31.

What is guaranteed
------------------

**The same seed gives the same answer**, on macOS, Linux and Windows alike.
The C backends carry an explicit splitmix64 generator rather than libc
``rand()``, which is implementation-defined and differs between platforms, and
they are compiled with ``-ffp-contract=off`` so that no compiler fuses a
multiply and an add into one rounding step where another would use two.

The claim is tested, not asserted: ``tests/test_cross_platform.py`` compares
whole trees — feature, size, impurity, V, PI and GPI at every node, to twelve
decimal places — against stored snapshots.

**Components do not perturb each other.**  Each generator carries its own
state, so fitting a ``Categorizer`` between two ``SLBT`` fits leaves them
identical.  This was not always true: both backends once shared libc's global
``rand()``.

What is not guaranteed
----------------------

**A different seed gives a different answer**, and sometimes a materially
different one.  The ALS converges to a local optimum that depends on where it
starts.  Fixing the seed makes a result repeatable; it does not make it the
best available one.  Varying ``random_state`` and reporting the spread is the
honest way to describe an SLBT result.

Running the tests
-----------------

.. code-block:: bash

   pytest                  # about a minute
   pytest --extended       # adds the long-running tests
