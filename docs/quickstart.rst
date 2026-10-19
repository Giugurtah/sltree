Quick start
===========

Three tasks, three examples.  Every one of them is exercised by the test suite,
so if the library installs, they run.

Growing a single tree
---------------------

SL-TREE partitions a categorical predictor space, so continuous columns are
discretised first.  :class:`~sltree.Categorizer` does that with k-means on each
column, choosing the number of bins itself when asked to.

.. code-block:: python

   from sltree import Categorizer, PredictabilityTree

   cat = Categorizer(method="elbow", k_max=5, random_state=0)
   X_cat = cat.fit_transform(X).astype(str)

   tree = PredictabilityTree(model="twoStage", max_depth=4, feats_viewed=10)
   tree.fit(X_cat, y)

   tree.predict(X_cat)
   tree.reporter.results        # one row per node: feature, N, impurity, PI, GPI

``model`` selects the splitting algorithm: ``"twoStage"``, ``"twoing"`` for
classification, ``"twoClass"`` for a numeric target.

Stratified data
---------------

When a control factor is expected to change *how* a predictor acts — not just
how much — :class:`~sltree.SLBT` gives each stratum its own split direction.

.. code-block:: python

   from sltree import SLBT, plot_html

   tree = SLBT(homogeneity="none", max_depth=6, feats_viewed=5, random_state=0)
   tree.fit(X_cat, y, x_s=stratum)

   plot_html(tree, "tree.html", visual_pruning=True)

``homogeneity`` controls what is shared across strata.  ``"none"`` and ``"B"``
leave the mixing parameters free and can therefore represent a relationship
that reverses between strata; ``"A"`` and ``"AB"`` impose one direction on all
of them.

.. warning::

   :meth:`~sltree.SLBT.predict` does not take the stratum.  With ``"none"`` or
   ``"B"`` it falls back to the union of the per-stratum thresholds, which
   collapses a reversed split.  For those settings the tree is currently a tool
   for interpretation — the per-node LIFT values in the HTML plot — rather than
   for prediction.

Visual pruning
--------------

``plot_html(..., visual_pruning=True)`` writes an interactive page where each
node sits at a height proportional to its impurity proportional reduction.
Reading it, you pick the node where the gain flattens and pass its id back:

.. code-block:: python

   tree.prune_after_vp(node_id)

Everything above the cutting line collapses into leaves.  A suggested cut is
marked automatically, using the criterion of Iorio et al. (2019), equation 3.7
with :math:`\varepsilon = 0.01`.

Missing values
--------------

.. code-block:: python

   from sltree import BINPI

   imputer = BINPI(ensemble="forest", n_estimators=50, max_depth=2,
                   random_state=0)
   X_filled = imputer.fit_transform(X_cat)

Columns are imputed from the least to the most incomplete, each one joining the
predictor block once filled.

.. note::

   ``max_depth`` defaults to 1, a stump.  That is a deliberate weak learner, but
   a stump makes one binary split and cannot separate more than two categories:
   raise it when the relationships are richer.
