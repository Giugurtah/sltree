SL-TREE
=======

Interpretable tree models grounded in statistical learning theory:
predictability-driven growth, latent budget trees for stratified data,
interactive visual pruning, and non-parametric imputation.

This site is the reference you keep open while writing code.  The methodology,
the derivations and the benchmark live in the paper; what follows is how to
call the library and what each argument does.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   install
   quickstart
   reproducibility

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

The five pillars
----------------

**Predictability-driven growth.**  Splits are chosen by the Global
Predictability Improvement (GPI) and Predictability Improvement (PI) metrics
rather than by impurity alone, through the TWO-STAGE, CART-TWOING and TWO-CLASS
algorithms.

**Mixture-model partitioning.**  Latent Budget Trees, and their simultaneous
variant SLBT, handle multi-class, imbalanced and stratified data.  A stratified
split may point in a different direction in each stratum — see
:class:`~sltree.SLBT`.

**Visual pruning.**  A Dendrogram-like Tree Graph maps the impurity
proportional reduction to branch height, so a human can cut the tree where the
gain stops being worth the complexity.  The construction follows Iorio, Aria,
D'Ambrosio and Siciliano (2019), *Informative trees by visual pruning*,
Expert Systems With Applications 127:228-240.

**Ensembles.**  Random forests and AdaBoost over any SL-TREE base learner.

**Imputation.**  BINPI fills missing values by growing ensembles on the
progressively completed predictor block.

Citation
--------

If SL-TREE contributes to your work, please cite the accompanying paper.  The
software itself is released under the GNU General Public License, version 3 or
later.
