# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/reporting.py
===================
TreeReporter — collects per-node information during tree growth and
exposes it as a tidy pandas DataFrame.

Works with both PredictabilityTree (no strat_labels / LIFT) and SLBT (with
strat_labels, LIFT values, and the mixing parameters and latent budgets of
each split: columns alpha, beta_left, beta_right).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .base import Node


@dataclass
class NodeRecord:
    """Single row of the tree report."""
    id: int
    node_type: str           # "Internal" | "Leaf"
    feature: Optional[str]
    threshold: Any           # list (PredictabilityTree) or dict {stratum: [values]} (SLBT)
    N: int
    impurity: Optional[float]
    distribution: Any        # list of floats
    labels: Any              # list of label values
    value: Any               # leaf prediction: majority class or numeric mean (twoClass)
    gpi: Optional[float]
    pi: Optional[float]
    gcr: Any                 # None or list of floats
    lift_left: Any           # None (PredictabilityTree) or list/nested list (SLBT)
    lift_right: Any
    lcr: Any = None          # SLBT leaves: {stratum: [LCR per class]} (Eq. 5)
    als_diag: Any = None     # SLBT internal nodes: convergence of the SLBM fit
    alpha: Any = None        # SLBT internal nodes: mixing parameter of the left budget,
                             #   {stratum: {modality: value}} or {modality: value}
    beta_left: Any = None    # SLBT internal nodes: latent budget of the left child,
                             #   {stratum: [per class]} or [per class]
    beta_right: Any = None   # same for the right child


class TreeReporter:
    """
    Accumulates node-level statistics while the tree is grown.

    Parameters
    ----------
    homogeneity : str or None
        Passed through to the report; does not affect logic.
    decimals : int, default 4
        Number of decimal places for rounded floats in the report.
    """

    def __init__(self, homogeneity: Optional[str] = None, decimals: int = 4):
        self.homogeneity = homogeneity
        self.decimals    = decimals
        self._records: List[NodeRecord] = []

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @property
    def results(self) -> pd.DataFrame:
        """Return the accumulated report as a pandas DataFrame."""
        if not self._records:
            return pd.DataFrame(columns=[
                "id", "node_type", "feature", "threshold", "N",
                "impurity", "distribution", "labels",
                "gpi", "pi", "gcr", "lift_left", "lift_right",
                "lcr", "als_diag", "alpha", "beta_left", "beta_right",
            ])
        return pd.DataFrame(asdict(r) for r in self._records)

    def add_node(self, node: Node, is_leaf: bool) -> None:
        """Register a node (internal or leaf) in the report."""
        node_type = "Leaf" if is_leaf else "Internal"

        # distribution & labels
        dist  = self._round_nested(self._to_python(node.distribution))
        labels = self._to_python(node.labels)

        # GCR
        gcr = self._round_nested(self._to_python(getattr(node, "GCR", None)))

        # LIFT
        lift_left, lift_right = self._extract_lift(node)

        # threshold
        threshold = self._extract_threshold(node)

        # Leaf predicted value: round if numeric (twoClass regression),
        # leave as-is for classification (string class label).
        raw_value = node.value if is_leaf else None
        if isinstance(raw_value, float):
            leaf_value = round(raw_value, self.decimals)
        else:
            leaf_value = raw_value

        rec = NodeRecord(
            id          = node.position,
            node_type   = node_type,
            feature     = node.feature if not is_leaf else None,
            threshold   = threshold,
            N           = node.N,
            impurity    = float(node.impurity) if node.impurity is not None else None,
            distribution = dist,
            labels      = labels,
            value       = leaf_value,
            gpi         = float(node.gpi) if node.gpi is not None else None,
            pi          = float(node.pi)  if node.pi  is not None else None,
            gcr         = gcr,
            lift_left   = lift_left,
            lift_right  = lift_right,
            lcr         = self._extract_lcr(node),
            als_diag    = getattr(node, "als_diag", None),
            alpha       = self._round_nested(getattr(node, "alpha", None)),
            beta_left   = self._round_nested(getattr(node, "beta_left", None)),
            beta_right  = self._round_nested(getattr(node, "beta_right", None)),
        )
        self._records.append(rec)

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _extract_threshold(self, node: Node) -> Any:
        thr = node.treshold
        if thr is None:
            return None

        strat_labels = getattr(node, "strat_labels", None)
        stratified = isinstance(thr, list) and all(
            isinstance(v, (np.ndarray, list, tuple)) for v in thr
        )
        if strat_labels is None or not stratified:
            # PredictabilityTree / SLBT homogeneity A or AB — one flat list.
            # (Mapping a flat threshold onto the strata, as earlier versions
            # did, split each value into its characters.)
            if isinstance(thr, (np.ndarray, list, tuple)):
                return self._to_python(list(thr))
            return thr

        # SLBT none / B — mapping  stratum → [values]
        return {str(s): self._to_python(list(v)) for s, v in zip(list(strat_labels), thr)}

    def _extract_lcr(self, node: Node) -> Any:
        lcr = getattr(node, "LCR", None)
        strat_labels = getattr(node, "strat_labels", None)
        if lcr is None or strat_labels is None:
            return None
        return {
            str(s): self._round_nested(self._to_python(v))
            for s, v in zip(list(strat_labels), lcr)
        }

    def _extract_lift(self, node: Node):
        L1 = getattr(node, "LIFT_1", None)
        L2 = getattr(node, "LIFT_2", None)
        return (
            self._round_nested(self._to_python(L1)),
            self._round_nested(self._to_python(L2)),
        )

    @staticmethod
    def _to_python(x: Any) -> Any:
        """Convert numpy types to plain Python."""
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [
                v.tolist() if isinstance(v, np.ndarray)
                else (v.item() if isinstance(v, np.generic) else v)
                for v in x
            ]
        if isinstance(x, np.generic):
            return x.item()
        return x

    def _round_nested(self, x: Any) -> Any:
        """Recursively round floats to ``self.decimals`` places."""
        d = self.decimals
        if x is None:
            return None
        if isinstance(x, (float, int)) and not isinstance(x, bool):
            x = float(x)
            return None if np.isnan(x) else round(x, d)
        if isinstance(x, np.ndarray):
            return [self._round_nested(v) for v in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [self._round_nested(v) for v in x]
        if isinstance(x, dict):
            return {k: self._round_nested(v) for k, v in x.items()}
        return x
