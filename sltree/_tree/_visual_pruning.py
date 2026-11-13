# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Cristian Buoncompagni, Giulia Vannucci, Roberta Siciliano
#
# This file is part of SL-TREE. SL-TREE is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""
sltree/_tree/_visual_pruning.py
===============================
Visual pruning and decision-tree selection (Eqs. 6-8 of the SLBT paper),
shared by every tree of the library.

Node quantities (set while the tree grows)
------------------------------------------
``impurity_decrease`` is V_Y(t) of Eq. 6:

    V_Y(t) = (i_Y(1) - i_Y(t)) / i_Y(1)

the proportional reduction of the impurity of Y from the root to node t.  In
the stratified trees (SLBT) i_Y(t) is the impurity of Y within the strata,

    i_Y(t) = sum_g p(g | t) i_{Y|(g)}(t),

the quantity whose reduction the split criterion measures (Eq. 2).  V_Y(t)
can be negative, and is not monotone along a path: a child can be less pure
than its parent.

The sequence of trees
---------------------
The visual-pruning canvas draws every node at a height that does not
decrease from parent to child,

    h(1) = 0,   h(t) = max(V_Y(t), h(parent of t)),

and the cut is horizontal.  Expanding the internal nodes by increasing
height (ties by position, so a parent always comes before its children)
gives the nested trees T_0 (the root alone), T_1, T_2, ...  For each,

    V_Y(T_nu) = sum over the leaves h of T_nu of V_Y(h) p(h)        (Eq. 7)

and the suggested tree is T_{nu-1} for the first nu with

    V_Y(T_nu) - V_Y(T_{nu-1}) <= epsilon                             (Eq. 8)

i.e. the split that turns T_{nu-1} into T_nu is not worth keeping.  The
comparison starts from T_1 against T_0, so the split of the root can be
pruned too.

Earlier versions (i) put the share p(t) of the node inside V_Y(t) as well as
in Eq. 7, so V_Y(T) grew with the number of leaves even without any gain in
purity; (ii) used the marginal impurity of Y, under which an effect that
reverses between strata looks like no effect; (iii) kept T_nu instead of
T_{nu-1}, and never compared T_1 with T_0.
"""

from __future__ import annotations


def _children(node):
    return [c for c in (node.left, node.right) if c is not None]


def set_heights(root) -> None:
    """Height of every node on the canvas (non-decreasing along each path)."""
    root.vp_height = 0.0
    stack = [root]
    while stack:
        node = stack.pop()
        for c in _children(node):
            c.vp_height = max(float(c.impurity_decrease), node.vp_height)
            stack.append(c)


def expansion_order(root) -> list:
    """Internal nodes by increasing canvas height, ties by position."""
    nodes, stack = [], [root]
    while stack:
        node = stack.pop()
        if node.left is not None:
            nodes.append(node)
            stack.extend(_children(node))
    return sorted(nodes, key=lambda n: (n.vp_height, n.position))


def _share(node, root):
    w = getattr(node, "mass", None)
    return float(w) if w is not None else node.N / root.N


def visual_pruning_path(root, epsilon: float):
    """
    Compute V_Y(T) along the sequence of trees and flag the suggested cut.

    Sets, on every node, ``vp_height`` and ``tree_partial_impurity_reduction``
    (V_Y of the smallest tree in the sequence in which the node is expanded,
    or, for a leaf, in which it first appears), and ``suggested_pruning =
    True`` on the internal node whose split is the first one with a gain of
    at most ``epsilon``: the suggested tree is the one before that split.
    Returns that node, or None when every split gains more than epsilon.
    """
    if root is None:
        return None
    set_heights(root)
    stack = [root]
    while stack:                                   # clear previous flags
        node = stack.pop()
        node.suggested_pruning = None
        stack.extend(_children(node))

    root.tree_partial_impurity_reduction = 0.0
    frontier = {root}
    v_prev = float(root.impurity_decrease) * _share(root, root)    # V_Y(T_0) = 0
    flagged = None
    for node in expansion_order(root):
        frontier.discard(node)
        frontier.update(_children(node))
        v_new = sum(float(h.impurity_decrease) * _share(h, root) for h in frontier)
        node.tree_partial_impurity_reduction = v_new
        for c in _children(node):
            c.tree_partial_impurity_reduction = v_new
        if flagged is None and v_new - v_prev <= epsilon:
            flagged = node
            node.suggested_pruning = True
        v_prev = v_new
    return flagged


def cut_at(root, node_id) -> list:
    """
    Expand the nodes in the order of the sequence until ``node_id`` is
    reached; return the internal nodes left unexpanded at that point (to be
    turned into leaves by the caller).
    """
    if getattr(root, "vp_height", None) is None:
        set_heights(root)
    expanded = set()
    for node in expansion_order(root):
        if node.position == node_id:
            break
        expanded.add(id(node))
    out, stack = [], [root]
    while stack:
        node = stack.pop()
        if node.left is None:
            continue
        if id(node) in expanded:
            stack.extend(_children(node))
        else:
            out.append(node)
    return out
