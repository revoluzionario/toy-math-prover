"""This file contains the `TreeNode` class and helper functions for converting a Sympy expression into a TreeNode."""

from typing import List
from sympy import AtomicExpr, sympify
from torch import Tensor


class TreeNode:
    """The `TreeNode` class is used for representing a Sympy expression in a neural network pipeline.

    Attributes
    ----------
    expr : AtomicExpr
        The Sympy expression for this node.

    children : List[TreeNode]
        List of child TreeNodes (derived from `expr.args`). By default, it is an empty list.

    embedding : Tensor
        A torch Tensor assigned later by an embedding function. By default, it is `None`.
    """

    # Define the attributes of the `TreeNode` class.
    __slots__ = ("expr", "children", "embedding")

    def __init__(self, expr: AtomicExpr):
        """Initialize an instance of the `TreeNode` class.

        Parameters
        ----------
        expr : AtomicExpr
            The Sympy expression for this node.
        """
        self.expr: AtomicExpr = expr
        self.children: List[TreeNode] = []
        self.embedding: Tensor = None


def sympy_expr_to_tree(expr: AtomicExpr) -> TreeNode:
    """
    Recursively build a TreeNode hierarchy from a Sympy expression.

    Parameters
    ----------
    expr : AtomicExpr
        A Sympy expression.

    Returns
    -------
    node: TreeNode
        A TreeNode representing the Sympy expression.
    """
    # Initialize the node with the expression.
    node: TreeNode = TreeNode(expr)

    # Recursively build the children nodes.
    for arg in expr.args:
        child: TreeNode = sympy_expr_to_tree(arg)
        node.children.append(child)

    # Return the node.
    return node


def parse_sympy_expression(expression_str: str) -> TreeNode:
    """
    Parse an expression string into a Sympy expression, then convert it into a TreeNode.

    Parameters
    ----------
    expression_str : str
        A string representing a Sympy expression.

    Returns
    -------
    node: TreeNode
        A TreeNode representing the Sympy expression.

    Notes
    -----
    Not to mistake `sympify` with `simplify`; both are Sympy functions but serve different purposes.
    """
    # Parse the expression string into a Sympy expression.
    expr: AtomicExpr = sympify(expression_str)

    # Convert the Sympy expression into a TreeNode.
    node: TreeNode = sympy_expr_to_tree(expr)

    # Return the TreeNode.
    return node
