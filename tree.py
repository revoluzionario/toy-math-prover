# symbolic_tree.py
import sympy

class TreeNode:
    """
    Tree node for representing a Sympy expression in a neural network pipeline.
    Each node has:
      - expr: the Sympy expression for this node
      - children: list of child TreeNodes (derived from expr.args)
      - embedding: a torch Tensor assigned later by an embedding function
    """
    __slots__ = ('expr', 'children', 'embedding')

    def __init__(self, expr):
        self.expr = expr
        self.children = []
        self.embedding = None

def sympy_expr_to_tree(expr):
    """
    Recursively build a TreeNode hierarchy from a Sympy expression.
    """
    node = TreeNode(expr)
    for arg in expr.args:
        child = sympy_expr_to_tree(arg)
        node.children.append(child)
    return node

def parse_sympy_expression(expression_str):
    """
    Convert string -> Sympy expression -> TreeNode.
    """
    expr = sympy.sympify(expression_str)
    return sympy_expr_to_tree(expr)
