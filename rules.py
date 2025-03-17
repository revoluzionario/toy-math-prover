# rewriting_rules.py
import sympy

def expand_rule(expr):
    """Expand algebraic products and powers."""
    return sympy.expand(expr)

def factor_rule(expr):
    """Factor the given expression."""
    return sympy.factor(expr)

def simplify_rule(expr):
    """Try a more general simplify on the expression."""
    return sympy.simplify(expr)

def cancel_rule(expr):
    """Cancel common factors in rational expressions."""
    return sympy.radsimp(sympy.cancel(expr))

def pow_simplify_rule(expr):
    """Attempt exponent simplification."""
    return sympy.powsimp(expr, force=True)

ALL_RULES = [
    expand_rule,
    factor_rule,
    simplify_rule,
    cancel_rule,
    pow_simplify_rule,
]

def apply_rule(expr, rule_index):
    """
    Apply a rewriting rule from the ALL_RULES list by index.
    """
    return ALL_RULES[rule_index](expr)
