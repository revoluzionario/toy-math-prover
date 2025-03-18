# environment.py
import sympy
from rules import apply_rule, ALL_RULES

class ProofEnvironment:
    """
    Environment for rewriting a start expression into a target expression.
    We apply rewriting 'actions' up to a max number of steps.
    Reward:
      +1 if expression is transformed to target (or they differ by 0)
      -0.1 if we exceed max steps without success
      0 otherwise
    """
    def __init__(self, start_expr, target_expr, max_steps=10):
        self.start_expr = start_expr
        self.target_expr = target_expr
        self.max_steps = max_steps
        self.current_expr = None
        self.step_count = 0

    def reset(self):
        self.current_expr = self.start_expr
        self.step_count = 0

    def step(self, action_index):
        """
        action_index: integer index in ALL_RULES.
        Returns: (new_expr, reward, done, info)
        """
        self.step_count += 1
        new_expr = apply_rule(self.current_expr, action_index)
        self.current_expr = new_expr

        # Check success
        if sympy.simplify(self.target_expr - self.current_expr) == 0:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        if self.step_count >= self.max_steps and not done:
            done = True
            reward = -0.1

        return new_expr, reward, done, {}

    def sample_action(self):
        import random
        return random.randrange(len(ALL_RULES))