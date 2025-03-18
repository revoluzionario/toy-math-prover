# dataset.py
import pandas as pd
import sympy

class ExpressionDataset:
    """
    Reads a CSV file with columns: start_expr, target_expr
    Stores them as Sympy expressions.
    Example usage:
        ds = ExpressionDataset('dataset.csv')
        for i in range(len(ds)):
            start_expr, target_expr = ds[i]
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.start_expressions = [sympy.sympify(s) for s in df['start_expr']]
        self.target_expressions = [sympy.sympify(t) for t in df['target_expr']]
        self.size = len(self.start_expressions)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.start_expressions[idx], self.target_expressions[idx]