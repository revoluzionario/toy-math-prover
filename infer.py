# infer.py
import argparse
import torch
import sympy

from environment import ProofEnvironment
from tree import parse_sympy_expression
from model import SYMBOL_MAP, SymbolEmbedding, TreeLSTM, PolicyValueNet, assign_embeddings
from rules import ALL_RULES
from dataset import ExpressionDataset  # if you want to compare to dataset or reuse

def load_model():
    embed_dim = 16
    hidden_dim = 32
    num_actions = len(ALL_RULES)

    embedder = SymbolEmbedding(len(SYMBOL_MAP), embed_dim)
    treelstm = TreeLSTM(embed_dim, hidden_dim)
    policy_value_net = PolicyValueNet(hidden_dim, num_actions)

    checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))
    embedder.load_state_dict(checkpoint['embedder_state'])
    treelstm.load_state_dict(checkpoint['treelstm_state'])
    policy_value_net.load_state_dict(checkpoint['policy_value_state'])

    embedder.eval()
    treelstm.eval()
    policy_value_net.eval()

    return embedder, treelstm, policy_value_net

def infer_expression(expr_str, target_str, max_steps=8, greedy=True):
    """
    Attempt to transform expr_str -> target_str with the loaded model.
    If greedy=True, pick the argmax action. Otherwise, sample from the policy.
    """
    embedder, treelstm, policy_value_net = load_model()

    start_expr = sympy.sympify(expr_str)
    target_expr = sympy.sympify(target_str)

    env = ProofEnvironment(start_expr, target_expr, max_steps=max_steps)
    env.reset()

    print(f"--- Inference: rewriting:\n  start:  {start_expr}\n  target: {target_expr}")
    for step_i in range(max_steps):
        root = parse_sympy_expression(str(env.current_expr))
        assign_embeddings(root, embedder)
        h, c = treelstm(root)
        policy_logits, value = policy_value_net(h)

        # Choose action
        probs = torch.softmax(policy_logits, dim=1).detach().numpy()[0]
        if greedy:
            action_index = probs.argmax()
        else:
            import random
            # Weighted random
            action_index = random.choices(range(len(ALL_RULES)), weights=probs)[0]

        new_expr, reward, done, _ = env.step(action_index)
        print(f"Step {step_i+1}: action={ALL_RULES[action_index].__name__}, expr={new_expr}, reward={reward:.2f}")

        if done:
            if reward > 0:
                print("SUCCESS: Reached target!")
            else:
                print("Episode ended but did not match target.")
            break
    else:
        print("Max steps reached without success.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exprs', type=str, nargs='*', default=None,
                        help='List of expressions to attempt rewriting.')
    parser.add_argument('--expr_file', type=str, default=None,
                        help='Path to a file containing expressions (one per line).')
    parser.add_argument('--target', type=str, default='x**2 + 2*x*y + y**2',
                        help='Target expression to rewrite to.')
    parser.add_argument('--max_steps', type=int, default=8,
                        help='Max rewriting steps.')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy argmax action selection instead of sampling.')
    args = parser.parse_args()

    if not args.exprs and not args.expr_file:
        print("No expressions provided. Use --exprs 'expr1' 'expr2' ... or --expr_file my_file.txt")
        return

    expr_list = []
    if args.exprs:
        expr_list.extend(args.exprs)
    if args.expr_file:
        with open(args.expr_file, 'r') as f:
            lines = f.read().strip().split('\n')
        expr_list.extend(lines)

    for expr_str in expr_list:
        infer_expression(expr_str, args.target, max_steps=args.max_steps, greedy=args.greedy)
        print()

if __name__ == "__main__":
    main()
