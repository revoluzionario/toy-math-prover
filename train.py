# train.py
import argparse
import random

import torch
import torch.optim as optim

from dataset import ExpressionDataset
from environment import ProofEnvironment
from tree import parse_sympy_expression
from model import (
    SYMBOL_MAP,
    SymbolEmbedding,
    TreeLSTM,
    PolicyValueNet,
    assign_embeddings,
)
from rules import ALL_RULES

def run_episode(env, embedder, tree_lstm, policy_value_net, gamma=0.95):
    """
    Run a single episode with on-policy REINFORCE updates (but we'll do one episode
    at a time, returning the transitions so we can do a single update).
    """
    states = []
    actions = []
    rewards = []
    done_flags = []

    env.reset()

    for _ in range(env.max_steps):
        # Build parse tree from current expr
        expr_str = str(env.current_expr)
        root = parse_sympy_expression(expr_str)

        # Assign embeddings
        assign_embeddings(root, embedder)

        # Forward pass
        h, c = tree_lstm(root)
        policy_logits, value = policy_value_net(h)

        # Sample from policy
        probs = torch.softmax(policy_logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action_index = dist.sample().item()

        # Step environment
        new_expr, reward, done, _ = env.step(action_index)

        # Save transition
        states.append((h, value))
        actions.append(action_index)
        rewards.append(reward)
        done_flags.append(done)

        if done:
            break

    # Now compute the discounted returns
    G = 0.0
    returns = []
    for r, d in reversed(list(zip(rewards, done_flags))):
        G = r + gamma * G
        returns.insert(0, G)
        if d:  # if episode ended with success, reset discount
            G = 0.0
    returns = torch.tensor(returns, dtype=torch.float).view(-1, 1)

    return states, actions, returns

def update_model(states, actions, returns, policy_value_net, optimizer):
    """
    Perform a single REINFORCE update with a value baseline.
    """
    policy_loss = []
    value_loss = []

    for (h_i, value_i), a_i, R_i in zip(states, actions, returns):
        logits_i, _ = policy_value_net(h_i)  # shape (1, num_actions)
        logp = torch.log_softmax(logits_i, dim=1)
        logp_a = logp[0, a_i]

        advantage = R_i - value_i  # shape (1,1)

        # Policy gradient
        policy_loss_i = -logp_a * advantage.detach()

        # Value loss
        value_loss_i = 0.5 * (value_i - R_i).pow(2)

        policy_loss.append(policy_loss_i)
        value_loss.append(value_loss_i)

    policy_loss = torch.stack(policy_loss).sum()
    value_loss = torch.stack(value_loss).sum()
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset.csv',
                        help='Path to CSV with start_expr, target_expr columns.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--max_steps', type=int, default=8,
                        help='Max rewriting steps per expression.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor.')
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    # Load dataset
    ds = ExpressionDataset(args.dataset)
    print(f"Loaded dataset with {len(ds)} expressions.")

    # Build model
    embed_dim = 16
    hidden_dim = 32
    num_actions = len(ALL_RULES)

    embedder = SymbolEmbedding(vocab_size=len(SYMBOL_MAP), embed_dim=embed_dim)
    tree_lstm = TreeLSTM(embed_dim, hidden_dim)
    policy_value_net = PolicyValueNet(hidden_dim, num_actions)

    # Optimizer
    params = list(embedder.parameters()) + list(tree_lstm.parameters()) + list(policy_value_net.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    # Train loop
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_reward = 0.0
        # Shuffle dataset each epoch
        indices = list(range(len(ds)))
        random.shuffle(indices)

        for idx in indices:
            start_expr, target_expr = ds[idx]
            env = ProofEnvironment(start_expr, target_expr, max_steps=args.max_steps)

            # Run an episode
            states, actions, returns = run_episode(env, embedder, tree_lstm, policy_value_net, gamma=args.gamma)
            # Update
            loss_val = update_model(states, actions, returns, policy_value_net, optimizer)
            total_loss += loss_val
            total_reward += sum(r.item() for r in returns)

        avg_loss = total_loss / len(ds)
        avg_return = total_reward / len(ds)
        print(f"Epoch {epoch+1}/{args.epochs} => avg_loss={avg_loss:.4f}, avg_return={avg_return:.4f}")

    # Save model
    checkpoint = {
        'embedder_state': embedder.state_dict(),
        'treelstm_state': tree_lstm.state_dict(),
        'policy_value_state': policy_value_net.state_dict(),
    }
    torch.save(checkpoint, 'model.pth')
    print("Training complete. Model saved to model.pth.")

if __name__ == "__main__":
    main()
