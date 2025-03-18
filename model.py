# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# A small dictionary mapping recognized symbolic tokens to IDs
SYMBOL_MAP = {
    'x': 0,
    'y': 1,
    'z': 2,
    '+': 3,
    '-': 4,
    '*': 5,
    '/': 6,
    'Pow': 7,
    'Integer': 8,
    'Symbol': 9,
    'Add': 10,
    'Mul': 11,
    'Number': 12,
    # You can extend further as you see fit
}

class SymbolEmbedding(nn.Module):
    """
    Convert Sympy node -> integer token -> learned embedding vector.
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_id):
        """
        token_id: int
        returns: embedding of shape (1, embed_dim)
        """
        tid_tensor = torch.tensor([token_id], dtype=torch.long)
        return self.embedding(tid_tensor)

def get_token_id(expr):
    """
    Heuristic to pick a token ID for a Sympy expression node.
    """
    if expr.is_Symbol:
        name = str(expr)
        return SYMBOL_MAP.get(name, SYMBOL_MAP['Symbol'])
    elif expr.is_Integer:
        return SYMBOL_MAP['Integer']
    else:
        func_name = expr.func.__name__
        return SYMBOL_MAP.get(func_name, SYMBOL_MAP['Symbol'])

def assign_embeddings(root, embedder):
    """
    Recursively assign .embedding to each node in the expression tree.
    """
    tid = get_token_id(root.expr)
    root.embedding = embedder(tid)
    for child in root.children:
        assign_embeddings(child, embedder)

class TreeLSTM(nn.Module):
    """
    N-ary TreeLSTM that sums child hidden states, plus node embedding, 
    to produce a hidden state.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ioux = nn.Linear(input_dim, 4 * hidden_dim)
        self.iouh = nn.Linear(hidden_dim, 4 * hidden_dim)

    def node_forward(self, node_embedding, child_h, child_c):
        # Sum child hidden states
        if len(child_h) == 0:
            child_h_sum = torch.zeros(1, self.hidden_dim, device=node_embedding.device)
        else:
            child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(node_embedding) + self.iouh(child_h_sum)
        i, o, u, f = torch.split(iou, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        f = torch.sigmoid(f)

        # Merge child cell states
        if len(child_c) > 0:
            c_sum = sum(f * c_sub for c_sub in child_c)
        else:
            c_sum = 0

        c = i * u + c_sum
        h = o * torch.tanh(c)
        return h, c

    def forward(self, root):
        """
        Recursively compute the hidden state from bottom to top.
        """
        child_states = [self.forward(child) for child in root.children]
        if child_states:
            child_h = torch.cat([x[0] for x in child_states], dim=0)
            child_c = torch.cat([x[1] for x in child_states], dim=0)
        else:
            child_h = []
            child_c = []
        h, c = self.node_forward(root.embedding, child_h, child_c)
        return h, c

class PolicyValueNet(nn.Module):
    """
    Given the final hidden state from TreeLSTM, produce:
     - policy_logits (distribution over rewriting rules)
     - value (scalar)
    """
    def __init__(self, hidden_dim, num_actions):
        super().__init__()
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        policy_logits = self.policy_head(h)
        value = self.value_head(h)  # shape (1,1)
        return policy_logits, value