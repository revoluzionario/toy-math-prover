# Hybrid Neural‑Symbolic Theorem Prover

This repository contains a codebase that uses **Sympy** for symbolic manipulation and a **TreeLSTM** + **Policy/Value network** for learning rewriting strategies to transform mathematical expressions from a start form to a target form. It can be trained on a dataset of expressions, and then used for inference on new expressions.
This project is a personal toy project based on the final-term idea of author, and inspired by these papers:
[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075)
[Mastering the game of Go with deep neural networks and tree search](https://doi.org/10.1038/nature16961)


## Project Structure

```plaintext
theorem_prover/
├── dataset.csv             # Sample training data
├── dataset.py              # Loads expression pairs from CSV
├── environment.py          # ProofEnvironment for rewriting
├── infer.py                # Script to run inference on new expressions
├── model.py                # TreeLSTM, Symbol Embeddings, Policy-Value net
├── rewriting_rules.py      # Set of rewriting functions
├── symbolic_tree.py        # Converts Sympy expressions to a tree
├── train.py                # Training script (REINFORCE with value baseline)
├── test_expressions.txt    # Example file with test expressions
└── README.md               # This file
