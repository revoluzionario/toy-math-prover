# Hybrid Neural‑Symbolic Theorem Prover

## Introduction

This repository contains a codebase that uses **Sympy** for symbolic manipulation and a **TreeLSTM** + **Policy/Value network** for learning rewriting strategies to transform mathematical expressions from a start form to a target form. It can be trained on a dataset of mathematical expressions, and then used for inference on new expressions.

This project is a personal toy project based on the idea for the Bachelor's thesis of the main author (Cong-Hoang Le / @revoluzionario); and it was inspired by these works:

- [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075)
- [Mastering the game of Go with deep neural networks and tree search](https://doi.org/10.1038/nature16961)

## Installation

The project currently uses Python 3.11, with dependencies listed in `pyproject.toml`. [UV](https://docs.astral.sh/uv/) for managing virtual environments is recommended.

To install Python and dependencies via UV (as it create a virtual environment for the project):

```bash
uv run hello.py
```

What it does under the hood: Check for the virtual environment, create it if it doesn't exist, and install the dependencies listed in `pyproject.toml`, then run the script. Any script will work, and `hello.py` is provided as an example.

## Project Structure

```plaintext
toy-math-prover/
├── .gitignore              # Files to ignore in Git.
├── .python-version         # Python version for the virtual environment, currently 3.11.
├── dataset.csv             # Sample training data.
├── dataset.py              # Loads expression pairs from CSV.
├── environment.py          # ProofEnvironment for rewriting.
├── hello.py                # Example script to run.
├── infer.py                # Script to run inference on new expressions.
├── model.py                # TreeLSTM, Symbol Embeddings, Policy-Value net.
├── pyproject.toml          # Project metadata and dependencies.
├── README.md               # This file.
├── rules.py                # Set of rewriting functions.
├── tree.py                 # Converts Sympy expressions to a tree.
├── train.py                # Training script (REINFORCE with value baseline).
├── test_expressions.txt    # Example file with test expressions.
└── uv.lock                 # Lock file for uv.
```

## Contributing

Currently there is no contribution guideline, but a few things to keep in mind:

- The code should be commented following the [Numpy style](https://numpydoc.readthedocs.io/en/latest/format.html), and formatted using [Ruff](https://docs.astral.sh/ruff/).

- The code should be type-anotated and checked with [MyPy](https://mypy.readthedocs.io/en/stable/) if possible.
