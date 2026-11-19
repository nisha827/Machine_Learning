"""Backward-compatible entry point for the Adult Income experiment.

Run ``python my_code.py`` from this directory, or use ``python -m src.train``.
Importing this module does not start a training run.
"""
from src.train import main, run

__all__ = ["main", "run"]


if __name__ == "__main__":
    main()
