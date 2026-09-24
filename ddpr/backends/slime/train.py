"""Delegate training to Slime with DDPR reward configuration in its Ray arguments."""

import os
import runpy
from argparse import ArgumentParser
from pathlib import Path


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--ddpr-reward", help="Importable shared Reward class")
    return parser


def main() -> None:
    root = Path(os.environ.get("DDPR_SLIME_ROOT", "/root/slime"))
    upstream = runpy.run_path(str(root / "train.py"))
    args = upstream["parse_args"](add_custom_arguments=add_arguments)
    upstream["train"](args)


if __name__ == "__main__":
    main()
