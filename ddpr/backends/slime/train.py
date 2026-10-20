"""Delegate training to Slime with DDPR reward configuration in its Ray arguments."""

import os
import runpy
from argparse import ArgumentParser
from pathlib import Path


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--ddpr-reward", help="Importable shared Reward class")
    return parser


def _reward_environment() -> dict[str, str]:
    """Forward explicit judge settings and credentials to Ray reward workers."""
    names = {
        "OPENAI_API_KEY",
        "ALIYUN_API_KEY",
        "AWS_BEDROCK_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
    }
    names.update(
        key for key in os.environ if key.startswith(("DDPR_CRITPT_AI_", "AUTO_CRITPT_"))
    )
    names.update(
        os.environ[key]
        for key in tuple(names)
        if key.endswith("_API_KEY_ENV") and key in os.environ
    )
    return {key: os.environ[key] for key in names if key in os.environ}


def main() -> None:
    root = Path(os.environ.get("DDPR_SLIME_ROOT", "/root/slime"))
    upstream = runpy.run_path(str(root / "train.py"))
    args = upstream["parse_args"](add_custom_arguments=add_arguments)
    if args.ddpr_reward:
        import ray

        # Actor-specific environments merge with this job environment.
        ray.init(runtime_env={"env_vars": _reward_environment()})
    upstream["train"](args)


if __name__ == "__main__":
    main()
