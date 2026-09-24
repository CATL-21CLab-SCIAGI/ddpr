"""Set up CPU development or install ddpr in a ready Slime/Swift GPU container."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "envs"


def _checkout(path: Path, repository: str, revision: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temporary:
            subprocess.run(["git", "init", temporary], check=True)
            subprocess.run(
                ["git", "-C", temporary, "remote", "add", "origin", repository],
                check=True,
            )
            subprocess.run(
                ["git", "-C", temporary, "fetch", "--depth=1", "origin", revision],
                check=True,
            )
            subprocess.run(
                ["git", "-C", temporary, "checkout", "--detach", revision], check=True
            )
            Path(temporary).rename(path)
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip()
    if head != revision or dirty:
        raise ValueError(f"{path} must be a clean checkout of {revision}")


def _setup_cpu(resolve: bool) -> None:
    if sys.version_info[:2] != (3, 12):
        raise ValueError("the CPU development profile requires Python 3.12")
    if not (Path(sys.prefix) / "conda-meta").is_dir() and sys.prefix == sys.base_prefix:
        raise ValueError("activate a dedicated Mamba environment or virtualenv first")
    if (Path(sys.prefix) / "conda-meta").is_dir() and (
        Path(sys.prefix) / "envs"
    ).is_dir():
        raise ValueError("use a dedicated environment, not the Mamba base environment")
    locked = (platform.system(), platform.machine()) == ("Darwin", "arm64")
    if not locked and not resolve:
        raise ValueError(
            "no tested snapshot for this platform; use --resolve, then verify"
        )

    sources = Path(sys.prefix) / "src/ddpr"
    frameworks = json.loads((CONFIG / "frameworks.json").read_text())
    for name, spec in frameworks.items():
        _checkout(sources / name, spec["repository"], spec["cpu_revision"])

    pip = [sys.executable, "-m", "pip", "install"]
    requirements = ["-r", str(CONFIG / "cpu/requirements.txt")]
    if locked and not resolve:
        requirements += ["-r", str(CONFIG / "cpu/requirements-macos-arm64-py312.txt")]
    else:
        print("Resolving an unverified development environment.", flush=True)
    # Swift's dependencies are resolved; Slime's GPU dependency set is omitted.
    subprocess.run([*pip, *requirements, str(sources / "ms-swift")], check=True)
    subprocess.run(
        [
            *pip,
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(sources / "slime"),
            "-e",
            str(sources / "ms-swift"),
            "-e",
            str(ROOT),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("cpu", "slime", "swift"))
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="CPU only: resolve dependencies instead of using the Mac snapshot",
    )
    args = parser.parse_args()
    if args.resolve and args.profile != "cpu":
        parser.error("--resolve is only supported for the cpu profile")
    if sys.version_info < (3, 12):  # noqa: UP036 - also run before installing ddpr
        parser.error("ddpr requires Python 3.12 or newer")
    if args.profile == "cpu":
        try:
            _setup_cpu(args.resolve)
        except ValueError as error:
            parser.error(str(error))
    else:
        if platform.system() != "Linux":
            parser.error("GPU setup requires a ready Linux/CUDA container")
        # The container owns its framework and CUDA dependencies.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                str(ROOT),
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/envs/check.py"), args.profile],
        check=True,
    )


if __name__ == "__main__":
    main()
