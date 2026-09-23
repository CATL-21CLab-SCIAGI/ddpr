"""Install the CPU development profile in the active Python environment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/environments"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="resolve dependencies on an unverified platform instead of using the Mac snapshot",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("the development profile requires Python 3.12")
    if not (Path(sys.prefix) / "conda-meta").is_dir() and sys.prefix == sys.base_prefix:
        parser.error("activate a dedicated Mamba environment or virtualenv first")
    if (Path(sys.prefix) / "conda-meta").is_dir() and (
        Path(sys.prefix) / "envs"
    ).is_dir():
        parser.error("use a dedicated environment, not the Mamba base environment")
    locked = (platform.system(), platform.machine()) == ("Darwin", "arm64")
    if not locked and not args.resolve:
        parser.error("no tested snapshot for this platform; use --resolve, then verify")

    sources = Path(sys.prefix) / "src/ddpr"
    frameworks = json.loads((CONFIG / "frameworks.json").read_text())
    for name, spec in frameworks.items():
        _checkout(sources / name, spec["repository"], spec["revision"])

    pip = [sys.executable, "-m", "pip", "install"]
    requirements = ["-r", str(CONFIG / "dev.txt")]
    if locked and not args.resolve:
        requirements += ["-r", str(CONFIG / "dev-macos-arm64-py312.txt")]
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
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_environment.py"), "dev"], check=True
    )


if __name__ == "__main__":
    main()
