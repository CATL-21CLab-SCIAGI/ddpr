"""Check a development or GPU environment without changing installed packages."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "envs"


def _revision(name: str) -> str:
    package = distribution(name)
    source = json.loads(package.read_text("direct_url.json") or "{}")
    if "vcs_info" in source:
        return source["vcs_info"]["commit_id"]
    if source.get("dir_info", {}).get("editable"):
        path = Path(unquote(urlparse(source["url"]).path))
        if not (path / ".git").exists():
            raise ValueError(f"{name}: editable source has no Git metadata")
        dirty = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        ).strip()
        if dirty:
            raise ValueError(f"{name}: framework checkout has local changes")
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    raise ValueError(
        f"{name}: revision unavailable; install from pinned Git or editable source"
    )


def _dependencies(profile: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    omitted = []
    failures = []
    for line in result.stdout.splitlines():
        if line == "No broken requirements found.":
            continue
        if profile == "cpu" and line.startswith("slime "):
            omitted.append(line)
        else:
            failures.append(line)
    if failures or (result.returncode and not omitted):
        raise ValueError(
            "dependency check failed: " + "\n".join(failures or [result.stderr])
        )
    return omitted


def check(profile: str) -> dict:
    report = {
        "profile": profile,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {},
        "frameworks": {},
        "warnings": [],
        "errors": [],
    }
    if sys.version_info < (3, 12):  # noqa: UP036 - also run before installing ddpr
        report["errors"].append("ddpr requires Python 3.12 or newer")
    if profile == "cpu" and sys.version_info[:2] != (3, 12):
        report["errors"].append("the CPU development profile requires Python 3.12")
    names = (
        ["slime", "ms-swift"]
        if profile == "cpu"
        else ["slime" if profile == "slime" else "ms-swift"]
    )
    specs = json.loads((CONFIG / "frameworks.json").read_text())
    for name in names:
        try:
            revision = _revision(name)
            key = "cpu_revision" if profile == "cpu" else "gpu_revision"
            expected = specs[name][key] if profile == "cpu" else specs[name].get(key)
            report["frameworks"][name] = revision
            if expected is None:
                report["warnings"].append(
                    f"{name}: no tested GPU revision recorded; validate in the target "
                    "container before recording gpu_revision"
                )
            elif revision != expected:
                raise ValueError(f"{name}: expected {key} {expected}, found {revision}")
        except (
            PackageNotFoundError,
            ValueError,
            OSError,
            subprocess.CalledProcessError,
        ) as error:
            report["errors"].append(str(error))

    for name in (
        "torch",
        "transformers",
        "datasets",
        "ray" if profile == "slime" else "trl",
    ):
        try:
            report["packages"][name] = version(name)
        except PackageNotFoundError as error:
            report["errors"].append(str(error))
    if profile == "cpu":
        for line in (CONFIG / "cpu/requirements.txt").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            requirement, expected = line.split("==", 1)
            name = requirement.split("[", 1)[0]
            try:
                actual = version(name)
                if actual != expected:
                    report["errors"].append(
                        f"{name}: expected {expected}, found {actual}"
                    )
            except PackageNotFoundError as error:
                report["errors"].append(str(error))
    try:
        report["omitted_training_dependencies"] = _dependencies(profile)
    except (ValueError, OSError) as error:
        report["errors"].append(str(error))

    try:
        torch = importlib.import_module("torch")
        report["cuda"] = {
            "build": torch.version.cuda,
            "available": torch.cuda.is_available(),
        }
        if profile != "cpu":
            if platform.system() != "Linux" or not torch.cuda.is_available():
                raise ValueError(
                    "GPU profiles require Linux and working CUDA; use cpu locally"
                )
            # An actual kernel catches driver/runtime errors missed by imports.
            value = torch.ones(2, device="cuda")
            if (value + value).tolist() != [2.0, 2.0]:
                raise ValueError("CUDA arithmetic check failed")
            report["cuda"]["devices"] = torch.cuda.device_count()
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        report["errors"].append(str(error))

    modules = ["ddpr.data.adapters.ddsr_bench"]
    if profile in {"cpu", "slime"}:
        modules += ["ddpr.backends.slime.plugin", "slime.utils.mask_utils"]
    if profile in {"cpu", "swift"}:
        modules += ["swift.dataset", "swift.template", "swift.rl_core.grpo_algorithm"]
    if profile == "slime" and report.get("cuda", {}).get("available"):
        modules += [
            "slime.utils.arguments",
            "megatron.core",
            "transformer_engine.pytorch",
        ]
    for module in modules:
        try:
            loaded = importlib.import_module(module)
            if module == "swift.template" and "qwen3_8" not in loaded.TEMPLATE_MAPPING:
                raise ValueError("Swift has no qwen3_8 template")
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            report["errors"].append(f"{module}: {error}")
    report["status"] = "failed" if report["errors"] else "passed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("cpu", "slime", "swift"))
    args = parser.parse_args()
    report = check(args.profile)
    print(json.dumps(report, indent=2))
    sys.exit(bool(report["errors"]))


if __name__ == "__main__":
    main()
