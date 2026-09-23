import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import check_environment, setup_dev


def test_dev_reports_omitted_slime_dependencies(monkeypatch):
    message = "slime 0.3.2 requires sglang-router, which is not installed."
    monkeypatch.setattr(
        check_environment.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout=message, stderr=""),
    )
    assert check_environment._dependencies("dev") == [message]
    with pytest.raises(ValueError, match="dependency check failed"):
        check_environment._dependencies("slime")


def test_dev_does_not_hide_other_dependency_conflicts(monkeypatch):
    monkeypatch.setattr(
        check_environment.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=1,
            stdout="slime 0.3.2 requires sglang-router\nms-swift requires transformers",
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match="ms-swift requires transformers"):
        check_environment._dependencies("dev")


@pytest.mark.parametrize("profile", ["slime", "swift"])
def test_gpu_profile_fails_without_cuda(monkeypatch, profile):
    specs = json.loads((check_environment.CONFIG / "frameworks.json").read_text())
    monkeypatch.setattr(
        check_environment, "_revision", lambda name: specs[name]["revision"]
    )
    monkeypatch.setattr(check_environment, "version", lambda name: "test")
    monkeypatch.setattr(check_environment, "_dependencies", lambda profile: [])
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(
        check_environment.importlib,
        "import_module",
        lambda name: (
            torch
            if name == "torch"
            else SimpleNamespace(TEMPLATE_MAPPING={"qwen3_8": None})
        ),
    )
    report = check_environment.check(profile)
    assert report["status"] == "failed"
    assert any("working CUDA" in error for error in report["errors"])


def test_unverifiable_local_install_is_rejected(monkeypatch):
    package = SimpleNamespace(
        read_text=lambda name: json.dumps({"url": "file:///tmp/source", "dir_info": {}})
    )
    monkeypatch.setattr(check_environment, "distribution", lambda name: package)
    with pytest.raises(ValueError, match="revision unavailable"):
        check_environment._revision("ms-swift")


def test_checkout_preserves_modified_source(tmp_path):
    source = tmp_path / "source"
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    (source / "file.txt").write_text("original")
    subprocess.run(["git", "-C", str(source), "add", "file.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "initial",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    (source / "file.txt").write_text("local work")
    with pytest.raises(ValueError, match="clean checkout"):
        setup_dev._checkout(source, "unused", revision)
    assert (source / "file.txt").read_text() == "local work"
