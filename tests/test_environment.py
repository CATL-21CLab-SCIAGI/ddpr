import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.envs import check, setup


def test_cpu_reports_omitted_slime_dependencies(monkeypatch):
    message = "slime 0.3.2 requires sglang-router, which is not installed."
    monkeypatch.setattr(
        check.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout=message, stderr=""),
    )
    assert check._dependencies("cpu") == [message]
    with pytest.raises(ValueError, match="dependency check failed"):
        check._dependencies("slime")


def test_cpu_does_not_hide_other_dependency_conflicts(monkeypatch):
    monkeypatch.setattr(
        check.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=1,
            stdout="slime 0.3.2 requires sglang-router\nms-swift requires transformers",
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match="ms-swift requires transformers"):
        check._dependencies("cpu")


@pytest.mark.parametrize("profile", ["slime", "swift"])
@pytest.mark.parametrize("revision_key", ["cpu_revision", "gpu_revision", "unknown"])
def test_gpu_checks_cuda_and_target_revision(monkeypatch, profile, revision_key):
    specs = json.loads((check.CONFIG / "frameworks.json").read_text())
    monkeypatch.setattr(
        check, "_revision", lambda name: specs[name].get(revision_key, "unknown")
    )
    monkeypatch.setattr(check, "version", lambda name: "test")
    monkeypatch.setattr(check, "_dependencies", lambda profile: [])
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(
        check.importlib,
        "import_module",
        lambda name: (
            torch
            if name == "torch"
            else SimpleNamespace(TEMPLATE_MAPPING={"qwen3_8": None})
        ),
    )
    report = check.check(profile)
    assert report["status"] == "failed"
    assert any("working CUDA" in error for error in report["errors"])
    name = "slime" if profile == "slime" else "ms-swift"
    expected = specs[name].get("gpu_revision")
    mismatch = (
        expected is not None and specs[name].get(revision_key, "unknown") != expected
    )
    assert (
        any("expected gpu_revision" in error for error in report["errors"]) == mismatch
    )
    assert bool(report["warnings"]) == (expected is None)


@pytest.mark.parametrize("profile", ["swift", "slime"])
def test_gpu_runs_checks_without_recording_revision(monkeypatch, tmp_path, profile):
    manifest = tmp_path / "frameworks.json"
    if profile == "slime":
        contents = (check.CONFIG / "frameworks.json").read_text()
        name, revision = "slime", "4c4adbbf3d1bab34d5123e93d1ad112a2bc597e2"
    else:
        contents = json.dumps({"ms-swift": {"cpu_revision": "cpu-commit"}})
        name, revision = "ms-swift", "container-commit"
    manifest.write_text(contents)
    monkeypatch.setattr(check, "CONFIG", tmp_path)
    monkeypatch.setattr(check, "_revision", lambda name: revision)
    monkeypatch.setattr(check, "version", lambda name: "test")
    monkeypatch.setattr(check, "_dependencies", lambda profile: [])
    monkeypatch.setattr(check.platform, "system", lambda: "Linux")

    class Tensor:
        def __add__(self, other):
            return SimpleNamespace(tolist=lambda: [2.0, 2.0])

    def ones(size, *, device):
        assert (size, device) == (2, "cuda")
        return Tensor()

    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="test"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        ones=ones,
    )
    monkeypatch.setattr(
        check.importlib,
        "import_module",
        lambda name: (
            torch
            if name == "torch"
            else SimpleNamespace(TEMPLATE_MAPPING={"qwen3_8": None})
        ),
    )
    report = check.check(profile)
    assert report["status"] == "passed"
    assert report["frameworks"] == {name: revision}
    assert report["cuda"]["devices"] == 1
    if profile == "swift":
        assert "no tested GPU revision recorded" in report["warnings"][0]
    else:
        assert not report["warnings"]
    assert manifest.read_text() == contents


def test_unverifiable_local_install_is_rejected(monkeypatch):
    package = SimpleNamespace(
        read_text=lambda name: json.dumps({"url": "file:///tmp/source", "dir_info": {}})
    )
    monkeypatch.setattr(check, "distribution", lambda name: package)
    with pytest.raises(ValueError, match="revision unavailable"):
        check._revision("ms-swift")


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
        setup._checkout(source, "unused", revision)
    assert (source / "file.txt").read_text() == "local work"


def test_cpu_setup_installs_pinned_frameworks_and_dependencies(monkeypatch, tmp_path):
    commands, checkouts = [], []
    monkeypatch.setattr(setup.sys, "argv", ["setup.py", "cpu"])
    monkeypatch.setattr(setup.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(setup, "_checkout", lambda *args: checkouts.append(args))
    monkeypatch.setattr(
        setup.subprocess, "run", lambda command, **kw: commands.append(command)
    )
    setup.main()
    specs = json.loads((setup.CONFIG / "frameworks.json").read_text())
    assert checkouts == [
        (tmp_path / "src/ddpr" / name, spec["repository"], spec["cpu_revision"])
        for name, spec in specs.items()
    ]
    assert str(setup.CONFIG / "cpu/requirements.txt") in commands[0]
    assert str(setup.CONFIG / "cpu/requirements-macos-arm64-py312.txt") in commands[0]
    assert commands[1][-6:] == [
        "-e",
        str(tmp_path / "src/ddpr/slime"),
        "-e",
        str(tmp_path / "src/ddpr/ms-swift"),
        "-e",
        str(setup.ROOT),
    ]
    assert commands[-1] == [
        setup.sys.executable,
        str(setup.ROOT / "scripts/envs/check.py"),
        "cpu",
    ]


@pytest.mark.parametrize("profile", ["slime", "swift"])
def test_gpu_setup_preserves_container_stack(monkeypatch, profile):
    commands = []
    monkeypatch.setattr(setup.sys, "argv", ["setup.py", profile])
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    # Ready containers can use their system Python without a separate virtualenv.
    monkeypatch.setattr(setup.sys, "prefix", setup.sys.base_prefix)
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    setup.main()
    assert commands == [
        [
            setup.sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(setup.ROOT),
        ],
        [setup.sys.executable, str(setup.ROOT / "scripts/envs/check.py"), profile],
    ]


@pytest.mark.parametrize("profile", ["slime", "swift"])
def test_gpu_setup_propagates_failed_check(monkeypatch, profile):
    monkeypatch.setattr(setup.sys, "argv", ["setup.py", profile])
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")

    def run(command, *, check):
        assert check
        if command[1].endswith("check.py"):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        setup.main()


@pytest.mark.parametrize(
    "arguments,system,error",
    [
        (["slime", "--resolve"], "Linux", "--resolve is only supported"),
        (["swift", "--resolve"], "Linux", "--resolve is only supported"),
        (["slime"], "Darwin", "ready Linux/CUDA container"),
        (["swift"], "Darwin", "ready Linux/CUDA container"),
    ],
)
def test_invalid_setup_does_not_install(monkeypatch, capsys, arguments, system, error):
    monkeypatch.setattr(setup.sys, "argv", ["setup.py", *arguments])
    monkeypatch.setattr(setup.platform, "system", lambda: system)
    monkeypatch.setattr(
        setup.subprocess, "run", lambda *a, **kw: pytest.fail("must not install")
    )
    with pytest.raises(SystemExit) as result:
        setup.main()
    assert result.value.code == 2
    assert error in capsys.readouterr().err
