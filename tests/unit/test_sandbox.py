import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("sandbox", Path(__file__).resolve().parents[2] / "evaluator/sandbox.py")
sandbox = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sandbox)


def test_sandbox_command_drops_the_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("LANG", "C.UTF-8")
    cmd = sandbox.command("bt-sandbox", ["python", "-m", "bittrellis.cli", "build"], "/home/bt-sandbox", "/venv/bin:/usr/bin")
    assert cmd[:6] == ["runuser", "-u", "bt-sandbox", "--", "env", "-i"]
    joined = " ".join(cmd)
    assert "ghp_secret" not in joined and "GITHUB_TOKEN" not in joined
    assert "CUDA_VISIBLE_DEVICES=" in cmd and "HOME=/home/bt-sandbox" in cmd and "LANG=C.UTF-8" in cmd
    assert cmd[-4:] == ["python", "-m", "bittrellis.cli", "build"]


def test_sandbox_steps_get_their_share_of_blas_threads(monkeypatch):
    from bittrellis import blas_threads

    monkeypatch.setattr(sandbox.os, "cpu_count", lambda: 24)
    monkeypatch.delenv("BITTRELLIS_BUILD_JOBS", raising=False)
    assert blas_threads() == 2   # 12 build workers on 24 CPUs
    cmd = sandbox.command("bt-sandbox", ["python", "-m", "bittrellis.cli", "build"], "/home/bt-sandbox", "/usr/bin")
    assert {"OPENBLAS_NUM_THREADS=2", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2"} <= set(cmd)
    monkeypatch.setenv("BITTRELLIS_BUILD_JOBS", "1")
    assert blas_threads() == 24  # a serial build may use the whole machine
