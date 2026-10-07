import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from bittrellis import quantizers as Q
from bittrellis.build import build
from bittrellis.manifest import Manifest
from bittrellis.quantizers.builtin import RTN
from bittrellis.safetensors_io import SafeTensorsDir
from bittrellis.track import load_track
from bittrellis.validate import audit

QUIET = {"log": lambda *_: None, "verify": False}


class _Counting(RTN):
    name, version, replay_mode = "countrtn", 1, "independent"
    calls = 0

    def encode(self, ctx, unit, lin, fmt):
        type(self).calls += 1
        return super().encode(ctx, unit, lin, fmt)


@pytest.fixture()
def counting():
    _Counting.calls = 0
    Q.register(_Counting())
    yield _Counting
    Q.REGISTRY.pop("countrtn", None)


def manifest():
    return Manifest.from_dict({"schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "t", "default": "NVFP4",
                               "rules": [{"match": "L*.gdn.*", "format": "FP8", "quantizer": "countrtn"}]})


def srcs(tiny_all):
    base, bl, ct = tiny_all
    return {"base": base, "gittensor_nvfp4": bl, "unsloth_nvfp4": ct}


def tensors(path):
    with SafeTensorsDir(path) as ck:
        return {n: bytes(ck.raw(n)) for n in ck.tensors}


def test_trusted_builds_reuse_encoder_outputs_byte_for_byte(tiny_all, counting, tmp_path, monkeypatch):
    monkeypatch.setenv("BITTRELLIS_BUILD_JOBS", "1")   # serial, so this process sees every encode() call
    track = load_track("HPC-01")
    build(manifest(), track, srcs(tiny_all), tmp_path / "plain", **QUIET)
    plain_calls = counting.calls
    monkeypatch.setenv("BITTRELLIS_ENCODE_CACHE", str(tmp_path / "cache"))
    build(manifest(), track, srcs(tiny_all), tmp_path / "first", **QUIET)
    counting.calls = 0
    build(manifest(), track, srcs(tiny_all), tmp_path / "second", **QUIET)
    assert plain_calls > 0 and counting.calls == 0          # every encoding came from the cache
    assert tensors(tmp_path / "second") == tensors(tmp_path / "first") == tensors(tmp_path / "plain")
    # a damaged entry is recomputed, not used
    blob = sorted((tmp_path / "cache").glob("*.bin"))[0]
    blob.write_bytes(bytes([blob.read_bytes()[0] ^ 1]) + blob.read_bytes()[1:])
    build(manifest(), track, srcs(tiny_all), tmp_path / "third", **QUIET)
    assert counting.calls == 1 and tensors(tmp_path / "third") == tensors(tmp_path / "plain")
    # parallel workers read the same cache
    monkeypatch.setenv("BITTRELLIS_BUILD_JOBS", "2")
    build(manifest(), track, srcs(tiny_all), tmp_path / "parallel", **QUIET)
    assert tensors(tmp_path / "parallel") == tensors(tmp_path / "plain")


def test_the_audit_never_trusts_the_cache(tiny_all, counting, tmp_path, monkeypatch):
    monkeypatch.setenv("BITTRELLIS_ENCODE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("BITTRELLIS_REPLAY_CACHE", str(tmp_path / "replay"))
    track = load_track("HPC-01")
    build(manifest(), track, srcs(tiny_all), tmp_path / "warm", **QUIET)
    # poison every entry with other, self-consistent bytes (as a broken or tampered cache would hold)
    for meta in (tmp_path / "cache").glob("*.json"):
        blob = meta.with_suffix(".bin")
        data = bytearray(blob.read_bytes())
        entries = json.loads(meta.read_text())
        for e in entries:
            data[e[3]] ^= 0x01
            e[5] = hashlib.sha256(bytes(data[e[3]:e[3] + e[4]])).hexdigest()
        blob.write_bytes(bytes(data))
        meta.write_text(json.dumps(entries))
    build(manifest(), track, srcs(tiny_all), tmp_path / "poisoned", **QUIET)
    res = audit(tmp_path / "poisoned", manifest(), srcs(tiny_all), verify_sources=False)
    assert not res.ok and "not produced by the declared quantizer" in res.errors[0]


def test_the_sandbox_never_sees_the_cache(monkeypatch):
    spec = importlib.util.spec_from_file_location("sandbox", Path(__file__).resolve().parents[2] / "evaluator/sandbox.py")
    sandbox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sandbox)
    monkeypatch.setenv("BITTRELLIS_ENCODE_CACHE", "/workspace/bt-eval/encode-cache")
    cmd = sandbox.command("bt-sandbox", ["python", "-m", "bittrellis.cli", "build"], "/home/bt-sandbox", "/usr/bin")
    assert not any("ENCODE_CACHE" in c for c in cmd)
