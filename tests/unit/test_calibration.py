from types import SimpleNamespace

import numpy as np
import pytest

from bittrellis import calibration as C
from bittrellis import quantizers as Q
from bittrellis.build import build, open_context
from bittrellis.fingerprint import probe
from bittrellis.manifest import Manifest
from bittrellis.model.qwen38 import Qwen38Arch
from bittrellis.precision import NVFP4
from bittrellis.quant.formats import dequantize_nvfp4, quantize_nvfp4
from bittrellis.quantizers.base import f32_weight, input_hessian
from bittrellis.safetensors_io import SafeTensorsDir
from bittrellis.synthetic import TINY_TEXT, make_tiny_calibration
from bittrellis.track import load_track
from bittrellis.validate import _replay_key, audit

QUIET = {"log": lambda *_: None, "verify": False}
MLP_GATE = "model.language_model.layers.0.mlp.gate_proj"


class _Weighted(Q.Quantizer):
    """Toy calibrated encoder: picks the NVFP4 global scale that minimises the input-weighted error."""

    name, version, formats, lineage, replay_mode = "toyhess", 1, (NVFP4,), "regenerable", "independent"

    def available(self, ctx, unit, fmt):
        return None if ctx.calibration is not None else "needs the calibration statistics (bittrellis calibration fetch)"

    def encode(self, ctx, unit, lin, fmt):
        w = f32_weight(ctx, lin)
        h = input_hessian(ctx, lin)
        amax = float(np.abs(w).max())
        best = None
        for clip in (1.0, 0.9, 0.8, 0.7):
            p, s, ws2 = quantize_nvfp4(w, global_amax=clip * amax)
            e = (w - dequantize_nvfp4(p, s, ws2)).astype(np.float64)
            cost = float(np.einsum("ij,jk,ik->", e, h, e)) if h is not None else float((e * e).sum())
            if best is None or cost < best[0]:
                best = (cost, p, s, ws2)
        _, p, s, ws2 = best
        return [(".weight", "U8", p.shape, p), (".weight_scale", "F8_E4M3", s.shape, s),
                (".weight_scale_2", "F32", (), np.asarray(ws2, "<f4").reshape(()))]


@pytest.fixture()
def toy():
    Q.register(_Weighted())
    yield Q.get("toyhess")
    Q.REGISTRY.pop("toyhess", None)


@pytest.fixture(scope="module")
def calib(tmp_path_factory):
    return make_tiny_calibration(tmp_path_factory.mktemp("calib"), seed=0)


def manifest():
    return Manifest.from_dict({"schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "t", "default": "NVFP4",
                               "rules": [{"match": "L*.mlp", "format": "NVFP4", "quantizer": "toyhess"}]})


def srcs(tiny_all, calib=None):
    base, bl, ct = tiny_all
    return {"base": base, "gittensor_nvfp4": bl, "unsloth_nvfp4": ct, **({"calibration": calib} if calib else {})}


def test_statistics_exist_only_for_linears_that_read_a_module_input():
    p = "model.language_model.layers.3."
    assert C.stat_name(p + "mlp.gate_proj") == C.stat_name(p + "mlp.up_proj") == p + "mlp.input_xtx"
    assert C.stat_name(p + "self_attn.q_proj") == p + "self_attn.input_xtx"
    assert C.stat_name(p + "linear_attn.in_proj_qkv") == C.stat_name(p + "linear_attn.in_proj_z") == p + "linear_attn.input_xtx"
    for leaf in ("mlp.down_proj", "self_attn.o_proj", "linear_attn.out_proj"):
        assert C.stat_name(p + leaf) is None


def test_encoders_read_the_statistics_through_the_build_context(tiny_all, calib, toy):
    units = Qwen38Arch.from_config(tiny_all[1] / "config.json").units()
    mlp = next(u for u in units if u.id == "L0.mlp")
    gate, up, down = mlp.linears
    ctx, handles = open_context(manifest().expand_assignments(units), srcs(tiny_all, calib), False, lambda *_: None)
    try:
        h = input_hessian(ctx, gate)
        assert h.shape == (gate.cols, gate.cols) and h.dtype == np.float32 and not h.flags.writeable
        assert np.array_equal(h, input_hessian(ctx, up)) and np.allclose(h, h.T)
        assert input_hessian(ctx, down) is None
    finally:
        for x in handles:
            x.close()
    ctx, handles = open_context(manifest().expand_assignments(units), srcs(tiny_all), False, lambda *_: None)
    assert ctx.calibration is None and input_hessian(ctx, gate) is None
    for x in handles:
        x.close()


def test_calibrated_build_is_deterministic_and_replayed_by_the_audit(tiny_all, calib, toy, tmp_path, monkeypatch):
    monkeypatch.setenv("BITTRELLIS_REPLAY_CACHE", str(tmp_path / "cache"))
    track = load_track("HPC-01")
    a, b = tmp_path / "a", tmp_path / "b"
    build(manifest(), track, srcs(tiny_all, calib), a, **QUIET)
    build(manifest(), track, srcs(tiny_all, calib), b, **QUIET)
    with SafeTensorsDir(a) as x, SafeTensorsDir(b) as y:
        assert all(bytes(x.raw(n)) == bytes(y.raw(n)) for n in x.tensors)
    res = audit(a, manifest(), srcs(tiny_all, calib), verify_sources=False)
    assert res.ok, res.errors
    assert res.lineage["toyhess@v1"]["replayed_tensors"] > 0
    # the replay reads the same statistics: other statistics regenerate other bytes
    other = make_tiny_calibration(tmp_path / "other", seed=5)
    monkeypatch.setenv("BITTRELLIS_REPLAY_CACHE", str(tmp_path / "cache2"))
    assert not audit(a, manifest(), srcs(tiny_all, other), verify_sources=False).ok
    # without the statistics the encoder refuses instead of guessing
    with pytest.raises(ValueError, match="needs the calibration"):
        build(manifest(), track, srcs(tiny_all), tmp_path / "c", **QUIET)


def test_the_fingerprint_probe_runs_calibrated_encoders(toy):
    first = probe(["toyhess"], seed=3)
    assert first and first == probe(["toyhess"], seed=3)


def test_the_replay_key_follows_the_pinned_calibration(toy, monkeypatch):
    from bittrellis import lineage

    units = Qwen38Arch.from_config({"text_config": TINY_TEXT}).units()
    assignments = manifest().expand_assignments(units)
    rtn = Q.get("rtn")
    lock = lineage.load_lock()
    monkeypatch.setattr(lineage, "load_lock", lambda *_: lock)
    before = _replay_key(rtn, units, assignments, units[0])
    pinned = {**lock, "sources": {**lock["sources"], "calibration": {"files": {"x.safetensors": {"sha256": "a" * 64, "size": 1}}}}}
    monkeypatch.setattr(lineage, "load_lock", lambda *_: pinned)
    assert _replay_key(rtn, units, assignments, units[0]) != before


class _Tok:
    def encode(self, s):
        return SimpleNamespace(ids=[ord(c) for c in s])


def test_packing_drops_overlapping_documents_and_opens_every_sequence_with_a_separator():
    scored = "The quick brown fox jumps over the lazy dog while the band plays on and on and on."
    excluded = C.windows(scored)
    docs = ["x" * 40 + " " + scored.upper() + " tail", *[f"document {i} " + "abcdefghij" * 30 for i in range(20)]]
    seqs, stats = C.pack(_Tok(), docs, excluded, n_seqs=3, seq_tokens=200)
    assert stats["documents_dropped_overlap"] == 1
    assert len(seqs) == 3 and all(len(s) == 200 for s in seqs)
    assert all("".join(map(chr, s)).startswith(C.DOC_SEP) for s in seqs)
    with pytest.raises(ValueError, match="not enough"):
        C.pack(_Tok(), docs[1:3], excluded, n_seqs=3, seq_tokens=200)


def test_glaive_conversations_become_qwen_chat():
    r = {"system": "SYSTEM: You can call get_time.",
         "chat": "USER: What time is it? ASSISTANT: <functioncall> {\"name\": \"get_time\"} <|endoftext|> "
                 "FUNCTION RESPONSE: {\"time\": \"10:00\"} ASSISTANT: It is 10:00. <|endoftext|>"}
    out = C.glaive_chat(r)
    roles = [line[len("<|im_start|>"):] for line in out.splitlines() if line.startswith("<|im_start|>")]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert "<|endoftext|>" not in out and "You can call get_time." in out


def test_saved_text_round_trips_and_is_hash_checked(tmp_path):
    body = {"version": C.VERSION, "tokens": 4, "sequences": [{"category": "math", "ids": [1, 2]}, {"category": "code", "ids": [3, 4]}]}
    body["sha256"] = C.text_hash(body)
    path = tmp_path / "t.json"
    C.save_text(body, path)
    assert C.load_text(path) == body
    path.write_text(path.read_text().replace("[3,4]", "[3,5]"))
    with pytest.raises(ValueError, match="hash"):
        C.load_text(path)
