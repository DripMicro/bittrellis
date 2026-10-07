import numpy as np
import pytest

from bittrellis.quant.formats import dequantize_nvfp4
from bittrellis.quantizers import nvfp4_blockfit as bf
from bittrellis.quantizers.nvfp4_gptq import CHUNK_ROWS, inverse_hessian_factor, quantize
from bittrellis.synthetic import make_tiny_calibration

QUIET = {"log": lambda *_: None, "verify": False}


def _weights(seed, shape=(48, 256)):
    return (np.random.default_rng(seed).standard_t(5, size=shape) * 0.02).astype(np.float32)


def _hessian(seed, n=256, tokens=1024):
    """A correlated, unevenly scaled input covariance, like real activations."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tokens, n)) @ (rng.standard_normal((n, n)) * 0.2 + np.eye(n))
    x *= np.exp(rng.standard_normal(n))
    return (x.T @ x / tokens).astype(np.float32)


def _out_err(w, result, h):
    e = w.astype(np.float64) - dequantize_nvfp4(*result)
    return float(np.einsum("ij,jk,ik->", e, h.astype(np.float64), e))


def test_bytes_repeat():
    w, h = _weights(1), _hessian(1)
    a, b = quantize(w, h), quantize(w, h)
    assert all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b, strict=True))


def test_lower_output_error_than_blockfit():
    w, h = _weights(2, (64, 256)), _hessian(2)
    ours, fit = quantize(w, h), bf.quantize(w, "mlp")
    assert ours[2] == fit[2]
    assert _out_err(w, ours, h) < 0.8 * _out_err(w, fit, h)


def test_identity_hessian_stays_close_to_blockfit():
    # without input correlations there is little to propagate: the result is a plain per-weight fit
    w = _weights(3, (32, 256))
    h = np.eye(256, dtype=np.float32)
    ours, fit = quantize(w, h), bf.quantize(w, "mlp")
    assert _out_err(w, ours, h) <= 1.05 * _out_err(w, fit, h)


def test_rows_are_independent_and_use_the_tensor_amax():
    w, h = _weights(4, (CHUNK_ROWS + 40, 64)), _hessian(4, 64)
    amax = float(np.abs(w).max())
    full = quantize(w, h, amax=amax)
    parts = [quantize(w[i:i + 24], h, amax=amax) for i in range(0, w.shape[0], 24)]
    for j in (0, 1):
        assert np.array_equal(full[j], np.concatenate([p[j] for p in parts]))


def test_dead_inputs_and_zero_weights():
    h = _hessian(5, 64)
    h[:, 7] = h[7, :] = 0
    assert np.isfinite(inverse_hessian_factor(h)).all()
    result = quantize(np.zeros((3, 64), np.float32), h)
    assert not dequantize_nvfp4(*result).any()


@pytest.mark.parametrize("w", [np.ones((2, 17)), np.full((1, 16), np.nan)])
def test_rejects_invalid_weights(w):
    with pytest.raises(ValueError):
        quantize(w, np.eye(w.shape[-1], dtype=np.float32))


def test_rejects_a_mismatched_hessian():
    with pytest.raises(ValueError):
        quantize(_weights(6, (4, 64)), np.eye(32, dtype=np.float32))


def _manifest():
    from bittrellis.manifest import Manifest

    return Manifest.from_dict({
        "schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "test-gptq", "default": "NVFP4",
        "rules": [{"match": "L*.mlp", "format": "NVFP4", "quantizer": "nvfp4_gptq"},
                  {"match": "L*.attn.*", "format": "NVFP4", "quantizer": "nvfp4_blockfit"},
                  {"match": "L*.gdn.*", "format": "NVFP4", "quantizer": "nvfp4_blockfit"}],
    })


def test_tiny_build_is_repeatable_and_audited(tiny_all, tmp_path, monkeypatch):
    from bittrellis.build import build
    from bittrellis.safetensors_io import SafeTensorsDir
    from bittrellis.track import load_track
    from bittrellis.validate import audit

    monkeypatch.setenv("BITTRELLIS_REPLAY_CACHE", str(tmp_path / "cache"))
    calib = make_tiny_calibration(tmp_path / "calib", seed=0)
    sources = {**dict(zip(("base", "gittensor_nvfp4", "unsloth_nvfp4"), tiny_all, strict=True)), "calibration": calib}
    first, second = tmp_path / "first", tmp_path / "second"
    record = build(_manifest(), load_track("HPC-01"), sources, first, **QUIET)
    again = build(_manifest(), load_track("HPC-01"), sources, second, **QUIET)
    assert record["files"] == again["files"]
    result = audit(first, _manifest(), sources, verify_sources=False)
    assert result.ok, result.errors
    assert result.lineage["nvfp4_gptq@v1"]["replayed_tensors"] > 0

    # down_proj has no statistics: it carries the blockfit MLP bytes; gate/up do not
    blockfit = tmp_path / "blockfit"
    plain = _manifest()
    plain.rules[0]["quantizer"] = "nvfp4_blockfit"
    build(plain, load_track("HPC-01"), sources, blockfit, **QUIET)
    with SafeTensorsDir(first) as a, SafeTensorsDir(blockfit) as b:
        down = [n for n in a.tensors if n.endswith("mlp.down_proj.weight")]
        gate = [n for n in a.tensors if n.endswith("mlp.gate_proj.weight")]
        assert down and all(bytes(a.raw(n)) == bytes(b.raw(n)) for n in down)
        assert any(bytes(a.raw(n)) != bytes(b.raw(n)) for n in gate)

    # other statistics regenerate other bytes, so the audit rejects the checkpoint
    monkeypatch.setenv("BITTRELLIS_REPLAY_CACHE", str(tmp_path / "cache2"))
    other = {**sources, "calibration": make_tiny_calibration(tmp_path / "other", seed=5)}
    assert not audit(first, _manifest(), other, verify_sources=False).ok


def test_refuses_to_build_without_the_statistics(tiny_all, tmp_path):
    from bittrellis.build import build
    from bittrellis.track import load_track

    sources = dict(zip(("base", "gittensor_nvfp4", "unsloth_nvfp4"), tiny_all, strict=True))
    with pytest.raises(ValueError, match="needs the calibration"):
        build(_manifest(), load_track("HPC-01"), sources, tmp_path / "x", **QUIET)


def test_the_fingerprint_probe_is_repeatable_and_not_blockfit():
    from bittrellis.fingerprint import by_quantizer, probe, similarity

    first = probe(["nvfp4_gptq"], seed=3)
    assert first and first == probe(["nvfp4_gptq"], seed=3)
    ours = by_quantizer(first)["nvfp4_gptq@v1"]
    fit = by_quantizer(probe(["nvfp4_blockfit"], seed=3))["nvfp4_blockfit@v1"]
    sim, compared = similarity(ours, fit)
    assert compared and sim < 0.99
