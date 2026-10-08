import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from bittrellis import kquant  # noqa: E402
from bittrellis.gguf_build import build_gguf, default_encoder  # noqa: E402


def bf16(a):
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


@pytest.fixture()
def template(tmp_path):
    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(str(tmp_path / "t.gguf"), "qwen35moe")
    w.add_uint32("qwen35moe.block_count", 1)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    vals = {"blk.0.attn_qkv.weight": rng.standard_normal((64, 512)) * 0.02,
            "blk.0.ffn_up_exps.weight": rng.standard_normal((4, 32, 256)) * 0.02,
            "blk.0.attn_norm.weight": rng.standard_normal(512)}
    for n, v in vals.items():
        if n.endswith("norm.weight"):
            w.add_tensor(n, v.astype(np.float32))
        else:
            w.add_tensor(n, bf16(v), raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return tmp_path / "t.gguf", vals


def test_assigned_tensors_are_encoded_and_the_rest_copied(template, tmp_path):
    path, vals = template
    assign = {"blk.0.attn_qkv.weight": ("Q4_K", default_encoder), "blk.0.ffn_up_exps.weight": ("Q6_K", default_encoder)}
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    build_gguf([path], a, assign, jobs=1, log=lambda *_: None)
    build_gguf([path], b, assign, jobs=2, log=lambda *_: None)
    assert a.read_bytes() == b.read_bytes()                     # deterministic, serial == parallel
    r, t = gguf.GGUFReader(str(a)), gguf.GGUFReader(str(path))
    assert r.fields["tokenizer.ggml.tokens"].contents() == ["a", "b", "c"]
    assert r.fields["qwen35moe.block_count"].contents() == 1
    out = {x.name: x for x in r.tensors}
    assert out["blk.0.attn_qkv.weight"].tensor_type.name == "Q4_K"
    assert out["blk.0.ffn_up_exps.weight"].tensor_type.name == "Q6_K"
    assert list(out["blk.0.ffn_up_exps.weight"].shape) == list(next(x for x in t.tensors if x.name == "blk.0.ffn_up_exps.weight").shape)
    for name, fmt in (("blk.0.attn_qkv.weight", "Q4_K"), ("blk.0.ffn_up_exps.weight", "Q6_K")):
        orig = vals[name].reshape(-1, vals[name].shape[-1])
        dq = kquant.dequantize(fmt, np.asarray(out[name].data).tobytes(), orig.shape[1])
        assert np.linalg.norm(dq - orig) / np.linalg.norm(orig) < (0.09 if fmt == "Q4_K" else 0.03)
    norm = next(x for x in t.tensors if x.name == "blk.0.attn_norm.weight")
    assert np.asarray(out["blk.0.attn_norm.weight"].data).tobytes() == np.asarray(norm.data).tobytes()


def test_formats_decode_with_the_reference_decoder():
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((8, 512)) * 0.02).astype(np.float32)
    errs = {}
    for fmt in ("Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        data = kquant.RTN[fmt](w)
        assert len(data) == 8 * kquant.row_bytes(fmt, 512)
        errs[fmt] = np.linalg.norm(kquant.dequantize(fmt, data, 512) - w) / np.linalg.norm(w)
    assert errs["Q8_0"] < errs["Q6_K"] < errs["Q5_K"] < errs["Q4_K"] < 0.1   # every extra bit helps


def test_chunked_encoding_gives_the_same_bytes(monkeypatch):
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((50, 512)) * 0.02).astype(np.float32)
    whole = {f: kquant.RTN[f](w) for f in ("Q4_K", "Q5_K", "Q6_K", "Q8_0")}
    monkeypatch.setattr(kquant, "CHUNK_ROWS", 7)
    assert {f: kquant.RTN[f](w) for f in whole} == whole
