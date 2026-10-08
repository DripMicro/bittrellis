import json

import numpy as np
import pytest
import yaml

gguf = pytest.importorskip("gguf")

from bittrellis import hpc02  # noqa: E402
from bittrellis.gguf_build import build_gguf, default_encoder  # noqa: E402

QUIET = {"log": lambda *_: None}


def bf16(a):
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def write_gguf(path, tensors, fields=True):
    w = gguf.GGUFWriter(str(path), "qwen35moe")
    if fields:
        w.add_uint32("qwen35moe.block_count", 2)
        w.add_array("tokenizer.ggml.tokens", ["a", "b"])
    for n, (v, kind) in tensors.items():
        if kind == "f32":
            w.add_tensor(n, np.asarray(v, np.float32))
        else:
            w.add_tensor(n, bf16(v), raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture()
def tiny(tmp_path):
    rng = np.random.default_rng(0)
    r = lambda *s: rng.standard_normal(s) * 0.02  # noqa: E731
    t = {"token_embd.weight": (r(64, 256), "bf16"), "output.weight": (r(64, 256), "bf16"),
         "output_norm.weight": (r(256), "f32")}
    for i in (0, 1):
        b = f"blk.{i}."
        t[b + "attn_norm.weight"] = (r(256), "f32")
        t[b + "ffn_gate_inp.weight"] = (r(4, 256), "f32")
        t[b + "ffn_gate_inp_shexp.weight"] = (r(1, 256), "f32")    # stored [256, 1] like the real template
        t[b + "ffn_gate_exps.weight"] = (r(4, 32, 256), "bf16")
        t[b + "ffn_up_exps.weight"] = (r(4, 32, 256), "bf16")
        t[b + "ffn_down_exps.weight"] = (r(4, 256, 256), "bf16")
        t[b + "ffn_gate_shexp.weight"] = (r(32, 256), "bf16")
        t[b + "ffn_up_shexp.weight"] = (r(32, 256), "bf16")
        t[b + "ffn_down_shexp.weight"] = (r(256, 256), "bf16")
        t[b + "attn_qkv.weight"] = (r(96, 256), "bf16")
        t[b + "attn_gate.weight"] = (r(64, 256), "bf16")
        t[b + "ssm_out.weight"] = (r(256, 256), "bf16")
        t[b + "ssm_alpha.weight"] = (r(4, 256), "bf16")
    tdir = tmp_path / "template"
    tdir.mkdir()
    write_gguf(tdir / "t.gguf", t)
    # a stand-in for the attested UD file: Q8_0 on attn_qkv, Q4_K experts
    template = hpc02.read_template(tdir)
    assign = {u.tensor: ("Q8_0" if u.kind == "gdn.qkv" else "Q4_K", default_encoder) for u in hpc02.units(template)}
    build_gguf(hpc02.template_paths(tdir), tmp_path / "ud.gguf", assign, jobs=1, **QUIET)
    return tdir, tmp_path / "ud.gguf"


def manifest(tmp_path, **kw):
    d = {"schema": "bittrellis/manifest@2", "track": "HPC-02", "name": "t", "default": "Q4_K", **kw}
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_units_and_expansion(tiny, tmp_path):
    tdir, ud = tiny
    us = hpc02.units(hpc02.read_template(tdir))
    ids = [u.id for u in us]
    assert ids[0] == "embed" and ids[-1] == "lm_head" and "L1.exps.down" in ids and "L0.gdn.z" in ids
    assert not any("ssm_alpha" in u.tensor for u in us)                     # fixed F32, not searchable
    d = yaml.safe_load(manifest(tmp_path, rules=[{"match": "L*.exps.down", "layers": "1", "format": "Q6_K"}]).read_text())
    a = hpc02.expand(d, us)
    assert a["L1.exps.down"].format == "Q6_K" and a["L0.exps.down"].format == "Q4_K"
    with pytest.raises(hpc02.ManifestError, match="unknown format"):
        hpc02.expand({**d, "default": "NVFP4"}, us)
    # a default that later rules override never has to be valid on its own
    ok = {**d, "default": "Q8_0", "encoders": {"Q8_0": "unsloth_ud", "Q4_K": "unsloth_ud"},
          "rules": [{"match": "L*.exps.*", "format": "Q4_K"}, {"match": "L*.gdn.*", "format": "Q4_K", "encoder": "kq_rtn"},
                    {"match": "L*.shexp.*", "format": "Q4_K", "encoder": "kq_rtn"}],
          "modules": {"embed": {"format": "Q4_K", "encoder": "kq_rtn"}, "lm_head": {"format": "Q4_K", "encoder": "kq_rtn"},
                      "L0.gdn.qkv": "Q8_0", "L1.gdn.qkv": "Q8_0"}}
    assert hpc02.expand(ok, us, hpc02.ud_formats(ud))["L0.exps.gate"].encoder == "unsloth_ud"
    with pytest.raises(hpc02.ManifestError, match="unsloth_ud stores"):
        hpc02.expand({**d, "rules": [{"match": "L*.gdn.qkv", "format": "Q4_K", "encoder": "unsloth_ud"}]}, us,
                     hpc02.ud_formats(ud))


def test_build_then_audit_and_tampering_is_caught(tiny, tmp_path):
    tdir, ud = tiny
    m = manifest(tmp_path, rules=[{"match": "L*.exps.*", "format": "Q5_K"},
                                  {"match": "L*.gdn.qkv", "format": "Q8_0", "encoder": "unsloth_ud"},
                                  {"match": "L0.exps.down", "format": "Q6_K"}], modules={"embed": "Q8_0"})
    out = tmp_path / "c.gguf"
    rec = hpc02.build(m, tdir, out, ud=ud, jobs=2, **QUIET)
    res = hpc02.audit(out, m, tdir, ud=ud, secret="s", verify=False, **QUIET)
    assert res["ok"], res["errors"]
    assert rec["candidate_id"] == res["candidate_id"]
    assert res["lineage"]["unsloth_ud@v1"]["checked"] == 2
    r = gguf.GGUFReader(str(out))
    t = {x.name: x for x in r.tensors}
    assert t["blk.0.ffn_down_exps.weight"].tensor_type.name == "Q6_K" and t["blk.0.ssm_alpha.weight"].tensor_type.name == "F32"
    assert list(map(int, t["blk.0.ffn_gate_inp_shexp.weight"].shape)) == [256]
    # a different manifest for the same file fails on formats
    other = manifest(tmp_path, rules=[{"match": "L*.exps.*", "format": "Q6_K"}])
    assert not hpc02.audit(out, other, tdir, ud=ud, verify=False, **QUIET)["ok"]
    # one flipped byte in a regenerable tensor that is always sampled (the first unit) fails the audit
    x = t["token_embd.weight"]
    with open(out, "r+b") as fh:
        fh.seek(int(x.data_offset) + 7)
        b = fh.read(1)
        fh.seek(int(x.data_offset) + 7)
        fh.write(bytes([b[0] ^ 1]))
    bad = hpc02.audit(out, m, tdir, ud=ud, secret="s", verify=False, **QUIET)
    assert not bad["ok"] and any("embed" in e for e in bad["errors"])


def test_committed_unit_list_reads_back():
    us, udf = hpc02.read_units_file()
    assert len(us) == 372 and us[0].id == "embed" and us[-1].id == "lm_head"
    assert sum(u.kind.startswith("exps.") for u in us) == 120 and set(udf.values()) <= set(hpc02.FORMATS)
    assert hpc02.units_doc(us, udf) == json.loads(hpc02.UNITS_FILE.read_text())   # round trip
