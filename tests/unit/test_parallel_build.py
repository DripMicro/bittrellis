"""A parallel build writes the same bytes as a serial one."""

import multiprocessing

import bittrellis.build as B
from bittrellis.build import build
from bittrellis.manifest import Manifest
from bittrellis.track import load_track


def test_parallel_build_is_byte_identical(tiny_all, tmp_path, monkeypatch):
    manifest = Manifest.from_dict({
        "schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "test-parallel", "default": "NVFP4",
        "rules": [{"match": "L*.mlp", "format": "NVFP4", "quantizer": "rtn"},
                  {"match": "L*.gdn.qkv", "format": "FP8"},
                  {"match": "L*.attn.*", "format": "Q4_K"}],
    })
    sources = dict(zip(("base", "gittensor_nvfp4", "unsloth_nvfp4"), tiny_all, strict=True))
    pools = []
    real = multiprocessing.get_context

    def counting(method):
        pools.append(method)
        return real(method)

    monkeypatch.setattr(B.multiprocessing, "get_context", counting)
    records = {}
    for jobs in ("1", "3"):
        monkeypatch.setenv("BITTRELLIS_BUILD_JOBS", jobs)
        records[jobs] = build(manifest, load_track("HPC-01"), sources, tmp_path / jobs, verify=False, log=lambda *_: None)
    assert records["1"]["files"] == records["3"]["files"]
    assert records["1"]["candidate_id"] == records["3"]["candidate_id"]
    assert pools == ["fork"]                                       # the 3-job build really used workers
