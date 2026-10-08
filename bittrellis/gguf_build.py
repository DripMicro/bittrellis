"""Build a GGUF checkpoint for the HPC-02 track from a pinned BF16 GGUF template.

The template (llama.cpp's own conversion of the BF16 model) supplies every metadata key and every tensor's
name, shape and order. Each tensor a recipe assigns is re-encoded from the template's BF16 values into its
GGUF block format; every other tensor is copied byte for byte. So the layout, tokenizer and model settings
are exactly what llama.cpp and SparkInfer expect, and only searchable weights change.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import kquant

SKIP_KEYS = ("GGUF.", "split.")


def _bf16_rows(t) -> np.ndarray:
    """A template tensor's values as float32 rows [rows, cols] (numpy order: cols = ne0)."""
    from gguf import GGMLQuantizationType as T

    raw = np.asarray(t.data)
    if t.tensor_type == T.BF16:
        u = raw.view(np.uint16).reshape(-1)
        out = (u.astype(np.uint32) << 16).view(np.float32)
    elif t.tensor_type == T.F32:
        out = raw.view(np.float32).reshape(-1)
    elif t.tensor_type == T.F16:
        out = raw.view(np.float16).reshape(-1).astype(np.float32)
    else:
        raise ValueError(f"{t.name}: template tensor is {t.tensor_type.name}, expected BF16/F16/F32")
    cols = int(t.shape[0])                     # gguf-py lists ne0 (the row length) first
    return out.reshape(-1, cols)


def _copied(t) -> np.ndarray:
    """A copied tensor's data. A [n, 1] vector (the template stores ffn_gate_inp_shexp so) is written with
    rank 1, as llama.cpp's quantized releases and SparkInfer's loader have it; the bytes are unchanged."""
    data = np.asarray(t.data)
    if len(t.shape) == 2 and int(t.shape[1]) == 1:
        data = data.reshape(-1)
    return data


_JOB: dict = {}


def _encode(name: str) -> bytes:
    t, fmt, enc = _JOB["tensors"][name], *_JOB["assign"][name]
    return enc(fmt, _bf16_rows(t))


def default_encoder(fmt: str, rows: np.ndarray) -> bytes:
    return kquant.RTN[fmt](rows)


def build_gguf(template: list[Path], out: Path, assign: dict[str, tuple[str, Callable]], jobs: int | None = None,
               log=print) -> dict:
    """`assign`: {tensor name: (format, encoder(fmt, float32 rows) -> bytes)}; other tensors are copied."""
    from gguf import GGMLQuantizationType as T
    from gguf import GGUFReader, GGUFValueType, GGUFWriter

    readers = [GGUFReader(p) for p in template]
    first = readers[0]
    arch = first.fields["general.architecture"].contents()
    tensors = {t.name: t for r in readers for t in r.tensors}
    unknown = sorted(set(assign) - set(tensors))
    if unknown:
        raise ValueError(f"not in the template: {unknown[:5]}")
    w = GGUFWriter(str(out), arch)
    for f in first.fields.values():
        if f.name == "general.architecture" or f.name.startswith(SKIP_KEYS):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == GGUFValueType.ARRAY else None)
    for name, t in tensors.items():
        if name in assign:
            fmt = assign[name][0]
            cols = int(t.shape[0])
            rows = int(np.prod(t.shape[1:])) if len(t.shape) > 1 else 1
            byte_shape = tuple(int(s) for s in reversed(t.shape[1:])) + (kquant.row_bytes(fmt, cols),)
            w.add_tensor_info(name, byte_shape, np.dtype(np.uint8), rows * kquant.row_bytes(fmt, cols), raw_dtype=T[fmt])
        else:
            data = _copied(t)
            w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    names = list(tensors)
    todo = [n for n in names if n in assign]
    jobs = jobs or max(1, min(12, (os.cpu_count() or 2) // 2))
    _JOB.update(tensors=tensors, assign=assign)
    pool = multiprocessing.get_context("fork").Pool(jobs) if jobs > 1 and todo else None
    try:
        results = iter(pool.imap(_encode, todo)) if pool else iter(_encode(n) for n in todo)
        for i, name in enumerate(names):
            t = tensors[name]
            if name in assign:
                fmt = assign[name][0]
                data = next(results)
                byte_shape = tuple(int(s) for s in reversed(t.shape[1:])) + (kquant.row_bytes(fmt, int(t.shape[0])),)
                w.write_tensor_data(np.frombuffer(data, np.uint8).reshape(byte_shape))
            else:
                w.write_tensor_data(_copied(t))
            if (i + 1) % 100 == 0:
                log(f"[gguf] {i + 1}/{len(names)} tensors")
    finally:
        if pool:
            pool.terminate()
            pool.join()
        _JOB.clear()
    w.close()
    return {"tensors": len(names), "encoded": len(todo), "bytes": out.stat().st_size}
