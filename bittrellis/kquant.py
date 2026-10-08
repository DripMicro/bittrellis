"""GGUF block formats for the HPC-02 track (llama.cpp layouts, as SparkInfer reads them).

Rows of float weights in, the bytes of one GGUF tensor out. The byte layouts are the inverse of gguf-py's
`dequantize_blocks` (gguf 0.19.0), which is the specification here: tests decode every encoding with gguf-py.

`rtn` encoders are the track's plain round-to-nearest baseline (built-in `kq_rtn`): per sub-block scale and
minimum from the block's own range, super-block scales from the largest sub-block scale. They are not
llama.cpp's iterative search (make_qkx2_quants); better rounding is what contributed encoders compete on.
Everything is float64 NumPy, deterministic.
"""

from __future__ import annotations

import numpy as np

QK_K = 256
# bytes per block and values per block
BLOCK = {"Q4_K": (144, 256), "Q5_K": (176, 256), "Q6_K": (210, 256), "Q8_0": (34, 32), "F32": (4, 1)}


def row_bytes(fmt: str, cols: int) -> int:
    nb, nv = BLOCK[fmt]
    if cols % nv:
        raise ValueError(f"{fmt} needs rows of a multiple of {nv} values, got {cols}")
    return cols // nv * nb


def _f16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16)


def _pack_scales_q4k(sc: np.ndarray, m: np.ndarray) -> np.ndarray:
    """6-bit sub-block scales and minimums [n, 8] -> the 12 packed bytes (see gguf-py Q4_K.get_scale_min)."""
    sc, m = sc.astype(np.uint8), m.astype(np.uint8)
    out = np.empty((len(sc), 12), np.uint8)
    out[:, 0:4] = (sc[:, 0:4] & 0x3F) | ((sc[:, 4:8] >> 4) << 6)
    out[:, 4:8] = (m[:, 0:4] & 0x3F) | ((m[:, 4:8] >> 4) << 6)
    out[:, 8:12] = (sc[:, 4:8] & 0x0F) | ((m[:, 4:8] & 0x0F) << 4)
    return out


def _asym_k(x: np.ndarray, nmax: int):
    """Shared Q4_K/Q5_K quantisation: x [n, 8, 32] -> (d, dmin as f16, sc, m [n, 8], q [n, 8, 32])."""
    lo = np.minimum(x.min(-1), 0.0)
    hi = x.max(-1)
    scale = (hi - lo) / nmax
    mins = -lo
    d = _f16(scale.max(-1) / 63.0)
    dmin = _f16(mins.max(-1) / 63.0)
    df, dmf = d.astype(np.float64), dmin.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(df[:, None] > 0, np.clip(np.rint(scale / df[:, None]), 0, 63), 0)
        m = np.where(dmf[:, None] > 0, np.clip(np.rint(mins / dmf[:, None]), 0, 63), 0)
        step = df[:, None] * sc
        q = np.where(step[..., None] > 0, np.rint((x + (dmf[:, None] * m)[..., None]) / step[..., None]), 0)
    return d, dmin, sc, m, np.clip(q, 0, nmax).astype(np.uint8)


def quantize_q4k(w: np.ndarray) -> bytes:
    x = np.asarray(w, np.float64).reshape(-1, 8, 32)
    d, dmin, sc, m, q = _asym_k(x, 15)
    qs = (q[:, 0::2, :] | (q[:, 1::2, :] << 4)).reshape(len(x), 128)
    blocks = np.concatenate([d.view(np.uint8).reshape(-1, 2), dmin.view(np.uint8).reshape(-1, 2),
                             _pack_scales_q4k(sc, m), qs], axis=1)
    return blocks.tobytes()


def quantize_q5k(w: np.ndarray) -> bytes:
    x = np.asarray(w, np.float64).reshape(-1, 8, 32)
    d, dmin, sc, m, q = _asym_k(x, 31)
    lo, hi = q & 0x0F, q >> 4                                   # hi: one bit
    qs = (lo[:, 0::2, :] | (lo[:, 1::2, :] << 4)).reshape(len(x), 128)
    qh = np.zeros((len(x), 32), np.uint8)
    for j in range(8):                                          # bit j of byte i: sub-block j, value i
        qh |= (hi[:, j, :] & 1) << j
    blocks = np.concatenate([d.view(np.uint8).reshape(-1, 2), dmin.view(np.uint8).reshape(-1, 2),
                             _pack_scales_q4k(sc, m), qh, qs], axis=1)
    return blocks.tobytes()


def quantize_q6k(w: np.ndarray) -> bytes:
    x = np.asarray(w, np.float64).reshape(-1, 16, 16)
    n = len(x)
    amax = np.abs(x).max(-1)
    scale = amax / 31.0
    d = _f16(np.abs(scale).max(-1) / 127.0)
    df = d.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(df[:, None] > 0, np.clip(np.rint(scale / df[:, None]), -128, 127), 0)
        step = df[:, None] * sc
        q = np.where(step[..., None] > 0, np.rint(x / step[..., None]), 0)
    q = (np.clip(q, -32, 31) + 32).astype(np.uint8).reshape(n, 256)   # 0..63, element e of the block
    # element e: row r = e // 32 (8 rows), column c = e % 32; half h = r // 4, rr = r % 4
    r = np.arange(256) // 32
    c = np.arange(256) % 32
    h, rr = r // 4, r % 4
    ql = np.zeros((n, 128), np.uint8)
    qh = np.zeros((n, 64), np.uint8)
    ql_idx = h * 64 + (rr % 2) * 32 + c
    ql_shift = (rr // 2) * 4
    qh_idx = h * 32 + c
    qh_shift = rr * 2
    for e in range(256):
        ql[:, ql_idx[e]] |= ((q[:, e] & 0x0F) << np.uint8(ql_shift[e])).astype(np.uint8)
        qh[:, qh_idx[e]] |= ((q[:, e] >> 4) << np.uint8(qh_shift[e])).astype(np.uint8)
    blocks = np.concatenate([ql, qh, sc.astype(np.int8).view(np.uint8), d.view(np.uint8).reshape(-1, 2)], axis=1)
    return blocks.tobytes()


def quantize_q8_0(w: np.ndarray) -> bytes:
    """ggml's Q8_0: one f16 scale (amax / 127) and 32 int8 values per block. float64, so a block whose scale
    is tiny or zero rounds to 0 instead of producing an undefined cast."""
    x = np.asarray(w, np.float64).reshape(-1, 32)
    d = _f16(np.abs(x).max(-1) / 127.0)
    df = d.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(df[:, None] > 0, np.rint(x / df[:, None]), 0)
    q = np.clip(q, -127, 127).astype(np.int8)
    return np.concatenate([d.view(np.uint8).reshape(-1, 2), q.view(np.uint8)], axis=1).tobytes()


def to_f32(w: np.ndarray) -> bytes:
    """Lossless for BF16 sources (small recurrent tensors the loader reads as F32)."""
    return np.asarray(w, "<f4").tobytes()


CHUNK_ROWS = 4096   # rows are independent: encoding in chunks bounds memory (an expert tensor has 131,072 rows)


def chunked(fn):
    """`fn` applied to row chunks; the bytes are those of one call over all rows."""
    def run(w: np.ndarray) -> bytes:
        w = np.asarray(w)
        w = w.reshape(-1, w.shape[-1])
        return b"".join(fn(w[r:r + CHUNK_ROWS]) for r in range(0, len(w), CHUNK_ROWS))
    return run


RTN = {k: chunked(f) for k, f in {"Q4_K": quantize_q4k, "Q5_K": quantize_q5k, "Q6_K": quantize_q6k,
                                    "Q8_0": quantize_q8_0, "F32": to_f32}.items()}


def dequantize(fmt: str, data: bytes, cols: int) -> np.ndarray:
    """Decode with gguf-py (the reference decoder) to float32 rows."""
    from gguf import quants

    if fmt == "F32":
        return np.frombuffer(data, "<f4").reshape(-1, cols).astype(np.float32)
    cls = getattr(quants, fmt)
    nb, nv = BLOCK[fmt]
    blocks = np.frombuffer(data, np.uint8).reshape(-1, nb)
    return cls.dequantize_blocks(blocks).reshape(-1, cols).astype(np.float32)
