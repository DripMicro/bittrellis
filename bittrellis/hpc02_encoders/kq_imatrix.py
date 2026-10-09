"""K-quant with an importance matrix: Q4_K / Q5_K / Q6_K grids searched per super-block on the pinned statistics.

`kq_rtn` fits every sub-block to its own range and every super-block scale to its largest sub-block. This encoder
keeps llama.cpp's byte layouts and chooses the bytes to minimise the importance-weighted error of each 256-value
super-block, sum_i w_i (x_i - q_i)^2, the way llama.cpp's `--imatrix` quantization does (no error feedback between
columns, no rotation: each super-block on its own):

* weights w_i: the input channel's mean square activation. Routed experts read their own expert's diagonal
  (`exps_input.sumsq[e] / count[e]` for gate/up, `exps_down.sumsq[e] / count[e]` for down; expert e's rows are
  e * rows/256 ... (e+1) * rows/256), shrunk towards the layer's mean expert by `count / (count + 8)` so a rarely
  routed expert is not fitted to a handful of tokens. Dense units read the diagonal of their input's x x^T
  (attention / recurrent q/k/v/z and qkv: `attn_input.xtx`; shared expert gate/up: `ffn_input.xtx`; shared expert
  down: `ffn_down_shexp.xtx`). Units without statistics (attention / recurrent outputs, lm_head, embeddings, or no
  calibration at all) use x^2 plus the super-block's mean x^2.
* Q4_K / Q5_K (8 sub-blocks of 32, asymmetric): per sub-block, llama.cpp's make_qkx3_quants search (the range
  divided by nmax - 0.9 + 0.05 k, k = 0..36, each rounding followed by a weighted least-squares refit of scale and
  minimum); the eight scales and minimums then become 6-bit multiples of f16 super-scales chosen by a weighted search
  (make_qp_quants, weights = each sub-block's total importance).
* Q6_K (16 sub-blocks of 16, symmetric, 8-bit scales): per sub-block llama.cpp's make_qx_quants search (19 steps
  around the largest value, weighted least-squares scale), int8 scales under an f16 super-scale.
* Then, beyond llama.cpp, two rounds of integer refinement on the stored grid: each sub-block tries its neighbouring
  6-bit scale/minimum codes (Q6_K: scale codes +-2) with the values re-rounded, and the f16 super-scale(s) are refitted
  by weighted least squares for the chosen codes; a change is kept only where the weighted error drops.

Q8_0 keeps `kq_rtn`'s bytes. Determinism: float64 NumPy on the CPU, element-wise operations and reductions over the
last axis only (no BLAS), fixed chunks of rows; rows are independent, so the bytes do not depend on the chunking.
"""

from __future__ import annotations

import numpy as np

from bittrellis import kquant
from bittrellis.hpc02 import Encoder, register

N_EXPERTS = 256
SHRINK = 8.0              # tokens of the layer-mean prior blended into each expert's importance
FLOOR = 1e-4              # minimum weight relative to the row's mean importance
CHUNK_SB = 4096           # super-blocks per vectorised step
REFINE_ROUNDS = 2
NMAX = {"Q4_K": 15, "Q5_K": 31}
DENSE = {"gdn.qkv": "attn_input", "gdn.z": "attn_input", "attn.q": "attn_input", "attn.k": "attn_input",
         "attn.v": "attn_input", "shexp.gate": "ffn_input", "shexp.up": "ffn_input", "shexp.down": "ffn_down_shexp"}
EXPERT_STAT = {"exps.gate": "exps_input", "exps.up": "exps_input", "exps.down": "exps_down"}


def _rint(a):
    return np.rint(a)


# ------------------------------------------------------------------ Q4_K / Q5_K


def _wsum(w, a):
    return (w * a).sum(-1)


def _search_qkx3(x: np.ndarray, w: np.ndarray, nmax: int):
    """llama.cpp make_qkx3_quants, vectorised. x, w [..., 32] -> (scale, min) [...] with x ~ scale * L - min, min >= 0."""
    xmin = np.minimum(x.min(-1), 0.0)
    xmax = x.max(-1)
    rng = xmax - xmin
    sw, swx, swxx = w.sum(-1), _wsum(w, x), _wsum(w, x * x)
    with np.errstate(divide="ignore", invalid="ignore"):
        iscale = np.where(rng > 0, nmax / rng, 0.0)
        best_s = np.where(rng > 0, rng / nmax, 0.0)
    best_m = -xmin
    lv = np.clip(_rint(iscale[..., None] * (x - xmin[..., None])), 0, nmax)
    best_e = _wsum(w, (best_s[..., None] * lv - best_m[..., None] - x) ** 2)
    xs = x - xmin[..., None]
    wx = w * x
    for k in range(37):
        with np.errstate(divide="ignore", invalid="ignore"):
            isc = np.where(rng > 0, (nmax - 0.9 + 0.05 * k) / rng, 0.0)
        lv = np.clip(_rint(isc[..., None] * xs), 0, nmax)
        wl = w * lv
        sl, sl2, sxl = wl.sum(-1), (wl * lv).sum(-1), (wx * lv).sum(-1)
        det = sw * sl2 - sl * sl
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(det > 0, (sw * sxl - swx * sl) / det, 0.0)
            mn = np.where(det > 0, (sl2 * swx - sl * sxl) / det, 0.0)          # x ~ s * L + mn, mn <= 0 wanted
            pos = mn > 0
            s = np.where(pos, np.where(sl2 > 0, sxl / sl2, 0.0), s)
            mn = np.where(pos, 0.0, mn)
        # sum w (s L + mn - x)^2 from the moments already summed
        e = swxx + s * s * sl2 + mn * mn * sw - 2 * s * sxl - 2 * mn * swx + 2 * s * mn * sl
        take = (det > 0) & (s > 0) & (e < best_e)
        best_e = np.where(take, e, best_e)
        best_s = np.where(take, s, best_s)
        best_m = np.where(take, -mn, best_m)
    return best_s, best_m


def _search_qp(v: np.ndarray, w: np.ndarray, nmax: int = 63) -> np.ndarray:
    """llama.cpp make_qp_quants: f16 super-scale for non-negative values v [n, k] with weights w [n, k]."""
    vmax = v.max(-1)
    best_d = np.zeros(len(v))
    best_e = np.full(len(v), np.inf)
    for i in range(-4, 5):
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(vmax > 0, vmax / (nmax + 0.1 * i), 0.0)
            lv = np.where(d[:, None] > 0, np.clip(_rint(v / d[:, None]), 0, nmax), 0)
        e = _wsum(w, (v - d[:, None] * lv) ** 2)
        take = e < best_e
        best_e, best_d = np.where(take, e, best_e), np.where(take, d, best_d)
    # weighted least-squares refit for the chosen codes
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.where(best_d[:, None] > 0, np.clip(_rint(v / best_d[:, None]), 0, nmax), 0)
        sl2 = _wsum(w, lv * lv)
        d = np.where(sl2 > 0, _wsum(w, lv * v) / sl2, best_d)
    lv2 = np.where(d[:, None] > 0, np.clip(_rint(v / np.where(d > 0, d, 1)[:, None]), 0, nmax), 0)
    e2 = _wsum(w, (v - d[:, None] * lv2) ** 2)
    return np.where(e2 < best_e, d, best_d).astype(np.float16)


def _codes(v, d16, nmax=63):
    df = d16.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(df[:, None] > 0, np.clip(_rint(v / df[:, None]), 0, nmax), 0)


def _levels_asym(x, step, off, nmax):
    """x [n, 8, 32], step/off [n, 8] -> levels and the weighted-error-ready reconstruction."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.where(step[..., None] > 0, np.clip(_rint((x + off[..., None]) / step[..., None]), 0, nmax), 0)
    return lv, step[..., None] * lv - off[..., None]


def fit_asym(x: np.ndarray, w: np.ndarray, nmax: int):
    """x, w [n, 256] -> (d, dmin f16 [n], sc, m uint8 [n, 8], L uint8 [n, 256])."""
    n = len(x)
    x = x.reshape(n, 8, 32)
    w = w.reshape(n, 8, 32)
    s, mn = _search_qkx3(x, w, nmax)
    sw = w.sum(-1)
    d = _search_qp(s, sw)
    dmin = _search_qp(mn, sw)
    sc, m = _codes(s, d), _codes(mn, dmin)

    def err_of(d, dmin, sc, m):
        lv, r = _levels_asym(x, d.astype(np.float64)[:, None] * sc, dmin.astype(np.float64)[:, None] * m, nmax)
        return lv, _wsum(w, (r - x) ** 2)                                     # [n, 8]

    lv, e = err_of(d, dmin, sc, m)
    for _ in range(REFINE_ROUNDS):
        # 1) neighbouring 6-bit codes per sub-block, super-scales fixed
        for ds in (-1, 0, 1):
            for dm in (-1, 0, 1):
                if ds == 0 and dm == 0:
                    continue
                sc2, m2 = np.clip(sc + ds, 0, 63), np.clip(m + dm, 0, 63)
                lv2, e2 = err_of(d, dmin, sc2, m2)
                take = e2 < e
                sc, m, e = np.where(take, sc2, sc), np.where(take, m2, m), np.where(take, e2, e)
                lv = np.where(take[..., None], lv2, lv)
        # 2) weighted least-squares refit of (d, dmin) for the chosen codes and levels: x ~ d*(sc L) - dmin*m
        a = sc[..., None] * lv
        b = np.broadcast_to(m[..., None], x.shape)
        waa, wbb, wab = _wsum(w, a * a).sum(-1), _wsum(w, b * b).sum(-1), _wsum(w, a * b).sum(-1)
        wax, wbx = _wsum(w, a * x).sum(-1), _wsum(w, b * x).sum(-1)
        det = waa * wbb - wab * wab
        with np.errstate(divide="ignore", invalid="ignore"):
            dn = np.where(det > 0, (wbb * wax - wab * wbx) / det, d.astype(np.float64))
            dmn = np.where(det > 0, (wab * wax - waa * wbx) / det, dmin.astype(np.float64))
        ok = (det > 0) & (dn > 0) & (dmn >= 0) & np.isfinite(dn) & np.isfinite(dmn) & (dn < 65000) & (dmn < 65000)
        d2 = np.where(ok, dn, d.astype(np.float64)).astype(np.float16)
        dmin2 = np.where(ok, dmn, dmin.astype(np.float64)).astype(np.float16)
        lv2, e2 = err_of(d2, dmin2, sc, m)
        take = e2.sum(-1) < e.sum(-1)
        d, dmin = np.where(take, d2, d), np.where(take, dmin2, dmin)
        e = np.where(take[:, None], e2, e)
        lv = np.where(take[:, None, None], lv2, lv)
    return d, dmin, sc.astype(np.uint8), m.astype(np.uint8), lv.reshape(n, 256).astype(np.uint8)


# ------------------------------------------------------------------ Q6_K


def _search_qx(x: np.ndarray, w: np.ndarray, nmax: int = 32) -> np.ndarray:
    """llama.cpp make_qx_quants (rmse_type 1), vectorised. x, w [..., 16] -> signed scale [...]."""
    idx = np.abs(x).argmax(-1)
    vmax = np.take_along_axis(x, idx[..., None], -1)[..., 0]
    best_s = np.zeros(vmax.shape)
    best_score = np.full(vmax.shape, -1.0)
    for i in range(-9, 10):
        with np.errstate(divide="ignore", invalid="ignore"):
            isc = np.where(vmax != 0, -(nmax + 0.1 * i) / vmax, 0.0)
        lv = np.clip(_rint(isc[..., None] * x), -nmax, nmax - 1)
        sxl, sl2 = _wsum(w, lv * x), _wsum(w, lv * lv)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(sl2 > 0, sxl * sxl / sl2, 0.0)
            s = np.where(sl2 > 0, sxl / sl2, 0.0)
        take = score > best_score
        best_score, best_s = np.where(take, score, best_score), np.where(take, s, best_s)
    return best_s


def _levels_sym(x, step):
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.where(step[..., None] != 0, np.clip(_rint(x / step[..., None]), -32, 31), 0)
    return lv, step[..., None] * lv


def fit_q6k(x: np.ndarray, w: np.ndarray):
    """x, w [n, 256] -> (d f16 [n], sc int [n, 16], L int [n, 256] in -32..31)."""
    n = len(x)
    x = x.reshape(n, 16, 16)
    w = w.reshape(n, 16, 16)
    s = _search_qx(x, w)
    idx = np.abs(s).argmax(-1)
    smax = np.take_along_axis(s, idx[:, None], -1)[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(smax != 0, smax / -128.0, 0.0).astype(np.float16)
        df = d.astype(np.float64)
        sc = np.where(df[:, None] != 0, np.clip(_rint(s / df[:, None]), -128, 127), 0)

    def err_of(d, sc):
        lv, r = _levels_sym(x, d.astype(np.float64)[:, None] * sc)
        return lv, _wsum(w, (r - x) ** 2)

    lv, e = err_of(d, sc)
    for _ in range(REFINE_ROUNDS):
        for ds in (-2, -1, 1, 2):
            sc2 = np.clip(sc + ds, -128, 127)
            lv2, e2 = err_of(d, sc2)
            take = e2 < e
            sc, e = np.where(take, sc2, sc), np.where(take, e2, e)
            lv = np.where(take[..., None], lv2, lv)
        a = sc[..., None] * lv
        saa, sax = _wsum(w, a * a).sum(-1), _wsum(w, a * x).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dn = np.where(saa > 0, sax / saa, d.astype(np.float64))
        ok = np.isfinite(dn) & (np.abs(dn) < 65000) & (dn != 0)
        d2 = np.where(ok, dn, d.astype(np.float64)).astype(np.float16)
        lv2, e2 = err_of(d2, sc)
        take = e2.sum(-1) < e.sum(-1)
        d = np.where(take, d2, d)
        e = np.where(take[:, None], e2, e)
        lv = np.where(take[:, None, None], lv2, lv)
    return d, sc.astype(np.int64), lv.reshape(n, 256).astype(np.int64)


# ------------------------------------------------------------------ packing (kquant.py's layouts)


def pack_q4k(d, dmin, sc, m, q) -> bytes:
    n = len(d)
    qb = q.reshape(n, 8, 32)
    qs = (qb[:, 0::2, :] | (qb[:, 1::2, :] << 4)).reshape(n, 128)
    return np.concatenate([d.view(np.uint8).reshape(-1, 2), dmin.view(np.uint8).reshape(-1, 2),
                           kquant._pack_scales_q4k(sc, m), qs], axis=1).tobytes()


def pack_q5k(d, dmin, sc, m, q) -> bytes:
    n = len(d)
    qb = q.reshape(n, 8, 32)
    lo, hi = qb & 0x0F, qb >> 4
    qs = (lo[:, 0::2, :] | (lo[:, 1::2, :] << 4)).reshape(n, 128)
    qh = np.zeros((n, 32), np.uint8)
    for j in range(8):
        qh |= (hi[:, j, :] & 1) << j
    return np.concatenate([d.view(np.uint8).reshape(-1, 2), dmin.view(np.uint8).reshape(-1, 2),
                           kquant._pack_scales_q4k(sc, m), qh, qs], axis=1).tobytes()


def pack_q6k(d, sc, lv) -> bytes:
    n = len(d)
    q = (lv + 32).astype(np.uint8).reshape(n, 2, 4, 32)                   # half, row-in-half, column
    ql = np.empty((n, 2, 2, 32), np.uint8)
    ql[:, :, 0] = (q[:, :, 0] & 0x0F) | ((q[:, :, 2] & 0x0F) << 4)
    ql[:, :, 1] = (q[:, :, 1] & 0x0F) | ((q[:, :, 3] & 0x0F) << 4)
    qh = (q[:, :, 0] >> 4) | ((q[:, :, 1] >> 4) << 2) | ((q[:, :, 2] >> 4) << 4) | ((q[:, :, 3] >> 4) << 6)
    return np.concatenate([ql.reshape(n, 128), qh.reshape(n, 64), sc.astype(np.int8).view(np.uint8),
                           d.view(np.uint8).reshape(-1, 2)], axis=1).tobytes()


def quantize(x: np.ndarray, w: np.ndarray, fmt: str) -> bytes:
    """Super-blocks x, w [n, 256] (float64, w > 0) -> GGUF `fmt` block bytes."""
    out = []
    for a in range(0, len(x), CHUNK_SB):
        xs, ws = x[a:a + CHUNK_SB], w[a:a + CHUNK_SB]
        if fmt == "Q6_K":
            out.append(pack_q6k(*fit_q6k(xs, ws)))
        else:
            out.append((pack_q4k if fmt == "Q4_K" else pack_q5k)(*fit_asym(xs, ws, NMAX[fmt])))
    return b"".join(out)


# ------------------------------------------------------------------ importance


def _stat(ctx, name: str) -> np.ndarray | None:
    cal = ctx.calibration
    if cal is None or cal.get(name) is None:
        return None
    return np.asarray(cal.array(name, "<f4"), np.float64)


def expert_importance(sumsq: np.ndarray, count: np.ndarray, shrink: float = SHRINK) -> np.ndarray:
    """Per-expert mean x^2 per channel [experts, cols], shrunk towards the token-weighted layer mean."""
    sumsq = np.maximum(np.asarray(sumsq, np.float64), 0)
    count = np.maximum(np.asarray(count, np.float64), 0)
    tot = count.sum()
    prior = sumsq.sum(0) / tot if tot > 0 else np.ones(sumsq.shape[1])
    return (sumsq + shrink * prior[None, :]) / (count + shrink)[:, None]


def importance(ctx) -> np.ndarray | None:
    """[rows, cols] or [1, cols] channel importance for this unit, or None (no usable statistics)."""
    unit, rows = ctx.unit, ctx.rows
    if ctx.calibration is None or unit.layer is None:
        return None
    nrows, cols = rows.shape
    if unit.kind in EXPERT_STAT:
        ss = _stat(ctx, f"blk.{unit.layer}.{EXPERT_STAT[unit.kind]}.sumsq")
        cnt = _stat(ctx, f"blk.{unit.layer}.exps.count")
        if ss is None or cnt is None or ss.shape != (N_EXPERTS, cols) or nrows % N_EXPERTS:
            return None
        imp = expert_importance(ss, cnt)
        return np.repeat(imp, nrows // N_EXPERTS, axis=0)
    if unit.kind in DENSE:
        name = f"blk.{unit.layer}.{DENSE[unit.kind]}.xtx"
        if ctx.calibration.get(name) is None:
            return None
        h = ctx.calibration.array(name, "<f4")
        if h.shape != (cols, cols):
            return None
        return np.maximum(np.diagonal(h).astype(np.float64), 0)[None, :]
    return None


def _weights(x: np.ndarray, imp: np.ndarray | None) -> np.ndarray:
    """Super-block weights [n, 256] for super-blocks x [n, 256]; imp already laid out like x (or None)."""
    if imp is None:
        x2 = x * x
        return x2 + x2.mean(-1, keepdims=True) + 1e-30
    mean = imp.mean(-1, keepdims=True)
    mean = np.where(mean > 0, mean, 1.0)
    return np.maximum(imp, FLOOR * mean) + 1e-30


def _encode(ctx, fmt: str) -> bytes:
    if fmt not in ("Q4_K", "Q5_K", "Q6_K"):
        return kquant.RTN[fmt](ctx.rows)
    rows = np.asarray(ctx.rows)
    rows = rows.reshape(-1, rows.shape[-1])
    nrows, cols = rows.shape
    kquant.row_bytes(fmt, cols)
    imp = importance(ctx)
    nb = cols // 256
    step = max(1, CHUNK_SB // nb)
    out = []
    for r in range(0, nrows, step):
        x = np.asarray(rows[r:r + step], np.float64).reshape(-1, 256)
        if imp is None:
            ib = None
        else:
            src = imp if len(imp) == 1 else imp[r:r + step]
            ib = np.broadcast_to(src, (min(step, nrows - r), cols)).reshape(-1, 256)
        out.append(quantize(x, _weights(x, ib), fmt))
    return b"".join(out)


register(Encoder("kq_imat", 1, "regenerable", _encode))
