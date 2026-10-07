"""NVFP4 with GPTQ rounding of the MLP gate and up projections, driven by the pinned calibration statistics.

Round-to-nearest (and `nvfp4_blockfit`) minimise each weight's own error. What the model sees is the error
of the projection's output, x·(w - q), over the inputs x it actually receives; for one row that is
(w - q) H (w - q)ᵀ with H = mean x xᵀ. Epoch hpc01-e6 pins H for every layer's MLP input
(`input_hessian`), which gate_proj and up_proj share. This encoder minimises that output error with GPTQ:
columns are rounded left to right and each column's rounding error is pushed onto the columns not yet
rounded, through the inverse Hessian, so later roundings correct for earlier ones.

NVFP4 specifics:

* the ModelOpt layout and the max-calibrated tensor scale are kept;
* every 16-column block scale is chosen when GPTQ reaches that block, on the weights as the earlier
  error feedback has left them, by the |e|^1.5 block fit `nvfp4_blockfit` uses for MLP blocks (its
  measured best for MLP weights), over the same E4M3 window;
* values round to the nearest E2M1 level, as everywhere else.

`down_proj` has no pinned statistics in this epoch, so it gets the `nvfp4_blockfit` MLP fit unchanged.

Determinism: NumPy on the CPU in float64 with fixed row chunks and a fixed column block. The only BLAS
calls are the Cholesky factorisation of the damped Hessian and the per-block error propagation; both run on
one machine for a build and its audit replay.
"""

from __future__ import annotations

import numpy as np

from ..precision import NVFP4
from ..quant.formats import E2M1_TABLE, E4M3_TABLE, FP4_E2M1_MAX, e4m3_encode, pack_nibbles
from . import nvfp4_blockfit as bf
from .base import Produced, QuantContext, Quantizer, f32_weight, input_hessian

DAMP = 0.01            # relative diagonal dampening of H, as in GPTQ
BLOCK = 128            # columns per lazy-update block (a multiple of 16)
CHUNK_ROWS = 2048      # rows per pass; GPTQ is independent per row given H
POWER, OFFSETS = bf.FITS["mlp"]

_E4M3_POS = E4M3_TABLE[:127].astype(np.float64)
_E2M1 = E2M1_TABLE.astype(np.float64)
_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# |x|/scale = y falls in bucket ceil(4y) - 1 (clipped to 0..25); every E2M1 rounding midpoint is a bucket
# edge, so the bucket gives the nearest level, ties toward the smaller magnitude (as e2m1_encode).
_BUCKET_LEVEL = bf._BUCKET_LEVEL


def _e2m1_index(a: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Nearest E2M1 level index (0..7) of non-negative `a` at scale `s`, in float64."""
    b = np.ceil(a * (4.0 / s))
    np.subtract(b, 1, out=b)
    np.clip(b, 0, 25, out=b)
    return _BUCKET_LEVEL[b.astype(np.uint8)]


def inverse_hessian_factor(h: np.ndarray, damp: float = DAMP) -> np.ndarray:
    """Upper Cholesky factor U of the damped inverse Hessian (H + λI)⁻¹ = UᵀU, float64."""
    h = np.array(h, dtype=np.float64)
    d = np.diag(h).copy()
    dead = d <= 0
    h[dead, dead] = 1.0
    h[np.diag_indices_from(h)] += damp * float(np.mean(np.diag(h)))
    lo = np.linalg.cholesky(h)
    inv = np.linalg.inv(lo)
    hinv = inv.T @ inv
    return np.linalg.cholesky(hinv).T


def _block_scales(w16: np.ndarray, ws2: float) -> np.ndarray:
    """Per row, the E4M3 code (1..126) of the 16-value block scale with the lowest |e|^1.5 error."""
    a = np.abs(w16)
    rtn = e4m3_encode((a.max(-1) / FP4_E2M1_MAX / ws2).astype(np.float32)).astype(np.int64)
    first = np.where(rtn & 0x80, 0, rtn).clip(1, 126)
    code = np.clip(first[:, None] + np.asarray(OFFSETS)[None, :], 1, 126)           # [rows, k]
    s = (_E4M3_POS[code] * ws2)[..., None]                                          # [rows, k, 1]
    q = _LEVELS[_e2m1_index(np.broadcast_to(a[:, None, :], (a.shape[0], code.shape[1], 16)), s)] * s
    loss = (np.abs(a[:, None, :] - q) ** POWER).sum(-1)                             # [rows, k]
    return code[np.arange(code.shape[0]), np.argmin(loss, axis=1)]


def gptq_rows(w: np.ndarray, u: np.ndarray, ws2: float) -> tuple[np.ndarray, np.ndarray]:
    """GPTQ over float rows `w` [rows, cols] with inverse-Hessian factor `u`.

    Returns (E2M1 codes [rows, cols] uint8, E4M3 block-scale codes [rows, cols/16] uint8)."""
    w = np.array(w, dtype=np.float64)
    rows, cols = w.shape
    codes = np.empty((rows, cols), np.uint8)
    scodes = np.empty((rows, cols // 16), np.uint8)
    for b0 in range(0, cols, BLOCK):
        b1 = min(b0 + BLOCK, cols)
        w1 = w[:, b0:b1].copy()
        err = np.empty_like(w1)
        u1 = u[b0:b1, b0:b1]
        for g in range(0, b1 - b0, 16):
            sc = _block_scales(w1[:, g:g + 16], ws2)
            scodes[:, (b0 + g) // 16] = sc
            s = _E4M3_POS[sc] * ws2
            wg = w1[:, g:g + 16]
            for j in range(16):
                col = wg[:, j]
                lv = _e2m1_index(np.abs(col), s)
                c = np.where(col < 0, lv | 0x8, lv).astype(np.uint8)
                codes[:, b0 + g + j] = c
                e = (col - _E2M1[c] * s) / u1[g + j, g + j]
                err[:, g + j] = e
                if j < 15:
                    wg[:, j + 1:] -= e[:, None] * u1[g + j, g + j + 1:g + 16]
            if g + 16 < b1 - b0:
                w1[:, g + 16:] -= err[:, g:g + 16] @ u1[g:g + 16, g + 16:]
        if b1 < cols:
            w[:, b1:] -= err @ u[b0:b1, b1:]
    return codes, scodes


def _quantize_with(w: np.ndarray, u: np.ndarray, amax: float | None = None) -> tuple[np.ndarray, np.ndarray, np.float32]:
    ws2 = bf.tensor_scale(float(np.abs(w).max()) if amax is None else float(amax))
    packed = np.empty((w.shape[0], w.shape[1] // 2), np.uint8)
    scales = np.empty((w.shape[0], w.shape[1] // 16), np.uint8)
    for r in range(0, w.shape[0], CHUNK_ROWS):
        c, s = gptq_rows(w[r:r + CHUNK_ROWS], u, float(ws2))
        packed[r:r + CHUNK_ROWS], scales[r:r + CHUNK_ROWS] = pack_nibbles(c), s
    return packed, scales, ws2


def quantize(w: np.ndarray, h: np.ndarray, amax: float | None = None) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """float32 [rows, cols] and H [cols, cols] -> ModelOpt NVFP4 (packed, block scales, weight_scale_2)."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % 16 or not np.isfinite(w).all():
        raise ValueError("expected finite [rows, cols] with cols divisible by 16")
    h = np.asarray(h)
    if h.shape != (w.shape[1], w.shape[1]) or not np.isfinite(h).all():
        raise ValueError("expected a finite [cols, cols] input Hessian")
    return _quantize_with(w, inverse_hessian_factor(h), amax)


class NVFP4GPTQ(Quantizer):
    name = "nvfp4_gptq"
    version = 1
    formats = (NVFP4,)
    lineage = "regenerable"
    replay_mode = "independent"
    description = "NVFP4 MLP: GPTQ rounding of gate/up on the pinned calibration statistics, blockfit for down"

    def supports(self, unit, fmt: str) -> bool:
        return fmt == NVFP4 and unit.kind == "mlp"

    def available(self, ctx: QuantContext, unit, fmt: str) -> str | None:
        return None if ctx.calibration is not None else "needs the calibration statistics (bittrellis calibration fetch)"

    def encode(self, ctx: QuantContext, unit, lin, fmt: str) -> list[Produced]:
        if not self.supports(unit, fmt):
            raise ValueError(f"{self.ref} encodes MLP blocks as NVFP4 only")
        if ctx.calibration is None:
            raise ValueError(f"{self.ref} needs the calibration statistics (bittrellis calibration fetch)")
        h = input_hessian(ctx, lin)
        if h is None:
            if not lin.prefix.endswith(".down_proj"):
                raise ValueError(f"{lin.prefix}: the pinned calibration has no statistics for this Linear")
            return bf.NVFP4BlockFit().encode(ctx, unit, lin, fmt)
        cache = ctx.state.setdefault("nvfp4_gptq", {})
        key = lin.prefix.rpartition(".")[0]
        if cache.get("key") != key:
            cache["key"], cache["u"] = key, inverse_hessian_factor(h)
        w = f32_weight(ctx, lin)
        if not np.isfinite(w).all():
            raise ValueError(f"{lin.prefix}: base weight is not finite")
        packed, scales, ws2 = _quantize_with(w, cache["u"])
        return [(".weight", "U8", packed.shape, packed),
                (".weight_scale", "F8_E4M3", scales.shape, scales),
                (".weight_scale_2", "F32", (), np.asarray(ws2, "<f4").reshape(()))]
