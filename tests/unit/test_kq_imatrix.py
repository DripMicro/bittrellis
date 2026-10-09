import numpy as np
import pytest

pytest.importorskip("gguf")

from bittrellis import kquant  # noqa: E402
from bittrellis.hpc02 import ENCODERS, EncodeContext, Unit  # noqa: E402
from bittrellis.hpc02_encoders import kq_imatrix as K  # noqa: E402

FORMATS = ("Q4_K", "Q5_K", "Q6_K")


class FakeCalibration:
    """The two methods encoders use on the pinned statistics (SafeTensorsDir.get / .array)."""

    def __init__(self, tensors):
        self.t = {k: np.asarray(v, "<f4") for k, v in tensors.items()}

    def get(self, name):
        return self.t.get(name)

    def array(self, name, dtype):
        return self.t[name].view(dtype)


def _rows(n, cols, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, cols)) * rng.uniform(0.005, 0.05, (n, 1)) * rng.uniform(0.3, 3, cols)
    w[rng.integers(0, n, 8), rng.integers(0, cols, 8)] *= 20.0
    return w.astype(np.float32)


def _expert_case(kind="exps.down", per=2, cols=512, seed=1):
    rng = np.random.default_rng(seed)
    rows = _rows(K.N_EXPERTS * per, cols, seed)
    stat = "exps_down" if kind == "exps.down" else "exps_input"
    count = rng.integers(0, 50, K.N_EXPERTS).astype(np.float32)
    count[:3] = 0                                                       # experts no token reached
    sumsq = (rng.gamma(0.5, 1.0, (K.N_EXPERTS, cols)) * count[:, None]).astype(np.float32)
    cal = FakeCalibration({f"blk.3.{stat}.sumsq": sumsq, "blk.3.exps.count": count})
    unit = Unit("L3." + kind, "t", 3, kind, cols, len(rows), rows.size)
    return EncodeContext(rows, unit, {}, cal), K.expert_importance(sumsq, count)


def _werr(fmt, data, rows, imp_rows):
    deq = kquant.dequantize(fmt, data, rows.shape[1]).astype(np.float64)
    return float((imp_rows * (deq - rows) ** 2).sum() / (imp_rows * rows.astype(np.float64) ** 2).sum())


def test_registered_and_deterministic():
    enc = ENCODERS["kq_imat"]
    assert enc.lineage == "regenerable" and enc.ref == "kq_imat@v1"
    ctx, _ = _expert_case()
    unit = Unit("probe", "probe.weight", 0, "exps.gate", 1024, 64, 64 * 1024)
    for fmt in FORMATS + ("Q8_0",):
        a = enc.fn(EncodeContext(_rows(64, 1024), unit, {}, None), fmt)
        b = enc.fn(EncodeContext(_rows(64, 1024), unit, {}, None), fmt)
        assert a == b
        assert enc.fn(ctx, fmt) == enc.fn(ctx, fmt)
        assert len(a) == 64 * kquant.row_bytes(fmt, 1024)


@pytest.mark.parametrize("fmt", FORMATS)
def test_bytes_do_not_depend_on_chunking(fmt, monkeypatch):
    ctx, _ = _expert_case(per=1, cols=512)
    ref = K._encode(ctx, fmt)
    monkeypatch.setattr(K, "CHUNK_SB", 7)
    assert K._encode(ctx, fmt) == ref


@pytest.mark.parametrize("fmt", FORMATS)
def test_layout_round_trips_through_the_reference_decoder(fmt):
    x = np.asarray(_rows(32, 512, 3), np.float64).reshape(-1, 256)
    w = K._weights(x, None)
    if fmt == "Q6_K":
        d, sc, lv = K.fit_q6k(x, w)
        mine = (d.astype(np.float64)[:, None] * sc)[..., None] * lv.reshape(-1, 16, 16)
        data = K.pack_q6k(d, sc, lv)
    else:
        d, dmin, sc, m, lv = K.fit_asym(x, w, K.NMAX[fmt])
        mine = ((d.astype(np.float64)[:, None] * sc)[..., None] * lv.reshape(-1, 8, 32)
                - (dmin.astype(np.float64)[:, None] * m)[..., None])
        data = (K.pack_q4k if fmt == "Q4_K" else K.pack_q5k)(d, dmin, sc, m, lv)
    ref = kquant.dequantize(fmt, data, 256).astype(np.float64)
    assert len(data) == len(x) * kquant.BLOCK[fmt][0]
    np.testing.assert_allclose(ref.reshape(-1), mine.reshape(-1), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("kind", ["exps.down", "exps.gate"])
def test_lower_weighted_error_than_rtn(fmt, kind):
    ctx, imp = _expert_case(kind)
    imp_rows = np.repeat(imp, len(ctx.rows) // K.N_EXPERTS, axis=0)
    ours = _werr(fmt, K._encode(ctx, fmt), ctx.rows, imp_rows)
    rtn = _werr(fmt, kquant.RTN[fmt](ctx.rows), ctx.rows, imp_rows)
    assert ours < 0.9 * rtn


def test_dense_units_read_the_xtx_diagonal_and_fall_back_without_it():
    rows = _rows(16, 256, 5)
    rng = np.random.default_rng(5)
    h = np.diag(rng.gamma(0.5, 1.0, 256)).astype(np.float32)
    cal = FakeCalibration({"blk.2.attn_input.xtx": h})
    q = Unit("L2.attn.q", "t", 2, "attn.q", 256, 16, 16 * 256)
    o = Unit("L2.attn.o", "t", 2, "attn.o", 256, 16, 16 * 256)
    imp = K.importance(EncodeContext(rows, q, {}, cal))
    np.testing.assert_allclose(imp[0], np.diag(h))
    assert K.importance(EncodeContext(rows, o, {}, cal)) is None
    assert K._encode(EncodeContext(rows, q, {}, cal), "Q4_K") != K._encode(EncodeContext(rows, q, {}, None), "Q4_K")
    # rows that are not 256 experts' worth fall back to the uncalibrated weights
    g = Unit("L2.exps.gate", "t", 2, "exps.gate", 256, 16, 16 * 256)
    assert K.importance(EncodeContext(rows, g, {}, cal)) is None


def test_differs_from_kq_rtn_on_the_probe():
    rows = _rows(64, 1024)
    unit = Unit("probe", "probe.weight", 0, "exps.gate", 1024, 64, 64 * 1024)
    for fmt in FORMATS:
        a = np.frombuffer(K._encode(EncodeContext(rows.copy(), unit, {}, None), fmt), np.uint8)
        b = np.frombuffer(kquant.RTN[fmt](rows), np.uint8)
        assert (a == b).mean() < 0.9
