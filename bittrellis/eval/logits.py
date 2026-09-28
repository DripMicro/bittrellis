"""Reference-Partition KL (RP-KL): how far a checkpoint's next-token distribution is from BF16.

For every scored position the BF16 reference fixes a partition of the vocabulary *before any
candidate exists*: its top-K token ids (K pinned by the track, 256) plus one bucket holding every
other token. The reference stores its log-probabilities on those ids; the candidate is scored on
exactly the same ids (tools/sparkinfer_refscore.cpp). Both distributions are projected onto that
partition and

    RP-KL = sum_i p_i (log p_i - log q_i) + p_tail (log p_tail - log q_tail)

Because both projections use exact probabilities over the same fixed partition, RP-KL is the KL
divergence between the projected distributions, which is never larger than the full-vocabulary KL
(data-processing inequality) and is computed on the same footing for every candidate. It is not
the full-vocabulary KL and is not called that.

Float32 storage and the reference's own rounding introduce a small numerical error; tail masses are
clamped to [1e-12, 1], which can move a position's value in either direction by a negligible amount.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..runtime import RefScore, ScoreDump

EPS = 1e-12


@dataclass
class StreamQuality:
    stream: str
    category: str
    n: int
    kl: np.ndarray            # RP-KL per position
    top1: np.ndarray          # bool per position: candidate argmax == BF16 argmax
    nll_delta: np.ndarray     # candidate NLL - BF16 NLL on the true next token
    outside_mass: np.ndarray  # candidate probability outside the BF16 top-K ids
    nonfinite: int            # candidate log-probs that were NaN/Inf
    needles: dict[str, dict]


def compare(ref: ScoreDump, cand: RefScore, stream: dict) -> StreamQuality:
    if len(ref.pos) != len(cand.pos) or not np.array_equal(ref.pos, cand.pos):
        raise ValueError(f"{stream['id']}: candidate positions do not match the reference partition")
    if cand.lp_ref.shape != ref.top_lp.shape:
        raise ValueError(f"{stream['id']}: candidate K={cand.lp_ref.shape[1]} != reference K={ref.top_lp.shape[1]}")
    lp = ref.top_lp.astype(np.float64)
    lq = cand.lp_ref.astype(np.float64)
    nonfinite = int((~np.isfinite(lq)).sum())
    lq = np.where(np.isfinite(lq), lq, np.log(EPS))
    p, q = np.exp(lp), np.exp(lq)
    p_tail = np.clip(1.0 - p.sum(axis=1), EPS, 1.0)
    q_tail = np.clip(1.0 - q.sum(axis=1), EPS, 1.0)
    kl = (p * (lp - lq)).sum(axis=1) + p_tail * (np.log(p_tail) - np.log(q_tail))
    kl = np.maximum(kl, 0.0)
    top1 = ref.top_ids[:, 0] == cand.argmax
    nll_delta = (-cand.lp_target.astype(np.float64)) - (-ref.lp_target.astype(np.float64))

    needles = {}
    ids = stream["ids"]
    at = {int(pp): k for k, pp in enumerate(cand.pos)}
    for nd in stream.get("needles", []):
        preds = [p_ - 1 for p_ in nd["target_positions"]]  # position i predicts token i+1
        want = [ids[p_] for p_ in nd["target_positions"]]
        cand_ok = all(int(cand.argmax[at[i]]) == w for i, w in zip(preds, want, strict=True))
        ref_ok = all(int(ref.top_ids[at[i], 0]) == w for i, w in zip(preds, want, strict=True))
        needles[nd["name"]] = {"depth": nd["depth"], "candidate": bool(cand_ok), "reference": bool(ref_ok)}
    return StreamQuality(stream["id"], stream["category"], len(kl), kl, top1, nll_delta, q_tail, nonfinite, needles)


def block_bootstrap_ci(values: list[np.ndarray], block: int = 128, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the pooled mean, resampling contiguous blocks (positions are autocorrelated)."""
    blocks = [v[i : i + block] for v in values for i in range(0, len(v), block)]
    sums = np.array([b.sum() for b in blocks])
    counts = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def save_positions(results: list[StreamQuality], path) -> None:
    """Per-position RP-KL/top-1 per stream, so any two artifacts can be compared *paired*."""
    arrays = {}
    for r in results:
        arrays[f"{r.stream}.kl"] = r.kl.astype(np.float32)
        arrays[f"{r.stream}.top1"] = r.top1
    np.savez_compressed(path, **arrays)


def section_of(stream: str) -> str:
    """Balancing sections: each short category on its own, the long-context streams together."""
    return "long" if stream.startswith("long-") else stream


def balance_scale(ref: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-stream factors that turn mean drift into section-balanced drift against `ref` (epoch hpc01-e4).

    Balanced drift = RP-KL(ref) × mean over sections of (drift in that section / ref's drift there), so ref
    keeps its own RP-KL and every section counts equally by its *relative* change. Without it, the section
    where the reference drifts most decides most of every score (the public maths stream did, in hpc01-e3).
    """
    streams = sorted(k[:-3] for k in ref if k.endswith(".kl"))
    total = np.concatenate([ref[f"{s}.kl"] for s in streams]).astype(np.float64)
    sections: dict[str, list[str]] = {}
    for s in streams:
        sections.setdefault(section_of(s), []).append(s)
    out = {}
    for members in sections.values():
        sec = np.concatenate([ref[f"{s}.kl"] for s in members]).astype(np.float64)
        for s in members:
            out[s] = float(total.mean() * len(total) / (len(sections) * sec.mean() * len(sec)))
    return out


def balanced(positions: dict[str, np.ndarray], scale: dict[str, float] | None) -> dict[str, np.ndarray]:
    """`positions` with each stream's per-position drift weighted so that its mean is section-balanced drift."""
    if not scale:
        return positions
    return {k: (v.astype(np.float64) * scale[k[:-3]] if k.endswith(".kl") and k[:-3] in scale else v)
            for k, v in positions.items()}


def mean_drift(positions: dict[str, np.ndarray]) -> float:
    return float(np.concatenate([v for k, v in sorted(positions.items()) if k.endswith(".kl")]).astype(np.float64).mean())


def paired_delta(a: dict[str, np.ndarray], b: dict[str, np.ndarray], block: int = 128, n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """Mean RP-KL(b) - RP-KL(a) over identical positions with a paired block-bootstrap 95% interval.

    Both artifacts score the same tokens against the same partition, so position-level variation
    common to every checkpoint (chaotic or high-entropy positions) cancels in the difference.
    """
    streams = sorted(k[:-3] for k in a if k.endswith(".kl") and k in b)
    if not streams:
        raise ValueError("no common streams to compare")
    diffs = [b[f"{s}.kl"].astype(np.float64) - a[f"{s}.kl"].astype(np.float64) for s in streams]
    lo, hi = block_bootstrap_ci(diffs, block=block, n_boot=n_boot, seed=seed)
    d = np.concatenate(diffs)
    return {"delta": float(d.mean()), "ci95": [lo, hi], "positions": int(len(d)), "significant": bool(lo > 0 or hi < 0)}


def summarize(results: list[StreamQuality], k: int) -> dict:
    kl_all = np.concatenate([r.kl for r in results])
    lo, hi = block_bootstrap_ci([r.kl for r in results])
    by_cat: dict[str, dict] = {}
    for cat in sorted({r.category for r in results}):
        rs = [r for r in results if r.category == cat]
        by_cat[cat] = {
            "positions": int(sum(r.n for r in rs)),
            "rp_kl": float(np.concatenate([r.kl for r in rs]).mean()),
            "top1": float(np.concatenate([r.top1 for r in rs]).mean()),
            "nll_delta": float(np.concatenate([r.nll_delta for r in rs]).mean()),
        }
    needles = {r.stream: r.needles for r in results if r.needles}
    counted = [n for s in needles.values() for n in s.values() if n["reference"]]
    by_length: dict[str, dict] = {}
    for stream, nd in needles.items():
        ok = [n["candidate"] for n in nd.values() if n["reference"]]
        by_length[stream] = {"required": len(ok), "retrieved": int(sum(ok))}
    return {
        "metric": "reference-partition-kl",
        "partition_k": k,
        "positions": int(len(kl_all)),
        "rp_kl": float(kl_all.mean()),
        "rp_kl_ci95": [lo, hi],
        "rp_kl_p99": float(np.percentile(kl_all, 99)),
        "top1": float(np.concatenate([r.top1 for r in results]).mean()),
        "nll_delta": float(np.concatenate([r.nll_delta for r in results]).mean()),
        "outside_mass_mean": float(np.concatenate([r.outside_mass for r in results]).mean()),
        "nonfinite_logprobs": int(sum(r.nonfinite for r in results)),
        "by_category": by_cat,
        "by_stream": {r.stream: {"rp_kl": float(r.kl.mean()), "top1": float(r.top1.mean())} for r in results},
        "needles": needles,
        "needles_by_length": by_length,
        "needle_recall": (sum(n["candidate"] for n in counted) / len(counted)) if counted else None,
    }
