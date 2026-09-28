"""Load artifact directories into rows, rank the internal frontier, and write reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..eval.logits import balance_scale, balanced, mean_drift, paired_delta
from ..track import REPO_ROOT, Track
from .pareto import FG_VERSION, OBJECTIVES, Row, apply_gates, rank

GIB = 1024**3


def _read(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def load_row(art: Path, speeds: Path | None = None) -> Row | None:
    """One artifact as a row. With `speeds` (hpc01-e5), a ranked result's speed and memory come from
    `<speeds>/<candidate id>/performance.json` -- re-measured on the machine that measures the PR -- when present."""
    cand, q, perf = _read(art / "candidate.json"), _read(art / "quality.json"), _read(art / "performance.json")
    if not (cand and q and perf) or "rp_kl" not in q:
        return None
    corr = _read(art / "correctness.json")
    hold = _read(art / "holdout.json")
    kind = "internal" if cand.get("kind") in ("candidate", "internal") else "external"
    perf = (_read(Path(speeds) / cand["id"] / "performance.json") if speeds and kind == "internal" else None) or perf
    return Row(
        id=cand["id"], name=cand["name"], kind=kind,
        rp_kl=q["rp_kl"], decode_tps=perf["decode_tps"], prefill_tps=perf["prefill_tps"], peak_gpu_gib=perf["peak_gpu_gib"],
        decode_spread=perf.get("decode_spread", 0.0), prefill_spread=perf.get("prefill_spread", 0.0),
        top1=q.get("top1"), needles_by_length=q.get("needles_by_length"),
        correctness_ok=(corr or {}).get("ok"), audit_ok=cand.get("audit_ok"), tasks=_read(art / "tasks.json"),
        holdout=(hold or {}).get("result"),
        extra={"path": str(art), "checkpoint_gib": (cand.get("checkpoint_bytes") or 0) / GIB,
               "resident_after_load_gib": perf.get("resident_after_load_gib"), "peak_host_gib": perf.get("peak_host_gib")},
    )


def drift_scale(track: Track, incumbent: Path | None) -> dict[str, float] | None:
    """Section weights for balanced drift (`frontier.drift: section-balanced`), taken from the incumbent's positions."""
    f = Path(incumbent) / "kl_positions.npz" if incumbent else None
    if track["frontier"].get("drift") != "section-balanced" or f is None or not f.exists():
        return None
    return balance_scale(dict(np.load(f)))


def apply_balance(rows: list[Row], scale: dict[str, float] | None) -> None:
    """Replace each row's RP-KL with section-balanced drift; the plain mean stays in `extra['rp_kl_raw']`."""
    for r in rows:
        f = Path(r.extra["path"]) / "kl_positions.npz"
        if scale and f.exists() and "rp_kl_raw" not in r.extra:
            r.extra["rp_kl_raw"] = r.rp_kl
            r.rp_kl = mean_drift(balanced(dict(np.load(f)), scale))


class PairedQuality:
    """RP-KL comparator: materially different only beyond the floor AND with a significant paired delta."""

    def __init__(self, floor: float, scale: dict[str, float] | None = None):
        self.floor, self.scale = floor, scale
        self._arrays: dict[str, dict] = {}
        self._cache: dict[tuple[str, str], dict] = {}

    def _positions(self, row: Row) -> dict:
        path = row.extra["path"]
        if path not in self._arrays:
            self._arrays[path] = balanced(dict(np.load(Path(path) / "kl_positions.npz")), self.scale)
        return self._arrays[path]

    def delta(self, a: Row, b: Row) -> dict:
        """RP-KL(b) - RP-KL(a), paired."""
        key = (a.extra["path"], b.extra["path"])
        if key not in self._cache:
            self._cache[key] = paired_delta(self._positions(a), self._positions(b), n_boot=1000)
        return self._cache[key]

    def __call__(self, a: Row, b: Row) -> int:
        if abs(a.rp_kl - b.rp_kl) <= self.floor:
            return 0
        d = self.delta(a, b)
        if not d["significant"]:
            return 0
        return -1 if d["delta"] > 0 else 1


def collect_artifacts(paths: list[Path]) -> list[Path]:
    arts: list[Path] = []
    for p in paths:
        if (p / "candidate.json").exists():
            arts.append(p)
        elif p.is_dir():
            arts.extend(sorted(d for d in p.iterdir() if (d / "candidate.json").exists()))
    return arts


def load_rows(paths: list[Path], track: Track, speeds: Path | None = None) -> tuple[list[Row], PairedQuality]:
    rows = [r for r in (load_row(a, speeds) for a in collect_artifacts(paths)) if r is not None]
    incumbent_name = track["frontier"]["incumbent"]
    incumbent = next((r for r in rows if r.name == incumbent_name), None)
    scale = drift_scale(track, Path(incumbent.extra["path"]) if incumbent else None)
    apply_balance(rows, scale)
    for r in rows:
        apply_gates(r, track["gates"], incumbent.tasks if incumbent and r is not incumbent else None)
    cmp = PairedQuality(track["frontier"]["epsilon_floor"]["rp_kl"], scale)
    rank(rows, track["frontier"]["box"], track["frontier"]["epsilon_floor"], cmp)
    return rows, cmp


def render_table(rows: list[Row]) -> str:
    head = f"  {'id':18s} {'name':32s} {'RP-KL':>7s} {'top1':>6s} {'decode':>7s} {'prefill':>8s} {'peakGPU':>7s} {'FG-2':>6s}  gates"
    out = []
    for kind, title in (("internal", "INTERNAL FRONTIER (legal BitTrellis artifacts on pinned SparkInfer)"),
                        ("external", "EXTERNAL REFERENCES (context only; never ranked)")):
        sel = sorted((r for r in rows if r.kind == kind), key=lambda r: r.rp_kl)
        if not sel:
            continue
        out += [title, head, "-" * len(head)]
        for r in sel:
            mark = "★" if r.frontier else " "
            gates = "PASS" if r.valid else "; ".join(r.gate_failures)
            fg = f"{100 * r.gain:5.2f}%" if kind == "internal" else "    —"
            out.append(f"{mark} {r.id:18s} {r.name[:32]:32s} {r.rp_kl:7.4f} {r.top1 or 0:6.3f} {r.decode_tps:7.1f} "
                       f"{r.prefill_tps:8.0f} {r.peak_gpu_gib:7.2f} {fg}  {gates}")
        out.append("")
    return "\n".join(out)


def row_dict(r: Row) -> dict:
    return {
        "id": r.id, "name": r.name, "kind": r.kind, "rp_kl": r.rp_kl, "rp_kl_raw": r.extra.get("rp_kl_raw", r.rp_kl), "top1": r.top1,
        "decode_tps": r.decode_tps, "decode_spread": r.decode_spread, "prefill_tps": r.prefill_tps,
        "prefill_spread": r.prefill_spread, "peak_gpu_gib": r.peak_gpu_gib,
        "resident_after_load_gib": r.extra.get("resident_after_load_gib"), "peak_host_gib": r.extra.get("peak_host_gib"),
        "checkpoint_gib": r.extra.get("checkpoint_gib"), "tasks_passed": (r.tasks or {}).get("passed"), "tasks_n": (r.tasks or {}).get("n"),
        "holdout": r.holdout, "valid": r.valid, "gate_failures": r.gate_failures,
        "frontier": r.frontier, "frontier_gain": r.gain if r.kind == "internal" else None, "dominated_by": r.dominated_by,
    }


def write_frontier(rows: list[Row], track: Track, out: Path | None) -> dict:
    doc = {
        "track": track.id, "track_version": track["version"], "evaluator_epoch": track["evaluation"]["epoch"],
        "frontier_gain_version": FG_VERSION, "objectives": dict(OBJECTIVES),
        "epsilon_floor": track["frontier"]["epsilon_floor"], "box": track["frontier"]["box"],
        "incumbent": track["frontier"]["incumbent"],
        "internal": [row_dict(r) for r in rows if r.kind == "internal"],
        "external": [row_dict(r) for r in rows if r.kind == "external"],
        "frontier": [r.name for r in rows if r.frontier],
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def compare(a: Path, b: Path, track: Track, speeds: Path | None = None) -> dict:
    """Paired comparison of two artifacts: RP-KL delta with CI, and every performance objective."""
    ra, rb = load_row(a, speeds), load_row(b, speeds)
    if ra is None or rb is None:
        raise ValueError("both artifacts need quality.json and performance.json")
    scale = drift_scale(track, REPO_ROOT / track["frontier"]["seeds"] / track["frontier"]["incumbent"])
    apply_balance([ra, rb], scale)
    cmp = PairedQuality(track["frontier"]["epsilon_floor"]["rp_kl"], scale)
    d = cmp.delta(ra, rb)
    return {
        "a": ra.name, "b": rb.name,
        "rp_kl": {"a": ra.rp_kl, "b": rb.rp_kl, "delta": d["delta"], "ci95": d["ci95"], "significant": d["significant"]},
        "decode_tps": {"a": ra.decode_tps, "b": rb.decode_tps, "rel": (rb.decode_tps - ra.decode_tps) / ra.decode_tps},
        "prefill_tps": {"a": ra.prefill_tps, "b": rb.prefill_tps, "rel": (rb.prefill_tps - ra.prefill_tps) / ra.prefill_tps},
        "peak_gpu_gib": {"a": ra.peak_gpu_gib, "b": rb.peak_gpu_gib, "delta": rb.peak_gpu_gib - ra.peak_gpu_gib},
    }


def _plots(rows: list[Row], out: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for key, label, fname in (("decode_tps", "decode tok/s @ 4K, batch 1  →  better", "quality_vs_decode.png"),
                              ("prefill_tps", "prefill tok/s @ 4K  →  better", "quality_vs_prefill.png"),
                              ("peak_gpu_gib", "peak GPU memory GiB  ←  better", "quality_vs_memory.png")):
        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
        for r in rows:
            tag = r.id if r.kind == "external" else r.name.split("-")[0]
            if r.kind == "external":
                style = {"marker": "s", "s": 60, "color": "none", "edgecolor": "#6b7280", "linewidths": 1.5}
            elif not r.valid:
                style = {"marker": "x", "s": 50, "color": "#dc2626"}
            else:
                style = {"marker": "o", "s": 70 if r.frontier else 38, "color": "#2563eb" if r.frontier else "#93c5fd",
                         "edgecolor": "#111827" if r.frontier else "none"}
            ax.scatter(getattr(r, key), r.rp_kl, zorder=3, **style)
            ax.annotate(tag, (getattr(r, key), r.rp_kl), textcoords="offset points", xytext=(6, -3), fontsize=8,
                        fontweight="bold" if r.frontier else "normal", color="#6b7280" if r.kind == "external" else "#111827")
        ax.set_xlabel(label)
        ax.set_ylabel("RP-KL vs BF16, nats/token  ↓ better")
        ax.set_title("HPC-01 · ● internal (bold = frontier) · □ external reference, not ranked", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname)
        plt.close(fig)
        made.append(fname)
    return made


def write_report(paths: list[Path], track: Track, out: Path) -> Path:
    rows, _ = load_rows(paths, track)
    out.mkdir(parents=True, exist_ok=True)
    write_frontier(rows, track, out / "frontier.json")
    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row_dict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            d = row_dict(r)
            d["gate_failures"] = "; ".join(d["gate_failures"])
            w.writerow(d)
    _plots(rows, out / "plots")
    (out / "frontier_table.txt").write_text(render_table(rows) + "\n")
    return out / "frontier.json"


def seed_rows(track: Track) -> list[Path]:
    return collect_artifacts([REPO_ROOT / track["frontier"]["seeds"]])
