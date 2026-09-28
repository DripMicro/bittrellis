"""Private rotating holdout: validator-side fidelity check on text miners never see.

Layout of a validator's private holdout directory (never committed, never published during its epoch):

    <private>/
      docs/general/*.txt   docs/math/*.txt   docs/code/*.txt   docs/tools/*.txt
      docs/multilingual/*.txt              docs/long/*.txt      (long filler documents)
      epoch.json            {"epoch": "hpc01-h2026w38", "seed": "<secret>"}
      corpus.json           built by `bittrellis holdout build`
      reference/            BF16 reference for corpus.json (`bittrellis reference --corpus ...`)
      results/<name>.json   full per-candidate numbers (private)

Each *.txt file is one document, already in the text form it should be scored in (chat-formatted
where that matters). The stream structure is identical to the public corpus.

Only PASS/FAIL leaves this directory: `holdout.json` in the candidate's artifact contains the epoch
and the verdict, nothing else, so repeated submissions cannot be used to fit the holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .eval import logits
from .eval.candidate import evaluate_quality
from .eval.corpus import SHORT_CATEGORIES, assemble_streams, corpus_hash
from .runtime import SparkInfer
from .track import Track

# Padding shows up as repeated long passages. Short repeats are normal (code idioms, "Answer:",
# log templates), so duplication is measured over 200-word spans, where natural text almost never repeats.
SHINGLE_WORDS = 200
MAX_REPEATED_SPANS = 0.25


class NotEnoughText(ValueError):
    """The holdout directory does not hold enough text yet; the message lists what is missing."""


def build_private_corpus(private_dir: Path, tokenizer_json: Path) -> dict:
    from tokenizers import Tokenizer

    private_dir = Path(private_dir)
    meta = json.loads((private_dir / "epoch.json").read_text())
    tok_bytes = Path(tokenizer_json).read_bytes()
    tok = Tokenizer.from_str(tok_bytes.decode())

    def docs(cat: str) -> list[str]:
        files = sorted((private_dir / "docs" / cat).glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"holdout needs documents in docs/{cat}/")
        ordered = sorted(files, key=lambda f: hashlib.sha256(f"{meta['seed']}:{f.name}".encode()).hexdigest())
        return [f.read_text() for f in ordered]

    short = {cat: docs(cat) for cat in SHORT_CATEGORIES}
    inv = inventory(private_dir, tokenizer_json)
    inv.pop("epoch", None)
    missing = {cat: d["missing"] for cat, d in inv.items() if d["missing"]}
    if missing:
        raise NotEnoughText("not enough text yet: " + ", ".join(
            f"{cat} needs {n:,} more tokens (~{n * 3 // 4:,} words)" for cat, n in sorted(missing.items())))
    repeated = {cat: d["repeated_spans"] for cat, d in inv.items() if d["repeated_spans"] > MAX_REPEATED_SPANS}
    if repeated:
        raise NotEnoughText(f"too much repeated text (a second copy of a passage is not new evidence), "
                            f"measured over {SHINGLE_WORDS}-word spans: " + ", ".join(
                                f"{cat} {share:.0%}" for cat, share in sorted(repeated.items())))
    streams = assemble_streams(tok, short, docs("long"), meta["seed"])
    body = {"version": meta["epoch"], "split": "private-holdout",
            "tokenizer": {"sha256": hashlib.sha256(tok_bytes).hexdigest()}, "streams": streams}
    body["sha256"] = corpus_hash(body)
    (private_dir / "corpus.json").write_text(json.dumps(body, separators=(",", ":")) + "\n")
    return body


def repeated_spans(text: str, n: int = SHINGLE_WORDS) -> float:
    """Share of `n`-word spans that occur more than once: the signature of padded text."""
    words = text.split()
    if len(words) < n:
        return 0.0
    spans = [hashlib.blake2b(" ".join(words[i : i + n]).encode(), digest_size=8).digest() for i in range(len(words) - n + 1)]
    return 1 - len(set(spans)) / len(spans)


def inventory(private_dir: Path, tokenizer_json: Path) -> dict:
    """What the holdout directory has and what it still needs, per category."""
    from tokenizers import Tokenizer

    from .eval.corpus import LONG_LENGTHS, SHORT_TOKENS

    tok = Tokenizer.from_str(Path(tokenizer_json).read_text())
    need = {cat: SHORT_TOKENS for cat in SHORT_CATEGORIES}
    need["long"] = max(LONG_LENGTHS)  # every needle stream is filled from these documents
    out = {}
    for cat, want in need.items():
        files = sorted((Path(private_dir) / "docs" / cat).glob("*.txt"))
        have = sum(len(tok.encode(f.read_text()).ids) for f in files)
        # Repeated passages make a holdout look easy: predictions on a second copy are not independent.
        repeated = repeated_spans("\n".join(f.read_text() for f in files))
        # Two or more long documents keep the 8K/16K/32K streams from being prefixes of each other.
        want_more = max(LONG_LENGTHS) * 2 if cat == "long" else want
        out[cat] = {"files": len(files), "tokens": have, "needs": want, "missing": max(0, want - have),
                    "recommended": want_more, "short_of_recommended": max(0, want_more - have),
                    "repeated_spans": round(repeated, 3)}
    out["epoch"] = json.loads((Path(private_dir) / "epoch.json").read_text()) if (Path(private_dir) / "epoch.json").exists() else None
    return out


TRANSFER_FAIL = "public gain does not carry over to the holdout"


def transfer(track: Track, pub_inc: dict, pub_cand: dict, inc_pos: dict, cand_pos: dict) -> dict:
    """Whether the candidate's public drift gain over the incumbent carries over to the holdout."""
    ratio = track["evaluation"]["holdout"]["min_gain_ratio"]
    if track["frontier"].get("drift") == "section-balanced":
        # hpc01-e4: gains are relative and section-balanced on each side, so an easier holdout, or one public
        # section where the incumbent drifts most, cannot decide the verdict. Every drift gain that FG-2 would
        # credit (above the RP-KL floor) must carry over, whether or not it is statistically significant.
        ps, hs = logits.balance_scale(pub_inc), logits.balance_scale(inc_pos)
        pub = logits.paired_delta(logits.balanced(pub_inc, ps), logits.balanced(pub_cand, ps))
        hold = logits.paired_delta(logits.balanced(inc_pos, hs), logits.balanced(cand_pos, hs))
        base_pub, base_hold = logits.mean_drift(pub_inc), logits.mean_drift(inc_pos)
        public_gain, holdout_gain = -pub["delta"] / base_pub, -hold["delta"] / base_hold
        claims_gain = public_gain > track["frontier"]["epsilon_floor"]["rp_kl"] / base_pub
    else:
        pub = logits.paired_delta(pub_inc, pub_cand)          # candidate - incumbent, public
        hold = logits.paired_delta(inc_pos, cand_pos)         # candidate - incumbent, holdout
        public_gain, holdout_gain = -pub["delta"], -hold["delta"]
        claims_gain = pub["significant"] and public_gain > 0
    return {"public_gain": public_gain, "holdout_gain": holdout_gain, "public_ci95": pub["ci95"], "holdout_ci95": hold["ci95"],
            "drift": track["frontier"].get("drift", "mean"), "fails": bool(claims_gain and holdout_gain < ratio * public_gain)}


def recheck(track: Track, artifact: Path, private_dir: Path, incumbent_artifact: Path) -> str:
    """Re-judge a candidate already scored on this holdout under the current rule, from its cached positions.

    The other holdout checks (audit, RP-KL ceiling, needles) do not depend on the rule and keep their recorded
    outcome; only the transfer test is recomputed. No GPU and no checkpoint are needed.
    """
    private_dir = Path(private_dir)
    meta = json.loads((private_dir / "epoch.json").read_text())
    name = json.loads((Path(artifact) / "candidate.json").read_text())["name"]
    inc = json.loads((Path(incumbent_artifact) / "candidate.json").read_text())["name"]
    path = private_dir / "results" / f"{name}.json"
    old = json.loads(path.read_text())
    t = transfer(track, dict(np.load(Path(incumbent_artifact) / "kl_positions.npz")),
                 dict(np.load(Path(artifact) / "kl_positions.npz")),
                 dict(np.load(private_dir / "work" / "incumbent" / inc / "kl_positions.npz")),
                 dict(np.load(private_dir / "work" / "candidate" / name / "kl_positions.npz")))
    reasons = [r for r in old["reasons"] if r != TRANSFER_FAIL] + ([TRANSFER_FAIL] if t["fails"] else [])
    verdict = "FAIL" if reasons else "PASS"
    path.write_text(json.dumps({**old, "verdict": verdict, "reasons": reasons, **{k: v for k, v in t.items() if k != "fails"},
                                "rechecked_from": old["verdict"]}, indent=2) + "\n")
    (Path(artifact) / "holdout.json").write_text(json.dumps({"epoch": meta["epoch"], "result": verdict}) + "\n")
    return verdict


def check(si: SparkInfer, track: Track, checkpoint: Path, artifact: Path, private_dir: Path,
          incumbent_checkpoint: Path, incumbent_artifact: Path, log=print) -> str:
    """Score a candidate (and, once per epoch, the incumbent) on the private holdout; write PASS/FAIL."""
    private_dir = Path(private_dir)
    corpus = json.loads((private_dir / "corpus.json").read_text())
    meta = json.loads((private_dir / "epoch.json").read_text())
    results = private_dir / "results"
    results.mkdir(exist_ok=True)
    cand_name = json.loads((Path(artifact) / "candidate.json").read_text())["name"]

    def score(ckpt: Path, name: str) -> tuple[dict, dict]:
        work = private_dir / "work" / name  # "incumbent/<name>" or "candidate/<name>": a candidate never overwrites the cache
        work.mkdir(parents=True, exist_ok=True)
        q, corr = evaluate_quality(si, track, ckpt, corpus, private_dir / "reference", work, log=log)
        return q, dict(np.load(work / "kl_positions.npz"))

    inc_ident = json.loads((Path(incumbent_artifact) / "candidate.json").read_text())
    inc_name = inc_ident["name"]
    # The incumbent is scored once per epoch and corpus, then reused by every later check.
    inc_work = private_dir / "work" / "incumbent" / inc_name
    stamp = {"epoch": meta["epoch"], "corpus_sha256": corpus.get("sha256"), "candidate_id": inc_ident.get("id")}
    cached = inc_work / "incumbent.json"
    if cached.exists() and (inc_work / "kl_positions.npz").exists() and json.loads(cached.read_text()).get("stamp") == stamp:
        inc_q, inc_pos = json.loads(cached.read_text())["quality"], dict(np.load(inc_work / "kl_positions.npz"))
        log(f"[holdout] incumbent {inc_name}: cached for epoch {meta['epoch']}")
    else:
        inc_q, inc_pos = score(Path(incumbent_checkpoint), f"incumbent/{inc_name}")
        cached.write_text(json.dumps({"stamp": stamp, "quality": inc_q}) + "\n")
    cand_q, cand_pos = score(Path(checkpoint), f"candidate/{cand_name}")

    gates = track["gates"]
    reasons = []
    audit_ok = json.loads((Path(artifact) / "candidate.json").read_text()).get("audit_ok")
    if audit_ok is not True:
        reasons.append("audit")
    if cand_q["rp_kl"] > gates["rp_kl_max"]:
        reasons.append("rp_kl_max")
    for stream, got in cand_q["needles_by_length"].items():
        if got["required"] and got["retrieved"] < got["required"]:
            reasons.append(f"needles:{stream}")
    pub_inc = dict(np.load(Path(incumbent_artifact) / "kl_positions.npz"))
    pub_cand = dict(np.load(Path(artifact) / "kl_positions.npz"))
    t = transfer(track, pub_inc, pub_cand, inc_pos, cand_pos)
    if t["fails"]:
        reasons.append(TRANSFER_FAIL)
    verdict = "FAIL" if reasons else "PASS"

    (results / f"{cand_name}.json").write_text(json.dumps({
        "epoch": meta["epoch"], "verdict": verdict, "reasons": reasons,
        "candidate_rp_kl": cand_q["rp_kl"], "incumbent_rp_kl": inc_q["rp_kl"],
        **{k: v for k, v in t.items() if k != "fails"},
    }, indent=2) + "\n")
    (Path(artifact) / "holdout.json").write_text(json.dumps({"epoch": meta["epoch"], "result": verdict}) + "\n")
    return verdict
