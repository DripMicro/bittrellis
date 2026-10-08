# Sourced by the other scripts: exports pins from configs/hpc01.yaml and configs/sources.lock.json.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
eval "$("$PY" - "$REPO_ROOT" <<'PYEOF'
import json, shlex, sys, yaml
root = sys.argv[1]
t = yaml.safe_load(open(f"{root}/configs/hpc01.yaml"))
lock = json.load(open(f"{root}/configs/sources.lock.json"))["sources"]
r2 = t["external_references"]["R2"]
pins = {"SPARKINFER_REPO": t["runtime"]["repo"], "SPARKINFER_COMMIT": t["runtime"]["commit"],
        "SPARKINFER_CMAKE_ARGS": " ".join(t["runtime"]["cmake_args"]), "SPARKINFER_TARGETS": " ".join(t["runtime"]["targets"]),
        "LLAMACPP_COMMIT": r2["llama_cpp_commit"], "R2_FILE": r2["file"],
        "DSPARK_REPO": t["runtime"]["dspark"]["repo"], "DSPARK_REVISION": t["runtime"]["dspark"]["revision"]}
t2 = yaml.safe_load(open(f"{root}/configs/hpc02.yaml"))   # HPC-02: its own SparkInfer pin and CUDA
pins.update({"SPARKINFER02_COMMIT": t2["runtime"]["commit"], "SPARKINFER02_CMAKE_ARGS": " ".join(t2["runtime"]["cmake_args"]),
             "SPARKINFER02_TARGETS": " ".join(t2["runtime"]["targets"]), "CUDA02": t2["runtime"]["cuda"]})
for sid, s in lock.items():
    pins[f"{sid.upper()}_REPO"] = s["repo"]
    pins[f"{sid.upper()}_REVISION"] = s["revision"]
for k, v in pins.items():
    print(f"export PIN_{k}={shlex.quote(str(v))}")
PYEOF
)"
export MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
export SPARKINFER_DIR="${BITTRELLIS_SPARKINFER:-$REPO_ROOT/third_party/sparkinfer}"
export SPARKINFER02_DIR="${BITTRELLIS_SPARKINFER02:-$REPO_ROOT/third_party/sparkinfer-hpc02}"
export LLAMACPP_DIR="${BITTRELLIS_LLAMACPP:-$REPO_ROOT/third_party/llama.cpp}"
