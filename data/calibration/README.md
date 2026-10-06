# Calibration text

`hpc01-calib-v1.json` holds the token ids the pinned calibration statistics were computed from
(`bittrellis/calibration.py`; encoders read the statistics through `input_hessian`, see
[quantizer_contract.md](../../docs/quantizer_contract.md#calibration-statistics-epoch-hpc01-e6)).
It is built deterministically by `bittrellis calibration text` from pinned sources and checked by hash
(`bittrellis calibration verify`).

| category | sequences × tokens | source (pinned revision in the file) |
|---|---:|---|
| general | 32 × 2,048 | WikiText-103 **train** (CC BY-SA 3.0); the drift corpus uses the test split |
| math | 32 × 2,048 | GSM8K **train**, chat-formatted (MIT); the drift corpus and the tasks use the test split |
| code | 32 × 2,048 | MBPP, all 974 problems, chat-formatted (CC BY 4.0) |
| tools | 32 × 2,048 | Glaive function-calling v2, as Qwen chat with tool turns (Apache-2.0) |
| multilingual | 32 × 2,048 | FineWeb-2 test files of the drift corpus's ten languages, alternating (ODC-By) |

327,680 tokens in all. Every sequence starts a fresh document with `<|endoftext|>`, as in the drift
corpus; documents are cut at 1,024 tokens so none dominates its category.

**Disjoint from what is scored:** a document is dropped when it shares a run of 50 normalized characters
(NFKC, lowercase, letters and digits only) with the public drift corpus or with any string of the 784
task questions (SparkInfer `bench/quality/data` at the pinned commit). One WikiText article was dropped.
The maintainers also checked the final text against the private holdout: no overlap.

The statistics (mean x xᵀ of each layer's MLP input under the BF16 model, 6.7 GB) are published as a
release and pinned file by file in `configs/sources.lock.json` (source `calibration`):

```bash
bittrellis calibration fetch       # -> models/hpc01-calib-v1/, verified against the lock
```
