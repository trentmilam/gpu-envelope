# gpu-envelope

[![CI](https://github.com/trentmilam/gpu-envelope/actions/workflows/ci.yml/badge.svg)](https://github.com/trentmilam/gpu-envelope/actions/workflows/ci.yml)

OOM defense for local LLMs on a single consumer GPU (llama.cpp / ollama).

- Models VRAM from model geometry: weights, KV cache, CUDA context, compute buffer
- Bisects the safe max context for a VRAM budget, with OOM/thermal margin
- Classifies a config as `safe` or `would-wedge`
- Emits a recovery ladder: soft-reboot, wake-on-LAN/SSH reboot, cold power-cycle

## Run

Python 3.9+.

```
# 0. install (from repo root):
pip install -e .

# self-test (RED/GREEN), from repo root:
python eval.py

# CLI, from repo root:
python -m gpuenvelope.cli classify --context 65536
python -m gpuenvelope.cli ceiling
python -m gpuenvelope.cli ceiling --kv-quant q4_0
```

Defaults model the RTX 3090 + Qwen3-30B-A3B IQ4_XS calibration case.

Overridable: `--vram`, `--n-layers`, `--hidden`, `--n-kv-heads`, `--head-dim`, `--weights`,
`--cuda-context-gib`, `--compute-act-count`, `--kv-quant`, `--safety-fraction` (0.9 = 90% of the card).

Other cards and models: give `--model-name`/`--gpu-name` and every geometry field, including
`--cuda-context-gib` and `--compute-act-count`. A field left at the reference default raises an error.

## Self-test results

`eval.py`: exit code `0`, all 30 checks pass. Measured on the reference machine 2026-07-04, numpy 2.5.0.

| Config | Predicted VRAM | Budget (90% of 24 GiB) | Verdict |
|--------|----------------|------------------------|---------|
| ctx 49152, f16 KV | 20.750 GiB | 21.600 GiB | safe (+0.850 GiB headroom) |
| ctx 65536, f16 KV | 22.500 GiB | 21.600 GiB | would-wedge (+0.900 GiB over) |

- Bisected safe-max context (f16 KV, step 256): 57088
- q4_0 KV ceiling: 159744
- Two-anchor check (~20.5 GiB @ 49152 clean, ~22 GiB @ 65536 wedged): within ~2.3% max relative error
- In-sample, two anchors. Held-out number is below

## Calibration harness

`gpuenvelope/calibrate.py` ingests an nvidia-smi telemetry sweep, fits on a train split, and
reports held-out validation RMSE.

- Fitted: `weights + cuda_context` (one intercept) and `COMPUTE_BUFFER_ACT_COUNT`
- KV slope fixed by model geometry
- Needs 5 or more points
- Ships a SYNTHETIC 8-point sweep: `gpuenvelope/data/rtx3090_qwen3_30b_synthetic.csv`
- Anchor rows pinned to the real 49152/65536 observations
- Real capture: command is in the CSV header
- Package: `gpuenvelope`

```
python -m gpuenvelope.cli calibrate                 # synthetic fixture
python -m gpuenvelope.cli calibrate --csv mycard.csv
```

Baseline A/B from `eval.py`, 2 interior points held out:

| Estimator | What it is | Held-out RMSE | @ ctx 65536 (train point, not held out) |
|-----------|-----------|---------------|---------------------|
| naive constant | weights + fixed overhead (HF Accelerate `estimate-memory` style), best least-squares constant | 1.875 GiB | predicts 19.38 GiB → "safe" (fail-open; would wedge the card) |
| GQA-aware (this tool) | KV/compute physics model, calibrated | 0.032 GiB | predicts 21.99 GiB → would-wedge (real: 22.0 GiB) |

- Held-out RMSE ~59× lower
- Min/max-context rows (including 65536) stay in TRAIN, so the right-hand column is in-sample
- Naive estimator under-predicts the wedge config by ~2.6 GiB
- Measured on the synthetic fixture
- Fit drives `COMPUTE_BUFFER_ACT_COUNT` slightly negative (~−0.15); shipped constant is 4.0
- `eval.py` also runs a mis-calibrated control (~23% weight-footprint error) that must be rejected

### RED/GREEN self-test

- RED: known-wedge context (65536) flagged `would-wedge`, predicted total over budget
- GREEN: proven-safe context (49152) passes with positive headroom
- Bisected ceiling lies between the two; ceiling is safe, `ceiling + step` is not
- VRAM monotonic in context
- KV-quant sizes order `q4_0 < q8_0 < f16`
- Bad inputs fail loud
- Predictions deterministic

## Limits

- Calibrated approximation, not a measurement
- KV sizing: `2 · n_layers · context · n_kv_heads · head_dim · bytes_per_elem`, plus a constant CUDA-context reservation and a linear compute buffer
- `CUDA_CONTEXT_GIB`, `COMPUTE_BUFFER_ACT_COUNT` and weight footprint fit to one rig
- Real VRAM varies with driver version, batch/ubatch size, flash-attention, KV-cache block overhead, allocator fragmentation
- Output is a planning ceiling to verify on hardware
- The recovery ladder is documentation; the tool does not reboot or power-cycle
- Wedged GPU: no remote reset; recovery is a cold power-cycle

## Related work

- llama.cpp KV-cache sizing (`llama_kv_cache_init`; `--cache-type-k` / `--cache-type-v`), GQA KV width (`n_kv_heads · head_dim`)
- Community "can I run this model" calculators, HF Accelerate `estimate-memory`: weights and a single context point

## License

MIT, copyright (c) 2026 Trent Milam. See `LICENSE`.
