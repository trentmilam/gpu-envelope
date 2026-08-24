# gpu-envelope

[![CI](https://github.com/trentmilam/gpu-envelope/actions/workflows/ci.yml/badge.svg)](https://github.com/trentmilam/gpu-envelope/actions/workflows/ci.yml)

When you run a large language model on your own GPU, one setting can quietly
wreck the session: context length, the amount of text the model holds in
memory at once. Push it too high and the card runs past its memory (VRAM) or
heat limits and locks up hard, a state this doc calls a wedge, and the only
fix is often a physical power-cycle. gpu-envelope estimates the largest safe
settings for a given GPU and model before you run anything, and it lays out
an escalating recovery ladder for when a card wedges anyway.

It targets local LLMs served on a single consumer GPU (llama.cpp / ollama).
Given a model's geometry and a card's VRAM, it:

1. Models VRAM deterministically: weights, KV cache (the memory buffer that
   grows with how much context the model is holding), CUDA context, and
   compute buffer, as a function of context length and KV-cache quantization.
2. Bisects the safe max context for a VRAM budget, leaving margin against
   running out of memory or exceeding the card's thermal budget (its heat
   headroom before it throttles or hangs).
3. Classifies a candidate config as `safe` or `would-wedge`, reporting the
   predicted VRAM against the budget.
4. Emits an escalating recovery ladder for when a card does wedge:
   soft-reboot, then wake-on-LAN/SSH reboot, then cold power-cycle.

## Why

A wedged consumer GPU, the kind of hang tagged by an Xid error (a public
NVIDIA GPU error code) for an MMU fault, a GSP hang, or illegal memory
access, has no remote reset: recovery is a physical cold power-cycle. On the
rig this was calibrated against (an RTX 3090 serving a 30B model), context
49152 runs clean at ~20.5 GiB, but 65536 pushed the KV cache to ~22 GiB and
cold-wedged the card. This tool turns that hard-won operational limit into a
reusable measured-ceiling finder plus a documented recovery path, so you
find the ceiling on paper instead of by wedging the card.

## Run

Requires Python 3.9+.

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

Defaults model the RTX 3090 + Qwen3-30B-A3B IQ4_XS calibration case. Every
geometry field is overridable (`--vram`, `--n-layers`, `--hidden`,
`--n-kv-heads`, `--head-dim`, `--weights`, `--cuda-context-gib`,
`--compute-act-count`, `--kv-quant`, `--safety-fraction` — the fraction of
VRAM you're willing to use; 0.9 caps you at 90% of the card) to model any
other card/model. `--cuda-context-gib` and `--compute-act-count` are the two
physics constants calibrated on the reference rig: to model a different
card, `--model-name`/`--gpu-name` and every geometry field (including those
two) must be given explicitly. Leaving any of them at the reference-rig
default raises an error instead of silently mixing your numbers with the old
rig's.

## Self-test results

From `eval.py` (exit code `0`, all 30 checks pass, measured on the reference machine
2026-07-04, numpy 2.5.0). These are the physics model's self-consistency
checks against two real on-card anchor points, not an independent hardware
benchmark; see "Honest scope" below and the held-out calibration harness
further down for the number that actually validates on unseen data:

| Config | Predicted VRAM | Budget (90% of 24 GiB) | Verdict |
|--------|----------------|------------------------|---------|
| ctx 49152, f16 KV | 20.750 GiB | 21.600 GiB | safe (+0.850 GiB headroom) |
| ctx 65536, f16 KV | 22.500 GiB | 21.600 GiB | would-wedge (+0.900 GiB over) |

- Bisected safe-max context (f16 KV, step 256): 57088.
- q4_0 KV raises the ceiling to 159744 (cheaper KV buys context).
- Two-anchor consistency: against the two real on-card observations (~20.5
  GiB @ 49152 clean, ~22 GiB @ 65536 wedged), which are not derived from the
  model, the shipped constants track to within ~2.3% max relative error.
  This is an in-sample consistency check on two anchors, not a validation
  set; see the calibration harness below for the honest held-out number.

## Calibration harness: held-out validation and a measured baseline A/B

The earlier story rested on two anecdotal points fit by three free
constants — underdetermined, so the residual is trivially ~0 and proves
nothing about predictive power. `gpuenvelope/calibrate.py` fixes that: it
ingests an nvidia-smi telemetry sweep, fits the identifiable free constants
on a train split, and reports a genuine held-out validation RMSE
(root-mean-squared error between predicted and actual VRAM, lower is
better) on points the fit never saw. (From telemetry alone only
`weights + cuda_context` (one intercept) and `COMPUTE_BUFFER_ACT_COUNT` are
identifiable; the KV slope is fixed by the model geometry, so it is a clean
2-parameter fit, well-determined from ≥5 points.)

It ships with a committed, clearly-labeled SYNTHETIC 8-point sweep
(`gpuenvelope/data/rtx3090_qwen3_30b_synthetic.csv`) so the test is offline
and deterministic; its two anchor rows are pinned to the real 49152/65536
observations. Drop in a real capture (command in the CSV header) for a real
number, a "fit to your card in 60s" demo:

```
python -m gpuenvelope.cli calibrate                 # synthetic fixture
python -m gpuenvelope.cli calibrate --csv mycard.csv
```

Fair baseline A/B (measured by `eval.py`, 2 interior points held out):

| Estimator | What it is | Held-out RMSE | @ ctx 65536 (train point, not held out) |
|-----------|-----------|---------------|---------------------|
| naive constant | weights + fixed overhead — one context-independent number (HF Accelerate `estimate-memory` / "can I run this" calculators), given its best least-squares constant (no strawman) | 1.875 GiB | predicts 19.38 GiB → "safe" (fail-open — would wedge the card) |
| GQA-aware (this tool) | the KV/compute physics model, calibrated | 0.032 GiB | predicts 21.99 GiB → would-wedge (real: 22.0 GiB) |

The GQA-aware model's held-out RMSE is ~59× lower on the genuinely held-out
interior points. The two right-hand columns are not held out:
`_holdout_split` always keeps the min/max-context rows (including 65536,
the wedge point) in TRAIN, so the "@ ctx 65536" column shows in-sample
accuracy, not validation. Even so, the naive estimator under-predicts the
wedge config by ~2.6 GiB and calls it safe — it would have cold-wedged the
card. These numbers are measured against the synthetic fixture; a number is
only as good as its telemetry, so run it on real captures before quoting it.

- Honest calibration finding: fitting to the real-anchored data drives
  `COMPUTE_BUFFER_ACT_COUNT` slightly negative (~−0.15) — the shipped
  constant (4.0) mildly over-states context growth, consistent with the
  ~2.3% over-prediction at high context. The harness surfaces this rather
  than hiding it.
- `eval.py` also runs a mis-calibrated control (wrong resident-weight
  footprint, ~23% error) that must be rejected, so the two-anchor check is
  non-vacuous rather than circular.

### RED/GREEN self-test

`eval.py` is the first-milestone self-test. It genuinely catches the injected
failure case and passes the clean case through the same physics model, with
no hard-coded verdicts:

- RED — the known-wedge context (65536) is flagged `would-wedge` with the
  predicted total strictly over budget.
- GREEN — the proven-safe context (49152) passes with positive headroom.
- The bisected ceiling also lies strictly between the two and is itself safe
  while `ceiling + step` is not; VRAM is monotonic in context; KV-quant sizes
  order `q4_0 < q8_0 < f16`; bad inputs fail loud; predictions are deterministic.

## Honest scope

- The VRAM model is a calibrated approximation, not a measurement. It uses
  the standard llama.cpp KV-cache sizing formula
  (`2 · n_layers · context · n_kv_heads · head_dim · bytes_per_elem`) plus a
  constant CUDA-context reservation and a linear compute buffer. The
  constants (`CUDA_CONTEXT_GIB`, `COMPUTE_BUFFER_ACT_COUNT`, weight
  footprint) are fit to one rig's observations.
- Real numbers need the physical card. Actual VRAM varies with driver
  version, batch/ubatch size, flash-attention, KV-cache block overhead, and
  allocator fragmentation. Treat the output as a planning ceiling to verify
  on hardware, not a guarantee.
- The recovery ladder describes operational procedure; the tool does not
  execute reboots or power-cycles.

## Prior art

- KV-cache sizing: the formula follows llama.cpp (`llama_kv_cache_init`;
  `--cache-type-k` / `--cache-type-v`) and the GQA KV width
  (`n_kv_heads · head_dim`), which is why grouped-query models fit far more
  context than the hidden dim implies.
- VRAM estimators exist (e.g. community "can I run this model" calculators
  and HF Accelerate's `estimate-memory`), but they target weights and a
  single context point. The combination here — a bisected safe ceiling with
  an explicit out-of-memory/thermal margin plus a wedge-recovery ladder — is
  not, to my knowledge, shipped as a standalone OSS tool.

## Name / registry note

`gpu-envelope` is a generic name. If published, verify availability on PyPI /
GitHub and rename if it collides (e.g. `llm-vram-envelope`,
`kv-ceiling-finder`). The importable package is `gpuenvelope`.

## License

MIT — Copyright (c) 2026 Trent Milam. See `LICENSE`.
