# Changelog

All notable changes to this project are documented in this file.

## [0.1.1] - 2026-07-31

- Reject an explicitly-passed geometry value (`--vram`, `--weights`, layer/head
  fields) that differs from the built-in preset while `--gpu-name`/`--model-name`
  still claim the preset rig. Previously `--vram 240` (a fat-fingered 24)
  printed a SAFE verdict for the documented 65536-context wedge configuration,
  still labeled `@ RTX 3090`. `--vram` now uses the same `None` sentinel as the
  other geometry flags, and a diverged label requires it explicitly.
- Upper sanity bound on `vram_gib` (> 2048 GiB rejected) so a MiB value entered
  as GiB errors instead of inflating the budget by three orders of magnitude.
- Five new regression checks in `tests/test_cli.py` covering the preset-label override paths.

## [0.1.0] - 2026-07-07

Initial release.

- VRAM envelope model (`predict_vram`, `classify`), safe-context bisection
  (`safe_max_context`), and a wedge-recovery ladder (`recovery_ladder`).
- `gpuenvelope.calibrate`: ingests an nvidia-smi telemetry sweep and reports a
  genuine held-out validation RMSE against a fair naive-constant baseline.
- CLI (`python -m gpuenvelope.cli {classify,ceiling,calibrate}`).
- `eval.py` RED/GREEN self-test; `tests/test_cli.py` CLI-level regression
  coverage.
- `__post_init__` validation on `ModelSpec`/`GpuSpec` and a `safety_fraction`
  range check in `classify`, so physically nonsensical geometry raises
  instead of silently reporting a false SAFE verdict.
