# Changelog

All notable changes to this project are documented in this file.

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
