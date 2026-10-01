# Changelog

## [0.1.1] - 2026-07-31

- Reject an explicit geometry value (`--vram`, `--weights`, layer/head fields) that differs from the built-in preset while `--gpu-name`/`--model-name` still claim the preset rig
- `--vram` uses the same `None` sentinel as the other geometry flags; a diverged label requires it explicitly
- Upper bound on `vram_gib` (> 2048 GiB rejected)
- Five regression checks in `tests/test_cli.py` for the preset-label override paths

## [0.1.0] - 2026-07-07

Initial release.

- VRAM envelope model (`predict_vram`, `classify`), safe-context bisection (`safe_max_context`), wedge-recovery ladder (`recovery_ladder`)
- `gpuenvelope.calibrate`: nvidia-smi telemetry sweep, held-out validation RMSE against a naive-constant baseline
- CLI (`python -m gpuenvelope.cli {classify,ceiling,calibrate}`)
- `eval.py` RED/GREEN self-test; `tests/test_cli.py` CLI regression checks
- `__post_init__` validation on `ModelSpec`/`GpuSpec`; `safety_fraction` range check in `classify`
