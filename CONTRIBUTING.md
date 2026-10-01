# Contributing

Before opening a PR:

1. `pip install -e .`
2. `python eval.py`
3. `python tests/test_cli.py`

Both scripts must exit 0 (`RESULT: PASS`). CI runs both on every push/PR (`.github/workflows/ci.yml`).

`tests/test_pytest_wrapper.py` wraps them for plain `pytest`.

Nonsensical input (geometry, VRAM, safety fraction, context, step size) must raise `ValueError`.
