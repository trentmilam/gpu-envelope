# Contributing

Before opening a PR:

1. Install the package: `pip install -e .`
2. Run the self-test and make sure it passes:
   ```
   python eval.py
   ```
3. Run the CLI regression tests:
   ```
   python tests/test_cli.py
   ```

Both must exit 0 (`RESULT: PASS`). CI runs the same two commands on every
push/PR (`.github/workflows/ci.yml`).

Keep the fail-loud contract intact: any physically meaningful input
(geometry, VRAM, safety fraction, context, step size) should raise
`ValueError` on a nonsensical value rather than silently producing a
misleading verdict.
