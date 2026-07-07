"""Thin pytest-discoverable wrapper around eval.py and tests/test_cli.py.

Both are standalone exit-code scripts by design (see CONTRIBUTING.md) and
predate this wrapper. This file exists purely so a reviewer who runs `pytest`
by reflex, with no other args, collects and runs *something* here instead of
silently collecting zero tests.

    pytest tests/test_pytest_wrapper.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval as _eval          # noqa: E402
import test_cli as _test_cli  # noqa: E402


def test_eval_self_test():
    assert _eval.main() == 0


def test_cli_regression():
    assert _test_cli.main() == 0
