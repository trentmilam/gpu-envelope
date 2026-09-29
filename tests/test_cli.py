"""gpu-envelope CLI test - argv-level coverage for the cli.main() entrypoint.

    python tests/test_cli.py

Covers two things the bespoke eval.py self-test never touched:
  1. Basic argv-level smoke coverage for each documented CLI subcommand
     (classify / ceiling / calibrate) so an argparse wiring regression can't
     hide behind a green `python eval.py`.
  2. Regression coverage for the two exact repro commands that used to return
     a false SAFE verdict on physically nonsensical geometry (undetected
     zero head_dim; a >100% safety_fraction) - both must now error instead.
"""
import contextlib
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gpuenvelope.cli import main as cli_main  # noqa: E402


def _run(argv):
    """Run cli_main(argv), capturing stdout/stderr. Returns (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


def main() -> int:
    checks = {}

    # ---- argv-level smoke coverage for each documented entrypoint -----------
    code, out, _ = _run(["classify", "--context", "49152"])
    checks["cli_classify_exit_0"] = code == 0
    checks["cli_classify_prints_verdict"] = "VERDICT: SAFE" in out

    code, out, _ = _run(["ceiling", "--step", "256"])
    checks["cli_ceiling_exit_0"] = code == 0
    checks["cli_ceiling_prints_result"] = "SAFE MAX CONTEXT" in out

    code, out, _ = _run(["calibrate"])
    checks["cli_calibrate_exit_0"] = code == 0
    checks["cli_calibrate_prints_rmse"] = "held-out RMSE" in out

    # ---- regression: previously-false-SAFE inputs must now error ------------
    # was: VERDICT: SAFE, headroom: 5.100 GiB under budget (kv_width collapses
    # to 0 when head_dim=0)
    code, out, err = _run(["classify", "--context", "65536", "--head-dim", "0"])
    checks["zero_head_dim_errors"] = code != 0
    checks["zero_head_dim_not_safe"] = "VERDICT: SAFE" not in out

    # was: VERDICT: SAFE, headroom: 1184.393 GiB under budget (safety_fraction
    # 50 == 5000% of VRAM)
    code, out, err = _run(["classify", "--context", "65536", "--safety-fraction", "50"])
    checks["oversized_safety_fraction_errors"] = code != 0
    checks["oversized_safety_fraction_not_safe"] = "VERDICT: SAFE" not in out

    # ---- bad input surfaces a one-line "error: ..." message on stderr,
    # not a multi-frame traceback ------------------------------------------
    checks["error_message_not_traceback"] = (
        "error:" in err and "Traceback" not in err
    )

    # ---- regression: numeric override under the trusted preset label --------
    # was: --vram 240 (a fat-fingered 24) printed VERDICT: SAFE for the
    # documented 65536-ctx wedge configuration, still labeled "@ RTX 3090"
    code, out, err = _run(["classify", "--context", "65536", "--vram", "240"])
    checks["vram_override_under_preset_label_errors"] = code != 0
    checks["vram_override_under_preset_label_not_safe"] = "VERDICT: SAFE" not in out

    # explicitly passing the preset's own value stays allowed
    code, out, _ = _run(["classify", "--context", "65536", "--vram", "24"])
    checks["explicit_preset_equal_vram_still_runs"] = (
        code == 0 and "WOULD-WEDGE" in out
    )

    # the legitimate path (a custom rig with full geometry) still works
    code, out, _ = _run([
        "classify", "--context", "49152", "--gpu-name", "RTX 5090", "--vram", "32",
        "--model-name", "custom-30b", "--weights", "18", "--n-layers", "48",
        "--hidden", "2048", "--n-kv-heads", "4", "--head-dim", "128",
        "--cuda-context-gib", "1.4", "--compute-act-count", "2",
    ])
    checks["full_custom_geometry_still_runs"] = code == 0

    # a MiB value entered as GiB is rejected even on a fully-specified rig
    code, out, err = _run([
        "classify", "--context", "49152", "--gpu-name", "big", "--vram", "32607",
        "--model-name", "custom-30b", "--weights", "18", "--n-layers", "48",
        "--hidden", "2048", "--n-kv-heads", "4", "--head-dim", "128",
        "--cuda-context-gib", "1.4", "--compute-act-count", "2",
    ])
    checks["mib_as_gib_unit_slip_errors"] = code != 0

    print("=== gpu-envelope CLI test (measured) ===")
    for k, v in checks.items():
        print(f"{'OK  ' if v else 'FAIL'} {k}")
    passed = all(checks.values())
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
