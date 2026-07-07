"""gpu-envelope CLI - classify a config, find the safe ceiling, show recovery.

Examples (from repo root, using the project venv):

  python -m gpuenvelope.cli classify --context 65536
  python -m gpuenvelope.cli ceiling
  python -m gpuenvelope.cli ceiling --kv-quant q8_0

Defaults model to Qwen3-30B-A3B IQ4_XS and gpu to the RTX 3090 (the calibration
case). To model a different card/model, override --model-name/--gpu-name
*and* every geometry flag (--weights, --n-layers, --hidden, --n-kv-heads,
--head-dim, --cuda-context-gib, --compute-act-count) -- leaving any of them
at the reference-rig default raises an error rather than silently mixing a
new label with the old rig's numbers.
"""
from __future__ import annotations

import argparse
import sys

from .envelope import (
    GpuSpec,
    ModelSpec,
    KV_QUANT_BYTES,
    DEFAULT_SAFETY_FRACTION,
    CUDA_CONTEXT_GIB,
    COMPUTE_BUFFER_ACT_COUNT,
    classify,
    safe_max_context,
    recovery_ladder,
    RTX_3090,
    QWEN3_30B_A3B,
)


def _build_gpu(args) -> GpuSpec:
    return GpuSpec(name=args.gpu_name, vram_gib=args.vram)


def _build_model(args) -> ModelSpec:
    # A `None` sentinel means the flag was not passed on the CLI; fill it from
    # the reference preset so the plain preset invocation behaves exactly as
    # before. (Value-equality can't tell "omitted" from "explicitly set to a
    # value that happens to equal the preset default", e.g. head_dim 128.)
    return ModelSpec(
        name=args.model_name,
        n_layers=args.n_layers if args.n_layers is not None else QWEN3_30B_A3B.n_layers,
        hidden=args.hidden if args.hidden is not None else QWEN3_30B_A3B.hidden,
        n_kv_heads=args.n_kv_heads if args.n_kv_heads is not None else QWEN3_30B_A3B.n_kv_heads,
        head_dim=args.head_dim if args.head_dim is not None else QWEN3_30B_A3B.head_dim,
        weights_gib=args.weights if args.weights is not None else QWEN3_30B_A3B.weights_gib,
    )


def _cuda_context_gib(args) -> float:
    return args.cuda_context_gib if args.cuda_context_gib is not None else CUDA_CONTEXT_GIB


def _compute_act_count(args) -> float:
    return args.compute_act_count if args.compute_act_count is not None else COMPUTE_BUFFER_ACT_COUNT


def _add_common(p: argparse.ArgumentParser) -> None:
    # The geometry flags default to a `None` sentinel (not the preset value) so
    # `_check_geometry_fully_specified` can tell an omitted flag from one the
    # user explicitly set to a value that equals the preset default. Omitted
    # (None) flags are filled from the preset in `_build_model` / the helpers.
    p.add_argument("--gpu-name", default=RTX_3090.name)
    p.add_argument("--vram", type=float, default=RTX_3090.vram_gib,
                   help="GPU VRAM in GiB")
    p.add_argument("--model-name", default=QWEN3_30B_A3B.name)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--weights", type=float, default=None,
                   help="quantized weight footprint in VRAM (GiB)")
    p.add_argument("--cuda-context-gib", type=float, default=None,
                   help="fixed CUDA runtime + driver reservation (GiB); "
                        "override for a non-reference-rig GPU")
    p.add_argument("--compute-act-count", type=float, default=None,
                   help="effective context-sized activation-buffer count; "
                        "override for a non-reference-rig GPU")
    p.add_argument("--kv-quant", default="f16", choices=sorted(KV_QUANT_BYTES))
    p.add_argument("--safety-fraction", type=float, default=DEFAULT_SAFETY_FRACTION)


def _check_geometry_fully_specified(args) -> None:
    """Refuse an ambiguous mix of a non-reference model/GPU label with any
    geometry field still at the reference-rig (Qwen3-30B-A3B / RTX 3090)
    default.

    Without this, forgetting a single geometry flag while modeling a
    different model/GPU silently falls back to the reference rig's numbers
    under the *new* label -- e.g. printing a Llama-3-8B verdict built from
    the Qwen3-30B-A3B weight footprint. Raises ValueError naming exactly
    which fields are still defaulted so the fix is obvious.
    """
    diverges = args.model_name != QWEN3_30B_A3B.name or args.gpu_name != RTX_3090.name
    if not diverges:
        return
    # A field counts as "not provided" only when it is genuinely omitted
    # (still the `None` sentinel) -- NOT when the user explicitly passed a value
    # that happens to equal the preset default (e.g. --head-dim 128).
    provided = {
        "--weights": args.weights,
        "--n-layers": args.n_layers,
        "--hidden": args.hidden,
        "--n-kv-heads": args.n_kv_heads,
        "--head-dim": args.head_dim,
        "--cuda-context-gib": args.cuda_context_gib,
        "--compute-act-count": args.compute_act_count,
    }
    omitted = [flag for flag, value in provided.items() if value is None]
    if omitted:
        raise ValueError(
            "--model-name/--gpu-name differ from the built-in "
            f"{QWEN3_30B_A3B.name} / {RTX_3090.name} preset, but "
            + ", ".join(omitted)
            + " was/were not provided. Pass explicit values for all of them "
              "(or leave --model-name/--gpu-name at the preset) -- otherwise "
              "the verdict would be silently computed from the wrong "
              "card/model's numbers."
        )


def _print_breakdown(res: dict) -> None:
    bd = res["breakdown"]
    print(f"  weights        {bd.weights_gib:7.3f} GiB")
    print(f"  kv_cache       {bd.kv_cache_gib:7.3f} GiB")
    print(f"  cuda_context   {bd.cuda_context_gib:7.3f} GiB")
    print(f"  compute_buffer {bd.compute_buffer_gib:7.3f} GiB")
    print(f"  --------------")
    print(f"  total          {bd.total_gib:7.3f} GiB   (budget {res['budget_gib']:.3f} of {res['vram_gib']:.1f})")


def cmd_classify(args) -> int:
    _check_geometry_fully_specified(args)
    gpu, model = _build_gpu(args), _build_model(args)
    res = classify(gpu, model, args.context, args.kv_quant, args.safety_fraction,
                   _cuda_context_gib(args), _compute_act_count(args))
    print(f"{model.name} @ {gpu.name}  context={args.context}  kv={args.kv_quant}")
    _print_breakdown(res)
    print(f"\nVERDICT: {res['verdict'].upper()}")
    if res["safe"]:
        print(f"  headroom: {res['headroom_gib']:.3f} GiB under budget")
    else:
        print(f"  OVER BUDGET by {res['over_budget_gib']:.3f} GiB")
        print("\nrecovery ladder if this config wedges the card:")
        for r in recovery_ladder(args.context, res["verdict"], gpu.name):
            print(f"  [{r['rung']}] {r['name']} ({r['disruptive']}): {r['action']}")
    return 0


def cmd_ceiling(args) -> int:
    _check_geometry_fully_specified(args)
    gpu, model = _build_gpu(args), _build_model(args)
    ctx = safe_max_context(gpu, model, args.kv_quant, args.safety_fraction, step=args.step,
                           cuda_context_gib=_cuda_context_gib(args),
                           compute_act_count=_compute_act_count(args))
    print(f"{model.name} @ {gpu.name}  kv={args.kv_quant}  safety={args.safety_fraction}")
    if ctx == 0:
        print("SAFE MAX CONTEXT: 0 (weights alone exceed the budget)")
        return 0
    res = classify(gpu, model, ctx, args.kv_quant, args.safety_fraction,
                   _cuda_context_gib(args), _compute_act_count(args))
    print(f"SAFE MAX CONTEXT: {ctx}  (predicted {res['predicted_total_gib']:.3f} GiB, "
          f"headroom {res['headroom_gib']:.3f} GiB)")
    return 0


def cmd_calibrate(args) -> int:
    from .calibrate import calibrate_and_validate, DEFAULT_TELEMETRY_CSV
    model = _build_model(args)
    gpu = _build_gpu(args)
    csv_path = args.csv or DEFAULT_TELEMETRY_CSV
    r = calibrate_and_validate(csv_path, model, kv_quant=args.kv_quant,
                               n_holdout=args.holdout,
                               budget_gib=gpu.budget_gib(args.safety_fraction))
    print(f"calibrating {model.name} @ {gpu.name} against {csv_path}")
    print(f"  {r['n_points']} load points | train {r['train_contexts']} | "
          f"held-out {r['holdout_contexts']}")
    g = r["gqa_fit"]
    print(f"  fitted intercept(weights+cuda) = {g.intercept_gib:.3f} GiB | "
          f"act_count = {g.act_count:.3f}")
    print()
    print(f"  GQA-aware (this tool) held-out RMSE : {r['gqa_holdout_rmse_gib']:.3f} GiB")
    print(f"  naive constant baseline held-out RMSE: {r['naive_holdout_rmse_gib']:.3f} GiB "
          f"({r['rmse_improvement_label']})")
    if r["true_vram_at_wedge_gib"] is not None:
        print(f"\n  @ ctx {r['wedge_ctx']} (real {r['true_vram_at_wedge_gib']:.2f} GiB, "
              f"budget {r['budget_gib']:.2f} GiB) -- IN-SAMPLE/TRAIN point, "
              f"not part of the held-out RMSE above:")
        print(f"    GQA-aware -> {r['gqa_wedge_pred_gib']:.2f} GiB  "
              f"[{'WOULD-WEDGE' if r['gqa_flags_wedge'] else 'safe'}]")
        print(f"    naive     -> {r['naive_wedge_pred_gib']:.2f} GiB  "
              f"[{'WOULD-WEDGE' if r['naive_flags_wedge'] else 'SAFE (fail-open!)'}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpu-envelope", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="classify one context as safe/would-wedge")
    _add_common(c)
    c.add_argument("--context", type=int, required=True)
    c.set_defaults(func=cmd_classify)

    e = sub.add_parser("ceiling", help="bisect the safe max context")
    _add_common(e)
    e.add_argument("--step", type=int, default=256)
    e.set_defaults(func=cmd_ceiling)

    cal = sub.add_parser("calibrate",
                         help="fit + validate the VRAM model against a telemetry CSV")
    _add_common(cal)
    cal.add_argument("--csv", default=None,
                     help="nvidia-smi sweep CSV (default: committed synthetic fixture)")
    cal.add_argument("--holdout", type=int, default=2,
                     help="number of interior points held out for validation RMSE")
    cal.set_defaults(func=cmd_calibrate)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
