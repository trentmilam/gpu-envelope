"""gpu-envelope — safe-operating-envelope finder + wedge-recovery ladder.

Public API:
    GpuSpec, ModelSpec, KV_QUANT_BYTES
    predict_vram, classify, safe_max_context, recovery_ladder
"""
from .envelope import (
    GpuSpec,
    ModelSpec,
    KV_QUANT_BYTES,
    VramBreakdown,
    predict_vram,
    classify,
    safe_max_context,
    recovery_ladder,
    RTX_3090,
    QWEN3_30B_A3B,
)
from .calibrate import (
    load_telemetry_csv,
    fit_gqa_aware,
    fit_naive_constant,
    calibrate_and_validate,
    Fit,
    DEFAULT_TELEMETRY_CSV,
)

__all__ = [
    "GpuSpec",
    "ModelSpec",
    "KV_QUANT_BYTES",
    "VramBreakdown",
    "predict_vram",
    "classify",
    "safe_max_context",
    "recovery_ladder",
    "RTX_3090",
    "QWEN3_30B_A3B",
    "load_telemetry_csv",
    "fit_gqa_aware",
    "fit_naive_constant",
    "calibrate_and_validate",
    "Fit",
    "DEFAULT_TELEMETRY_CSV",
]
