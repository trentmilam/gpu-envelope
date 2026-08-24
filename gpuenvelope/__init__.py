"""gpu-envelope - safe-operating-envelope finder + wedge-recovery ladder.

Public API (zero third-party dependencies):
    GpuSpec, ModelSpec, KV_QUANT_BYTES
    predict_vram, classify, safe_max_context, recovery_ladder

Calibration API (backed by .calibrate, which needs numpy) is lazily imported
on first access so the core safety API above never pulls in numpy.
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

__version__ = "0.1.1"

_CALIBRATE_NAMES = {
    "load_telemetry_csv",
    "fit_gqa_aware",
    "fit_naive_constant",
    "calibrate_and_validate",
    "Fit",
    "DEFAULT_TELEMETRY_CSV",
}

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
    "__version__",
]


def __getattr__(name):
    """PEP 562 lazy import: only pulls in .calibrate (and numpy) on demand."""
    if name in _CALIBRATE_NAMES:
        from . import calibrate
        return getattr(calibrate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
