"""VRAM envelope model, safe-context bisection, and wedge-recovery ladder.

The VRAM model is a CALIBRATED APPROXIMATION of llama.cpp GPU memory use for a
decoder-only transformer served fully on one card. It is deterministic and
depends only on the config numbers passed in -- no wall-clock, no RNG, no I/O.

Memory decomposition (all in GiB):

    total = weights + kv_cache + cuda_context + compute_buffer

  * weights      : the quantized weight footprint resident in VRAM. Passed in
                   per-model (measured from the GGUF on disk); we do not try to
                   re-derive it from bits-per-weight here.
  * kv_cache     : llama.cpp sizes the KV cache as
                       2 (K and V)  x  n_layers  x  context
                       x  (n_kv_heads * head_dim)  x  bytes_per_element
                   n_kv_heads * head_dim is the *grouped* KV width -- this is
                   why GQA models (few KV heads) fit far more context than the
                   hidden dim alone would suggest. bytes_per_element comes from
                   the KV-cache quantization (f16 / q8_0 / q4_0).
                   Ref: llama.cpp llama_kv_cache_init / `--cache-type-k|-v`.
  * cuda_context : fixed CUDA runtime + driver reservation (a constant).
  * compute_buffer: activation / attention scratch that scales with context.

Calibration target (measured on a reference RTX 3090 rig):
  RTX 3090 24 GiB, Qwen3-30B-A3B IQ4_XS, f16 KV.
    - ctx 49152 sits at ~20.5 GiB and runs 0-Xid (SAFE).
    - ctx 65536 pushed KV to ~22 GiB and cold-wedged the card (WOULD-WEDGE).
  This model predicts ~20.7 GiB @ 49152 and ~22.5 GiB @ 65536 -- consistent
  with both observations. See README for the honest-scope note: exact bytes
  need the physical card; this tool finds the *ceiling* and the recovery path.
"""
from __future__ import annotations

from dataclasses import dataclass

GIB = 1024 ** 3

# bytes per stored KV element by llama.cpp cache-type. q8_0/q4_0 carry a small
# block-scale overhead in practice; these are the nominal element sizes.
KV_QUANT_BYTES = {
    "f16": 2.0,
    "q8_0": 1.0,
    "q4_0": 0.5,
}

# --- fixed model constants (calibrated to the 3090 / llama.cpp case) ---------
# CUDA runtime + driver reservation, roughly constant regardless of context.
CUDA_CONTEXT_GIB = 0.60
# Effective number of context-sized activation buffers held during a forward
# pass (attention scratch, logits staging, etc.). Folded into one factor so the
# compute buffer scales linearly with context * hidden * 2 bytes.
COMPUTE_BUFFER_ACT_COUNT = 4.0
# Fraction of raw VRAM we allow the predicted total to occupy. The gap is the
# OOM / thermal safety margin -- consumer cards wedge before hitting the metal.
DEFAULT_SAFETY_FRACTION = 0.90


@dataclass(frozen=True)
class GpuSpec:
    name: str
    vram_gib: float

    def budget_gib(self, safety_fraction: float = DEFAULT_SAFETY_FRACTION) -> float:
        """Usable VRAM after the safety margin."""
        return self.vram_gib * safety_fraction


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int
    hidden: int          # embedding / residual-stream width
    n_kv_heads: int      # grouped-query KV heads (== attn heads if no GQA)
    head_dim: int        # per-head dimension
    weights_gib: float   # measured quantized weight footprint resident in VRAM

    @property
    def kv_width(self) -> int:
        """Grouped KV width = n_kv_heads * head_dim (bytes below multiply this)."""
        return self.n_kv_heads * self.head_dim


@dataclass(frozen=True)
class VramBreakdown:
    weights_gib: float
    kv_cache_gib: float
    cuda_context_gib: float
    compute_buffer_gib: float

    @property
    def total_gib(self) -> float:
        return (
            self.weights_gib
            + self.kv_cache_gib
            + self.cuda_context_gib
            + self.compute_buffer_gib
        )


def predict_vram(model: ModelSpec, context: int, kv_quant: str = "f16") -> VramBreakdown:
    """Deterministic VRAM prediction (GiB) for a model at a given context.

    Raises ValueError on an unknown kv_quant or a non-positive context so the
    tool fails loud rather than silently guessing.
    """
    if kv_quant not in KV_QUANT_BYTES:
        raise ValueError(
            f"unknown kv_quant {kv_quant!r}; expected one of {sorted(KV_QUANT_BYTES)}"
        )
    if context <= 0:
        raise ValueError(f"context must be positive, got {context}")

    bytes_per_elem = KV_QUANT_BYTES[kv_quant]
    kv_bytes = 2 * model.n_layers * context * model.kv_width * bytes_per_elem
    compute_bytes = COMPUTE_BUFFER_ACT_COUNT * context * model.hidden * 2.0

    return VramBreakdown(
        weights_gib=model.weights_gib,
        kv_cache_gib=kv_bytes / GIB,
        cuda_context_gib=CUDA_CONTEXT_GIB,
        compute_buffer_gib=compute_bytes / GIB,
    )


def classify(
    gpu: GpuSpec,
    model: ModelSpec,
    context: int,
    kv_quant: str = "f16",
    safety_fraction: float = DEFAULT_SAFETY_FRACTION,
) -> dict:
    """Classify a candidate config as 'safe' or 'would-wedge'.

    Returns a dict with the verdict, the predicted total, the budget, the
    over/under-budget delta, and the full breakdown.
    """
    bd = predict_vram(model, context, kv_quant)
    budget = gpu.budget_gib(safety_fraction)
    total = bd.total_gib
    safe = total <= budget
    return {
        "verdict": "safe" if safe else "would-wedge",
        "safe": safe,
        "context": context,
        "kv_quant": kv_quant,
        "predicted_total_gib": total,
        "budget_gib": budget,
        "vram_gib": gpu.vram_gib,
        "over_budget_gib": total - budget,   # positive => over budget
        "headroom_gib": budget - total,      # positive => headroom left
        "breakdown": bd,
    }


def safe_max_context(
    gpu: GpuSpec,
    model: ModelSpec,
    kv_quant: str = "f16",
    safety_fraction: float = DEFAULT_SAFETY_FRACTION,
    step: int = 256,
    ctx_hi: int = 1_048_576,
) -> int:
    """Bisect the largest context (multiple of `step`) that classifies as safe.

    Returns 0 if even one `step` of context does not fit (weights alone exceed
    budget). The result is rounded DOWN to a multiple of `step`, matching how
    llama.cpp contexts are usually configured.
    """
    def fits(ctx: int) -> bool:
        return classify(gpu, model, ctx, kv_quant, safety_fraction)["safe"]

    lo_units = 1                      # 1 * step is the smallest candidate
    if not fits(lo_units * step):
        return 0

    hi_units = ctx_hi // step
    if fits(hi_units * step):         # even the ceiling fits -> return it
        return hi_units * step

    # invariant: fits(lo_units*step) is True, fits(hi_units*step) is False
    while hi_units - lo_units > 1:
        mid_units = (lo_units + hi_units) // 2
        if fits(mid_units * step):
            lo_units = mid_units
        else:
            hi_units = mid_units
    return lo_units * step


def recovery_ladder(context: int, verdict: str, gpu_name: str) -> list[dict]:
    """Escalating wedge-recovery ladder, least->most disruptive.

    The rungs mirror a standard anti-wedge recovery ladder: try the cheap
    software restart first, then a network-triggered OS reboot, and only fall
    back to a physical AC cold-cycle if the card is truly hung (no remote reset
    on a wedged consumer GPU).
    """
    ladder = [
        {
            "rung": 1,
            "name": "soft-reboot",
            "action": "restart the llama.cpp/ollama server process (frees KV; "
                      "no OS reboot). Reload at a context <= the safe ceiling.",
            "disruptive": "low",
            "recovers_hard_wedge": False,
        },
        {
            "rung": 2,
            "name": "wake-on-LAN / SSH reboot",
            "action": f"SSH into {gpu_name} host and `sudo reboot`; if the box is "
                      "down, send a Wake-on-LAN magic packet to bring it back.",
            "disruptive": "medium",
            "recovers_hard_wedge": False,
        },
        {
            "rung": 3,
            "name": "cold power-cycle",
            "action": "hard AC cold-cycle via smart plug (only cure for an "
                      "Xid-wedged card: GSP hang, MMU fault, illegal-memory). "
                      "Wait for the host to self-boot, then reload under ceiling.",
            "disruptive": "high",
            "recovers_hard_wedge": True,
        },
    ]
    for rung in ladder:
        rung["context"] = context
        rung["triggered_by"] = verdict
    return ladder


# --- calibrated reference specs ---------------------------------------------
# RTX 3090: 24 GiB consumer card (a secondary GPU node).
RTX_3090 = GpuSpec(name="RTX 3090", vram_gib=24.0)

# Qwen3-30B-A3B geometry (48 layers, hidden 2048, GQA 4 KV heads, head_dim 128).
# weights_gib = measured IQ4_XS footprint resident in VRAM (~14.9 GiB).
QWEN3_30B_A3B = ModelSpec(
    name="Qwen3-30B-A3B IQ4_XS",
    n_layers=48,
    hidden=2048,
    n_kv_heads=4,
    head_dim=128,
    weights_gib=14.9,
)
