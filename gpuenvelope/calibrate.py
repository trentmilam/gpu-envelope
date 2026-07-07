"""Real-card calibration harness for the VRAM envelope model.

The rest of the tool ships a *physics* VRAM model whose free constants were fit
to only two anecdotal on-card points -- with three free constants and two
observations that fit is underdetermined, so its in-sample residual is
trivially ~0 and says nothing about predictive power. This module closes that
gap: it INGESTS an nvidia-smi telemetry sweep (>=5 load points), fits the
identifiable free constants by least squares on a TRAIN split, and reports a
genuine **held-out validation RMSE** on points the fit never saw.

It also ships the fair head-to-head baseline the portfolio claim rests on: the
naive incumbent estimator (weights + fixed overhead, i.e. what HF Accelerate's
`estimate-memory` and community "can I run this" calculators report -- a single
context-independent number). We give that baseline its *best possible* constant
(the least-squares constant == the training mean), so it is not a strawman, and
then MEASURE that it still fails to track context-driven KV growth and would
call the known-wedge config "safe".

Identifiability note: from (context, vram) telemetry alone, `weights_gib` and
`CUDA_CONTEXT_GIB` are not separately identifiable -- only their sum (the
intercept) is. The KV slope is FIXED by the model geometry
(`2*n_layers*n_kv_heads*head_dim*bytes_per_elem`), so the only free slope
parameter is `COMPUTE_BUFFER_ACT_COUNT`. Two free parameters (intercept,
act_count) => identifiable from >=2 points, well-determined from >=5.

Deterministic, offline, numpy + stdlib only. No GPU workload is run here; a
live capture on your own card is optional (see the CSV header for the capture
command).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources

import numpy as np

from .envelope import GIB, KV_QUANT_BYTES, CUDA_CONTEXT_GIB, ModelSpec

MIB = 1024 ** 2

# Floor for the held-out-RMSE ratio denominator, and the bound past which the
# ratio is reported qualitatively rather than as a raw number (see
# calibrate_and_validate: on clean/near-linear telemetry the GQA-aware fit's
# held-out RMSE can land at a near-zero, but nonzero, floating-point residual,
# which would otherwise blow the ratio up to an absurd figure like
# "549755813888000x worse" on an otherwise perfectly healthy fit).
_RMSE_FLOOR_GIB = 1e-6
_RMSE_RATIO_SANE_BOUND = 1000.0


def load_telemetry_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse an nvidia-smi-style sweep CSV into (contexts, vram_gib).

    Expected columns: a `context` column and a memory column. Memory values may
    carry an nvidia-smi ``MiB``/``GiB`` unit suffix (stripped here); the memory
    header decides the unit (``MiB`` -> divide by 1024, ``GiB`` -> as-is).
    ``#`` comment lines and the header row are skipped. Raises ValueError on a
    malformed / empty file so a bad capture fails loud rather than silently
    yielding garbage constants.
    """
    if not os.path.isfile(path):
        raise ValueError(f"telemetry CSV not found: {path}")

    rows: list[tuple[int, float]] = []
    mem_unit_gib = 1.0 / 1024.0  # default: memory column is MiB
    header_seen = False
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                raise ValueError(f"malformed CSV row (need >=2 columns): {raw!r}")
            # header row: first cell is not an integer context
            if not header_seen and not parts[0].lstrip("-").isdigit():
                header_seen = True
                mem_hdr = parts[1].lower()
                if "gib" in mem_hdr:
                    mem_unit_gib = 1.0
                elif "mib" in mem_hdr:
                    mem_unit_gib = 1.0 / 1024.0
                continue
            header_seen = True
            try:
                ctx = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"bad context value {parts[0]!r}") from exc
            mem_tok = parts[1].replace("MiB", "").replace("GiB", "").strip()
            try:
                mem = float(mem_tok)
            except ValueError as exc:
                raise ValueError(f"bad memory value {parts[1]!r}") from exc
            if ctx <= 0 or mem <= 0:
                raise ValueError(f"non-positive telemetry row: ctx={ctx} mem={mem}")
            rows.append((ctx, mem * mem_unit_gib))

    if len(rows) < 2:
        raise ValueError(f"need >=2 telemetry rows, got {len(rows)} from {path}")
    rows.sort(key=lambda r: r[0])
    contexts = np.array([r[0] for r in rows], dtype=float)
    vram_gib = np.array([r[1] for r in rows], dtype=float)
    return contexts, vram_gib


def _kv_slope_gib_per_ctx(model: ModelSpec, kv_quant: str) -> float:
    """GiB of KV cache added per one token of context (FIXED by geometry)."""
    if kv_quant not in KV_QUANT_BYTES:
        raise ValueError(f"unknown kv_quant {kv_quant!r}")
    return 2 * model.n_layers * model.kv_width * KV_QUANT_BYTES[kv_quant] / GIB


def _compute_unit_gib_per_ctx(model: ModelSpec) -> float:
    """GiB of compute buffer per (act_count * one token of context)."""
    return model.hidden * 2.0 / GIB


@dataclass(frozen=True)
class Fit:
    """A calibrated predictor plus its identifiable constants."""
    kind: str                 # "gqa-aware" or "naive-constant"
    intercept_gib: float      # weights + cuda_context (gqa) / best constant (naive)
    act_count: float          # fitted COMPUTE_BUFFER_ACT_COUNT (0.0 for naive)
    kv_slope: float           # fixed KV GiB/ctx (0.0 for naive)
    weights_gib: float        # intercept - CUDA_CONTEXT_GIB (gqa) / nan (naive)
    compute_unit: float = 0.0  # GiB compute buffer per (act_count * ctx token)

    def predict(self, context) -> np.ndarray:
        c = np.asarray(context, dtype=float)
        return (self.intercept_gib
                + self.kv_slope * c
                + self.act_count * self.compute_unit * c)


def fit_gqa_aware(contexts: np.ndarray, vram_gib: np.ndarray,
                  model: ModelSpec, kv_quant: str = "f16") -> Fit:
    """Least-squares fit of the GQA-aware physics model's free constants.

    Fits [intercept, act_count]; the KV slope is fixed by geometry. Returns a
    Fit whose ``predict`` reproduces the same physics as ``predict_vram`` but
    with the calibrated constants.
    """
    contexts = np.asarray(contexts, dtype=float)
    vram_gib = np.asarray(vram_gib, dtype=float)
    if contexts.size < 2:
        raise ValueError("need >=2 training points to fit 2 free constants")
    kv_slope = _kv_slope_gib_per_ctx(model, kv_quant)
    comp_unit = _compute_unit_gib_per_ctx(model)
    # y with the KNOWN kv contribution removed; solve y = C0 + act_count*(comp_unit*ctx)
    y = vram_gib - kv_slope * contexts
    design = np.column_stack([np.ones_like(contexts), comp_unit * contexts])
    (c0, act_count), *_ = np.linalg.lstsq(design, y, rcond=None)
    return Fit(
        kind="gqa-aware",
        intercept_gib=float(c0),
        act_count=float(act_count),
        kv_slope=kv_slope,
        weights_gib=float(c0) - CUDA_CONTEXT_GIB,
        compute_unit=comp_unit,
    )


def fit_naive_constant(contexts: np.ndarray, vram_gib: np.ndarray) -> Fit:
    """Fair naive incumbent: a context-INDEPENDENT estimate (weights + fixed
    overhead, HF-Accelerate style). Its best least-squares constant is the mean
    of the training VRAM, which we hand it -- no crippling."""
    vram_gib = np.asarray(vram_gib, dtype=float)
    return Fit(
        kind="naive-constant",
        intercept_gib=float(vram_gib.mean()),
        act_count=0.0,
        kv_slope=0.0,
        weights_gib=float("nan"),
    )


def _holdout_split(n: int, n_holdout: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/holdout split: hold out `n_holdout` INTERIOR points
    spread across the (context-sorted) range, so validation tests interpolation
    rather than pure extrapolation."""
    if n_holdout < 1 or n_holdout >= n - 1:
        raise ValueError(f"n_holdout must be in [1, n-2]; got {n_holdout} for n={n}")
    idx = np.unique(np.round(np.linspace(1, n - 2, n_holdout)).astype(int))
    holdout = idx
    train = np.array([i for i in range(n) if i not in set(holdout.tolist())])
    return train, holdout


def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(actual)) ** 2)))


def calibrate_and_validate(csv_path: str, model: ModelSpec,
                           kv_quant: str = "f16", n_holdout: int = 2,
                           budget_gib: float = 21.6,
                           wedge_ctx: int = 65536) -> dict:
    """Full harness: ingest telemetry, fit on TRAIN, validate on HELD-OUT points.

    Runs the calibrated GQA-aware model and the fair naive-constant baseline
    head to head and reports each one's held-out validation RMSE, plus the
    safety consequence at `wedge_ctx` (does the estimator's prediction clear the
    budget?). All numbers are MEASURED from the fit, not asserted.
    """
    contexts, vram_gib = load_telemetry_csv(csv_path)
    n = contexts.size
    train_idx, hold_idx = _holdout_split(n, n_holdout)

    gqa = fit_gqa_aware(contexts[train_idx], vram_gib[train_idx], model, kv_quant)
    naive = fit_naive_constant(contexts[train_idx], vram_gib[train_idx])

    gqa_hold_rmse = _rmse(gqa.predict(contexts[hold_idx]), vram_gib[hold_idx])
    naive_hold_rmse = _rmse(naive.predict(contexts[hold_idx]), vram_gib[hold_idx])
    rmse_improvement_factor = naive_hold_rmse / max(gqa_hold_rmse, _RMSE_FLOOR_GIB)
    rmse_improvement_label = (
        "effectively exact fit (>1000x better)"
        if rmse_improvement_factor > _RMSE_RATIO_SANE_BOUND
        else f"{rmse_improvement_factor:.0f}x worse"
    )

    # safety consequence at the known-wedge context
    gqa_wedge_pred = float(gqa.predict(wedge_ctx))
    naive_wedge_pred = float(naive.predict(wedge_ctx))
    # the true measured VRAM at wedge_ctx if it is in the telemetry
    true_at_wedge = None
    match = np.where(contexts == wedge_ctx)[0]
    if match.size:
        true_at_wedge = float(vram_gib[match[0]])

    return {
        "n_points": n,
        "n_train": int(train_idx.size),
        "n_holdout": int(hold_idx.size),
        "train_contexts": contexts[train_idx].astype(int).tolist(),
        "holdout_contexts": contexts[hold_idx].astype(int).tolist(),
        "gqa_fit": gqa,
        "naive_fit": naive,
        "gqa_holdout_rmse_gib": gqa_hold_rmse,
        "naive_holdout_rmse_gib": naive_hold_rmse,
        "rmse_improvement_factor": rmse_improvement_factor,
        "rmse_improvement_label": rmse_improvement_label,
        "wedge_ctx": wedge_ctx,
        "budget_gib": budget_gib,
        "true_vram_at_wedge_gib": true_at_wedge,
        "gqa_wedge_pred_gib": gqa_wedge_pred,
        "naive_wedge_pred_gib": naive_wedge_pred,
        "gqa_flags_wedge": gqa_wedge_pred > budget_gib,
        "naive_flags_wedge": naive_wedge_pred > budget_gib,
    }


# default fixture path (the committed synthetic sweep). Resolved as real
# package data via importlib.resources so it resolves correctly from a real
# (non-editable) install too, not just the git-checkout layout.
DEFAULT_TELEMETRY_CSV = str(
    resources.files("gpuenvelope").joinpath("data", "rtx3090_qwen3_30b_synthetic.csv")
)
