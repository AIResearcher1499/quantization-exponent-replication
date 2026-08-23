"""GPU selection, done before torch initialises CUDA.

The quantization backend places the model across every *visible* CUDA device.
On a box with a small auxiliary card that means an out-of-memory on the small
one and a silent fallback to CPU for the Hessian inverse -- after which some
layers of a model were quantized through one code path and some through
another, depending on where the OOM landed. Across a ladder of checkpoints that
varies per run, so the measured degradation would partly encode which layers
OOMed rather than how much training the checkpoint had.

The fix is to make only one large card visible, and to decide *which* without
importing torch: `CUDA_VISIBLE_DEVICES` has no effect once CUDA is initialised,
so the query goes through nvidia-smi.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

MIN_GIB = 20.0


@dataclass
class Gpu:
    index: int
    name: str
    total_gib: float
    free_gib: float

    @property
    def usable(self) -> bool:
        return self.total_gib >= MIN_GIB


def list_gpus() -> list[Gpu]:
    """Enumerate GPUs via nvidia-smi. Returns [] if it is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx, name, total_mib, free_mib = parts
        gpus.append(Gpu(int(idx), name,
                        float(total_mib) / 1024, float(free_mib) / 1024))
    return gpus


def pick(requested: str | int | None = None) -> int | None:
    """Choose one GPU index to expose.

    `requested` may be an explicit index, or "auto"/None to take the emptiest
    card that is large enough. Returns None if nvidia-smi is unavailable, in
    which case the caller should leave the environment alone rather than guess.
    """
    gpus = list_gpus()
    if not gpus:
        return None
    if requested not in (None, "auto", ""):
        idx = int(requested)
        chosen = next((g for g in gpus if g.index == idx), None)
        if chosen is None:
            raise SystemExit(f"gpu {idx} not found; available: "
                             f"{[g.index for g in gpus]}")
        if not chosen.usable:
            raise SystemExit(
                f"gpu {idx} is {chosen.total_gib:.1f} GiB, below the "
                f"{MIN_GIB:.0f} GiB needed. Quantizing there OOMs and falls "
                f"back to CPU, which makes layers within one model "
                f"incomparable.")
        return idx
    usable = [g for g in gpus if g.usable]
    if not usable:
        raise SystemExit(
            f"no GPU with at least {MIN_GIB:.0f} GiB: "
            + ", ".join(f"{g.index}={g.name} {g.total_gib:.1f}GiB" for g in gpus))
    return max(usable, key=lambda g: g.free_gib).index


def restrict_to(index: int) -> None:
    """Make exactly one device visible. Must run before torch touches CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
