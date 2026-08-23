"""Run provenance.

Every result row carries where it was produced and with what. Merged files from
different machines are otherwise indistinguishable, and a kernel or library
change between the BF16 and quantized halves of a comparison is
indistinguishable from the effect under test.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Any


def _version(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def gpu_info() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu": None, "n_gpu": 0}
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return {"gpu": names[0], "n_gpu": len(names), "gpu_all": names,
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))}
    except Exception:
        return {"gpu": None, "n_gpu": 0}


def collect() -> dict[str, Any]:
    info: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "commit": _git_commit(),
    }
    info.update(gpu_info())
    for mod in ("torch", "transformers", "datasets", "gptqmodel", "accelerate"):
        info[f"v_{mod}"] = _version(mod)
    return info
