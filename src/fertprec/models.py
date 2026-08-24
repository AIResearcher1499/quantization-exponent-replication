"""The G0 model list, in two layers that must not be pooled.

Layer 1 has a disclosed token count, so it can carry the quantitative axis.
Layer 2 does not; it exists only to answer "does the phenomenon still appear on
current models", as sign and ordering. See the (unpublished) G0 pre-registration §8(f).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    key: str
    hf_id: str
    params: float
    tokens: float | None      # disclosed pretraining tokens; None = layer 2
    family: str

    @property
    def layer(self) -> int:
        return 1 if self.tokens else 2

    @property
    def tokens_per_param(self) -> float | None:
        return self.tokens / self.params if self.tokens else None


LAYER1 = [
    # Qwen3 dense: five parameter counts at a COMMON D of 36T and one
    # tokenizer -- a clean N axis. 32B excluded: 64 GB at BF16.
    Model("qwen3-0.6b", "Qwen/Qwen3-0.6B-Base", 0.6e9, 36e12, "qwen3"),
    Model("qwen3-1.7b", "Qwen/Qwen3-1.7B-Base", 1.7e9, 36e12, "qwen3"),
    Model("qwen3-4b",   "Qwen/Qwen3-4B-Base",   4.0e9, 36e12, "qwen3"),
    Model("qwen3-8b",   "Qwen/Qwen3-8B-Base",   8.2e9, 36e12, "qwen3"),
    Model("qwen3-14b",  "Qwen/Qwen3-14B-Base", 14.8e9, 36e12, "qwen3"),
    # Gemma 3: D and N moving together within one recipe -- the beta axis.
    Model("gemma3-1b",  "google/gemma-3-1b-pt",  1.0e9,  2e12, "gemma3"),
    Model("gemma3-4b",  "google/gemma-3-4b-pt",  4.3e9,  4e12, "gemma3"),
    Model("gemma3-12b", "google/gemma-3-12b-pt",12.2e9, 12e12, "gemma3"),
    # Bridges to the competing literature and to our own earlier work.
    Model("llama31-8b", "meta-llama/Llama-3.1-8B", 8.0e9, 15e12, "llama3"),
    Model("qwen25-7b",  "Qwen/Qwen2.5-7B",         7.6e9, 18e12, "qwen25"),
]

LAYER2 = [
    Model("qwen38-27b",   "Qwen/Qwen3.8-27B",              27.8e9, None, "qwen38"),
    Model("gemma4-12b",   "google/gemma-4-12b-pt",         12.0e9, None, "gemma4"),
    Model("ministral3-8b", "mistralai/Ministral-3-8B-Base", 8.0e9, None, "ministral3"),
]

ALL = {m.key: m for m in LAYER1 + LAYER2}


def by_layer(layer: int) -> list[Model]:
    return [m for m in ALL.values() if m.layer == layer]
