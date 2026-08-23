"""The frozen K6 checkpoint ladder.

Steps are hard-coded rather than recomputed. They were selected once, log-spaced
over stage 1, and written into docs/prereg-k6.md; recomputing them from the Hub
would let the ladder drift silently if Ai2 adds or removes branches.
"""

from __future__ import annotations

MODEL_ID = "allenai/Olmo-3-1025-7B"

# Frozen 2026-08-23. See docs/prereg-k6.md §2.
STEPS: tuple[int, ...] = (70000, 108000, 165000, 254000, 390000, 599000,
                          920000, 1413814)


def revision(step: int) -> str:
    return f"stage1-step{step}"


def ladder() -> list[tuple[int, str]]:
    return [(s, revision(s)) for s in STEPS]


def verify_available(model_id: str = MODEL_ID) -> dict[int, bool]:
    """Check every frozen step still exists as a branch on the Hub.

    Run before a long job: a missing branch discovered eight hours in is an
    avoidable loss.
    """
    from huggingface_hub import HfApi

    refs = HfApi().list_repo_refs(model_id)
    have = {b.name for b in refs.branches}
    return {s: revision(s) in have for s in STEPS}
