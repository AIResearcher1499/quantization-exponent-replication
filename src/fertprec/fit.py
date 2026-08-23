"""The frozen K6 fit: `log dqLoss ~ beta log(step)`, with its decision rule."""

from __future__ import annotations

import math
from dataclasses import dataclass

PUBLISHED_BETA = 0.5251
WEAK_BAND = (0.40, 0.65)
MIN_POINTS = 6  # prereg §5: fewer usable checkpoints -> UNUSABLE, not fitted


@dataclass
class Fit:
    bits: int
    eval_set: str
    n: int
    beta: float | None
    se: float | None
    ci: tuple[float, float] | None
    verdict: str

    def contains_published(self) -> bool:
        return self.ci is not None and self.ci[0] <= PUBLISHED_BETA <= self.ci[1]

    def inside_weak_band(self) -> bool:
        return (self.ci is not None
                and WEAK_BAND[0] <= self.ci[0] and self.ci[1] <= WEAK_BAND[1])


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    beta = sxy / sxx
    alpha = my - beta * mx
    resid = [y - (alpha + beta * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se = math.sqrt(s2 / sxx)
    return beta, se


def fit_one(rows: list[dict], bits: int, eval_set: str) -> Fit:
    """Fit one (bits, eval set) cell after applying the pre-registered guards."""
    sel = [r for r in rows if r["bits"] == bits and r["eval_set"] == eval_set]
    usable = [r for r in sel if r.get("dq_loss", 0) > 0 and not r.get("flagged")]
    if len(usable) < MIN_POINTS:
        return Fit(bits, eval_set, len(usable), None, None, None, "UNUSABLE")
    xs = [math.log(r["step"]) for r in usable]
    ys = [math.log(r["dq_loss"]) for r in usable]
    beta, se = ols(xs, ys)
    ci = (beta - 1.96 * se, beta + 1.96 * se)
    return Fit(bits, eval_set, len(usable), beta, se, ci, "FITTED")


def flag_outliers(rows: list[dict]) -> list[dict]:
    """Prereg §5: flag checkpoints whose BF16 perplexity is 3x the median.

    A checkpoint that failed to load looks exactly like an early checkpoint
    unless this is checked explicitly.
    """
    ppls = sorted(r["ppl_bf16"] for r in rows if r.get("ppl_bf16"))
    if not ppls:
        return rows
    med = ppls[len(ppls) // 2]
    for r in rows:
        if r.get("ppl_bf16") and r["ppl_bf16"] > 3 * med:
            r["flagged"] = "ppl_bf16 > 3x median"
    return rows


def verdict(fits: list[Fit]) -> str:
    """Prereg §4. Two of four cells decide it."""
    fitted = [f for f in fits if f.verdict == "FITTED"]
    if sum(f.contains_published() for f in fitted) >= 2:
        return "REPLICATES"
    if sum(f.inside_weak_band() for f in fitted) >= 2:
        return "WEAK"
    excludes_both = sum(
        (not f.contains_published())
        and f.ci is not None
        and (f.ci[1] < WEAK_BAND[0] or f.ci[0] > WEAK_BAND[1])
        for f in fitted
    )
    if excludes_both >= 2:
        return "FAILS"
    return "INCONCLUSIVE"
