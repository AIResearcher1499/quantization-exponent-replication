"""The frozen G0 analysis: does fertility out-predict typological distance?

One fit per model family (the (unpublished) G0 pre-registration §4, and §9(b) for why that is now
load-bearing rather than merely tidy). Coefficients are standardized so their
magnitudes are comparable, and are never pooled across families.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass

from . import covariates


@dataclass
class FamilyFit:
    family: str
    n: int
    beta_fertility: float
    ci_fertility: tuple[float, float]
    beta_typo: float
    ci_typo: tuple[float, float]
    sigma: float          # residual SD -- a deliverable, see prereg §9(d)
    covariates_used: list[str]

    @property
    def fertility_positive(self) -> bool:
        return self.ci_fertility[0] > 0

    @property
    def fertility_beats_typo(self) -> bool:
        return abs(self.beta_fertility) > abs(self.beta_typo)


def _standardize(xs: list[float]) -> list[float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return [(x - m) / sd if sd else 0.0 for x in xs]


def _ols_multi(X: list[list[float]], y: list[float]):
    """Least squares with an intercept, via normal equations.

    Small and dense (11 languages, <=3 predictors), so the numerics are not
    delicate and a dependency is not worth it.
    """
    n, k = len(y), len(X[0])
    A = [[1.0] + row for row in X]
    XtX = [[sum(A[i][a] * A[i][b] for i in range(n)) for b in range(k + 1)]
           for a in range(k + 1)]
    Xty = [sum(A[i][a] * y[i] for i in range(n)) for a in range(k + 1)]
    # Gauss-Jordan
    M = [XtX[i] + [1.0 if i == j else 0.0 for j in range(k + 1)]
         for i in range(k + 1)]
    for col in range(k + 1):
        piv = max(range(col, k + 1), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("singular design -- predictors are collinear")
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [v / d for v in M[col]]
        for r in range(k + 1):
            if r != col and M[r][col]:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    inv = [row[k + 1:] for row in M]
    beta = [sum(inv[a][b] * Xty[b] for b in range(k + 1)) for a in range(k + 1)]
    pred = [sum(A[i][a] * beta[a] for a in range(k + 1)) for i in range(n)]
    resid = [y[i] - pred[i] for i in range(n)]
    dof = max(n - (k + 1), 1)
    s2 = sum(r * r for r in resid) / dof
    se = [math.sqrt(max(s2 * inv[a][a], 0.0)) for a in range(k + 1)]
    return beta, se, math.sqrt(s2)


def fit_family(rows: list[dict], family: str, bits: int = 4,
               cov: dict | None = None) -> FamilyFit:
    """Fit one family at one bit width. INT4 is primary (prereg §8a)."""
    cov = cov or covariates.load()
    typo = cov["typo_distance"]["value"]
    share = (cov.get("data_share_proxy") or {}).get("value")

    sel = [r for r in rows if r["family"] == family and r["bits"] == bits
           and r["dq_loss"] > 0]
    if len(sel) < 6:
        raise ValueError(f"{family}: only {len(sel)} usable rows")

    # Damage relative to that model's own BF16 baseline, never absolute
    # perplexity -- absolute thresholds mislabelled models before.
    y = [math.log(r["dq_loss"]) for r in sel]
    fert = _standardize([math.log(r["flores_tokens"]) for r in sel])
    tp = _standardize([typo[r["lang"]] for r in sel])
    cols, names = [fert, tp], ["log_fertility", "typo_distance"]
    if share:
        cols.append(_standardize([math.log(share[r["lang"]]) for r in sel]))
        names.append("log_data_share")

    X = [[c[i] for c in cols] for i in range(len(sel))]
    beta, se, sigma = _ols_multi(X, y)
    return FamilyFit(
        family=family, n=len(sel),
        beta_fertility=beta[1],
        ci_fertility=(beta[1] - 1.96 * se[1], beta[1] + 1.96 * se[1]),
        beta_typo=beta[2],
        ci_typo=(beta[2] - 1.96 * se[2], beta[2] + 1.96 * se[2]),
        sigma=sigma, covariates_used=names)


def verdict(fits: list[FamilyFit]) -> str:
    """Prereg §5, evaluated on layer 1 and INT4 only."""
    go = sum(f.fertility_positive and f.fertility_beats_typo for f in fits)
    if go >= 2:
        return "GO"
    dominated = sum(abs(f.beta_typo) > abs(f.beta_fertility) for f in fits)
    if dominated >= 2 or all(f.ci_fertility[0] <= 0 <= f.ci_fertility[1] for f in fits):
        return "NO-GO"
    return "UNDECIDED"


def analyse(path: pathlib.Path, bits: int = 4) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("layer") == 1]      # never pool layer 2
    fits, errors = [], {}
    for fam in sorted({r["family"] for r in rows}):
        try:
            fits.append(fit_family(rows, fam, bits))
        except ValueError as exc:
            errors[fam] = str(exc)
    return {
        "bits": bits,
        "verdict": verdict(fits) if fits else "NO DATA",
        "fits": [f.__dict__ for f in fits],
        "sigma_by_family": {f.family: f.sigma for f in fits},
        "not_fitted": errors,
    }
