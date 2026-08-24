"""Covariates for the G0 regression, computed rather than typed in.

The pre-registration names two competing explanations for per-language
quantization damage, and both need a number:

- `typo_distance` -- the explanation `2608.11786` offers ("typological distance
  from English"). Computed from URIEL via lang2vec as the mean cosine distance
  to English over the `syntax_knn` and `fam` feature vectors.
- `data_share_proxy` -- how much of the pretraining corpus a language plausibly
  occupies. No model discloses its mixture, so this is a proxy and is treated
  as one: it comes from published Common Crawl language statistics, is weak by
  construction, and its weakness is the reason G1 exists.

Nothing here is hand-entered from memory. Distances are derived from URIEL at
build time and cached to `data/covariates.json`; the corpus shares carry their
source in the file.
"""

from __future__ import annotations

import json
import math
import pathlib

from .corpus import LANGS

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "covariates.json"

# ISO 639-3 codes as URIEL knows them.
ISO3 = {"en": "eng", "zh": "cmn", "es": "spa", "fr": "fra", "ja": "jpn",
        "de": "deu", "ru": "rus", "sw": "swh", "th": "tha", "bn": "ben",
        "te": "tel"}

# URIEL calls the genetic/phylogenetic vector "fam" (language family
# membership), not "genetic".
FEATURE_SETS = ("syntax_knn", "fam")


def _cosine_distance(a, b) -> float:
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None and x != "--" and y != "--"]
    if not pairs:
        raise ValueError("no shared defined features")
    dot = sum(float(x) * float(y) for x, y in pairs)
    na = math.sqrt(sum(float(x) ** 2 for x, _ in pairs))
    nb = math.sqrt(sum(float(y) ** 2 for _, y in pairs))
    if na == 0 or nb == 0:
        raise ValueError("zero feature vector")
    return 1.0 - dot / (na * nb)


def compute_typological_distances() -> dict[str, float]:
    """Mean cosine distance to English over the URIEL feature sets."""
    import lang2vec.lang2vec as l2v

    codes = [ISO3[l] for l in LANGS]
    out: dict[str, list[float]] = {l: [] for l in LANGS}
    for fs in FEATURE_SETS:
        feats = l2v.get_features(codes, fs)
        eng = feats[ISO3["en"]]
        for lang in LANGS:
            out[lang].append(_cosine_distance(eng, feats[ISO3[lang]]))
    return {l: sum(v) / len(v) for l, v in out.items()}


def load(path: pathlib.Path | None = None) -> dict:
    """Read the cached covariates, or say exactly how to regenerate them."""
    path = path or CACHE
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Regenerate with:\n"
            f"  uv run python scripts/build_covariates.py\n"
            "It needs lang2vec (URIEL) and network access the first time.")
    return json.loads(path.read_text())
