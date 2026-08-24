"""Corpus loading for F0.

Two corpora with different jobs:

- **Wikipedia** (`data/raw/wiki/<lang>.txt`) trains tokenizers. It only has to
  be large and natural; it is not parallel and is never measured on.
- **FLORES-200 devtest** (`data/raw/flores200_dataset/devtest/<code>.devtest`)
  is what fertility is measured on. It is parallel by construction: every
  language expresses the same 1012 sentences, so a difference in token count
  between two languages is a difference in tokenization, not in content.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
WIKI = ROOT / "data" / "raw" / "wiki"
FLORES = ROOT / "data" / "raw" / "flores200_dataset" / "devtest"

# Frozen 11-language set (the (unpublished) G0 pre-registration §2), with FLORES-200 codes.
LANGS: dict[str, str] = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ja": "jpn_Jpan",
    "de": "deu_Latn",
    "ru": "rus_Cyrl",
    "sw": "swh_Latn",
    "th": "tha_Thai",
    "bn": "ben_Beng",
    "te": "tel_Telu",
}


def wiki_text(lang: str, max_bytes: int) -> str:
    """Read up to `max_bytes` of UTF-8 from one language's Wikipedia dump.

    Truncation is done on the decoded string, not on raw bytes, so a cut never
    lands inside a multi-byte character. Non-Latin scripts spend 3 bytes per
    character where English spends 1, so a byte budget is the only cut that
    means the same thing across languages.
    """
    raw = (WIKI / f"{lang}.txt").read_bytes()
    if len(raw) <= max_bytes:
        return raw.decode("utf-8", errors="ignore")
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def flores_sentences(lang: str) -> list[str]:
    path = FLORES / f"{LANGS[lang]}.devtest"
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def available() -> dict[str, int]:
    """Bytes of Wikipedia available per language, for budget sanity checks."""
    return {lang: (WIKI / f"{lang}.txt").stat().st_size for lang in LANGS}
