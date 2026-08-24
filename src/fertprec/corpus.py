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


FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def ensure_flores() -> pathlib.Path:
    """Download and unpack FLORES-200 if it is not already present.

    Public, ungated, ~25 MB. Fetching it automatically is worth a small amount
    of magic: the alternative is a run that dies on the first language after
    the model has already been downloaded, which is what happened once.
    """
    if FLORES.is_dir() and any(FLORES.glob("*.devtest")):
        return FLORES

    import tarfile
    import urllib.request

    dest = ROOT / "data" / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    tarball = dest / "flores200.tar.gz"
    if not tarball.exists():
        print(f"downloading FLORES-200 ({FLORES_URL}) ...", flush=True)
        urllib.request.urlretrieve(FLORES_URL, tarball)
    print(f"unpacking into {dest} ...", flush=True)
    with tarfile.open(tarball) as tf:
        tf.extractall(dest)
    if not (FLORES.is_dir() and any(FLORES.glob("*.devtest"))):
        raise RuntimeError(f"FLORES unpacked but {FLORES} is not where expected")
    print(f"FLORES-200 ready: {len(list(FLORES.glob('*.devtest')))} languages",
          flush=True)
    return FLORES


def flores_sentences(lang: str) -> list[str]:
    path = FLORES / f"{LANGS[lang]}.devtest"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `fertprec doctor`, or call "
            "corpus.ensure_flores() to fetch it.")
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def available() -> dict[str, int]:
    """Bytes of Wikipedia available per language, for budget sanity checks."""
    return {lang: (WIKI / f"{lang}.txt").stat().st_size for lang in LANGS}
