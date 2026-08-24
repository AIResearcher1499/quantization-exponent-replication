"""G0 analysis tests. Private: these exercise the design of the main paper.

The decision rule has to distinguish the two hypotheses it was written to
separate. A rule that returns GO on any input is worse than no rule at all.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


# --- G0 analysis -------------------------------------------------------------
# The decision rule has to distinguish the two hypotheses it was written to
# separate. A rule that says GO on any input is worse than no rule.

import math as _m
import random as _r

from fertprec import covariates as _cov
from fertprec.fit_g0 import fit_family, verdict

_LANG_TOKENS = [('en',29520),('de',34518),('ru',60071),('es',36064),
                ('fr',37668),('zh',43089),('bn',93702),('th',76978),
                ('te',97415),('ja',47281),('sw',34027)]


def _synth(driver, seed=0, noise=0.15):
    """Rows whose damage is driven either by fertility or by typology."""
    typo = _cov.load()["typo_distance"]["value"]
    rng = _r.Random(seed)
    rows = []
    for fam in ("A", "B", "C"):
        for lang, tokens in _LANG_TOKENS:
            signal = (0.6 * _m.log(tokens) if driver == "fertility"
                      else 3.0 * typo[lang])
            rows.append({"family": fam, "bits": 4, "lang": lang,
                         "flores_tokens": tokens, "layer": 1,
                         "dq_loss": _m.exp(-4 + signal + rng.gauss(0, noise))})
    return rows


def test_go_when_fertility_drives_the_damage():
    fits = [fit_family(_synth("fertility"), f) for f in ("A", "B", "C")]
    assert verdict(fits) == "GO"
    assert all(f.fertility_positive for f in fits)


def test_no_go_when_typology_drives_the_damage():
    """LCD's explanation, simulated. The rule must not return GO here."""
    fits = [fit_family(_synth("typology"), f) for f in ("A", "B", "C")]
    assert verdict(fits) != "GO"


def test_sigma_is_reported_because_it_sizes_g1():
    fits = [fit_family(_synth("fertility", noise=0.4), f) for f in ("A","B","C")]
    assert all(f.sigma > 0 for f in fits)


def test_layer2_rows_never_enter_the_fit(tmp_path):
    """A model with no disclosed token count cannot carry the quantitative
    axis, and must not rescue a layer-1 result."""
    import json as _json
    from fertprec import fit_g0
    rows = _synth("fertility")
    for r in rows[:11]:
        r = dict(r); r["layer"] = 2; r["family"] = "LAYER2"; rows.append(r)
    p = tmp_path / "g0.jsonl"
    p.write_text("\n".join(_json.dumps(r) for r in rows))
    out = fit_g0.analyse(p)
    assert "LAYER2" not in out["sigma_by_family"]
