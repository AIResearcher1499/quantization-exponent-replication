"""K6 logic tests. No GPU, no network -- these run anywhere.

They cover the parts that decide what the paper says (the frozen decision rule)
and the two mechanisms that have silently corrupted results in earlier projects
(resume keys and overwrite-instead-of-merge).
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fertprec import fit, k6  # noqa: E402


def synth(beta: float, bits: int, eval_set: str, noise: float = 0.0,
          steps=(70000, 108000, 165000, 254000, 390000, 599000, 920000, 1413814)):
    """dqLoss generated to follow log dq = beta * log step + c exactly."""
    rows = []
    for i, s in enumerate(steps):
        dq = math.exp(beta * math.log(s) - 8.0 + (noise if i % 2 else -noise))
        rows.append({"step": s, "bits": bits, "eval_set": eval_set,
                     "dq_loss": dq, "ppl_bf16": 12.0})
    return rows


def test_ols_recovers_a_known_slope():
    rows = synth(fit.PUBLISHED_BETA, 4, "wikitext2")
    f = fit.fit_one(rows, 4, "wikitext2")
    assert f.verdict == "FITTED"
    assert abs(f.beta - fit.PUBLISHED_BETA) < 1e-9


def test_replicates_needs_two_cells():
    rows = synth(0.5251, 4, "wikitext2", noise=0.02) + synth(0.5251, 3, "c4val", noise=0.02)
    fits = [fit.fit_one(rows, b, e) for b in (4, 3) for e in ("wikitext2", "c4val")]
    assert fit.verdict(fits) == "REPLICATES"


def test_fails_when_slope_is_far_off():
    rows = synth(0.15, 4, "wikitext2") + synth(0.15, 3, "c4val")
    fits = [fit.fit_one(rows, b, e) for b in (4, 3) for e in ("wikitext2", "c4val")]
    assert fit.verdict(fits) == "FAILS"


def test_weak_band_between_the_two():
    rows = synth(0.45, 4, "wikitext2", noise=0.001) + synth(0.45, 3, "c4val", noise=0.001)
    fits = [fit.fit_one(rows, b, e) for b in (4, 3) for e in ("wikitext2", "c4val")]
    assert fit.verdict(fits) == "WEAK"


def test_too_few_points_is_unusable_not_fitted():
    """Prereg §5: fit on fewer than 6 usable checkpoints is refused."""
    rows = synth(0.5251, 4, "wikitext2")[:5]
    assert fit.fit_one(rows, 4, "wikitext2").verdict == "UNUSABLE"


def test_nonpositive_dqloss_is_excluded():
    rows = synth(0.5251, 4, "wikitext2")
    rows[0]["dq_loss"] = -0.01
    rows[1]["dq_loss"] = 0.0
    f = fit.fit_one(rows, 4, "wikitext2")
    assert f.n == 6


def test_broken_checkpoint_is_flagged_by_bf16_perplexity():
    rows = synth(0.5251, 4, "wikitext2")
    rows[3]["ppl_bf16"] = 500.0
    fit.flag_outliers(rows)
    assert rows[3].get("flagged")
    assert fit.fit_one(rows, 4, "wikitext2").n == 7


def test_resume_key_covers_every_parameter_that_moves_the_number():
    """A key missing a parameter returns a stale row from another config."""
    base = k6.Cell(step=70000, bits=4, eval_set="wikitext2")
    for field_, value in (("step", 108000), ("bits", 3), ("eval_set", "c4val"),
                          ("seq_len", 1024), ("calib_n", 64), ("calib_seed", 1),
                          ("group_size", 64)):
        other = k6.Cell(step=70000, bits=4, eval_set="wikitext2")
        setattr(other, field_, value)
        assert other.key() != base.key(), f"key ignores {field_}"


def test_results_file_merges_and_never_overwrites(tmp_path):
    out = tmp_path / "k6.jsonl"
    k6.append_row(out, {"key": "a", "step": 1})
    k6.append_row(out, {"key": "b", "step": 2})
    done = k6.load_done(out)
    assert set(done) == {"a", "b"}
    # a second process appending must not lose the first process's rows
    k6.append_row(out, {"key": "c", "step": 3})
    assert set(k6.load_done(out)) == {"a", "b", "c"}


def test_plan_is_the_frozen_ladder():
    cells = k6.plan()
    assert len(cells) == 8 * 2 * 2
    assert sorted({c.step for c in cells}) == [70000, 108000, 165000, 254000,
                                               390000, 599000, 920000, 1413814]
