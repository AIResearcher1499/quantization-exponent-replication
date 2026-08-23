"""K6 runner: does the published exponent replicate on an OLMo-3 ladder?

Protocol is frozen in docs/prereg-k6.md. This module only executes it.

Two failure modes from earlier projects are designed against here:

- **Overwrite instead of merge.** Results are appended to a JSONL file and the
  file is re-read on start. A run that crashes at checkpoint six does not lose
  checkpoints one to five, and a second invocation does not destroy the first.
- **Incomplete resume keys.** The key includes every parameter that changes the
  number: step, bits, eval set, sequence length, calibration size and seed,
  group size. A key missing one of these silently returns a stale row from a
  different configuration and looks like a successful run.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any

from . import checkpoints, fit, ppl, provenance, quantize

BITS = (4, 3)
EVAL_SETS = ("wikitext2", "c4val")


@dataclass
class Cell:
    step: int
    bits: int
    eval_set: str
    seq_len: int = ppl.SEQ_LEN
    calib_n: int = quantize.CALIB_N
    calib_seed: int = quantize.CALIB_SEED
    group_size: int = 128

    def key(self) -> str:
        d = asdict(self)
        return "|".join(f"{k}={d[k]}" for k in sorted(d))


@dataclass
class Row:
    step: int
    bits: int
    eval_set: str
    ppl_bf16: float
    ppl_quant: float
    dq_loss: float
    nll_bf16: float
    nll_quant: float
    n_windows: int
    n_tokens: int
    key: str
    prov: dict[str, Any] = field(default_factory=dict)


def load_done(out: pathlib.Path) -> dict[str, dict]:
    if not out.exists():
        return {}
    done = {}
    for line in out.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        done[r["key"]] = r
    return done


def append_row(out: pathlib.Path, row: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def plan(steps=None, bits=BITS, eval_sets=EVAL_SETS) -> list[Cell]:
    steps = steps or checkpoints.STEPS
    return [Cell(step=s, bits=b, eval_set=e)
            for s in steps for b in bits for e in eval_sets]


def run(out: pathlib.Path, steps=None, bits=BITS, eval_sets=EVAL_SETS,
        max_windows: int | None = None, device: str = "cuda",
        model_id: str = checkpoints.MODEL_ID) -> None:
    """Execute the ladder, skipping cells already present in `out`.

    Ordered checkpoint-major so that each checkpoint is downloaded and its BF16
    baseline computed once, then reused for both bit widths.
    """
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    done = load_done(out)
    prov = provenance.collect()
    cells = plan(steps, bits, eval_sets)
    todo = [c for c in cells if c.key() not in done]
    print(f"{len(cells)} cells, {len(cells) - len(todo)} cached, {len(todo)} to run",
          flush=True)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    windows = {}
    for e in eval_sets:
        if any(c.eval_set == e for c in todo):
            windows[e] = ppl.make_windows(ppl.load_eval_text(e), tok,
                                          max_windows=max_windows)
            print(f"eval {e}: {windows[e].size(0)} windows", flush=True)

    calib = None
    for step in sorted({c.step for c in todo}):
        rev = checkpoints.revision(step)
        print(f"\n=== step {step} ({rev})", flush=True)

        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=rev, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True)
        bf16 = {e: ppl.perplexity(base, windows[e], device=device)
                for e in windows}
        for e, v in bf16.items():
            print(f"  bf16 {e}: ppl={v['ppl']:.4f}", flush=True)
        del base
        gc.collect()
        torch.cuda.empty_cache()

        for b in sorted({c.bits for c in todo if c.step == step}, reverse=True):
            if calib is None:
                calib = quantize.build_calibration(tok)
                print(f"  calibration: {len(calib)} sequences", flush=True)
            qmodel = quantize.quantize_gptq(model_id, rev, b, calib)
            for e in windows:
                cell = Cell(step=step, bits=b, eval_set=e)
                if cell.key() in done:
                    continue
                q = ppl.perplexity(qmodel.model, windows[e], device=device)
                row = Row(step=step, bits=b, eval_set=e,
                          ppl_bf16=bf16[e]["ppl"], ppl_quant=q["ppl"],
                          dq_loss=q["nll"] - bf16[e]["nll"],
                          nll_bf16=bf16[e]["nll"], nll_quant=q["nll"],
                          n_windows=q["n_windows"], n_tokens=q["n_tokens"],
                          key=cell.key(), prov=prov)
                append_row(out, asdict(row))
                print(f"  int{b} {e}: ppl={q['ppl']:.4f} dqLoss={row.dq_loss:+.5f}",
                      flush=True)
            del qmodel
            gc.collect()
            torch.cuda.empty_cache()


def analyse(out: pathlib.Path) -> dict:
    """Apply the frozen guards and decision rule to whatever is in `out`."""
    rows = list(load_done(out).values())
    if not rows:
        return {"verdict": "NO DATA", "fits": []}
    rows = fit.flag_outliers(rows)
    fits = [fit.fit_one(rows, b, e) for b in BITS for e in EVAL_SETS]
    return {"verdict": fit.verdict(fits), "fits": [asdict(f) for f in fits],
            "n_rows": len(rows),
            "excluded_nonpositive": sum(1 for r in rows if r.get("dq_loss", 0) <= 0),
            "flagged": [r["key"] for r in rows if r.get("flagged")]}
