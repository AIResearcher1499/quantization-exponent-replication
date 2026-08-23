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
import shutil
import tempfile
import time
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
    # Wall-clock, seconds. Recorded for planning only -- nothing in the
    # pre-registration is scored on time, and these numbers are
    # hardware-dependent so they must never be compared across machines.
    t_snapshot: float = 0.0
    t_bf16: float = 0.0
    t_quantize: float = 0.0
    t_eval_quant: float = 0.0
    prov: dict[str, Any] = field(default_factory=dict)


def _stamp(msg: str, t0: float) -> None:
    """Print with an absolute clock time and elapsed-since-start.

    A long run is unattended; knowing that a phase took 40 minutes matters more
    afterwards than watching it happen.
    """
    el = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')} +{el/60:6.1f}m] {msg}", flush=True)


class AlreadyRunning(SystemExit):
    pass


def acquire_lock(out: pathlib.Path, steps: list[int]) -> pathlib.Path:
    """Refuse to start if another process is writing the same cells.

    Two processes started on the same output file both read it as empty, run
    the same cells, and append duplicate rows -- while sharing one card. The
    duplicates are not obviously wrong afterwards, which is what makes this
    worth preventing rather than detecting.

    Splitting a ladder across two cards stays legal: the lock is per
    (output file, step), so disjoint `--steps` never collide.
    """
    import os

    lock = out.with_suffix(out.suffix + ".lock")
    if lock.exists():
        try:
            held = json.loads(lock.read_text())
        except (json.JSONDecodeError, OSError):
            held = None
        if held:
            alive = True
            try:
                os.kill(held["pid"], 0)
            except (ProcessLookupError, PermissionError, TypeError):
                alive = held.get("pid") is None
            overlap = sorted(set(held.get("steps", [])) & set(steps))
            if alive and overlap:
                raise AlreadyRunning(
                    f"pid {held['pid']} is already running steps {overlap} "
                    f"into {out}.\n"
                    f"Two processes on the same cells duplicate rows and share "
                    f"one card.\n"
                    f"Use disjoint --steps, a different --out, or stop that "
                    f"process.\n"
                    f"If it is dead, remove {lock}")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "steps": sorted(steps)}))
    return lock


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

    t0 = time.time()
    lock = acquire_lock(out, list(steps or checkpoints.STEPS))
    done = load_done(out)
    prov = provenance.collect()
    # Re-read thermal state per checkpoint rather than once per run: a ladder
    # takes many hours and the machine it starts on is not the machine it
    # finishes on, thermally speaking.
    cells = plan(steps, bits, eval_sets)
    todo = [c for c in cells if c.key() not in done]
    _stamp(f"{len(cells)} cells, {len(cells) - len(todo)} cached, "
           f"{len(todo)} to run", t0)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    windows = {}
    for e in eval_sets:
        if any(c.eval_set == e for c in todo):
            windows[e] = ppl.make_windows(ppl.load_eval_text(e), tok,
                                          max_windows=max_windows)
            _stamp(f"eval {e}: {windows[e].size(0)} windows", t0)

    calib = None
    for step in sorted({c.step for c in todo}):
        rev = checkpoints.revision(step)
        _stamp(f"=== step {step} ({rev})", t0)

        # One local snapshot per checkpoint, shared by the BF16 and quantized
        # loads, so both are provably the same weights.
        t = time.time()
        local = quantize.snapshot(model_id, rev)
        t_snap = time.time() - t
        _stamp(f"  snapshot ready ({t_snap / 60:.1f}m)", t0)

        t = time.time()
        base = AutoModelForCausalLM.from_pretrained(
            local, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True)
        bf16 = {e: ppl.perplexity(base, windows[e], device=device)
                for e in windows}
        t_bf16 = time.time() - t
        for e, v in bf16.items():
            _stamp(f"  bf16 {e}: ppl={v['ppl']:.4f}", t0)
        _stamp(f"  bf16 total ({t_bf16 / 60:.1f}m)", t0)
        del base
        gc.collect()
        torch.cuda.empty_cache()

        for b in sorted({c.bits for c in todo if c.step == step}, reverse=True):
            if calib is None:
                t = time.time()
                calib = quantize.build_calibration(tok)
                _stamp(f"  calibration: {len(calib)} sequences "
                       f"({(time.time() - t) / 60:.1f}m)", t0)
            t = time.time()
            # Scratch beside the results file, not in /tmp: a quantized 7B is
            # 3-5 GB and /tmp is often tmpfs, i.e. RAM.
            scratch = out.parent / "_quant"
            scratch.mkdir(parents=True, exist_ok=True)
            qdir = tempfile.mkdtemp(prefix=f"q{b}-{step}-", dir=str(scratch))
            quantize.quantize_gptq(local, b, calib, out_dir=qdir)
            t_quant = time.time() - t
            _stamp(f"  int{b} quantized ({t_quant / 60:.1f}m)", t0)
            gc.collect()
            torch.cuda.empty_cache()
            # Reload from disk: quantizing in-process leaves part of the model
            # on the meta device, which cannot be moved without destroying it.
            qmodel = quantize.load_quantized(qdir, device)
            quantize.consolidate(qmodel.model, device)
            t_eval = time.time()
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
                          key=cell.key(),
                          prov={**prov, "thermal": provenance.thermal_state()},
                          t_snapshot=t_snap, t_bf16=t_bf16,
                          t_quantize=t_quant,
                          t_eval_quant=time.time() - t_eval)
                append_row(out, asdict(row))
                _stamp(f"  int{b} {e}: ppl={q['ppl']:.4f} "
                       f"dqLoss={row.dq_loss:+.5f}", t0)
            del qmodel
            gc.collect()
            torch.cuda.empty_cache()
            shutil.rmtree(qdir, ignore_errors=True)

    lock.unlink(missing_ok=True)


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
