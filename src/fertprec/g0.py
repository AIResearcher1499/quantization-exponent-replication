"""G0: does fertility out-predict typological distance on existing models?

Observational. It cannot establish causality -- fertility and pretraining data
share are entangled in every public checkpoint, which is what G1 exists to
break. G0 kills the idea cheaply if the correlational precondition already
fails, and it measures `sigma`, which sets how many arms G1 needs.

Protocol: the (unpublished) G0 pre-registration (frozen 2026-08-23, amended 2026-08-24 before any
row existed).
"""

from __future__ import annotations

import gc
import json
import pathlib
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import corpus, models, ppl, provenance, quantize
from .k6 import _stamp, acquire_lock, append_row, load_done

BITS = (4, 3)          # INT4 is primary (prereg §8a); INT3 collected, not scored


@dataclass
class Cell:
    model: str
    lang: str
    bits: int
    seq_len: int = ppl.SEQ_LEN
    calib_n: int = quantize.CALIB_N
    calib_seed: int = quantize.CALIB_SEED
    group_size: int = 128

    def key(self) -> str:
        d = asdict(self)
        return "|".join(f"{k}={d[k]}" for k in sorted(d))


@dataclass
class Row:
    model: str
    family: str
    layer: int
    params: float
    tokens: float | None
    lang: str
    bits: int
    # Absolute tokens for the identical FLORES content under THIS model's
    # tokenizer. Never a premium -- see the (unpublished) G0 pre-registration §9(a).
    flores_tokens: int
    ppl_bf16: float
    ppl_quant: float
    dq_loss: float
    nll_bf16: float
    nll_quant: float
    n_windows: int
    key: str
    t_quantize: float = 0.0
    prov: dict[str, Any] = field(default_factory=dict)


def plan(model_keys=None, langs=None, bits=BITS) -> list[Cell]:
    model_keys = model_keys or [m.key for m in models.LAYER1]
    langs = langs or list(corpus.LANGS)
    return [Cell(model=mk, lang=l, bits=b)
            for mk in model_keys for l in langs for b in bits]


def run(out: pathlib.Path, model_keys=None, langs=None, bits=BITS,
        device: str = "cuda", max_windows: int | None = None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_keys = model_keys or [m.key for m in models.LAYER1]
    langs = langs or list(corpus.LANGS)

    t0 = time.time()
    # Before anything expensive: a missing corpus should not surface after a
    # 30 GB model download.
    corpus.ensure_flores()
    # Lock on model keys, not a hash of them: the refusal message has to name
    # what is already running for it to be actionable.
    lock = acquire_lock(out, list(model_keys))
    stale = out.parent / "_quant"
    if stale.exists():
        shutil.rmtree(stale, ignore_errors=True)

    done = load_done(out)
    prov = provenance.collect()
    cells = plan(model_keys, langs, bits)
    todo = [c for c in cells if c.key() not in done]
    _stamp(f"{len(cells)} cells, {len(cells)-len(todo)} cached, {len(todo)} to run", t0)

    for mk in model_keys:
        pend = [c for c in todo if c.model == mk]
        if not pend:
            continue
        m = models.ALL[mk]
        _stamp(f"=== {m.key} ({m.hf_id}) layer {m.layer}", t0)

        tok = AutoTokenizer.from_pretrained(m.hf_id, trust_remote_code=True)
        # Windows are built per model because the tokenizer differs; the same
        # FLORES sentences therefore become a different number of tokens, which
        # is the independent variable.
        windows, ntok = {}, {}
        for lang in sorted({c.lang for c in pend}):
            text = "\n".join(corpus.flores_sentences(lang))
            windows[lang] = ppl.make_windows(text, tok, max_windows=max_windows)
            ntok[lang] = int(tok(text, return_tensors="pt").input_ids.numel())
        _stamp("  tokens on identical FLORES content: "
               + " ".join(f"{l}={ntok[l]}" for l in sorted(ntok)), t0)

        base = AutoModelForCausalLM.from_pretrained(
            m.hf_id, torch_dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True)
        bf16 = {l: ppl.perplexity(base, windows[l], device=device) for l in windows}
        del base
        gc.collect(); torch.cuda.empty_cache()

        calib = quantize.build_calibration(tok)
        for b in sorted({c.bits for c in pend}, reverse=True):
            t = time.time()
            scratch = out.parent / "_quant"
            scratch.mkdir(parents=True, exist_ok=True)
            qdir = tempfile.mkdtemp(prefix=f"{mk}-q{b}-", dir=str(scratch))
            quantize.quantize_gptq(m.hf_id, b, calib, out_dir=qdir)
            t_quant = time.time() - t
            gc.collect(); torch.cuda.empty_cache()
            qmodel = quantize.load_quantized(qdir, device)
            quantize.consolidate(qmodel.model, device)
            _stamp(f"  int{b} quantized ({t_quant/60:.1f}m)", t0)

            for lang in windows:
                cell = Cell(model=mk, lang=lang, bits=b)
                if cell.key() in done:
                    continue
                q = ppl.perplexity(qmodel.model, windows[lang], device=device)
                row = Row(model=mk, family=m.family, layer=m.layer,
                          params=m.params, tokens=m.tokens, lang=lang, bits=b,
                          flores_tokens=ntok[lang],
                          ppl_bf16=bf16[lang]["ppl"], ppl_quant=q["ppl"],
                          dq_loss=q["nll"] - bf16[lang]["nll"],
                          nll_bf16=bf16[lang]["nll"], nll_quant=q["nll"],
                          n_windows=q["n_windows"], key=cell.key(),
                          t_quantize=t_quant,
                          prov={**prov, "thermal": provenance.thermal_state()})
                append_row(out, asdict(row))
                _stamp(f"  int{b} {lang:<3} tok={ntok[lang]:>6} "
                       f"dqLoss={row.dq_loss:+.5f}", t0)
            del qmodel
            gc.collect(); torch.cuda.empty_cache()
            shutil.rmtree(qdir, ignore_errors=True)

    lock.unlink(missing_ok=True)
