# K6 runbook — OLMo-3 ladder on the 2x A6000 box

Protocol: `docs/prereg-k6.md` (frozen — do not edit once `data/k6.jsonl`
exists). This file is how to run it.

## What it does

Eight `stage1` checkpoints of `allenai/Olmo-3-1025-7B`, each evaluated at BF16
and after GPTQ at INT4 and INT3, on two English evaluation sets. Fits
`log dqLoss ~ beta log(step)` and applies the frozen decision rule.

K6 only asks whether `beta = 0.5251` from
`2411.17691` reproduces outside the setting it was fitted in.

## Setup

```bash
uv venv -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e ".[gpu,stats]"
```

7B at BF16 is ~14 GB, so one A6000 is enough; GPTQ needs headroom above that
and still fits in 48 GB.

## Before a long run

```bash
uv run fertprec doctor             # imports resolve? GPU visible?
uv run fertprec k6 --verify        # every frozen branch still exists?
uv run fertprec k6 --dry-run       # what would run, what is already cached
```

`doctor` and `--verify` matter for the same reason: both failures are cheap now
and expensive eight hours in. `doctor` in particular catches lazily-resolved
imports -- `transformers` pulls in torchvision on some paths even for text-only
models, and that error would otherwise surface only after a 14 GB download.

## Smoke first

```bash
uv run fertprec k6 --steps 70000 --bits 4 --eval-sets wikitext2 \
                   --max-windows 8 --out data/k6_smoke.jsonl
```

Exercises the whole stack — Hub revision download, BF16 load, calibration
build, GPTQ, perplexity — in minutes rather than hours. **Write smoke output to
a different file**: `--max-windows` changes the number, and although the resume
key does not include it, a smoke row in the real file would still be a row
measured on 8 windows sitting beside rows measured on hundreds.

## Full run

```bash
uv run fertprec k6 --out data/k6.jsonl
```

Resumable: rows are appended, and a restart skips whatever is already in the
file. Killing the job mid-checkpoint loses at most that checkpoint.

Rough cost: 8 checkpoints x (one BF16 pass + two GPTQ quantizations + four
quantized perplexity passes). Expect **10-14 hours** on one A6000, dominated by
GPTQ. Also budget **~110 GB of downloads** — each revision is a full 7B
checkpoint — and disk for the Hub cache.

### Pin the process to one large card, always

The backend uses every visible CUDA device. On a machine with a small auxiliary
GPU that means an OOM on the small one and a silent CPU fallback for the
Hessian inverse, which makes layers within a single model incomparable. Check
with `nvidia-smi`, then pin.

### Using both cards

The cells are independent per checkpoint, so split the ladder:

```bash
CUDA_VISIBLE_DEVICES=0 uv run fertprec k6 --steps 70000,108000,165000,254000 &
CUDA_VISIBLE_DEVICES=1 uv run fertprec k6 --steps 390000,599000,920000,1413814 &
```

Both write to `data/k6.jsonl` by append, which is safe for line-sized writes.
Nothing here is scored on wall-clock, so contention between the two processes

## Read the result

```bash
uv run fertprec analyse --out data/k6.jsonl
```

Prints one fit per (bits, eval set) with 95% CI, plus the verdict:
**REPLICATES / WEAK / FAILS / INCONCLUSIVE / UNUSABLE**, per `prereg-k6.md` §4.

Then write `docs/k6-result-<date>.md`. Report the registered verdict as
primary. If a threshold looks wrong in hindsight, say so there — do not edit
the pre-registration.

## Things that would invalidate the run

- Mixing checkpoints from stages 2 or 3 into the fit (different data mixture).
- Rebuilding calibration per checkpoint. It is built once and reused; if the
  process restarts, verify the same C4 stream order is still being produced.
- Upgrading `gptqmodel` or `transformers` midway. Provenance records versions
  per row; if they differ across rows, split the fit rather than pooling.
- Comparing BF16 and quantized perplexity computed on different windows. The
  runner builds windows once per eval set and reuses them; keep it that way.
