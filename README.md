# fertprec — quantization damage across languages, and the exponent behind it

`2411.17691` fits quantization-induced degradation as

```
dqLoss(N, D, P) = k * D^beta / (N^alpha * P^gamma)
k = 0.017, beta = 0.5251, alpha = 0.2261, gamma = 5.4967
```

on RefinedWeb and Wikitext-2, in English, on one model family, over experiments
the authors describe as amounting to ~300B tokens. Whether `beta` transfers to a
different corpus, tokenizer and training recipe has not been checked.

The K6 stage checks it, using the one public artefact that makes a clean `D`
axis possible: the intermediate `stage1` checkpoints of
`allenai/Olmo-3-1025-7B`. Same model, same data, same recipe — only the number
of optimizer steps differs.

The G0 stage asks a different question on existing multilingual checkpoints:
does a language's **token cost** predict how badly quantization damages it,
better than its typological distance from English does?

- K6 protocol, frozen before any data existed: [`docs/prereg-k6.md`](docs/prereg-k6.md)
- How to run K6: [`docs/k6-runbook.md`](docs/k6-runbook.md)
- G0's pre-registration, results, and the design rationale are not published
  here yet. This repository ships the runnable code.

## Method in one paragraph

Eight log-spaced `stage1` checkpoints are evaluated at BF16 and after GPTQ at
INT4 and INT3, on two English evaluation sets, with a calibration set held fixed
across every checkpoint. `dqLoss = log(ppl_quant) - log(ppl_bf16)` in nats. The
fit is `log dqLoss ~ beta log(step)`; because batch size and sequence length are
constant within stage 1, `log D = log(step) + constant` and the constant is
absorbed into the intercept, so no token-per-step conversion is needed and the
fitted slope is unaffected.

The decision rule — REPLICATES / WEAK / FAILS — was written down before the
first row was measured and is not editable afterwards.

## Quick start

```bash
uv venv -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e ".[gpu,stats]"

uv run fertprec doctor           # imports and GPUs, before anything downloads
uv run fertprec k6 --verify      # frozen branches still on the Hub?
uv run fertprec k6 --dry-run     # what would run
uv run fertprec k6 --steps 70000 --bits 4 --eval-sets wikitext2 \
                   --max-windows 8 --out data/k6_smoke.jsonl
uv run fertprec k6 --out data/k6.jsonl
uv run fertprec analyse --out data/k6.jsonl
```

One A6000-class card is enough (7B at BF16 is ~14 GB). Expect 10-14 hours and
~110 GB of checkpoint downloads for the full ladder; the run is resumable and
can be split across two cards.

## Tests

```bash
uv run pytest
```

The tests cover the frozen decision rule and the two mechanisms most likely to
corrupt a long run silently: incomplete resume keys, and results files that
overwrite instead of merging.

## G0

```bash
uv run fertprec g0 --dry-run
uv run fertprec g0 --gpu 0 --models qwen3-8b,llama31-8b
uv run fertprec analyse-g0
```

Models are declared in two layers that are never pooled. Layer 1 has a
disclosed pretraining token count and carries the fit; layer 2 does not, and
answers only whether the phenomenon still appears on current models. Neither
Qwen3.8 nor Gemma 4 discloses a token count, which is why they sit in layer 2.

Covariates are computed rather than typed in: typological distance comes from
URIEL via `lang2vec`. Regenerate with `python scripts/build_covariates.py`.
