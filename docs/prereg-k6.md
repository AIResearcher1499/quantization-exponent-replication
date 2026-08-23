# Pre-registration — K6: does the published exponent replicate?

- **Frozen:** 2026-08-23, before any row of `data/k6.jsonl` exists.

## 1. What is at stake

This work **tests a published exponent** rather than fitting its own curve. That exponent is `beta = 0.5251` from `2411.17691`,
fit on RefinedWeb and Wikitext-2 — English, one model family, and by the
authors' own statement experiments amounting to ~300B tokens.

Nobody has checked whether it transfers to a different corpus, tokenizer and
training recipe. If it does not, and this is discovered late, a
measured exponent of (say) 0.31 is uninterpretable: it could mean the law does
not generalise, or it could mean the mechanism under test is wrong. Those two
causes cannot be separated after the fact. This check separates them beforehand, in English, with no new machinery.

## 2. Materials

**Model:** `allenai/Olmo-3-1025-7B`. Chosen because Ai2 publishes intermediate
checkpoints as branches: same model, same data, same recipe, different `D`.
1421 `stage1-step*` branches exist.

**Checkpoints (8, frozen), log-spaced over stage 1:**

```
70000, 108000, 165000, 254000, 390000, 599000, 920000, 1413814
```

Spanning log10 4.85 -> 6.15, a 20x range in `D`. Under `beta = 0.5251` that
predicts a ~4.8x range in degradation — far more than measurement noise.

**Stage 1 only.** Stages 2 and 3 change the data mixture, which would confound
`D` with composition. The earliest 5% of stage 1 is excluded: a barely-trained
model's perplexity is not a meaningful baseline.

**`D` is measured in optimizer steps, not tokens.** Batch size and sequence
length are constant within stage 1, so `log D = log(step) + constant`, and the
constant is absorbed into the intercept. **The fitted slope is unaffected**, so
no token-per-step conversion is needed and none may be invented later.

**Quantization:** GPTQ, `bits in {4, 3}`, group size 128, act-order on.
Calibration frozen: 128 sequences of 2048 tokens from C4 (en), seed 0, and the
**same calibration set for every checkpoint**. Calibration data never overlaps
the evaluation sets.

**Evaluation (two sets, both English):**

1. `wikitext-2-raw-v1`, test split — the set `2411.17691` itself used.
2. `c4` (en), validation shard — a different domain, to see whether the
   exponent is a property of the law or of Wikitext.

Sequence length 2048, non-overlapping windows, identical windows for BF16 and
quantized runs.

## 3. Quantity

For each (checkpoint, bits, eval set):

```
dqLoss = log(ppl_quantized) - log(ppl_bf16)      # nats, matching 2411.17691
```

Recorded per row: `step`, `bits`, `eval_set`, `ppl_bf16`, `ppl_quant`,
`dqLoss`, `n_windows`, `n_tokens`, plus provenance (hardware, library versions,
kernel, commit).

## 4. Fit and decision rule (frozen)

Ordinary least squares, one fit per (bits, eval set) — four fits:

```
log(dqLoss) ~ beta * log(step) + c
```

95% CI on `beta` from the OLS standard error.

- **REPLICATES** — the framing stands: the CI contains **0.5251** in **>= 2 of
  the 4** fits.
- **WEAK** — keep the point prediction but hedge it explicitly: no fit contains
  0.5251, but >= 2 of 4 have a CI lying entirely within **[0.40, 0.65]**.
- **FAILS** — every prediction elsewhere in the project downgrades from
  "consistent with 0.5251" to "positive with the correct sign", and the paper
  is rewritten as sign-and-channel before anything else is claimed: >= 2 of 4
  fits have a CI excluding both 0.5251 and [0.40, 0.65].

Reported as effect size and CI per fit. No pooling across bit widths or eval
sets — `gamma = 5.4967` means 4-bit and 3-bit are far apart, and pooling would
hide a disagreement that is itself informative.

## 5. Data-quality guards (decided in advance)

- Any row with `dqLoss <= 0` (quantization improving loss) is flagged and
  excluded from the fit, and reported separately. It indicates a broken
  quantization run, not a finding.
- Any row with `ppl_bf16` more than 3x the median BF16 perplexity across
  checkpoints is flagged: a checkpoint that failed to load correctly looks
  exactly like an early checkpoint otherwise.
- If fewer than 6 of the 8 checkpoints yield usable rows for a given (bits,
  eval set), that fit is reported as UNUSABLE rather than fitted on the
  remainder.

## 6. What K6 does not do

K6 is a precondition check, not evidence for any downstream hypothesis. A
REPLICATES verdict only means the published exponent is usable as a reference
point; it supports nothing on its own.
