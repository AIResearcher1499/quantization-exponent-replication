# CLAUDE.md

Operating rules for this repository. Read `docs/prereg-k6.md` first, then
`docs/k6-runbook.md`.

## What this repo is

A single pre-registered replication check: does `beta = 0.5251` from
`2411.17691` reproduce on a corpus, tokenizer and training recipe it was not
fitted on? It answers that and nothing else. Do not extend the scope.

## Hard rules

- **Never edit `docs/prereg-k6.md` once `data/k6.jsonl` exists.** If a
  threshold looks wrong once data is in, report the registered rule as primary
  and put the alternative beside it in the result document. Changing the rule
  after seeing the data destroys the only thing that makes this check worth
  running.
- **Never pool across bit widths or evaluation sets.** `gamma = 5.4967` puts
  INT4 and INT3 far apart, and a disagreement between eval sets is itself the
  finding. Four fits, reported separately.
- **Smoke runs write to a different file.** `--max-windows` changes the number;
  a smoke row sitting beside full rows in `data/k6.jsonl` is indistinguishable
  from a real measurement afterwards.
- **Only `stage1` checkpoints.** Stages 2 and 3 change the data mixture, which
  confounds `D` with composition.
- **Do not invent a token-per-step conversion.** Steps are used deliberately:
  `log D = log(step) + constant` within stage 1, and the constant is absorbed
  into the intercept. A conversion factor guessed from a blog post would add
  error without adding information.
- **Never put a model name, assistant name, or session link in a commit
  message.** Write commits as a person would: subject, then what and why.
- **Do not upgrade `gptqmodel` or `transformers` mid-ladder.** Provenance
  records versions per row. If rows disagree on versions, split the fit rather
  than pooling and say so.

## Running

Order matters, and skipping it costs hours:

```bash
uv run pytest                 # no GPU needed; logic only
uv run fertprec doctor        # imports + GPU, before anything is downloaded
uv run fertprec k6 --verify   # all eight frozen branches still on the Hub?
uv run fertprec k6 --dry-run  # what would run, what is cached
# smoke, to a separate file:
uv run fertprec k6 --steps 70000 --bits 4 --eval-sets wikitext2 \
                   --max-windows 8 --out data/k6_smoke.jsonl
uv run fertprec k6 --gpu 0 --out data/k6.jsonl
uv run fertprec analyse --out data/k6.jsonl
```

`--gpu` defaults to `auto`, which takes the emptiest card with enough memory.
Pass an explicit index to split the ladder across two cards.

The run is resumable: rows are appended and a restart skips what is already
present. Killing a job loses at most the checkpoint in progress.

## When something breaks

- **`No module named torchvision`.** `transformers` reaches into torchvision on
  some import paths even for text-only models. Install it (`uv pip install
  torchvision`) or reinstall the extras. `fertprec doctor` catches this before
  a checkpoint is downloaded, which is the whole reason it exists.
- **`gptqmodel` does not recognise the architecture.** Most likely failure, and
  the reason the smoke run exists. Check `transformers` is new enough to load
  `Olmo3ForCausalLM`, and that `trust_remote_code=True` reaches both the loader
  and the tokenizer. Report the exact error rather than switching quantization
  backends: a different backend is a different measurement.
- **Out of memory during quantization, or a CPU fallback warning.** The backend
  spreads the model across every *visible* CUDA device. If a small GPU is
  visible it will OOM there and fall back to CPU for the Hessian inverse, which
  is not merely slow: layers then differ in how they were quantized, within one
  model, depending on where the OOM landed. `--gpu N` (default `auto`) pins the
  process to one large card before torch initialises CUDA, so the backend
  cannot spill onto another; a card below 20 GiB is refused outright rather
  than used. `fertprec doctor` lists every card and says which are usable.
  **Any run that logged a CPU fallback must be discarded, not resumed** -- its
  layers were not all quantized the same way.
- **Disk fills.** Each revision is a full 7B checkpoint; the ladder pulls
  ~110 GB. Clear the Hub cache between checkpoints if needed — the results file
  is what matters, not the weights.
- **A branch is missing.** Stop. Do not substitute a nearby step: the ladder is
  frozen in the pre-registration. Report which step is gone.

## Reporting

Write `docs/k6-result-<date>.md` with the four fits, their 95% CIs, the
registered verdict, and every excluded or flagged row with the reason. Negative
and inconclusive outcomes are reported exactly as prominently as positive ones.
