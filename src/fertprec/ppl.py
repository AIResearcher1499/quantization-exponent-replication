"""Perplexity on fixed, non-overlapping windows.

Windows are built once per (eval set, tokenizer) and reused for the BF16 and
quantized halves of a comparison. If the two halves saw different windows the
difference between them would not be the quantization effect.
"""

from __future__ import annotations

import math

SEQ_LEN = 2048


def load_eval_text(eval_set: str, max_chars: int = 6_000_000) -> str:
    """Return one concatenated string for the named evaluation set."""
    from datasets import load_dataset

    if eval_set == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(ds["text"])
    if eval_set == "c4val":
        ds = load_dataset("allenai/c4", "en", split="validation",
                          streaming=True)
        out, n = [], 0
        for row in ds:
            out.append(row["text"])
            n += len(row["text"])
            if n >= max_chars:
                break
        return "\n\n".join(out)
    raise ValueError(f"unknown eval set: {eval_set!r}")


def make_windows(text: str, tokenizer, seq_len: int = SEQ_LEN,
                 max_windows: int | None = None):
    """Tokenize once, then cut into non-overlapping windows of `seq_len`."""
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // seq_len
    if max_windows is not None:
        n = min(n, max_windows)
    if n == 0:
        raise ValueError("evaluation text is shorter than one window")
    return ids[: n * seq_len].view(n, seq_len)


def perplexity(model, windows, device: str = "cuda", batch_size: int = 1) -> dict:
    """Token-weighted perplexity over the given windows.

    Returns the mean negative log-likelihood as well, because `dqLoss` is a
    difference of losses in nats and going through perplexity and back loses
    precision for no reason.
    """
    import torch

    model.eval()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, windows.size(0), batch_size):
            batch = windows[i : i + batch_size].to(device)
            out = model(batch, labels=batch)
            # HF averages over (seq_len - 1) predicted positions per sequence.
            n_pred = (batch.size(1) - 1) * batch.size(0)
            total_nll += out.loss.item() * n_pred
            total_tok += n_pred
    mean_nll = total_nll / total_tok
    return {"nll": mean_nll, "ppl": math.exp(mean_nll), "n_tokens": total_tok,
            "n_windows": int(windows.size(0))}
