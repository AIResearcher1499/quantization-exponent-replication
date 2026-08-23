"""GPTQ quantization with a calibration set frozen across every checkpoint.

The calibration set is built once from a fixed C4 slice with a fixed seed and
reused for every checkpoint and both bit widths. If calibration varied with the
checkpoint, the ladder would measure calibration luck rather than `D`.
"""

from __future__ import annotations

CALIB_N = 128
CALIB_SEQ = 2048
CALIB_SEED = 0


def build_calibration(tokenizer, n: int = CALIB_N, seq_len: int = CALIB_SEQ,
                      seed: int = CALIB_SEED) -> list[dict]:
    """`n` sequences of exactly `seq_len` tokens from C4 (en) train.

    Streamed, so nothing large is written to disk. Deterministic given the seed:
    documents are taken in stream order, which is fixed for a given snapshot.
    """
    import torch
    from datasets import load_dataset

    torch.manual_seed(seed)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    buf: list[int] = []
    samples: list[dict] = []
    for row in ds:
        buf.extend(tokenizer(row["text"]).input_ids)
        while len(buf) >= seq_len and len(samples) < n:
            chunk = buf[:seq_len]
            buf = buf[seq_len:]
            ids = torch.tensor([chunk])
            samples.append({"input_ids": ids,
                            "attention_mask": torch.ones_like(ids)})
        if len(samples) >= n:
            break
    if len(samples) < n:
        raise RuntimeError(f"calibration short: {len(samples)}/{n}")
    return samples


def quantize_gptq(model_id: str, revision: str, bits: int, calib: list[dict],
                  group_size: int = 128, out_dir: str | None = None):
    """Quantize one checkpoint and return the loaded quantized model.

    `gptqmodel` is the maintained successor to auto-gptq and is the only backend
    supported here. Pin its version and record it in provenance: INT3 support in
    particular has moved between releases, and a kernel change between the BF16
    and quantized halves of a comparison is indistinguishable from the effect
    being measured.
    """
    from gptqmodel import GPTQModel, QuantizeConfig

    cfg = QuantizeConfig(bits=bits, group_size=group_size, desc_act=True,
                         sym=True)
    model = GPTQModel.load(model_id, cfg, revision=revision,
                           trust_remote_code=True)
    model.quantize(calib)
    if out_dir:
        model.save(out_dir)
    return model
