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


def snapshot(model_id: str, revision: str) -> str:
    """Download one revision to the local cache and return its path.

    Both halves of a comparison then load from the *same directory* rather than
    resolving a branch name twice. That removes a real hazard -- a branch
    resolving differently between the BF16 and quantized loads would be
    indistinguishable from the effect under test -- and it sidesteps backends
    that forward unknown kwargs like `revision` into the model constructor.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, revision=revision)


def quantize_gptq(local_path: str, bits: int, calib: list[dict],
                  group_size: int = 128, out_dir: str | None = None):
    """Quantize one checkpoint and return the loaded quantized model.

    Takes a local path, not a repo id plus revision: `GPTQModel.load` forwards
    unknown keyword arguments to the model constructor, and
    `Olmo3ForCausalLM` rejects `revision`.

    `gptqmodel` is the maintained successor to auto-gptq and is the only backend
    supported here. Pin its version and record it in provenance: INT3 support in
    particular has moved between releases, and a kernel change between the BF16
    and quantized halves of a comparison is indistinguishable from the effect
    being measured.
    """
    from gptqmodel import GPTQModel, QuantizeConfig

    cfg = QuantizeConfig(bits=bits, group_size=group_size, desc_act=True,
                         sym=True)
    model = GPTQModel.load(local_path, cfg, trust_remote_code=True)
    model.quantize(calib)
    if out_dir:
        model.save(out_dir)
    return model


def load_quantized(path: str, device: str):
    """Load a saved quantized model onto one device.

    Quantizing and evaluating in the same process leaves parts of the model on
    the meta device -- shapes without data -- and `.to()` cannot move those.
    The error message suggests `to_empty()`; that would silently fill the model
    with uninitialised weights and measure garbage without crashing, so it must
    never be used here.

    Saving and reloading costs a minute and a few GB of scratch, and returns a
    model that is simply complete. It also means a failure during evaluation no
    longer throws away the quantization that preceded it.
    """
    from gptqmodel import GPTQModel

    for kwargs in ({"device": device}, {"device_map": {"": device}}, {}):
        try:
            return GPTQModel.load(path, trust_remote_code=True, **kwargs)
        except TypeError:
            continue
    raise RuntimeError(f"could not load quantized model from {path}")


def consolidate(module, device: str):
    """Put every parameter and buffer of a quantized model on one device.

    GPTQ only moves the Linear layers it is quantizing onto the GPU; norms,
    embeddings and rotary buffers can be left on CPU. The returned model is
    therefore a hybrid, and feeding it CUDA inputs fails inside RMSNorm with
    "expected all tensors to be on the same device".

    Moving is not enough on its own -- anything left behind afterwards is a
    silent hazard rather than a crash, since a stray buffer might still
    broadcast correctly and quietly change a number. So this verifies, and
    raises with the offending names rather than continuing.
    """
    import torch

    meta = [n for n, t in list(module.named_parameters())
            + list(module.named_buffers()) if t.device.type == "meta"]
    if meta:
        raise RuntimeError(
            f"{len(meta)} tensors are on the meta device (e.g. {meta[:3]}). "
            "They hold no data, so moving them would require to_empty(), which "
            "fills them with uninitialised values and would measure garbage. "
            "Save the quantized model and load it back instead.")

    module.to(device)
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())

    stray = {}
    for name, t in list(module.named_parameters()) + list(module.named_buffers()):
        if t.device.type != dev.type or (
                t.device.type == "cuda" and t.device.index != dev.index):
            stray[name] = str(t.device)
    if stray:
        shown = ", ".join(f"{n}={d}" for n, d in list(stray.items())[:6])
        raise RuntimeError(
            f"{len(stray)} tensors are not on {dev} after moving: {shown}"
            + (" ..." if len(stray) > 6 else "")
            + ". Refusing to measure a model split across devices.")
    return module
