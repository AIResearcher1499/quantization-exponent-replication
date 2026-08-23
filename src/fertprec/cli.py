"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def doctor() -> int:
    """Import every dependency the run needs, then report the GPU.

    Exists because the expensive failures happen after a 14 GB download: an
    import that resolves lazily inside `transformers` or `gptqmodel` will not
    fail until a model is loaded. Running this first turns a two-hour discovery
    into a two-second one.
    """
    import importlib

    ok = True
    checks = [
        ("torch", None),
        ("torchvision", "transformers imports it on some paths, text-only or not"),
        ("transformers", None),
        ("datasets", None),
        ("gptqmodel", "quantization backend"),
        ("huggingface_hub", None),
    ]
    for mod, note in checks:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  ok      {mod:<16} {ver}")
        except Exception as exc:
            ok = False
            print(f"  MISSING {mod:<16} {type(exc).__name__}: {exc}"
                  + (f"  ({note})" if note else ""))

    from . import gpu as gpumod

    gpus = gpumod.list_gpus()
    if not gpus:
        print("  no GPU reported by nvidia-smi -- the ladder needs one")
        ok = False
    else:
        for g in gpus:
            mark = "" if g.usable else "   <-- too small, must not be visible"
            print(f"  gpu {g.index}: {g.name} "
                  f"{g.free_gib:.1f}/{g.total_gib:.1f} GiB free{mark}")
        usable = [g.index for g in gpus if g.usable]
        if not usable:
            ok = False
            print(f"\n  no card has the {gpumod.MIN_GIB:.0f} GiB this needs")
        else:
            print(f"\n  will pin to one of {usable}; pass --gpu N to choose")

    print("\nready" if ok else "\nnot ready -- fix the above before running k6")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fertprec")
    sub = ap.add_subparsers(dest="cmd", required=True)

    k6 = sub.add_parser("k6", help="OLMo-3 ladder: does beta=0.5251 replicate?")
    k6.add_argument("--out", default="data/k6.jsonl")
    k6.add_argument("--steps", default="", help="comma-separated subset")
    k6.add_argument("--bits", default="4,3")
    k6.add_argument("--eval-sets", default="wikitext2,c4val")
    k6.add_argument("--max-windows", type=int, default=None,
                    help="cap windows per eval set (smoke runs only)")
    k6.add_argument("--device", default="cuda")
    k6.add_argument("--gpu", default="auto",
                    help="CUDA index to pin to, or 'auto' for the emptiest "
                         "card with enough memory, or 'off' to leave "
                         "CUDA_VISIBLE_DEVICES alone")
    k6.add_argument("--dry-run", action="store_true")
    k6.add_argument("--verify", action="store_true",
                    help="check every frozen branch still exists, then exit")

    an = sub.add_parser("analyse", help="apply the frozen K6 decision rule")
    an.add_argument("--out", default="data/k6.jsonl")

    sub.add_parser("doctor", help="check the environment before downloading anything")

    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if args.cmd == "doctor":
        return doctor()

    if args.cmd == "analyse":
        from . import k6 as k6mod
        print(json.dumps(k6mod.analyse(pathlib.Path(args.out)), indent=2))
        return 0

    from . import checkpoints
    from . import k6 as k6mod

    if args.verify:
        avail = checkpoints.verify_available()
        for step, ok in avail.items():
            print(f"  step {step}: {'ok' if ok else 'MISSING'}")
        return 0 if all(avail.values()) else 1

    # Pin before torch is imported anywhere: CUDA_VISIBLE_DEVICES has no
    # effect once CUDA is initialised, and leaving a small auxiliary card
    # visible makes the backend OOM there and fall back to CPU mid-model.
    if args.gpu != "off":
        from . import gpu as gpumod
        chosen = gpumod.pick(args.gpu)
        if chosen is not None:
            gpumod.restrict_to(chosen)
            print(f"pinned to cuda:{chosen} (CUDA_VISIBLE_DEVICES={chosen})")

    steps = [int(s) for s in args.steps.split(",") if s] or None
    bits = tuple(int(b) for b in args.bits.split(","))
    eval_sets = tuple(e for e in args.eval_sets.split(",") if e)

    if args.dry_run:
        out = pathlib.Path(args.out)
        done = k6mod.load_done(out)
        for c in k6mod.plan(steps, bits, eval_sets):
            print(f"  {'CACHED' if c.key() in done else 'RUN   '} "
                  f"step={c.step} int{c.bits} {c.eval_set}")
        return 0

    k6mod.run(pathlib.Path(args.out), steps=steps, bits=bits,
              eval_sets=eval_sets, max_windows=args.max_windows,
              device=args.device)
    return 0
