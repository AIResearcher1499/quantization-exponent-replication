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

    try:
        import torch
        MIN_GIB = 20.0
        if torch.cuda.is_available():
            small = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                tot_gib = total / 2**30
                mark = "" if tot_gib >= MIN_GIB else "   <-- TOO SMALL"
                print(f"  gpu {i}: {torch.cuda.get_device_name(i)} "
                      f"{free/2**30:.1f}/{tot_gib:.1f} GiB free{mark}")
                if tot_gib < MIN_GIB:
                    small.append(i)
            if small:
                ok = False
                print(f"\n  gpu {small} are below {MIN_GIB:.0f} GiB and are visible to "
                      f"this process.\n"
                      "  The quantization backend spreads the model across every "
                      "visible device,\n"
                      "  runs out of memory on the small one, and silently falls "
                      "back to CPU for\n"
                      "  the Hessian inverse. That is not just slow: some layers "
                      "then get a\n"
                      "  CPU-computed inverse and others a GPU one, within the same "
                      "model, and\n"
                      "  which layers depends on where the OOM happened. Across a "
                      "ladder of\n"
                      "  checkpoints that is a confound sitting directly on the "
                      "measurement.\n"
                      "  Restrict the process, e.g. CUDA_VISIBLE_DEVICES=0")
        else:
            print("  no CUDA device visible -- the ladder needs one")
            ok = False
    except Exception as exc:
        print(f"  gpu check failed: {exc}")
        ok = False

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
