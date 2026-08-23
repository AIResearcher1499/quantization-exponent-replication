"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


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

    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

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
