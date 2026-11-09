"""Find the largest batch that fits: double until out of memory, then bisect.

    uv run probe.py --data data/fwe50 --out runs/probe_gpt --arch gpt
    uv run probe.py --data data/fwe50 --out runs/probe_rev --arch coupling --stream-dtype fixed

Each candidate batch runs as its own process (train.py --bench-steps 3: forward, backward and
AdamW, three times, so the optimizer state and the allocator's blocks are all in place). A
fresh process per try means one out-of-memory failure cannot leave fragments behind that make
the next try fail. Batches go up in steps of --granularity sequences (8: the tensor cores like
multiples of 8).

Writes <out>/probe.json: every try with its outcome and peak memory, and the answer.
"""

import argparse
import json
import os
import subprocess
import sys

OOM_EXIT = 3


def try_batch(args, b, extra):
    """Run 3 training steps at batch b. Returns (fits, bench summary or None)."""
    out = os.path.join(args.out, f"b{b}")
    cmd = [sys.executable, "train.py", "--data", args.data, "--out", out, "--arch", args.arch,
           "--micro-batch", str(b), "--bench-steps", "3", "--seq-len", str(args.seq_len)] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode == 0:
        return True, json.load(open(os.path.join(out, "bench.json")))
    if p.returncode == OOM_EXIT:
        return False, None
    raise SystemExit(f"batch {b} failed for a reason other than memory (exit {p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--start", type=int, default=64)
    ap.add_argument("--cap", type=int, default=1 << 16, help="never try above this (the smoke run keeps it small)")
    ap.add_argument("--granularity", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    args, extra = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)
    tries = []

    def attempt(b):
        fits, s = try_batch(args, b, extra)
        tries.append({"batch": b, "fits": fits, **({"peak_alloc": s["peak_alloc"], "peak_reserved": s["peak_reserved"],
                                                    "tok_per_s": s["tok_per_s"]} if s else {})})
        print(f"  batch {b:>6}: {'fits, peak ' + format(s['peak_alloc'] / 1e9, '.2f') + ' GB' if fits else 'out of memory'}",
              flush=True)
        return fits

    print(f"probing {args.arch} {' '.join(extra)}: doubling from {args.start}")
    ok, bad = 0, None
    b = args.start
    while b <= args.cap:
        if attempt(b):
            ok, b = b, b * 2
        else:
            bad = b
            break
    if bad is None:
        print(f"reached the cap {args.cap} without running out of memory")
    else:
        print(f"bisecting between {ok} and {bad}")
        while bad - ok > args.granularity:
            mid = (ok + bad) // 2 // args.granularity * args.granularity
            if mid <= ok:
                break
            if attempt(mid):
                ok = mid
            else:
                bad = mid
    res = {"arch": args.arch, "extra": extra, "max_batch": ok, "first_oom": bad, "capped": bad is None,
           "granularity": args.granularity, "tries": sorted(tries, key=lambda t: t["batch"])}
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"largest batch that fits: {ok} sequences x {args.seq_len} tokens = {ok * args.seq_len:,} tokens per step"
          + (" (cap reached)" if bad is None else f"; {bad} does not"))


if __name__ == "__main__":
    main()
