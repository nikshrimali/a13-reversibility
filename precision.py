"""Two measurements behind the fp16-vs-bf16 choice: how fast each dtype runs here, and how fast each one drifts.

    uv run precision.py speed --out runs/precision_speed.json
    uv run precision.py drift --runs runs/screen --out runs/precision_drift.json

speed: a 4096 x 4096 matmul in fp32, fp16 and bf16, then 20 training steps of the plain GPT
    with autocast in each dtype (fp32 meaning no autocast). On a T4, fp16 runs on tensor cores
    and bf16 has no hardware path, so bf16 is expected near fp32 speed. A dtype that fails
    (an old GPU without bf16 at all) is reported as unsupported instead of crashing.

drift: for each reversible variant, load its screening checkpoint (or a fresh model if there is
    none), and measure how far the undo drifts per layer when the streams between layers are
    stored as fixed point, fp32, fp16 or bf16. Fewer bits per stream number means a bigger
    rounding step per layer, and the undo has to subtract its way back through all of them.
"""

import argparse
import contextlib
import json
import os
import statistics
import time

import torch

import flops
from model import GPTConfig, token_loss
from rev import RevConfig, RevGPT, build

DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def speed(args):
    dev = flops.device()
    out = {"device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type, "matmul_tflops": {},
           "step": {}}
    for name, dt in DT.items():
        try:
            out["matmul_tflops"][name] = flops.matmul_ceiling(dev, dt) / 1e12
        except (RuntimeError, TypeError) as e:
            out["matmul_tflops"][name] = None
            print(f"{name} matmul unsupported: {str(e).splitlines()[0]}")
    for name, dt in DT.items():
        try:
            torch.manual_seed(0)
            model = build(GPTConfig(layers=args.layers), "gpt").to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler(dev.type) if dt == torch.float16 else None
            amp = torch.autocast(dev.type, dtype=dt) if dt != torch.float32 else contextlib.nullcontext()
            x = torch.randint(0, 4096, (args.batch, 256), device=dev)
            y = torch.randint(0, 4096, (args.batch, 256), device=dev)
            times = []
            for _ in range(args.steps):
                flops.sync(dev)
                t0 = time.perf_counter()
                with amp:
                    loss = token_loss(model(x), y)
                (scaler.scale(loss) if scaler else loss).backward()
                (scaler.step(opt), scaler.update()) if scaler else opt.step()
                opt.zero_grad(set_to_none=True)
                flops.sync(dev)
                times.append(time.perf_counter() - t0)
            med = statistics.median(times[5:])
            out["step"][name] = {"median_step_s": med, "tok_per_s": args.batch * 256 / med}
            del model, opt
        except (RuntimeError, TypeError) as e:
            out["step"][name] = None
            print(f"{name} training step unsupported: {str(e).splitlines()[0]}")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\n{'dtype':<6} {'matmul TFLOPS':>14} {'train tok/s':>12}")
    for name in DT:
        mm, st = out["matmul_tflops"][name], out["step"][name]
        mm_txt = f"{mm:.2f}" if mm is not None else "n/a"
        st_txt = f"{st['tok_per_s']:,.0f}" if st else "n/a"
        print(f"{name:<6} {mm_txt:>14} {st_txt:>12}")
    return out


def drift(args):
    dev = flops.device()
    amp_dtype = flops.autocast_dtype(dev)
    amp = torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype != torch.float32 else contextlib.nullcontext()
    data_eval = None
    if args.data and os.path.exists(f"{args.data}/eval.bin"):
        import numpy as np
        toks = np.memmap(f"{args.data}/eval.bin", dtype=np.uint16, mode="r")[: 8 * 257]
        data_eval = torch.from_numpy(toks.astype(np.int64)).view(8, 257)[:, :256].to(dev)
    x = data_eval if data_eval is not None else torch.randint(0, 4096, (8, 256), device=dev)
    out = {"device": str(dev), "amp_dtype": str(amp_dtype), "variants": {}}
    for variant in ("coupling", "midpoint", "momentum"):
        ckpt = os.path.join(args.runs, variant, "ckpt.pt") if args.runs else None
        state = torch.load(ckpt, map_location=dev)["model"] if ckpt and os.path.exists(ckpt) else None
        out["variants"][variant] = {"trained": state is not None}
        for sd in ("fixed", "fp32", "fp16", "bf16"):
            torch.manual_seed(0)
            m = RevGPT(GPTConfig(layers=args.layers), RevConfig(variant=variant, stream_dtype=sd)).to(dev)
            if state:
                m.load_state_dict(state)
            with amp:
                errs = m.reconstruction_error(x)
            out["variants"][variant][sd] = errs
            print(f"{variant:<9} streams {sd:<5} ({'trained' if state else 'fresh'}): worst layer {max(errs):.2e}, "
                  f"per layer from the top {', '.join(f'{e:.0e}' for e in errs)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["speed", "drift"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--runs", default=None, help="drift: folder holding <variant>/ckpt.pt from the screening")
    ap.add_argument("--data", default=None, help="drift: data folder, to measure on real text")
    args = ap.parse_args()
    res = speed(args) if args.what == "speed" else drift(args)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
