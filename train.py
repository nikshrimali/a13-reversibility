"""Train the plain GPT or a reversible one, logging loss, grad norm, tokens per second, MFU and peak memory.

    uv run train.py --data data/fwe50 --out runs/base --arch gpt                  # plain GPT, 50M tokens
    uv run train.py --data data/fwe50 --out runs/rev --arch coupling              # reversible, same batch
    uv run train.py --data data/fwe50 --out runs/smoke --arch midpoint --max-tokens 2M
    uv run train.py --data data/fwe50 --out runs/b --arch coupling --micro-batch 512 --bench-steps 3   # does it fit?

The loop is a10-pipeline's train.py. Kept: the batching, AdamW groups, warmup + cosine schedule,
loss scaling, held-out eval, checkpoints, the memory bill and the compile check. Removed: a10's
learning-rate spike and shape-trace experiments. Added for a13:

    --arch          gpt (a10's model) or a reversible variant: coupling, midpoint, euler, momentum
    peak memory     every step, reset at the start of the step, so the log says what each step needed
    HFU             MFU counts 6N per token; HFU adds the FLOPs the reversible backward spends recomputing
    recon           every --recon-every-tokens, how far the undo drifts from the true layer inputs
    --bench-steps   run N steps and stop: no eval, no checkpoint. Writes bench.json, or exits with
                    code 3 on out-of-memory, which is how probe.py finds the largest batch
    done.json       written at the very end; the notebook skips a run that has one
    --resume        continues from ckpt.pt (model, AdamW, loss scaler, step) if it exists

Eval and recon are scheduled in tokens, not steps, so runs with different batch sizes are
measured at the same points of the 50M.
"""

import os

# Let the CUDA allocator grow and shrink its blocks instead of keeping fixed-size ones. Without it,
# a long run's memory can fragment and fail at a batch the probe said fits. Must be set before
# torch touches the GPU, so before `import torch`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import contextlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
from tokenizers import Tokenizer

import flops
from model import GPTConfig, count_params, token_loss
from rev import RevGPT, build

OOM_EXIT = 3


def load_tokens(data_dir, split):
    """Open <data_dir>/<split>.bin as a read-only array of uint16 token ids without loading it into RAM."""
    return np.memmap(f"{data_dir}/{split}.bin", dtype=np.uint16, mode="r")


def get_batch(tokens, seq_ids, seq_len, dev):
    """Cut sequences out of the token array and make (input, target) pairs.

    Each sequence is seq_len + 1 consecutive tokens. The input is the first seq_len tokens and
    the target is the same window shifted one to the left, so target[t] is the token that comes
    after input[t].

    Returns:
        x: (B, T) int64 inputs, y: (B, T) int64 targets.
    """
    w = seq_len + 1
    batch = np.stack([tokens[i * w : (i + 1) * w] for i in seq_ids]).astype(np.int64)
    batch = torch.from_numpy(batch).to(dev)
    return batch[:, :-1], batch[:, 1:]


def configure_optimizer(model, lr, weight_decay, fused=False):
    """AdamW. Matrices get weight decay; LayerNorm gains and biases (1-D tensors) do not."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=fused or None)


def lr_at(step, peak, warmup, total, min_ratio):
    """Linear warmup to `peak`, then a cosine curve down to peak * min_ratio."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, tokens, seq_len, batch, amp, dev, max_tokens=None):
    """Average next-token loss on held-out text the model never trains on.

    Under no_grad a reversible model runs its plain path: nothing is stored either way.
    """
    model.eval()
    n_seq = len(tokens) // (seq_len + 1)
    if max_tokens:
        n_seq = min(n_seq, max_tokens // seq_len)
    total, count = 0.0, 0
    for start in range(0, n_seq, batch):
        x, y = get_batch(tokens, range(start, min(start + batch, n_seq)), seq_len, dev)
        with amp:
            total += token_loss(model(x), y, reduction="sum").item()
        count += y.numel()
    model.train()
    return total / count


def check_compiled_grads(model, fwd, x, y, amp, tol=0.01):
    """Compare the compiled model's gradients with plain PyTorch's on one batch (from a10).

    Returns the largest relative difference in any parameter's gradient norm; exits if above tol.
    """
    norms = []
    for f in (model, fwd):
        model.zero_grad(set_to_none=True)
        with amp:
            token_loss(f(x), y).backward()
        norms.append({n: p.grad.float().norm().item() for n, p in model.named_parameters()})
    model.zero_grad(set_to_none=True)
    worst_name, worst = max(((n, abs(norms[1][n] - norms[0][n]) / max(norms[0][n], 1e-12)) for n in norms[0]),
                            key=lambda kv: kv[1])
    print(f"compile check: largest gradient difference vs eager is {100 * worst:.3f}% ({worst_name})")
    if worst > tol:
        raise SystemExit(f"compiled gradients disagree with eager ones by {100 * worst:.1f}% on {worst_name}; "
                         "run without --compile")
    return worst


def print_memory_bill(model, n_params, cfg, batch_shape, amp_dtype, dev, measured, saved):
    """Print the textbook memory bill, autograd's real saved-tensor list, and the device's own count."""
    if isinstance(model, RevGPT):
        n_streams = 1 if model.rcfg.variant == "euler" else 2
        rows = flops.rev_memory_bill(n_params, cfg, batch_shape, amp_dtype, n_streams, 8 if model.fixed else 4)
    else:
        rows = flops.memory_bill(n_params, cfg, batch_shape, amp_dtype)
    print("\nWhat one step costs in memory, textbook prediction:")
    for item, b, why in rows:
        print(f"  {item:<28} {b / 1e6:8.1f} MB   {why}")
    print(f"  {'total':<28} {sum(b for _, b, _ in rows) / 1e6:8.1f} MB")
    print(f"What autograd actually saved for backward: {saved.total_bytes / 1e6:.1f} MB")
    for shape, dtype, n, b in saved.by_kind():
        if b >= 1e6:
            print(f"  {b / 1e6:8.1f} MB  {n:3d} x {str(shape):<20} {dtype}")
    if dev.type != "cpu":
        print("Measured on the device (bytes held by tensors):")
        print(f"  before forward   {measured['before'] / 1e6:8.1f} MB   weights + AdamW state")
        print(f"  after forward    {measured['after_forward'] / 1e6:8.1f} MB   + activations saved for backward")
        print(f"  after backward   {measured['after_backward'] / 1e6:8.1f} MB   activations freed, gradients added")
        if dev.type == "cuda":
            print(f"  peak in the step {measured['peak'] / 1e6:8.1f} MB   highest point, inside backward")
    print()
    return rows


def parse_count(s):
    """"50M" -> 50_000_000, "2.5M" -> 2_500_000, "300K" -> 300_000, plain numbers pass through."""
    s = str(s).upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(s[-1], 1)
    return int(float(s.rstrip("KMB")) * mult)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="gpt", choices=["gpt", "coupling", "midpoint", "euler", "momentum"])
    ap.add_argument("--rev-h", type=float, default=0.5, help="midpoint step size")
    ap.add_argument("--rev-gamma", type=float, default=0.9, help="momentum coefficient")
    ap.add_argument("--euler-iters", type=int, default=3, help="fixed-point repeats per layer undo")
    ap.add_argument("--plain", action="store_true",
                    help="diagnostic: train a reversible architecture with ordinary autograd (stores every layer)")
    ap.add_argument("--stream-dtype", default="fp32", choices=["fp32", "fp64", "fixed", "fp16", "bf16"])
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--max-tokens", default=None, help="train on at most this many tokens, e.g. 50M or 5M")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fused-adam", action="store_true")
    ap.add_argument("--peak-tflops", type=float, default=None)
    ap.add_argument("--eval-every-tokens", default="2.5M")
    ap.add_argument("--eval-tokens", type=int, default=None, help="default: the whole eval split")
    ap.add_argument("--eval-batch", type=int, default=64, help="sequences per eval forward, kept small so eval never OOMs")
    ap.add_argument("--recon-every-tokens", default="2.5M", help="reversible only; 0 turns it off")
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=None)
    ap.add_argument("--sample-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = flops.device()
    amp_dtype = flops.autocast_dtype(dev, args.dtype)
    amp = torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype != torch.float32 else contextlib.nullcontext()
    scaler = torch.amp.GradScaler(dev.type) if amp_dtype == torch.float16 else None
    bench = args.bench_steps is not None

    meta = json.load(open(f"{args.data}/meta.json"))
    cfg = GPTConfig(vocab=meta["vocab_size"], context=args.seq_len, layers=args.layers, width=args.width,
                    heads=args.heads)
    rkw = {} if args.arch == "gpt" else {"h": args.rev_h, "gamma": args.rev_gamma, "iters": args.euler_iters,
                                          "stream_dtype": args.stream_dtype}
    model = build(cfg, args.arch, **rkw).to(dev)
    if args.plain:
        model.plain = True
    n_params = count_params(model)
    block_params = sum(p.numel() for p in model.blocks.parameters())
    extra = flops.recompute_flops_per_token(block_params, args.arch, args.euler_iters)
    peak, peak_src = flops.peak_flops(dev, amp_dtype, args.peak_tflops)
    print(f"arch {args.arch}: {n_params:,} params ({n_params / 1e6:.2f}M), {block_params:,} in the blocks")
    print(f"FLOPs per token: {flops.flops_per_token(n_params) / 1e6:.1f}M useful (6N) + {extra / 1e6:.1f}M recompute")
    print(f"device: {dev} ({torch.cuda.get_device_name(dev) if dev.type == 'cuda' else dev.type}), "
          f"matmuls in {amp_dtype}, loss scaling {'on' if scaler else 'off'}")
    print(f"peak: {peak_src}")

    train_tokens = load_tokens(args.data, "train")
    eval_tokens = load_tokens(args.data, "eval")
    n_seq = len(train_tokens) // (args.seq_len + 1)
    order = np.random.default_rng(args.seed).permutation(n_seq)
    tokens_per_step = args.micro_batch * args.seq_len
    total_steps = n_seq // args.micro_batch
    if args.max_tokens:
        total_steps = min(total_steps, parse_count(args.max_tokens) // tokens_per_step)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if bench:
        total_steps = args.bench_steps
    eval_every = max(1, round(parse_count(args.eval_every_tokens) / tokens_per_step))
    recon_every = max(1, round(parse_count(args.recon_every_tokens) / tokens_per_step)) if parse_count(args.recon_every_tokens) else 0
    print(f"{total_steps} steps x {tokens_per_step:,} tokens = {total_steps * tokens_per_step / 1e6:.2f}M tokens, "
          f"batch {args.micro_batch} x {args.seq_len}")

    opt = configure_optimizer(model, args.lr, args.weight_decay, args.fused_adam)
    step = 0
    ckpt_path = f"{args.out}/ckpt.pt"
    if args.resume and os.path.exists(ckpt_path) and not bench:
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        step = state["step"]
        print(f"resumed from {ckpt_path} at step {step}")
    fwd = torch.compile(model) if args.compile else model
    if args.compile:
        check_compiled_grads(model, fwd, *get_batch(train_tokens, order[: args.micro_batch], args.seq_len, dev), amp)

    resuming = step > 0
    log = open(f"{args.out}/{'bench' if bench else 'metrics'}.jsonl", "a" if resuming else "w")
    run_cfg = {**vars(args), "model": asdict(cfg), "n_params": n_params, "block_params": block_params,
               "recompute_flops_per_token": extra, "peak_flops": peak, "peak_source": peak_src, "device": str(dev),
               "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type,
               "amp_dtype": str(amp_dtype), "tokens_per_step": tokens_per_step, "total_steps": total_steps,
               "eval_every_steps": eval_every, "recon_every_steps": recon_every, "torch": torch.__version__}
    if not bench:
        with open(f"{args.out}/config.json", "w") as f:
            json.dump(run_cfg, f, indent=2)

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": asdict(cfg),
                    "scaler": scaler.state_dict() if scaler else None}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    recon_x = get_batch(eval_tokens, range(8), args.seq_len, dev)[0]

    def recon():
        if isinstance(model, RevGPT) and recon_every:
            with amp:
                errs = model.reconstruction_error(recon_x)
            write({"type": "recon", "step": step, "tokens": step * tokens_per_step, "per_layer": errs, "max": max(errs)})
            print(f"recon step {step}: worst layer input rebuilt to {max(errs):.2e} (relative)", flush=True)

    if step == 0 and not bench:
        loss0 = evaluate(model, eval_tokens, args.seq_len, args.eval_batch, amp, dev, args.eval_tokens)
        write({"type": "eval", "step": 0, "tokens": 0, "loss": loss0})
        print(f"eval step 0: held-out loss {loss0:.4f} (random guessing would be ln({cfg.vocab}) = {math.log(cfg.vocab):.4f})")
        recon()

    last_ckpt = time.time()
    measure_at = step + 1  # the first step of this process; by then AdamW state exists if resumed, else after step 1
    measure_at = max(measure_at, 2)  # step 2 on a fresh run: AdamW state is created by step 1's update
    window, step_times, peaks = [], [], []
    flops.sync(dev)
    t_start = time.time()
    try:
        while step < total_steps:
            t0 = time.perf_counter()
            lr = lr_at(step, args.lr, args.warmup, total_steps, args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            flops.reset_peak(dev)
            measure = step + 1 == measure_at and not bench
            mem = {"before": flops.memory_allocated(dev)} if measure else None
            ids = order[(step % (n_seq // args.micro_batch)) * args.micro_batch :][: args.micro_batch]
            x, y = get_batch(train_tokens, ids, args.seq_len, dev)
            with amp, (flops.saved_tensors() if measure else contextlib.nullcontext()) as saved:
                logits = fwd(x)
                loss = token_loss(logits, y)
            # Apple GPUs keep no peak counter: sample the memory in use after forward and after
            # backward, where it is highest, and keep the larger.
            mps_hi = torch.mps.current_allocated_memory() if dev.type == "mps" else 0
            if measure:
                flops.sync(dev)
                saved.freeze()
                mem["after_forward"] = flops.memory_allocated(dev)
            (scaler.scale(loss) if scaler else loss).backward()
            loss_val = loss.item()
            if dev.type == "mps":
                mps_hi = max(mps_hi, torch.mps.current_allocated_memory())
            # Drop the logits now, or they stay alive into the next step's forward (a10's 134 MB leak).
            del logits, loss
            if measure:
                flops.sync(dev)
                mem["after_backward"] = flops.memory_allocated(dev)
            if scaler:
                scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            flops.sync(dev)
            dt = time.perf_counter() - t0
            step += 1
            peak_alloc, peak_res = flops.peak_memory(dev)
            peak_alloc = max(peak_alloc, mps_hi)
            if measure:
                mem["peak"] = peak_alloc

            tps = tokens_per_step / dt
            step_times.append(dt)
            peaks.append((peak_alloc, peak_res))
            rec = {"type": "train", "step": step, "tokens": step * tokens_per_step, "loss": loss_val,
                   "grad_norm": grad_norm, "lr": lr, "step_time": dt, "tok_per_s": tps,
                   "mfu": flops.mfu(n_params, tps, peak), "hfu": flops.hfu(n_params, extra, tps, peak),
                   "peak_alloc": peak_alloc, "peak_reserved": peak_res}
            if scaler:
                rec["loss_scale"] = scaler.get_scale()
            if getattr(model, "fp_residual", None) is not None:
                rec["fp_residual"] = model.fp_residual
            write(rec)
            if measure:
                bill = print_memory_bill(model, n_params, cfg, (args.micro_batch, args.seq_len), amp_dtype, dev, mem, saved)
                write({"type": "memory", "step": step, **mem, "saved_for_backward": saved.total_bytes,
                       "bill": [[item, b] for item, b, _ in bill]})
            window = (window + [dt])[-args.print_every:]
            if step <= 5 or step % args.print_every == 0 or step == total_steps:
                w_tps = tokens_per_step * len(window) / sum(window)
                w_mfu = flops.mfu(n_params, w_tps, peak)
                mfu_txt = f"{100 * w_mfu:.1f}%" if w_mfu is not None else "n/a"
                print(f"step {step}/{total_steps} loss {loss_val:.4f} grad_norm {grad_norm:.3f} lr {lr:.2e} "
                      f"{w_tps:,.0f} tok/s mfu {mfu_txt} peak {peak_alloc / 1e9:.2f} GB ({dt * 1000:.0f} ms)", flush=True)

            # Everything below is outside the timed region.
            if not bench and (step % eval_every == 0 or step == total_steps):
                ev = evaluate(model, eval_tokens, args.seq_len, args.eval_batch, amp, dev, args.eval_tokens)
                write({"type": "eval", "step": step, "tokens": step * tokens_per_step, "loss": ev})
                print(f"eval step {step}: held-out loss {ev:.4f}", flush=True)
            if not bench and recon_every and (step % recon_every == 0 or step == total_steps):
                recon()
            if not bench and time.time() - last_ckpt > args.ckpt_minutes * 60 and step < total_steps:
                save()
                last_ckpt = time.time()
                print(f"saved {ckpt_path} at step {step}", flush=True)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        # CUDA raises OutOfMemoryError; Apple GPUs raise a RuntimeError saying "out of memory".
        if not isinstance(e, torch.OutOfMemoryError) and "out of memory" not in str(e).lower():
            raise
        print(f"OUT OF MEMORY at batch {args.micro_batch} x {args.seq_len}, step {step}: {str(e).splitlines()[0]}")
        sys.exit(OOM_EXIT)

    # Median over steps after the first 10 (warmup of kernels, allocator and loss scale).
    timed = step_times[10:] if len(step_times) > 20 else step_times[1:] or step_times
    med = statistics.median(timed)
    summary = {"arch": args.arch, "micro_batch": args.micro_batch, "seq_len": args.seq_len, "steps": step,
               "tokens": step * tokens_per_step, "median_step_time": med, "tok_per_s": tokens_per_step / med,
               "mfu": flops.mfu(n_params, tokens_per_step / med, peak),
               "hfu": flops.hfu(n_params, extra, tokens_per_step / med, peak),
               "peak_alloc": max(p[0] for p in peaks), "peak_reserved": max(p[1] for p in peaks),
               "n_params": n_params, "device": run_cfg["gpu"], "amp_dtype": str(amp_dtype),
               "stream_dtype": args.stream_dtype if args.arch != "gpt" else None,
               "gpu_total": torch.cuda.get_device_properties(dev).total_memory if dev.type == "cuda" else None}
    if bench:
        with open(f"{args.out}/bench.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return

    save()
    print(f"saved {ckpt_path}")
    evals = [json.loads(l) for l in open(f"{args.out}/metrics.jsonl") if '"eval"' in l]
    summary.update(final_eval_loss=evals[-1]["loss"], wall_seconds=time.time() - t_start)
    if args.sample_tokens and os.path.exists(f"{args.data}/tokenizer.json"):
        tok = Tokenizer.from_file(f"{args.data}/tokenizer.json")
        prompt = "The most important thing about learning is"
        torch.manual_seed(0)
        with amp:
            out = model.generate(torch.tensor([tok.encode(prompt).ids], device=dev), args.sample_tokens)
        text = tok.decode(out[0].tolist())
        print(f"\nsample:\n{text}")
        with open(f"{args.out}/sample.txt", "w") as f:
            f.write(text + "\n")
    with open(f"{args.out}/done.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
