"""Check every reversible variant's gradients against ordinary autograd on the same weights.

    uv run gradcheck.py                         # fp64 on CPU, then the device in its training precision
    uv run gradcheck.py --out runs/gradcheck.json

The reversible backward never sees the layers' stored inputs; it rebuilds them. If the rebuild
or the hand-written gradient bookkeeping were wrong, the weights would get wrong gradients while
the loss looked normal (a10 met exactly that with torch.compile). So before training, each
variant runs the same batch twice: once through RevStack (reversible=True) and once with plain
autograd through the same equations (reversible=False), and every parameter's gradient is
compared.

Reported per variant: the largest relative difference over parameters, |g_rev - g_ref| / |g_ref|
(norms over each parameter tensor), the largest reconstruction error of a layer input, and in
training precision how far each gradient is from an fp64 answer on the same weights.
"""

import argparse
import contextlib
import json

import torch

import flops
from model import GPTConfig, token_loss
from rev import VARIANTS, RevConfig, RevGPT


def grads(model, x, y, reversible, amp):
    """Loss and {name: gradient} for one forward/backward."""
    model.zero_grad(set_to_none=True)
    with amp:
        loss = token_loss(model(x, reversible=reversible), y)
    loss.backward()
    return loss.item(), {n: p.grad.detach().cpu().double() for n, p in model.named_parameters()}


def rel(a, b):
    """{name: |a - b| / |b|} for two gradient dicts."""
    return {n: ((a[n] - b[n]).norm() / b[n].norm().clamp_min(1e-30)).item() for n in b}


def check(variant, cfg, dev, dtype, amp, stream, B=4, T=64, seed=0, iters=3):
    """Compare reversible and plain gradients for one variant. Returns a result dict.

    The reference is ordinary autograd through the same equations with float streams (fp64 on
    the CPU check, fp32 otherwise; fixed-point integers cannot carry gradients). In fp64 the two
    should agree to about 1e-14. In training precision both carry rounding error of their own,
    so each is also compared with an fp64 answer computed on the CPU from the same weights.
    Reversibility costs nothing if its error against fp64 is no bigger than plain autograd's.
    """
    torch.manual_seed(seed)
    ref_stream = "fp64" if dtype == torch.float64 else "fp32"
    ref = RevGPT(cfg, RevConfig(variant=variant, iters=iters, stream_dtype=ref_stream)).to(dev, dtype)
    model = RevGPT(cfg, RevConfig(variant=variant, iters=iters, stream_dtype=stream)).to(dev, dtype)
    model.load_state_dict(ref.state_dict())
    x = torch.randint(0, cfg.vocab, (B, T), device=dev)
    y = torch.randint(0, cfg.vocab, (B, T), device=dev)
    l_ref, g_ref = grads(ref, x, y, False, amp)
    l_rev, g_rev = grads(model, x, y, True, amp)
    d = rel(g_rev, g_ref)
    worst = max(d, key=d.get)
    out = {"variant": variant, "stream": stream, "dtype": str(dtype).replace("torch.", ""), "loss_ref": l_ref,
           "loss_rev": l_rev, "max_rel_grad_diff": d[worst], "worst_param": worst, "fp_residual": model.fp_residual}
    with amp:
        rec = model.reconstruction_error(x)
    out.update(max_recon_err=max(rec), recon_err_per_layer=rec)
    if dtype != torch.float64 or dev.type != "cpu":
        m64 = RevGPT(cfg, RevConfig(variant=variant, iters=iters, stream_dtype="fp64")).double()
        m64.load_state_dict({k: v.cpu().double() for k, v in ref.state_dict().items()})
        _, g64 = grads(m64, x.cpu(), y.cpu(), False, contextlib.nullcontext())
        e_ref, e_rev = rel(g_ref, g64), rel(g_rev, g64)
        out.update(plain_vs_fp64=max(e_ref.values()), rev_vs_fp64=max(e_rev.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--euler-iters", type=int, default=3)
    args = ap.parse_args()
    cfg = GPTConfig(layers=args.layers, width=args.width, heads=args.heads)

    dev = flops.device()
    amp_dtype = flops.autocast_dtype(dev, "auto")
    amp_dev = torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype != torch.float32 else contextlib.nullcontext()
    a = str(amp_dtype).replace("torch.", "")
    setups = [("cpu fp64 weights and streams, exact arithmetic up to 1e-16", torch.device("cpu"), torch.float64,
               contextlib.nullcontext(), "fp64"),
              (f"{dev} fp32 weights, {a} autocast, fp32 streams", dev, torch.float32, amp_dev, "fp32"),
              (f"{dev} fp32 weights, {a} autocast, fixed-point streams", dev, torch.float32, amp_dev, "fixed")]
    results = []
    for title, d, dtype, amp, stream in setups:
        print(f"\n== {title} ==")
        print(f"{'variant':<10} {'loss (plain)':>13} {'loss (rev)':>13} {'rev vs plain':>13} {'worst parameter':<26} "
              f"{'recon err':>10} {'plain vs fp64':>14} {'rev vs fp64':>12}")
        for v in VARIANTS:
            r = check(v, cfg, d, dtype, amp, stream, iters=args.euler_iters)
            r["setup"] = title
            results.append(r)
            tail = f" {r['plain_vs_fp64']:>14.2e} {r['rev_vs_fp64']:>12.2e}" if "rev_vs_fp64" in r else ""
            print(f"{v:<10} {r['loss_ref']:>13.6f} {r['loss_rev']:>13.6f} {r['max_rel_grad_diff']:>13.2e} "
                  f"{r['worst_param']:<26} {r['max_recon_err']:>10.2e}" + tail)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
