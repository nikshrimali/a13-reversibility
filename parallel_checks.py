"""Six ways to split training across GPUs, each checked on small numbers against doing it on one device.

    uv run parallel_checks.py          # all six, prints every number; runs on the CPU in seconds

Nothing here needs a GPU. Each "GPU" is a slice of a tensor, and each message between GPUs is
an ordinary Python assignment that the code counts in bytes. The point is to show that every
split computes exactly what one device would, and to put a number on what it saves and what it
sends. parallelism.ipynb runs these one per cell, with the explanation in between; guide.md
Part 10 is the long version.

    data        each GPU gets different examples; gradients are averaged          (a12 goes deep)
    tensor      one layer's weight matrix is cut into column or row pieces
    pipeline    different layers live on different GPUs; micro-batches flow through
    sequence    the token-by-token parts of a layer (LayerNorm) are split along the sentence
    context     attention itself is split along the sentence; keys and values travel round a ring
    expert      each expert of a mixture-of-experts lives on its own GPU; tokens travel to it  (a14)
"""

import math

import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
DT = torch.float64  # exact enough that "equal" means equal to 1e-15


def show(name, a, b):
    """Print the largest difference between the split answer and the one-device answer."""
    d = (a - b).abs().max().item()
    print(f"  {name}: largest difference {d:.1e} -> {'SAME' if d < 1e-12 else 'DIFFERENT'}")
    return d


def nbytes(*ts):
    return sum(t.numel() * 4 for t in ts)  # counted as fp32, 4 bytes per number


# ---------------------------------------------------------------------------------------------
def data_parallel():
    """2 GPUs, each with a full copy of the model and half the batch. Average the gradients."""
    print("\n== Data parallelism: split the batch ==")
    torch.manual_seed(0)
    W = torch.tensor([[0.5, -1.0], [2.0, 0.25]], dtype=DT)
    X = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 1.0], [2.0, 2.0]], dtype=DT)  # 4 examples
    Y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=DT)

    def grad(x, y):
        w = W.clone().requires_grad_()
        ((x @ w - y) ** 2).mean().backward()
        return w.grad

    g_full = grad(X, Y)
    g0, g1 = grad(X[:2], Y[:2]), grad(X[2:], Y[2:])  # GPU 0: examples 0-1, GPU 1: examples 2-3
    g_avg = (g0 + g1) / 2  # the all-reduce: every GPU ends with this average
    print("  gradient on one device:\n", g_full)
    print("  GPU 0's gradient (examples 0,1):\n", g0)
    print("  GPU 1's gradient (examples 2,3):\n", g1)
    print("  their average:\n", g_avg)
    d = show("average vs one device", g_avg, g_full)
    print(f"  each GPU holds: the whole model ({nbytes(W)} bytes) and half the examples; "
          f"sends its whole gradient ({nbytes(g0)} bytes) to be averaged, every step")
    return {"diff": d}


# ---------------------------------------------------------------------------------------------
def tensor_parallel():
    """An MLP: widen with W1 (split by columns), GELU, narrow with W2 (split by rows), one all-reduce."""
    print("\n== Tensor parallelism: split one layer's weights ==")
    torch.manual_seed(0)
    x = torch.tensor([[1.0, 2.0]], dtype=DT)          # one token, width 2
    W1 = torch.arange(1, 9, dtype=DT).view(2, 4) / 10  # widen 2 -> 4
    W2 = torch.arange(1, 9, dtype=DT).view(4, 2) / 10  # narrow 4 -> 2
    full = F.gelu(x @ W1) @ W2

    # GPU 0 owns columns 0-1 of W1 and rows 0-1 of W2; GPU 1 owns the other halves.
    h0 = F.gelu(x @ W1[:, :2])   # GPU 0's two hidden numbers, computed alone
    h1 = F.gelu(x @ W1[:, 2:])   # GPU 1's two hidden numbers, computed alone
    p0, p1 = h0 @ W2[:2], h1 @ W2[2:]  # each GPU's partial output
    out = p0 + p1                       # the one all-reduce: add the partial outputs
    r = lambda t: [round(v, 4) for v in t.flatten().tolist()]
    print("  x:", r(x), " W1:", W1.tolist(), " W2:", W2.tolist())
    print("  GPU 0 hidden (columns 0,1):", r(h0), "  GPU 1 hidden (columns 2,3):", r(h1))
    print("  GPU 0 partial output:", r(p0), "  GPU 1 partial output:", r(p1))
    print("  sum:", r(out), "  one device:", r(full))
    d = show("split MLP vs one device", out, full)
    print(f"  each GPU holds half of W1 and W2 ({nbytes(W1, W2) // 2} of {nbytes(W1, W2)} bytes); "
          f"sends its partial output ({nbytes(p0)} bytes per token) once per layer, forward and backward")
    return {"diff": d}


# ---------------------------------------------------------------------------------------------
def pipeline_parallel(stages=2, micro=4):
    """4 layers on 2 GPUs (2 layers each); the batch cut into micro-batches that flow through."""
    print("\n== Pipeline parallelism: split the layers ==")
    torch.manual_seed(0)
    Ws = [torch.randn(3, 3, dtype=DT) * 0.5 for _ in range(4)]
    X = torch.randn(8, 3, dtype=DT)

    def run(layers, x):
        for w in layers:
            x = torch.tanh(x @ w)
        return x

    full = run(Ws, X)
    gpu0, gpu1 = Ws[:2], Ws[2:]
    outs, sent = [], 0
    for mb in X.chunk(micro):           # micro-batches of 2 examples
        a = run(gpu0, mb)               # GPU 0 runs layers 0-1 ...
        sent += nbytes(a)               # ... and sends the activations to GPU 1
        outs.append(run(gpu1, a))       # GPU 1 runs layers 2-3
    d = show("pipelined output vs one device", torch.cat(outs), full)
    # The schedule (GPipe, forward only): time slot t, GPU s works on micro-batch t - s.
    print(f"  forward schedule, {stages} GPUs x {micro} micro-batches (number = micro-batch, . = idle):")
    slots = micro + stages - 1
    for s in range(stages):
        row = "".join(f" {t - s}" if 0 <= t - s < micro else " ." for t in range(slots))
        print(f"    GPU {s}:{row}")
    bubble = (stages - 1) / (micro + stages - 1)
    print(f"  idle share (the 'bubble') = (GPUs - 1) / (micro-batches + GPUs - 1) = {stages - 1}/{slots} = {bubble:.0%}")
    print(f"  each GPU holds 2 of 4 layers; GPU 0 sent {sent} bytes of activations this step")
    return {"diff": d, "bubble": bubble}


# ---------------------------------------------------------------------------------------------
def sequence_parallel():
    """LayerNorm looks at one token at a time, so each GPU can normalize its own half of the sentence."""
    print("\n== Sequence parallelism: split the sentence for the token-by-token parts ==")
    x = torch.tensor([[1.0, 2.0, 3.0, 6.0], [0.0, 0.0, 4.0, 4.0], [2.0, 2.0, 2.0, 10.0], [5.0, 1.0, 1.0, 1.0]],
                     dtype=DT)  # 4 tokens, width 4
    full = F.layer_norm(x, (4,))
    gpu0, gpu1 = F.layer_norm(x[:2], (4,)), F.layer_norm(x[2:], (4,))  # tokens 0-1, tokens 2-3
    print("  token 0 by hand: mean = (1+2+3+6)/4 = 3, spread = sqrt(((1-3)^2+(2-3)^2+(3-3)^2+(6-3)^2)/4) = sqrt(3.5)")
    print(f"    -> ({1 - 3}, {2 - 3}, {3 - 3}, {6 - 3}) / {math.sqrt(3.5):.4f} = {[round(v, 4) for v in full[0].tolist()]}")
    d = show("two halves vs whole sentence", torch.cat([gpu0, gpu1]), full)
    print(f"  each GPU stores the LayerNorm's saved input for half the tokens ({nbytes(x) // 2} of {nbytes(x)} bytes); "
          "attention still needs every token, so the halves are gathered before it and split again after")
    return {"diff": d}


# ---------------------------------------------------------------------------------------------
def context_parallel(T=8, D=2, gpus=2):
    """Causal attention with the sentence split across GPUs; key/value chunks travel round a ring.

    Softmax needs every score of a row before it can divide, but a GPU only sees one chunk of
    keys at a time. The fix is the running ("online") softmax: keep, per query, the largest score
    so far (m), the sum of exp(score - m) so far (l) and the weighted sum of values so far (acc).
    When a new chunk arrives with a bigger maximum, rescale what you have by exp(old m - new m).
    """
    print("\n== Context parallelism: split attention itself along the sentence ==")
    torch.manual_seed(0)
    q, k, v = (torch.randn(T, D, dtype=DT) for _ in range(3))
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    scores = (q @ k.T) / math.sqrt(D)
    full = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v

    c = T // gpus
    out, sent = [], 0
    for g in range(gpus):                       # GPU g owns queries, keys and values of tokens g*c .. g*c+c-1
        qg = q[g * c:(g + 1) * c]
        m = torch.full((c, 1), float("-inf"), dtype=DT)
        l = torch.zeros(c, 1, dtype=DT)
        acc = torch.zeros(c, D, dtype=DT)
        for step in range(gpus):                # ring: at step s, GPU g holds the K/V chunk of GPU (g - s) mod gpus
            src = (g - step) % gpus
            if src > g:                         # causal: later tokens' keys are never needed
                continue
            if step:
                sent += nbytes(k[:c], v[:c])    # this chunk arrived from the neighbour
            kc, vc = k[src * c:(src + 1) * c], v[src * c:(src + 1) * c]
            s = (qg @ kc.T) / math.sqrt(D)
            if src == g:
                s = s.masked_fill(~torch.ones(c, c, dtype=torch.bool).tril(), float("-inf"))
            m_new = torch.maximum(m, s.max(dim=1, keepdim=True).values)
            scale = torch.exp(m - m_new)        # shrink the old totals to the new maximum
            p = torch.exp(s - m_new)
            l = l * scale + p.sum(dim=1, keepdim=True)
            acc = acc * scale + p @ vc
            m = m_new
        out.append(acc / l)
    d = show("ring attention vs one device", torch.cat(out), full)
    print(f"  each GPU holds queries, keys and values for {c} of {T} tokens, and never builds the full "
          f"{T} x {T} score table; key/value bytes passed round the ring: {sent}")
    return {"diff": d}


# ---------------------------------------------------------------------------------------------
def expert_parallel():
    """Top-1 mixture of experts with expert 0 on GPU 0 and expert 1 on GPU 1; tokens go to their expert."""
    print("\n== Expert parallelism: each expert on its own GPU ==")
    torch.manual_seed(0)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]], dtype=DT)  # 4 tokens
    router = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=DT)
    experts = [torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=DT), torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=DT)]
    choice = (x @ router).argmax(dim=1)                                    # which expert each token goes to
    full = torch.stack([x[i] @ experts[choice[i]] for i in range(len(x))])  # one device: all experts at hand
    out = torch.empty_like(x)
    sent = 0
    for e in range(2):                                                     # all-to-all: send each token to its expert's GPU
        idx = (choice == e).nonzero().flatten()
        sent += 2 * nbytes(x[idx])                                         # there, and the answer back
        out[idx] = x[idx] @ experts[e]
    print(f"  router picks expert {choice.tolist()} for tokens 0..3")
    d = show("experts on separate GPUs vs one device", out, full)
    print(f"  each GPU holds one expert ({nbytes(experts[0])} of {nbytes(*experts)} bytes of expert weights); "
          f"tokens sent there and back: {sent} bytes")
    return {"diff": d}


def main():
    results = {f.__name__: f() for f in (data_parallel, tensor_parallel, pipeline_parallel, sequence_parallel,
                                         context_parallel, expert_parallel)}
    print("\nall six give the same answer as one device:", all(r["diff"] < 1e-12 for r in results.values()))
    return results


if __name__ == "__main__":
    main()
