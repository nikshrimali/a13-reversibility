"""Reversible versions of a10's GPT: the same weights, wired so each layer's input can be rebuilt from its output.

    from rev import RevGPT, RevConfig
    model = RevGPT(GPTConfig(layers=10), RevConfig(variant="coupling"))
    logits = model(x)                      # trains keeping only the last layer's output
    logits = model(x, reversible=False)    # same equations, ordinary autograd (stores everything)

A normal network keeps every layer's input for the backward pass, because backward needs it.
A reversible network throws those inputs away and, during backward, computes each one from the
layer's output instead: "if I know you added 5, I can subtract 5". Only the last layer's output
is kept, so activation memory no longer grows with the number of layers.

Every variant uses a10's Block unchanged (LayerNorm, attention, LayerNorm, MLP) and so has
exactly the same parameters as the plain GPT. What changes is how the pieces are wired. Write
A(x) = attn(ln1(x)), M(x) = mlp(ln2(x)), and F(x) = the whole block's correction, A(x) + M(x + A(x)).

    coupling   two streams.  y1 = x1 + A(x2),  y2 = x2 + M(y1)            (Reformer / RevNet)
               undo:         x2 = y2 - M(y1),  x1 = y1 - A(x2)            exact
    midpoint   two states.   x[n+1] = x[n-1] + 2h F(x[n])                  (leapfrog)
               undo:         x[n-1] = x[n+1] - 2h F(x[n])                  exact
    euler      one state.    x[n+1] = x[n] + F(x[n])                        (the plain GPT itself)
               undo:         solve x = y - F(x) by repeating it `iters` times, approximate
    momentum   x and speed v. v' = g v + (1-g) F(x),  x' = x + v'           (MomentumNet)
               undo:         x = x' - v',  v = (v' - (1-g) F(x)) / g        exact on paper, /g grows rounding

How the streams are stored decides whether "exact" survives real arithmetic (`stream_dtype`):

    fp32, fp16, bf16   floats. (x + f) - f is not always x: the last bit can change. The rebuilt
                       x then differs from the true x by about one rounding step, the block sees
                       a slightly different input, and inside the block (which runs in fp16 or
                       bf16) that tiny difference can flip a value to the next fp16/bf16 step. The
                       error grows layer by layer; reconstruction_error() measures it.
    fixed              integers. Each stream value is stored as round(value x 2^24) in int64, and
                       each correction f is rounded to the same grid before it is added. Integer
                       addition and subtraction are exact, so coupling and midpoint rebuild every
                       input bit for bit, and the recomputed blocks see exactly what forward saw.
                       The grid step, 2^-24 = 6e-8, is finer than fp32's own step for any value
                       above 1. Momentum's division by g is still rounded, so it stays approximate.
                       (MacKay et al. 2018 used the same trick for reversible RNNs.)
"""

import contextlib
from dataclasses import dataclass

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from model import GPT

VARIANTS = ("coupling", "midpoint", "euler", "momentum")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "fp64": torch.float64}
FIXED_SCALE = 2.0**24  # fixed-point grid: one unit of the stored integer is 2^-24


@dataclass
class RevConfig:
    """How the reversible network is wired.

    Attributes:
        variant: one of VARIANTS.
        h: midpoint step size. 0.5 makes 2h = 1, so each layer adds F at full strength, like the plain GPT.
        gamma: momentum's memory of its previous speed. MomentumNet's default, 0.9.
        iters: euler fixed-point repeats used to undo a layer. Each costs one extra block forward.
        stream_dtype: how streams are stored between layers: "fixed", "fp32", "fp16", "bf16" or "fp64".
    """

    variant: str = "coupling"
    h: float = 0.5
    gamma: float = 0.9
    iters: int = 3
    stream_dtype: str = "fp32"


def attention_backend(dev):
    """Pin one attention kernel on CUDA so forward and backward's recompute always run the same code.

    The memory-efficient kernel runs on every CUDA GPU from the T4 on. Elsewhere this does nothing.
    """
    if dev.type == "cuda":
        return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    return contextlib.nullcontext()


class RevGPT(GPT):
    """a10's GPT with the blocks run as a reversible stack. Same parameters, same names, same init."""

    def __init__(self, cfg, rcfg):
        super().__init__(cfg)
        assert rcfg.variant in VARIANTS, rcfg.variant
        self.rcfg = rcfg
        self.fixed = rcfg.stream_dtype == "fixed"
        self.sd = None if self.fixed else DTYPES[rcfg.stream_dtype]
        self.plain = False  # True: train with ordinary autograd, a diagnostic (train.py --plain)
        self.fp_residual = None  # euler only: how far the last backward's undo missed, see _rev_euler

    # ---- how a stream is stored --------------------------------------------------------------

    def store(self, t):
        """Turn a float tensor into the stored form of a stream."""
        if self.fixed:
            return torch.round(t.float() * FIXED_SCALE).long()
        return t.to(self.sd)

    def value(self, s):
        """The float value of a stored stream: what the blocks and the output head read."""
        if self.fixed:
            return s.float() / FIXED_SCALE
        return s

    def add(self, s, f, sign=1):
        """s + sign * f, where s is a stored stream and f a float correction from a block.

        With fixed point, f is rounded to the grid first and the add is an integer add, so
        add(add(s, f), f, -1) gives back s bit for bit.
        """
        if self.fixed:
            q = torch.round(f.float() * FIXED_SCALE).long()
            return s + q if sign > 0 else s - q
        return (s + f if sign > 0 else s - f).to(self.sd)

    # ---- the pieces of one block -------------------------------------------------------------

    @staticmethod
    def _A(blk, x):
        return blk.attn(blk.ln1(x))

    @staticmethod
    def _M(blk, x):
        return blk.mlp(blk.ln2(x))

    def _F(self, blk, x):
        """The whole block's correction: what the plain GPT adds to x in one block."""
        a = self._A(blk, x)
        return a + self._M(blk, x + a)

    # ---- start and end of the stack ----------------------------------------------------------

    def init_state(self, h0):
        """Turn the embedding output into the starting state of the stack (stored form)."""
        v = self.rcfg.variant
        s = self.store(h0)
        if v in ("coupling", "midpoint"):
            return (s, s.clone())  # midpoint: x[-1] = x[0], start at rest
        if v == "momentum":
            return (s, torch.zeros_like(s))  # speed 0 before the first layer
        return (s,)

    def init_grad(self, grads):
        """Gradient of the starting state, folded back onto the embedding output."""
        if self.rcfg.variant in ("coupling", "midpoint"):
            return grads[0] + grads[1]
        return grads[0]  # momentum: v0 is a constant zero; euler: one state

    def readout(self, vals):
        """The vector the output head reads, from the final state's float values."""
        if self.rcfg.variant == "coupling":
            return (vals[0] + vals[1]) / 2
        return vals[1] if self.rcfg.variant == "midpoint" else vals[0]

    # ---- one layer forward, and its undo -----------------------------------------------------

    def step(self, blk, s):
        """Run one layer forward on stored state s. Returns the stored output state."""
        v, val, add = self.rcfg.variant, self.value, self.add
        if v == "coupling":
            x1, x2 = s
            y1 = add(x1, self._A(blk, val(x2)))
            return (y1, add(x2, self._M(blk, val(y1))))
        if v == "midpoint":
            p, cur = s
            return (cur, add(p, 2 * self.rcfg.h * self._F(blk, val(cur))))
        if v == "momentum":
            x, vel = s
            g = self.rcfg.gamma
            vel = self.store(g * val(vel) + (1 - g) * self._F(blk, val(x)))
            return (add(x, val(vel)), vel)
        (x,) = s
        return (add(x, self._F(blk, val(x))),)

    def invert(self, blk, s):
        """Rebuild one layer's input state from its output state, without gradients."""
        v, val, add = self.rcfg.variant, self.value, self.add
        if v == "coupling":
            y1, y2 = s
            x2 = add(y2, self._M(blk, val(y1)), -1)
            return (add(y1, self._A(blk, val(x2)), -1), x2)
        if v == "midpoint":
            cur, nxt = s
            return (add(nxt, 2 * self.rcfg.h * self._F(blk, val(cur)), -1), cur)
        if v == "momentum":
            x2, v2 = s
            g = self.rcfg.gamma
            x = add(x2, val(v2), -1)
            return (x, self.store((val(v2) - (1 - g) * self._F(blk, val(x))) / g))
        (y,) = s
        x = y
        for _ in range(self.rcfg.iters):
            x = add(y, self._F(blk, val(x)), -1)
        return (x,)

    def reverse(self, blk, s, d):
        """Backward through one layer without its stored input.

        Rebuilds the layer's input from its output s, and turns the gradient d of the output into
        the gradient of the input. The weights' gradients are added to their .grad on the way.
        Each block piece is evaluated once with autograd on, and that one evaluation serves both
        the undo and the gradient, so the extra cost is one forward of the blocks.

        Args:
            blk: the Block.
            s: output state, stored form.
            d: gradient of the loss with respect to each stream's float value (fp32).

        Returns:
            (input state in stored form, gradient with respect to the input state).
        """
        v, val, add = self.rcfg.variant, self.value, self.add

        def run(fn, stored):
            """Evaluate fn on a fresh leaf holding the stream's value, autograd on. Returns (leaf, output)."""
            leaf = val(stored).detach().requires_grad_()
            with torch.enable_grad():
                out = fn(blk, leaf)
                # Upcast inside the graph so the output's gradient arrives in fp32.
                return leaf, out.to(torch.promote_types(out.dtype, torch.float32))

        def back(out, grad, leaf):
            """Push grad back through out; return the gradient that reached leaf."""
            torch.autograd.backward(out, grad)
            return leaf.grad

        if v == "coupling":
            (y1, y2), (d1, d2) = s, d
            y1r, m = run(self._M, y1)
            x2 = add(y2, m.detach(), -1)
            dz1 = d1 + back(m, d2, y1r)
            x2r, a = run(self._A, x2)
            x1 = add(y1, a.detach(), -1)
            dx2 = d2 + back(a, dz1, x2r)
            return (x1, x2), (dz1, dx2)
        if v == "midpoint":
            (cur, nxt), (dcur, dnxt) = s, d
            k = 2 * self.rcfg.h
            cr, f = run(self._F, cur)
            p = add(nxt, k * f.detach(), -1)
            return (p, cur), (dnxt, dcur + back(f, k * dnxt, cr))
        if v == "momentum":
            (x2, v2), (dx2, dv2) = s, d
            g = self.rcfg.gamma
            x = add(x2, val(v2), -1)
            xr, f = run(self._F, x)
            vel = self.store((val(v2) - (1 - g) * f.detach()) / g)
            dvt = dv2 + dx2  # v' reaches the loss directly and through x' = x + v'
            return (x, vel), (dx2 + back(f, (1 - g) * dvt, xr), g * dvt)
        return self._rev_euler(blk, s, d, run, back)

    def _rev_euler(self, blk, s, d, run, back):
        """Euler's backward: undo by fixed-point repeats, then a normal gradient at the rebuilt input.

        Also records how badly the undo missed: |x + F(x) - y| / |y| for the worst layer, in
        self.fp_residual. Zero would mean x really is the input that produced y.
        """
        (y,), (dy,) = s, d
        (x,) = self.invert(blk, s)
        xr, f = run(self._F, x)
        yv = self.value(y).float()
        miss = ((self.value(x).float() + f.detach()) - yv).norm() / yv.norm().clamp_min(1e-30)
        self.fp_residual = miss.item() if self.fp_residual is None else max(self.fp_residual, miss.item())
        return (x,), (dy + back(f, dy, xr),)

    # ---- the model ---------------------------------------------------------------------------

    def embed(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).expand(B, T)  # (B, T) grid, see a10's model.py
        return self.tok_emb(idx) + self.pos_emb(pos)

    def forward(self, idx, trace=None, reversible=True):
        """Next-token logits for every position.

        Args:
            idx: (B, T) token ids.
            trace: unused, accepted so train.py can call every model the same way.
            reversible: True runs the stack through RevStack (stores only the final state);
                False runs the same equations with ordinary autograd (stores every layer).

        Returns:
            (B, T, V) logits.
        """
        h0 = self.embed(idx)
        with attention_backend(idx.device):
            if reversible and not self.plain and torch.is_grad_enabled():
                vals = RevStack.apply(self, h0)
            else:
                if self.fixed and torch.is_grad_enabled():
                    raise RuntimeError("fixed-point streams are integers; ordinary autograd cannot pass through them")
                state = self.init_state(h0)
                for blk in self.blocks:
                    state = self.step(blk, state)
                vals = tuple(self.value(s) for s in state)
        x = self.readout(vals).to(self.ln_f.weight.dtype)
        return self.lm_head(self.ln_f(x))

    def reconstruction_error(self, idx):
        """How far the undo drifts from the truth, per layer.

        Runs forward keeping every layer's true state, then undoes the layers one by one from the
        final state, as backward does, and compares. Both directions run with autograd on and
        their outputs detached, which is how RevStack runs them (see same_path).

        Returns:
            A list with one number per layer input, from the last layer down to the first:
            |rebuilt state - true state| / |true state|, with the state's tensors taken together
            (momentum's starting speed is exactly zero, so it cannot be measured on its own).
        """
        with attention_backend(idx.device):
            states = [self.init_state(self.embed(idx).detach())]
            for blk in self.blocks:
                states.append(same_path(self.step, blk, states[-1]))
            s, errs = states[-1], []
            for i in range(len(self.blocks) - 1, -1, -1):
                s = same_path(self.invert, self.blocks[i], s)
                errs.append(_rel_err(s, states[i]))
        return errs


def same_path(fn, blk, state):
    """Run one layer step or undo with autograd on, then cut the result loose from the graph.

    Some kernels compute slightly different bits depending on whether autograd is recording: on
    Apple GPUs, attention on bf16 inputs differs in the 4th digit between the two modes (found
    while building this, see the guide). Backward recomputes each block with autograd on, so
    forward must run it the same way, or even integer streams would rebuild a different input.
    Detaching the outputs drops the graph straight away, so memory holds one layer at a time.
    """
    with torch.enable_grad():
        return tuple(t.detach() for t in fn(blk, state))


def _rel_err(rebuilt, true):
    """|rebuilt - true| / |true| over all tensors of a state, computed in fp64 on the CPU."""
    num = sum(((r.detach().cpu().double() - t.detach().cpu().double()) ** 2).sum() for r, t in zip(rebuilt, true))
    den = sum((t.detach().cpu().double() ** 2).sum() for t in true)
    return (num.sqrt() / den.sqrt().clamp_min(1e-300)).item()


class RevStack(torch.autograd.Function):
    """The whole stack of blocks as one autograd node that saves only its output.

    forward runs every layer and keeps none of its insides (same_path drops each layer's graph
    as soon as the layer is done). backward walks the layers from last to first, rebuilding each
    layer's input from its output (RevGPT.reverse). The weights are not inputs of this node:
    their gradients are written straight into .grad inside backward, so the only gradient this
    node returns is the embedding output's.

    The node's outputs are the final streams' float values (integers cannot carry gradients);
    the stored form, which may be fixed-point integers, is what it saves for backward.
    """

    @staticmethod
    def forward(ctx, model, h0):
        dev = h0.device.type
        ctx.model = model
        ctx.amp = (torch.is_autocast_enabled(dev), torch.get_autocast_dtype(dev), dev)
        state = model.init_state(h0.detach())
        for blk in model.blocks:
            state = same_path(model.step, blk, state)
        ctx.save_for_backward(*state)
        return tuple(model.value(s) for s in state)

    @staticmethod
    def backward(ctx, *grads):
        model = ctx.model
        state = ctx.saved_tensors
        grads = tuple(torch.zeros(s.shape, device=s.device) if g is None else g.float() for s, g in zip(state, grads))
        enabled, dtype, dev = ctx.amp
        model.fp_residual = None
        # Autocast is not active inside backward by default; switch it back on as forward had it,
        # so the recomputed blocks run in the same precision and give the same bits.
        with torch.autocast(dev, dtype=dtype, enabled=enabled), attention_backend(state[0].device):
            for blk in reversed(model.blocks):
                state, grads = model.reverse(blk, state, grads)
        return None, model.init_grad(grads)


def build(cfg, arch, **rkw):
    """Make the model train.py asks for: arch "gpt" is a10's plain GPT, anything else a RevGPT variant."""
    if arch == "gpt":
        return GPT(cfg)
    return RevGPT(cfg, RevConfig(variant=arch, **rkw))
