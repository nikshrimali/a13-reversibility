# Reversible GPT: undo instead of remember

A 19.4M-parameter GPT trained on 50M tokens three ways: plain, reversible at the same batch, and reversible at the largest batch that fits. The reversible backward pass rebuilds each layer's input from its output instead of storing it. Four reversible designs were built and screened (coupling, midpoint, euler, momentum), each checked against ordinary autograd before training.

`guide.md` explains every piece from scratch, for a reader who has never trained a neural network.

## Setup

| | |
|---|---|
| Model | a10's GPT at 10 layers: width 384, 6 heads, context 256, tied embeddings, 4,096-token BPE, **19,382,016 params** |
| Data | FineWeb-Edu, **50M train tokens** (one pass), 0.5M held-out tokens. a10's tokenizer; the held-out file is byte for byte a10's |
| Hardware | Kaggle, 2x NVIDIA T4 (16 GB each). One run per GPU, two runs side by side |
| Precision | fp16 autocast with loss scaling, fp32 weights and AdamW state. Reversible streams between layers stored as fixed-point int64 (see Findings) |
| Batch | runs 1 and 2: 64 x 256 = 16,384 tokens/step, 3,039 steps. Run 3: the largest batch that fits, found by probing |
| Optimizer | AdamW (0.9, 0.95), lr 1e-3, 50 warmup steps, cosine to 1e-4, wd 0.1, clip 1.0 (a10's). Run 3: lr x sqrt(batch / 64), capped at 3e-3, warmup 1.6% of steps |
| Notebooks | `reversibility.ipynb` (everything, one cell per stage, Kaggle / Colab / Mac); `parallelism.ipynb` (six ways to split training across GPUs, checked on CPU) |

## The four reversible designs

All four reuse a10's block unchanged (LayerNorm, attention, LayerNorm, MLP) and so have exactly the plain GPT's 19,382,016 parameters. A(x) = attn(ln1(x)), M(x) = mlp(ln2(x)), F(x) = the whole block's correction.

| variant | forward | undo |
|---|---|---|
| coupling (RevNet / Reformer) | y1 = x1 + A(x2), y2 = x2 + M(y1) | x2 = y2 - M(y1), x1 = y1 - A(x2) |
| midpoint (leapfrog) | x[n+1] = x[n-1] + F(x[n]) | x[n-1] = x[n+1] - F(x[n]) |
| euler (the plain GPT) | x[n+1] = x[n] + F(x[n]) | 3 fixed-point repeats of x = y - F(x) |
| momentum (MomentumNet) | v' = 0.9 v + 0.1 F(x), x' = x + v' | x = x' - v', v = (v' - 0.1 F(x)) / 0.9 |

`RevStack` (in `rev.py`) runs the stack keeping only the final state. Its backward walks the layers from the top, recomputes each block once with autograd on, and uses that one evaluation both to rebuild the layer's input and to push the gradient through. Extra work: one forward pass of the blocks per step (euler: four).

## Results

*Pending the Kaggle run. This section is filled in only from the executed `reversibility.ipynb`; no number here will be estimated.*

## Findings from building it (Mac, before Kaggle)

These come from the gradient check and the smoke run on a MacBook Air M4 (bf16); the files are in `runs/mac/`. The notebook rechecks each one on the T4.

- **fp32 streams do not undo exactly; fixed-point streams do.** (x + f) - f in fp32 can be off by one last bit. The rebuilt input then differs by about 1e-7, which inside a bf16/fp16 block can flip a value to the next 16-bit step, and that error compounds down the stack. After 18 training steps, midpoint with fp32 streams rebuilt the first layer's input 38% off; coupling 4%. Storing the streams as int64 counts of 2^-24 and rounding each correction to that grid makes every add and subtract exact: coupling and midpoint then rebuild every input bit for bit (error exactly 0), and their gradients are as close to an fp64 answer as plain autograd's (ratio 0.96 to 0.99). MacKay et al. (2018) used fixed point for reversible RNNs.
- **Euler cannot be undone on this model.** Its fixed-point repeats diverge: LayerNorm divides by the spread of its input, which is about 0.03 at initialization, so the block is steep and two inputs can give one output. At initialization its rebuilt input was off by 209% (fp64) and its bf16 gradients were 174x further from the fp64 answer than plain autograd's.
- **Momentum drifts.** Its undo divides by 0.9 per layer, so rounding grows 2.87x over 10 layers, fixed point or not. Its bf16 gradient error was 3x plain autograd's.
- **Apple GPUs compute attention differently with and without autograd.** On MPS, `scaled_dot_product_attention` on bf16 inputs gives different results (5e-4) depending on whether gradients are being recorded. The forward ran without autograd and the recompute with it, so the undo was off by bf16 rounding. `same_path` in `rev.py` runs the forward blocks with autograd on and detaches their outputs straight away: same kernels both ways, and still one layer's activations in memory at a time.
- **a10's saved-tensor hook leaked the measured step's whole graph.** Returning the tensor itself from a `saved_tensors_hooks` pack function creates a cycle (tensor, its grad_fn, the saved copy) that Python's garbage collector does not collect: 481 MB per measured step at batch 16. Returning `t.detach()` fixes it; gradients are bit for bit the same.
- **Single-step loss spikes, in every model.** At lr 1e-3 with a fast warmup, the plain GPT and every reversible variant have the odd step where the loss jumps by 1 to 4 and recovers on the next step. The screening therefore uses a 50-step warmup and ranks variants by the average of their last three held-out measurements.

## Files

| file | what | from |
|---|---|---|
| `guide.md` | the walkthrough: models, gradients, memory, reversibility by hand, rounding, parallelism, then every result explained | new |
| `reversibility.ipynb` | the whole experiment, one cell per stage | new |
| `parallelism.ipynb` | data, tensor, pipeline, sequence, context and expert parallelism, each checked on small numbers | new |
| `rev.py` | `RevGPT` (four variants, fixed-point or float streams) and `RevStack`, the hand-written backward | new |
| `train.py` | the training loop: loss, grad norm, tok/s, MFU, HFU, peak memory, reconstruction error, bench / OOM probe mode, resume | a10's loop, extended |
| `gradcheck.py` | each variant's gradients against autograd (fp64) and against fp64 truth (training precision) | new |
| `precision.py` | fp32 / fp16 / bf16 speed; reconstruction drift by stream format | new |
| `probe.py` | largest batch that fits: double, then bisect; one process per try | new |
| `jobs.py` | one run per GPU at a time; skip finished runs, resume unfinished ones | new |
| `plot.py` | the figures in `plots/` | new |
| `parallel_checks.py` | the code behind `parallelism.ipynb` | new |
| `model.py` | the GPT | a10, unchanged |
| `data.py` | FineWeb-Edu download and tokenization | a10, plus `--tokenizer` |
| `flops.py` | device, dtype, MFU, memory bill, saved-tensor list | a10, plus peak memory, HFU, reversible bill, leak fix |
| `tokenizer.json` | a10's 4,096-token BPE | a10 |

```bash
uv sync
uv run data.py --out data/fwe50 --train-tokens 50M --eval-tokens 500K --tokenizer tokenizer.json
uv run gradcheck.py                                    # every variant against autograd, ~10 s
uv run jupyter notebook reversibility.ipynb            # set SMOKE = True for a 32-minute check on an M4
```
