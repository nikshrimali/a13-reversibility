# Reversible GPT: undo instead of remember

A 19.4M-parameter GPT trained on 50M tokens three ways: plain, reversible at the same batch, and reversible at the largest batch that fits. The reversible backward pass rebuilds each layer's input from its output instead of storing it. Four reversible designs were built and screened (coupling, midpoint, euler, momentum), each checked against ordinary autograd before training.

`guide.md` explains every piece from scratch, for a reader who has never trained a neural network.

## Setup

| | |
|---|---|
| Model | a10's GPT at 10 layers: width 384, 6 heads, context 256, tied embeddings, 4,096-token BPE, **19,382,016 params** |
| Data | FineWeb-Edu, **50M train tokens** (one pass), 0.5M held-out tokens. a10's tokenizer; the held-out file is byte for byte a10's |
| Hardware | Google Colab, 1x NVIDIA T4 (15.6 GB), one run at a time. The notebook also runs on Kaggle's 2x T4, one run per GPU |
| Precision | fp16 autocast with loss scaling, fp32 weights and AdamW state. Reversible streams between layers stored as fixed-point int64 (see Findings) |
| Batch | runs 1 and 2: 64 x 256 = 16,384 tokens/step, 3,039 steps. Run 3: the largest batch that fits, found by probing (888 x 256, 219 steps) |
| Optimizer | AdamW (0.9, 0.95), lr 1e-3, 50 warmup steps, cosine to 1e-4, wd 0.1, clip 1.0 (a10's). Run 3: lr x sqrt(batch / 64), capped at 3e-3, warmup 1.6% of steps. This schedule broke run 3, see Results |
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

Every number here is copied from the executed `reversibility.ipynb`: Google Colab, one T4, torch 2.14.1, November 2026. None is estimated.

| run | batch | steps | final held-out loss | tok/s | peak memory | MFU | HFU | job time |
|---|---|---|---|---|---|---|---|---|
| 1. plain GPT | 64 | 3,039 | **3.5421** | 85,047 | 3.54 GB | 15.2% | 15.2% | 10.7 min |
| 2. midpoint, fixed-point streams | 64 | 3,039 | 3.6328 | 56,523 | **1.33 GB** | 10.1% | 13.2% | 15.7 min |
| 3. midpoint, largest batch | 888 | 219 | 6.0816 | 57,028 | 15.21 GB | 10.2% | 13.3% | 15.4 min |

Speed, memory, MFU and HFU come from separate 100-step bench runs, each alone on the GPU. Job time is the whole training job, evals included. Peak memory is what tensors held, as `torch.cuda.max_memory_allocated` reports it.

The reversible model does what it promises on memory and pays for it in speed. At the same batch it needs 2.7x less memory and runs 1.5x slower. Its loss ends 0.091 behind the plain GPT. Run 3 failed, and the cause is its learning-rate schedule, not reversibility.

### Memory

- At batch 64, autograd saves 410 MB for the reversible model and 2,617 MB for the plain one, 6.4x less. Peak memory drops from 3.54 GB to 1.33 GB.
- The largest batch that fits is 296 for the plain GPT and 888 for the reversible one, 3.0x more. `probe.py` finds these with 3 training steps per try. A 100-step bench of the plain GPT at 296 then ran out of memory and fit at 288 (14.91 GB peak), so 296 is right at the edge.
- Of the reversible model's 410 MB, 268 MB is the fp32 log-probabilities cross-entropy keeps for backward. The 10 blocks store only two int64 streams, 101 MB. Computing the loss in chunks would be the next large saving.

### Speed

- The reversible model trains at 56,523 tok/s against 85,047, 34% fewer. Rebuilding each layer means one extra forward pass of the blocks, which adds 31% to the arithmetic per token (HFU 13.2% over MFU 10.1%). That covers most of the gap.
- Fixed-point streams cost 10%. The same model with fp32 streams runs at 62,943 tok/s, but fp32 streams do not undo exactly (see Checks below).
- A bigger batch speeds up the plain GPT by 13% (95,779 tok/s at 288) and the reversible one by under 1% (57,028 at 888). The reversible model is already limited by arithmetic at batch 64.

### Screening, 5M tokens each at batch 64

A variant is eligible when its gradient error against fp64 is at most 1.5x plain autograd's and no layer's rebuilt input is off by more than 1e-3. The lowest held-out loss wins, and among variants within 0.02 of it, the fastest.

| run | held-out loss, last 3 avg | tok/s | gradient error / plain | worst undo error | eligible |
|---|---|---|---|---|---|
| plain GPT | 5.3451 | 88,919 | 1.00 | - | reference |
| coupling, fixed point | 5.5284 | 53,489 | 0.97 | 0 | yes |
| coupling, fp32 | 5.5202 | 64,724 | 1.24 | 2.9e-2 | no |
| **midpoint, fixed point** | **5.3441** | 57,494 | 0.95 | 0 | **winner** |
| midpoint, fp32 | 5.3471 | 63,466 | 1.02 | 6.2e-2 | no |
| momentum, fp32 | 5.0994 | 60,623 | 0.99 | 6.4e-3 | no |
| euler, fp32 | 6.9230 | 38,632 | 379.72 | 3.6e+3 | no |

Momentum had the lowest loss of all seven, 0.25 below the plain GPT, and the rule shut it out only for its undo error of 6.4e-3. Its gradients were as accurate as plain autograd's. I haven't tested whether that lead survives to 50M tokens, and it is the most interesting open question here. Midpoint tied the plain GPT at 5M tokens (5.344 against 5.345) and fell 0.091 behind by 50M.

### Run 3 failed because of its learning-rate schedule

Run 3 ended at 6.08, and its samples are common words in no order. The notebook also ran a 100-step bench at the same batch of 888. Both start from the same weights and see the same batches, so the learning rate is the only difference. Training loss:

| step | 5 | 20 | 40 | 60 | 80 |
|---|---|---|---|---|---|
| bench: lr 1e-3, 50-step warmup | 7.83 | 7.01 | 6.46 | 6.04 | 5.79 |
| run 3: lr 3e-3, 4-step warmup | 7.06 | 7.00 | 6.80 | 6.54 | 6.34 |

Run 3 drops faster for 5 steps, then stalls, and the bench passes it by step 40.

The schedule came from two rules. Scaling lr by sqrt(888 / 64) gave 3.7e-3, capped at 3e-3. Keeping warmup at 1.6% of training, as in run 2, cut it from 50 steps to 4. AdamW's first updates move each weight by about the learning rate, whatever its gradient. So from step 4, every step moved a typical starting weight (std 0.02) by 15%, and an output-projection weight (std 0.0045) by 67%. The square-root rule assumes gradient noise is the limit on step size. A bigger batch does lower that noise. It does not make the loss surface any less sharp at the start of training, and that sharpness is what makes big early steps unsafe. Warmup is there to get past this region, and that takes some number of steps, not some share of training.

After step 20, run 3's loss creeps from 6.80 to 6.10 while its gradient norm sits at 0.1 to 0.2. At step 220, run 2 had a training loss of 5.37 after 3.6M tokens, and run 3 had 6.10 after 49.8M. Each of run 3's steps averaged 13.9x more text and still did worse.

So the 6.08 measures this schedule, not a reversible model at large batch. Rerunning run 3 with lr 1e-3 and 50 warmup steps would test the real question. I haven't done that yet. Even then, 219 steps is very few for this model, and it will most likely finish behind run 2.

### Checks on the T4

- **Gradients.** With fixed-point streams and fp16 matmuls, the gradient error of coupling, midpoint and momentum against fp64 is 0.95x to 0.97x plain autograd's. Euler's is 379x. In fp64 on the CPU, coupling, midpoint and momentum match autograd to 3e-8. Euler is off by 1.7.
- **Undo.** After 5M tokens of training, fixed-point coupling and midpoint rebuild every layer's input exactly (error 0). With float streams, midpoint's layer-0 input is 6% off in fp32, 17% in fp16 and 102% in bf16. Momentum drifts to 0.5% even with fixed point.
- **Precision.** On the T4, fp16 matmuls run at 27.0 TFLOPS, fp32 at 4.3 and bf16 at 2.5, because the T4 emulates bf16. Training runs at 97k, 28k and 19k tok/s. That's why the runs use fp16.

## Findings from building it (Mac, before the T4 run)

These come from the gradient check and the smoke run on a MacBook Air M4 (bf16); the files are in `runs/mac/`. The T4 run confirmed the first three (see Checks on the T4 above).

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
