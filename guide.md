# Undo instead of remember: training a language model that forgets on purpose

A step by step guide for someone who has never trained a neural network.

You will train a small language model (19.4 million numbers) on 50 million words' worth of educational web text, three times. The first time, it trains the ordinary way. The second time, it is wired so that it can throw away almost everything it would normally keep in memory, and rebuild it when needed by running its arithmetic backwards. The third time, you use the memory that saves to feed it much bigger batches of text at once, and see what that does.

By the end you will have:

- seen what a model, a gradient and a training step are, with numbers small enough to check on paper
- worked out where a training step's memory goes, and why most of it is "scratch paper"
- undone a layer of a neural network by hand, with a pencil
- met four ways of making layers undoable, and seen which one survives real computer arithmetic
- watched a rounding error of one part in ten million grow, layer by layer, until it was 1 part in 100, and fixed it by storing numbers as whole integers
- met six ways big models are split across many GPUs, each one checked on a handful of numbers
- read the memory, speed and loss numbers of three real training runs, and said what each one costs

You do not need to know machine learning or much maths. You need to be able to add, subtract, multiply, and run a notebook. Every term is explained the first time it appears, and there is a glossary at the end.

How to read this guide. Parts 1 to 11 are the "lecture": what each piece is, with small numbers you can check yourself. Parts 12 to 20 are the "lab": how to run the notebook, how to read what it prints, and what our runs measured. Each lecture part ends with a few "check yourself" questions; the answers are folded underneath, so try first. If you are impatient, read Parts 5 and 7, then jump to the lab.

This project builds on three earlier ones: [a10-pipeline](../a10-pipeline/guide.md) trained the same kind of model on a laptop and checked every number it reported, [a12](../a12-DistributedTraining/guide.md) split training across 32 pretend GPUs, and [a14](../a14-MOE/guide.md) turned a model into a "mixture of experts". You do not need them. Where they go deeper, the text says so.

---

## Contents

**Lecture**

1. [Numbers in, numbers out: what a model is](#part-1-numbers-in-numbers-out-what-a-model-is)
2. [Learning by nudging knobs](#part-2-learning-by-nudging-knobs)
3. [Text as numbers](#part-3-text-as-numbers)
4. [Our model, station by station](#part-4-our-model-station-by-station)
5. [Why training eats memory](#part-5-why-training-eats-memory)
6. [The batch, and why memory caps it](#part-6-the-batch-and-why-memory-caps-it)
7. [Undo instead of remember](#part-7-undo-instead-of-remember)
8. [Four ways to make a layer undoable](#part-8-four-ways-to-make-a-layer-undoable)
9. [The price: time, and rounding](#part-9-the-price-time-and-rounding)
10. [Splitting the work across GPUs](#part-10-splitting-the-work-across-gpus)
11. [Measuring speed and memory](#part-11-measuring-speed-and-memory)

**Lab**

12. [Setting up: Kaggle, Colab or a Mac](#part-12-setting-up-kaggle-colab-or-a-mac)
13. [Running the notebook](#part-13-running-the-notebook)
14. [The gradient check](#part-14-the-gradient-check)
15. [The screening, and the winner](#part-15-the-screening-and-the-winner)
16. [The three runs](#part-16-the-three-runs)
17. [What we found that we did not plan for](#part-17-what-we-found-that-we-did-not-plan-for)
18. [When things go wrong](#part-18-when-things-go-wrong)
19. [A tour of the code](#part-19-a-tour-of-the-code)
20. [Experiments to try next, and glossary](#part-20-experiments-to-try-next-and-glossary)

---

## Part 1: Numbers in, numbers out: what a model is

### 1.1 A recipe with knobs

Imagine a machine with one input slot, one output slot and two knobs. You put a number in, it multiplies by the first knob, adds the second knob, and gives you the answer:

```
output = knob1 x input + knob2
```

With knob1 = 3 and knob2 = 1, putting in 2 gives 3 x 2 + 1 = 7. Turn knob1 to 4 and the same input gives 9.

That is a model. A neural network is the same idea with many more knobs and a few more kinds of step. The knobs are called **weights** or **parameters** (the two words mean the same here). Our model has 19,382,016 of them. The steps it runs are almost all multiplications and additions, with a few other simple operations in between.

### 1.2 Many numbers at once

Real inputs are not one number. A word, inside our model, is a list of 384 numbers. A list of numbers is called a **vector**, and a table of numbers is called a **matrix**. When the machine takes a list of 384 numbers and produces another list of 384, each output number is a weighted sum of all 384 inputs, with its own 384 knobs:

```
output[0] = k[0,0] x in[0] + k[0,1] x in[1] + ... + k[0,383] x in[383]
output[1] = k[1,0] x in[0] + ...
```

That is 384 x 384 = 147,456 knobs for one such step. This "every output is a weighted sum of every input" step is called a **matrix multiply**, and it is where a GPU spends almost all its time.

### 1.3 Layers

A network runs the same kind of step many times in a row. Each round is a **layer**. Our model has 10. You can think of them as stations on an assembly line: the item (our list of numbers) enters station 1, gets worked on, moves to station 2, and so on. Part 4 says what each station does.

<details>
<summary>Check yourself</summary>

1. The machine is output = knob1 x input + knob2, with knob1 = 5 and knob2 = -2. What comes out for input 3?
2. A step turns a list of 4 numbers into a list of 4 numbers, each output a weighted sum of all inputs. How many knobs?
3. Why is "parameters" a better word than "knobs" in a paper, and why is "knobs" better for this guide?

**Answers.** 1: 5 x 3 - 2 = 13. 2: 4 x 4 = 16. 3: "parameters" is the standard term everyone will understand; "knobs" says what they are, things you turn to change the output.
</details>

---

## Part 2: Learning by nudging knobs

### 2.1 Score the guess

Training means turning the knobs until the machine's outputs are good. To do that we need a number that says how bad an output is. That number is the **loss**. Smaller is better, zero is perfect.

A one-knob machine: output = w x input. We want input 3 to give 9. With w = 2 the machine says 6. A common loss is the squared miss:

```
loss = (guess - target)^2 = (6 - 9)^2 = 9
```

### 2.2 Which way to turn

We could try w = 2.1 and w = 1.9 and see which loss is lower. With 19 million knobs that is far too slow. Calculus gives a shortcut: for every knob, a single number that says "if you turn this knob up a tiny bit, the loss changes by this much per unit of turn". That number is the knob's **gradient**.

For our machine the gradient works out to 2 x (guess - target) x input = 2 x (6 - 9) x 3 = -18. Negative means "turning w up makes the loss go down". So we turn w up, by a small step: the **learning rate**, here 0.01, times the gradient.

```
new w = 2 - 0.01 x (-18) = 2.18
new guess = 2.18 x 3 = 6.54
new loss = (6.54 - 9)^2 = 6.05     (was 9)
```

That is one training step. Repeat it many times and w creeps toward 3, where the loss is zero. A real-life picture: you are on a hillside in thick fog, and you want to reach the valley. You cannot see the valley, but you can feel which way the ground slopes under your feet. Take a small step downhill. Feel again. Step again.

### 2.3 Forward and backward

Every training step has two halves.

- The **forward pass** runs the input through all the layers and computes the loss.
- The **backward pass** works out every knob's gradient. It starts at the loss and walks back through the layers, last layer first, using a rule from calculus (the chain rule) to pass the "which way" information from each layer to the one before it.

Then the **optimizer** turns every knob by a small step against its gradient. Our optimizer is called **AdamW**. It is a smarter version of "learning rate times gradient": it keeps two running averages per knob so each knob's step size adapts to how noisy its gradient is. Remember that it keeps two extra numbers for every knob, because Part 5 counts them.

PyTorch, the library we use, computes the backward pass for you. That feature is called **autograd**. Part 7 replaces autograd's usual way of doing it with our own.

<details>
<summary>Check yourself</summary>

1. The machine is output = w x input, w = 4, input 2, target 6. What is the loss, the gradient (2 x (guess - target) x input), and w after one step with learning rate 0.05?
2. The gradient of a knob is +3. Do you turn the knob up or down to lower the loss?
3. In which order does the backward pass visit the layers?

**Answers.** 1: guess 8, loss 4, gradient 2 x 2 x 2 = 8, new w = 4 - 0.05 x 8 = 3.6. 2: down; a positive gradient means turning up raises the loss. 3: last layer first, first layer last.
</details>

---

## Part 3: Text as numbers

### 3.1 The one job

A **language model** reads some text and guesses what comes next. That is the whole job. Give it "The capital of France is" and it gives a score to every possible next piece. A good model scores " Paris" highest.

### 3.2 Tokens

The "pieces" are called **tokens**. A token is a word, a part of a word, or a punctuation mark. The program that cuts text into tokens is the **tokenizer**, and every token has an ID number. Here is our tokenizer (the one a10 trained, 4,096 different tokens) on a short sentence:

```
"The cat sat on the mat."
-> ['The', ' cat', ' sat', ' on', ' the', ' mat', '.']
-> [451, 2927, 2657, 332, 262, 1259, 14]
```

The model never sees letters, only these ID numbers. Our training text is 50,000,000 tokens of FineWeb-Edu, a collection of educational web pages. That is about 171 million characters, or roughly 300 novels.

### 3.3 Turning an ID into a vector

The first thing the model does with a token ID is look it up in a table with 4,096 rows, one per token, each row a list of 384 numbers. That table is the **embedding**, and its numbers are knobs too (4,096 x 384 = 1,572,864 of them). A second table adds a list of 384 numbers for the token's position in the text (first, second, ...), so the model knows word order.

### 3.4 The score at the end, and the loss

At the end, the model turns each position's 384 numbers back into 4,096 scores, one per possible next token. These scores go through **softmax**, which turns them into probabilities that add up to 1. The loss is then:

```
loss = -log(probability the model gave to the token that really came next)
```

If the model gave the right token probability 1, the loss is -log(1) = 0. If it gave probability 1/4,096, as a model that knows nothing does, the loss is -log(1/4096) = 8.32. That is why every run in this project starts near 8.3. a10's 12-million-knob model got down to 3.9 after 25 million tokens.

We measure the loss on text the model never trains on, the **held-out** set (500,000 tokens). Training loss can drop because a model memorizes its training text; held-out loss only drops if it learned something that carries over.

<details>
<summary>Check yourself</summary>

1. The model gives the right next token probability 0.5. What is the loss? (log here is the natural log; -log(0.5) = 0.69.)
2. Why does an untrained model start with a loss near 8.32?
3. Why do we report held-out loss and not training loss?

**Answers.** 1: 0.69. 2: it spreads probability evenly, 1/4,096 per token, and -log(1/4096) = 8.32. 3: held-out text was never trained on, so a lower loss there means real learning, not memorizing.
</details>

---

## Part 4: Our model, station by station

### 4.1 The assembly line

Our model is a **GPT**, the same design (much smaller) as the models behind chatbots. For every token position it carries a list of 384 numbers through 10 identical stations, called **blocks**. That moving list is the **residual stream**: think of it as the item on the conveyor belt.

Every block does two jobs, and each job adds a correction to the item rather than replacing it:

```
x = x + Attention(LayerNorm(x))     the tokens look at each other
x = x + MLP(LayerNorm(x))           each token is worked on alone
```

The "x = x + correction" shape is called a **residual connection**. It matters a great deal for this project, because addition is easy to undo: if you know what was added, you can subtract it.

### 4.2 The four pieces of a block

- **LayerNorm** rescales each token's 384 numbers so their average is 0 and their spread is 1. Like adjusting the volume of a recording before editing it, so loud and quiet recordings are handled the same way. Keep this one in mind: Part 8 finds that it causes trouble for one of the four reversible designs.
- **Attention** lets each token collect information from the tokens before it. For the word "it" in "The cat sat because it was tired", attention is how the model can pull in information from "cat". Each token computes how much to listen to each earlier token, then takes a weighted mix of what they offer. Our model does this 6 ways in parallel, called 6 **heads**, each looking for a different kind of relationship.
- The **MLP** (multi-layer perceptron) works on each token alone: it widens the 384 numbers to 1,536, applies a simple bend called GELU (negative numbers get squashed toward zero), and narrows back to 384. Most of the model's knobs live here.
- The two **residual adds** put each job's correction onto the item.

### 4.3 Where the 19,382,016 knobs are

| part | knobs | how |
|---|---|---|
| token embedding (also used at the output) | 1,572,864 | 4,096 tokens x 384 |
| position embedding | 98,304 | 256 positions x 384 |
| one block | 1,771,008 | attention 4 x 384 x 384, MLP 8 x 384 x 384, plus 1,536 for two LayerNorms |
| 10 blocks | 17,710,080 | |
| final LayerNorm | 768 | |
| **total** | **19,382,016** | |

The model reads up to 256 tokens at once (its **context**). It is a10's model with 10 blocks instead of 6: same width, same tokenizer, same code. Adding blocks, not width, was deliberate: Part 7's trick saves memory in proportion to the number of blocks.

<details>
<summary>Check yourself</summary>

1. A block's attention job adds a correction c to the stream x. Afterwards the stream holds 7.5 and you know c was 2.25. What was x?
2. Which piece of a block works on each token alone, and which mixes tokens?
3. Why did we add blocks instead of making each block wider?

**Answers.** 1: 7.5 - 2.25 = 5.25. 2: the MLP (and LayerNorm) work on each token alone; attention mixes them. 3: the reversible trick saves memory per block, so more blocks means more saving to measure.
</details>

---

## Part 5: Why training eats memory

### 5.1 Four piles

While a model trains, the GPU's memory holds four piles of numbers:

| pile | what it is | size grows with |
|---|---|---|
| weights | the knobs | the model |
| gradients | one per knob: which way to turn it | the model |
| optimizer state | AdamW's two running averages per knob | the model |
| activations | numbers from the forward pass kept for the backward pass | the batch, the text length, and the number of layers |

The first three are fixed by the model. For ours, in the precision we train in:

```
weights           19,382,016 x 4 bytes = 77.5 MB
gradients         19,382,016 x 4 bytes = 77.5 MB
AdamW state       19,382,016 x 8 bytes = 155.1 MB
fp16 copies of the weights                38.5 MB   (Part 9 explains fp16)
                                         --------
                                          348.6 MB
```

A T4 GPU has 16 GB. So the model itself uses about 2% of it. Everything else in this project is about the fourth pile.

### 5.2 Scratch paper

Remember long multiplication from school:

```
     47
   x 23
   ----
    141      <- 47 x 3
    940      <- 47 x 20
   ----
   1081
```

The middle rows are scratch work. If you only wanted the answer, you could throw them away. But imagine a teacher who, after you finish, asks "how much would the answer change if the 4 had been a 5?". To answer quickly, you look at your scratch rows. That is the backward pass. To work out gradients, it needs numbers from the forward pass: the input to every matrix multiply, the output of every softmax, and so on. So the forward pass keeps them. Those kept numbers are the **activations**, and autograd keeps them automatically, for every layer, until the backward pass has used them.

### 5.3 How big the scratch paper is

For one sequence of 256 tokens, one of our blocks keeps about 3.5 MB of activations (the 384 numbers of each token at several points, the 1,536 MLP numbers before and after GELU, and so on). Ten blocks keep 35.5 MB. The loss at the end keeps another 4.2 MB (4,096 probabilities for each of 256 positions, 4 bytes each).

So one sequence costs about 40 MB of scratch paper. Train on 64 sequences at once and it is about 2.5 GB, seven times the model itself:

```
the model (weights + gradients + AdamW + fp16 copies)     348.6 MB
activations, 10 blocks x 64 sequences                   2,268.9 MB
activations, the loss, 64 sequences                       268.4 MB
                                                        ----------
                                                        2,885.9 MB
```

These are textbook predictions, from `flops.memory_bill`. The notebook prints them next to what autograd really kept and what the GPU really used, so you can check them (Part 16).

The key line is the second one. It grows with the number of blocks: a 100-block model would keep 100 blocks of scratch paper. Part 7 is about making that line stop growing.

<details>
<summary>Check yourself</summary>

1. A model has 1 million knobs and trains with AdamW in fp32. How many MB for weights, gradients and optimizer state together?
2. Our model at 128 sequences instead of 64: roughly how much activation memory?
3. Which pile would shrink if we used 5 blocks instead of 10?

**Answers.** 1: 4 + 4 + 8 = 16 bytes per knob, 16 MB. 2: about twice 2.5 GB, so about 5 GB. 3: the activations of the blocks (and a little of the model piles, since there are fewer knobs).
</details>

---

## Part 6: The batch, and why memory caps it

### 6.1 Many examples per step

The gradient from one sequence is noisy: it tells you which way to turn the knobs to do better on that one piece of text. Averaging the gradients of many sequences gives a steadier direction. The set of sequences processed in one step is the **batch**, and the number of them is the **batch size**. Our main runs use 64 sequences of 256 tokens, 16,384 tokens per step. 50,000,000 tokens then take 3,039 steps (the last few hundred tokens that do not fill a step are left out).

### 6.2 Bigger batches

A bigger batch has two attractions. Each step does more work at once, which keeps a GPU busier: a GPU is like a bus, and a bus with 60 passengers costs about the same to drive as one with 6. And the gradient is steadier. The cost is that you get fewer steps from the same text: 50M tokens at 1,024 sequences per batch is only 190 steps. With fewer, steadier steps you usually need a bigger learning rate, and there is a limit past which a bigger batch stops helping at all. Our third run sits right in that trade.

### 6.3 Memory caps the batch

Activations grow in step with the batch: double the sequences, double the scratch paper. The model's own 349 MB does not change. So on a 16 GB GPU there is a largest batch that fits, and past it the GPU runs **out of memory** and the program stops. For our plain model that limit is set almost entirely by the activations of the blocks, 35.5 MB per sequence. Shrink that, and the limit moves.

There is an old trick for going past the limit anyway: run several small batches one after another and add up their gradients before turning the knobs (**gradient accumulation**, a10's Part 6). It gives the steadier gradient of a big batch, but not the speed, since the GPU still does small batches one at a time. This project does not use it: run 3 asks how big a batch fits in one go.

<details>
<summary>Check yourself</summary>

1. How many tokens per step does a batch of 96 sequences of 256 tokens hold, and how many steps does 50M tokens take?
2. Why does a batch twice as big not need twice as much total memory?
3. What does gradient accumulation give you, and what does it not?

**Answers.** 1: 24,576 tokens; 50,000,000 / 24,576 = about 2,034 steps. 2: the model's own memory (weights, gradients, optimizer) does not depend on the batch; only the activations double. 3: it gives the gradient of a big batch; it does not give the speed of one.
</details>

---

## Part 7: Undo instead of remember

### 7.1 The idea

The backward pass needs each block's input. The ordinary way is to keep every block's input, plus all the scratch work inside it, from the forward pass. The reversible way is to keep only the output of the last block, and during the backward pass compute each block's input from its output, one block at a time, last block first. Then redo that block's forward work (now with the scratch paper kept, briefly), use it for the gradients, and throw it away before moving down to the next block.

Real-life picture: a hiker who wants to walk back to the start without having dropped breadcrumbs. If every step was "3 paces north, then 2 paces east", and the hiker remembers the rule, they can walk backwards from where they stand: 2 paces west, 3 paces south, and so on. They never needed a map of every point along the way. They need two things: the final position, and steps that can be undone.

"If I know you added 5, I can subtract 5." That is the whole trick, and it is why the residual connection (Part 4) matters: a block's output is its input plus a correction. If we can compute the correction from something we still have, we can subtract it.

### 7.2 The catch

There is a problem with the plain GPT block. Its output is

```
y = x + F(x)
```

where F is the block's correction. To subtract F(x) we need to compute it, and to compute it we need x, which is exactly what we are trying to find. Circular.

The fix is to change the wiring so the correction is always computed from something we still have after the step. The simplest version splits the stream into two halves that take turns.

### 7.3 Two streams taking turns, by hand

Keep two copies of the stream, x1 and x2. A block now does:

```
y1 = x1 + A(x2)        correction for stream 1, computed from stream 2
y2 = x2 + M(y1)        correction for stream 2, computed from the NEW stream 1
```

Here A is the attention job and M the MLP job, the same pieces as in a plain block. To undo, go in reverse order:

```
x2 = y2 - M(y1)        we have y1, so we can compute M(y1)
x1 = y1 - A(x2)        we just got x2, so we can compute A(x2)
```

With pencil numbers. Let A(x) = x times x, and M(y) = y / 10 rounded to a whole number. Start with x1 = 3, x2 = 5.

```
forward:  y1 = 3 + A(5)  = 3 + 25 = 28
          y2 = 5 + M(28) = 5 + 3  = 8           keep only (28, 8)

undo:     x2 = 8  - M(28) = 8 - 3   = 5
          x1 = 28 - A(5)  = 28 - 25 = 3          back to (3, 5)
```

Look at what A and M are. A(5) = 25 and A(-5) = 25, so from A's output alone you cannot tell which went in: A cannot be undone. M(28) = M(29) = 3: M cannot be undone either. The layer can be undone anyway, because we never undo A or M. We only recompute them and subtract. That is why this works for attention and MLPs, which are nowhere near undoable themselves.

Real-life picture: two bank accounts, Alice's and Bob's. Each month, Alice receives an amount computed from Bob's balance; then Bob receives an amount computed from Alice's new balance. You are shown only the final balances. Can you find last month's? Yes: compute what Bob received from Alice's balance (which you have), subtract it, then compute what Alice received from Bob's old balance (which you now have), subtract that.

This design is the **coupling** layer, also called a RevNet block (Gomez et al., 2017), and the Reformer (Kitaev et al., 2020) used it for transformers.

### 7.4 What it saves

The forward pass now keeps only the last block's output: two streams of 64 x 256 x 384 numbers. During backward it holds one block's scratch work at a time. The textbook bill at 64 sequences:

```
                                  plain GPT      reversible
the model                          348.6 MB       348.6 MB
activations of the 10 blocks     2,268.9 MB          -
final state (2 streams)               -           100.7 MB
one block, while it is redone         -           226.9 MB
the loss                           268.4 MB       268.4 MB
                                  ---------      ---------
                                  2,885.9 MB       944.6 MB
```

(The two streams take 8 bytes per number, not 4, for a reason Part 9 explains.)

Per sequence, the scratch paper drops from about 40 MB to about 9 MB. The block activations no longer grow with depth: 10 blocks or 100 blocks, the stack keeps one final state and one block's work. The loss at the end does not shrink at all, and at very big batches it becomes the biggest pile. Part 16 shows where that leaves the largest batch.

### 7.5 What it costs

Nothing is free. During backward, every block runs its forward work a second time. Part 9 counts the cost: about a third more arithmetic per step.

<details>
<summary>Check yourself</summary>

1. With A(x) = x times x and M(y) = y / 10 rounded, the output is y1 = 19, y2 = 6. What were x1 and x2?
2. Why can the coupling layer be undone even though A and M cannot?
3. Which pile of Part 5 does reversibility shrink, and which one does it not touch?

**Answers.** 1: x2 = 6 - M(19) = 6 - 2 = 4, x1 = 19 - A(4) = 19 - 16 = 3. 2: undoing only ever recomputes A and M on numbers we have, and subtracts; it never has to run them backwards. 3: it shrinks the activations of the blocks; it does not touch the model's own memory or the loss's activations.
</details>

---

## Part 8: Four ways to make a layer undoable

The coupling layer is one answer. There are others, and the task of this project is to try them and report which one works. Write F(x) for a whole block's correction (attention, then MLP), the thing a plain GPT block adds.

### 8.1 Coupling

Part 7. Two streams, each corrected from the other. Undoing is exact in exact arithmetic. Each block runs attention on one stream and the MLP on the other.

### 8.2 Midpoint: remember two steps, not one

Keep the last two positions, and step from the older one:

```
x[next] = x[previous] + F(x[now])
```

To undo, we need x[previous], and we have x[now] and x[next]:

```
x[previous] = x[next] - F(x[now])
```

By hand, with F(x) = 2x, starting with x[-1] = x[0] = 1:

```
forward:  x[1] = x[-1] + F(x[0]) = 1 + 2  = 3
          x[2] = x[0]  + F(x[1]) = 1 + 6  = 7
          x[3] = x[1]  + F(x[2]) = 3 + 14 = 17          keep only (7, 17)

undo:     x[1] = x[3] - F(x[2]) = 17 - 14 = 3
          x[0] = x[2] - F(x[1]) = 7 - 6   = 1           back to the start
```

Real-life picture: hopscotch where each landing is decided by where you were two hops ago plus a push from where you are now. Knowing your last two squares is enough to walk the whole game back.

The name comes from a method for simulating motion (the leapfrog or midpoint rule): physicists use it because it can be run backwards exactly. In our code the step is written x[n+1] = x[n-1] + 2h F(x[n]) with h = 0.5, so each block adds its full correction, as a plain block does. Unlike coupling, each step runs the whole block (attention and MLP) on one state.

### 8.3 Euler: the plain GPT, undone by guessing

The plain block itself, y = x + F(x), is a step of Euler's method, the simplest way to simulate motion. Part 7 showed why it cannot be undone directly. It can be undone by guessing: start with the guess x = y, compute y - F(guess) as a better guess, and repeat.

When F is gentle, the guesses close in. F(x) = x times x / 8, and the true input is 2, so y = 2 + 0.5 = 2.5:

```
guesses: 2.5, 1.7188, 2.1307, 1.9325, 2.0332, 1.9833, 2.0083, ...   -> 2
```

When F is steep, it fails. F(x) = x times x / 2, true input 2, so y = 2 + 2 = 4:

```
guesses: 4, -4, -4, -4, ...
```

Stuck at -4. And -4 is a "correct" answer: -4 + F(-4) = -4 + 8 = 4. Two different inputs give the same output, so no amount of guessing can know which one it was. This is not undoable.

Our blocks start with LayerNorm, which divides by the spread of the numbers. At the start of training, the token vectors coming out of the embedding are tiny (spread about 0.03), so LayerNorm multiplies small differences by about 35. That makes F steep. Euler's guessing has a real chance of failing on our model, and Part 14 shows that it does. Euler also costs a forward pass per guess: with 3 guesses, a step does 4 extra block passes instead of 1.

### 8.4 Momentum: a rolling ball

Carry a speed v next to the position x. Each block nudges the speed, and the speed moves the position:

```
v[next] = 0.9 x v + 0.1 x F(x)
x[next] = x + v[next]
```

Like a ball rolling over hills: the speed keeps 90% of what it had and picks up a bit from the slope under the ball. Undo:

```
x = x[next] - v[next]
v = (v[next] - 0.1 x F(x)) / 0.9
```

By hand: x = 1, v = 0, F(1) = 10. Forward: v[next] = 0 + 1 = 1, x[next] = 1 + 1 = 2. Undo: x = 2 - 1 = 1, v = (1 - 0.1 x 10) / 0.9 = 0. Exact.

But look at the divide by 0.9. Every undo multiplies whatever small error the speed carries by 1 / 0.9 = 1.11. Ten blocks: 1.11 to the 10th power = 2.87 times the error. In exact arithmetic there is no error, so it does not matter. On a real computer there always is (Part 9), so momentum is exact on paper and drifting in practice. This design is MomentumNet (Sander et al., 2021).

### 8.5 Side by side

| | streams kept | undo | each block runs | extra work per step |
|---|---|---|---|---|
| coupling | 2 | exact on paper | attention on one stream, MLP on the other | 1 block pass |
| midpoint | 2 | exact on paper | the whole block on one state | 1 block pass |
| euler | 1 | by guessing, can fail | the whole block | 1 + one pass per guess |
| momentum | 2 (position and speed) | exact on paper, errors grow by 1.11 per block | the whole block | 1 block pass |

All four use exactly the same knobs as the plain GPT (the same LayerNorms, attention and MLP in every block), so all four have 19,382,016 parameters. What differs is how the pieces are wired, and wiring changes what the model can learn. That is why the notebook trains all four on the same 5M tokens before choosing one (Part 15).

<details>
<summary>Check yourself</summary>

1. Midpoint with F(x) = 3x. You have x[now] = 2 and x[next] = 10. What was x[previous]?
2. Why can the guessing for Euler fail, in one sentence?
3. Momentum, 20 blocks: by how much does the undo multiply an error in the speed?

**Answers.** 1: 10 - F(2) = 10 - 6 = 4. 2: if two different inputs give the same output, no guess can tell them apart (and when F is steep, the guesses jump around instead of closing in). 3: 1.11 to the 20th power, about 8.2.
</details>

---

## Part 9: The price: time, and rounding

### 9.1 Time

Count the work of one step in units of "one forward pass of the blocks". The backward pass costs about two units (for each matrix multiply it computes two things: the gradient for the knobs and the gradient to pass down). So:

```
plain:      forward 1 + backward 2                       = 3 units
reversible: forward 1 + redo forward 1 + backward 2      = 4 units    (+33%)
euler, 3 guesses: forward 1 + guesses 3 + redo 1 + backward 2 = 7 units  (+133%)
```

So a reversible step should take about a third longer than a plain one at the same batch. The question run 3 answers is whether the bigger batches it allows win that time back.

### 9.2 How computers store numbers

A computer stores a number in a fixed number of bits, like writing it on a ruler with a fixed number of tick marks. Between two tick marks there is nothing: every number lands on the nearest tick. Our three formats, with the gap between ticks near the value 1:

| format | bits | gap near 1 | about this many correct digits | largest number |
|---|---|---|---|---|
| fp32 | 32 | 0.00000012 | 7 | 3.4 x 10^38 |
| fp16 | 16 | 0.00098 | 3 | 65,504 |
| bf16 | 16 | 0.0078 | 2 to 3 | 3.4 x 10^38 |

(a10's Part 8 takes the number 0.1 apart bit by bit in each of these.) Training stores the knobs in fp32, but runs the matrix multiplies in a 16-bit format, because GPUs do 16-bit arithmetic much faster. This is **mixed precision**, and PyTorch's **autocast** does the switching.

fp16 and bf16 split their 16 bits differently. fp16 spends more on precision (finer ticks), bf16 more on range (bigger largest number). fp16's largest number is only 65,504, so tiny gradients can round to zero; the fix, **loss scaling**, multiplies the loss by a big number before backward and divides the gradients after. bf16 does not need it.

### 9.3 Why we train in fp16, not bf16

bf16 needs no loss scaling, so why not use it? On the T4, the GPU Kaggle gives us, there are two separate reasons not to:

1. **Speed.** The T4 has special circuits, **tensor cores**, for fp16 matrix multiplies: 65 trillion operations per second on the datasheet. It has none for bf16. bf16 still works, but it runs about as slowly as fp32 (8.1 trillion per second). Part 14 shows the measured numbers.
2. **Undoing.** bf16's ticks are 8 times further apart than fp16's. Part 9.4 shows why that matters for a reversible network: the undo has to subtract its way back through every rounding.

On newer GPUs (A100, H100) bf16 has tensor cores too, and the speed reason goes away. The undo reason does not, which is why our streams between blocks are not stored in either 16-bit format.

### 9.4 Undo meets rounding: a 3-digit calculator

Pretend our calculator keeps 3 significant digits. The stream holds 1.24, and a block adds a correction of 9.87:

```
forward:  1.24 + 9.87 = 11.11  -> stored as 11.1
undo:     11.1 - 9.87 = 1.23                        (it was 1.24)
```

The correction was bigger than the stream, so the stream's last digit fell off the end when they were added, and no subtraction brings it back. That is exactly the situation at the bottom of our model: the embedding's numbers are small (spread about 0.03), and the first block adds corrections much bigger than that.

It gets worse. The rebuilt 1.23 is wrong by one tick, and it goes into the next undo as the input of a block. Inside the block, numbers are rounded to fp16 or bf16, whose ticks are far coarser. A difference of one fp32 tick can push a number across an fp16 tick boundary, and then the block's output differs by a whole fp16 tick, about a thousand times bigger. That error goes into the next subtraction, and so on down the stack. Part 14 shows this happening: the rebuilt inputs drift a little more at each block on the way down.

### 9.5 The fix: count in whole numbers

Run the example again, but count in hundredths as whole numbers. 1.24 is 124 hundredths; round the correction to hundredths too, 987.

```
forward:  124 + 987 = 1111
undo:     1111 - 987 = 124                          exact
```

Whole-number addition and subtraction never round. As long as the correction is rounded to hundredths the same way both times (it is: it is recomputed from the same input by the same program), the undo returns exactly what went in. And because the rebuilt input is exact, the block that gets recomputed on it sees exactly the same numbers as in the forward pass, so every later undo is exact too. The fp16 rounding inside the blocks stops mattering.

Our model does this with a much finer unit: 2^-24 (about 0.00000006). Each stream number is stored as a whole number of those units in a 64-bit integer, which is why the final state takes 8 bytes per number. This is **fixed-point** storage, and it is how MacKay et al. (2018) made reversible recurrent networks exact. The notebook trains coupling and midpoint both ways, fixed-point and fp32, so you can see the difference it makes.

Fixed point does not save momentum. Its undo divides by 0.9, and division by 0.9 does not give a whole number of units, so it must round. MacKay's paper fixes that with an extra buffer of the bits that division throws away; we did not build that.

<details>
<summary>Check yourself</summary>

1. The 3-digit calculator: the stream holds 2.47 and the correction is 8.51. Add, store with 3 digits, then subtract. What comes back?
2. Same numbers, counting in hundredths as whole numbers. What comes back?
3. Give the two reasons bf16 is a bad choice on a T4 for this project.

**Answers.** 1: 2.47 + 8.51 = 10.98, stored as 11.0; 11.0 - 8.51 = 2.49. Wrong by 0.02. 2: 247 + 851 = 1098; 1098 - 851 = 247. Exact. 3: the T4 has no fast bf16 path (it runs near fp32 speed), and bf16's ticks are 8 times coarser than fp16's, so rounding inside the blocks is bigger.
</details>

---

## Part 10: Splitting the work across GPUs

Reversibility makes one GPU hold less. The other way to fit a bigger job is to use more GPUs. Big models are trained on thousands at once, and there are six standard ways to split the work. Each one cuts a different thing into pieces. Every method below is checked on a handful of numbers in `parallelism.ipynb`, where each "GPU" is a slice of a table and each message between GPUs is counted in bytes; every check gives exactly the answer one device gives.

A restaurant kitchen is a useful picture throughout. One cook (one GPU) cannot keep up, so you hire more. How do you divide the work?

### 10.1 Data parallelism: split the batch

Several identical kitchens, each cooking different orders from the same recipe book. Every GPU holds the whole model and gets a different part of the batch. Each computes gradients for its own examples, then they average their gradients (an **all-reduce**: every GPU sends its numbers, every GPU ends up with the average) and all turn their knobs the same way.

By hand: 4 examples, 2 GPUs. GPU 0 takes examples 0 and 1, GPU 1 takes 2 and 3. The average of their two gradients equals the gradient of all 4 together, exactly, because the loss is itself an average over examples. The notebook checks it: largest difference 0.

What it saves: time. What it does not save: memory, because every GPU holds the whole model. [a12](../a12-DistributedTraining/guide.md) is about removing that duplication (ZeRO).

### 10.2 Tensor parallelism: split one layer's knobs

One dish, several cooks: each chops half the vegetables. Our MLP widens 384 numbers to 1,536 with one matrix, then narrows back with another. Cut the first matrix by columns: GPU 0 makes hidden numbers 0 to 767, GPU 1 makes 768 to 1,535. Each does it alone, and GELU works on each number alone, so no talking is needed. Cut the second matrix by rows: each GPU multiplies its own half of the hidden numbers and gets a partial answer, and one all-reduce adds the two partial answers.

By hand, a token (1, 2), a 2 x 4 first matrix, a 4 x 2 second one: GPU 0's partial output is (0.4812, 0.7049), GPU 1's is (2.1803, 2.5381), their sum (2.6614, 3.2431) is what one device gets. Each GPU holds half the knobs of the layer. The price is one message per layer in the forward pass and one in the backward pass, every layer, so the GPUs need a very fast link between them (in practice, GPUs in one machine).

### 10.3 Pipeline parallelism: split the layers

An assembly line spread over buildings: GPU 0 holds blocks 0 to 4, GPU 1 holds blocks 5 to 9. Only the stream between the two halves travels. The problem is waiting: GPU 1 has nothing to do until GPU 0 finishes. So the batch is cut into small **micro-batches**: while GPU 1 works on the first, GPU 0 starts the second.

```
2 GPUs, 4 micro-batches (number = micro-batch, . = idle):
  GPU 0:  0 1 2 3 .
  GPU 1:  . 0 1 2 3
```

The idle slots are the **bubble**: (GPUs - 1) / (micro-batches + GPUs - 1) = 1/5 = 20% here, and 1/17 = 6% with 16 micro-batches. Each GPU holds only its own layers' knobs and their activations.

### 10.4 Sequence parallelism: split the sentence, for the parts that look at one token at a time

LayerNorm works on each token alone (Part 4), so its work can be cut by token: GPU 0 normalizes tokens 0 to 127, GPU 1 tokens 128 to 255, and each stores only its own half of the scratch paper. It is used next to tensor parallelism, for the parts of a block that tensor parallelism leaves whole on every GPU. Before attention, which needs every token, the halves are gathered; after it, they are split again.

By hand, one token (1, 2, 3, 6): its average is 3 and its spread is the square root of ((1-3)^2 + (2-3)^2 + 0 + (6-3)^2) / 4 = the square root of 3.5, so LayerNorm gives (-2, -1, 0, 3) / 1.8708. Nothing about the other tokens enters, which is why the split gives the same answer.

### 10.5 Context parallelism: split attention itself along the sentence

A reading group passing chapters around a circle. Each GPU keeps the queries for its own part of the text (say tokens 0 to 3 on GPU 0, 4 to 7 on GPU 1) and passes its keys and values to its neighbour in a ring, one chunk at a time, until each GPU has seen every chunk it needs.

Attention's softmax normally needs all the scores of a row before it can divide by their total. A GPU here only ever has one chunk. The fix is to keep running totals: the biggest score so far, the total of the exponentials so far, and the weighted sum of values so far. When a chunk arrives with a bigger score, scale the old totals down to match, then add the new chunk. This is the **online softmax**, the same idea that makes flash attention work. No GPU ever builds the full table of scores, which is what lets models read texts of a million tokens.

The notebook checks it on 8 tokens over 2 GPUs and 12 tokens over 3: the ring's answer matches one device to 1e-16.

### 10.6 Expert parallelism: specialist cooks

In a **mixture of experts** ([a14](../a14-MOE/guide.md)), each MLP is replaced by several smaller ones (experts), and a router sends each token to one of them. Put each expert on its own GPU, like a pastry chef and a grill chef at their own stations, and send each token to its expert's GPU and back (an **all-to-all**). Each GPU holds only its own experts' knobs.

### 10.7 Where reversibility fits

Each method cuts a different thing, and each pays in a different way:

| method | what it cuts | what each GPU holds less of | what it costs |
|---|---|---|---|
| data | the batch | nothing (each holds the whole model) | one all-reduce of all gradients per step |
| tensor | the width of each layer | knobs, gradients, optimizer state, most activations | two messages per layer, every step; needs a fast link |
| pipeline | the layers | knobs and activations of the other layers | idle bubble at the start and end of each step |
| sequence | the sentence, for one-token-at-a-time parts | those parts' activations | gathering before attention, splitting after |
| context | the sentence, for attention too | attention's activations and scores | passing keys and values round the ring |
| expert | the experts | the other experts' knobs | all-to-all sends of tokens, there and back |
| **reversible** (this project) | the depth, in time instead of space | the stored activations of every block but one | about a third more arithmetic; one GPU, no messages |

Reversibility is the odd one out: it splits nothing across GPUs. It trades memory for arithmetic on a single GPU, which also makes it combinable with all of the above. Large runs stack several methods at once: for example tensor parallelism inside each machine (fast links), pipeline parallelism across a few machines, and data parallelism across many copies of that arrangement, with ZeRO on top.

<details>
<summary>Check yourself</summary>

1. A model is too big for one GPU's memory even at batch 1. Does data parallelism help?
2. Pipeline with 4 GPUs and 12 micro-batches: what is the bubble?
3. Which methods split the sentence, and what is the difference between them?

**Answers.** 1: no; every GPU still holds the whole model. Tensor or pipeline parallelism (or ZeRO) is needed. 2: 3 / 15 = 20%. 3: sequence parallelism splits the parts that work on each token alone; context parallelism also splits attention, passing keys and values round a ring.
</details>

---

## Part 11: Measuring speed and memory

### 11.1 Tokens per second

The plainest speed number: how many tokens of text one training step gets through, divided by how long the step took. Our main runs do 16,384 tokens per step, so a step of 0.25 seconds is 65,536 tokens per second. We time each step with the GPU fully finished on both sides (a GPU accepts work and does it later, so a timer without that wait measures nothing), and we report the median step, after skipping the first 10 while everything warms up.

The two main runs train side by side on Kaggle's two GPUs, which share 4 CPU cores. To make sure that sharing did not slow them, the speed and memory numbers in the results come from separate 100-step runs with each configuration alone on the machine.

### 11.2 Peak memory

PyTorch tracks the highest memory any step reached, the **peak**, in two ways:

- **allocated**: bytes actually held by numbers in use. This is what the memory bill of Part 5 predicts.
- **reserved**: bytes PyTorch has taken from the GPU, including free blocks it keeps for reuse. This is what runs out first and what the GPU's own tools show.

The peak of a step is inside the backward pass, when gradients are being written while activations have not yet been freed.

### 11.3 MFU and HFU

Every token costs about 6 multiplications-or-additions per knob to train (2 forward, 4 backward; a10's Part 10 derives it). With N = 19,382,016 knobs that is 116 million per token. **MFU** (model FLOPs utilization) compares the useful work done per second with what the chip could do at most:

```
MFU = 6 x N x tokens per second / the chip's peak per second
```

For the T4 in fp16 the datasheet peak is 65 trillion per second. At 65,536 tokens per second, MFU would be 6 x 19.4M x 65,536 / 65 trillion = 11.7%.

A reversible step also redoes every block's forward work, 2 operations per block knob per token, which MFU does not count because it is not useful work. **HFU** (hardware FLOPs utilization) counts it. MFU says how much training you got per second; HFU says how busy the chip was.

<details>
<summary>Check yourself</summary>

1. A step of 16,384 tokens takes 0.4 s. Tokens per second?
2. Why can reserved memory be higher than allocated?
3. A reversible model and a plain one reach the same MFU. Which has the higher HFU?

**Answers.** 1: 40,960. 2: PyTorch keeps freed blocks to reuse later instead of giving them back to the GPU. 3: the reversible one; it did the same useful work plus the redone forward passes.
</details>

---

## Part 12: Setting up: Kaggle, Colab or a Mac

The notebook `reversibility.ipynb` runs anywhere PyTorch runs. Our results come from Kaggle's free 2 x T4.

### 12.1 Kaggle (what we used)

1. **The data.** On a computer with this repo, build the token files once:
   ```bash
   uv sync
   uv run data.py --out data/fwe50 --train-tokens 50M --eval-tokens 500K --tokenizer tokenizer.json
   ```
   This downloads about 300 million characters of FineWeb-Edu (about a minute and a half on a fast connection) and writes `train.bin` (100 MB), `eval.bin` (1 MB), `meta.json` and `tokenizer.json`. On kaggle.com, go to Datasets, New Dataset, upload those four files, and keep it private.
2. **The notebook.** On Kaggle, Create, New Notebook, then File, Import Notebook, and upload `reversibility.ipynb` (or paste its GitHub link).
3. **Settings** (the panel on the right): Accelerator GPU T4 x2; Internet on (the first cell clones this repo for the scripts). Add Input, and pick your dataset. The notebook finds `train.bin` under `/kaggle/input` by itself.
4. **Run it as a commit**: Save Version, Save & Run All. A commit runs in the background on Kaggle's machines, keeps going if you close the browser, and keeps everything written to `/kaggle/working` as the version's output. An interactive session dies when the tab sleeps; a14 lost a whole run that way.
5. **Collect the results.** When the version finishes, open it. Download the notebook (with its outputs) and the `a13` output folder, and copy them into the repo as `reversibility.ipynb` and `runs/`.

### 12.2 Colab

Upload the notebook, Runtime, Change runtime type, a GPU. Set `USE_DRIVE = True` in the first cell to keep outputs on Google Drive, so a disconnect does not lose them. Without a Kaggle dataset the notebook builds the data itself (a few minutes). With one GPU, the runs go one after another instead of two at a time.

### 12.3 A Mac

```bash
uv sync
uv run jupyter notebook reversibility.ipynb
```

Set `SMOKE = True` first. That runs every cell at toy size in about 32 minutes on an M4 (with the data already built), which is how this project checked the notebook before sending it to Kaggle. The full-size run would need about 7 hours of training on an M4 at the speeds the smoke run measured (9,400 tokens per second plain, 6,400 reversible), more once the fanless chip heats up. Apple GPUs have no fp16 tensor cores, so the notebook runs in bf16 there (it chooses by itself), and they keep no peak-memory counter, so the peak column shows the higher of two readings taken in every step, after forward and after backward.

---

## Part 13: Running the notebook

The notebook has 17 numbered sections, each a markdown cell saying what is about to happen and a code cell doing it. Heavy work never runs inside the notebook's own process. Each training run is a separate `python train.py ...` process, started by `jobs.py` and pinned to one GPU, writing its log to `<run>/log.txt`. That way the notebook never holds GPU memory that would spoil a memory measurement, and with two GPUs two runs train at once.

While runs are going, the queue prints each one's latest line every minute:

```
step 1240/3039 loss 3.9187 grad_norm 0.412 lr 7.31e-04 61,402 tok/s mfu 11.0% peak 2.96 GB (267 ms)
```

That is: the step number, this step's training loss, the size of the gradient before clipping (a10's Part 7: it moves before the loss does when training goes wrong), the learning rate, speed over the last 20 steps, MFU, the step's peak allocated memory, and how long this one step took. (The numbers in that line are an illustration; Part 16 has the real ones.) Every 2.5M tokens a run also prints its held-out loss, and a reversible run prints how far its rebuilt layer inputs are from the true ones.

If a session dies, run all the cells again. A run with a `done.json` is skipped; an unfinished one continues from its last checkpoint (saved every 10 minutes).

---

<!-- RESULTS:START -->
## Part 14: The gradient check

## Part 15: The screening, and the winner

## Part 16: The three runs

## Part 17: What we found that we did not plan for

*Parts 14 to 17 are written from the executed Kaggle notebook. They are not filled in yet: every number in them will come from a cell of `reversibility.ipynb`, and none is estimated.*
<!-- RESULTS:END -->

---

## Part 18: When things go wrong

**"CUDA out of memory" in a main run.** The batch is too big for this GPU. In the 64-batch runs that should never happen on a 16 GB card (Part 5 predicts under 3 GB). In the max-batch run, the probe found the limit with 3 steps, and a long run can need slightly more (memory gets fragmented over time). The notebook catches it, lowers the batch by 8 and restarts the run, up to three times.

**A run's loss becomes `nan`.** In fp16, the loss scaler skips steps whose gradients overflow, and a few skipped steps early on are normal (watch `loss_scale` in `metrics.jsonl`). A `nan` that stays means the weights themselves broke. For the euler variant this is a known way to fail (Part 8.3); for the others it is a bug worth reporting.

**The reversible run's loss drifts away from the plain run's.** Look at its `recon` lines in the log. With fixed-point streams, coupling and midpoint should print exactly 0. If they do not, the forward and the redone forward computed different numbers; the usual cause is a GPU kernel that picks a different method depending on whether gradients are being recorded (see Part 17, and `same_path` in `rev.py`).

**`git clone` fails on Kaggle.** Internet is off in the notebook settings. Turn it on, or upload the repo's `.py` files as a dataset and set the working directory to it.

**The notebook cannot find the data.** It searches `/kaggle/input` for `train.bin`. Check that the dataset is attached (Add Input) and that you uploaded the files themselves, not a zip.

**Colab disconnects.** Set `USE_DRIVE = True` and run again: finished runs are skipped, unfinished ones resume.

---

## Part 19: A tour of the code

| file | what it does | from |
|---|---|---|
| `model.py` | the GPT: embedding, blocks (LayerNorm, attention, MLP), output head | a10, unchanged |
| `data.py` | downloads FineWeb-Edu, tokenizes it, writes `train.bin` / `eval.bin` | a10, plus `--tokenizer` to reuse a10's tokenizer |
| `flops.py` | device, dtype, peak FLOPs, MFU, the memory bill, the saved-tensor list | a10, plus peak memory, HFU, the reversible bill, and a leak fix (Part 17) |
| `rev.py` | the four reversible variants and `RevStack`, the hand-written backward | new |
| `train.py` | the training loop: logs loss, grad norm, tok/s, MFU, HFU, peak memory, rebuild error | a10's loop, extended |
| `gradcheck.py` | every variant's gradients against ordinary autograd and against fp64 | new, in the spirit of a10's |
| `precision.py` | bf16 / fp16 / fp32 speed, and drift by stream format | new |
| `probe.py` | largest batch that fits: doubling, then bisecting | new |
| `jobs.py` | one run per GPU at a time, resume and skip | new |
| `plot.py` | the figures in `plots/` | new |
| `parallel_checks.py` | the six ways to split across GPUs, checked on small numbers | new |

The heart of the project is about 60 lines of `rev.py`. `RevStack.forward` runs every block and keeps only the last state. `RevStack.backward` walks the blocks from last to first; for each it calls `RevGPT.reverse`, which redoes the block's pieces with autograd on, uses the result twice (once to subtract and rebuild the block's input, once to push the gradient through), and adds the knobs' gradients straight into their `.grad`. The coupling case:

```python
y1r, m = run(self._M, y1)          # redo M on y1, with autograd recording
x2 = add(y2, m.detach(), -1)       # undo: x2 = y2 - M(y1)
dz1 = d1 + back(m, d2, y1r)        # gradient through M, added to y1's
x2r, a = run(self._A, x2)          # redo A on the rebuilt x2
x1 = add(y1, a.detach(), -1)       # undo: x1 = y1 - A(x2)
dx2 = d2 + back(a, dz1, x2r)       # gradient through A, added to x2's
```

`add` is ordinary float addition for fp32 streams and integer addition for fixed-point ones; nothing else changes between the two.

---

## Part 20: Experiments to try next, and glossary

### 20.1 Experiments

1. **Make the loss reversible too.** At big batches the loss's 4.2 MB per sequence becomes the largest pile (Part 7.4). Computing the loss in chunks of positions, and redoing each chunk's logits in backward, would cut it to almost nothing. How much further does the batch go?
2. **Go deeper.** Train 30 blocks at width 256 instead of 10 at 384 (similar knob count). The plain model's activations triple; the reversible model's stay put.
3. **Save momentum.** Add MacKay's buffer of the bits that division throws away, and see if momentum becomes exact.
4. **bf16 on a newer GPU.** On an L4 or A100, bf16 has tensor cores. Does it change the speed comparison? (It should not change the drift: the fixed-point streams make the blocks' precision irrelevant to the undo.)
5. **More layers of Euler guessing.** Raise `--euler-iters` from 3 to 10. Does Euler ever undo correctly on this model, and at what cost?

### 20.2 Glossary

- **activations**: numbers from the forward pass that are kept for the backward pass. The scratch paper.
- **AdamW**: the optimizer we use; keeps two running averages per knob.
- **all-reduce**: every GPU sends its numbers, every GPU ends up with the sum or average.
- **all-to-all**: every GPU sends a different piece to every other GPU.
- **attention**: the part of a block where each token gathers information from earlier tokens.
- **autocast**: PyTorch's switch that runs matrix multiplies in 16 bits while keeping the knobs in 32.
- **autograd**: PyTorch's automatic computation of gradients.
- **backward pass**: the half of a training step that computes every gradient, last layer first.
- **batch, batch size**: the sequences processed in one step, and how many.
- **bf16, fp16, fp32**: three ways to store a number in bits (Part 9.2).
- **block**: one of the 10 identical stations of our model.
- **bubble**: idle time in pipeline parallelism.
- **context**: how many tokens the model reads at once (256).
- **coupling**: a reversible block with two streams, each corrected from the other.
- **embedding**: the table that turns a token ID into 384 numbers.
- **fixed-point**: storing a number as a whole number of small units (here 2^-24), so adding and subtracting are exact.
- **forward pass**: running the input through the model to get the loss.
- **gradient**: for one knob, how much the loss changes per unit of turning it.
- **held-out**: text the model never trains on, used to measure it.
- **HFU**: hardware FLOPs utilization; like MFU, but also counting redone work.
- **LayerNorm**: rescales a token's numbers to average 0 and spread 1.
- **learning rate**: how big a step the optimizer takes.
- **loss**: one number saying how bad the model's guesses are; smaller is better.
- **loss scaling**: multiplying the loss before backward so fp16 gradients do not round to zero.
- **MFU**: model FLOPs utilization; useful work per second over the chip's peak.
- **midpoint**: a reversible design that steps from the state two positions back.
- **mixed precision**: knobs in 32 bits, arithmetic in 16.
- **MLP**: the part of a block that works on each token alone (widen, bend, narrow).
- **momentum**: a reversible design that carries a speed alongside the position.
- **online softmax**: softmax computed chunk by chunk with running totals.
- **optimizer**: the rule that turns the knobs from their gradients.
- **parameters, weights, knobs**: the numbers the model learns.
- **peak memory (allocated, reserved)**: the most memory a step used; held by numbers, or taken from the GPU.
- **residual connection, residual stream**: "x = x + correction"; the list of numbers each token carries through the blocks.
- **reversible**: wired so each layer's input can be computed from its output.
- **softmax**: turns scores into probabilities that add to 1.
- **tensor cores**: GPU circuits that do 16-bit matrix multiplies fast.
- **token, tokenizer**: a piece of text, and the program that cuts text into pieces.
