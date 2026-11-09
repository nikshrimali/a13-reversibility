"""Draw the figures in plots/ from a finished run folder.

    uv run plot.py --runs runs --out plots

Reads what reversibility.ipynb writes under --runs (screen/, main/, probe/, bench/, the precision
JSONs) and skips any figure whose inputs are missing, so it works on a half-finished run too.

    screening.png   held-out loss over tokens, every variant of the 5M-token screening
    main_loss.png   held-out loss over the 50M tokens: plain GPT, reversible, reversible at max batch
    probe.png       peak memory against batch size for both models, with the GPU's size marked
    memory.png      what one step holds at its peak, solo runs
    speed.png       tokens per second, solo runs
    recon.png       how far the undo drifts per layer, by how the streams are stored

Colors follow the architecture everywhere (plain GPT is always blue, coupling always orange...),
in the reference palette's fixed order. Line style carries the second thing that varies.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (dataviz skill), light mode, slots in fixed order.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ARCH_COLOR = {"gpt": SLOT[0], "coupling": SLOT[1], "midpoint": SLOT[2], "euler": SLOT[3], "momentum": SLOT[4]}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {"gpt": "plain GPT", "coupling": "coupling", "midpoint": "midpoint", "euler": "euler", "momentum": "momentum"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})


def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def evals(folder):
    return [(r["tokens"] / 1e6, r["loss"]) for r in load_jsonl(os.path.join(folder, "metrics.jsonl")) if r["type"] == "eval"]


def arch_of(folder):
    cfg = load_json(os.path.join(folder, "config.json")) or {}
    return cfg.get("arch", "gpt"), cfg.get("stream_dtype", "fp32"), cfg.get("micro_batch")


def screening(runs, out):
    root = os.path.join(runs, "screen")
    if not os.path.isdir(root):
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    rows = []
    for name in sorted(os.listdir(root), key=lambda n: (n != "gpt", n)):
        ev = evals(os.path.join(root, name))
        if len(ev) < 2:
            continue
        arch, sd, _ = arch_of(os.path.join(root, name))
        xs, ys = zip(*ev[1:])
        style = "--" if (arch != "gpt" and sd == "fp32") else "-"
        label = NAMES[arch] + ("" if arch == "gpt" else f" ({'fixed-point' if sd == 'fixed' else sd} streams)")
        ax.plot(xs, ys, style, color=ARCH_COLOR[arch], label=label)
        rows.append((ys[-1], label))
    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("held-out loss")
    ax.set_title("Screening: every variant on the same 5M tokens")
    finite = [y for y, _ in rows if y == y and y < 20]
    if finite:
        top = min(max(finite) + 0.6, 9)
        ax.set_ylim(min(finite) - 0.15, top)
        if max(y for y, _ in rows) > top:
            ax.annotate(f"lines that leave the top are cut at {top:.1f}", (0.01, 0.97), xycoords="axes fraction",
                        va="top", color=INK2, fontsize=8)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "screening.png"), dpi=150)
    plt.close(fig)


def main_loss(runs, out):
    root = os.path.join(runs, "main")
    if not os.path.isdir(root):
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, style in (("gpt", "-"), ("rev", "-"), ("rev_max", "--")):
        folder = os.path.join(root, name)
        ev = evals(folder)
        if len(ev) < 2:
            continue
        arch, sd, b = arch_of(folder)
        xs, ys = zip(*ev[1:])
        ax.plot(xs, ys, style, color=ARCH_COLOR[arch], label=f"{NAMES[arch]}, batch {b}: final {ys[-1]:.3f}")
    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("held-out loss")
    ax.set_title("Held-out loss over 50M tokens")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "main_loss.png"), dpi=150)
    plt.close(fig)


def probe(runs, out):
    root = os.path.join(runs, "probe")
    if not os.path.isdir(root):
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    total = None
    for name in ("gpt", "rev"):
        p = load_json(os.path.join(root, name, "probe.json"))
        if not p:
            continue
        arch = p["arch"]
        ok = [t for t in p["tries"] if t["fits"]]
        xs, ys = [t["batch"] for t in ok], [t["peak_alloc"] / 1e9 for t in ok]
        ax.plot(xs, ys, "-o", color=ARCH_COLOR[arch], ms=6, label=f"{NAMES[arch]}: fits up to {p['max_batch']}")
        if p.get("first_oom"):
            ax.plot([p["first_oom"]], [ys[-1]], "x", color=ARCH_COLOR[arch], ms=9, mew=2)
            ax.annotate(f"{p['first_oom']}: out of memory", (p["first_oom"], ys[-1]), xytext=(6, -12),
                        textcoords="offset points", color=INK2, fontsize=9)
        b = load_json(os.path.join(root, name, f"b{p['max_batch']}", "bench.json")) or {}
        total = total or b.get("gpu_total")
    if total:
        ax.axhline(total / 1e9, ls=":", color=INK2, lw=1.2)
        ax.annotate(f"the GPU's memory: {total / 1e9:.1f} GB", (0.01, total / 1e9), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch (sequences of 256 tokens), log scale")
    ax.set_ylabel("peak memory held by tensors (GB)")
    ax.set_title("How much memory each batch size needs")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "probe.png"), dpi=150)
    plt.close(fig)


def bench_rows(runs):
    root = os.path.join(runs, "bench")
    rows = []
    if not os.path.isdir(root):
        return rows
    for name in sorted(os.listdir(root)):
        b = load_json(os.path.join(root, name, "bench.json"))
        if b:
            rows.append((f"{NAMES[b['arch']]}, batch {b['micro_batch']}", b))
    return rows


def bars(rows, value, xlabel, title, path, fmt):
    """One series, so one color for every bar (dataviz: no value ramp on nominal categories)."""
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.4))
    labels = [r[0] for r in rows]
    vals = [value(r[1]) for r in rows]
    ax.barh(labels, vals, color=SLOT[0], height=0.55)
    for i, v in enumerate(vals):
        ax.annotate(fmt(v), (v, i), xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def recon(runs, out):
    d = load_json(os.path.join(runs, "precision_drift.json"))
    if not d:
        return
    variants = list(d["variants"])
    fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 4.4), sharey=True)
    styles = {"fixed": ("-", SLOT[0]), "fp32": ("-", SLOT[1]), "fp16": ("-", SLOT[2]), "bf16": ("-", SLOT[3])}
    floor = 1e-12
    for ax, v in zip(axes, variants):
        for sd, (ls, col) in styles.items():
            errs = d["variants"][v].get(sd)
            if errs is None:
                continue
            layers = list(range(len(errs) - 1, -1, -1))  # errs run from the top layer down
            ys = [max(e, floor) for e in errs]
            ax.plot(layers, ys, ls, color=col, marker="o", ms=4, label=sd if sd != "fixed" else "fixed-point")
        ax.set_yscale("log")
        ax.set_xlabel("layer whose input is rebuilt")
        ax.set_title(f"{v}{'' if d['variants'][v]['trained'] else ' (untrained)'}")
        ax.invert_xaxis()
    axes[0].set_ylabel("relative error of the rebuilt input")
    for ax in axes:
        ax.annotate("exactly 0 (drawn at 1e-12)", (0.5, floor), xycoords=("axes fraction", "data"), xytext=(0, 6),
                    textcoords="offset points", ha="center", color=INK2, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="streams stored as", loc="lower center", ncol=4, fontsize=9, title_fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Undoing the layers from the top down: how far each stream format drifts", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "recon.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    screening(args.runs, args.out)
    main_loss(args.runs, args.out)
    probe(args.runs, args.out)
    rows = bench_rows(args.runs)
    bars(rows, lambda b: b["peak_alloc"] / 1e9, "peak memory held by tensors (GB)", "Peak memory in one training step",
         os.path.join(args.out, "memory.png"), lambda v: f"{v:.2f} GB")
    bars(rows, lambda b: b["tok_per_s"] / 1e3, "thousand tokens per second (solo run, median step)",
         "Training speed", os.path.join(args.out, "speed.png"), lambda v: f"{v:.1f}k")
    recon(args.runs, args.out)
    print(f"wrote {', '.join(sorted(f for f in os.listdir(args.out) if f.endswith('.png')))} to {args.out}/")


if __name__ == "__main__":
    main()
