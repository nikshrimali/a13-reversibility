"""Spread training runs over the GPUs this machine has: one run per GPU at a time, the next run starts when a GPU frees up.

    from jobs import detect_gpus, run_queue
    gpus = detect_gpus("auto")          # Kaggle 2x T4: [0, 1]; Colab: [0]; Mac or CPU: [None]
    run_queue([Job("base", ["--arch", "gpt", ...], "runs/base")], gpus)

Each job is `python train.py <args> --out <out> --resume`, started with CUDA_VISIBLE_DEVICES set to
its GPU, so inside the process that GPU is simply "cuda:0" and train.py needs no device logic.
Every job's output goes to <out>/log.txt. A job whose done file (done.json for a training run,
bench.json for a bench, probe.json for a probe) already exists is skipped, which is what makes re-running the notebook after
a dead session pick up where it left off; an unfinished run resumes from its checkpoint.

While jobs run, the queue prints each one's latest step line every `report_every` seconds.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field


@dataclass
class Job:
    """One run of a script (train.py unless said otherwise).

    Attributes:
        name: short label for the printout.
        args: the script's arguments, without --out.
        out: output folder.
        done_file: the file in `out` whose existence means the job is finished.
        resume: pass --resume (training runs only).
        script: the script to run.
        returncode: filled in by run_queue. 0 means finished, 3 means out of memory.
    """

    name: str
    args: list
    out: str
    done_file: str = "done.json"
    resume: bool = True
    script: str = "train.py"
    returncode: int = None
    seconds: float = None
    gpu: object = field(default=None, repr=False)

    def done(self):
        return os.path.exists(os.path.join(self.out, self.done_file))


def bench_job(name, args, out):
    """A --bench-steps run: finished when bench.json exists, never resumed."""
    return Job(name, args, out, done_file="bench.json", resume=False)


def probe_job(name, args, out):
    """A probe.py max-batch search: finished when probe.json exists."""
    return Job(name, args, out, done_file="probe.json", resume=False, script="probe.py")


def detect_gpus(spec="auto"):
    """Turn the GPUS parameter into a list of device ids.

    Args:
        spec: "auto" (every CUDA GPU, or one Apple GPU / CPU worker), an int (use the first n
            CUDA GPUs) or a list of CUDA ids.

    Returns:
        A list of CUDA device ids, or [None] when there is no CUDA GPU (one worker on MPS or CPU).
    """
    import torch

    n = torch.cuda.device_count()
    if spec == "auto":
        return list(range(n)) if n else [None]
    if isinstance(spec, int):
        return list(range(min(spec, n))) if n else [None]
    return list(spec)


def _last_line(path, key="step "):
    """The last line of a log that starts with `key`, or ''."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            lines = f.read().decode(errors="replace").splitlines()
        return next((l for l in reversed(lines) if l.startswith(key)), "")
    except OSError:
        return ""


def run_queue(jobs, gpus, report_every=60, python=sys.executable):
    """Run jobs, one per GPU at a time, until all are finished. Returns the jobs with returncodes.

    Jobs that are already done are skipped (returncode 0). A failed job does not stop the queue;
    its returncode is non-zero and the last lines of its log are printed.
    """
    todo = []
    for j in jobs:
        if j.done():
            j.returncode = 0
            print(f"[skip] {j.name}: already finished in {j.out}")
        else:
            todo.append(j)
    free, running = list(gpus), {}
    last_report = time.time()
    while todo or running:
        while todo and free:
            j, g = todo.pop(0), free.pop(0)
            os.makedirs(j.out, exist_ok=True)
            env = dict(os.environ)
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            cmd = [python, j.script] + [str(a) for a in j.args] + ["--out", j.out] + (["--resume"] if j.resume else [])
            log = open(os.path.join(j.out, "log.txt"), "a")
            j.gpu, j.seconds = g, time.time()
            running[j.name] = (j, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log)
            print(f"[start] {j.name} on {'GPU ' + str(g) if g is not None else 'the local device'}", flush=True)
        time.sleep(2)
        for name in list(running):
            j, p, log = running[name]
            if p.poll() is None:
                continue
            log.close()
            j.returncode, j.seconds = p.returncode, time.time() - j.seconds
            free.append(j.gpu)
            del running[name]
            status = "finished" if p.returncode == 0 else ("OUT OF MEMORY" if p.returncode == 3 else f"FAILED (exit {p.returncode})")
            print(f"[{'done' if p.returncode == 0 else 'fail'}] {j.name} {status} in {j.seconds / 60:.1f} min", flush=True)
            if p.returncode not in (0, 3):
                with open(os.path.join(j.out, "log.txt")) as f:
                    print("".join(f.readlines()[-30:]))
        if running and time.time() - last_report > report_every:
            last_report = time.time()
            for j, _, _ in running.values():
                key = "  batch" if j.script == "probe.py" else "step "
                print(f"  {j.name:<16} {_last_line(os.path.join(j.out, 'log.txt'), key).strip() or 'starting...'}", flush=True)
    return jobs


def run_script(args, gpu=None, python=sys.executable):
    """Run one script to the end on one GPU, printing its output as it comes. Returns the exit code.

    Used for the short scripts (gradcheck.py, precision.py) so the notebook's own process never
    touches the GPU: a CUDA context in the notebook would hold memory on GPU 0 for the rest of
    the session and skew every memory number measured there afterwards.
    """
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    p = subprocess.Popen([python] + [str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=env, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()
