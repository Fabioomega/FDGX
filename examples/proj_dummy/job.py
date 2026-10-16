"""A fake AI project used to exercise fdgx: prepare -> train (checkpointed) -> pack / eval, plus a summary.

  python job.py prepare --item configs/a.yaml
  python job.py train   --item configs/a.yaml --steps 20 --bs 32 [--oom-above 8] [--fail-first 1] [--hang-at 5]
                        [--fail-delay 2] [--sigkill-first 1] [--vram-gb 30]
  python job.py pack    --item configs/a.yaml
  python job.py eval    --item configs/a.yaml --suite ood     (fails if the training result is missing)
  python job.py summary                                       (lists results/*.json into results/summary.json)
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def stem(item: str) -> str:
    return Path(item).stem


def prepare(a):
    Path("work").mkdir(exist_ok=True)
    print(f"preparing {a.item}", flush=True)
    time.sleep(a.step_seconds)
    Path(f"work/{stem(a.item)}.prep").write_text("ok")


def train(a):
    s = stem(a.item)
    Path("work").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"train {a.item} bs={a.bs} gpus=[{gpus}] attempt={os.environ.get('FDGX_ATTEMPT')}", flush=True)

    if a.vram_gb is not None and os.environ.get("FDGX_VRAM_FILE"):
        Path(os.environ["FDGX_VRAM_FILE"]).write_text(str(int(a.vram_gb * 1024 ** 3)))

    for flag, how in ((a.fail_first, "crash"), (a.sigkill_first, "sigkill")):
        if not flag:
            continue
        counter = Path(f"work/{s}.{how}")
        n = int(counter.read_text()) if counter.exists() else 0
        if n < flag:
            counter.write_text(str(n + 1))
            time.sleep(a.fail_delay)
            print(f"simulated {how} #{n + 1}", flush=True)
            if how == "sigkill":
                os.kill(os.getpid(), signal.SIGKILL)  # what the kernel's OOM killer does
            sys.exit(1)

    if a.oom_above is not None and a.bs > a.oom_above:
        time.sleep(a.step_seconds)
        print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", flush=True)
        sys.exit(1)

    ckpt = Path(f"work/{s}.ckpt")
    start = int(ckpt.read_text()) if ckpt.exists() else 0
    if start:
        print(f"resuming from checkpoint step {start}", flush=True)
    for step in range(start, a.steps):
        if a.hang_at is not None and step == a.hang_at:
            print("hanging (no more output)...", flush=True)
            time.sleep(10**6)
        time.sleep(a.step_seconds)
        ckpt.write_text(str(step + 1))
        print(f"step {step + 1}/{a.steps} loss {1.0 / (step + 1):.4f}", flush=True)

    out = {"item": a.item, "bs": a.bs, "gpus": gpus, "resumed_from": start, "steps": a.steps,
           "omp": os.environ.get("OMP_NUM_THREADS"), "log_file": os.environ.get("FDGX_LOG_FILE"),
           "state_dir": os.environ.get("FDGX_STATE_DIR")}
    Path(f"results/{s}.json").write_text(json.dumps(out))
    print("done", flush=True)


def pack(a):
    s = stem(a.item)
    data = Path(f"results/{s}.json").read_text()
    Path(f"results/{s}.packed").write_text(data)
    print(f"packed results/{s}.json", flush=True)


def evaluate(a):
    s = stem(a.item)
    trained = Path(f"results/{s}.json")
    if not trained.exists():
        print(f"missing training result {trained}", flush=True)
        sys.exit(1)
    for step in range(a.steps):
        time.sleep(a.step_seconds)
        print(f"eval {a.suite} {s} batch {step + 1}/{a.steps}", flush=True)
    out = {"item": a.item, "suite": a.suite, "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    Path(f"results/{s}.{a.suite}.json").write_text(json.dumps(out))


def summary(a):
    results = sorted(p.name for p in Path("results").glob("*.json") if p.name != "summary.json")
    Path("results/summary.json").write_text(json.dumps({"results": results}))
    print(f"summary of {len(results)} result file(s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train", "pack", "eval", "summary"])
    ap.add_argument("--item", default="")
    ap.add_argument("--suite", default="default")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--step-seconds", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--oom-above", type=int)
    ap.add_argument("--fail-first", type=int, default=0)
    ap.add_argument("--fail-delay", type=float, default=0.0, help="run this long before a simulated failure")
    ap.add_argument("--sigkill-first", type=int, default=0, help="SIGKILL itself on the first N attempts")
    ap.add_argument("--vram-gb", type=float, help="report this usage through $FDGX_VRAM_FILE")
    ap.add_argument("--hang-at", type=int)
    args = ap.parse_args()
    {"prepare": prepare, "train": train, "pack": pack, "eval": evaluate, "summary": summary}[args.cmd](args)
