"""Choose an RL checkpoint of scripts/rl_jev.py on validation scenarios, never on the test groups.

Validation = the 9 training junctions x VAL_ENVS (training patterns only) at SUMO seed 100; the test results use
seeds 0-2, the held-out pattern and the held-out junctions. Candidates are every update<k>/ of every --runs dir
plus each run's final adapter; update0 is the starting policy itself, so the chosen checkpoint is never worse
than not doing RL on validation. The winner is copied to --out (adapter + critic.pt + normalizer.json) with
selection.json listing every candidate's mean validation ATT.

    python scripts/select_rl_checkpoint.py --model ~/zzy/hf_models/Qwen2.5-0.5B-Instruct \
        --adapters ckpt/unitsa-jev-0.5B-full --runs ckpt/rl5a-0.5B-full ckpt/rl5b-0.5B-full \
        --gpus 0 1 2 3 4 5 --out ckpt/rl5-0.5B-full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jevtraffic.sumo_env import TRAIN_JUNCTIONS  # noqa: E402

VAL_ENVS = ["easy_high_density", "normal_high_density", "easy_fluctuating_commuter", "normal_increasing_demand"]
VAL_SEED = 100


def candidates(runs):
    out = []
    for r in runs:
        ups = sorted(glob.glob(os.path.join(r, "update*")), key=lambda d: int(d.rsplit("update", 1)[1]))
        out += [d for d in ups if os.path.exists(os.path.join(d, "adapter_model.safetensors"))]
        if os.path.exists(os.path.join(r, "adapter_model.safetensors")):
            out.append(r)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapters", nargs="*", default=[], help="adapters under the RL one (the distilled student)")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--state_format", default="full")
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--slots", type=int, default=3, help="processes per GPU")
    p.add_argument("--junctions", nargs="+", default=TRAIN_JUNCTIONS, help="(smoke tests: fewer)")
    p.add_argument("--envs", nargs="+", default=VAL_ENVS)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cands = candidates(args.runs)
    val_root = os.path.join(ROOT, "results", "sumo", "val_rl")
    jobs = queue.Queue()
    for c in cands:
        for j in args.junctions:
            jobs.put((c, j))
    print(f"{len(cands)} candidates x {len(args.junctions)} junctions x {len(args.envs)} envs", flush=True)

    def out_dir(c):
        return os.path.join(val_root, os.path.relpath(c, ROOT).replace("/", "__"))

    def worker(gpu):
        while True:
            try:
                c, j = jobs.get_nowait()
            except queue.Empty:
                return
            cmd = [os.path.join(ROOT, ".venv", "bin", "python"), os.path.join(ROOT, "run_sumo.py"), "--junction", j,
                   "--env_name", *args.envs, "--agent", "jev", "--action_type", "next_or_not", "--state_format",
                   args.state_format, "--model", args.model, "--adapters", *args.adapters, c, "--seed",
                   str(VAL_SEED), "--out_dir", out_dir(c), "--skip_existing"]
            log = os.path.join(ROOT, "logs", "val_rl", f"{os.path.basename(out_dir(c))}_{j}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus for _ in range(args.slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    scores = {}
    for c in cands:
        atts = [json.load(open(f))["metrics"].get("ATT") for f in glob.glob(os.path.join(out_dir(c), "*.json"))]
        atts = [a for a in atts if a]
        scores[os.path.relpath(c, ROOT)] = {"ATT": round(sum(atts) / len(atts), 3) if atts else None,
                                           "episodes": len(atts)}
    full = len(args.junctions) * len(args.envs)
    ok = {c: s for c, s in scores.items() if s["episodes"] == full}
    best = min(ok, key=lambda c: ok[c]["ATT"])
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    shutil.copytree(os.path.join(ROOT, best), args.out, ignore=shutil.ignore_patterns("update*"))
    for f in ("rl_meta.json",):  # the run's settings travel with the chosen checkpoint
        src = os.path.join(ROOT, best.split("/update")[0], f)
        if os.path.exists(src):
            shutil.copy(src, args.out)
    sel = {"selected": best, "val_envs": args.envs, "val_seed": VAL_SEED, "junctions": args.junctions,
           "scores": scores}
    json.dump(sel, open(os.path.join(args.out, "selection.json"), "w"), indent=1)
    for c, s in scores.items():
        print(f"{s['ATT']}\t{s['episodes']}\t{c}{'  <- selected' if c == best else ''}")


if __name__ == "__main__":
    main()
