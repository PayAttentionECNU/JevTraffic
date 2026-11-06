"""Validation gate for few-shot adaptation: keep an adapted model only if it beats the zero-shot one on validation.

For every learning method, budget N and held-out junction J, the adapted model (fs<N>_<J>) and the zero-shot model
are run on J x VAL_ENVS (training patterns, no events) at SUMO seed 100, the validation seed of
scripts/select_rl_checkpoint.py; the test results use seeds 0-2. If the adapted model is not better, the deployed
model for (method, N, J) is the zero-shot one, so a few-shot budget is never worse than none on validation.
The decisions go to results/sumo/fewshot_gate.json, which scripts/main_tables.py applies to Table 2.

    python scripts/fewshot_gate.py --gpus 0 1 2 3 4 5 6 7 --cpu_jobs 32
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from jevtraffic.sumo_env import HELDOUT_JUNCTIONS  # noqa: E402
from select_rl_checkpoint import VAL_ENVS, VAL_SEED  # noqa: E402

PY = os.path.join(ROOT, ".venv", "bin", "python")
VAL_ROOT = os.path.join(ROOT, "results", "sumo", "val_fs")
DQN = {"presslight": "PressLight", "attendlight": "AttendLight", "intellilight": "IntelliLight"}


def out_dir(path):
    return os.path.join(VAL_ROOT, os.path.relpath(path, ROOT).replace("/", "__"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", nargs="+", type=int, default=[1, 10, 50])
    p.add_argument("--junctions", nargs="+", default=HELDOUT_JUNCTIONS)
    p.add_argument("--model", default=os.path.expanduser("~/zzy/hf_models/Qwen2.5-0.5B-Instruct"))
    p.add_argument("--student", default="ckpt/unitsa-jev-0.5B-full")
    p.add_argument("--rl", default="ckpt/rl5-0.5B-full")
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--slots", type=int, default=3)
    p.add_argument("--cpu_jobs", type=int, default=32)
    args = p.parse_args()

    # (row names, budget, junction, candidate path, command builder); budget 0 = the zero-shot model
    cands, cpu, gpu = [], queue.Queue(), queue.Queue()
    for algo, row in [*DQN.items(), ("unitsa", "UniTSA")]:
        flag = "--unitsa_model" if algo == "unitsa" else "--rl_model"
        for j in args.junctions:
            for b in [0, *args.budgets]:
                path = os.path.join(ROOT, "ckpt", algo, "universal" if b == 0 else f"fs{b}_{j}", "model.zip")
                cands.append(([row], b, j, path))
                cpu.put([PY, "run_sumo.py", "--junction", j, "--env_name", *VAL_ENVS, "--agent", algo, flag, path,
                         "--seed", str(VAL_SEED), "--out_dir", out_dir(path), "--skip_existing"])
    jev_rows = ["Jev (distilled + RL)", "Jev (distilled + RL) + event plan"]
    for j in args.junctions:
        for b in [0, *args.budgets]:
            path = os.path.join(ROOT, args.rl if b == 0 else f"{args.rl}-fs{b}_{j}")
            adapters = [args.student, args.rl] + ([] if b == 0 else [path])
            cands.append((jev_rows, b, j, path))
            gpu.put([PY, "run_sumo.py", "--junction", j, "--env_name", *VAL_ENVS, "--agent", "jev", "--action_type",
                     "next_or_not", "--state_format", "full", "--model", args.model, "--adapters", *adapters,
                     "--seed", str(VAL_SEED), "--out_dir", out_dir(path), "--skip_existing"])

    def worker(q, env):
        while True:
            try:
                cmd = q.get_nowait()
            except queue.Empty:
                return
            log = os.path.join(ROOT, "logs", "val_fs", os.path.basename(cmd[cmd.index("--out_dir") + 1])
                               + f"_{cmd[3]}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, **env})

    threads = [threading.Thread(target=worker, args=(cpu, {"CUDA_VISIBLE_DEVICES": ""}))
               for _ in range(args.cpu_jobs)]
    threads += [threading.Thread(target=worker, args=(gpu, {"CUDA_VISIBLE_DEVICES": g}))
                for g in args.gpus for _ in range(args.slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    def val_att(path, j):
        atts = [json.load(open(f))["metrics"].get("ATT") for f in glob.glob(os.path.join(out_dir(path), f"{j}__*.json"))]
        atts = [a for a in atts if a]
        return round(sum(atts) / len(atts), 3) if len(atts) == len(VAL_ENVS) else None

    zero = {(tuple(rows), j): val_att(path, j) for rows, b, j, path in cands if b == 0}
    gate = {}
    for rows, b, j, path in cands:
        if b == 0:
            continue
        z, a = zero[(tuple(rows), j)], val_att(path, j)
        use = "adapted" if a is not None and z is not None and a < z else "zero"
        for row in rows:
            gate[f"{row}|{b}|{j}"] = {"adapted": a, "zero": z, "use": use}
        print(f"{rows[0]:22s} fs{b:<3d} {j:18s} adapted {a}  zero {z}  -> {use}", flush=True)
    meta = {"val_envs": VAL_ENVS, "val_seed": VAL_SEED, "gate": gate}
    json.dump(meta, open(os.path.join(ROOT, "results", "sumo", "fewshot_gate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
