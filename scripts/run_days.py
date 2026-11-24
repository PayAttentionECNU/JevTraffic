"""E5: long-term memory and optional per-junction fine-tuning over simulated days (one junction per process).

Every day: generate the day's demand (days.py), run MaxPressure on it (reference) and the deployed Jev model,
store the decisions in long-term memory, and let the adaptation policy decide whether to fine-tune a junction
LoRA (initialised from the shared adapter, trained on the junction's last `window` days with MaxPressure labels).
A candidate is deployed only if it beats the current model on a re-perturbed replay of that day.
The decider is freed while training, so one GPU suffices.

    CUDA_VISIBLE_DEVICES=0 python scripts/run_days.py --junction France_Massy --adapt auto --shared ckpt/sumo-distill
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jevtraffic.days import make_day  # noqa: E402
from jevtraffic.case_memory import CaseMemory  # noqa: E402
from jevtraffic.longterm import AdaptationPolicy, LongTermMemory  # noqa: E402
from jevtraffic.sumo_agents import Controller, mp_action  # noqa: E402
from jevtraffic.sumo_env import ROOT, SumoTSCEnv  # noqa: E402


def episode(junction, info, work, policy, seed, record=False, cases=None):
    env = SumoTSCEnv(junction, info["env_name"], work, route_file=info["route_file"], seed=seed)
    obs = env.reset()
    ctl = Controller(env, base="jev", decider=policy, record_states=record, case_memory=cases) if policy is not None else None
    while not env.done:
        before = obs
        obs = env.step(ctl.act(obs) if ctl else mp_action(obs))
        if cases:
            cases.record(before, obs)
    env.close()
    return env.metrics(), ctl


def load_decider(model, adapters):
    from jevtraffic.decider import TypedDecider
    from jevtraffic.sumo_prompts import SYSTEM_PROMPT

    dec = TypedDecider(model, system_prompt=SYSTEM_PROMPT, adapters=adapters, batch_size=8)
    calib = os.path.join(adapters[0], "calib.json")  # temperatures of the shared model
    if os.path.exists(calib):
        d = json.load(open(calib))
        dec.temperatures.update({k: d[k] for k in ("phase", "situation") if k in d})
    return dec


def release():
    """Return GPU memory of deciders whose last reference the caller has just dropped."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--junction", required=True)
    p.add_argument("--net", default="normal")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--adapt", default="off", choices=["off", "daily", "weekly", "auto"])
    p.add_argument("--case_mode", default="off", choices=["off", "collect", "read", "online"])
    p.add_argument("--case_store", help="case archive shared by collection and read-only evaluation")
    p.add_argument("--shared", required=True, help="shared (all-junction) adapter")
    p.add_argument("--model", default=os.path.expanduser("~/zzy/hf_models/Qwen2.5-7B-Instruct"))
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "sumo", "days"))
    args = p.parse_args()

    tag = f"{args.junction}__{args.net}__{args.adapt}"
    run_group = args.adapt
    if args.case_mode != "off":
        tag += f"__cases_{args.case_mode}"
        run_group += f"_{args.net}_cases_{args.case_mode}"
    default_source = "collect" if args.case_mode == "read" else args.case_mode
    case_root = args.case_store or os.path.join(ROOT, "records", "day_cases", args.adapt, default_source)
    mem = LongTermMemory(os.path.join(ROOT, "records", "longterm", run_group), args.junction)
    for f in (mem.decisions_path, mem.days_path):
        if os.path.exists(f):
            os.remove(f)
    policy = AdaptationPolicy(args.adapt, window=args.window)
    routes = os.path.join(ROOT, "records", "days_routes")
    work = os.path.join(ROOT, "records", "days_work", tag)
    deployed = None
    dec = load_decider(args.model, [args.shared])
    rows = []
    for day in range(args.days):
        info = make_day(args.junction, args.net, day, routes)
        mp_m, _ = episode(args.junction, info, work, None, seed=0)
        cases = (CaseMemory(case_root, args.junction, args.net, "choose_next_phase", day, args.case_mode)
                 if args.case_mode != "off" else None)
        m, ctl = episode(args.junction, info, work, dec, seed=0, record=True, cases=cases)
        n = mem.log_decisions(day, ctl.steps)
        s = ctl.summary()
        ctl = None  # the controller holds a reference to the decider
        row = {"day": day, "pattern": info["pattern"], "scale": info["scale"], "vehicles": info["vehicles"],
               "ATT": m["ATT"], "AWT": m["AWT"], "mp_ATT": mp_m["ATT"], "mp_AWT": mp_m["AWT"],
               "agree_mp": s.get("agree_mp"), "mean_conf": s.get("mean_conf"), "ms_per_call": s.get("ms_per_call"),
               "decisions": n, "deployed": os.path.basename(deployed) if deployed else "shared"}
        if cases:
            cases.publish()
            row["case_memory"] = s["case_memory"]
        adapt, why = policy.should_adapt(day, rows + [row])
        row["adapt_reason"] = why
        if adapt:
            t0 = time.time()
            cand = mem.adapter_dir(day)
            train = mem.training_file(day, args.window)
            dec = None
            release()
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "train_lora.py"), "--model", args.model,
                   "--train", train, "--init_adapter", args.shared, "--out", cand, "--rank", str(args.rank),
                   "--alpha", str(2 * args.rank), "--epochs", str(args.epochs), "--lr", str(args.lr),
                   "--batch_size", "8", "--save_every", "0", "--log_every", "10"]
            subprocess.run(cmd, check=True, stdout=open(os.path.join(mem.dir, f"train_day{day:03d}.log"), "w"),
                           stderr=subprocess.STDOUT)
            train_s = time.time() - t0
            # replay a re-perturbed version of the day (other demand jitter / thinning; the SUMO seed alone barely
            # changes these scenarios): current vs candidate
            replay = make_day(args.junction, args.net, day, routes + "_replay", seed=1)
            # Both validation arms see exactly the same earlier days. Replays never publish.
            frozen = (CaseMemory(case_root, args.junction, args.net, "choose_next_phase", day, "read")
                      if args.case_mode in ("online", "read") else None)
            cur = load_decider(args.model, [args.shared] + ([deployed] if deployed else []))
            cur_m, _ = episode(args.junction, replay, work, cur, seed=0, cases=frozen)
            cur = None
            release()
            cand_dec = load_decider(args.model, [args.shared, cand])
            cand_m, _ = episode(args.junction, replay, work, cand_dec, seed=0, cases=frozen)
            accept = cand_m["ATT"] < cur_m["ATT"]
            row.update(adapted=True, train_records=sum(1 for _ in open(train)), train_s=round(train_s, 1),
                       replay_current_ATT=cur_m["ATT"], replay_candidate_ATT=cand_m["ATT"], accepted=accept)
            if accept:
                if deployed:
                    shutil.rmtree(deployed, ignore_errors=True)
                deployed, dec, cand_dec = cand, cand_dec, None
            else:
                shutil.rmtree(cand, ignore_errors=True)
                cand_dec = None
                release()
                dec = load_decider(args.model, [args.shared] + ([deployed] if deployed else []))
            row["adapt_s"] = round(time.time() - t0, 1)
        row["storage_MB"] = round(mem.storage_bytes() / 2**20, 2)
        mem.log_day(row)
        rows.append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, tag + ".json"), "w") as f:
        json.dump({"junction": args.junction, "net": args.net, "adapt": args.adapt, "shared": args.shared,
                   "days": rows}, f, indent=1)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # skip interpreter teardown: libsumo + CUDA sometimes crash there after all results are written
