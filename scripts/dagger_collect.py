"""DAgger data for the keep/switch Jev student: the student drives, the UniTSA teacher labels (v6).

v3 distilled the student on states visited by the teacher only (scripts/sumo_collect_unitsa.py). On new
junctions the student's own mistakes lead it into states the teacher never visited, so its agreement with the
teacher drops (0.86 on validation vs 0.75 on held-out junctions). Here the student's policy (argmax, plus
epsilon-random actions) runs the episode, and every decision it faces is labelled with the teacher's
distribution; records have the format of sumo_collect_unitsa.py, so train_lora.py aggregates them with the v3 data.
Only training junctions and training patterns are used (never the test groups); random events as in v3.
One process loads the student once and runs a list of episodes:

    python scripts/dagger_collect.py --model ~/zzy/hf_models/Qwen2.5-0.5B-Instruct \
        --adapters ckpt/unitsa-jev-0.5B-full --teacher ckpt/unitsa/universal/model.zip \
        --episodes Beijing_Beihuan:easy_high_density:0.1:7 ... --out_dir data/sumo_dagger/r1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jevtraffic.sumo_env import ROOT, SumoTSCEnv, random_events  # noqa: E402
from jevtraffic.sumo_prompts import (KEEP, SWITCH, SYSTEM_PROMPT, action_to_option, option_to_action,  # noqa: E402
                                     render_compact, render_state, switch_question)
from jevtraffic.unitsa import MAX_GREEN, UniTSAAgent  # noqa: E402


def run_episode(decider, teacher_path, junction, env_name, eps, seed, out_dir, tag):
    name = f"{junction}__{env_name}__{tag}__eps{eps:g}__s{seed}"
    out = os.path.join(out_dir, f"{name}.jsonl")
    if os.path.exists(out):
        return None
    rng = random.Random(name)
    events = random_events(junction, env_name, rng, p_accident=0.6, p_special=0.8)
    work = os.path.join(ROOT, "records", "collect_dagger", name)
    env = SumoTSCEnv(junction, env_name, work, events=events, seed=seed, action_type="next_or_not")
    teacher = UniTSAAgent(env, teacher_path)
    meta = {"kind": "switch", "junction": junction, "env_name": env_name, "eps": eps, "seed": seed, "dagger": tag}
    n, agree, t0 = 0, 0, time.time()
    obs = env.reset()
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        while not env.done:
            t_action = teacher.act(obs)  # once per step: it keeps its own green timer
            if obs.green_seconds >= MAX_GREEN:
                action = 0  # max-green guard, as the controller (not asked, not recorded)
            else:
                q = switch_question(obs)
                state = render_state(obs)
                d = decider.decide(state, q)
                action = option_to_action(d.index)
                p_switch, p_keep = teacher.last_probs
                probs = [0.0, 0.0]
                probs[KEEP], probs[SWITCH] = p_keep, p_switch
                f.write(json.dumps({**meta, "t": obs.time, "state": state, "state_compact": render_compact(obs),
                                    "question": q.question, "options": list(q.options),
                                    "label": action_to_option(t_action), "probs": [round(x, 5) for x in probs],
                                    "student": action_to_option(action), "green_seconds": obs.green_seconds}) + "\n")
                n += 1
                agree += action == t_action
                if rng.random() < eps:
                    action = rng.choice([0, 1])
            obs = env.step(action)
    env.close()
    os.replace(tmp, out)
    shutil.rmtree(work, ignore_errors=True)
    return {"file": out, "records": n, "agree_teacher": round(agree / max(n, 1), 4), **env.metrics(),
            "seconds": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--teacher", required=True, help="UniTSA model.zip")
    p.add_argument("--episodes", nargs="+", required=True, help="junction:env_name:eps:seed")
    p.add_argument("--tag", default="dagger", help="round name, part of the file names")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    from jevtraffic.decider import TypedDecider

    decider = TypedDecider(args.model, device="cuda", system_prompt=SYSTEM_PROMPT, adapters=args.adapters,
                           batch_size=8)
    os.makedirs(args.out_dir, exist_ok=True)
    for ep in args.episodes:
        j, e, eps, seed = ep.split(":")
        res = run_episode(decider, args.teacher, j, e, float(eps), int(seed), args.out_dir, args.tag)
        if res:
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # libsumo + CUDA sometimes crash at interpreter teardown
