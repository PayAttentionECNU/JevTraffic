"""One SingleTSCBaselines (SUMO) episode with one controller; writes results/sumo/runs/<run>.json.

  python run_sumo.py --junction France_Massy --env_name normal_low_density --agent maxpressure
  python run_sumo.py --junction France_Massy --env_name normal_low_density --agent mp --detector rule --events event_1
  python run_sumo.py ... --agent jev --adapters ckpt/sumo-distill --memory triggered --detector jev --events event_1
  python run_sumo.py --junction France_Massy --env_name normal_low_density easy_high_density --events none event_1 \
      --adapters ckpt/sumo-distill --variants jev:none:none jev:triggered:jev     (model loaded once)

--agent fixtime|webster|sotl|maxpressure  the benchmark's own controllers (unchanged)
--agent mp | jev                          our controller with MaxPressure / fine-tuned Jev as the phase policy,
                                          plus --memory / --detector for context and special events
--events none | event_1 | random:<seed>   the junction's official event set, or random accidents / special vehicles
tshub keeps per-process state, so run one junction per process (this script does).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

from jevtraffic.sumo_env import SumoTSCEnv, random_events

ROOT = os.path.dirname(os.path.abspath(__file__))
RL_BASELINES = ("presslight", "attendlight", "intellilight")  # jevtraffic/rl_baselines.py


def run_label(args) -> str:
    if args.agent == "unitsa":
        return f"unitsa[{os.path.basename(os.path.dirname(os.path.abspath(args.unitsa_model)))}]"
    if args.agent in RL_BASELINES:
        return f"{args.agent}[{os.path.basename(os.path.dirname(os.path.abspath(args.rl_model)))}]"
    if args.agent not in ("mp", "jev"):
        return args.agent
    label = args.agent
    if args.action_type == "next_or_not":
        label += "-ns"  # keep/switch (UniTSA's action space)
    if args.state_format != "full":
        label += f"-{args.state_format}"
    if args.detector != "none":
        label += f"-{args.memory}-{args.detector}"
    if args.agent == "jev" or args.detector == "jev":
        tag = "+".join(os.path.basename(a.rstrip("/")) for a in args.adapters) or os.path.basename(args.model)
        if args.adapters and "7B" not in args.model:
            tag = f"{os.path.basename(args.model)}+{tag}"  # the backbone is part of a small Jev's identity; no ":" (SUMO reads host:port)
        label += f"[{tag}]"
    return label


def load_temperatures(adapters) -> dict:
    """calib.json of the last adapter that has one (an RL adapter on top of a distilled one keeps its temperatures)."""
    for a in reversed(adapters or []):
        path = os.path.join(a, "calib.json")
        if os.path.exists(path):
            d = json.load(open(path))
            return {k: d[k] for k in ("phase", "situation", "switch") if k in d}
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--junction", required=True)
    p.add_argument("--env_name", nargs="+", required=True, help="one or more envs of this junction")
    p.add_argument("--agent", default="mp",
                   choices=["fixtime", "webster", "sotl", "maxpressure", "mp", "jev", "unitsa", *RL_BASELINES])
    p.add_argument("--unitsa_model", default=None, help="UniTSA model.zip (scripts/train_unitsa.py)")
    p.add_argument("--rl_model", default=None, help="PressLight / AttendLight / IntelliLight model.zip "
                                                    "(scripts/train_rl_baseline.py)")
    p.add_argument("--action_type", default="choose_next_phase", choices=["choose_next_phase", "next_or_not"],
                   help="mp/jev: phase choice (v2) or keep/switch (v3); unitsa always uses next_or_not")
    p.add_argument("--state_format", default="full", choices=["full", "compact"])
    p.add_argument("--teacher", default=None, help="UniTSA model.zip run alongside jev (next_or_not) to log agreement")
    p.add_argument("--prefix_cache", action="store_true", help="reuse the KV cache of the fixed prompt prefix")
    p.add_argument("--compile", action="store_true", help="CUDA-graph the forward pass (torch.compile)")
    p.add_argument("--memory", default="triggered", choices=["none", "always", "triggered"])
    p.add_argument("--detector", default="none", choices=["none", "rule", "jev"])
    p.add_argument("--variants", nargs="*", default=[],
                   help="agent:memory:detector combos (e.g. jev:none:none jev:triggered:jev); overrides the three above")
    p.add_argument("--events", nargs="+", default=["none"])
    p.add_argument("--route_file", default=None, help="day route file (days.py) instead of the pattern's file")
    p.add_argument("--model", default=os.path.expanduser("~/zzy/hf_models/Qwen2.5-7B-Instruct"))
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--tau_event", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0, help="SUMO seed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "sumo", "runs"))
    p.add_argument("--tag", default="")
    p.add_argument("--save_steps", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    variants = [v.split(":") for v in args.variants] or [[args.agent, args.memory, args.detector]]
    decider = None  # loaded once, shared by all episodes of this process
    for env_name in args.env_name:
        for events in args.events:
            for agent, memory, detector in variants:
                a = argparse.Namespace(**{**vars(args), "env_name": env_name, "events": events, "agent": agent,
                                          "memory": memory, "detector": detector})
                if (a.agent == "jev" or a.detector == "jev") and decider is None:
                    from jevtraffic.decider import TypedDecider
                    from jevtraffic.sumo_prompts import SYSTEM_PROMPT

                    decider = TypedDecider(a.model, device=a.device, system_prompt=SYSTEM_PROMPT,
                                           adapters=a.adapters, batch_size=8, prefix_cache=a.prefix_cache, compile=a.compile)
                    decider.temperatures.update(load_temperatures(a.adapters))
                run_one(a, decider)


def run_one(args, decider):
    label = run_label(args)
    run = f"{args.junction}__{args.env_name}__{label}__ev-{args.events.replace(':', '')}__s{args.seed}"
    run += f"__{args.tag}" if args.tag else ""
    out = os.path.join(args.out_dir, run.replace("/", "_") + ".json")
    if args.skip_existing and os.path.exists(out):
        print("exists", out)
        return

    events = None
    if args.events.startswith("random"):
        rng = random.Random(f"{args.junction}|{args.env_name}|{args.events}")
        events = random_events(args.junction, args.env_name, rng, p_accident=1.0, p_special=1.0)  # eval: both kinds
    elif args.events != "none":
        events = args.events
    work = os.path.join(ROOT, "records", "sumo_work", run.replace("/", "_"))
    env = SumoTSCEnv(args.junction, args.env_name, work, events=events, route_file=args.route_file, seed=args.seed,
                     action_type="next_or_not" if args.agent == "unitsa" else args.action_type)

    obs = env.reset()
    if args.agent in ("mp", "jev"):
        from jevtraffic.sumo_agents import Controller

        notify = os.path.join(args.out_dir, "..", "events", run.replace("/", "_") + ".jsonl")
        teacher = None
        if args.teacher and args.action_type == "next_or_not":
            from jevtraffic.unitsa import UniTSAAgent

            teacher = UniTSAAgent(env, args.teacher)
        agent = Controller(env, base=args.agent, decider=decider, memory=args.memory, detector=args.detector,
                           notify_path=notify if args.detector != "none" else None, tau_event=args.tau_event,
                           state_format=args.state_format, teacher=teacher)
    elif args.agent == "unitsa":
        from jevtraffic.unitsa import UniTSAAgent

        agent = UniTSAAgent(env, args.unitsa_model)
    elif args.agent in RL_BASELINES:
        from jevtraffic.rl_baselines import RLBaselineAgent

        agent = RLBaselineAgent(env, args.agent, args.rl_model)
    else:
        from jevtraffic.sumo_agents import BenchmarkAgent

        agent = BenchmarkAgent(env, args.agent)
    t0 = time.time()
    while not env.done:
        obs = env.step(agent.act(obs))
    wall = time.time() - t0
    env.close()

    result = {"junction": args.junction, "env_name": args.env_name, "agent": label, "events": args.events,
              "seed": args.seed, "route_file": args.route_file, "decisions": env.steps, "wall_s": round(wall, 1),
              "metrics": env.metrics(), "summary": agent.summary()}
    if args.agent in ("mp", "jev"):
        from jevtraffic.sumo_agents import score_detection

        result["notifications"] = agent.notifier.events
        if args.detector != "none":
            result["detection"] = score_detection(env, agent.notifier.events)
        if args.save_steps:
            result["steps"] = agent.steps
    result["ground_truth"] = {"accidents": env.accidents, "special_vehicles": env.specials, "v2x_seen": env.v2x_seen}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f)
    shutil.rmtree(work, ignore_errors=True)  # tripinfo is summarised in the result; the server disk is shared
    print(json.dumps({k: result[k] for k in ("agent", "events", "metrics", "summary")} |
                     {"env_name": args.env_name, "detection": result.get("detection")}), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # skip interpreter teardown: libsumo + CUDA sometimes crash there after all results are written
