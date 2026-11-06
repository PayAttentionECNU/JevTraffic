"""Train PressLight / AttendLight / IntelliLight (DQN, the benchmark's settings) universally or few-shot.

Universal model (zero-shot on the held-out junctions), SUMO workers on CPU, DQN update on --device:
    python scripts/train_rl_baseline.py --algo presslight --junctions train --out ckpt/presslight/universal \
        --total_timesteps 2000000 --device cuda
Few-shot adaptation on a held-out junction from the universal model, with a budget of N episodes:
    python scripts/train_rl_baseline.py --algo presslight --junctions France_Massy --episodes 10 \
        --init ckpt/presslight/universal/model.zip --out ckpt/presslight/fs10_France_Massy

Workers: one per (junction, net, copy); each samples a training pattern on every reset (random_perturbation is
never trained on). Few-shot runs use min(episodes, --copies x nets) workers and stop after `episodes` episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jevtraffic.sumo_env import HELDOUT_PATTERN, JUNCTIONS, NETS, PATTERNS, TRAIN_JUNCTIONS  # noqa: E402
from jevtraffic import rl_baselines as rb  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=sorted(rb.ALGOS))
    p.add_argument("--junctions", nargs="+", default=["train"], help="'train' = the 9 training junctions")
    p.add_argument("--nets", nargs="+", default=NETS)
    p.add_argument("--copies", type=int, default=2, help="workers per (junction, net)")
    p.add_argument("--total_timesteps", type=int, default=2_000_000)
    p.add_argument("--episodes", type=int, default=0, help="few-shot budget: stop after this many episodes")
    p.add_argument("--init", default=None, help="model.zip to adapt (few-shot)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu", help="DQN update device; SUMO workers always run on CPU")
    p.add_argument("--checkpoint_every", type=int, default=200_000)
    args = p.parse_args()

    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    torch.set_num_threads(args.threads)
    junctions = TRAIN_JUNCTIONS if args.junctions == ["train"] else args.junctions
    assert all(j in JUNCTIONS for j in junctions), junctions
    patterns = [q for q in PATTERNS if q != HELDOUT_PATTERN]
    os.makedirs(os.path.join(args.out, "monitor"), exist_ok=True)
    fns, k = [], 0
    for c in range(args.copies):
        for j in junctions:
            for net in args.nets:
                fns.append(rb.train_env(args.algo, j, net, patterns, seed=args.seed * 1000 + k,
                                        log_file=os.path.join(args.out, "monitor", f"{j}_{net}_{c}")))
                k += 1
    if args.episodes:
        fns = fns[:max(1, min(args.episodes, len(fns)))]
    env = SubprocVecEnv(fns)
    if rb.ALGOS[args.algo]["vec_normalize"]:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.99)
        vn = args.init and os.path.join(os.path.dirname(args.init), "vec_normalize.pkl")
        if vn and os.path.exists(vn):  # keep the reward scale the Q-network was trained on
            old = VecNormalize.load(vn, SubprocVecEnv([fns[0]]))
            env.ret_rms = old.ret_rms
            old.close()

    params = rb.dqn_params(len(fns), few_shot=bool(args.episodes))
    if args.init:
        model = DQN.load(args.init, env=env, device=args.device, custom_objects={"policy_class": rb.masked_policy()},
                         **params)  # fresh replay buffer
    else:
        policy_kwargs = dict(features_extractor_class=rb.model_class(args.algo),
                             features_extractor_kwargs=rb.ALGOS[args.algo]["model_kwargs"])
        model = DQN(rb.masked_policy(), env, verbose=1, policy_kwargs=policy_kwargs, device=args.device, seed=args.seed,
                    **params)
    meta = {"algo": args.algo, "junctions": junctions, "nets": args.nets, "patterns": patterns, "workers": len(fns),
            "total_timesteps": args.total_timesteps, "episodes": args.episodes, "init": args.init, "dqn": params,
            "history_len": rb.ALGOS[args.algo]["history_len"], "phase_pad": rb.PHASE_PAD}
    json.dump(meta, open(os.path.join(args.out, "train_meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)
    t0 = time.time()
    callbacks = [CheckpointCallback(save_freq=max(args.checkpoint_every // len(fns), 1),
                                    save_path=os.path.join(args.out, "checkpoints"))]
    if args.episodes:
        callbacks = [rb.episode_budget_callback(args.episodes)]
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=not args.init)
    model.save(os.path.join(args.out, "model.zip"))
    if isinstance(env, VecNormalize):
        env.save(os.path.join(args.out, "vec_normalize.pkl"))
    meta["train_s"] = round(time.time() - t0, 1)
    meta["timesteps_used"] = int(model.num_timesteps)
    json.dump(meta, open(os.path.join(args.out, "train_meta.json"), "w"), indent=1)
    print("done", meta["train_s"], flush=True)
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)
