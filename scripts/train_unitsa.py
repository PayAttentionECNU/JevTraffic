"""Train UniTSA (PPO; SUMO workers on CPU, update on --device) on one or many junctions with SingleTSCBaselines' settings.

Universal model (zero-shot on the held-out junctions):
    python scripts/train_unitsa.py --junctions train --out ckpt/unitsa/universal --total_timesteps 2000000
Per-junction fine-tune from the universal model (the "UniTSA fine-tuned" upper bound):
    python scripts/train_unitsa.py --junctions France_Massy --init ckpt/unitsa/universal/model.zip \
        --out ckpt/unitsa/ft_France_Massy --total_timesteps 300000

Few-shot adaptation with a budget of N episodes (stops after N finished episodes; PPO rollouts shortened to fit):
    python scripts/train_unitsa.py --junctions France_Massy --episodes 10 --init ckpt/unitsa/universal/model.zip \
        --out ckpt/unitsa/fs10_France_Massy

Workers: one per (junction, net, copy); each samples a training pattern on every reset. The held-out pattern
(random_perturbation) is never trained on. Few-shot runs use min(episodes, workers) workers. Observations are
phase-padded (jevtraffic.unitsa.PadPhases) so every model, universal or per junction, has the same input shape.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jevtraffic.sumo_env import HELDOUT_PATTERN, NETS, PATTERNS, TRAIN_JUNCTIONS, JUNCTIONS  # noqa: E402
from jevtraffic import unitsa  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--junctions", nargs="+", default=["train"], help="'train' = the 9 training junctions")
    p.add_argument("--nets", nargs="+", default=NETS)
    p.add_argument("--copies", type=int, default=2, help="workers per (junction, net)")
    p.add_argument("--total_timesteps", type=int, default=2_000_000)
    p.add_argument("--episodes", type=int, default=0, help="few-shot budget: stop after this many episodes")
    p.add_argument("--init", default=None, help="model.zip to continue from (per-junction fine-tuning)")
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu", help="PPO update device; SUMO workers always run on CPU")
    p.add_argument("--checkpoint_every", type=int, default=200_000)
    args = p.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from tsc_algos.rl.unitsa.model import UniTSAMovementTransformer

    torch.set_num_threads(args.threads)
    junctions = TRAIN_JUNCTIONS if args.junctions == ["train"] else args.junctions
    assert all(j in JUNCTIONS for j in junctions), junctions
    patterns = [q for q in PATTERNS if q != HELDOUT_PATTERN]
    os.makedirs(os.path.join(args.out, "monitor"), exist_ok=True)
    fns, k = [], 0
    for c in range(args.copies):  # copies outermost: a few-shot prefix of the list mixes nets
        for j in junctions:
            for net in args.nets:
                fns.append(unitsa.train_env(j, net, patterns, seed=args.seed * 1000 + k,
                                            log_file=os.path.join(args.out, "monitor", f"{j}_{net}_{c}")))
                k += 1
    if args.episodes:
        fns = fns[:max(1, min(args.episodes, len(fns)))]
    env = VecNormalize(SubprocVecEnv(fns), norm_obs=False, norm_reward=True, gamma=0.99)

    params = dict(unitsa.PPO_PARAMS)
    if args.learning_rate:
        params["learning_rate"] = args.learning_rate
    if args.episodes:  # ~200 decisions per episode: update every 64 steps per worker instead of 256; conservative
        # (v5): learning rate 1e-5 and 2 epochs per rollout instead of 1e-4 and 10, so it cannot undo the shared model
        params.update(n_steps=64, batch_size=64, learning_rate=args.learning_rate or 1e-5, n_epochs=2)
    if args.init:
        model = PPO.load(args.init, env=env, device=args.device, **params)
        vn = os.path.join(os.path.dirname(args.init), "vec_normalize.pkl")
        if os.path.exists(vn):  # keep the reward scale the value head was trained on
            old = VecNormalize.load(vn, SubprocVecEnv([fns[0]]))
            env.ret_rms = old.ret_rms
            old.close()
    else:
        policy_kwargs = dict(features_extractor_class=UniTSAMovementTransformer,
                             features_extractor_kwargs=unitsa.POLICY_KWARGS, activation_fn=torch.nn.GELU,
                             share_features_extractor=False)
        model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device=args.device, seed=args.seed,
                    **params)
    meta = {"junctions": junctions, "nets": args.nets, "patterns": patterns, "workers": len(fns),
            "total_timesteps": args.total_timesteps, "init": args.init, "ppo": params,
            "history_len": unitsa.HISTORY_LEN, "max_green": unitsa.MAX_GREEN, "phase_pad": unitsa.PHASE_PAD}
    json.dump(meta, open(os.path.join(args.out, "train_meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)
    t0 = time.time()
    ckpt = CheckpointCallback(save_freq=max(args.checkpoint_every // len(fns), 1),
                              save_path=os.path.join(args.out, "checkpoints"))
    if args.episodes:
        from jevtraffic.rl_baselines import episode_budget_callback

        ckpt = episode_budget_callback(args.episodes)
    model.learn(total_timesteps=args.total_timesteps, callback=ckpt, reset_num_timesteps=not args.init)
    model.save(os.path.join(args.out, "model.zip"))
    env.save(os.path.join(args.out, "vec_normalize.pkl"))
    meta["train_s"] = round(time.time() - t0, 1)
    meta["episodes"], meta["timesteps_used"] = args.episodes, int(model.num_timesteps)
    json.dump(meta, open(os.path.join(args.out, "train_meta.json"), "w"), indent=1)
    print("done", meta["train_s"], flush=True)
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)
