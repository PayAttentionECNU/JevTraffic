"""PPO fine-tuning of a distilled keep/switch Jev student on SUMO, with UniTSA's reward and critic.

Zero-shot (universal): all training junctions x nets x training patterns, many workers, a step budget
(v5: --reward little --kl_coef 0.2 --critic_warmup 5; checkpoints update<k>/ are then chosen on validation
scenarios by scripts/select_rl_checkpoint.py):
    python scripts/rl_jev.py --model ~/zzy/hf_models/Qwen2.5-0.5B-Instruct --adapters ckpt/unitsa-jev-0.5B-full \
        --state_format full --junctions train --workers 32 --total_steps 500000 --out ckpt/rl-jev-0.5B-full
Few-shot on a held-out junction, a budget of N episodes (as the baselines' few-shot runs):
    python scripts/rl_jev.py ... --junctions France_Massy --episodes 10 --out ckpt/rl-jev-0.5B-full-fs10_France_Massy

The distilled adapter is merged into the base and a fresh LoRA is trained on top, so the reference policy of the
KL penalty is the same model with that LoRA disabled. The result evaluates with
`run_sumo.py --adapters <distilled> <out>` (adapters are merged in order). See jevtraffic/rl_jev.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jevtraffic.sumo_env import HELDOUT_PATTERN, JUNCTIONS, NETS, PATTERNS, TRAIN_JUNCTIONS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapters", nargs="*", default=[], help="distilled student adapter(s), merged into the base")
    p.add_argument("--state_format", default="full", choices=["full", "compact"])
    p.add_argument("--critic", default=os.path.join(ROOT, "ckpt", "unitsa", "universal", "model.zip"))
    p.add_argument("--critic_state", default=None, help="critic.pt of an earlier rl_jev run (few-shot from it)")
    p.add_argument("--reward", default="wait", choices=["wait", "little"], help="see jevtraffic/rl_jev.py")
    p.add_argument("--wait_coef", type=float, default=0.1, help="little: weight of the waiting-time penalty")
    p.add_argument("--normalizer", default=None, help="normalizer.json of an earlier run (little; few-shot)")
    p.add_argument("--junctions", nargs="+", default=["train"])
    p.add_argument("--nets", nargs="+", default=NETS)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--total_steps", type=int, default=500_000, help="student decisions over all workers")
    p.add_argument("--episodes", type=int, default=0, help="few-shot budget: stop after this many episodes")
    p.add_argument("--n_steps", type=int, default=64, help="decisions per worker per rollout")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.05, help="KL(pi || distilled student) penalty")
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (larger backbones)")
    p.add_argument("--critic_warmup", type=int, default=1, help="rollouts that train only the critic")
    p.add_argument("--save_every", type=int, default=20, help="updates between adapter checkpoints")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from stable_baselines3 import PPO

    from jevtraffic.decider import Choice, build_prompt, load_model, single_token_ids
    from jevtraffic.rl_jev import RewardNormalizer, VecWorkers, gae, normalize_reward
    from jevtraffic.sumo_prompts import SYSTEM_PROMPT, option_to_action

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from train_lora import letter_logprobs

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")

    # ------------------------------------------------------------------ policy: student + fresh LoRA
    tokenizer, model = load_model(args.model, args.adapters, device="cuda")
    model.config.use_cache = False
    if args.grad_ckpt:  # not needed for 0.5B / 1.5B at minibatch 64; recomputation costs ~1/3 of the update time
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    letters = single_token_ids(tokenizer, "AB")
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)

    # ------------------------------------------------------------------ critic: UniTSA's value network
    teacher = PPO.load(args.critic, device="cuda")
    policy = teacher.policy
    if args.critic_state:
        policy.load_state_dict(torch.load(args.critic_state, map_location=device))
    critic_params = (list(policy.vf_features_extractor.parameters()) + list(policy.mlp_extractor.value_net.parameters())
                     + list(policy.value_net.parameters()))
    critic_opt = torch.optim.Adam(critic_params, lr=args.critic_lr)
    vn = os.path.join(os.path.dirname(args.critic), "vec_normalize.pkl")
    import pickle

    ret_var = float(pickle.load(open(vn, "rb")).ret_rms.var) if os.path.exists(vn) else 1.0
    unitsa_shape = tuple(teacher.observation_space.shape)

    def values(obs_batch):
        x = torch.as_tensor(np.stack(obs_batch), device=device)
        return policy.predict_values(x).squeeze(-1)

    # ------------------------------------------------------------------ workers
    junctions = TRAIN_JUNCTIONS if args.junctions == ["train"] else args.junctions
    assert all(j in JUNCTIONS for j in junctions), junctions
    scenarios = [(j, f"{n}_{q}") for j in junctions for n in args.nets for q in PATTERNS if q != HELDOUT_PATTERN]
    n_workers = min(args.workers, args.episodes) if args.episodes else args.workers
    work_dir = os.path.join(ROOT, "records", "rl_jev", os.path.basename(args.out.rstrip("/")))
    specs = [{"index": i, "seed": args.seed * 1000 + i, "scenarios": scenarios, "state_format": args.state_format,
              "unitsa_shape": unitsa_shape, "work_dir": work_dir, "reward": args.reward,
              "wait_coef": args.wait_coef} for i in range(n_workers)]
    workers = VecWorkers(specs)
    normalizer = RewardNormalizer(n_workers, args.gamma)
    if args.normalizer:
        normalizer.count, normalizer.mean, normalizer.var = json.load(open(args.normalizer)).values()

    def scale(rewards, dones):
        if args.reward == "wait":  # the critic's own (UniTSA's) return scale
            return [normalize_reward(r, ret_var) for r in rewards]
        return normalizer(rewards, dones)

    def checkpoint(path):
        model.save_pretrained(path)
        torch.save(policy.state_dict(), os.path.join(path, "critic.pt"))
        json.dump(normalizer.state(), open(os.path.join(path, "normalizer.json"), "w"))

    def prompts_of(obs_list):
        return [build_prompt(tokenizer, o["state"], Choice(o["question"], o["options"], "switch"), SYSTEM_PROMPT)
                for o in obs_list]

    def logprobs(prompts, grad: bool, ref: bool = False):
        out = []
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for i in range(0, len(prompts), args.minibatch):
                chunk = prompts[i:i + args.minibatch]
                if ref:
                    with model.disable_adapter():
                        out.append(letter_logprobs(model, tokenizer, chunk, letters, device, 1024))
                else:
                    out.append(letter_logprobs(model, tokenizer, chunk, letters, device, 1024))
        return torch.cat(out)

    meta = {**vars(args), "junctions": junctions, "workers": n_workers, "ret_var": ret_var,
            "scenarios": len(scenarios)}
    json.dump(meta, open(os.path.join(args.out, "rl_meta.json"), "w"), indent=1)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    print(json.dumps(meta), flush=True)

    episodes_done, steps_done, update, t0 = 0, 0, 0, time.time()
    recent = deque(maxlen=50)
    if args.save_every:
        checkpoint(os.path.join(args.out, "update0"))  # = the starting policy (fresh LoRA is the identity)
    while True:
        if args.episodes and episodes_done >= args.episodes:
            break
        if not args.episodes and steps_done >= args.total_steps:
            break
        # -------------------------------------------------------------- rollout
        model.eval()
        policy.set_training_mode(False)  # the critic's Transformer has dropout
        buf = {k: [] for k in ("prompts", "critic", "actions", "logp", "ref", "rewards", "dones", "values")}
        t_roll = time.time()
        for _ in range(args.n_steps):
            obs = workers.obs
            prompts = prompts_of(obs)
            lp = logprobs(prompts, grad=False)
            lr_ = logprobs(prompts, grad=False, ref=True)
            acts = torch.distributions.Categorical(logits=lp).sample()
            with torch.no_grad():
                v = values([o["critic_obs"] for o in obs])
            rewards, dones, infos = workers.step([option_to_action(int(a)) for a in acts])
            for info in infos:
                if "episode" in info:
                    episodes_done += 1
                    recent.append(info["episode"])
                    log.write(json.dumps({"update": update, **info["episode"]}) + "\n")
            buf["prompts"].append(prompts)
            buf["critic"].append([o["critic_obs"] for o in obs])
            buf["actions"].append(acts.cpu())
            buf["logp"].append(lp.gather(1, acts[:, None]).squeeze(1).cpu())
            buf["ref"].append(lr_.cpu())
            buf["rewards"].append(scale(rewards, dones))
            buf["dones"].append(dones)
            buf["values"].append(v.float().cpu().numpy())
            steps_done += len(obs)
            if args.episodes and episodes_done >= args.episodes:
                break
        with torch.no_grad():
            last_v = values([o["critic_obs"] for o in workers.obs]).float().cpu().numpy()
        adv, ret = gae(buf["rewards"], buf["values"], buf["dones"], last_v, args.gamma, args.lam)
        t_roll = time.time() - t_roll

        # -------------------------------------------------------------- update
        T, N = adv.shape
        flat = lambda xs: [x for row in xs for x in row]  # noqa: E731
        prompts = flat(buf["prompts"])
        critic_obs = flat(buf["critic"])
        actions = torch.cat(buf["actions"]).to(device)
        old_logp = torch.cat(buf["logp"]).to(device)
        ref_logp = torch.cat(buf["ref"]).to(device)
        adv_t = torch.as_tensor(adv.reshape(-1), device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret.reshape(-1), device=device)
        train_policy = update >= args.critic_warmup
        model.train()
        policy.set_training_mode(True)
        stats = {"pg": [], "kl": [], "clipfrac": [], "vf": [], "ent": []}
        t_upd = time.time()
        for _ in range(args.epochs):
            idx = torch.randperm(T * N).tolist()
            for i in range(0, len(idx), args.minibatch):
                mb = idx[i:i + args.minibatch]
                v = values([critic_obs[j] for j in mb])
                vf_loss = ((v.float() - ret_t[mb]) ** 2).mean()
                critic_opt.zero_grad()
                vf_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_params, 0.5)
                critic_opt.step()
                stats["vf"].append(vf_loss.item())
                if not train_policy:
                    continue
                lp = letter_logprobs(model, tokenizer, [prompts[j] for j in mb], letters, device, 1024)
                new = lp.gather(1, actions[mb][:, None]).squeeze(1)
                ratio = torch.exp(new - old_logp[mb])
                a = adv_t[mb]
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                kl = (lp.exp() * (lp - ref_logp[mb])).sum(-1).mean()  # exact KL over the two options
                ent = -(lp.exp() * lp).sum(-1).mean()
                loss = pg + args.kl_coef * kl - args.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
                opt.step()
                stats["pg"].append(pg.item())
                stats["kl"].append(kl.item())
                stats["ent"].append(ent.item())
                stats["clipfrac"].append(((ratio - 1).abs() > args.clip).float().mean().item())
        update += 1
        row = {"update": update, "steps": steps_done, "episodes": episodes_done, "policy": train_policy,
               "rollout_s": round(t_roll, 1), "update_s": round(time.time() - t_upd, 1),
               "elapsed_s": round(time.time() - t0), "mean_reward": round(float(np.mean(buf["rewards"])), 4),
               "switch_rate": round(float((torch.cat(buf["actions"]) == 1).float().mean()), 3),
               **{k: round(float(np.mean(v)), 5) for k, v in stats.items() if v}}
        if recent:
            row["recent_ATT"] = round(float(np.mean([e["ATT"] for e in recent if e.get("ATT")])), 2)
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
        if args.save_every and update % args.save_every == 0:
            checkpoint(os.path.join(args.out, f"update{update}"))

    checkpoint(args.out)
    meta.update(updates=update, steps=steps_done, episodes_done=episodes_done, train_s=round(time.time() - t0))
    json.dump(meta, open(os.path.join(args.out, "rl_meta.json"), "w"), indent=1)
    print("saved", args.out, flush=True)
    workers.close()


if __name__ == "__main__":
    main()
    os._exit(0)
