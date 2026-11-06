"""PPO for the keep/switch Jev student, with UniTSA's critic (v3/v4 RL stage).

The policy is the student's answer distribution over the two options of `switch_question` (one forward pass,
letters A = keep / B = switch, identity order as at inference). Rewards (`REWARDS`), per decision interval:
  * "wait" (v4): UniTSA's own, -(mean accumulated waiting time of the approaching vehicles) x 0.1 per env step,
    divided by the universal UniTSA's running return std (its VecNormalize) and clipped to +-10;
  * "little" (v5): by Little's law (L = lambda W, so at a given demand the time-average number of vehicles L
    is proportional to their mean time in the system W), the vehicle-seconds spent on the approach lanes during
    the interval, plus a small penalty `wait_coef` x interval x mean accumulated waiting time (anti-starvation:
    vehicle counts alone do not care who waits); normalised by the running std of the discounted return
    (`RewardNormalizer`, as VecNormalize) and clipped to +-10.
The critic is the universal UniTSA's value network (its own Transformer over the benchmark movement state),
fine-tuned alongside the policy (for "little" after a critic-only warm-up, since its scale was UniTSA's reward).
Forced switches (45 s max green) are not decisions: as in the controller the model is not asked, and their
reward is added to the previous decision's transition.
`EnvWorker` runs one SumoTSCEnv in a subprocess and returns the student's prompt parts + UniTSA's observation at
each decision point; `gae` turns a rollout into advantages.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import shutil
import traceback

import numpy as np

REWARD_SCALE = 0.1  # jevtraffic.unitsa.train_env
LITTLE_SCALE = 0.01  # vehicle-seconds -> reward units (only the ratio to wait_coef matters after normalisation)
REWARD_CLIP = 10.0  # VecNormalize default clip_reward


def wait_reward(env, wait_coef: float = 0.0) -> float:
    """UniTSA's reward for the last decision interval of `env` (raw TSCInfoWrapper output)."""
    from tsc_algos.rl.unitsa.unitsa_env.reward_funcs import waiting_time_reward

    return waiting_time_reward(env.raw["tls_dynamic_features_seq"]) * REWARD_SCALE


def little_reward(env, wait_coef: float = 0.1) -> float:
    """-(vehicle-seconds on the approach lanes + wait_coef x interval x mean accumulated wait) x LITTLE_SCALE.
    One frame of lane_dynamic_features_seq is one simulated second of the interval."""
    from tsc_algos.rl.unitsa.unitsa_env.reward_funcs import movement_avg_accumulated_waiting_time

    frames = env.raw["lane_dynamic_features_seq"]
    incoming = env.incoming_lanes
    veh_s = sum(sum(c["vehicle_count"] for c in f.get(lid, ())) for f in frames for lid in incoming)
    wait = movement_avg_accumulated_waiting_time(env.raw["tls_dynamic_features_seq"][-1])
    return -(veh_s + wait_coef * len(frames) * wait) * LITTLE_SCALE


REWARDS = {"wait": wait_reward, "little": little_reward}


class RewardNormalizer:
    """VecNormalize's reward scaling: divide by the running std of each worker's discounted return."""

    def __init__(self, n: int, gamma: float = 0.99):
        self.ret = np.zeros(n)
        self.gamma = gamma
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def _update(self, x):
        b_mean, b_var, b_n = float(np.mean(x)), float(np.var(x)), len(x)
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def __call__(self, rewards, dones):
        r = np.asarray(rewards, dtype=np.float64)
        self.ret = self.ret * self.gamma + r
        self._update(self.ret)
        self.ret[np.asarray(dones, dtype=bool)] = 0.0
        return np.clip(r / np.sqrt(self.var + 1e-8), -REWARD_CLIP, REWARD_CLIP).tolist()

    def state(self) -> dict:
        return {"count": self.count, "mean": self.mean, "var": self.var}


class DecisionStream:
    """Steps a next_or_not SumoTSCEnv to the next point where the student decides (skipping forced switches).

    `advance(action)` applies the student's action (None at reset), then auto-applies forced switches; it returns
    the summed raw reward since the last decision and whether the episode ended."""

    def __init__(self, env, unitsa_shape, reward_fn, max_green: float):
        from .unitsa import UniTSAState

        self.env, self.reward_fn, self.max_green = env, reward_fn, max_green
        self.ustate = UniTSAState(env, unitsa_shape)
        self.forced = 0

    def _step(self, action: int) -> float:
        self.env.step(action)
        self.critic_obs = self.ustate.update()
        return self.reward_fn(self.env)

    def reset(self):
        self.obs = self.env.reset()
        self.critic_obs = self.ustate.update()
        return self._skip_forced(0.0)

    def advance(self, action: int):
        return self._skip_forced(self._step(action))

    def _skip_forced(self, reward: float):
        while not self.env.done and self.env.obs.green_seconds >= self.max_green:
            reward += self._step(0)
            self.forced += 1
        self.obs = self.env.obs
        return reward, self.env.done


def _worker(conn, spec):
    """Subprocess: one SUMO scenario at a time, sampled from spec['scenarios'] on every reset."""
    import torch

    torch.set_num_threads(1)
    from functools import partial

    from .sumo_agents import MAX_GREEN_SECONDS
    from .sumo_env import SumoTSCEnv, _quiet_loguru
    from .sumo_prompts import STATE_FORMATS, switch_question

    _quiet_loguru()
    rng = random.Random(spec["seed"])
    render = STATE_FORMATS[spec["state_format"]]
    stream = None
    episodes = 0

    def start():
        nonlocal stream, episodes
        junction, env_name = rng.choice(spec["scenarios"])
        work = os.path.join(spec["work_dir"], f"w{spec['index']}_{episodes}")
        env = SumoTSCEnv(junction, env_name, work, seed=rng.randrange(10**6), action_type="next_or_not")
        reward_fn = partial(REWARDS[spec.get("reward", "wait")], wait_coef=spec.get("wait_coef", 0.1))
        stream = DecisionStream(env, spec["unitsa_shape"], reward_fn, MAX_GREEN_SECONDS)
        reward, _ = stream.reset()
        episodes += 1
        return reward

    def observation():
        q = switch_question(stream.obs)
        return {"state": render(stream.obs), "question": q.question, "options": list(q.options),
                "critic_obs": stream.critic_obs}

    try:
        start()
        conn.send(("ok", observation()))
        while True:
            cmd, arg = conn.recv()
            if cmd == "step":
                reward, done = stream.advance(arg)
                info = {}
                if done:
                    env = stream.env
                    env.close()  # writes tripinfo
                    info = {"episode": {"junction": env.junction, "env_name": env.env_name, **env.metrics(),
                                        "decisions": env.steps, "forced": stream.forced}}
                    shutil.rmtree(env.work_dir, ignore_errors=True)  # the server disk is shared
                    start()  # (the reward of forced steps before the new episode's first decision is dropped)
                conn.send(("ok", (reward, done, info, observation())))
            elif cmd == "close":
                if stream is not None:
                    stream.env.close()
                conn.send(("ok", None))
                break
    except Exception:
        conn.send(("error", traceback.format_exc()))


class VecWorkers:
    """N EnvWorkers in subprocesses (spawn: libsumo and CUDA do not survive fork)."""

    def __init__(self, specs: list[dict]):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for spec in specs:
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, spec), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.obs = [self._recv(c) for c in self.conns]

    @staticmethod
    def _recv(conn):
        status, payload = conn.recv()
        if status != "ok":
            raise RuntimeError(f"worker failed:\n{payload}")
        return payload

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", int(a)))
        out = [self._recv(c) for c in self.conns]
        self.obs = [o[3] for o in out]
        return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
                c.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)


def normalize_reward(r: float, ret_var: float) -> float:
    return float(np.clip(r / np.sqrt(ret_var + 1e-8), -REWARD_CLIP, REWARD_CLIP))


def gae(rewards, values, dones, last_values, gamma: float = 0.99, lam: float = 0.95):
    """Generalised advantage estimates for a (T, N) rollout. dones[t] = episode ended after step t."""
    rewards, values, dones = np.asarray(rewards), np.asarray(values), np.asarray(dones, dtype=np.float32)
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values
