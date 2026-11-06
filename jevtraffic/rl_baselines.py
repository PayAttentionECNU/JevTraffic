"""The DQN baselines of SingleTSCBaselines (PressLight, AttendLight, IntelliLight) as universal models.

The benchmark trains each of them per scenario (one junction, one demand pattern). For the zero-shot / few-shot
tables we train one model across junctions, as for UniTSA (jevtraffic/unitsa.py), and add what that needs:
  * `RandomScenarioEnv`: one worker per (junction, net) that samples a demand pattern on every reset;
  * `PadPhases`: the movement features end with a phase-binding block of `num_phases` columns (3 or 4), so they
    are zero-padded to PHASE_PAD columns;
  * `PadActions` + `MaskedDQNPolicy`: the action space is Discrete(PHASE_PAD) everywhere; the Q-values of phases
    the junction does not have (no movement bound to them in the observation) are masked, for acting and for the
    TD target alike; only an epsilon-random exploration step can still pick one, and it then keeps the phase;
  * `RLBaselineAgent`: runs a trained model inside our `SumoTSCEnv` (choose_next_phase, safety layer off, as in
    the benchmark), rebuilding the benchmark state from the env's raw TSCInfoWrapper output.
Everything else (state, reward, network, DQN settings) is the benchmark's, unchanged.
"""
from __future__ import annotations

import importlib
import random
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import sumo_env  # noqa: F401  (puts SingleTSCBaselines on sys.path, sets SUMO_HOME)
from .unitsa import PHASE_PAD, fit_shape

NUM_MOVEMENTS = 12
# per algorithm: env package, feature extractor, its kwargs, history length, reward options (the benchmark's train.py)
ALGOS = {
    "presslight": dict(pkg="presslight.presslight_env", model=("presslight.model", "PressLightMovementModel"),
                       model_kwargs=dict(features_dim=128, hidden_dims=[128, 128]), history_len=5,
                       reward_scale=0.01, env_kwargs=dict(reward_time_decay=1.0), vec_normalize=False),
    "attendlight": dict(pkg="attendlight.attendlight_env",
                        model=("attendlight.model", "AttendLightMovementTransformer"),
                        model_kwargs=dict(features_dim=128, embedding_dim=128, num_heads=4, num_layers=2,
                                          dim_feedforward=256, dropout=0.1), history_len=4,
                        reward_scale=0.01, env_kwargs=dict(reward_time_decay=1.0), vec_normalize=False),
    "intellilight": dict(pkg="intellilight.intellilight_env",
                         model=("intellilight.model", "IntelliLightMovementModel"),
                         model_kwargs=dict(features_dim=128, hidden_dims=[128, 128]), history_len=4,
                         reward_scale=0.01, env_kwargs={}, vec_normalize=True),
}
EPISODE_STEPS = 150  # the benchmark's --episode_steps (sets learning_starts / target_update_interval)


FEW_SHOT_LR = 1e-5  # v5: 10x below training; v4 used 1e-4 and num_envs gradient steps, which could make it worse


def dqn_params(num_envs: int, few_shot: bool = False) -> dict:
    """The benchmark's DQN settings. Few-shot adaptation of a trained model starts learning (almost) at once,
    explores little (1 episode, ~200 decisions, is less than the benchmark's learning_starts), and is conservative:
    a small learning rate and one gradient step per 4 env steps, so a few episodes cannot undo the shared model."""
    p = dict(learning_rate=1e-4, buffer_size=50000, learning_starts=max(EPISODE_STEPS * 8, 1000), batch_size=64,
             train_freq=4, gradient_steps=max(num_envs, 1), gamma=0.99, exploration_fraction=0.2,
             exploration_final_eps=0.05, target_update_interval=max(EPISODE_STEPS * 5, 500))
    if few_shot:
        p.update(learning_rate=FEW_SHOT_LR, gradient_steps=1, learning_starts=64, exploration_initial_eps=0.05,
                 exploration_fraction=1.0, target_update_interval=200)  # (exploration constant at 0.05)
    return p


def masked_policy():
    """DQNPolicy whose Q-network masks phases that no movement is bound to (padding of 3-phase junctions)."""
    import torch
    from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

    class MaskedQNetwork(QNetwork):
        def forward(self, obs):
            q = super().forward(obs)
            binding = obs[:, -1, :, -q.shape[1]:]  # (B, movements, phases) of the latest frame
            valid = binding.abs().sum(dim=1) > 0
            valid = valid | ~valid.any(dim=1, keepdim=True)  # no binding at all: leave unmasked
            return q.masked_fill(~valid, torch.finfo(q.dtype).min / 2)

    class MaskedDQNPolicy(DQNPolicy):
        def make_q_net(self):
            net_args = self._update_features_extractor(self.net_args, features_extractor=None)
            return MaskedQNetwork(**net_args).to(self.device)

    return MaskedDQNPolicy


def episode_budget_callback(episodes: int):
    """SB3 callback that stops training once `episodes` episodes have finished (few-shot budgets)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeBudget(BaseCallback):
        def __init__(self):
            super().__init__()
            self.done = 0

        def _on_step(self) -> bool:
            self.done += int(np.sum(self.locals["dones"]))
            return self.done < episodes

    return EpisodeBudget()


def _module(algo: str, name: str):
    return importlib.import_module(f"tsc_algos.rl.{ALGOS[algo]['pkg']}.{name}")


def model_class(algo: str):
    mod, cls = ALGOS[algo]["model"]
    return getattr(importlib.import_module(f"tsc_algos.rl.{mod}"), cls)


def base_dim(algo: str) -> int:
    return _module(algo, "state_funcs").MOVEMENT_BASE_DIM


class PadPhases(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, base: int, phase_pad: int = PHASE_PAD):
        super().__init__(env)
        t, m, _ = env.observation_space.shape
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(t, m, base + phase_pad), dtype=np.float32)

    def observation(self, obs):
        return fit_shape(np.asarray(obs, dtype=np.float32), self.observation_space.shape)


class PadActions(gym.Wrapper):
    """Discrete(PHASE_PAD) on every junction; an action beyond the junction's phases keeps the last phase."""

    def __init__(self, env: gym.Env, phase_pad: int = PHASE_PAD):
        super().__init__(env)
        self.n = env.action_space.n
        self.action_space = spaces.Discrete(phase_pad)
        self.last = 0

    def reset(self, **kw):
        self.last = 0
        return self.env.reset(**kw)

    def step(self, action):
        action = int(action)
        if action >= self.n:
            action = self.last
        self.last = action
        return self.env.step(action)


class RandomScenarioEnv(gym.Env):
    """One junction and network; a demand pattern sampled on every reset (cf. UniTSA's RandomScenarioEnv)."""

    metadata = {"render_modes": []}

    def __init__(self, algo: str, junction: str, env_names: list[str], seed: int, reward_scale: float | None = None):
        from junction_configs import load_junction_config

        a = ALGOS[algo]
        self.cfgs = {e: load_junction_config(junction, e) for e in env_names}
        self.env_names, self.rng = list(env_names), random.Random(seed)
        cfg = self.cfgs[self.env_names[0]]
        make_env = _module(algo, "make_env").make_env
        env = make_env(tls_id=cfg["tls_id"], num_seconds=cfg["num_seconds"], num_phases=cfg["num_phases"],
                       sumo_cfg=cfg["sumo_cfg"], net_file=cfg["net_file"], use_gui=False, log_file=None,
                       env_index=f"{junction}_{seed}", num_movements=NUM_MOVEMENTS, history_len=a["history_len"],
                       reward_scale=a["reward_scale"] if reward_scale is None else reward_scale, **a["env_kwargs"])()
        self.env = PadActions(PadPhases(env, base_dim(algo)))
        self.observation_space, self.action_space = self.env.observation_space, self.env.action_space
        self.current = self.env_names[0]

    def _base(self):
        e = self.env
        while isinstance(e, gym.Wrapper):
            e = e.env
        return e

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.current = self.rng.choice(self.env_names)
        cfg = self.cfgs[self.current]
        self._base().configure_sumo(sumo_cfg=cfg["sumo_cfg"], net_file=cfg["net_file"], num_seconds=cfg["num_seconds"])
        obs, info = self.env.reset(seed=seed)
        return obs, {**info, "env_name": self.current}

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r, term, trunc, {**info, "env_name": self.current}

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def train_env(algo: str, junction: str, net: str, patterns: list[str], seed: int, log_file: str | None = None):
    """Factory for one training worker: `junction` on network `net`, a random pattern per episode."""

    def _init():
        from stable_baselines3.common.monitor import Monitor

        sumo_env._quiet_loguru()
        env = RandomScenarioEnv(algo, junction, [f"{net}_{p}" for p in patterns], seed)
        if log_file:
            env = Monitor(env, filename=log_file, info_keywords=("env_name",))
        return env

    return _init


class RLBaselineAgent:
    """A trained PressLight / AttendLight / IntelliLight model driving `SumoTSCEnv(choose_next_phase)`."""

    def __init__(self, env, algo: str, model_path: str, device: str = "cpu"):
        import torch
        from stable_baselines3 import DQN

        assert env.action_type == "choose_next_phase"
        torch.set_num_threads(1)
        env.constrain = False  # the benchmark's controllers own their phase choice (as BenchmarkAgent)
        self.env, self.algo = env, algo
        self.state_fn = _module(algo, "state_funcs").movement_sequence_state
        self.model = DQN.load(model_path, device=device, custom_objects={"policy_class": masked_policy()})
        self.shape = self.model.observation_space.shape
        self.n = env.cfg["num_phases"]
        self.last = 0
        self.calls, self.seconds, self.padded = 0, 0.0, 0

    def act(self, obs) -> int:
        t0 = time.perf_counter()
        state = self.state_fn(self.env.raw["tls_dynamic_features_seq"], self.n,
                              history_len=self.shape[0]).astype(np.float32)
        action, _ = self.model.predict(fit_shape(state, self.shape), deterministic=True)
        action = int(action)
        if action >= self.n:  # as PadActions in training
            action, self.padded = self.last, self.padded + 1
        self.last = action
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return action

    def summary(self) -> dict:
        return {"calls": self.calls, "ms_per_call": round(1000 * self.seconds / max(self.calls, 1), 3),
                "padded_actions": self.padded, "params": sum(p.numel() for p in self.model.q_net.parameters())}
