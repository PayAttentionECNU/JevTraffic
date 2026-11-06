"""CPU tests for v4: main-table label parsing, PPO pieces of rl_jev, action padding of the DQN baselines."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def test_main_table_labels_and_budgets():
    import main_tables as mt

    rows = dict(mt.ROWS)
    student = "jev-ns[Qwen2.5-0.5B-Instruct+unitsa-jev-0.5B-full]"
    rl = "jev-ns[Qwen2.5-0.5B-Instruct+unitsa-jev-0.5B-full+rl5-0.5B-full]"
    fs = "jev-ns[Qwen2.5-0.5B-Instruct+unitsa-jev-0.5B-full+rl5-0.5B-full+rl5-0.5B-full-fs10_France_Massy]"
    v4 = "jev-ns[Qwen2.5-0.5B-Instruct+unitsa-jev-0.5B-full+rl-jev-0.5B-full]"
    assert re.match(rows["Jev (distilled)"], student) and not re.match(rows["Jev (distilled)"], rl)
    assert not re.match(rows["Jev (distilled + RL)"], v4)  # the v4 RL run is not part of the tables
    m = re.match(rows["Jev (distilled + RL)"], rl)
    assert m and mt.budget_of(m.group("m"), "France_Massy") == 0
    m = re.match(rows["Jev (distilled + RL)"], fs)
    assert m and mt.budget_of(m.group("m"), "France_Massy") == 10
    assert mt.budget_of(m.group("m"), "Hongkong_YMT") is None
    m = re.match(rows["UniTSA"], "unitsa[fs1_SouthKorea_Songdo]")
    assert mt.budget_of(m.group("m"), "SouthKorea_Songdo") == 1
    assert mt.budget_of("universal", "France_Massy") == 0 and mt.budget_of(None, "x") == 0
    assert mt.budget_of("ft_France_Massy", "France_Massy") is None  # UniTSA's 300k fine-tunes are not few-shot
    assert mt.budget_of("v4_fs10_France_Massy", "France_Massy") is None  # archived v4 few-shot runs


def test_gae_matches_hand_computation():
    from jevtraffic.rl_jev import gae, normalize_reward

    rewards = [[1.0], [0.0], [2.0]]
    values = [[0.5], [0.5], [0.5]]
    dones = [[False], [True], [False]]
    adv, ret = gae(rewards, values, dones, np.array([1.0], dtype=np.float32), gamma=0.5, lam=1.0)
    # t=2: 2 + 0.5*1 - 0.5 = 2.0 ; t=1 (terminal): 0 - 0.5 = -0.5 ; t=0: 1 + 0.5*0.5 - 0.5 + 0.5*(-0.5) = 0.5
    assert np.allclose(adv[:, 0], [0.5, -0.5, 2.0]) and np.allclose(ret, adv + 0.5)
    assert normalize_reward(-100.0, 1.0) == -10.0 and normalize_reward(-1.0, 4.0) == pytest.approx(-0.5, 1e-6)


class FakeEnv:
    """next_or_not env whose green clock is given per step; reward_fn reads raw['r']."""

    action_type = "next_or_not"

    def __init__(self, greens):
        self.greens, self.k, self.done, self.steps = greens, 0, False, 0
        self.applied = []

    class Obs:
        def __init__(self, g):
            self.green_seconds = g

    def _set(self):
        self.obs = self.Obs(self.greens[self.k])
        self.raw = {"tls_dynamic_features_seq": [[{"r": -float(self.k)}]]}
        self.done = self.k == len(self.greens) - 1

    def reset(self):
        self.k = 0
        self._set()
        return self.obs

    def step(self, action):
        self.applied.append(action)
        self.k += 1
        self.steps += 1
        self._set()
        return self.obs


def test_decision_stream_merges_forced_switches(monkeypatch):
    from jevtraffic import rl_jev

    class NoState:
        def __init__(self, env, shape, max_green=45):
            pass

        def update(self):
            return np.zeros(1)

    monkeypatch.setattr("jevtraffic.unitsa.UniTSAState", NoState)
    env = FakeEnv([10, 50, 0, 5])  # after the first decision the green is 50 s >= 45: one forced switch
    s = rl_jev.DecisionStream(env, (1,), lambda e: e.raw["tls_dynamic_features_seq"][-1][0]["r"], max_green=45)
    assert s.reset() == (0.0, False)
    reward, done = s.advance(1)  # keep -> step 1 (forced: switch) -> step 2 (decision)
    assert env.applied == [1, 0] and s.forced == 1 and not done
    assert reward == pytest.approx(-1.0 - 2.0)
    reward, done = s.advance(0)
    assert done and reward == pytest.approx(-3.0)


def test_little_reward_counts_vehicle_seconds_on_approaches(monkeypatch):
    import types

    from jevtraffic import rl_jev

    mod = types.ModuleType("rf")
    mod.movement_avg_accumulated_waiting_time = lambda mv: 20.0
    monkeypatch.setitem(sys.modules, "tsc_algos.rl.unitsa.unitsa_env.reward_funcs", mod)
    cells = lambda *n: [{"vehicle_count": k} for k in n]  # noqa: E731
    env = types.SimpleNamespace(incoming_lanes=["in_0", "in_1"], raw={
        "lane_dynamic_features_seq": [{"in_0": cells(1, 2), "in_1": cells(0), "out_0": cells(9)},  # 3 vehicles
                                      {"in_0": cells(1), "in_1": cells(1, 1), "out_0": cells(9)}],  # 3 vehicles
        "tls_dynamic_features_seq": [[{}]]})
    # 6 vehicle-seconds over 2 s + 0.1 x 2 s x 20 s waiting = 10, outgoing lanes ignored
    assert rl_jev.little_reward(env, wait_coef=0.1) == pytest.approx(-10 * rl_jev.LITTLE_SCALE)
    assert rl_jev.little_reward(env, wait_coef=0.0) == pytest.approx(-6 * rl_jev.LITTLE_SCALE)


def test_reward_normalizer_scales_by_return_std():
    from jevtraffic.rl_jev import RewardNormalizer

    n = RewardNormalizer(2, gamma=0.0)  # gamma 0: return = reward
    for _ in range(200):
        out = n([-4.0, 4.0], [False, False])
    assert n.var == pytest.approx(16.0, rel=0.01) and out == pytest.approx([-1.0, 1.0], rel=0.01)
    n([1.0, 1.0], [True, False])
    assert n.ret[0] == 0.0 and n.ret[1] == 1.0


def test_pad_actions_keeps_last_phase():
    gym = pytest.importorskip("gymnasium")
    from gymnasium import spaces

    from jevtraffic.rl_baselines import PadActions

    class E(gym.Env):
        action_space = spaces.Discrete(3)
        observation_space = spaces.Box(0, 1, (1,))

        def __init__(self):
            self.seen = []

        def reset(self, seed=None, options=None):
            return np.zeros(1), {}

        def step(self, a):
            self.seen.append(a)
            return np.zeros(1), 0.0, False, False, {}

    e = E()
    w = PadActions(e)
    w.reset()
    for a in (2, 3, 0, 3):
        w.step(a)
    assert w.action_space.n == 4 and e.seen == [2, 2, 0, 0]


def test_main_tables_applies_fewshot_gate(tmp_path, monkeypatch):
    import json

    import main_tables as mt

    runs = tmp_path / "results" / "sumo" / "runs"
    runs.mkdir(parents=True)
    j = "France_Massy"
    for model, att in [("universal", 50.0), (f"fs10_{j}", 60.0)]:
        r = {"agent": f"unitsa[{model}]", "junction": j, "env_name": "easy_low_density", "events": "none",
             "metrics": {"ATT": att}}
        (runs / f"{model}.json").write_text(json.dumps(r))
    gate = {"gate": {f"UniTSA|10|{j}": {"adapted": 61.0, "zero": 51.0, "use": "zero"}}}
    (tmp_path / "results" / "sumo" / "fewshot_gate.json").write_text(json.dumps(gate))
    monkeypatch.setattr(mt, "ROOT", str(tmp_path))
    monkeypatch.setattr(mt, "RUNS", str(runs))
    mt.main()
    text = (tmp_path / "results" / "sumo" / "main_tables.md").read_text()
    gated = text.split("validation-gated)")[2].split("###")[0]
    raw = text.split("raw adapted models)")[1]
    assert "| UniTSA | 50.0 | - | 50.0 | - |" in gated and "| UniTSA | 50.0 | - | 60.0 | - |" in raw
    assert "UniTSA 10 eps 0/1" in text
