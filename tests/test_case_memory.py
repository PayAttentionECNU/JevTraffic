"""Chronological SUMO case retrieval and the normal controller/prompt paths, without model weights."""
from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from jevtraffic.case_memory import CaseMemory
from jevtraffic.sumo_agents import Controller
from jevtraffic.sumo_env import write_sumocfg
from test_v2 import make_obs


def test_simulated_day_config_preserves_route_paths(tmp_path):
    route = str(tmp_path / "交通 & 路线.rou.xml")
    config = write_sumocfg(str(tmp_path / "day.sumocfg"), "network.xml", route, [])
    assert ElementTree.parse(config).find("input/route-files").get("value") == route


def archive(tmp_path, day, mode="collect", junction="Massy", network="normal", task="choose_next_phase"):
    return CaseMemory(tmp_path, junction, network, task, day, mode)


def collect(tmp_path, day, times=(100, 105, 110), **scope):
    store = archive(tmp_path, day, **scope)
    for t in times:
        before = make_obs(t=t)
        after = replace(before, time=t+5, lane_halting={"s_0": 4}, lane_discharged={"s_0": 3})
        store.record(before, after)
    store.publish()
    return store


def test_backoff_distinct_days_and_no_future(tmp_path):
    collect(tmp_path, 0, times=range(100, 200, 5))
    collect(tmp_path, 9)
    assert archive(tmp_path, 2, "read").retrieve(make_obs(), False) is None
    collect(tmp_path, 1)
    read = archive(tmp_path, 2, "read")
    fine = read.retrieve(make_obs(), False)
    o, = fine["outcomes"]
    assert fine["depth"] == 4 and o["steps"] == 23 and o["days"] == 2
    assert o["example"]["day"] == 1 and o["departures"] == 3 and o["queue_change"] == -2
    assert "Example: day 1" in read.render(fine)
    coarse = read.retrieve(make_obs(t=620), False)
    assert coarse["depth"] == 3 and "example" not in coarse["outcomes"][0]
    assert read.retrieve(replace(make_obs(), pressures=[0, 4]), False)["depth"] == 2
    # Different load split, same observable family.
    assert read.retrieve(make_obs(halting={"s_0": 3, "s_1": 0, "w_0": 3, "w_1": 0}), False)["depth"] == 1
    assert read.retrieve(make_obs(discharged={"s_0": 0}), False) is None
    assert read.retrieve(make_obs(special=[{"kind": "ambulance", "movement": 0}]), False) is None
    assert archive(tmp_path, 29, "read").retrieve(make_obs(), False) is None


def test_recency_frequency_and_similarity(tmp_path):
    collect(tmp_path, 0)
    collect(tmp_path, 1)
    def outcome(day, **obs):
        return archive(tmp_path, day, "read").retrieve(make_obs(**obs), False)["outcomes"][0]
    old = outcome(2)
    assert outcome(3)["score"] < old["score"]
    assert outcome(2, halting={"s_0": 6, "w_0": 3})["similarity"] < old["similarity"]
    collect(tmp_path, 2)
    assert outcome(3)["support"] > old["support"]
    assert outcome(3)["score"] > old["score"]


def test_archive_retry_and_scoping(tmp_path):
    first = collect(tmp_path, 0)
    first.publish()
    with pytest.raises(ValueError, match="another trajectory"):
        collect(tmp_path, 0, times=(120,))
    collect(tmp_path, 1)
    assert archive(tmp_path, 2, "read").retrieve(make_obs(), False)["outcomes"][0]["steps"] == 6
    for kw in (dict(network="easy"), dict(junction="YMT"), dict(task="next_or_not")):
        with pytest.raises(FileNotFoundError):
            archive(tmp_path, 2, "read", **kw)
    read = archive(tmp_path, 2, "read")
    read.record(make_obs(), make_obs(t=105))
    read.publish()
    assert not read.pending


def test_online_day_is_invisible_until_next_day(tmp_path):
    collect(tmp_path, 0)
    online = archive(tmp_path, 1, "online")
    for t in (100, 105, 110):
        before = make_obs(t=t)
        assert online.retrieve(before, False) is None
        online.record(before, make_obs(t=t+5))
    online.publish()
    assert archive(tmp_path, 1, "read").retrieve(make_obs(), False) is None
    assert archive(tmp_path, 2, "read").retrieve(make_obs(), False)["outcomes"][0]["steps"] == 6


def test_observed_switch_rate_and_legal_phases(tmp_path):
    for d in (0, 1):
        store = archive(tmp_path, d)
        for t in (0, 10):
            store.record(make_obs(t=t), make_obs(t=t+10, phase=1))
        store.publish()
    read = archive(tmp_path, 2, "read")
    assert read.retrieve(make_obs(allowed=(0,)), False) is None
    before, after = make_obs(), make_obs(t=110, phase=1)
    match = read.retrieve(before, False)
    o, = match["outcomes"]
    assert o["phase"] == 1 and o["departures"] == 1 and o["action"] == "switch to west left"
    read.record(before, after)
    assert read.summary()["paired_intervals"] == 1 and set(read.summary()["paired_mae"].values()) == {0.0}


@pytest.mark.parametrize("action_type", ["next_or_not", "choose_next_phase"])
def test_controller_uses_evidence_in_existing_calls(tmp_path, action_type):
    for d in (0, 1):
        collect(tmp_path, d, task=action_type)
    env = SimpleNamespace(junction="Massy", action_type=action_type, situation_label=lambda: "normal")
    class Recorder:
        def __init__(self):
            self.prompts = []

        def decide(self, state, question):
            self.prompts.append(state)
            return SimpleNamespace(index=0, confidence=0.9, probabilities=[0.9, 0.1])

    obs = replace(make_obs(), green_seconds=15)
    def run(cases):
        model = Recorder()
        ctl = Controller(env, decider=model, detector="jev", memory="always", case_memory=cases, record_states=True)
        return ctl.act(obs), model.prompts, ctl

    off_action, old_prompts, _ = run(None)
    cold = archive(tmp_path, 1, "read", task=action_type)
    assert run(cold)[:2] == (off_action, old_prompts)
    cases = archive(tmp_path, 2, "read", task=action_type)
    action, prompts, ctl = run(cases)
    assert action == off_action and len(prompts) == len(old_prompts) == 2
    assert all(p.startswith(old + "\nPrior local observations") for p, old in zip(prompts, old_prompts))
    assert ctl.steps[0]["state"] == old_prompts[1]  # case text is not fed back into LoRA training records
    assert cases.summary()["queries"] == 1 and ctl.summary()["context"]["extra_calls"] == 1
