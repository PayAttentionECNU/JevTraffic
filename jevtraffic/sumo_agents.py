"""Controllers for the SUMO single-intersection environment (v2).

`Controller` is the full system: a base phase controller (fine-tuned Jev, or MaxPressure as a reference)
plus, optionally, the context agent (short-term memory + triggers), a situation detector (Jev or rule) and the
event manager (notifications + preemption / blocked-lane plans). Everything obeys the env's min/max-green rule.
In next_or_not mode (v3, UniTSA's action space) the base controller answers keep/switch instead, under UniTSA's
45 s max-green guard, and an optional UniTSA `teacher` is run alongside to log agreement.

The benchmark's own rule controllers (FixTime / Webster / SOTL / MaxPressure) are wrapped by `BenchmarkAgent`.
"""
from __future__ import annotations

import time

from .case_memory import CaseMemory
from .events import EventManager, Notifier, rule_situation
from .memory import ContextAgent, TriggerConfig
from .sumo_env import SITUATIONS, SumoObs, SumoTSCEnv
from .sumo_prompts import (STATE_FORMATS, SITUATION_QUESTION, option_to_action, phase_question,
                           situation_state, switch_question)

MAX_GREEN_SECONDS = 45  # next_or_not guard, as jevtraffic.unitsa.MAX_GREEN (not imported: needs sb3)


def mp_action(obs: SumoObs, env: SumoTSCEnv | None = None, exclude: set | frozenset = frozenset()) -> int:
    """Benchmark MaxPressure over the allowed phases (first maximum, as in their agent)."""
    pressures = env.pressures(obs.lane_wait, exclude) if (exclude and env is not None) else obs.pressures
    return max(obs.allowed, key=lambda p: pressures[p])


class BenchmarkAgent:
    """One of SingleTSCBaselines' traditional agents, driven through our env (safety layer off, as in theirs)."""

    def __init__(self, env: SumoTSCEnv, name: str):
        n = env.cfg["num_phases"]  # constructor arguments as in the benchmark's run.py scripts
        if name == "maxpressure":
            from tsc_algos.traditional.maxpressure.maxpressure_agent import MaxPressureAgent as A
            kw = {"min_green_steps": env.min_green, "max_green_steps": env.max_green}
        elif name == "fixtime":
            from tsc_algos.traditional.fixtime.fixtime_agent import FixTimeAgent as A
            kw = {"num_phases": n, "phase_durations": env.cfg["fix_phase_durations"]}
        elif name == "webster":
            from tsc_algos.traditional.webster.webster_agent import WebsterAgent as A
            kw = {"num_phases": n, "max_green_steps": 12}
        elif name == "sotl":
            from tsc_algos.traditional.sotl.sotl_agent import SOTLAgent as A
            kw = {"num_phases": n, "threshold": 20, "max_green_steps": 12}
        else:
            raise ValueError(name)
        self.env, self.name, self.agent = env, name, A(**kw)
        env.constrain = name == "maxpressure"  # MaxPressure has the same rule built in; keep it explicit

    def act(self, obs: SumoObs) -> int:
        return int(self.agent.choose_action(self.env.env.lane_dynamic_features_seq, self.env.env.static_lane_features))

    def summary(self) -> dict:
        return {}


class Controller:
    def __init__(self, env: SumoTSCEnv, base: str = "jev", decider=None, memory: str = "none",
                 detector: str = "none", notify_path: str | None = None, tau_event: float = 0.5,
                 trigger_config: TriggerConfig | None = None, log_steps: bool = True, record_states: bool = False,
                 state_format: str = "full", teacher=None, case_memory: CaseMemory | None = None):
        assert base in ("jev", "mp") and detector in ("none", "rule", "jev")
        assert base == "mp" or decider is not None
        assert detector != "jev" or decider is not None
        self.env, self.base, self.decider, self.detector = env, base, decider, detector
        self.context = ContextAgent(memory if detector != "none" else "none", config=trigger_config)
        self.notifier = Notifier(notify_path, env.junction)
        self.events = EventManager(self.notifier, tau=tau_event) if detector != "none" else None
        self.log_steps = log_steps
        self.record_states = record_states  # keep state / options per Jev decision (long-term memory)
        self.render = STATE_FORMATS[state_format]
        self.switch_mode = env.action_type == "next_or_not"
        self.teacher = teacher
        self.case_memory = case_memory
        self.steps: list[dict] = []
        self.timing = {"phase_calls": 0, "phase_s": 0.0, "situation_calls": 0, "situation_s": 0.0, "skipped": 0}

    # ---------------------------------------------------------------- situation
    def _situation(self, obs: SumoObs, triggers, note: str) -> tuple[str, float, list[float] | None]:
        if self.detector == "rule":
            label, conf = rule_situation(obs, triggers)
            return label, conf, None
        t0 = time.time()
        d = self.decider.decide(situation_state(obs, self.context.memory.items()) + note, SITUATION_QUESTION)
        self.timing["situation_calls"] += 1
        self.timing["situation_s"] += time.time() - t0
        return SITUATIONS[d.index], d.confidence, d.probabilities

    # ---------------------------------------------------------------- phase
    def act(self, obs: SumoObs) -> int:
        self.context.observe(obs)
        rec = {"t": obs.time, "allowed": list(obs.allowed), "mp": mp_action(obs)}
        note = ""
        if self.case_memory:
            match = self.case_memory.retrieve(obs, self.switch_mode)
            if match:
                note = "\n" + self.case_memory.render(match)
                rec["case_match"] = match
        if self.switch_mode:  # MaxPressure's choice as keep (1) / switch (0)
            rec["mp"] = int(rec["mp"] == obs.current_phase)
        if self.events:
            self.events.update(obs)
            triggers = self.context.triggers(obs)
            if triggers:
                label, conf, probs = self._situation(obs, triggers, note)
                self.events.on_situation(obs, triggers, label, conf, self.detector)
                rec.update(triggers=[t.kind for t in triggers], situation=label, situation_conf=round(conf, 3),
                           situation_probs=probs and [round(p, 4) for p in probs])
        rec["gt"] = self.env.situation_label()
        if self.teacher is not None:
            rec["teacher"] = self.teacher.act(obs)  # once per step: it keeps its own green timer
        blocked = self.events.blocked_lanes() if self.events else set()
        action = self.events.override(obs) if self.events else None
        if self.switch_mode:
            action = self._act_switch(obs, rec, blocked, action, note)
        else:
            action = self._act_phase(obs, rec, blocked, action, note)
        rec["action"] = action
        if blocked:
            rec["blocked"] = sorted(blocked)
        if self.log_steps:
            self.steps.append(rec)
        return action

    def _decide(self, state, q):
        t0 = time.time()
        d = self.decider.decide(state, q)
        self.timing["phase_calls"] += 1
        self.timing["phase_s"] += time.time() - t0
        return d

    def _act_switch(self, obs: SumoObs, rec: dict, blocked, target, note: str) -> int:
        """next_or_not: 1 keep, 0 switch to the next phase."""
        conf = None
        if target is not None:  # event plan: keep while its phase is green, otherwise cycle towards it
            rec["preempt"] = True
            action = 1 if target == obs.current_phase else 0
        elif obs.green_seconds >= MAX_GREEN_SECONDS:
            action = 0
            self.timing["skipped"] += 1
        elif self.base == "mp":
            action = 1 if mp_action(obs, self.env, blocked) == obs.current_phase else 0
        else:
            q = switch_question(obs)
            state = self.render(obs, blocked)
            d = self._decide(state + note, q)
            action, conf = option_to_action(d.index), d.confidence
            rec["probs"] = [round(p, 4) for p in d.probabilities]
            rec["conf"] = round(conf, 4)
            if self.record_states:
                rec.update(state=state, question=q.question, options=list(q.options))
        self.context.memory.set_confidence(conf)
        return action

    def _act_phase(self, obs: SumoObs, rec: dict, blocked, action, note: str) -> int:
        conf = None
        if action is not None:
            rec["preempt"] = True
        elif len(obs.allowed) == 1:
            action = obs.allowed[0]
            self.timing["skipped"] += 1
        elif self.base == "mp":
            action = mp_action(obs, self.env, blocked)
        else:
            q, phases = phase_question(obs)
            state = self.render(obs, blocked)
            d = self._decide(state + note, q)
            action, conf = phases[d.index], d.confidence
            rec["probs"] = [round(p, 4) for p in d.probabilities]
            rec["conf"] = round(conf, 4)
            if self.record_states:
                rec.update(state=state, question=q.question, options=list(q.options), phases=phases,
                           mp_blocked=mp_action(obs, self.env, blocked))
        self.context.memory.set_confidence(conf)
        return action

    def summary(self) -> dict:
        decided = [s for s in self.steps if "probs" in s]
        out = {"context": self.context.summary(), "timing": dict(self.timing),
               "events": self.events.stats if self.events else {}, "notifications": len(self.notifier.events)}
        if self.case_memory:
            out["case_memory"] = self.case_memory.summary()
        if decided:
            out["agree_mp"] = round(sum(s["action"] == s["mp"] for s in decided) / len(decided), 4)
            out["mean_conf"] = round(sum(s["conf"] for s in decided) / len(decided), 4)
            if self.teacher is not None:
                out["agree_teacher"] = round(sum(s["action"] == s["teacher"] for s in decided) / len(decided), 4)
        if self.teacher is not None:
            out["teacher_ms_per_call"] = self.teacher.summary()["ms_per_call"]
        calls = self.timing["phase_calls"] + self.timing["situation_calls"]
        total_s = self.timing["phase_s"] + self.timing["situation_s"]
        if calls:
            out["ms_per_call"] = round(1000 * total_s / calls, 1)
        out["ms_per_step"] = round(1000 * total_s / max(len(self.steps), 1), 1)
        out["calls_per_step"] = round(calls / max(len(self.steps), 1), 3)
        return out


def score_detection(env: SumoTSCEnv, notifications: list[dict], slack: float = 30.0) -> dict:
    """Event-level detection quality against the ground truth of the episode.

    special vehicles: every vehicle that came within V2X range is an event; detected if a preemption started for it.
    blockages: every accident lane (on an incoming lane) is an event over [depart, depart + duration]; detected if a
    blockage notification names that lane inside the interval (+slack). Other blockage notifications are false alarms.
    """
    out = {}
    starts = {n["vehicle"]: n["time"] for n in notifications if n["action"] == "preemption_start"}
    ev = env.v2x_seen
    out["special_events"] = len(ev)
    out["special_detected"] = sum(v in starts for v in ev)
    lat = [starts[v] - ev[v]["first"] for v in ev if v in starts]
    out["special_latency"] = round(sum(lat) / len(lat), 1) if lat else None
    incoming = {l for m in env.obs.movements for l in m.lanes}
    acc = [a for a in env.accidents if f'{a["edge_id"]}_{a["lane_index"]}' in incoming]
    flags = [n for n in notifications if n["action"] == "lane_excluded"]
    hit, lat, used = 0, [], set()
    for a in acc:
        lane = f'{a["edge_id"]}_{a["lane_index"]}'
        t0, t1 = a["depart_time"], a["depart_time"] + a["duration"] + slack
        match = [n for n in flags if n["lane"] == lane and t0 <= n["time"] <= t1]
        if match:
            hit += 1
            lat.append(match[0]["time"] - t0)
            used.update(id(n) for n in match)
    out["blockage_events"] = len(acc)
    out["blockage_detected"] = hit
    # observable: the blocked lane stalled on green at least once (in SUMO, traffic often changes lanes around a
    # one-lane barrier, so many accidents never show up in the signal's detectors at all)
    obs_lanes = {f'{a["edge_id"]}_{a["lane_index"]}' for a in acc} & set(env.blockage_observable)
    out["blockage_observable"] = len(obs_lanes)
    out["blockage_observable_detected"] = sum(
        any(n["lane"] == l and n["time"] >= env.blockage_observable[l] - 60 for n in flags) for l in obs_lanes)
    out["blockage_latency"] = round(sum(lat) / len(lat), 1) if lat else None
    out["blockage_false_alarms"] = sum(id(n) not in used for n in flags)
    return out
