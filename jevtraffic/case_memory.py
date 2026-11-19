"""Retrievable SUMO transitions. One immutable episode per simulated day and local road layout."""
from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .sumo_env import SumoObs
from .sumo_prompts import STEP_SECONDS, phase_name

LOOKBACK_DAYS = 28
HALF_LIFE_DAYS = 7
MIN_DAYS, MIN_INTERVALS = 2, 3
MIN_SIMILARITY, MIN_SCORE = 0.75, 0.42


@dataclass(frozen=True)
class Scene:
    path: tuple[str, ...]
    features: tuple[float, ...]

    @classmethod
    def observe(cls, obs: SumoObs) -> Scene:
        lanes = obs.lane_halting
        total = sum(lanes.values())
        green = [lane for lane, phases in obs.lane_phases.items() if obs.current_phase in phases]
        served = sum(lanes.get(lane, 0) for lane in green)
        share = served / total if total else 0.5
        stalls = sorted(lane for lane in green if obs.green_steps >= 2 and lanes.get(lane, 0) >= 2
                        and obs.lane_discharged.get(lane, 0) == 0)
        kinds = sorted({s["kind"] for s in obs.special})
        family = "V2X " + "+".join(kinds) if kinds else ("queued green without discharge" if stalls else "ordinary")
        if not total:
            load = "empty"
        elif share > 0.65:
            load = "green-heavy"
        elif share < 0.35:
            load = "red-heavy"
        else:
            load = "mixed"
        locations = sorted({s["movement"] for s in obs.special})
        detail = f"P{obs.current_phase}/peak P{obs.pressures.index(max(obs.pressures))}"
        if locations:
            detail += "/V2X " + ",".join(str(i) for i in locations)
        if stalls:
            detail += "/stalled " + ",".join(stalls)
        held = obs.green_seconds if obs.green_seconds is not None else STEP_SECONDS * obs.green_steps
        age_band = "<15s" if held < 15 else "15-29s" if held < 30 else "30s+"
        slot = f"window {int(obs.time // 300)}, green {age_band}"
        density = total / max(1, len(lanes))
        flow = sum(obs.lane_discharged.values()) / max(1, len(lanes))
        features = (density / (density + 2), share, held / (held + 15), flow / (flow + 1))
        return cls((family, load, detail, slot), features)


@dataclass
class _Evidence:
    count: int = 0
    days: set[int] = field(default_factory=set)
    weight: float = 0
    features: list[float] = field(default_factory=lambda: [0.] * 4)
    queue_change: float = 0
    departures: float = 0
    latest: tuple = ()

    def add(self, row, day, weight):
        _, features, _, t, change, left = row
        self.count += 1
        self.days.add(day)
        self.weight += weight
        for i, x in enumerate(features):
            self.features[i] += weight * x
        self.queue_change += weight * change
        self.departures += weight * left
        self.latest = max(self.latest, (day, t, change, left))

    def match(self, scene, day, detailed):
        n_days = len(self.days)
        if n_days < MIN_DAYS or self.count < MIN_INTERVALS:
            return None
        distance = sum(abs(x - s / self.weight) for x, s in zip(scene.features, self.features))
        similarity = 1 - distance / len(scene.features)
        last, t, change, left = self.latest
        recency = 2 ** (-(day - last) / HALF_LIFE_DAYS)
        support = math.sqrt((n_days / (n_days + 2)) * (self.count / (self.count + 3)))
        score = similarity * math.sqrt(support * recency)
        if similarity < MIN_SIMILARITY or score < MIN_SCORE:
            return None
        result = dict(steps=self.count, days=n_days, last_day=last, similarity=round(similarity, 3),
                      support=round(support, 3), recency=round(recency, 3), score=round(score, 3),
                      queue_change=round(self.queue_change / self.weight, 2),
                      departures=round(self.departures / self.weight, 2))
        if detailed:
            result["example"] = dict(day=last, t=t, queue_change=round(change, 2), departures=round(left, 2))
        return result


class CaseMemory:
    """Load prior days once, then look up at most four tree nodes per decision."""

    def __init__(self, root, junction, network, action_type, day: int, mode: str):
        if mode not in ("collect", "read", "online") or day < 0:
            raise ValueError("case memory requires a non-negative simulated day and collect/read/online mode")
        self.day, self.mode = day, mode
        self.db = Path(root) / junction / network / action_type / "cases-v1.sqlite"
        self.nodes = defaultdict(lambda: defaultdict(_Evidence))
        self.by_action = defaultdict(_Evidence)
        self.pending = []
        self.depths = Counter()
        self.scene_queries = Counter()
        self.scene_hits = Counter()
        self.queries = 0
        self.lookup_s = 0.
        self.last_lookup = None
        self.scored = 0
        self.errors = Counter()
        if mode == "read" and not self.db.is_file():
            raise FileNotFoundError(f"no case archive at {self.db}; collect earlier days first")
        if mode != "collect" and self.db.is_file():
            with self._connect(readonly=True) as db:
                # One snapshot per episode. Today's rows, future days and old cases are ineligible.
                if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='episodes'").fetchone():
                    return  # a concurrent first publisher has created the file but not yet committed its table
                for past, payload in db.execute("SELECT day, cases FROM episodes WHERE day >= ? AND day < ? "
                                                "ORDER BY day", (day - LOOKBACK_DAYS, day)):
                    weight = 2 ** (-(day - past) / HALF_LIFE_DAYS)
                    for row in json.loads(payload):
                        path, _, action, *_ = row
                        self.by_action[action].add(row, past, weight)
                        for depth in range(1, len(path) + 1):
                            self.nodes[tuple(path[:depth])][action].add(row, past, weight)

    @contextmanager
    def _connect(self, readonly=False):
        db = (sqlite3.connect(self.db.resolve().as_uri() + "?mode=ro", uri=True) if readonly
              else sqlite3.connect(self.db, timeout=30))
        try:
            with db:
                yield db
        finally:
            db.close()

    def record(self, before: SumoObs, after: SumoObs) -> None:
        """Use the *observed* new green and next interval, including the final step and forced actions."""
        elapsed = after.time - before.time
        if elapsed <= 0:
            return
        action = "keep" if after.current_phase == before.current_phase else f"P{after.current_phase}"
        scale = STEP_SECONDS / elapsed  # switching can include yellow; compare rates per five seconds
        delta = scale * (sum(after.lane_halting.values()) - sum(before.lane_halting.values()))
        departures = scale * sum(after.lane_discharged.values())
        # Score predictions only after execution, for the same observed action and intervals in both arms.
        if self.last_lookup is not None and self.last_lookup[0] == before.time:
            for o in self.last_lookup[1]["outcomes"]:
                if o["phase"] == after.current_phase:
                    baseline = self.by_action[action]
                    self.scored += 1
                    for key, actual in (("queue_change", delta), ("departures", departures)):
                        self.errors["tree_" + key] += abs(actual - o[key])
                        mean = round(getattr(baseline, key) / baseline.weight, 2)
                        self.errors["action_only_" + key] += abs(actual - mean)
            self.last_lookup = None
        if self.mode != "read":
            scene = Scene.observe(before)
            self.pending.append((scene.path, scene.features, action, before.time, delta, departures))

    def publish(self) -> None:
        """Atomic, idempotent publication. A day cannot silently acquire a different trajectory on retry."""
        if self.mode == "read":
            return
        self.db.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.pending, separators=(",", ":"), allow_nan=False)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS episodes (day INTEGER PRIMARY KEY, cases TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO episodes VALUES (?, ?)", (self.day, payload))
            saved, = db.execute("SELECT cases FROM episodes WHERE day = ?", (self.day,)).fetchone()
            if saved != payload:
                raise ValueError(f"day {self.day} already contains another trajectory in {self.db}")

    def retrieve(self, obs: SumoObs, switch_mode: bool) -> dict | None:
        if self.mode == "collect":
            return None
        start = time.perf_counter()
        self.queries += 1
        scene = Scene.observe(obs)
        self.scene_queries[scene.path[0]] += 1
        phases = ([obs.current_phase, (obs.current_phase + 1) % obs.num_phases] if switch_mode else obs.allowed)
        legal = {"keep" if phase == obs.current_phase else f"P{phase}": phase for phase in phases}
        match = None
        for depth in range(len(scene.path), 0, -1):
            path = scene.path[:depth]
            outcomes = []
            for action, stats in self.nodes.get(path, {}).items():
                if action not in legal:
                    continue
                outcome = stats.match(scene, self.day, detailed=depth == len(scene.path))
                if outcome is not None:
                    name = "keep" if action == "keep" else "switch to " + phase_name(obs, legal[action])
                    outcomes.append(dict(action=name, phase=legal[action], **outcome))
            if outcomes:
                outcomes.sort(key=lambda x: (-x["score"], x["action"]))
                match = dict(day=self.day, depth=depth, path=path, outcomes=outcomes[:2])
                self.depths[depth] += 1
                self.scene_hits[path[0]] += 1
                break
        self.last_lookup = (obs.time, match) if match else None
        self.lookup_s += time.perf_counter() - start
        return match

    @staticmethod
    def render(match: dict) -> str:
        lines = ["Prior local observations (" + " / ".join(match["path"]) + "), averages per 5 s:"]
        for o in match["outcomes"]:
            lines.append(f"- {o['action']}: queue change {o['queue_change']:+.1f}, departures {o['departures']:.1f}"
                         f" ({o['steps']} intervals on {o['days']} days, last {match['day'] - o['last_day']}d ago, "
                         f"similarity {o['similarity']:.2f}).")
        first = match["outcomes"][0]
        if "example" in first:
            ex = first["example"]
            lines.append(f"Example: day {ex['day']} at {ex['t']:g}s, {first['action']}, "
                         f"queue change {ex['queue_change']:+.1f}, departures {ex['departures']:.1f} per 5 s.")
        return "\n".join(lines)

    def summary(self):
        return dict(mode=self.mode, day=self.day, queries=self.queries, hits=sum(self.depths.values()),
                    depth_counts=dict(self.depths), recorded=len(self.pending),
                    by_scene={k: dict(queries=n, hits=self.scene_hits[k]) for k, n in self.scene_queries.items()},
                    paired_intervals=self.scored,
                    paired_mae={key: round(error / self.scored, 3) for key, error in self.errors.items()},
                    lookup_ms=round(1000 * self.lookup_s / max(1, self.queries), 3))
