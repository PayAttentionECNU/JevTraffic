"""Single-intersection SUMO environment from SingleTSCBaselines (TransSimHub), for the v2 experiments.

We reuse the benchmark's own pipeline (`TSCEnvironment` -> optional `TSCEventWrapper` -> `TSCInfoWrapper`),
so the simulation, special-event injection and per-movement features are exactly the benchmark's. One
`step` is one decision: 5 s of green, plus 3 s of yellow when the phase changes, 1000 s per episode.

What we add on top:
  * the benchmark MaxPressure's min/max-green rule as an explicit constraint (`allowed_phases`), so every
    controller (rule, Jev, event plans) obeys the same safety layer, and `mp_action` reproduces their MP exactly;
  * V2X: special vehicles (ambulance / police / fire engine) on incoming lanes, with the phase that serves them;
  * ground-truth event labels (for training / evaluating the situation question) and tripinfo metrics,
    with special vehicles reported separately and accident barriers excluded;
  * day-level route perturbation (`days.py`) through a generated sumocfg.
"""
from __future__ import annotations

import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINGLETSC_DIR = os.environ.get("SINGLETSC_DIR", os.path.join(ROOT, "third_party", "SingleTSCBaselines"))
if SINGLETSC_DIR not in sys.path:
    sys.path.insert(0, SINGLETSC_DIR)
if "SUMO_HOME" not in os.environ:
    try:
        import sumo  # eclipse-sumo wheel

        os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    except ImportError:
        pass

JUNCTIONS = [
    "Beijing_Beihuan", "Beijing_Beishahe", "Beijing_Changjianglu", "Beijing_Gaojiaoyuan", "Beijing_Pinganli",
    "Beijing_Yongrunlu", "Chengdu_Chenghannanlu", "Chengdu_Guanghua", "Tianjin_zhijingdao",
    "France_Massy", "SouthKorea_Songdo", "Hongkong_YMT",
]
HELDOUT_JUNCTIONS = ["France_Massy", "SouthKorea_Songdo", "Hongkong_YMT"]
TRAIN_JUNCTIONS = [j for j in JUNCTIONS if j not in HELDOUT_JUNCTIONS]
PATTERNS = ["low_density", "high_density", "fluctuating_commuter", "increasing_demand", "random_perturbation"]
HELDOUT_PATTERN = "random_perturbation"
NETS = ["easy", "normal"]

# SUMO vehicle types used by the benchmark's EVENTS
SPECIAL_TYPES = {"emergency": "ambulance", "fire_engine": "fire_engine", "police": "police"}
BARRIER_TYPE = "barrier"
SITUATIONS = ["normal", "blockage", "ambulance", "police", "fire_engine"]
V2X_RANGE = 150.0  # m: special vehicles closer than this to the stop line are reported / labelled

COMPASS = ["east", "north", "west", "south"]


def _quiet_loguru():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("JEV_SUMO_LOG", "WARNING"))


def approach_angle(vec) -> float:
    """Bearing (deg, counter-clockwise from east) of where traffic comes from, given a lane heading vector
    (direction of travel): traffic heading north comes *from* the south."""
    return math.degrees(math.atan2(-vec[1], -vec[0])) % 360


def compass_from(vec) -> str:
    return COMPASS[int(((approach_angle(vec) + 45) % 360) // 90)]


def compass_names(angles: dict[str, float]) -> dict[str, str]:
    """Compass name per approach (edge -> bearing). The 90-degree bins are rotated by the smallest angle that gives
    every approach its own name; junctions whose approaches already fall into distinct bins keep the plain
    binning (rotation 0). Songdo is rotated ~45 degrees: plain binning named its four approaches north, north-2,
    south, south-2. With more than four approaches (or no rotation that separates them) duplicates get a suffix."""
    def binned(rot):
        return {e: COMPASS[int(((a - rot + 45) % 360) // 90)] for e, a in angles.items()}

    if len(angles) <= len(COMPASS):
        for d in range(46):
            for rot in ((d,) if d == 0 else (d, -d)):
                names = binned(rot)
                if len(set(names.values())) == len(names):
                    return names
    names, used = {}, {}
    for e, base in sorted(binned(0).items()):
        used[base] = used.get(base, 0) + 1
        names[e] = base if used[base] == 1 else f"{base}-{used[base]}"
    return names


@dataclass
class Movement:
    id: str
    approach: str        # where the traffic comes from (north/east/...), with a suffix if two approaches collide
    turn: str            # through / left / right / u-turn
    phases: list[int]
    queue: float         # waiting (speed < 1 m/s) vehicles on the incoming lanes
    moving: float
    wait: float          # mean waiting time (s) of vehicles on the incoming lanes
    down_queue: float    # waiting vehicles on the outgoing lane
    lanes: list[str]
    from_edge: str
    to_edge: str


@dataclass
class SumoObs:
    time: float
    movements: list[Movement]
    phase_movements: list[list[int]]   # phase -> movement indices it serves (right turns are usually free)
    current_phase: int
    green_steps: int
    allowed: list[int]                 # phases allowed by the min/max-green rule
    pressures: list[float]             # benchmark MaxPressure phase pressures
    special: list[dict] = field(default_factory=list)   # V2X: {id, kind, movement, phase, dist}
    discharged: list[float] = field(default_factory=list)  # per movement: vehicles that left the incoming lanes last step
    lane_halting: dict = field(default_factory=dict)       # incoming lane -> halting vehicles (speed < 0.1 m/s)
    lane_discharged: dict = field(default_factory=dict)    # incoming lane -> vehicles that left it last step
    lane_wait: dict = field(default_factory=dict)          # lane -> MaxPressure waiting count (incoming and outgoing)
    lane_phases: dict = field(default_factory=dict)        # lane -> phases it is bound to (static)
    green_seconds: float | None = None  # next_or_not mode: seconds since the current phase began (UniTSA's timer)

    @property
    def num_phases(self) -> int:
        return len(self.phase_movements)


def _turn(direction) -> str:
    d = str(direction or "").lower()
    return {"s": "through", "l": "left", "r": "right", "t": "u-turn"}.get(d, d or "through")


def scenario_paths(junction: str, env_name: str) -> dict:
    from junction_configs import load_junction_config

    cfg = load_junction_config(junction, env_name)
    root = os.path.dirname(cfg["sumo_cfg"])
    inp = ET.parse(cfg["sumo_cfg"]).getroot().find("input")
    routes = inp.find("route-files").get("value")
    adds = inp.find("additional-files").get("value") if inp.find("additional-files") is not None else ""
    cfg["route_file"] = os.path.normpath(os.path.join(root, routes))
    cfg["additional_files"] = [os.path.normpath(os.path.join(root, a.strip())) for a in adds.split(",") if a.strip()]
    return cfg


def write_sumocfg(path: str, net_file: str, route_file: str, additional_files: list[str]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    adds = ", ".join(additional_files)
    with open(path, "w") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
        <additional-files value="{adds}"/>
    </input>
</configuration>
""")
    return path


def route_pairs(route_file: str) -> list[list[str]]:
    """Distinct routes (edge lists) used by the scenario's background traffic."""
    seen, out = set(), []
    for veh in ET.parse(route_file).getroot().iter("vehicle"):
        r = veh.find("route")
        if r is None:
            continue
        edges = r.get("edges").split()
        if tuple(edges) not in seen:
            seen.add(tuple(edges))
            out.append(edges)
    return out


def random_events(junction: str, env_name: str, rng: random.Random, num_seconds: int = 1000,
                  p_accident: float = 0.6, p_special: float = 0.8) -> tuple[list, list]:
    """Random accidents (barrier on 1-2 lanes of an incoming edge near the stop line) and special vehicles
    (random type, on a route the background traffic uses). Same schema as the benchmark's EVENTS."""
    import sumolib

    paths = scenario_paths(junction, env_name)
    net = sumolib.net.readNet(paths["net_file"])
    routes = route_pairs(paths["route_file"])
    in_edges = sorted({r[0] for r in routes})
    accidents, specials = [], []
    if rng.random() < p_accident:
        for k in range(rng.choice([1, 1, 2])):
            e = net.getEdge(rng.choice(in_edges))
            nl, length = e.getLaneNumber(), e.getLength()
            pos = max(5.0, length - rng.uniform(10, 60))
            t0 = rng.uniform(50, num_seconds - 300)
            dur = rng.uniform(60, 180)
            lanes = rng.sample(range(nl), k=min(nl, rng.choice([1, 2]) if nl > 2 else 1))
            for li in lanes:
                accidents.append({"id": f"accident_{k}_{li}", "depart_time": round(t0), "edge_id": e.getID(),
                                  "lane_index": li, "position": round(pos, 1), "type": BARRIER_TYPE,
                                  "duration": round(dur)})
    if rng.random() < p_special:
        for k in range(rng.choice([1, 2, 3])):
            vtype = rng.choice(list(SPECIAL_TYPES))
            specials.append({"id": f"{SPECIAL_TYPES[vtype]}_{k}", "type": vtype,
                             "depart_time": round(rng.uniform(30, num_seconds - 150)), "route": rng.choice(routes)})
    return accidents, specials


class SumoTSCEnv:
    def __init__(self, junction: str, env_name: str, work_dir: str, events=None, route_file: str | None = None,
                 min_green: int = 2, max_green: int = 10, num_seconds: int | None = None, cell_length: float = 15.0,
                 seed: int | str = 0, action_type: str = "choose_next_phase"):
        """events: None, an EVENTS name of the junction config (e.g. "event_1"), or (accidents, special_vehicles).
        seed: SUMO seed; the benchmark scripts use 'random', which makes every run (even MaxPressure) differ.
        action_type: "choose_next_phase" (action = phase index) or "next_or_not" (UniTSA: 1 keep, 0 switch to the
        next phase); in next_or_not mode the phase safety layer is off and the agent owns its max-green guard."""
        _quiet_loguru()
        from junction_configs import load_event_config
        from tsc_env import TSCEnvironment, TSCEventWrapper, TSCInfoWrapper

        self.junction, self.env_name = junction, env_name
        self.min_green, self.max_green = min_green, max_green
        assert action_type in ("choose_next_phase", "next_or_not")
        self.action_type = action_type
        paths = scenario_paths(junction, env_name)
        self.cfg = paths
        self.tls_id = paths["tls_id"]
        self.num_seconds = num_seconds or paths["num_seconds"]
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        sumo_cfg = paths["sumo_cfg"]
        if route_file:
            sumo_cfg = write_sumocfg(os.path.join(work_dir, "day.sumocfg"), paths["net_file"], route_file,
                                     paths["additional_files"])
        if isinstance(events, str):
            events = load_event_config(junction, events)
        self.accidents, self.specials = events if events else ([], [])
        self.trip_info = os.path.join(work_dir, "tripinfo.xml")
        env = TSCEnvironment(sumo_cfg=sumo_cfg, net_file=paths["net_file"], num_seconds=self.num_seconds,
                             tls_ids=[self.tls_id], tls_action_type=action_type, use_gui=False,
                             trip_info=self.trip_info)
        env.tsc_env.sumo_seed = seed  # TSCEnvironment does not expose it; read at reset
        if self.accidents or self.specials:
            env = TSCEventWrapper(env, accident_configs=self.accidents, special_vehicle_configs=self.specials)
        self.base = env
        self.env = TSCInfoWrapper(env, tls_id=self.tls_id, cell_length=cell_length)
        self.done = False
        self.steps = 0
        self.constrain = action_type == "choose_next_phase"  # min/max-green safety layer; off for the benchmark's baselines

    # ------------------------------------------------------------------ simulation
    @property
    def incoming_lanes(self) -> list[str]:
        """Lanes approaching the junction (io_type incoming), set at reset."""
        return list(self._lane_phases)

    @property
    def conn(self):
        e = self.base
        while not hasattr(e, "tsc_env"):
            e = e.env
        return e.tsc_env.sumo

    def reset(self) -> SumoObs:
        raw, info = self.env.reset()
        self.raw, self.info = raw, info
        self.done, self.steps = False, 0
        self.current_phase, self.green_steps = 0, self.min_green  # as in the benchmark MaxPressureAgent
        if self.action_type == "next_or_not":
            self.current_phase = raw["tls_dynamic_features_seq"][-1][0].get("this_phase_index", 0)
            self.green_steps = 0
        self.green_seconds, self._last_t = 0.0, int(info.get("step_time", 0))
        self._static = self.env.static_lane_features
        self._num_phases = max((len(f.get("phase_binding", [])) for f in self._static.values()), default=1) or 1
        self._lane_len = {lid: self.conn.lane.getLength(lid) for lid in self._static}
        self._build_layout(raw)
        self._prev_ids = self._prev_mv = None
        self.v2x_seen: dict[str, dict] = {}  # ground truth: special vehicles that came within V2X range
        self.blockage_observable: dict[str, float] = {}  # accident lane -> first time it stalled on green
        self._lane_phases = {lid: [p for p, b in enumerate(f.get("phase_binding", [])) if b]
                             for lid, f in self._static.items() if f["io_type"][0] == 1}
        self.obs = self._observe(raw)
        return self.obs

    def step(self, action: int) -> SumoObs:
        assert not self.done
        action = int(action)
        if self.constrain and action not in self.obs.allowed:  # the safety layer has the last word
            action = self.obs.allowed[0] if len(self.obs.allowed) == 1 else max(
                self.obs.allowed, key=lambda p: self.obs.pressures[p])
        if self.action_type == "next_or_not":
            raw, _, _, dones, info = self.env.step(action)
            phase = raw["tls_dynamic_features_seq"][-1][0].get("this_phase_index", self.current_phase)
            t = int(info.get("step_time", self._last_t))
            if phase == self.current_phase:
                self.green_steps, self.green_seconds = self.green_steps + 1, self.green_seconds + max(t - self._last_t, 0)
            else:
                self.current_phase, self.green_steps, self.green_seconds = phase, 1, 0.0
            self._last_t = t
        else:
            if self.green_steps < self.min_green:
                self.green_steps += 1
            elif action == self.current_phase:
                self.green_steps += 1
            else:
                self.current_phase, self.green_steps = action, 1
            raw, _, _, dones, info = self.env.step(action)
        self.raw, self.info = raw, info
        self.steps += 1
        self.done = bool(dones)
        self.obs = self._observe(raw)
        self.last_action = action
        return self.obs

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ observation
    def _build_layout(self, raw):
        mv = raw["tls_dynamic_features_seq"][-1]
        angles = {}
        for m in mv:
            for lid in m["incoming_lane_ids"]:
                if lid in self._static:
                    angles[m["from_edge"]] = approach_angle(self._static[lid]["heading"])
                    break
        self._approach = compass_names(angles)
        self._phase_movements = [[i for i, m in enumerate(mv) if p in m["phase_indices"]]
                                 for p in range(self._num_phases)]

    def allowed_phases(self) -> list[int]:
        if not self.constrain:
            return list(range(self._num_phases))
        if self.green_steps < self.min_green:
            return [self.current_phase]
        if self.green_steps >= self.max_green and self._num_phases > 1:
            return [p for p in range(self._num_phases) if p != self.current_phase]
        return list(range(self._num_phases))

    def _lane_wait(self, lane_seq) -> dict:
        """Per-lane waiting count exactly as the benchmark MaxPressureAgent (max over the decision interval)."""
        from tsc_env.tools import aggregate_features_seq

        lanes = aggregate_features_seq(lane_seq, method="max")
        return {lid: sum(c["vehicle_count"] for c in cells if c["avg_speed"] < 1.0 and c["vehicle_count"] > 0)
                for lid, cells in lanes.items()}

    def pressures(self, waiting: dict, exclude: set | frozenset = frozenset()) -> list[float]:
        """Benchmark MaxPressure phase pressures; lanes in `exclude` (e.g. blocked lanes) are ignored."""
        out = [0.0] * self._num_phases
        for p in range(self._num_phases):
            for lid, f in self._static.items():
                pb = f.get("phase_binding", [])
                if p >= len(pb) or pb[p] != 1 or lid in exclude:
                    continue
                if f["io_type"][0] == 1:
                    out[p] += waiting.get(lid, 0)
                elif f["io_type"][1] == 1:
                    out[p] -= waiting.get(lid, 0)
        return out

    def _observe(self, raw) -> SumoObs:
        mv = raw["tls_dynamic_features_seq"][-1]
        movements = [Movement(
            id=m["movement_id"], approach=self._approach.get(m["from_edge"], m["from_edge"] or "?"),
            turn=_turn(m["direction"]), phases=list(m["phase_indices"]), queue=m["waiting_vehicle_count"],
            moving=m["moving_vehicle_count"], wait=m["avg_waiting_time"], down_queue=m["outgoing_waiting_vehicle_count"],
            lanes=list(m["incoming_lane_ids"]), from_edge=m["from_edge"], to_edge=m["to_edge"]) for m in mv]
        # discharge: vehicles that were on an incoming lane at the previous decision and are gone now
        lanes = sorted({l for m in movements for l in m.lanes})
        lane_ids = {l: set(self.conn.lane.getLastStepVehicleIDs(l)) for l in lanes}
        prev = self._prev_ids or {}
        lane_dis = {l: float(len(prev.get(l, set()) - lane_ids[l] - set().union(*lane_ids.values()))) for l in lanes}
        self._prev_ids = lane_ids
        ids = {i: set().union(*(lane_ids[l] for l in m.lanes)) for i, m in enumerate(movements)}
        prev_mv = self._prev_mv or {}
        discharged = [float(len(prev_mv.get(i, set()) - ids[i] - set().union(*lane_ids.values()))) for i in ids]
        self._prev_mv = ids
        wait = self._lane_wait(raw["lane_dynamic_features_seq"])
        t = float(self.conn.simulation.getTime())
        halting = {l: float(self.conn.lane.getLastStepHaltingNumber(l)) for l in lanes}
        for lane in self.blocked_lanes(t):  # ground truth: did the blockage ever show as a stall on green?
            if (self.current_phase in self._lane_phases.get(lane, []) and halting.get(lane, 0) >= 2
                    and lane_dis.get(lane, 0) == 0):
                self.blockage_observable.setdefault(lane, t)
        return SumoObs(time=float(self.conn.simulation.getTime()), movements=movements,
                       phase_movements=self._phase_movements, current_phase=self.current_phase,
                       green_steps=self.green_steps, allowed=self.allowed_phases(), pressures=self.pressures(wait),
                       special=self._v2x(movements, ids), discharged=discharged,
                       lane_halting=halting,
                       lane_discharged=lane_dis, lane_wait=wait, lane_phases=self._lane_phases,
                       green_seconds=self.green_seconds if self.action_type == "next_or_not" else None)

    def _v2x(self, movements, ids) -> list[dict]:
        out = []
        for i, m in enumerate(movements):
            for vid in ids[i]:
                vtype = self.conn.vehicle.getTypeID(vid)
                if vtype not in SPECIAL_TYPES:
                    continue
                lane = self.conn.vehicle.getLaneID(vid)
                dist = self._lane_len.get(lane, 0.0) - self.conn.vehicle.getLanePosition(vid)
                route = self.conn.vehicle.getRoute(vid)
                k = self.conn.vehicle.getRouteIndex(vid)
                nxt = route[k + 1] if 0 <= k < len(route) - 1 else None
                if nxt is not None and m.to_edge and nxt != m.to_edge:
                    continue  # the vehicle is on these lanes but takes another movement
                if dist <= V2X_RANGE:
                    self.v2x_seen.setdefault(vid, {"kind": SPECIAL_TYPES[vtype], "first": self.conn.simulation.getTime(),
                                                   "movement": i})
                    out.append({"id": vid, "kind": SPECIAL_TYPES[vtype], "movement": i,
                                "phase": m.phases[0] if m.phases else None, "dist": round(dist, 1),
                                "speed": round(self.conn.vehicle.getSpeed(vid), 1)})
        out.sort(key=lambda s: s["dist"])
        return out

    # ------------------------------------------------------------------ ground truth
    def blocked_lanes(self, t: float | None = None) -> list[str]:
        """Lanes with an active accident barrier at time t (ground truth, not visible to controllers)."""
        t = self.obs.time if t is None else t
        return [f'{a["edge_id"]}_{a["lane_index"]}' for a in self.accidents
                if a["depart_time"] <= t < a["depart_time"] + a["duration"]]

    def situation_label(self) -> str:
        """Ground truth for the situation question: nearest special vehicle wins, then an incoming-lane blockage."""
        if self.obs.special:
            return self.obs.special[0]["kind"]
        incoming = {l for m in self.obs.movements for l in m.lanes}
        if any(l in incoming for l in self.blocked_lanes()):
            return "blockage"
        return "normal"

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> dict:
        """Mean tripinfo duration / waitingTime / timeLoss over finished background trips, plus special vehicles."""
        if not os.path.exists(self.trip_info):
            return {}
        barrier_ids = {a["id"] for a in self.accidents}
        normal, special = [], []
        for t in ET.parse(self.trip_info).getroot().iter("tripinfo"):
            if t.get("id") in barrier_ids or t.get("vType") == BARRIER_TYPE:
                continue
            row = (float(t.get("duration")), float(t.get("waitingTime")), float(t.get("timeLoss")))
            (special if t.get("vType") in SPECIAL_TYPES else normal).append(row)

        def mean(rows, k):
            return round(sum(r[k] for r in rows) / len(rows), 3) if rows else None

        return {"ATT": mean(normal, 0), "AWT": mean(normal, 1), "timeLoss": mean(normal, 2), "trips": len(normal),
                "special_trips": len(special), "special_ATT": mean(special, 0), "special_AWT": mean(special, 1),
                "special_timeLoss": mean(special, 2)}
