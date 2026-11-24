"""Run paired PR module drops and one-at-a-time parameter changes on the SUMO server."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "jev-pr1-four-scenarios"
sys.path[:0] = [str(PRIOR / "src"), str(PRIOR)]
import replay
from screen import DISTILLED, PY, SRC, read, save

SWEEPS = {
    "min_similarity": (0.45, 0.60, 0.75, 0.85, 0.95),
    "half_life_days": (2, 4, 7, 14, 28),
    "min_days": (1, 2, 3, 4, 5),
}
DEFAULTS = {"min_similarity": 0.75, "half_life_days": 7, "min_days": 2}
ABLATIONS = ("no_event_response", "no_case", "no_temporal_prompt", "rules_for_events")
BASE = {"kind": "jev", "memory": "pr", "detector": "jev", "adapters": [str(DISTILLED)]}
WINDOW = 60


def key(field, value):
    return "full" if value == DEFAULTS[field] else f"{field}_{value:g}"


def prepare():
    from fixtures import CONDITIONS, DEMAND, OFFSETS, digest, event_bundle, object_digest, rows_for
    from jevtraffic.case_memory import CaseSettings
    from jevtraffic.sumo_env import HELDOUT_JUNCTIONS

    for field, value in DEFAULTS.items():
        assert getattr(CaseSettings(), field) == value
    OFFSETS["module-study"] = 731000000
    import fixtures

    fixtures.ROOT = ROOT
    rows = list(rows_for("module-study", HELDOUT_JUNCTIONS, 2, DEMAND[:2]))
    # Rotate vehicle class and same/different approaches by junction and network.
    routes = {}
    for row in rows:
        j = HELDOUT_JUNCTIONS.index(row["junction"])
        n = ("easy", "normal").index(row["network"])
        event_index = 2 * (2 * j + n) + row["replicate"]
        row["events"] = event_bundle(row["junction"], row["environment"], row["condition"],
                                     event_index, row["seed"], row["horizon"])
        row["event_sha256"] = object_digest(row["events"])
        bundles = routes.setdefault(row["route"], {})
        bundles[row["condition"]] = row["events"]

    for row in rows:
        bundles = routes[row["route"]]
        # One common temporal anchor for none/special, another for accident/both.
        # The accident condition borrows the combined fixture's priority dispatch time.
        target = "both" if row["condition"] in ("accident", "both") else "special"
        anchor = bundles[target][1][0]["depart_time"]
        assert WINDOW <= anchor <= row["horizon"] - WINDOW
        row["anchor"] = anchor

    archives = {}
    for row in rows:
        rel = Path(row["junction"]) / row["network"] / "next_or_not" / "cases-v1.sqlite"
        if str(rel) not in archives:
            archives[str(rel)] = digest(PRIOR / "history" / rel)
    assert len(rows) == 96 and all(set(b) == set(CONDITIONS) for b in routes.values())
    sensitivity = [i for i, row in enumerate(rows) if row["replicate"] == 0]
    return {
        "rows": rows, "sensitivity_indices": sensitivity,
        "ablations": list(ABLATIONS), "sweeps": {field: list(values) for field, values in SWEEPS.items()},
        "defaults": DEFAULTS, "window_seconds": WINDOW,
        "archive_sha256": archives, "manifest_sha256": object_digest(rows),
        "weights_sha256": digest(DISTILLED / "adapter_model.safetensors"),
        "runner_sha256": digest(Path(__file__)),
        "source_sha256": {rel: digest(SRC / rel) for rel in (
            "jevtraffic/case_memory.py", "jevtraffic/memory.py", "jevtraffic/events.py",
            "jevtraffic/sumo_agents.py", "jevtraffic/sumo_env.py", "jevtraffic/sumo_prompts.py")},
        "parent_replay_sha256": digest(PRIOR / "replay.py"),
    }


def arm_settings(plan):
    settings = {
        "full": BASE,
        "no_case": dict(BASE, memory="off"),
        "no_temporal_prompt": dict(BASE, temporal_prompt=False),
        "rules_for_events": dict(BASE, detector="rule"),
        "no_event_response": dict(BASE, event_response=False),
    }
    for field, values in plan["sweeps"].items():
        for value in values:
            label = key(field, value)
            if label != "full":
                settings[label] = dict(BASE, case_settings={field: value})
    return settings


def storage(row, arm):
    from jevtraffic.case_memory import CaseMemory, CaseSettings

    if arm["memory"] == "off":
        return None
    settings = CaseSettings(**arm.get("case_settings", {}))
    return CaseMemory(PRIOR / "history", row["junction"], row["network"], "next_or_not", 7, "read", settings)


def current_only(obs, history):
    from jevtraffic.sumo_prompts import situation_state

    return situation_state(obs, [])


def disabled_override(obs):
    return None


def disabled_blocked():
    return set()


def replay_study(row, arm, policy, path):
    from unittest.mock import patch

    from jevtraffic import sumo_agents

    sampled = []
    ctl = None
    window_end = row["anchor"] + WINDOW

    def instrumented_controller(arm, env, policy, cases):
        nonlocal ctl
        ctl = sumo_agents.Controller(env, base="jev", decider=policy, memory="triggered",
                                    detector=arm["detector"], state_format="full", case_memory=cases)
        if arm.get("event_response") is False:
            ctl.events.override = disabled_override
            ctl.events.blocked_lanes = disabled_blocked

        def act(obs):
            if obs.time <= window_end or not sampled or sampled[-1][0] < window_end:
                sampled.append((obs.time, sum(obs.lane_halting.values())))
            return ctl.act(obs)

        return act

    prompt = current_only if arm.get("temporal_prompt") is False else sumo_agents.situation_state
    with patch.object(replay, "controller", instrumented_controller), patch.object(replay, "storage", storage), \
            patch.object(sumo_agents, "situation_state", prompt):
        result = replay.replay(row, arm, policy, path)
    assert ctl is not None

    def average(start, end):
        area = sum(halting * max(0, min(finish, end) - max(time, start))
                   for (time, halting), (finish, _) in zip(sampled, sampled[1:]))
        assert sampled[0][0] <= start and sampled[-1][0] >= end
        return area / (end - start)

    trigger_labels = Counter(step["situation"] for step in ctl.steps if "situation" in step)
    result["instrumentation"] = {
        "snapshot_calls": ctl.timing["phase_calls"], "situation_calls": ctl.timing["situation_calls"],
        "labels": dict(trigger_labels), "triggered": dict(ctl.context.fired),
        "case_prompt_calls": ctl.case_calls, "event_manager": dict(ctl.events.stats),
        "preemption_decisions": sum(bool(step.get("preempt")) for step in ctl.steps),
        "blocked_decisions": sum(bool(step.get("blocked")) for step in ctl.steps),
        "anchor": row["anchor"], "halting_before": average(row["anchor"] - WINDOW, row["anchor"]),
        "halting_after": average(row["anchor"], window_end),
    }
    if arm.get("event_response") is False:
        assert result["instrumentation"]["preemption_decisions"] == result["instrumentation"]["blocked_decisions"] == 0
    return result


def work(index, junction, network):
    path = ROOT / "jobs" / f"{index:02d}.json"
    plan = read(ROOT / "plan.json")
    jobs = read(path)
    settings = arm_settings(plan)
    output = path.with_suffix(".jsonl")
    existing = {}
    if output.is_file():
        # Resume after an interrupted write, also catching inconsistent duplicate records.
        payload = output.read_bytes()
        lines = payload.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            with output.open("r+b") as stream:
                stream.truncate(len(payload) - len(lines.pop()))
        for line in lines:
            record = json.loads(line)
            label = record["arm"]
            row = record["result"]
            existing[label, row["id"]] = row
        assert len(existing) == len(lines)
    all_expected = {(label, plan["rows"][i]["id"]) for label, i in jobs}
    assert len(all_expected) == len(jobs) and existing.keys() <= all_expected
    jobs = [(label, i) for label, i in jobs
            if (plan["rows"][i]["junction"], plan["rows"][i]["network"]) == (junction, network)]
    jobs = [(label, i) for label, i in jobs if (label, plan["rows"][i]["id"]) not in existing]
    if not jobs:
        return
    policy = replay.make_policy(BASE)
    with output.open("a", buffering=1) as stream:
        for label, i in jobs:
            row = plan["rows"][i]
            with tempfile.TemporaryDirectory(prefix=f"sim-{index:02d}-", dir=ROOT / "jobs") as folder:
                result = replay_study(row, settings[label], policy, Path(folder))
            record = json.dumps({"arm": label, "result": result}, separators=(",", ":"), default=float)
            stream.write(record + "\n")
    print(f"worker {index} {junction} {network} saved {len(jobs)} episodes", flush=True)


def run_slot(index):
    plan = read(ROOT / "plan.json")
    jobs = read(ROOT / "jobs" / f"{index:02d}.json")
    geometries = sorted({(plan["rows"][i]["junction"], plan["rows"][i]["network"]) for _, i in jobs})
    offset = index // 8 % len(geometries)
    geometries = geometries[offset:] + geometries[:offset]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(index % 8), "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONPATH": str(PRIOR / "deps"), "PYTHONUNBUFFERED": "1"}
    for junction, network in geometries:
        run = [str(PY), str(Path(__file__).resolve()), "--worker", str(index), "--geometry", junction, network]
        for attempt in range(2):
            with (ROOT / "jobs" / f"{index:02d}.log").open("a") as log:
                child = subprocess.run(run, env=env, cwd=SRC, stdout=log, stderr=subprocess.STDOUT)
            if child.returncode == 0:
                break
            print("retry", index, junction, network, child.returncode, attempt, flush=True)
        else:
            raise RuntimeError(f"worker {index} {junction} {network} failed twice, see jobs/{index:02d}.log")


def run(slots):
    path = ROOT / "plan.json"
    locked = read(path) if path.exists() else prepare()
    if not path.exists():
        save(path, locked)
    # Spread long and short networks while making complete module-drop pairs arrive early.
    tasks = [(label, i) for i in range(len(locked["rows"])) for label in ("full", *ABLATIONS)]
    others = [label for label in arm_settings(locked) if label not in ("full", *ABLATIONS)]
    tasks.extend((label, i) for i in locked["sensitivity_indices"] for label in others)
    for index in range(slots):
        loc = ROOT / "jobs" / f"{index:02d}.json"
        jobs = [list(t) for t in tasks[index::slots]]
        if loc.exists():
            assert read(loc) == jobs
        else:
            save(loc, jobs)
    print(f"locked {len(locked['rows'])} ablation and {len(locked['sensitivity_indices'])} sensitivity fixtures, "
          f"{len(tasks)} total episodes across {slots} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
        list(pool.map(run_slot, range(slots)))
    save(ROOT / "complete.json", {"episodes": len(tasks), "workers": slots,
                                  "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print("all episodes complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--geometry", nargs=2)
    args = parser.parse_args()
    if args.worker is not None:
        work(args.worker, *args.geometry)
        sys.stdout.flush()
        os._exit(0)
    run(args.slots)
