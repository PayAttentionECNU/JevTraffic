"""Separate the event response into its control components and measure trigger overhead."""
import argparse
import concurrent.futures
import hashlib
import os
import sys
from functools import partial
from pathlib import Path

import run_study as study
from screen import read, save

HERE = Path(__file__).resolve()
PARENT = HERE.parent
ROOT = PARENT / "components"
COMPONENTS = {
    "always_assess": dict(study.BASE, context_mode="always"),
    "no_preemption": dict(study.BASE, preemption=False),
    "no_blockage_mask": dict(study.BASE, blockage_mask=False),
}
study.ROOT = ROOT


def run(slots):
    parent = PARENT / "plan.json"
    plan = read(parent)
    # Every component is matched against the unmodified arm in the locked main experiment.
    selections = {
        "always_assess": plan["sensitivity_indices"],
        "no_preemption": [i for i, row in enumerate(plan["rows"]) if row["condition"] in ("special", "both")],
        "no_blockage_mask": [i for i, row in enumerate(plan["rows"]) if row["condition"] in ("accident", "both")],
    }
    plan = {"rows": plan["rows"], "components": COMPONENTS, "selections": selections,
            "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
            "core_sha256": hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest()}
    path = ROOT / "plan.json"
    if path.exists():
        assert read(path) == plan
    else:
        save(path, plan)
    tasks = [(label, i) for i in range(len(plan["rows"])) for label, cases in selections.items() if i in cases]
    for index in range(slots):
        path = ROOT / "jobs" / f"{index:02d}.json"
        jobs = [list(t) for t in tasks[index::slots]]
        if path.exists():
            assert read(path) == jobs
        else:
            save(path, jobs)
    print(f"locked {len(tasks)} component episodes", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
        list(pool.map(partial(study.run_slot, launcher=HERE), range(slots)))
    save(ROOT / "complete.json", {"episodes": len(tasks), "workers": slots})
    print("all component episodes complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--geometry", nargs=2)
    args = parser.parse_args()
    if args.worker is not None:
        study.arm_settings = lambda plan: plan["components"]
        study.work(args.worker, *args.geometry)
        sys.stdout.flush()
        os._exit(0)
    run(args.slots)
