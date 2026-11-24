"""Paired SUMO case-memory benchmark. MP measures retrieval quality; --model/--adapters evaluate Jev control."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevtraffic.days import make_day
from jevtraffic.sumo_env import HELDOUT_JUNCTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junction", choices=HELDOUT_JUNCTIONS, required=True)
    parser.add_argument("--action_type", choices=["choose_next_phase", "next_or_not"], default="next_or_not")
    parser.add_argument("--model", help="optional Jev backbone; default evaluator is MP + rule detector")
    parser.add_argument("--adapters", nargs="+", default=[])
    parser.add_argument("--history_days", type=int, default=2, help="collect days 0 to N-1 with official events")
    parser.add_argument("--first_test_day", type=int, help="start of three frozen test days (default N)")
    parser.add_argument("--out", type=Path, default=ROOT / "records" / "case_eval")
    args = parser.parse_args()
    if bool(args.model) != bool(args.adapters):
        parser.error("provide --model together with --adapters for a Jev evaluation")
    first_test = args.first_test_day if args.first_test_day is not None else args.history_days
    if args.history_days < 1 or first_test < args.history_days:
        parser.error("test days must follow at least one history day")
    folder = args.out.resolve() / args.junction / args.action_type
    protocol = f"h{args.history_days}t{first_test}"
    if (args.history_days, first_test) != (2, 2):
        folder /= protocol
    folder.mkdir(parents=True, exist_ok=True)
    reference = ["--agent", "mp", "--detector", "rule"]
    policy = (["--agent", "jev", "--detector", "jev", "--model", args.model, "--adapters", *args.adapters]
              if args.model else reference)
    results = []
    history = list(range(args.history_days))
    schedule = [(d, "event_1") for d in history] + list(enumerate(("event_1", "random:7", "none"), first_test))
    for day, event in schedule:
        info = make_day(args.junction, "normal", day, str(folder / "routes"))
        pair = {}
        for mode in (["collect"] if day in history else ["off", "read"]):
            output = folder / f"day{day}_{mode}"
            cmd = [sys.executable, str(ROOT / "run_sumo.py"), "--junction", args.junction,
                   "--env_name", info["env_name"], "--route_file", info["route_file"], "--events", event,
                   "--action_type", args.action_type, "--case_store", str(folder / "history"), "--case_day", str(day),
                   "--case_mode", mode, "--out_dir", str(output), "--save_steps", "--tag", protocol,
                   *(reference if mode == "collect" else policy)]
            subprocess.run(cmd, check=True)
            # A separate output root per policy/checkpoint makes the paired result unambiguous.
            files = list(output.glob(f"*__{protocol}.json"))
            if len(files) != 1:
                raise ValueError(f"expected one run in {output}; use a separate --out for each model")
            pair[mode] = json.loads(files[0].read_bytes())
        if day in history:
            continue
        off, read = pair["off"], pair["read"]
        same_actions = [s["action"] for s in off["steps"]] == [s["action"] for s in read["steps"]]
        if not args.model and not (same_actions and off["metrics"] == read["metrics"]):
            raise RuntimeError(f"MP control changed during read-only diagnostics on day {day}")
        results.append(dict(day=day, pattern=info["pattern"], events=event, same_actions=same_actions,
                            case_memory=read["summary"]["case_memory"], off_metrics=off["metrics"],
                            read_metrics=read["metrics"], off_detection=off["detection"],
                            read_detection=read["detection"]))
    summary = dict(junction=args.junction, action_type=args.action_type, model=args.model, adapters=args.adapters,
                   history_days=history, evaluation=results)
    (folder / "paired.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(folder / "paired.json")


if __name__ == "__main__":
    main()
