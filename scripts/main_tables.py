"""The two main tables (v4): zero-shot and few-shot ability across scenarios -> results/sumo/main_tables.md.

Table 1, zero-shot: every method as trained on the 9 training junctions (or rule-based), on
  seen (training junctions x training patterns), pattern (held-out pattern), junction (held-out junctions),
  each without events, with the official event_1 and with the random event sets. ATT in s, mean over runs.
Table 2, few-shot: held-out junctions only; each learning method adapted on the target junction with a budget of
  N episodes (0 = its zero-shot model), then evaluated on all 10 envs x events. The main table shows N = 10;
  the budget curve follows.
Methods are matched to run labels by `ROWS` (prefix of the label, then the bracketed model / adapter name).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jevtraffic.sumo_env import HELDOUT_JUNCTIONS, HELDOUT_PATTERN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "sumo", "runs")
GROUPS = ["seen", "pattern", "junction"]
EVENTS = ["none", "event_1", "random"]
BUDGETS = [0, 1, 10, 50]
STUDENT = os.environ.get("STUDENT", "unitsa-jev-0.5B-full")  # the distilled student the Jev rows are built on
RL = os.environ.get("RL", "rl5-0.5B-full")  # its RL adapter (scripts/select_rl_checkpoint.py); few-shot: RL-fs<N>_<J>
_S, _R = re.escape(STUDENT), re.escape(RL)

# (row name, regex on the run label; group "m" = model name for the learning methods)
ROWS = [
    ("FixTime", r"^fixtime$"),
    ("Webster", r"^webster$"),
    ("SOTL", r"^sotl$"),
    ("MaxPressure", r"^maxpressure$"),
    ("IntelliLight", r"^intellilight\[(?P<m>[^\]]+)\]$"),
    ("PressLight", r"^presslight\[(?P<m>[^\]]+)\]$"),
    ("AttendLight", r"^attendlight\[(?P<m>[^\]]+)\]$"),
    ("UniTSA", r"^unitsa\[(?P<m>[^\]]+)\]$"),
    ("Jev (distilled)", rf"^jev-ns\[[^\]]*\+{_S}\]$"),
    ("Jev (distilled) + event plan", rf"^jev-ns-triggered-jev\[[^\]]*\+{_S}\]$"),
    ("Jev (distilled + RL)", rf"^jev-ns\[[^\]]*\+{_S}\+{_R}(?:\+(?P<m>{_R}-fs\d+_\w+))?\]$"),
    ("Jev (distilled + RL) + event plan",
     rf"^jev-ns-triggered-jev\[[^\]]*\+{_S}\+{_R}(?:\+(?P<m>{_R}-fs\d+_\w+))?\]$"),
]
LEARNED = {"IntelliLight", "PressLight", "AttendLight", "UniTSA", "Jev (distilled + RL)",
           "Jev (distilled + RL) + event plan"}


def group(r) -> str:
    if r["junction"] in HELDOUT_JUNCTIONS:
        return "junction"
    return "pattern" if r["env_name"].endswith(HELDOUT_PATTERN) else "seen"


def ev_kind(e: str) -> str:
    return "none" if e == "none" else ("event_1" if e == "event_1" else "random")


def budget_of(model: str | None, junction: str) -> int | None:
    """Budget of a model on this junction: 0 = zero-shot (no model name, or "universal"), N = few-shot model of
    this junction ("fs<N>_<junction>", for Jev "<RL>-fs<N>_<junction>"); None = not part of the tables
    (models of other junctions, UniTSA's scratch_/ft_ runs, archived v4_fs runs)."""
    if model is None or model == "universal":
        return 0
    m = re.fullmatch(rf"(?:{_R}-)?fs(\d+)_(.+)", model)
    if m:
        return int(m.group(1)) if m.group(2) == junction else None
    return None


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return "-" if x is None else f"{x:.1f}"


def main():
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(RUNS, "*.json"))]
    zero = defaultdict(list)  # (row, group, event) -> ATT
    few = defaultdict(list)  # (row, budget, junction, event) -> ATT on held-out junctions
    for r in runs:
        att = r["metrics"].get("ATT")
        if not att:
            continue
        for name, pat in ROWS:
            m = re.match(pat, r["agent"])
            if not m:
                continue
            model = m.groupdict().get("m")
            b = budget_of(model, r["junction"])
            if b == 0 or name not in LEARNED:
                zero[(name, group(r), ev_kind(r["events"]))].append(att)
            if b is not None and r["junction"] in HELDOUT_JUNCTIONS:
                few[(name, b if name in LEARNED else 0, r["junction"], ev_kind(r["events"]))].append(att)

    gate_file = os.path.join(ROOT, "results", "sumo", "fewshot_gate.json")
    gate = json.load(open(gate_file))["gate"] if os.path.exists(gate_file) else {}

    def fs_runs(name, b, e, gated=True):
        """Held-out runs of (method, budget, event kind); gated: a junction whose adapted model lost to the
        zero-shot one on validation (scripts/fewshot_gate.py) deploys the zero-shot model."""
        out = []
        for j in sorted(HELDOUT_JUNCTIONS):
            g = gate.get(f"{name}|{b}|{j}") if gated and b else None
            out += few[(name, 0 if g and g["use"] == "zero" else b, j, e)]
        return out

    cols = [(e, g) for e in EVENTS for g in GROUPS]
    out = ["# Main tables", "", "ATT (s, lower is better), mean over runs; scenario groups and budgets as in "
           "scripts/main_tables.py.", "", "## Table 1: zero-shot", "",
           "| method | " + " | ".join(f"{e}: {g}" for e, g in cols) + " |", "|---|" + "---|" * len(cols)]
    for name, _ in ROWS:
        cells = [mean(zero[(name, g, e)]) for e, g in cols]
        if any(c is not None for c in cells):
            best = [min((mean(zero[(n, g, e)]) for n, _ in ROWS if zero[(n, g, e)]), default=None) for e, g in cols]
            out.append(f"| {name} | " + " | ".join(("**" + fmt(c) + "**") if c is not None and c == b else fmt(c)
                                                   for c, b in zip(cells, best)) + " |")
    out += ["", "## Table 2: few-shot on the held-out junctions (budget 10 episodes"
            + (", validation-gated" if gate else "") + ")", "",
            "| method | " + " | ".join(EVENTS) + " | runs |", "|---|" + "---|" * (len(EVENTS) + 1)]
    for name, _ in ROWS:
        b = 10 if name in LEARNED else 0
        cells = [mean(fs_runs(name, b, e)) for e in EVENTS]
        n = sum(len(fs_runs(name, b, e)) for e in EVENTS)
        if n:
            out.append(f"| {name}{'' if name in LEARNED else ' (no adaptation)'} | "
                       + " | ".join(fmt(c) for c in cells) + f" | {n} |")
    curves = [(True, "validation-gated")] if gate else []
    for gated, title in curves + [(False, "raw adapted models")]:
        out += ["", f"### Budget curve (held-out junctions, all events, {title})", "",
                "| method | " + " | ".join(f"{b} eps" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
        for name, _ in ROWS:
            if name not in LEARNED:
                continue
            cells = [mean([x for e in EVENTS for x in fs_runs(name, b, e, gated)]) for b in BUDGETS]
            if any(c is not None for c in cells):
                out.append(f"| {name} | " + " | ".join(fmt(c) for c in cells) + " |")
    if gate:
        kept = defaultdict(lambda: [0, 0])
        for k, g in gate.items():
            name, b, _ = k.split("|")
            kept[(name, int(b))][0] += g["use"] == "adapted"
            kept[(name, int(b))][1] += 1
        out += ["", "Adapted models kept by the gate (of the held-out junctions): "
                + "; ".join(f"{n} {b} eps {k}/{t}" for (n, b), (k, t) in sorted(kept.items())
                            if "event plan" not in n)]
    path = os.path.join(ROOT, "results", "sumo", "main_tables.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
