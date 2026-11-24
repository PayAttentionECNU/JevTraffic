"""Validate complete matched outputs and build analysis tables and publication figures."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "deliverables"
FORMAL = ROOT / "formal-input"
CONDITIONS = {"none": "No event", "accident": "Incident", "special": "Priority", "both": "Combined"}
DEMAND = {"low_density": "Low", "high_density": "High", "random_perturbation": "Perturbed"}
ABLATIONS = {
    "full": "Full JevTraffic",
    "no_event_response": "Without event response",
    "no_case": "Without case memory",
    "no_temporal_prompt": "Without short-term prompt",
    "rules_for_events": "Rule-based detector",
}
FIELDS = {
    "min_similarity": ("Similarity threshold", r"$\tau_{\mathrm{sim}}$"),
    "half_life_days": ("Recency half-life", r"$h$ (days)"),
    "min_days": ("Minimum day support", r"$d_{\min}$ (days)"),
}
METRICS = {
    "ordinary_att": ("ordinary", "ATT"),
    "ordinary_wait": ("ordinary", "AWT"),
    "priority_loss": ("special", "timeLoss"),
    "priority_wait": ("special", "AWT"),
}
BOOTSTRAPS = 20000


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    assert rows
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_row(row, fixture):
    for field in ("id", "junction", "network", "environment", "demand", "seed", "condition", "horizon",
                  "route_sha256", "event_sha256"):
        assert row[field] == fixture[field], (field, row["id"])
    assert fixture["horizon"] <= row["clock"] <= fixture["horizon"] + 15
    assert row["metrics"]["ordinary"]["scheduled"] == fixture["vehicles"]
    assert row["metrics"]["special"]["scheduled"] == len(fixture["events"][1])
    for group in row["metrics"].values():
        assert group["scheduled"] == group["completed"] + group["not_departed"] + group["not_arrived"]
        assert (group["timeLoss"] is not None) == bool(group["completed"])


def route_id(fixture):
    return fixture["junction"], fixture["network"], fixture["demand"], fixture["seed"]


def values_by_route(fixtures, values):
    blocks = defaultdict(list)
    for fixture, value in zip(fixtures, values, strict=True):
        if value is not None:
            blocks[route_id(fixture)].append(value)
    assert blocks
    return {key: mean(values) for key, values in blocks.items()}


def interval(blocks, *, stratified=True, coverage=.95):
    block_values = np.array(list(blocks.values()))
    center = float(block_values.mean())
    if stratified:
        strata = defaultdict(list)
        for key, value in blocks.items():
            strata[key[:3]].append(value)
        assert all(len(values) >= 2 for values in strata.values())
        terms = [(len(values), np.var(values, ddof=1) * len(values) / len(blocks) ** 2)
                 for values in strata.values()]
        variance = sum(term for _, term in terms)
        if variance:
            df = variance ** 2 / sum(term ** 2 / (size - 1) for size, term in terms)
            margin = float(t.ppf((1 + coverage) / 2, df) * np.sqrt(variance))
        else:
            margin = 0.0
        low, high = center - margin, center + margin
    else:
        rng = np.random.default_rng(241905)
        boot = block_values[rng.integers(0, len(blocks), (BOOTSTRAPS, len(blocks)))].mean(axis=1)
        tail = (1 - coverage) / 2
        low, high = np.quantile(boot, (tail, 1 - tail))
    return {"mean": center, "lo": float(low), "hi": float(high), "routes": len(blocks)}


def paired(fixtures, before, after, *, stratified=True, coverage=.95):
    aligned = [(base, new) for base, new in zip(before, after, strict=True) if base is not None and new is not None]
    assert aligned
    differences = [new - base if base is not None and new is not None else None
                   for base, new in zip(before, after, strict=True)]
    n = len(aligned)
    before_mean, after_mean = (float(mean(vals)) for vals in zip(*aligned))
    return {"before": before_mean, "after": after_mean, "paired_episodes": n,
            **interval(values_by_route(fixtures, differences), stratified=stratified, coverage=coverage)}


def measure(row, metric):
    group, key = METRICS[metric]
    return row["metrics"][group][key]


def records(policies, ids, metric):
    values = []
    for fixture_id in ids:
        scores = [measure(rows[fixture_id], metric) for rows in policies]
        values.append(mean(scores) if all(score is not None for score in scores) else None)
    return values


def completion(policies, ids, group):
    groups = [rows[i]["metrics"][group] for rows in policies for i in ids]
    done = sum(value["completed"] for value in groups)
    total = sum(value["scheduled"] for value in groups)
    return done, total


def verify_formal_arm(fixtures, stage, arm):
    folder = FORMAL / "evaluations" / stage / arm
    config = load(folder / "arm.json")
    rows = {}
    for file in sorted(folder.glob("*.json")):
        if file.stem != "arm":
            data = load(file)
            assert data["arm"] == config
            if config.get("engine") == "original":
                assert data["case_sha256"] == "70c6ecc3e4af526bdd0dd16f1575cecca4b5efdf1b107dcf3bf63dae46a20269"
            for row in data["rows"]:
                assert row["id"] not in rows
                rows[row["id"]] = row
    assert set(rows) == {fixture["id"] for fixture in fixtures}
    for fixture in fixtures:
        validate_row(rows[fixture["id"]], fixture)
    return config, rows


def formal_data():
    fixtures = load(FORMAL / "fixtures/formal/manifest.json")["rows"]
    stages = {name: {arm: verify_formal_arm(fixtures, name, arm)
                     for arm in ("unitsa", "phase_s70", "phase_s71", "phase_s72")}
              for name in ("formal_native", "formal_shared")}

    for arm in stages["formal_native"]:
        old, new = (stages[stage][arm][0] for stage in ("formal_native", "formal_shared"))
        assert new["shared_event_plan"] is True and new["detector"] == "none"
        assert old == {k: v for k, v in new.items() if k not in ("shared_event_plan", "detector")}

    rows = []
    for method, arms in (("UniTSA", ("unitsa",)), ("Phase PPO", ("phase_s70", "phase_s71", "phase_s72"))):
        off, on = ([stages[stage][arm][1] for arm in arms] for stage in ("formal_native", "formal_shared"))
        for condition in CONDITIONS:
            for demand in ("all", *DEMAND):
                subset = [f for f in fixtures if f["condition"] == condition and (demand == "all" or demand == f["demand"])]
                ids = [f["id"] for f in subset]
                metrics = ["ordinary_att", "ordinary_wait"]
                if condition in ("special", "both"):
                    metrics.extend(("priority_loss", "priority_wait"))
                for metric in metrics:
                    before, after = (records(policy, ids, metric) for policy in (off, on))
                    group, _ = METRICS[metric]
                    off_done, off_scheduled = completion(off, ids, group)
                    on_done, on_scheduled = completion(on, ids, group)
                    rows.append({"method": method, "condition": condition, "demand": demand, "metric": metric,
                                 **paired(subset, before, after),
                                 "before_p95": float(np.quantile([v for v in before if v is not None], .95)),
                                 "after_p95": float(np.quantile([v for v in after if v is not None], .95)),
                                 "off_completed": off_done, "off_scheduled": off_scheduled,
                                 "on_completed": on_done, "on_scheduled": on_scheduled})
    write_csv(OUTPUT / "exploratory_baselines.csv", rows)
    return rows


def formal_memory():
    fixtures = load(FORMAL / "fixtures/formal/manifest.json")["rows"]
    off, on = (verify_formal_arm(fixtures, "formal_native", arm) for arm in ("A_no_memory", "A_pr"))
    assert off[0]["memory"] == "off" and on[0]["memory"] == "pr"
    assert {**off[0], "memory": "pr"} == on[0]
    rows = []
    for condition in CONDITIONS:
        subset = select(fixtures, condition)
        ids = [f["id"] for f in subset]
        for metric in metric_set(condition):
            before, after = (records([entry[1]], ids, metric) for entry in (off, on))
            rows.append({"condition": condition, "metric": metric, **paired(subset, before, after)})
    write_csv(OUTPUT / "formal_memory_verification.csv", rows)
    return rows


def new_data():
    import hashlib

    plan = load(ROOT / "plan.json")
    complete = load(ROOT / "complete.json")
    assert complete["plan_sha256"] == hashlib.sha256((ROOT / "plan.json").read_bytes()).hexdigest()
    fixtures = plan["rows"]
    canonical = json.dumps(fixtures, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == plan["manifest_sha256"]
    assert set(plan["ablations"]) == set(ABLATIONS) - {"full"}
    mapping = {f["id"]: f for f in fixtures}
    outputs = defaultdict(dict)
    for jsonl in sorted((ROOT / "jobs").glob("*.jsonl")):
        expected = {(label, fixtures[i]["id"]) for label, i in load(jsonl.with_suffix(".json"))}
        lines = [json.loads(line) for line in jsonl.read_text().splitlines()]
        assert len(lines) == len(expected)
        for value in lines:
            label = value["arm"]
            row = value["result"]
            assert (label, row["id"]) in expected
            assert row["id"] not in outputs[label]
            validate_row(row, mapping[row["id"]])
            assert row["instrumentation"]["anchor"] == mapping[row["id"]]["anchor"]
            outputs[label][row["id"]] = row
    assert sum(map(len, outputs.values())) == complete["episodes"]
    indices = plan["sensitivity_indices"]
    sensitivity_ids = {fixtures[i]["id"] for i in indices}
    sweep_labels = {setting_name(plan, param, value)
                    for param, values in plan["sweeps"].items() for value in values}
    assert set(outputs) == set(ABLATIONS) | sweep_labels
    for label, rows in outputs.items():
        assert set(rows) == (set(mapping) if label in ABLATIONS else sensitivity_ids)
    return plan, outputs


def component_data(plan, outputs):
    import hashlib

    root = ROOT / "components"
    study = load(root / "plan.json")
    complete = load(root / "complete.json")
    assert study["parent_sha256"] == hashlib.sha256((ROOT / "plan.json").read_bytes()).hexdigest()
    assert study["rows"] == plan["rows"]
    fixtures = {row["id"]: row for row in plan["rows"]}
    variants = defaultdict(dict)
    for path in sorted((root / "jobs").glob("*.jsonl")):
        expected = {(label, plan["rows"][i]["id"]) for label, i in load(path.with_suffix(".json"))}
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(lines) == len(expected)
        for value in lines:
            label, row = value["arm"], value["result"]
            assert (label, row["id"]) in expected and row["id"] not in variants[label]
            validate_row(row, fixtures[row["id"]])
            inst = row["instrumentation"]
            assert inst["anchor"] == fixtures[row["id"]]["anchor"]
            if label == "always_assess":
                assert inst["situation_calls"] == row["control_steps"]
            if label == "no_preemption":
                assert inst["preemption_decisions"] == 0
            if label == "no_blockage_mask":
                assert inst["blocked_decisions"] == 0
            variants[label][row["id"]] = row
    assert sum(map(len, variants.values())) == complete["episodes"]
    assert set(variants) == set(study["selections"])
    rows = []
    for label, selected in study["selections"].items():
        chosen = [plan["rows"][i] for i in selected]
        assert set(variants[label]) == {f["id"] for f in chosen}
        combined = {"full": outputs["full"], label: variants[label]}
        conditions = sorted({f["condition"] for f in chosen}, key=tuple(CONDITIONS).index)
        for condition in ("all", *conditions):
            subset = select(chosen, condition)
            ids = [f["id"] for f in subset]
            stats = {}
            for arm in ("full", label):
                items = [combined[arm][i] for i in ids]
                stats[arm] = {
                    "control_steps": sum(r["control_steps"] for r in items),
                    "situation_calls": sum(r["instrumentation"]["situation_calls"] for r in items),
                    "preemption_decisions": sum(r["instrumentation"]["preemption_decisions"] for r in items),
                    "blocked_decisions": sum(r["instrumentation"]["blocked_decisions"] for r in items),
                }
            metrics = metric_set(condition)
            if condition == "all" and "special" in conditions:
                metrics += ["priority_loss", "priority_wait"]
            for metric in metrics:
                group, _ = METRICS[metric]
                before, after = (completion([combined[arm]], ids, group) for arm in ("full", label))
                rows.append({"component": label, "condition": condition, "metric": metric,
                             **deltas(subset, combined, "full", label, metric, stratified=label != "always_assess"),
                             **{f"full_{k}": v for k, v in stats["full"].items()},
                             **{f"changed_{k}": v for k, v in stats[label].items()},
                             "full_completed": before[0], "full_scheduled": before[1],
                             "changed_completed": after[0], "changed_scheduled": after[1]})
    write_csv(OUTPUT / "event_components.csv", rows)
    return rows


def select(fixtures, condition, demand="all"):
    return [f for f in fixtures if (condition == "all" or f["condition"] == condition
                                    or condition == "priority_pooled" and f["condition"] in ("special", "both"))
            and (demand == "all" or demand == f["demand"])]


def metric_set(condition):
    if condition == "priority_pooled":
        return ["priority_loss", "priority_wait"]
    base = ["ordinary_att", "ordinary_wait"]
    return base + ["priority_loss", "priority_wait"] if condition in ("special", "both") else base


def deltas(fixtures, outputs, base, label, metric, *, stratified=True, coverage=.95):
    ids = [f["id"] for f in fixtures]
    before, after = (records([outputs[name]], ids, metric) for name in (base, label))
    return paired(fixtures, before, after, stratified=stratified, coverage=coverage)


def ablation_data(fixtures, outputs):
    rows = []
    for label in ABLATIONS:
        if label == "full":
            continue
        for condition in ("all", "priority_pooled", *CONDITIONS):
            subset = select(fixtures, condition)
            ids = [f["id"] for f in subset]
            for metric in metric_set(condition):
                group, _ = METRICS[metric]
                before, after = (completion([outputs[name]], ids, group) for name in ("full", label))
                rows.append({"ablation": label, "condition": condition, "metric": metric,
                             **deltas(subset, outputs, "full", label, metric),
                             "full_completed": before[0], "full_scheduled": before[1],
                             "ablated_completed": after[0], "ablated_scheduled": after[1]})
    write_csv(OUTPUT / "ablation.csv", rows)
    primary = []
    for label, condition, metric in (
        ("no_event_response", "priority_pooled", "priority_loss"),
        ("no_case", "all", "ordinary_att"),
        ("no_temporal_prompt", "priority_pooled", "priority_loss"),
        ("rules_for_events", "priority_pooled", "priority_loss"),
    ):
        subset = select(fixtures, condition)
        primary.append({"ablation": label, "condition": condition, "metric": metric,
                        **deltas(subset, outputs, "full", label, metric, coverage=.9875)})
    write_csv(OUTPUT / "ablation_primary.csv", primary)

    mechanism = []
    for label in ABLATIONS:
        for condition in CONDITIONS:
            subset = select(fixtures, condition)
            rows_for_label = [outputs[label][f["id"]] for f in subset]
            inst = [row["instrumentation"] for row in rows_for_label]
            mechanism.append({
                "ablation": label, "condition": condition, "episodes": len(inst),
                "situation_calls": sum(x["situation_calls"] for x in inst),
                "v2x_triggers": sum(x["triggered"].get("v2x", 0) for x in inst),
                "blocked_decisions": sum(x["blocked_decisions"] for x in inst),
                "preemption_decisions": sum(x["preemption_decisions"] for x in inst),
                "episode_any_preemption": sum(x["preemption_decisions"] > 0 for x in inst),
                "episode_any_blockage_mask": sum(x["blocked_decisions"] > 0 for x in inst),
                "full_metrics_match_episodes": sum(outputs["full"][row["id"]]["metrics"] == row["metrics"]
                                                   for row in rows_for_label),
                "case_hits": sum(row["memory"]["hits"] for row in rows_for_label if row["memory"]),
                "case_queries": sum(row["memory"]["queries"] for row in rows_for_label if row["memory"]),
            })
    write_csv(OUTPUT / "ablation_mechanism.csv", mechanism)
    return rows, primary, mechanism


def time_data(fixtures, outputs):
    rows = []
    for condition in CONDITIONS:
        for demand in ("all", "low_density", "high_density"):
            subset = select(fixtures, condition, demand)
            ids = [f["id"] for f in subset]
            for metric in ("halting_before", "halting_after", "halting_growth"):
                arms = []
                for label in ("no_event_response", "full"):
                    inst = [outputs[label][i]["instrumentation"] for i in ids]
                    arms.append([x["halting_after"] - x["halting_before"] if metric == "halting_growth" else x[metric]
                                 for x in inst])
                rows.append({"condition": condition, "demand": demand, "metric": metric,
                             **paired(subset, *arms)})
    write_csv(OUTPUT / "event_windows.csv", rows)
    return rows


def setting_name(plan, param, value):
    return "full" if value == plan["defaults"][param] else f"{param}_{value:g}"


def sweep_data(plan, outputs):
    fixtures = [plan["rows"][i] for i in plan["sensitivity_indices"]]
    rows = []
    for param, values in plan["sweeps"].items():
        for value in values:
            label = setting_name(plan, param, value)
            for condition in CONDITIONS:
                subset = select(fixtures, condition)
                for metric in metric_set(condition) + ["memory_coverage"]:
                    if metric == "memory_coverage":
                        coverage = []
                        for name in ("full", label):
                            mem = [outputs[name][f["id"]]["memory"] for f in subset]
                            coverage.append([100 * m["hits"] / m["queries"] for m in mem])
                        before, after = coverage
                    else:
                        ids = [f["id"] for f in subset]
                        before, after = (records([outputs[name]], ids, metric) for name in ("full", label))
                    assert all(v is not None for v in after)
                    absolute = interval(values_by_route(subset, after), stratified=False)
                    diff = paired(subset, before, after, stratified=False)
                    rows.append({"parameter": param, "value": value, "default": value == plan["defaults"][param],
                                 "condition": condition, "metric": metric, "mean": absolute["mean"],
                                 "lo": absolute["lo"], "hi": absolute["hi"], "routes": absolute["routes"],
                                 "delta_from_default": diff["mean"], "delta_lo": diff["lo"],
                                 "delta_hi": diff["hi"], "paired_episodes": diff["paired_episodes"]})
    write_csv(OUTPUT / "sensitivity.csv", rows)
    return rows


def graphics(baselines, ablation=None, sensitivity=None, windows=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    colors = ("#226b9e", "#bf640d", "#267044", "#a73946")
    markers = ("o", "s", "^", "x")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})

    def export(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(OUTPUT / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), layout="constrained")
    for i, condition in enumerate(("special", "both")):
        axis = axes[i]
        subset = [x for x in baselines if x["demand"] == "all" and x["condition"] == condition
                  and x["metric"] == "priority_loss"]
        for j, row in enumerate(subset):
            axis.plot((0, 1), (row["before"], row["after"]), marker=markers[j], color=colors[j],
                      label=row["method"], lw=1.7)
            offset = (9, -11)[j]
            axis.annotate(f'{row["before"]:.1f}', (0, row["before"]), (-8, offset),
                          textcoords="offset points", ha="right", color=colors[j])
            axis.annotate(f'{row["after"]:.1f}', (1, row["after"]), (8, offset),
                          textcoords="offset points", ha="left", color=colors[j])
        axis.set_xticks((0, 1), ("Baseline", "+ Shared event response"))
        axis.set_xlim(-.23, 1.23)
        axis.set_ylim(0, 33)
        axis.set_title(CONDITIONS[condition] + " scenarios")
        axis.grid(axis="y", color="#d7dbe0", alpha=.8)
    axes[0].set_ylabel("Priority vehicle time loss (s)")
    axes[0].legend(loc="lower right", frameon=False)
    export(fig, "exploratory_priority")
    if ablation is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.1, 3.8), layout="constrained")
    for index, (metric, condition, title) in enumerate((
        ("ordinary_att", "all", "Ordinary travel time"),
        ("priority_loss", "priority_pooled", "Priority vehicle time loss"),
    )):
        axis = axes[index]
        subset = [x for x in ablation if x["condition"] == condition and x["metric"] == metric]
        assert len(subset) == len(ABLATIONS) - 1
        for i, row in enumerate(subset):
            axis.errorbar(row["mean"], i, xerr=[[max(0, row["mean"] - row["lo"])],
                                               [max(0, row["hi"] - row["mean"])]],
                          fmt="o", color=colors[i], capsize=3, linewidth=1.6)
        axis.axvline(0, color="#5f656a", linewidth=1, linestyle="--")
        axis.grid(axis="x", color="#d7dbe0", alpha=.8)
        axis.set_title(title)
        axis.set_xlabel("Ablated − full (s)")
        axis.set_yticks(range(len(subset)), [ABLATIONS[row["ablation"]] if index == 0 else "" for row in subset])
        axis.invert_yaxis()
    export(fig, "ablation_effects")

    fig, axes = plt.subplots(2, 3, figsize=(13.1, 6.0), sharex="col", sharey="row", layout="constrained")
    for col, (param, (title, symbol)) in enumerate(FIELDS.items()):
        for row, (metric, shown) in enumerate((
            ("ordinary_att", tuple(CONDITIONS)),
            ("priority_loss", ("special", "both")),
        )):
            axis = axes[row, col]
            ticks = []
            for i, condition in enumerate(CONDITIONS):
                if condition not in shown:
                    continue
                curve = [point for point in sensitivity if point["metric"] == metric and
                         point["parameter"] == param and point["condition"] == condition]
                x, y = [v["value"] for v in curve], [v["mean"] for v in curve]
                ticks = x
                if i == tuple(CONDITIONS).index(shown[0]):
                    default = next(v["value"] for v in curve if v["default"])
                    axis.axvline(default, color="#757982", linestyle="--", linewidth=.9, alpha=.55)
                axis.plot(x, y, label=CONDITIONS[condition], color=colors[i], marker=markers[i],
                          linewidth=1.65, markersize=5)
            axis.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
            axis.grid(color="#d7dbe0", alpha=.75)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if row == 0:
                axis.set_title(f"({chr(97 + col)}) {title}", pad=10)
            else:
                axis.set_xlabel(symbol)
            if col == 0:
                axis.set_ylabel("Ordinary travel time (s)" if row == 0 else "Priority vehicle time loss (s)")
            if col == 2:
                axis.legend(loc="best", frameon=True, facecolor="white", framealpha=.91, fontsize=9)
    export(fig, "sensitivity")

    fig, axes = plt.subplots(1, 3, figsize=(13.1, 3.25), sharey=True, layout="constrained")
    for col, (param, (title, symbol)) in enumerate(FIELDS.items()):
        axis = axes[col]
        for i, condition in enumerate(CONDITIONS):
            curve = [v for v in sensitivity if v["metric"] == "memory_coverage" and
                     v["parameter"] == param and v["condition"] == condition]
            x, y = [v["value"] for v in curve], [v["mean"] for v in curve]
            if i == 0:
                axis.axvline(next(v["value"] for v in curve if v["default"]), color="#757982", linestyle="--",
                             linewidth=.9, alpha=.55)
                axis.set_xticks(x, [f"{tick:g}" for tick in x])
            axis.plot(x, y, label=CONDITIONS[condition], color=colors[i], marker=markers[i],
                      linewidth=1.55, markersize=5)
        axis.set_title(f"({chr(97 + col)}) {title}")
        axis.set_xlabel(symbol)
        axis.grid(color="#d7dbe0", alpha=.75)
        axis.set_ylim(-2, 103)
    axes[0].set_ylabel("Case retrieval hit rate (%)")
    axes[-1].legend(loc="best", frameon=True, fontsize=9)
    export(fig, "sensitivity_retrieval")

    fig, ax = plt.subplots(figsize=(9.7, 4), layout="constrained")
    for group, metric in enumerate(("halting_before", "halting_after")):
        subset = [x for x in windows if x["demand"] == "all" and x["metric"] == metric]
        pos = np.arange(len(subset), dtype=float) + (-.11 if group == 0 else .11)
        values = [row["mean"] for row in subset]
        ax.errorbar(pos, values, yerr=[[max(0, row["mean"] - row["lo"]) for row in subset],
                                      [max(0, row["hi"] - row["mean"]) for row in subset]],
                    fmt=markers[group], label=("60 s before", "60 s after")[group],
                    color=colors[group], linewidth=1.5, capsize=3)
    ax.axhline(0, color="#5f656a", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(CONDITIONS)), tuple(CONDITIONS.values()))
    ax.set_ylabel("Mean halted vehicles\nFull − without event response")
    ax.grid(axis="y", color="#d7dbe0", alpha=.8)
    ax.legend(frameon=False)
    export(fig, "exploratory_event_window")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    base = formal_data()
    if args.formal_only:
        graphics(base)
        for row in base:
            if row["demand"] == "all" and row["metric"] in ("ordinary_att", "priority_loss"):
                print(row["method"], row["condition"], row["metric"],
                      *[round(row[k], 3) for k in ("before", "after", "mean", "lo", "hi")])
        return
    memory_verification = formal_memory()
    plan, outputs = new_data()
    ablation, primary, mechanism = ablation_data(plan["rows"], outputs)
    windows = time_data(plan["rows"], outputs)
    sensitivity = sweep_data(plan, outputs)
    components = component_data(plan, outputs)
    graphics(base, ablation, sensitivity, windows)
    save(OUTPUT / "results.json", {"exploration_baselines": base, "ablation": ablation,
                                   "ablation_primary": primary, "ablation_mechanism": mechanism,
                                   "event_windows": windows, "sensitivity": sensitivity,
                                   "event_components": components, "formal_memory_verification": memory_verification})
    main_count = load(ROOT / "complete.json")["episodes"]
    component_count = load(ROOT / "components/complete.json")["episodes"]
    print(f"Validated {main_count + component_count} new episodes and wrote {OUTPUT}")


if __name__ == "__main__":
    main()
