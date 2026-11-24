"""Write the Chinese experiment report from the validated numerical outputs."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "deliverables"
CONDITIONS = {"none": "无事件", "accident": "事故封道", "special": "特种车", "both": "复合事件"}
ABLATIONS = {
    "no_event_response": "移除事件执行",
    "no_case": "移除长期案例记忆",
    "no_temporal_prompt": "移除短期历史提示",
    "rules_for_events": "以规则替代 Jev 情景识别",
}
COMPONENTS = {
    "no_preemption": "仅关闭特种车抢占",
    "no_blockage_mask": "仅关闭封道车道屏蔽",
    "always_assess": "每步进行情景识别",
}
PARAMS = {"min_similarity": "相似度阈值", "half_life_days": "衰减半衰期（天）", "min_days": "最少历史天数"}


def find(table, **filters):
    rows = [row for row in table if all(row.get(k) == v for k, v in filters.items())]
    assert len(rows) == 1, (filters, len(rows))
    return rows[0]


def number(value, signed=False):
    if abs(value) < .005:
        value = 0.
    return f"{value:+.2f}" if signed and value != 0 else f"{value:.2f}"


def cf(row):
    return f'{number(row["mean"], True)} [{number(row["lo"], True)}, {number(row["hi"], True)}]'


def table(header, rows):
    def render(columns):
        return "| " + " | ".join(str(v) for v in columns) + " |"

    return "\n".join([render(header), render(["---"] * len(header)), *(render(row) for row in rows)])


def report(data, plan):
    baseline = data["exploration_baselines"]
    windows = data["event_windows"]
    ablation = data["ablation"]
    primary = data["ablation_primary"]
    memory_check = data["formal_memory_verification"]
    component = data["event_components"]
    sweep = data["sensitivity"]
    blocks = len({f["seed"] for f in plan["rows"]})
    assert blocks == len(plan["rows"]) / 4
    listed = [row for row in baseline if row["metric"] == "priority_loss" and row["demand"] == "all"]
    gains = [-row["mean"] for row in listed]
    percent = [-100 * row["mean"] / row["before"] for row in listed]
    ordinary = [row["mean"] for row in baseline if row["metric"] == "ordinary_att" and row["demand"] == "all"
                and row["condition"] in ("special", "both")]
    priority_rows = []
    for row in listed:
        low, high = [find(baseline, method=row["method"], condition=row["condition"], demand=demand,
                          metric="priority_loss") for demand in ("low_density", "high_density")]
        assert row["off_completed"] == row["off_scheduled"] and row["on_completed"] == row["on_scheduled"]
        priority_rows.append([row["method"], CONDITIONS[row["condition"]],
                              f'{row["before"]:.2f} → {row["after"]:.2f}',
                              cf(row), f'{low["mean"]:.2f} / {high["mean"]:.2f}',
                              f'{row["before_p95"]:.2f} → {row["after_p95"]:.2f}'])
    queue_rows = [[name,
                   *[cf(find(windows, condition=condition, demand="all", metric=metric))
                     for metric in ("halting_before", "halting_after", "halting_growth")]]
                  for condition, name in CONDITIONS.items()]

    intervals = []
    for name, display in ABLATIONS.items():
        ordinary_row = find(ablation, ablation=name, condition="all", metric="ordinary_att")
        priority_row = find(ablation, ablation=name, condition="priority_pooled", metric="priority_loss")
        main = find(primary, ablation=name)
        metric = {"ordinary_att": "普通 ATT", "priority_loss": "特种车 timeLoss"}[main["metric"]]
        intervals.append([display, cf(ordinary_row), cf(priority_row), f"{metric} {cf(main)}"])
    full_ordinary = find(ablation, ablation="no_case", condition="all", metric="ordinary_att")["before"]
    full_priority = find(ablation, ablation="no_event_response", condition="priority_pooled", metric="priority_loss")["before"]
    count = find(ablation, ablation="no_event_response", condition="priority_pooled", metric="priority_loss")
    arm_completed = []
    for name, label in ABLATIONS.items():
        att = find(ablation, ablation=name, condition="all", metric="ordinary_att")
        loss = find(ablation, ablation=name, condition="priority_pooled", metric="priority_loss")
        arm_completed.append([label, f'{att["ablated_completed"]}/{att["ablated_scheduled"]}',
                              f'{loss["ablated_completed"]}/{loss["ablated_scheduled"]}'])
    full_counts = find(ablation, ablation="no_case", condition="all", metric="ordinary_att")
    arm_completed.insert(0, ["完整 JevTraffic", f'{full_counts["full_completed"]}/{full_counts["full_scheduled"]}',
                             f'{count["full_completed"]}/{count["full_scheduled"]}'])
    memory_rows = []
    for condition, label in CONDITIONS.items():
        ordinary_check = find(memory_check, condition=condition, metric="ordinary_att")
        priority_check = find(memory_check, condition=condition, metric="priority_loss") if condition in ("special", "both") else None
        memory_rows.append([label, cf(ordinary_check), cf(priority_check) if priority_check else "—"])

    comp_rows = []
    for name, display in COMPONENTS.items():
        row = find(component, component=name, condition="all", metric="ordinary_att")
        has_priority = name in ("always_assess", "no_preemption")
        priority = find(component, component=name, condition="all", metric="priority_loss") if has_priority else None
        if name == "no_blockage_mask":
            priority = find(component, component=name, condition="both", metric="priority_loss")
        comp_rows.append([display, row["paired_episodes"], cf(row), cf(priority)])
    call = find(component, component="always_assess", condition="all", metric="ordinary_att")
    call_fraction = 100 * call["full_situation_calls"] / call["full_control_steps"]
    comp_scope = [find(component, component="no_blockage_mask", condition=c, metric="ordinary_att")
                  for c in ("accident", "both")]

    sensitivity_rows = []
    for param, label in PARAMS.items():
        values = plan["sweeps"][param]
        columns = []
        for metric in ("memory_coverage", "ordinary_att", "priority_loss"):
            series = [mean(p["mean"] for p in sweep if p["parameter"] == param and p["value"] == value
                           and p["metric"] == metric) for value in values]
            unit = "%" if metric == "memory_coverage" else " s"
            columns.append(f"{min(series):.2f}–{max(series):.2f}{unit}")
        values_text = ", ".join(f"**{value:g}**" if value == plan["defaults"][param] else f"{value:g}"
                                for value in values)
        sensitivity_rows.append([label, values_text, *columns])
    main_supported = [ABLATIONS[r["ablation"]] for r in primary if r["lo"] > 0]
    main_favored_variant = [ABLATIONS[r["ablation"]] for r in primary if r["hi"] < 0]
    mechanism = data["ablation_mechanism"]
    equivalent = []
    for variant in ("no_temporal_prompt", "rules_for_events"):
        compared = [row for row in mechanism if row["ablation"] == variant]
        if sum(row["full_metrics_match_episodes"] for row in compared) == len(plan["rows"]):
            equivalent.append(variant)
    main_uncertain = [ABLATIONS[r["ablation"]] for r in primary
                      if r["lo"] <= 0 <= r["hi"] and r["ablation"] not in equivalent]
    masked = sum(row["episode_any_blockage_mask"] for row in mechanism if row["ablation"] == "full")
    default_similarity = find(sweep, parameter="min_similarity", value=.75, metric="memory_coverage", condition="none")
    tight_similarity = find(sweep, parameter="min_similarity", value=.95, metric="memory_coverage", condition="none")
    tight_att = mean(row["delta_from_default"] for row in sweep if row["parameter"] == "min_similarity"
                     and row["value"] == .95 and row["metric"] == "ordinary_att")
    short_decay = find(sweep, parameter="half_life_days", value=2, metric="priority_loss", condition="both")
    default_decay = find(sweep, parameter="half_life_days", value=7, metric="priority_loss", condition="both")
    day_att = [mean(row["mean"] for row in sweep if row["parameter"] == "min_days"
                    and row["value"] == v and row["metric"] == "ordinary_att") for v in plan["sweeps"]["min_days"]]
    conclusions = []
    if main_supported:
        conclusions.append(f'校正后的主对照支持完整方案保留这些能力。对应移除项为{"、".join(main_supported)}。')
    if main_favored_variant:
        conclusions.append(f'校正后的主对照中，以下变体的该项指标更好。{"、".join(main_favored_variant)}。')
    if equivalent:
        names = "、".join(ABLATIONS[label] for label in equivalent)
        conclusions.append(f'{names}这两项对照的交通指标在全部 {len(plan["rows"])} 个场景中均与完整方案一致。')
    if main_uncertain:
        conclusions.append(f'{"、".join(main_uncertain)}的校正区间包含零，当前样本不足以确认其主要收益。')

    primary_table = table(
        ["消融组", "普通 ATT Δ (s)", "特种车 timeLoss Δ (s)", "主对照 Δ（98.75% CI）"], intervals)
    replay_completion = table(["组别", "普通车辆完成数", "特种车完成数"], arm_completed)
    memory_table = table(["场景", "普通 ATT Δ (s)", "特种车 timeLoss Δ (s)"], memory_rows)
    window_table = table(["场景", "前 60 秒", "后 60 秒", "前后变化之差"], queue_rows)
    baseline_table = table(
        ["基线", "事件", "timeLoss 未适配 → 适配 (s)", "Δ (s) [95% CI]",
         "低 / 高流量 Δ (s)", "P95 未适配 → 适配 (s)"], priority_rows)
    component_table = table(["对照", "场景数", "普通 ATT Δ (s)", "特种车 timeLoss Δ (s)"], comp_rows)
    sensitivity_table = table(["参数", "扫描值（粗体为默认）", "案例命中率", "普通 ATT", "特种车 timeLoss"],
                              sensitivity_rows)
    return f"""# JevTraffic 事件适配、模块消融与敏感性分析

## 1. 事件适配探索

对 UniTSA 和 Phase PPO 复用 864 个固定 SUMO 场景，覆盖 Massy、Songdo、YMT，两个路网版本、三种交通需求和四种事件条件。每个基线的网络权重、动作空间、车流随机种子和事件注入保持一致。适配组增加统一事件规则，使用 60 m V2X 接近检测、有限连续干预和排队停滞响应。Phase PPO 的三个训练权重先对同一场景求平均。每种事件有 216 个场景，每个流量子组有 72 个。

加入适配后，两种基线在仅特种车和复合事件中的特种车平均时间损失减少 **{min(gains):.2f}–{max(gains):.2f} 秒（{min(percent):.1f}%–{max(percent):.1f}%）**。两种基线在无事件条件下的普通车 ATT 完全一致。有特种车的场景中，普通车 ATT 的平均差为 {min(ordinary):.2f} 至 {max(ordinary):.2f} 秒，特种车在各组都完成行程。P95 按场景统计，Phase PPO 先对训练权重求平均。

{baseline_table}

![特种车优先对照](exploratory_priority.png)

下表用新场景比较完整 JevTraffic 和关闭事件执行的 JevTraffic。时间锚点是排定的特种车发车时刻。无事件与事故单独场景使用同车流下的对应锚点。统计前后 60 秒内各进口车道的停车车辆数，用相邻控制时刻间隔加权。差值为完整方案减去关闭事件执行。高低流量是仿真中的需求档位。复合事件按排程定义，车辆接近与物理封道的实际重合由逐场景数据单独记录。

{window_table}

![事件时间窗口](exploratory_event_window.png)

## 2. 模块消融

新回放包含 {blocks} 套车流种子和 **{len(plan["rows"])} 个场景**，覆盖三个留出路口、两个路网版本、两档流量和四种事件。各组使用同一份蒸馏模型，启用记忆的组使用同一份七天只读案例库。关闭事件执行同时关闭信号抢占和封闭车道屏蔽。移除短期历史提示只清空提供给情景分类器的历史行，触发器仍读取最近五步。规则对照保留触发器及事件管理器。

完整方案普通车 ATT 为 **{full_ordinary:.2f} 秒**，特种车 timeLoss 为 **{full_priority:.2f} 秒**。表中 Δ 为消融组减完整方案，正值表示通行代价上升。两列通行指标为逐项 95% 区间，最右列为四个预定主对照的 Bonferroni 校正结果。

{primary_table}

{"".join(conclusions)}

![各消融组的配对影响](ablation_effects.png)

{replay_completion}

### 案例记忆的独立种子复核

另用原始合并 PR 在 864 个固定场景上的回放，核对同一模型、有无案例记忆的差异。该批车流种子与新消融不重叠。这里的 Δ 为开启减关闭。四种条件下普通车 ATT 的 95% 区间均位于零下方。两种含特种车条件的 timeLoss 区间均跨过零。

{memory_table}

### 事件执行与触发效率

以下对照各跑 48 个配对场景。关闭抢占使用特种车与复合事件，关闭车道屏蔽使用事故与复合事件，每步识别覆盖全部四类。车道屏蔽对照的特种车指标来自复合事件；其余普通车指标各自按上面的场景范围汇总。

{component_table}

触发式方案在 {call["full_control_steps"]:,} 次控制决策中发起 {call["full_situation_calls"]:,} 次情景调用（**{call_fraction:.2f}%**），每步识别组为 {call["changed_situation_calls"]:,}/{call["changed_control_steps"]:,}。这一对照衡量情景查询频率及策略效果，墙钟耗时受并发负载影响。完整方案实际使用车道屏蔽的场景共有 {masked} 个。关闭屏蔽的普通车 ATT 差在事故、复合场景分别为 {cf(comp_scope[0])} 和 {cf(comp_scope[1])} 秒。

## 3. 超参数敏感性

选取上述预先锁定的第一组复现种子，形成 48 个场景，每个事件条件 12 个。每次只变一个参数，每个参数评估五个取值，共 13 套唯一配置。虚线标记当前默认值。下表区间是各档位的指标均值范围，普通 ATT 和案例命中率按四种条件等权汇总，特种车 timeLoss 按两种含特种车的条件等权汇总。曲线和 CSV 保留各场景的差异。

{sensitivity_table}

相似度阈值由 0.75 调到 0.95 后，无事件场景的命中率从 {default_similarity["mean"]:.2f}% 降到 {tight_similarity["mean"]:.2f}%，四类场景平均普通 ATT 增加 {tight_att:.2f} 秒。复合事件对衰减半衰期更敏感，2 天与默认 7 天的特种车 timeLoss 均值分别为 {short_decay["mean"]:.2f} 和 {default_decay["mean"]:.2f} 秒，配对区间为 [{short_decay["delta_lo"]:.2f}, {short_decay["delta_hi"]:.2f}] 秒，包含零。最少历史天数从 1 到 5 的各档普通 ATT 均值相差最多 {max(day_att) - min(day_att):.2f} 秒。

![三个超参数的通行指标敏感性](sensitivity.png)

![三个超参数的案例检索命中率](sensitivity_retrieval.png)

## 统计与复现

- ATT 为完成普通车在仿真中的平均行程时间。timeLoss 为 SUMO 记录的低于理想行驶速度造成的时间损失。指标先在每个场景内汇总，车辆完成率按排定车辆统计。
- 正式基线和主消融以配对车流种子为统计单位，按路口、路网、需求分层，使用层内方差和 Satterthwaite 自由度构建 t 区间。敏感性和每步情景识别每层各一组种子，CSV 给出按配对场景整体重抽样的描述性区间。敏感性属于单因素扫描，未估计参数之间的交互。
- 回放继承相同的 SUMO 时窗、种子核验与未完程统计。遇到跨拓扑的 SUMO 订阅残留后，计算按拓扑分进程隔离，保留可核验的原进程首拓扑结果，其余恢复重跑。
- `exploratory_baselines.csv`、`event_windows.csv`、`ablation.csv`、`ablation_primary.csv`、`ablation_mechanism.csv`、`formal_memory_verification.csv`、`event_components.csv`、`sensitivity.csv` 保留分组指标和配对差值。原始回放、场景清单及权重和案例库哈希另行打包。
"""


def main():
    data = json.loads((OUTPUT / "results.json").read_text())
    plan = json.loads((ROOT / "plan.json").read_text())
    target = OUTPUT / "report.md"
    target.write_text(report(data, plan), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
