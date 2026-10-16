"""Summarize the frozen six-category comparison without treating seeds as maps.

Input is the runner's run_rows.json (one row per map, method, and planner seed).
This reporting script uses only the standard library. Main comparisons are
fixed below before running the experiment, not selected from their p-values.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uav_path_planning.statistical_reporting import wilcoxon_signed_rank


FAMILIES = ("de", "jade", "shade", "lshade", "pso")
NAMES = {"de": "DE", "jade": "JADE", "shade": "SHADE", "lshade": "L-SHADE", "pso": "PSO"}
METHODS = (
    "full", *("bare_" + x for x in FAMILIES),
    *("risk_" + x for x in FAMILIES),
    *("risk_polish_" + x for x in FAMILIES),
    "no_terminal_polish", "no_risk", "binary_risk",
)
LABELS = {
    "full": "完整方法（条件风险 + 末端切向精修）",
    "no_terminal_polish": "自己：去掉末端精修",
    "no_risk": "自己：去掉全部扰动风险处理",
    "binary_risk": "自己：条件估计换成普通二值采样",
    "bare_astar": "A* 3D（每图 1 条确定性路径）",
}
for family in FAMILIES:
    LABELS["bare_" + family] = NAMES[family]
    LABELS["risk_" + family] = NAMES[family] + " + 条件风险"
    LABELS["risk_polish_" + family] = NAMES[family] + " + 条件风险 + 同款精修"

CATEGORIES = (
    ("E1", "与普通基线比较", ["full", *("bare_" + x for x in FAMILIES), "bare_astar"]),
    ("E2", "与同样考虑风险的基线比较", ["full", *("risk_" + x for x in FAMILIES)]),
    ("E3", "与同样考虑风险、使用同款精修的基线比较", ["full", *("risk_polish_" + x for x in FAMILIES)]),
    ("E4", "去掉末端精修", ["full", "no_terminal_polish"]),
    ("E5", "去掉全部扰动风险处理", ["full", "no_risk"]),
    ("E6", "更换风险估计器", ["full", "binary_risk"]),
)
PRIMARY = (
    ("H1", "risk_polish_lshade", "length", "同样风险与精修预算下，完整方法路径更短"),
    ("H2", "bare_lshade", "cfr_0.6", "完整方法比普通 L-SHADE 的 CFR 更高"),
    ("H3", "no_terminal_polish", "length", "末端精修比全部预算用于全局搜索产生更短路径"),
)


def mean(values):
    clean = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.fmean(clean) if clean else None


def finite(value):
    return value is not None and math.isfinite(float(value))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def normalize(rows):
    seen = set()
    required = {"map_root", "scene", "method", "seed", "length", "nominal_success", "cfr_0.3", "cfr_0.6", "cfr_0.9", "pass_0.6", "evaluations", "risk_checks", "runtime_sec"}
    for row in rows:
        absent = required.difference(row)
        if absent:
            raise ValueError("Missing fields: " + ", ".join(sorted(absent)))
        key = int(row["map_root"]), str(row["scene"]), str(row["method"]), int(row["seed"])
        if key in seen:
            raise ValueError("Duplicate run key: " + repr(key))
        seen.add(key)
        for field in ("cfr_0.3", "cfr_0.6", "cfr_0.9", "pass_0.6"):
            if not 0 <= float(row[field]) <= 1:
                raise ValueError("Rates must be fractions in [0, 1]: " + field)
        if not bool(row["nominal_success"]):
            if any(float(row[k]) != 0.0 for k in ("cfr_0.3", "cfr_0.6", "cfr_0.9", "pass_0.6")):
                raise ValueError("Nominal failures must have unconditional CFR/pass equal to zero")
        elif not finite(row["length"]):
            raise ValueError("A nominal success must have finite path length")
    roots = sorted({int(r["map_root"]) for r in rows})
    for root in roots:
        by_method = defaultdict(set)
        for row in rows:
            if int(row["map_root"]) == root:
                by_method[row["method"]].add((row["scene"], int(row["seed"])))
        missing = set(METHODS).difference(by_method)
        if missing:
            raise ValueError(f"Incomplete bank {root}; missing methods: {sorted(missing)}")
        reference = by_method["full"]
        for method in METHODS:
            if by_method[method] != reference:
                raise ValueError(f"Planner-seed/map keys differ for {root}/{method}")
        for method in ("bare_astar",):
            if method in by_method:
                scenes = [scene for scene, _ in by_method[method]]
                if len(scenes) != len(set(scenes)):
                    raise ValueError("A* must be recorded once per map, not replicated across planner seeds")
    return roots


def map_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["map_root"]), row["scene"], row["method"]].append(row)
    output = []
    for (root, scene, method), group in sorted(grouped.items()):
        good = [r for r in group if r["nominal_success"]]
        item = {"map_root": root, "scene": scene, "method": method, "runs": len(group),
                "nominal_successes": len(good), "length_feasible": mean(r["length"] for r in good),
                "nominal_success": len(good) / len(group)}
        for metric in ("cfr_0.3", "cfr_0.6", "cfr_0.9", "pass_0.6", "evaluations", "risk_checks", "runtime_sec", "cpu_seconds"):
            item[metric] = mean(r.get(metric) for r in group)
        output.append(item)
    return output


def summarize_maps(map_rows):
    groups = defaultdict(list)
    for row in map_rows:
        groups[str(row["map_root"]), row["method"]].append(row)
        groups["combined", row["method"]].append(row)
    output = []
    for (scope, method), group in sorted(groups.items()):
        item = {"scope": scope, "method": method, "label": LABELS.get(method, method),
                "maps": len(group), "runs": sum(r["runs"] for r in group),
                "nominal_successes": sum(r["nominal_successes"] for r in group),
                "maps_without_nominal_success": sum(r["length_feasible"] is None for r in group)}
        for metric in ("length_feasible", "nominal_success", "cfr_0.3", "cfr_0.6", "cfr_0.9", "pass_0.6", "evaluations", "risk_checks", "runtime_sec", "cpu_seconds"):
            item[metric] = mean(r[metric] for r in group)
        output.append(item)
    return output


def quantile(values, q):
    data = sorted(values)
    index = q * (len(data) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    return data[lo] + (data[hi] - data[lo]) * (index - lo)


def effect(pairs, metric):
    ours = statistics.fmean(a for a, _ in pairs)
    theirs = statistics.fmean(b for _, b in pairs)
    if metric == "length":
        return 100.0 * (ours - theirs) / theirs if theirs > 0 else None
    return 100.0 * (ours - theirs)


def bootstrap_maps(pairs, metric, count, seed):
    if not pairs:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(count):
        draw = effect([pairs[rng.randrange(len(pairs))] for _ in pairs], metric)
        if draw is not None:
            draws.append(draw)
    return (quantile(draws, 0.025), quantile(draws, 0.975)) if draws else (None, None)


def holm(items, field="p_value", output_field="p_holm"):
    finite_items = [x for x in items if finite(x[field])]
    ordered = sorted(finite_items, key=lambda row: row[field])
    previous = 0.0
    for rank, item in enumerate(ordered):
        previous = min(1.0, max(previous, (len(ordered) - rank) * item[field]))
        item[output_field] = previous
    for item in items:
        item.setdefault(output_field, None)


def compare(rows, roots, bootstrap):
    # For length, match *both successful* runs before taking each map's mean.
    # This sensitivity removes unequal failure selection from the paired sample,
    # while the displayed main table retains all nominally successful paths.
    index = {(int(r["map_root"]), r["scene"], int(r["seed"]), r["method"]): r for r in rows}
    reference = [(key, row) for key, row in index.items() if key[3] == "full"]
    outputs = []
    primary_keys = {(baseline, metric): hypothesis for hypothesis, baseline, metric, _ in PRIMARY}
    for root in roots:
        for method in METHODS:
            if method == "full":
                continue
            for metric in ("length", "cfr_0.6"):
                pairs_by_map = defaultdict(list)
                total, excluded = 0, 0
                for key, proposed in reference:
                    if key[0] != root:
                        continue
                    baseline = index[key[:3] + (method,)]
                    total += 1
                    if metric == "length":
                        if not proposed["nominal_success"] or not baseline["nominal_success"]:
                            excluded += 1
                            continue
                    pairs_by_map[key[1]].append((float(proposed[metric]), float(baseline[metric])))
                pairs = [(statistics.fmean(x for x, _ in values), statistics.fmean(y for _, y in values))
                         for _, values in sorted(pairs_by_map.items())]
                differences = [x - y for x, y in pairs]
                test = wilcoxon_signed_rank(differences)
                # Stable seed remains independent of row order / Python hash salt.
                digest = hashlib.sha256(f"{root}/{method}/{metric}".encode()).digest()
                lo, hi = bootstrap_maps(pairs, metric, bootstrap, int.from_bytes(digest[:8], "big"))
                entry = {"map_root": root, "baseline": method, "baseline_label": LABELS[method],
                         "metric": metric, "hypothesis": primary_keys.get((method, metric), "exploratory"),
                         "maps": len(pairs), "matched_runs": total, "excluded_failed_pairs": excluded,
                         "effect_full_minus_baseline": effect(pairs, metric) if pairs else None,
                         "effect_unit": "% relative path length" if metric == "length" else "percentage points CFR",
                         "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                         "bootstrap_resamples": bootstrap, "p_value": test.p_value if finite(test.p_value) else None,
                         "wilcoxon_method": test.method, "nonzero_map_differences": test.n,
                         "maps_full_better": sum(d < -1e-12 if metric == "length" else d > 1e-12 for d in differences),
                         "map_mean_full": mean(x for x, _ in pairs), "map_mean_baseline": mean(y for _, y in pairs)}
                outputs.append(entry)
    primary = [r for r in outputs if r["hypothesis"] != "exploratory"]
    exploratory = [r for r in outputs if r["hypothesis"] == "exploratory"]
    holm(primary)
    holm(exploratory)
    for row in outputs:
        row["correction_family_size"] = len(primary if row["hypothesis"] != "exploratory" else exploratory)
        row["correction_family"] = "fixed_primary_across_banks" if row["hypothesis"] != "exploratory" else "remaining_comparisons_across_banks"
        row["supports_intended_direction"] = bool(
            finite(row["p_holm"]) and row["p_holm"] < 0.05 and finite(row["effect_full_minus_baseline"])
            and (row["effect_full_minus_baseline"] < 0 if row["metric"] == "length" else row["effect_full_minus_baseline"] > 0))
    return outputs


def number(value, digits=4):
    return f"{float(value):.{digits}f}" if finite(value) else "NA"


def percent(value):
    return f"{100.0 * float(value):.2f}%" if finite(value) else "NA"


def table(rows):
    lines = ["| 方法 | 平均路径 ↓ | 名义成功率 ↑ | CFR .3 ↑ | CFR .6 ↑ | CFR .9 ↑ | .6 达标率 ↑ | 耗时/CPU 秒 ↓ | NFE / 风险评估量 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join([
            row["label"], number(row["length_feasible"]), percent(row["nominal_success"]),
            percent(row["cfr_0.3"]), percent(row["cfr_0.6"]), percent(row["cfr_0.9"]),
            percent(row["pass_0.6"]), number(row["runtime_sec"], 3) + " / " + number(row["cpu_seconds"], 3),
            number(row["evaluations"], 0) + " / " + number(row["risk_checks"], 0),
        ]) + " |")
    return "\n".join(lines)


def primary_table(comparisons):
    lines = ["| 假设 | 图库 | 对手 / 指标 | 完整方法减对手 | 地图 bootstrap 95% 区间 | 原始 p | Holm p | 支持预定方向 |",
             "|---|---|---|---:|---|---:|---:|---|"]
    for row in sorted((r for r in comparisons if r["hypothesis"] != "exploratory"), key=lambda r: (r["hypothesis"], r["map_root"])):
        unit = "%" if row["metric"] == "length" else " pp"
        lines.append("| " + " | ".join([
            row["hypothesis"], str(row["map_root"]), row["baseline_label"] + " / " + row["metric"],
            number(row["effect_full_minus_baseline"], 3) + unit,
            "[" + number(row["bootstrap_ci_low"], 3) + ", " + number(row["bootstrap_ci_high"], 3) + "]" + unit,
            number(row["p_value"], 6), number(row["p_holm"], 6), "是" if row["supports_intended_direction"] else "否",
        ]) + " |")
    return "\n".join(lines)


def report(summary, comparisons, roots, input_path, bootstrap):
    lookup = {(row["scope"], row["method"]): row for row in summary}
    total_maps = lookup["combined", "full"]["maps"]
    primary_count = len([r for r in comparisons if r["hypothesis"] != "exploratory"])
    lines = ["# 六类实验：固定当前完整方法的公平比较", "",
             "本轮先测清当前方法，不在结果出来后换主方法、改 α 或挑地图。19 个连续优化配置来自 1 个完整方法、5 个普通基线、5 个风险适配基线、5 个风险加精修基线、3 个消融。六类表反复列出的是同一个完整方法，不是六个新算法。A* 只作额外几何参照。", "",
             f"## 先看输入中全部 {total_maps} 张研究地图的描述性总表", "",
             "两套图库此前都已使用过，因此这里是已知地图上的固定配置比较，不称未见盲测或分布外验证。合并表方便直观比较，正式主检验分别在每套图库按地图进行。", "",
             table([lookup["combined", m] for m in (*METHODS, "bare_astar") if ("combined", m) in lookup]), "",
             "## 六类比较分别回答什么", ""]
    for category, title, methods in CATEGORIES:
        lines.extend([f"### {category}：{title}", ""])
        if category == "E1":
            lines.append("普通基线只看到名义地图；这组回答完整系统与普通规划算法相比如何，不能单独证明搜索器更强。A* 每张地图只规划一次，不把同一路径复制成 20 个独立样本；其搜索预算含义与连续算法不同。")
        elif category == "E2":
            lines.append("所有适配基线使用同款条件风险信息、扰动模型、内部样本数和门限，完整方法额外包含末端精修。本组判断优势是否仅来自让自己知道风险。")
        elif category == "E3":
            lines.append("所有适配基线进一步使用同款末端切向精修及相同预算划分。本组更接近比较全局搜索底座，但具体接口、适应规则和缓存统计仍须以运行清单为准。")
        elif category == "E4":
            lines.append("去掉的仅是末端精修，腾出的评估预算归还给全局搜索；全局搜索内部原有几何捷径保留。本组回答相同总预算怎样分配更有效，不能解读成“去掉了所有局部操作”。")
        elif category == "E5":
            lines.append("去掉全局及末端的扰动风险处理，保留名义碰撞检查与普通几何精修。它不是把风险门限随便调大。不同信息与开销属于消融本身，风险评估量应为零。")
        else:
            lines.append("只把条件风险估计换成普通二值采样，主搜索、门限、样本数和末端预算保持一致。K=16 时二值估计分辨率为 1/16；差别同时包括量化、方差和精修梯度可用性，不能全部归因于降方差。")
        lines.append("")
        for root in roots:
            lines.extend([f"**研究图库 {root}**", "", table([lookup[str(root), m] for m in methods if (str(root), m) in lookup]), ""])
    lines.extend(["## 提前指定的三项主问题", "",
                  f"完整设计为每个问题在两套图库分别检验，合计 6 项构成同一个 Holm 家族。当前输入包含 {primary_count} 项；只有两套图库、全部样本齐全的完整输出才用于正式比较。这里的“提前指定”仅指本轮执行前冻结，不能称为公开预注册或完全独立确认。", ""])
    for code, _, _, text in PRIMARY:
        lines.append(f"- {code}：{text}。")
    lines.extend(["", primary_table(comparisons), "",
                  "路径差值为 100 ×（完整方法均长 − 对手均长）/ 对手均长，负值表示路径缩短；CFR 差值为完整方法减对手，正值表示无碰撞率提升，单位为百分点（pp）。区间按地图成对重采样，不把 240 条运行当作 240 张独立地图。", "",
                  "两侧 Wilcoxon 使用每张地图的配对均值差；在每套图库 12 张图时枚举符号得到精确 p。Holm 校正控制已指定检验家族的多重比较。95% bootstrap 区间为单项描述区间，不是多重校正后的同时置信区间。p 不显著不能证明两方法相同或非劣。", "",
                  "其余每图库的完整方法对比见 paired_comparisons.csv；除主问题外，其余比较使用另一整体 Holm 家族并明确标为探索分析。未对合并两图库另行挑选“更好看”的主 p 值。", "",
                  "## 指标与公平性口径", "",
                  "- 平均路径：先在每张地图内对名义可行路径求均值，再对地图等权平均。坐标单位未做物理标定，不能标成米。某地图没有可行路径时记 NA，并单独报告缺失地图数，不能当作零长度。",
                  "- 名义成功率：算法输出路径在原始地图满足名义约束的比例；失败仍计入分母。",
                  "- CFR：路径固定后，在独立于优化器内部采样的外部障碍位移上检查无碰撞比例。这里没有在每张扰动地图上重新规划。名义失败的无条件 CFR 记 0。",
                  "- .6 达标率：直接汇总运行器记录的 pass_0.6，不在分析阶段重新选门限。本轮外部目标 α_target=0.175，即单条路径 CFR ≥ 82.5%；在正式运行的 500 次扰动下，至少 413 次无碰撞才达标。它不同于平均 CFR，也不是带置信保证的安全认证；α_target 也不同于优化器内部 α_search=0.10。",
                  "- 路径统计检验：仅配对同地图同种子且双方名义成功的路径，再算地图内均值，排除数量保存在 paired_comparisons.csv。这是成功条件下的长度比较，不能掩盖可行率差异。",
                  "- 时间：同时列出规划墙钟耗时与进程 CPU 时间；当前机器并行调度可能造成墙钟时间抖动，CPU 时间也不等同于目标飞行器的时延，只作描述。未声称严格等时预算，NFE 相同不代表 CPU 时间或风险评估量相同。",
                  "- 风险评估量：保留运行器实际计数；条件积分项与二值碰撞检查的单项代价不同，不能直接叫相同数量的精确碰撞检查。",
                  "- E1–E3 的连续基线均是本仓库实现/适配结果；不冒充原作者论文中的结果或官方排行榜数值。", "",
                  "## 结果完整性与可追溯文件", "",
                  f"- 输入：`{input_path.resolve()}`。",
                  f"- 地图 bootstrap 次数：{bootstrap}；固定种子由图库、方法与指标标识的 SHA-256 产生。",
                  "- `summary.csv` / `summary.json`：每图库及合并图库的地图等权结果。",
                  "- `map_summary.csv`：每张地图结果，可检查是否只在少数地图取胜。",
                  "- `paired_comparisons.csv` / `primary_tests.csv`：配对效果量、区间、p 值、校正家族与失败排除数量。",
                  "- `analysis_receipt.json`：输入哈希、分析脚本哈希、运行条数与统计规则。", ""])
    missing = [r for r in summary if r["scope"] != "combined" and r["maps_without_nominal_success"]]
    if missing:
        lines.extend(["**存在没有名义成功运行的地图：**", ""])
        for row in missing:
            lines.append(f"- {row['scope']} / {row['label']}：{row['maps_without_nominal_success']} 张。其平均路径的地图集合与其他方法不同，主表均长不能单独作为优越性证据。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    roots = normalize(rows)
    maps = map_summaries(rows)
    summary = summarize_maps(maps)
    comparisons = compare(rows, roots, args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "summary.csv", summary)
    write_json(args.output / "summary.json", summary)
    write_csv(args.output / "map_summary.csv", maps)
    write_csv(args.output / "paired_comparisons.csv", comparisons)
    write_csv(args.output / "primary_tests.csv", [r for r in comparisons if r["hypothesis"] != "exploratory"])
    (args.output / "六类比较结果.md").write_text(report(summary, comparisons, roots, args.input, args.bootstrap), encoding="utf-8")
    receipt = {"completed_at": datetime.now(timezone.utc).isoformat(), "input": str(args.input.resolve()),
               "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "runs": len(rows), "map_roots": roots, "methods": sorted({r["method"] for r in rows}),
               "statistical_unit": "map; seeds averaged within each map; nominally successful matched pairs for length",
               "primary_family": [dict(hypothesis=h, baseline=b, metric=m, meaning=t) for h, b, m, t in PRIMARY],
               "primary_family_tests": len([r for r in comparisons if r["hypothesis"] != "exploratory"]),
               "primary_test": "two-sided map-level Wilcoxon; Holm across both map banks",
               "bootstrap": args.bootstrap, "bootstrap_scope": "paired maps; unadjusted percentile 95% intervals",
               "claim_limit": "Known map banks, not blind/OOD. Frozen for this run, not public preregistration. Empirical pass, not safety certification."}
    write_json(args.output / "analysis_receipt.json", receipt)
    print(json.dumps({"output": str(args.output), "runs": len(rows), "summary_rows": len(summary), "contrasts": len(comparisons)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
