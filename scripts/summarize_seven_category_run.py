"""Analyze a complete, validated E0--E6 run; never promote a partial matrix.

Statistics use maps, not planner seeds, as independent units. All 23 planned
non-deterministic contrasts, two metrics and both banks share one exploratory
Holm family (92 tests for the full protocol). A* stays descriptive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from summarize_six_category_comparison import (
    bootstrap_maps, effect, finite, holm, map_summaries, mean, number,
    percent, summarize_maps, wilcoxon_signed_rank, write_csv, write_json,
)


def planned_contrasts(protocol):
    """Read category semantics from the frozen protocol, including E0."""
    categories = protocol["categories"]
    if set(categories) != {f"E{i}" for i in range(7)}:
        raise ValueError("The report requires all seven categories E0--E6")
    output = []
    for category in sorted(categories):
        entry = categories[category]
        reference = entry["reference"]
        for comparator in entry["comparators"]:
            if protocol["methods"][comparator].get("deterministic", False):
                continue
            output.append((category, reference, comparator))
    if len(output) != 23:
        raise ValueError(f"Expected 23 planned contrasts, found {len(output)}")
    return output


def collect_valid_rows(run_dir, protocol):
    """Require exactly the planned files and validate each saved job bundle."""
    from seven_category_protocol import make_jobs, minimalvalidate_result
    jobs = make_jobs(protocol)
    directory = Path(run_dir) / "results"
    expected = {job["id"] + ".json" for job in jobs}
    actual = {p.name for p in directory.glob("*.json")}
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(
            "Only a complete result matrix can be summarized: "
            f"missing={len(missing)}, unexpected={len(extra)}; "
            f"examples={sorted(missing or extra)[:3]}"
        )
    rows, receipts = [], []
    for job in jobs:
        path = directory / (job["id"] + ".json")
        content = path.read_bytes()
        bundle = json.loads(content)
        minimalvalidate_result(job, protocol, bundle)
        rows.append(bundle["row"])
        receipts.append({"job_id": job["id"], "sha256": hashlib.sha256(content).hexdigest()})
    return rows, receipts


def paired_statistics(rows, protocol, bootstrap=5000):
    """Within-map pair seeds first; then compare independent map means."""
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    index = {}
    for row in rows:
        key = (int(row["map_root"]), row["scene"], int(row["seed"]), row["method"])
        if key in index:
            raise ValueError(f"Duplicate run key: {key}")
        index[key] = row
    outputs = []
    for root in protocol["map_roots"]:
        for category, reference, comparator in planned_contrasts(protocol):
            for metric in ("length", "cfr_0.6"):
                pairs_by_map = defaultdict(list)
                total = excluded = 0
                for key, proposed in sorted(index.items()):
                    if key[0] != int(root) or key[3] != reference:
                        continue
                    comparison_key = key[:3] + (comparator,)
                    if comparison_key not in index:
                        raise ValueError(f"Missing paired result: {comparison_key}")
                    baseline = index[comparison_key]
                    total += 1
                    if metric == "length" and (
                        not proposed["nominal_success"] or not baseline["nominal_success"]
                    ):
                        excluded += 1
                        continue
                    pairs_by_map[key[1]].append((float(proposed[metric]), float(baseline[metric])))
                pairs = [
                    (statistics.fmean(x for x, _ in group), statistics.fmean(y for _, y in group))
                    for _, group in sorted(pairs_by_map.items())
                ]
                differences = [x - y for x, y in pairs]
                test = wilcoxon_signed_rank(differences)
                digest = hashlib.sha256(f"{root}/{category}/{reference}/{comparator}/{metric}".encode()).digest()
                low, high = bootstrap_maps(pairs, metric, bootstrap, int.from_bytes(digest[:8], "big"))
                pvalue = test.p_value if finite(test.p_value) else None
                outputs.append({
                    "map_root": int(root), "category": category,
                    "reference": reference, "comparator": comparator, "metric": metric,
                    "maps": len(pairs), "matched_runs": total,
                    "excluded_failed_pairs": excluded,
                    "effect_reference_minus_comparator": effect(pairs, metric) if pairs else None,
                    "effect_unit": "relative percent path length" if metric == "length" else "percentage points CFR",
                    "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                    "bootstrap_resamples": bootstrap, "p_value": pvalue,
                    # Retain unavailable tests as p=1 within the planned family.
                    "p_for_correction": pvalue if pvalue is not None else 1.0,
                    "wilcoxon_method": test.method, "nonzero_map_differences": test.n,
                    "maps_reference_better": sum(d < -1e-12 if metric == "length" else d > 1e-12 for d in differences),
                    "map_mean_reference": mean(x for x, _ in pairs),
                    "map_mean_comparator": mean(y for _, y in pairs),
                })
    holm(outputs, field="p_for_correction", output_field="p_holm")
    for entry in outputs:
        entry["correction_family"] = "all_planned_exploratory_comparisons_across_banks"
        entry["correction_family_size"] = len(outputs)
        entry["supports_reference_direction"] = bool(
            entry["p_value"] is not None and entry["p_holm"] < .05
            and finite(entry["effect_reference_minus_comparator"])
            and (entry["effect_reference_minus_comparator"] < 0
                 if entry["metric"] == "length"
                 else entry["effect_reference_minus_comparator"] > 0)
        )
    return outputs


def _table(summary, methods, protocol):
    lines = [
        "| 方法 | 规划次数 | 平均路径↓ | 名义成功率↑ | CFR σ=.3↑ | CFR σ=.6↑ | CFR σ=.9↑ | σ=.6 经验达标率↑ | 平均评价数 | 风险计算项数 | 平均规划时间/s↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = summary[method]
        label = protocol["methods"][method]["label"]
        lines.append(
            f"| {label} | {row['runs']} | {number(row['length_feasible'])} | "
            f"{percent(row['nominal_success'])} | {percent(row['cfr_0.3'])} | "
            f"{percent(row['cfr_0.6'])} | {percent(row['cfr_0.9'])} | "
            f"{percent(row['pass_0.6'])} | {number(row['evaluations'], 1)} | "
            f"{number(row['risk_checks'], 1)} | {number(row['runtime_sec'])} |"
        )
    return lines


def render_report(protocol, rows, summaries, comparisons):
    summary_by_scope = defaultdict(dict)
    for row in summaries:
        summary_by_scope[row["scope"]][row["method"]] = row
    roots = protocol["map_roots"]
    is_smoke = protocol.get("profile", protocol.get("mode")) == "smoke" or any(
        len(protocol["scenarios"][str(root)]) < 12 for root in roots
    )
    is_ood = protocol.get("study_kind", "").startswith("ood_v2_domain_shift_")
    is_pilot = protocol.get("study_kind") == "ood_v2_domain_shift_pilot"
    title = (
        "七类实验：OOD小规模流程验证（不是正式实验结果）" if is_ood and is_smoke
        else "七类OOD先导实验描述结果（非盲测、非确认性）" if is_pilot
        else "七类OOD分布偏移对照结果" if is_ood
        else "七类实验：小规模流程验证（不是正式实验结果）" if is_smoke
        else "七类完整对照结果（当前已知双图库）"
    )
    lines = [f"# {title}", "", "本报告只在冻结协议要求的任务全部完成、保存结果通过校验后生成。", "",
        f"本次包含 {len(rows)} 次规划、{len(protocol['methods'])} 个不同配置；E0—E6 是七类比较，同一完整方法结果在各类中复用，不重复计作独立证据。", "",
        "主方法固定为现有全局搜索（保留内部捷径）＋固定 z 条件风险估计＋原有限差分末端切向精修。解析梯度候选未替换主方法。", "",
        ("本轮使用冻结的 v2-OOD 分布偏移图库；障碍密度、高度与拓扑改变，边界和起终点保持一致。地图生成只做几何有效性检查，不调用规划器或鲁棒性评分。配置冻结后不得根据 OOD 结果调参。各 DE 家族与 PSO 是本项目的本地复现/适配，不是原作者发布的标准 benchmark 数值。" if is_ood else
         "两图库都已经用于研究开发，本轮属于已知地图上的完整对照，不称为盲测或未见地图泛化证据。各 DE 家族与 PSO 是本项目的本地复现/适配，不是原作者发布的标准 benchmark 数值。"), "",
        "## 如何读指标", "",
        "路径长度使用场景坐标单位，不宣称米。表中长度先对每张图名义成功的路径求均值，再对地图等权平均；若一图全部失败，该图长度记 NA，并在 CSV 的 maps_without_nominal_success 中记录。名义成功率及 CFR 均保留所有规划尝试作分母，名义失败的 CFR 记 0。", "",
        "CFR 是固定最终路径在外部扰动下的无碰撞比例；σ 越大，扰动越强。σ=.6 的经验达标率是外部 CFR≥82.5% 的规划占比，不是安全证书。外部扰动仅评分，不回传搜索。", "",
        "随机优化器评价预算相同；条件估计和二值估计的单次工作量不同，风险计算项数不等同于运行时间。A* 每图只运行一次，使用自身的图搜索工作量，不纳入随机优化器配对检验。并行运行的耗时受调度影响，只作本轮运行描述，不能替代专门计时实验。", "",
        "## 七类比较：24 图合并描述", "",
    ]
    if is_smoke:
        lines[-2] = "## 七类比较：本次小规模地图合并描述"
    if protocol.get("profile") == "manuscript":
        lines[2:2] = [
            "本轮是锁定 soft_beta050 的对照补齐，不是全部方法重新计算。完整方法保持 β=.05；E2/E3 的风险选优也使用同一软风险权重；E4/E6 保留此权重，避免同时改动两项因素。",
            "3864 条原有规划记录按原文件复用，5760 条修正对照新计算；逐条来源见 reuse_receipt.json。跨批次运行时间仅作描述，不据此宣称效率优势。",
            "原生基线保留各自搜索和适应规则，门外排序采用已披露的标量适配；E3 比较全局搜索方案，不能称为只替换一个变异算子。", "",
        ]
    if is_ood:
        lines[2:2] = ["证据范围：" + protocol.get("evidence_scope", "OOD descriptive pilot"), ""]
    for category in sorted(protocol["categories"]):
        entry = protocol["categories"][category]
        methods = [entry["reference"], *entry["comparators"]]
        lines.extend([f"### {category}：{entry['label']}", ""])
        if category == "E0":
            lines.extend(["只检验名义全局搜索底座；保留内部捷径，风险评价与末端精修都关闭。", ""])
        elif category in ("E2", "E3"):
            lines.extend(["双方使用同一固定 z 条件风险评价；E3 进一步共享同款末端切向精修及预算分配。", ""])
        elif category == "E4":
            lines.extend(["仅去掉末端精修；内部捷径继续保留，省出的预算全部交给全局搜索。", ""])
        elif category == "E5":
            lines.extend(["关闭扰动风险处理，但保留名义长度末端精修；这与 E0 只有全局搜索的配置不同。", ""])
        elif category == "E6":
            lines.extend(["其余搜索与精修规则保持对应，仅将条件风险估计替换成普通 IID 二值碰撞采样。二值估计的离散台阶同时影响有限差分精修，因此它检验完整方法中的估计器替换效果；固定路径方差需另外测量。", ""])
        lines.extend(_table(summary_by_scope["combined"], methods, protocol))
        lines.append("")
    lines.extend([( "## 冻结 v2-OOD 图库列出全部配置" if is_ood else "## 两图库分别列出全部配置"), ""])
    for root in roots:
        lines.extend([f"### 图库 rootseed={root}", ""])
        lines.extend(_table(summary_by_scope[str(root)], list(protocol["methods"]), protocol))
        lines.append("")
    lines.extend(["## 地图层级配对统计", "",
        f"全部 {len(comparisons)} 项计划比较统一进行一次探索性 Holm 校正（23 个对照×2 指标×{len(roots)} 个图库）。这不是重新声明的独立预注册确认性检验。每个图库按地图独立配对，先在每图内部配对相同规划种子，绝不把 20 个种子当作 20 张独立地图。", "",
        "长度统计只配对双方均名义成功的同图同种子路径；CFR 统计包含全部规划，失败记 0。效应定义为“参照方法−对手”：长度用相对百分比（负数更好），CFR 用百分点（正数更好）。Bootstrap 区间在地图层级重采样，是逐项区间，不是经多重校正的同时置信区间。不显著不代表等价，某一指标显著也不代表全面优越。", "",
        "| 图库 | 类别 | 参照 | 对手 | 指标 | 配对地图数 | 差异 | 95% bootstrap 区间 | 原始 p | Holm p |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|",
    ])
    for row in comparisons:
        labels = protocol["methods"]
        lines.append(
            f"| {row['map_root']} | {row['category']} | {labels[row['reference']]['label']} | "
            f"{labels[row['comparator']]['label']} | {row['metric']} | {row['maps']} | "
            f"{number(row['effect_reference_minus_comparator'])} | "
            f"[{number(row['bootstrap_ci_low'])}, {number(row['bootstrap_ci_high'])}] | "
            f"{number(row['p_value'], 6)} | {number(row['p_holm'], 6)} |"
        )
    lines.extend(["", "原始逐次结果见 run_rows.json/csv；地图均值见 map_summary.csv；失败配对排除数、实际统计地图数、检验方式与完整效应见 paired_statistics.csv。正式报告不自动给出会议等级或替换主方法的决定。", ""])
    return "\n".join(lines)


def summarize_run(run_dir: Path, bootstrap=5000) -> dict:
    run_dir = Path(run_dir)
    protocol_path = run_dir / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    planned_contrasts(protocol)
    rows, inputs = collect_valid_rows(run_dir, protocol)
    maps = map_summaries(rows)
    summaries = summarize_maps(maps)
    for entry in summaries:
        entry["label"] = protocol["methods"][entry["method"]]["label"]
    comparisons = paired_statistics(rows, protocol, bootstrap)
    directory = run_dir / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "run_rows.json", rows)
    write_csv(directory / "run_rows.csv", rows)
    write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "map_summary.csv", maps)
    write_csv(directory / "paired_statistics.csv", comparisons)
    report = directory / "七类完整对照.md"
    report.write_text(render_report(protocol, rows, summaries, comparisons), encoding="utf-8")
    receipt = {
        "status": "complete_validated_matrix", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()), "rows": len(rows), "methods": len(protocol["methods"]),
        "map_count": sum(len(protocol["scenarios"][str(r)]) for r in protocol["map_roots"]),
        "bootstrap_resamples": bootstrap, "holm_family_size": len(comparisons),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": inputs,
    }
    write_json(directory / "analysis_receipt.json", receipt)
    return {"report": str(report), "rows": len(rows), "methods": len(protocol["methods"]),
            "holm_family_size": len(comparisons), "status": receipt["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.run_dir, args.bootstrap), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
