"""Summarize the eight-method P6 blind OOD supplement at map level."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from ood_category_protocol import P6_METHODS
from summarize_seven_category_run import collect_valid_rows
from summarize_six_category_comparison import (
    bootstrap_maps, effect, finite, holm, map_summaries, mean,
    number, percent, summarize_maps, wilcoxon_signed_rank, write_csv, write_json,
)


def paired_statistics(rows, protocol, bootstrap=5000):
    """Seven comparisons x two metrics; seeds paired inside each map."""
    index = {(r["map_root"], r["scene"], r["seed"], r["method"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate P6 result identity")
    if len(protocol["map_roots"]) != 1:
        raise ValueError("P6 requires one frozen OOD bank")
    outputs = []
    for comparator in P6_METHODS[1:]:
        for metric in ("length", "cfr_0.6"):
            groups = defaultdict(list)
            total = excluded = 0
            for key, ours in sorted(index.items()):
                if key[3] != "full":
                    continue
                other = index[key[:3] + (comparator,)]
                total += 1
                if metric == "length" and (not ours["nominal_success"] or not other["nominal_success"]):
                    excluded += 1
                    continue
                groups[key[1]].append((float(ours[metric]), float(other[metric])))
            pairs = [(statistics.fmean(x for x, _ in values), statistics.fmean(y for _, y in values))
                     for _, values in sorted(groups.items())]
            differences = [a-b for a, b in pairs]
            test = wilcoxon_signed_rank(differences)
            pvalue = test.p_value if finite(test.p_value) else None
            seed = int.from_bytes(hashlib.sha256(f"P6/{comparator}/{metric}".encode()).digest()[:8], "big")
            low, high = bootstrap_maps(pairs, metric, bootstrap, seed)
            outputs.append({"reference": "full", "comparator": comparator, "metric": metric,
                "maps": len(pairs), "matched_runs": total, "excluded_failed_pairs": excluded,
                "effect_reference_minus_comparator": effect(pairs, metric) if pairs else None,
                "effect_unit": "relative percent path length" if metric == "length" else "percentage points CFR",
                "bootstrap_ci_low": low, "bootstrap_ci_high": high, "bootstrap_resamples": bootstrap,
                "p_value": pvalue, "p_for_correction": pvalue if pvalue is not None else 1.,
                "wilcoxon_method": test.method, "nonzero_map_differences": test.n,
                "map_mean_reference": mean(a for a, _ in pairs), "map_mean_comparator": mean(b for _, b in pairs)})
    holm(outputs, field="p_for_correction", output_field="p_holm")
    for row in outputs:
        row["correction_family_size"] = 14
        row["supports_reference_direction"] = bool(row["p_holm"] < .05 and finite(row["effect_reference_minus_comparator"]) and (
            row["effect_reference_minus_comparator"] < 0 if row["metric"] == "length" else row["effect_reference_minus_comparator"] > 0))
    return outputs


def summarize_run(run_dir, bootstrap=5000):
    run_dir = Path(run_dir)
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8-sig"))
    if protocol.get("profile") != "p6" or tuple(protocol["methods"]) != P6_METHODS or "reuse" in protocol:
        raise ValueError("P6 requires the fixed eight-method matrix without cross-map reuse")
    rows, receipts = collect_valid_rows(run_dir, protocol)
    maps = map_summaries(rows)
    summaries = summarize_maps(maps)
    comparisons = paired_statistics(rows, protocol, bootstrap)
    output = run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("run_rows", rows), ("map_summary", maps), ("summary", summaries), ("paired_statistics", comparisons)):
        write_csv(output / (name + ".csv"), records)
    lines = ["# P6：固定主方法的分布偏移盲测", "",
        "证据范围：" + protocol["evidence_scope"], "",
        "这是独立的外部有效性补充：主方法、五个条件风险加精修的匹配强对照，以及裸 DE、裸 L-SHADE。此处不重复 E0 或组件消融；所有地图上的规划均新计算，没有旧图结果迁移。主方法在地图生成前冻结，不按 P6 结果换方法或挑参数。", "",
        "| 方法 | 规划次数 | 平均路径↓ | 名义成功率↑ | CFR .3↑ | CFR .6↑ | CFR .9↑ | .6经验达标率↑ | 平均时间/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {row["method"]: row for row in summaries if row["scope"] == "combined"}
    for method in P6_METHODS:
        r = summary[method]
        lines.append(f"| {protocol['methods'][method]['label']} | {r['runs']} | {number(r['length_feasible'])} | {percent(r['nominal_success'])} | {percent(r['cfr_0.3'])} | {percent(r['cfr_0.6'])} | {percent(r['cfr_0.9'])} | {percent(r['pass_0.6'])} | {number(r['runtime_sec'])} |")
    lines += ["", "路径以场景坐标单位报告，长度按图内名义成功路径均值再图间等权；全失败地图在 CSV 单独记录。名义失败保留 CFR 与达标率分母、记 0。长度配对检验只保留双方成功的同图同种子，排除数不隐去。时间仅作运行描述。", "",
        "| 对手 | 指标 | 配对地图 | 差异 | 地图bootstrap95%区间 | 原始p | Holm p |",
        "|---|---|---:|---:|---|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['comparator']} | {r['metric']} | {r['maps']} | {number(r['effect_reference_minus_comparator'])} | [{number(r['bootstrap_ci_low'])}, {number(r['bootstrap_ci_high'])}] | {number(r['p_value'], 6)} | {number(r['p_holm'], 6)} |")
    lines += ["", "七个固定对照×路径/CFR两指标共14项，统一 Holm 校正。地图为统计单位，规划种子先在地图内配对平均；bootstrap 为逐项区间，不是同时置信区间。长度差异用相对百分比（负为好），CFR 用百分点（正为好）。不显著不等于等效，也不自动支持全面领先。", "",
        "P6 是人工定义密度、高度、拓扑偏移下的泛化实验，不代表真实传感器误差或真实飞行安全认证。", ""]
    (output / "P6分布偏移盲测.md").write_text("\n".join(lines), encoding="utf-8")
    receipt = {"status": "complete_validated_P6_matrix", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows), "maps": len(maps)//len(P6_METHODS), "methods": len(P6_METHODS),
        "holm_family_size": 14, "protocol_sha256": hashlib.sha256((run_dir / "protocol.json").read_bytes()).hexdigest(),
        "input_results": receipts, "report": str(output / "P6分布偏移盲测.md")}
    write_json(output / "analysis_receipt.json", receipt)
    return {key: value for key, value in receipt.items() if key != "input_results"}
