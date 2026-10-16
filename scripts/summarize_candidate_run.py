"""Analyze the complete diverse-candidate matrix against frozen references."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from summarize_seven_category_run import summarize_run as summarize_base
from summarize_six_category_comparison import (
    bootstrap_maps, effect, finite, holm, wilcoxon_signed_rank, write_csv,
)


def _f(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", "NA", "None") else None


def _pct(value):
    return "NA" if value is None else f"{100*value:.3f}%"


def _num(value):
    return "NA" if value is None else f"{value:.6f}"


def _success(value):
    return value is True or value == 1 or value == "1" or value == "True"


def _combined(summary_rows, method):
    rows = [r for r in summary_rows if r.get("scope") == "combined" and r.get("method") == method]
    return rows[0] if rows else None


def _raw_mean(rows, method, metric):
    values = [_f(row, metric) for row in rows if row.get("method") == method]
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _candidate_statistics(rows, protocol, bootstrap):
    index = {}
    for row in rows:
        key = (int(row["map_root"]), row["scene"], int(row["seed"]), row["method"])
        if key in index:
            raise ValueError(f"duplicate candidate run key: {key}")
        index[key] = row
    candidates = [name for name, spec in protocol["methods"].items()
                  if spec.get("category") == "CANDIDATE"]
    references = ("full", "risk_polish_lshade")
    metrics = ["length", "cfr_0.6"] + [
        f"cfr_{model}_{protocol['distribution_external_sigma']}"
        for model in protocol.get("distribution_external_models", [])
    ]
    outputs = []
    for root in protocol["map_roots"]:
        for candidate in candidates:
            for reference in references:
                for metric in metrics:
                    by_map = {}
                    total = excluded = 0
                    for key, proposed in index.items():
                        if key[0] != int(root) or key[3] != candidate:
                            continue
                        base_key = key[:3] + (reference,)
                        if base_key not in index:
                            raise ValueError(f"missing paired reference result: {base_key}")
                        baseline = index[base_key]
                        total += 1
                        if metric == "length" and not (_success(proposed["nominal_success"])
                                                        and _success(baseline["nominal_success"])):
                            excluded += 1
                            continue
                        by_map.setdefault(key[1], []).append((float(proposed[metric]), float(baseline[metric])))
                    pairs = [(statistics.fmean(a for a, _ in group),
                              statistics.fmean(b for _, b in group))
                             for _, group in sorted(by_map.items())]
                    differences = [a - b for a, b in pairs]
                    test = wilcoxon_signed_rank(differences)
                    seed = int.from_bytes(hashlib.sha256(
                        f"{root}/{candidate}/{reference}/{metric}".encode()).digest()[:8], "big")
                    low, high = bootstrap_maps(pairs, metric, bootstrap, seed)
                    pvalue = test.p_value if finite(test.p_value) else None
                    outputs.append({
                        "map_root": int(root), "candidate": candidate, "reference": reference,
                        "metric": metric, "maps": len(pairs), "matched_runs": total,
                        "excluded_failed_pairs": excluded,
                        "candidate_minus_reference": effect(pairs, metric) if pairs else None,
                        "effect_unit": "relative percent path length" if metric == "length" else "percentage points CFR",
                        "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                        "bootstrap_resamples": bootstrap, "p_value": pvalue,
                        "p_for_correction": pvalue if pvalue is not None else 1.0,
                        "wilcoxon_method": test.method,
                        "nonzero_map_differences": test.n,
                    })
    holm(outputs, field="p_for_correction", output_field="p_holm")
    family_size = len(outputs)
    for row in outputs:
        row["correction_family"] = "all_candidates_x_two_references_x_two_metrics_x_map_banks"
        row["correction_family_size"] = family_size
        effect_value = row["candidate_minus_reference"]
        row["supports_candidate_direction"] = bool(
            row["p_value"] is not None and row["p_holm"] < .05 and finite(effect_value)
            and (effect_value < 0 if row["metric"] == "length" else effect_value > 0)
        )
    return outputs


def summarize_run(run_dir: Path, bootstrap=5000):
    run_dir = Path(run_dir)
    result = summarize_base(run_dir, bootstrap)
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8-sig"))
    with (run_dir / "analysis/run_rows.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (run_dir / "analysis/summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    methods = [name for name, spec in protocol["methods"].items()
               if spec.get("category") == "CANDIDATE"]
    full = _combined(summaries, "full")
    lshade = _combined(summaries, "risk_polish_lshade")
    lines = [
        "# 多机制候选全量对照结果", "",
        "本报告只在完整冻结矩阵完成后生成。`full` 是锁定的 soft_beta050 主方法；所有候选与它以及相同条件风险+精修的 L-SHADE 配对。两张已有地图库是已暴露开发/复核图，不是第三图库盲测。",
        "",
        "候选覆盖全局搜索、路径几何、扰动分布假设和末端编辑。内部 `internal_risk` 的含义随风险评分机制变化，不作为横向概率指标；跨算法风险比较以同一独立外部地图扰动样本得到的 CFR 为准。",
        "",
        f"完整结果：{result['rows']} 条规划记录，{len(protocol['methods'])} 个方法；本候选矩阵含 {len(methods)} 个机制候选。",
        "",
        "## 合并描述统计", "",
        "| 方法 | 平均路径↓ | 名义成功率↑ | CFR σ=.3↑ | 高斯 .6↑ | 高斯 .9↑ | 均匀 .6↑ | t3 .6↑ | 共模 .6↑ | 平均规划时间/s↓ | vs 主方法路径变化 | vs 主方法高斯 CFR .6 变化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("full", "risk_polish_lshade", *methods):
        current = _combined(summaries, method)
        if current is None:
            continue
        length = _f(current, "length_feasible")
        c06 = _f(current, "cfr_0.6")
        full_length = _f(full, "length_feasible") if full else None
        full_c06 = _f(full, "cfr_0.6") if full else None
        dlength = None if length is None or not full_length else 100 * (length - full_length) / full_length
        dcfr = None if c06 is None or full_c06 is None else 100 * (c06 - full_c06)
        shifted_cfr = {
            model: _raw_mean(rows, method, f"cfr_{model}_{protocol['distribution_external_sigma']}")
            for model in protocol.get("distribution_external_models", [])
        }
        label = protocol["methods"].get(method, {}).get("label", method)
        lines.append(
            f"| {label} | {_num(length)} | {_pct(_f(current, 'nominal_success'))} | "
            f"{_pct(_f(current, 'cfr_0.3'))} | {_pct(c06)} | {_pct(_f(current, 'cfr_0.9'))} | "
            f"{_pct(shifted_cfr.get('uniform'))} | {_pct(shifted_cfr.get('student_t3'))} | "
            f"{_pct(shifted_cfr.get('common_translation'))} | "
            f"{_num(_f(current, 'runtime_sec'))} | "
            f"{('—' if method == 'full' else _num(dlength) + '%')} | "
            f"{('—' if method == 'full' else _num(dcfr) + ' pp')} |"
        )
    comparisons = _candidate_statistics(rows, protocol, bootstrap)
    write_csv(run_dir / "analysis/candidate_paired_statistics.csv", comparisons)
    lines.extend([
        "", "## 地图层配对检验", "",
        "每张地图先对共同随机种子求均值，再以地图作为独立单位做 Wilcoxon 符号秩检验和地图层 bootstrap 区间。对所有候选 × 两个参照 × 路径长度、高斯 σ=.6 CFR、均匀/t3/共模 σ=.6 CFR × 两个图库统一做 Holm 校正。路径变化为候选减参照的相对百分比，负值代表更短；CFR 变化为百分点，正值代表更高。每种扰动分布的 500 个固定外部样本在方法间完全相同；这些属于已暴露开发图上的分布压力测试，不替代冻结后的盲测。完整逐项结果见 `candidate_paired_statistics.csv`。",
        "",
        "| 候选 | 参照 | 图库根种子 | 指标 | 效应 | 95% 地图 bootstrap 区间 | p | Holm p |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in comparisons:
        label = protocol["methods"][row["candidate"]]["label"]
        reference_label = protocol["methods"][row["reference"]]["label"]
        unit = "%" if row["metric"] == "length" else "pp"
        ci = ("NA" if row["bootstrap_ci_low"] is None else
              f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}] {unit}")
        effect_value = row["candidate_minus_reference"]
        lines.append(
            f"| {label} | {reference_label} | {row['map_root']} | {row['metric']} | "
            f"{_num(effect_value)} {unit} | {ci} | {_num(row['p_value'])} | {_num(row['p_holm'])} |"
        )
    lines.extend([
        "", "## 解读边界", "",
        "这些候选是跨机制的可证伪方案，不是已经成立的新算法贡献；只要某个候选胜出，也必须先在冻结后第三图库上做未见分布偏移盲测，不能用这两张已暴露图库挑选后就宣称泛化。混合分布候选按三种扰动子集的最大经验碰撞率评分，K=16 时每类只有 5–6 个样本，评分很粗，只能作为分布偏移压力候选；它的外部 CFR 仍按共同的高斯测试分布独立评分。",
        "",
    ])
    report = run_dir / "analysis/多机制候选全量对照.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    result["candidate_report"] = str(report)
    result["candidate_holm_family_size"] = len(comparisons)
    receipt_path = run_dir / "analysis/analysis_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        receipt["candidate_holm_family_size"] = len(comparisons)
        receipt["candidate_analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        receipt["candidate_report"] = report.name
        (run_dir / "analysis/analysis_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.run_dir, args.bootstrap), ensure_ascii=False, indent=2))
