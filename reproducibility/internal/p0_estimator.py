"""P0: compare risk estimators on preselected frozen paths, without planning.

The conditional and binary estimates share x/y draws. Repeated banks are
independent; K values use nested prefixes. An independent, finite Monte Carlo
reference has its own reported uncertainty and is never called ground truth.
NumPy is an analysis dependency only; the optimizer remains standard-library.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from reproducibility.cohorts import equivalent
import time

import numpy as np

ROOT = CODE_ROOT
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from run_frozen_stress_suite import collision_free_batch
from uav_path_planning.conditional_risk_de import conditional_path_collision
from uav_path_planning.models import Path3D
from uav_path_planning.robustness import translate_environment, wilson_interval
from uav_path_planning.scenarios import Scenario3D

DEFAULT_SOURCE = CODE_ROOT / "data/reference/e"
OUTPUT_ROOT = CODE_ROOT / "outputs"
KS = (8, 16, 32, 64)
METHODS = ("full", "bare_lshade")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from reproducibility.atomic_io import write_json as dump_json


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot save an empty measurement table")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(base_seed, stream, root, scene, repeat=None):
    payload = json.dumps(["P0-fixed-path-v1", base_seed, stream, root, scene, repeat])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")


def select_paths(source, smoke=False):
    """Choose by IDs before any new risk measurements, never by outcome."""
    protocol_file = source / "protocol.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8-sig"))
    if protocol["main_method"]["module"] != "uav_path_planning.soft_risk_projected_polish":
        raise ValueError("Source does not contain the locked soft-risk method")
    if protocol["main_method"]["options"].get("beta") != .05:
        raise ValueError("Expected locked soft_beta050")
    seed = min(protocol["planner_seeds"])
    selected = []
    for root in protocol["map_roots"]:
        for index, scene in enumerate(protocol["scenarios"][str(root)][:1] if smoke else protocol["scenarios"][str(root)]):
            for method in METHODS:
                task_id = f"r{root}_m{index:02d}_{method}_s{seed:04d}"
                filename = source / "results" / (task_id + ".json")
                result = json.loads(filename.read_text(encoding="utf-8-sig"))
                row = result["row"]
                if (row["map_root"], row["scene"], row["method"], row["seed"]) != (
                        root, scene["name"], method, seed):
                    raise ValueError("Selected result has mismatched identity")
                points = result["detail"]["waypoints"]
                if points is None or len(points) < 2:
                    raise ValueError(f"Missing path; do not silently drop {task_id}")
                if not row["nominal_success"]:
                    raise ValueError(f"Selected path is nominally infeasible: {task_id}")
                selected.append({"id": task_id, "map_root": root,
                    "scene": scene["name"], "method": method, "planner_seed": seed,
                    "source_file": str(filename), "source_sha256": digest(filename),
                    "waypoints": points, "nominal_success": row["nominal_success"],
                    "original_internal_risk": row["internal_risk"], "scenario": scene})
    if len(selected) != (4 if smoke else 48) or seed != 9:
        raise ValueError("This diagnostic requires 24 maps x two paths and planner seed 9")
    return protocol, selected


def measure_timing(path, scene, shifts, sigma, margin, repeats=8):
    """Time equal K calls with preprocessing outside every timed region.

    Production scalar binary uses already translated environments, like the
    planner's cached bank. The vectorized binary evaluator is also reported as
    a distinct implementation; its speed cannot be attributed to an estimator.
    Alternating call order reduces systematic first/last effects.
    """
    results = defaultdict(list)
    for repeat in range(repeats):
        bank = shifts[repeat]
        python_bank = bank.tolist()
        environments = [translate_environment(scene.environment, sample)
                        for sample in python_bank]
        for k in KS:
            calls = {
                "conditional_stdlib": lambda: sum(conditional_path_collision(
                    path, scene.environment.obstacles, sample, sigma, axis=2,
                    margin=margin) for sample in python_bank[:k]) / k,
                "binary_stdlib": lambda: sum(path.collision_count(
                    environment, safety_margin=margin) > 0
                    for environment in environments[:k]) / k,
                "binary_vectorized": lambda: float((~collision_free_batch(
                    path, scene, bank[:k], margin=margin)).mean()),
            }
            order = list(calls)
            if repeat % 2:
                order.reverse()
            values = {}
            for backend in order:
                started = time.perf_counter()
                value = calls[backend]()
                elapsed = time.perf_counter() - started
                results[(k, backend)].append(elapsed)
                values[backend] = value
            if values["binary_stdlib"] != values["binary_vectorized"]:
                raise AssertionError("Scalar and vectorized binary predicates disagree")
    return [{"K": k, "backend": backend, "timing_repeats": len(times),
             "median_estimate_seconds": statistics.median(times),
             "mean_estimate_seconds": statistics.mean(times)}
            for (k, backend), times in sorted(results.items())]


def aggregate(details, timing, bootstrap_seed):
    """Equal-map summaries; the independent repeats are not independent maps."""
    summaries = []
    keys = sorted({(r["map_root"], r["scene"]) for r in details})
    rng = np.random.default_rng(bootstrap_seed)
    banks = defaultdict(list)
    for index, (root, _) in enumerate(keys):
        banks[root].append(index)
    # Stratify by existing bank; resample the paired map once for all methods/K.
    sampled = np.concatenate([rng.choice(indices, (5000, len(indices)), replace=True)
                              for indices in banks.values()], axis=1)
    for origin in (*METHODS, "both_path_origins"):
        for k in KS:
            subset = [r for r in details if r["K"] == k and (
                origin == "both_path_origins" or r["method"] == origin)]
            lookup = defaultdict(list)
            for row in subset:
                lookup[(row["map_root"], row["scene"], row["estimator"])].append(row)
            variances = {}
            for estimator in ("conditional", "binary"):
                rows = [r for r in subset if r["estimator"] == estimator]
                variances[estimator] = np.array([np.mean([r["empirical_variance"]
                    for r in lookup[(root, scene, estimator)]]) for root, scene in keys])
                times = [r["median_estimate_seconds"] for r in timing
                         if r["K"] == k and r["backend"] == (
                             "conditional_stdlib" if estimator == "conditional" else "binary_stdlib")
                         and (origin == "both_path_origins" or r["method"] == origin)]
                summaries.append({"path_origin": origin, "K": k,
                    "estimator": estimator, "maps": len(keys), "paths": len(rows),
                    "mean_empirical_variance": float(variances[estimator].mean()),
                    "mean_MSE_vs_finite_MC_reference": statistics.mean(r["MSE_vs_MC_reference"] for r in rows),
                    "mean_signed_bias_vs_finite_MC_reference": statistics.mean(r["bias_vs_MC_reference"] for r in rows),
                    "mean_gate_disagreement_vs_reference": statistics.mean(r["gate_disagreement_vs_reference"] for r in rows),
                    "mean_scalar_estimate_seconds": statistics.mean(times)})
            binary_mean = variances["binary"].mean()
            ratios = variances["conditional"][sampled].mean(axis=1) / variances["binary"][sampled].mean(axis=1)
            low, high = np.quantile(ratios, [.025, .975])
            for row in summaries[-2:]:
                row.update(conditional_to_binary_variance_ratio=float(variances["conditional"].mean()/binary_mean),
                           ratio_descriptive_map_bootstrap95_low=float(low),
                           ratio_descriptive_map_bootstrap95_high=float(high))
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2026101007)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--reference-samples", type=int, default=32768)
    args = parser.parse_args(argv)
    if args.repeats < 8 or args.reference_samples < 4096:
        raise ValueError("At least eight repeats and 4096 MC reference draws are required")
    source = args.source.resolve()
    if args.smoke:
        args.repeats = 8
        args.reference_samples = 4096
    output = args.output or OUTPUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_P0固定路径风险估计")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol, selected = select_paths(source, smoke=args.smoke)
    # Guard against silently measuring another evaluator than the frozen run.
    source_hashes = {}
    for relative in ("src/uav_path_planning/conditional_risk_de.py", "src/uav_path_planning/models.py",
                     "src/uav_path_planning/scenarios.py", "src/uav_path_planning/robustness.py",
                     "scripts/run_frozen_stress_suite.py"):
        actual, frozen = ROOT / relative, CODE_ROOT / relative
        if digest(actual) != digest(frozen):
            raise ValueError(f"Evaluator source differs from frozen source: {relative}")
        source_hashes[relative] = digest(actual)
    source_hashes["scripts/measure_locked_estimator.py"] = digest(__file__)
    margin = protocol["benchmark_config"]["safety_margin"]
    manifest = {"experiment": "P0-fixed-path-estimator-v1", "status": "started", "smoke": args.smoke,
        "created_at": datetime.now().astimezone().isoformat(), "source_run": str(source),
        "source_protocol_sha256": digest(source / "protocol.json"), "code_sha256": source_hashes,
        "path_selection": "first declared planner seed (9), full and bare_lshade on every existing map; no outcome filtering",
        "new_optimizations": 0, "maps": len(selected)//2, "fixed_paths": len(selected), "K": list(KS),
        "repeat_banks": args.repeats, "reference_samples": args.reference_samples,
        "reference_chunk": 4096, "sigma": .6, "axis": 2, "safety_margin": margin,
        "search_gate": .10, "seed": args.seed, "python": platform.python_version(),
        "numpy": np.__version__, "noise": "independent Gaussian translations across obstacles and coordinates",
        "pairing": "same draws across path origins and estimators within map; nested K; independent repeat banks; independent MC reference stream",
        "timing": "separate equal-K forward evaluation; prebuilt sample banks/environments; eight alternating-order repeats per path; scalar and vectorized binary reported separately",
        "limitations": ["Fixed paths on previously exposed maps; no new generalization evidence.",
            "Finite MC reference is not exact truth; MC standard error and Wilson interval reported.",
            "The repeated banks measure estimator variability; they are not extra independent maps.",
            "Equal K is neither equal computation nor equal elapsed time.",
            "Post-selection diagnostic; no optimizer tuning, new parameter choice or formal safety certification."]}
    # Selection and design are committed to disk before any new evaluation.
    dump_json(output / "selected_paths.json", selected)
    dump_json(output / "manifest.json", manifest)
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["map_root"], row["scene"])].append(row)
    details, repeat_rows, timing_rows, reference_rows = [], [], [], []
    started = time.perf_counter()
    for map_index, ((root, name), plans) in enumerate(sorted(grouped.items()), 1):
        scene = Scenario3D.from_dict(plans[0]["scenario"])
        paths = {p["method"]: Path3D(p["waypoints"]) for p in plans}
        for path in paths.values():
            if path.collision_count(scene.environment, safety_margin=margin):
                raise ValueError("Selected path contradicts nominal feasibility")
        n_obstacles = len(scene.environment.obstacles)
        ref_rng = np.random.default_rng(stable_seed(args.seed, "independent-reference", root, name))
        counts = {method: 0 for method in METHODS}
        for first in range(0, args.reference_samples, 4096):
            n = min(4096, args.reference_samples-first)
            shifts = .6 * ref_rng.normal(size=(n, n_obstacles, 3))
            for method, path in paths.items():
                counts[method] += int((~collision_free_batch(path, scene, shifts, margin)).sum())
        draws = np.array([.6 * np.random.default_rng(stable_seed(
            args.seed, "independent-repeat", root, name, repeat)).normal(size=(max(KS), n_obstacles, 3))
            for repeat in range(args.repeats)])
        for plan in plans:
            method, path = plan["method"], paths[plan["method"]]
            identity = {"map_root": root, "scene": name, "method": method, "path_id": plan["id"]}
            probability = counts[method] / args.reference_samples
            reference_se = math.sqrt(probability*(1-probability)/args.reference_samples)
            ref_low, ref_high = wilson_interval(counts[method], args.reference_samples)
            reference_rows.append({**identity, "MC_collision_count": counts[method],
                "MC_samples": args.reference_samples, "MC_risk_reference": probability,
                "MC_standard_error": reference_se, "MC_Wilson95_low": ref_low,
                "MC_Wilson95_high": ref_high, "gate_ambiguous_within_MC_interval": ref_low <= .10 <= ref_high})
            estimates = defaultdict(list)
            for repeat, bank in enumerate(draws):
                binary = (~collision_free_batch(path, scene, bank, margin)).astype(float)
                conditional = np.array([conditional_path_collision(path, scene.environment.obstacles,
                    sample, .6, axis=2, margin=margin) for sample in bank.tolist()])
                if not np.all((conditional >= 0) & (conditional <= 1)):
                    raise AssertionError("Invalid conditional probability")
                for k in KS:
                    for estimator, values in (("binary", binary), ("conditional", conditional)):
                        estimate = float(values[:k].mean())
                        estimates[(k, estimator)].append(estimate)
                        repeat_rows.append({**identity, "repeat": repeat, "K": k,
                            "estimator": estimator, "estimated_risk": estimate,
                            "MC_reference": probability, "within_search_gate": estimate <= .10})
            for (k, estimator), values in sorted(estimates.items()):
                values = np.asarray(values)
                details.append({**identity, "K": k, "estimator": estimator,
                    "repeats": args.repeats, "MC_reference": probability, "MC_standard_error": reference_se,
                    "mean_estimated_risk": float(values.mean()), "empirical_variance": float(values.var(ddof=1)),
                    "bias_vs_MC_reference": float(values.mean()-probability),
                    "MSE_vs_MC_reference": float(np.mean((values-probability)**2)),
                    "gate_disagreement_vs_reference": float(np.mean((values <= .10) != (probability <= .10))),
                    "reference_gate_ambiguous": ref_low <= .10 <= ref_high})
            timing_rows.extend({**identity, **row} for row in measure_timing(path, scene, draws, .6, margin))
        print(f"P0 fixed maps {map_index}/{len(grouped)}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
    summaries = aggregate(details, timing_rows, args.seed + 1)
    save_csv(output / "fixed_path_reference.csv", reference_rows)
    save_csv(output / "repeat_estimates.csv", repeat_rows)
    save_csv(output / "fixed_path_metrics.csv", details)
    save_csv(output / "map_summary.csv", details)  # one selected path per origin per map
    save_csv(output / "timing.csv", timing_rows)
    save_csv(output / "summary.csv", summaries)
    detail_lookup = {(r["path_id"], r["K"], r["estimator"]): r for r in details}
    worse = [{"path_id": p["id"], "K": k,
              "conditional_variance": detail_lookup[(p["id"], k, "conditional")]["empirical_variance"],
              "binary_variance": detail_lookup[(p["id"], k, "binary")]["empirical_variance"]}
             for p in selected for k in KS if detail_lookup[(p["id"], k, "conditional")]["empirical_variance"] >
             detail_lookup[(p["id"], k, "binary")]["empirical_variance"]]
    dump_json(output / "finite_repeat_variance_exceptions.json", worse)
    if args.smoke:
        manifest.update(status="complete_smoke", scope="Four paths, eight repeats; no full-study conclusions")
        dump_json(output / "execution_manifest.json", manifest)
        return
    lookup = {(r["path_origin"], r["K"], r["estimator"]): r for r in summaries}
    lines = ["# P0：固定路径条件风险估计诊断", "",
        "24 张既有地图，主方法与裸 L-SHADE 各选第一规划种子 9 的路径，共 48 条。选取规则与路径 ID 在计算前落盘；没有挑结果，没有新增规划，也未修改主方法。",
        f"每条路径使用 {args.repeats} 套独立重复样本；K=8/16/32/64 使用嵌套前缀。普通二元估计与条件积分使用同一套 x/y 扰动。高斯 σ=.6，各障碍独立平移；条件积分固定 z 轴。",
        f"独立二元 Monte Carlo 参考每图 {args.reference_samples} 个扰动，两种路径共享；这只是有限样本参考，逐路径 SE 和 Wilson 区间在 fixed_path_reference.csv。MSE 与偏差均相对此参考，不能称为对精确真值的误差。", "",
        "| 路径来源 | K | 二元估计方差 | 条件估计方差 | 条件/二元方差比 | 条件 MSE / 二元 MSE | 条件 / 二元门限判断分歧率 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for origin in METHODS:
        for k in KS:
            cond, binary = [lookup[(origin, k, est)] for est in ("conditional", "binary")]
            lines.append(f"| {origin} | {k} | {binary['mean_empirical_variance']:.6g} | {cond['mean_empirical_variance']:.6g} | {cond['conditional_to_binary_variance_ratio']:.3f} | {cond['mean_MSE_vs_finite_MC_reference']:.6g} / {binary['mean_MSE_vs_finite_MC_reference']:.6g} | {cond['mean_gate_disagreement_vs_reference']:.2%} / {binary['mean_gate_disagreement_vs_reference']:.2%} |")
    lines += ["", "门限判断：风险估计是否 ≤0.10 与独立 MC 参考是否 ≤0.10 一致。参考区间跨门限的路径有不确定标签，已逐路径标注；此项不是安全认证。", "",
        f"有限重复下出现条件经验方差大于二元经验方差的路径×K 单元：{len(worse)}/192，全部保留在 finite_repeat_variance_exceptions.json。总体理论方差不增，不等于每次有限重复测量都必然不增。", "",
        "计时单独进行：同 K、预先准备好扰动/平移环境、交替顺序，记录条件标准库、生产二元标准库与 NumPy 批量二元三种后端。每条路径每 K 测量 8 次取中位数，再按图等权。它是估计器前向计算时间，不含随机采样、规划与外部验收；不同后端不能混为估计器的算法加速。", "",
        "| K | 条件标准库 ms | 二元标准库 ms | 二元批量 ms |",
        "|---:|---:|---:|---:|"]
    for k in KS:
        means = {backend: statistics.mean(r["median_estimate_seconds"] for r in timing_rows
                   if r["K"] == k and r["backend"] == backend)*1000
                 for backend in ("conditional_stdlib", "binary_stdlib", "binary_vectorized")}
        lines.append(f"| {k} | {means['conditional_stdlib']:.4f} | {means['binary_stdlib']:.4f} | {means['binary_vectorized']:.4f} |")
    lines += ["", "图层级描述区间在 summary.csv：在两个既有图库内各自重采样地图，配对保留路径来源与估计器；24 张地图是汇总单位，不能把 64 次估计重复或 48 条路径宣称为 3072 张独立地图。区间没有多重比较校正，不作显著优越声明。", "",
        "本实验回答条件积分是否降低固定路径风险估计的随机波动；不单独证明优化后路径更优、消除选择偏差或在新地图泛化。原全量试验里的 E6 回答的是另一层问题。", "",
        "新增优化次数：**0**。主方法保持 soft_beta050，不基于本诊断调整参数。", ""]
    (output / "P0结果.md").write_text("\n".join(lines), encoding="utf-8")
    manifest.update(status="complete", completed_at=datetime.now().astimezone().isoformat(),
                    elapsed_seconds=time.perf_counter()-started, variance_exception_cells=len(worse),
                    artifacts_sha256={p.name: digest(p) for p in output.iterdir() if p.name != "manifest.json"})
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "status": "complete", "new_optimizations": 0,
                      "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    _outcome = main()
    if _outcome not in (None, 0):
        raise SystemExit(_outcome)
    from reproducibility.cohorts import write as _atomic_write, sha as _artifact_sha
    _destination = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    _smoke = "--smoke" in sys.argv
    _artifacts = {p.relative_to(_destination).as_posix(): _artifact_sha(p)
                  for p in _destination.rglob("*") if p.is_file() and p.name != "stage_complete.json"}
    _atomic_write(_destination / "stage_complete.json", {
        "schema": "RiskAwareDE-diagnostic-stage-v270", "status": "complete_smoke" if _smoke else "complete",
        "smoke": _smoke, "runner_sha256": _artifact_sha(Path(__file__)), "artifacts_sha256": _artifacts,
        "scope": "Smoke only; reduced computation and no complete-study conclusions" if _smoke else "Complete declared diagnostic stage"})
