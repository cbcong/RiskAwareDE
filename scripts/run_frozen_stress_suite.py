"""Evaluate frozen original-map paths across noise intensity and model shifts.

No replanning, candidate selection, or optimizer tuning occurs in this script.
All perturbations are synthetic sensitivity models, not real sensor models.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uav_path_planning import make_research_scenarios, split_research_scenarios
from uav_path_planning.models import Path3D
from uav_path_planning.robustness import _feasible, make_realizations, translate_environment, wilson_interval
from uav_path_planning.run_artifacts import recorded_run

BASELINES = ("de", "pso", "astar3d", "jade", "shade", "lshade", "adaptive_de", "uad_de")
VARIANTS = ("historical_main_276170", "partial070", "partial040_a0125")
MODELS = {
    "gaussian": {"sigmas": [0., .15, .3, .45, .6, .75, .9], "seed_offset": 0,
        "distribution": "independent standard Gaussian per obstacle and coordinate, scaled by sigma"},
    "uniform": {"sigmas": [.3, .6], "seed_offset": 1,
        "distribution": "independent Uniform(-sqrt(3), sqrt(3)) per obstacle and coordinate, scaled by sigma"},
    "student_t3": {"sigmas": [.3, .6], "seed_offset": 2,
        "distribution": "independent Student-t(df=3)/sqrt(3) per obstacle and coordinate, scaled by sigma; no clipping"},
    "common_translation": {"sigmas": [.3, .6], "seed_offset": 3,
        "distribution": "one isotropic Gaussian translation shared by every obstacle in a trial; coordinates independent; scaled by sigma"},
}


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(base, model, scene):
    raw = json.dumps(["frozen-stress-v1", base, model, scene]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def collision_free_batch(path, scenario, shifts, margin=0.0):
    """Exact closed AABB slab predicate over every trial, segment, and box.

    This is the same geometric predicate as ObstacleBox.intersects_segment;
    boundary contact is collision and zero-length directions are explicit.
    """
    if path is None:
        return np.zeros(len(shifts), dtype=bool)
    points = np.asarray(path.waypoints, dtype=float)
    bounds = scenario.environment.bounds
    if np.any(points < np.asarray(bounds.minimum)) or np.any(points > np.asarray(bounds.maximum)):
        return np.zeros(len(shifts), dtype=bool)
    if not scenario.environment.obstacles:
        return np.ones(len(shifts), dtype=bool)
    lo = np.asarray([o.minimum for o in scenario.environment.obstacles])[None, :, None, :] + shifts[:, :, None, :] - margin
    hi = np.asarray([o.maximum for o in scenario.environment.obstacles])[None, :, None, :] + shifts[:, :, None, :] + margin
    starts = points[:-1][None, None, :, :]
    direction = np.diff(points, axis=0)[None, None, :, :]
    parallel = direction == 0.0
    outside = np.any(parallel & ((starts < lo) | (starts > hi)), axis=-1)
    denominator = np.where(parallel, 1.0, direction)
    first, second = (lo - starts) / denominator, (hi - starts) / denominator
    near = np.where(parallel, -np.inf, np.minimum(first, second))
    far = np.where(parallel, np.inf, np.maximum(first, second))
    enter = np.maximum(0., np.max(near, axis=-1))
    leave = np.minimum(1., np.min(far, axis=-1))
    collision = (~outside) & (enter <= leave)
    return ~np.any(collision, axis=(1, 2))


def load_sources(historical, previous, scenarios):
    plans, sources = [], {}
    sources_spec = [(historical, None), (previous / "full" / "partial070", "partial070"),
                    (previous / "full" / "partial040_a0125", "partial040_a0125")]
    for directory, expected in sources_spec:
        protocol_path = directory / "robustness" / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
        if protocol["scenarios"] != [s.to_dict() for s in scenarios]:
            raise ValueError(f"Source geometry does not match the original bank: {directory}")
        if float(protocol.get("safety_margin", 0.)) != 0.:
            raise ValueError("This protocol requires historical zero safety margin")
        source = directory / "robustness" / "nominal_paths.json"
        for f in (source, protocol_path):
            sources[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
        for row in json.loads(source.read_text(encoding="utf-8-sig")):
            if expected is None and row["algorithm"] not in BASELINES + ("risk_aware_de",):
                continue
            if expected is not None and row["algorithm"] != expected:
                raise ValueError("Unexpected algorithm in candidate source")
            row = dict(row)
            if row["algorithm"] == "risk_aware_de":
                row["algorithm"] = "historical_main_276170"
            plans.append(row)
    keys = [(p["algorithm"], p["scenario"], p["planner_seed"]) for p in plans]
    design = {(s.name, seed) for s in scenarios for seed in range(4, 9)}
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate frozen path key")
    for algorithm in BASELINES + VARIANTS:
        if {(s, seed) for a, s, seed in keys if a == algorithm} != design:
            raise ValueError(f"Incomplete original 12-map x 5-seed design: {algorithm}")
    return plans, sources


def make_shift_bank(scenarios, model, samples, base_seed):
    bank = {}
    if model == "gaussian":
        realizations = make_realizations(scenarios, sigmas=(1.0,), samples=samples, perturbation_seed=base_seed)
        groups = defaultdict(list)
        for row in realizations:
            groups[row.scenario].append(row.obstacle_shifts)
        return {name: np.asarray(values) for name, values in groups.items()}
    seed = base_seed + MODELS[model]["seed_offset"]
    for scene in scenarios:
        rng = np.random.default_rng(stable_seed(seed, model, scene.name))
        shape = (samples, len(scene.environment.obstacles), 3)
        if model == "uniform":
            shifts = rng.uniform(-math.sqrt(3), math.sqrt(3), size=shape)
        elif model == "student_t3":
            shifts = rng.standard_t(df=3, size=shape) / math.sqrt(3)
        elif model == "common_translation":
            shifts = np.broadcast_to(rng.normal(size=(samples, 1, 3)), shape).copy()
        else:
            raise ValueError(model)
        bank[scene.name] = shifts
    return bank


def summarize(fixed_rows, scenarios, output, bootstrap_seed):
    grouped = defaultdict(list)
    for row in fixed_rows:
        grouped[(row["noise_model"], row["sigma"], row["algorithm"], row["scenario"])].append(row)
    scene_rows = []
    for (model, sigma, algorithm, scene), rows in sorted(grouped.items()):
        scene_rows.append({"noise_model": model, "sigma": sigma, "algorithm": algorithm,
            "scenario": scene, "family": scene.split("_")[1], "planner_seeds": len(rows),
            "nominal_feasible_rate": float(np.mean([r["nominal_feasible"] for r in rows])),
            "unconditional_collision_free_rate": float(np.mean([r["unconditional_collision_free_rate"] for r in rows])),
            "mean_path_length": float(np.mean([r["path_length"] for r in rows if r["path_length"] is not None]))})
    save_csv(output / "scenario_summary.csv", scene_rows)
    names = sorted(s.name for s in scenarios)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(names), size=(10000, len(names)))
    # Identical resampled scene indices for every algorithm/model/intensity.
    np.save(output / "scene_bootstrap_indices.npy", indices)
    cells = {(r["noise_model"], r["sigma"], r["algorithm"], r["scenario"]): r for r in scene_rows}
    summaries, family_rows, paired = [], [], []
    for model, spec in MODELS.items():
        for sigma in spec["sigmas"]:
            arrays = {a: np.asarray([cells[(model, sigma, a, s)]["unconditional_collision_free_rate"] for s in names]) for a in BASELINES + VARIANTS}
            strongest = max(BASELINES, key=lambda a: float(arrays[a].mean()))
            for algorithm in BASELINES + VARIANTS:
                values = arrays[algorithm]
                low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
                families = {}
                for family in ("clutter", "urban", "barriers"):
                    family_values = [values[i] for i, name in enumerate(names) if name.split("_")[1] == family]
                    families[family] = float(np.mean(family_values))
                    family_rows.append({"noise_model": model, "sigma": sigma, "algorithm": algorithm,
                        "family": family, "scenarios": len(family_values), "unconditional_collision_free_rate": families[family]})
                worst = min(families, key=families.get)
                summaries.append({"noise_model": model, "sigma": sigma, "algorithm": algorithm,
                    "scenarios": len(names), "nominal_plans": 5 * len(names),
                    "unconditional_collision_free_rate": float(values.mean()),
                    "scene_bootstrap95_low": float(low), "scene_bootstrap95_high": float(high),
                    "worst_family": worst, "worst_family_rate": families[worst],
                    "min_scene_rate": float(values.min()),
                    "strongest_baseline": strongest, "baseline_envelope": float(arrays[strongest].mean()),
                    "delta_vs_baseline_envelope_pp": float(100 * (values.mean() - arrays[strongest].mean())),
                    "mean_path_length": float(np.mean([cells[(model, sigma, algorithm, s)]["mean_path_length"] for s in names]))})
            for algorithm in VARIANTS:
                for baseline in BASELINES:
                    delta = 100 * (arrays[algorithm] - arrays[baseline])
                    low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
                    paired.append({"noise_model": model, "sigma": sigma, "algorithm": algorithm,
                        "baseline": baseline, "paired_scenes": len(names), "mean_delta_pp": float(delta.mean()),
                        "paired_scene_bootstrap95_low_pp": float(low), "paired_scene_bootstrap95_high_pp": float(high),
                        "scene_wins": int(np.sum(delta > 1e-12)), "scene_ties": int(np.sum(np.abs(delta) <= 1e-12)),
                        "scene_losses": int(np.sum(delta < -1e-12)),
                        "interval_scope": "pointwise descriptive interval; no simultaneous coverage or corrected superiority claim"})
    save_csv(output / "summary.csv", summaries)
    save_csv(output / "family_summary.csv", family_rows)
    save_csv(output / "paired_scene_comparisons.csv", paired)
    save_csv(output / "gaussian_curve.csv", [r for r in summaries if r["noise_model"] == "gaussian"])
    lines = ["# 冻结路径的扰动强度与分布失配实验", "",
        "原 root=20260923 的 12 个已暴露场景、每图 5 个规划种子；所有原路径冻结。8 个基线与 3 个方法版本逐一共同受测。",
        "每条件每路径 150 个外部试验；障碍不截断、不拒绝碰到起终点的扰动；边界固定。高斯、均匀、t3 各障碍各轴独立；共同平移模型把同一个三维位移施加于所有障碍。",
        "四种模型均为零均值、每坐标理论方差 sigma²；t3 使用理论归一化而不按样本方差重新缩放。同一模型各强度共用单位随机样本，方法及规划种子共用扰动。不同模型独立采样。",
        "无条件成功率要求名义可行且扰动后无碰撞；失败规划保留在分母。规划种子先在场景内平均，12 个场景等权汇总。",
        "场景 bootstrap 区间使用相同的 10,000 组场景重采样索引，不能把百万次共享扰动当作百万独立样本；这里仅提供逐条件描述区间，没有同时覆盖或多重校正优越性结论。每族仅 4 张地图，最坏族结果也仅作描述。",
        "这些是人为定义的分布敏感性检查，不代表真实传感器噪声、不证明分布外安全性，也不构成新的独立地图测试。评估结果不会反馈本轮规划。", "",
        "| 噪声模型 | sigma | 原版 CFR | 稳健版 CFR | 路径版 CFR | 最高基线 | 基线 CFR | 稳健版差值 pp | 路径版差值 pp |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    lookup = {(r["noise_model"], r["sigma"], r["algorithm"]): r for r in summaries}
    for model, spec in MODELS.items():
        for sigma in spec["sigmas"]:
            old, safe, short = [lookup[(model, sigma, a)] for a in VARIANTS]
            lines.append(f"| {model} | {sigma:.2f} | {old['unconditional_collision_free_rate']:.2%} | {safe['unconditional_collision_free_rate']:.2%} | {short['unconditional_collision_free_rate']:.2%} | {safe['strongest_baseline']} | {safe['baseline_envelope']:.2%} | {safe['delta_vs_baseline_envelope_pp']:+.2f} | {short['delta_vs_baseline_envelope_pp']:+.2f} |")
    lines += ["", "完整 11 方法与区间见 summary.csv；按 3 图族汇总见 family_summary.csv；逐场景配对差异见 paired_scene_comparisons.csv。",
        "原始逐扰动结果在各模型子文件夹 robustness_trials.csv.gz；实际单位位移存在 unit_shifts.npz，乘以 sigma 即为实际位移；原路径与来源 SHA256 已归档。",
        "每条路径/条件的首尾试验与现有 Path3D.collision_count 完整谓词一致后才接受结果；汇总不包含距离采样近似。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=96301)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("samples must be at least two")
    scenarios = list(split_research_scenarios(make_research_scenarios(20260923), protocol="research24_6_6_12").test)
    plans, sources = load_sources(args.historical, args.previous, scenarios)
    groups = defaultdict(list)
    for p in plans:
        groups[p["scenario"]].append(p)
    protocol = {"experiment": "frozen-path-noise-intensity-and-distribution-shift-v1", "root_seed": 20260923,
        "evidence_scope": "known original maps; frozen historical plans; no replanning or tuning",
        "models": MODELS, "unit_shift_seed": args.seed, "samples_per_condition": args.samples,
        "source_sha256": sources, "scenarios": [s.to_dict() for s in scenarios],
        "planner_seeds": list(range(4, 9)), "algorithms": list(BASELINES + VARIANTS),
        "planned_path_perturbation_checks": len(plans) * args.samples * sum(len(m['sigmas']) for m in MODELS.values()),
        "bounds_fixed": True, "no_clipping_or_rejection": True, "safety_margin": 0.,
        "aggregation": "all failures included; equal planner-seed mean then equal scene mean",
        "bootstrap": "10000 paired scene resamples, common indices, pointwise descriptive intervals only",
        "statistical_claim": "synthetic sensitivity experiment; no safety certification, sensor realism, or fresh-map claim",
        "clearance_computed": False, "collision_predicate": "vectorized exact closed segment-AABB slab test with public predicate checks"}
    fixed_rows, raw_count, reference_checks = [], 0, 0
    with recorded_run(args.output, protocol):
        dump_json(args.output / "protocol.json", protocol)
        dump_json(args.output / "nominal_paths.json", plans)
        fields = ["noise_model", "sigma", "algorithm", "scenario", "planner_seed", "trial", "path_exists", "nominal_feasible", "perturbed_collision_free", "robust_success"]
        for model, spec in MODELS.items():
            directory = args.output / model
            directory.mkdir()
            bank = make_shift_bank(scenarios, model, args.samples, args.seed)
            np.savez_compressed(directory / "unit_shifts.npz", **bank)
            with gzip.open(directory / "robustness_trials.csv.gz", "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for scene in scenarios:
                    for row in groups[scene.name]:
                        path = Path3D(row["waypoints"]) if row["waypoints"] is not None else None
                        nominal = _feasible(path, scene, 0.)
                        for sigma in spec["sigmas"]:
                            shifts = sigma * bank[scene.name]
                            free = collision_free_batch(path, scene, shifts)
                            for trial in (0, args.samples - 1):
                                environment = translate_environment(scene.environment, shifts[trial])
                                expected = path is not None and path.collision_count(environment, safety_margin=0.) == 0
                                if bool(free[trial]) != expected:
                                    raise AssertionError(f"Geometry mismatch: {model} {sigma} {scene.name} {row['algorithm']} {row['planner_seed']} {trial}")
                                reference_checks += 1
                            robust = nominal & free
                            success_count = int(np.sum(robust))
                            low, high = wilson_interval(success_count, args.samples) if nominal else (None, None)
                            fixed_rows.append({"noise_model": model, "sigma": sigma, "algorithm": row["algorithm"],
                                "scenario": scene.name, "planner_seed": row["planner_seed"], "path_exists": path is not None,
                                "nominal_feasible": nominal, "trials": args.samples, "robust_successes": success_count,
                                "unconditional_collision_free_rate": success_count / args.samples,
                                "conditional_wilson95_low": low, "conditional_wilson95_high": high,
                                "path_length": path.length if path is not None else None})
                            for trial in range(args.samples):
                                writer.writerow({"noise_model": model, "sigma": sigma, "algorithm": row["algorithm"],
                                    "scenario": scene.name, "planner_seed": row["planner_seed"], "trial": trial,
                                    "path_exists": int(path is not None), "nominal_feasible": int(nominal),
                                    "perturbed_collision_free": int(free[trial]), "robust_success": int(robust[trial])})
                                raw_count += 1
                    print(json.dumps({"model": model, "scene_complete": scene.name, "raw_trials": raw_count}), flush=True)
            save_csv(directory / "fixed_path_summary.csv", [r for r in fixed_rows if r["noise_model"] == model])
        save_csv(args.output / "fixed_path_summary.csv", fixed_rows)
        summaries = summarize(fixed_rows, scenarios, args.output, args.seed + 100)
        dump_json(args.output / "completion.json", {"raw_trials": raw_count, "public_predicate_checks": reference_checks,
            "algorithms": len(BASELINES + VARIANTS), "frozen_plans": len(plans), "conditions": sum(len(m['sigmas']) for m in MODELS.values()),
            "files_sha256": {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.output.rglob("*") if p.is_file() and p.name != "execution_manifest.json"}})
        print(json.dumps({"completed": str(args.output), "raw_trials": raw_count, "summary_rows": len(summaries)}), flush=True)


if __name__ == "__main__":
    main()
