"""Evaluate frozen nominal paths against independent obstacle translations.

All algorithms and planner seeds share the same perturbations for a scenario.
No optimizer sees a perturbed environment.  These are empirical stress tests,
not a chance-constraint certificate or a model of real sensor uncertainty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
from typing import Any, Callable, Iterable, Sequence

from .benchmarks import BenchmarkConfig, _result_for_astar, _result_for_continuous
from .models import Environment3D, ObstacleBox, Path3D
from .scenarios import Scenario3D


DEFAULT_SIGMAS = (0.0, 0.1, 0.3, 0.6)


@dataclass(frozen=True)
class NominalPlan:
    algorithm: str
    scenario: str
    planner_seed: int
    path: Path3D | None
    runtime_seconds: float = 0.0
    evaluations: int = 0
    # Risk-aware planners perform additional perturbation checks inside one
    # nominal evaluation.  Preserve the actual count here so downstream
    # tuning reports do not reconstruct it as ``evaluations * K`` when a
    # screening ablation intentionally skips checks for nominally infeasible
    # candidates.
    internal_collision_checks: int = 0


@dataclass(frozen=True)
class PerturbationRealization:
    scenario: str
    sigma: float
    trial: int
    perturbation_seed: int
    obstacle_shifts: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class RobustnessTrial:
    algorithm: str
    scenario: str
    planner_seed: int
    sigma: float
    trial: int
    perturbation_seed: int
    path_exists: bool
    nominal_feasible: bool
    perturbed_collision_free: bool
    robust_success: bool
    clearance: float | None
    path_length: float | None = None


@dataclass
class RobustnessResult:
    nominal_plans: list[NominalPlan]
    realizations: list[PerturbationRealization]
    trials: list[RobustnessTrial]
    fixed_path_summaries: list[dict[str, Any]]
    scenario_summaries: list[dict[str, Any]]
    aggregate_summaries: list[dict[str, Any]]
    protocol: dict[str, Any] = field(default_factory=dict)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval, only used conditional on one fixed feasible path."""
    if isinstance(trials, bool) or isinstance(successes, bool) or not isinstance(trials, int) or not isinstance(successes, int):
        raise ValueError("successes and trials must be integers")
    if trials <= 0 or not 0 <= successes <= trials or not 0.0 < confidence < 1.0:
        raise ValueError("require trials > 0, 0 <= successes <= trials, 0 < confidence < 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denominator
    low = max(0.0, centre - radius)
    high = min(1.0, centre + radius)
    # Avoid tiny floating-point residues at the closed endpoints.
    if successes == 0:
        low = 0.0
    if successes == trials:
        high = 1.0
    return low, high


def _trial_seed(base: int, scenario: str, trial: int) -> int:
    # Stable across processes and independent of algorithm/planner seed/sigma.
    payload = json.dumps(["obstacle-translation-v1", base, scenario, trial], ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def translate_environment(environment: Environment3D, shifts: Sequence[Sequence[float]]) -> Environment3D:
    """Translate complete boxes without clipping/rejecting endpoint collisions.

    The world boundary is fixed. Obstacles may move partially outside it;
    clipping or rejecting such trials would change the Gaussian distribution.
    """
    if len(shifts) != len(environment.obstacles):
        raise ValueError("one translation is required per obstacle")
    obstacles = []
    for obstacle, shift in zip(environment.obstacles, shifts):
        if len(shift) != 3 or not all(math.isfinite(x) for x in shift):
            raise ValueError("translations must be finite 3-D vectors")
        obstacles.append(ObstacleBox(
            tuple(x + d for x, d in zip(obstacle.minimum, shift)),
            tuple(x + d for x, d in zip(obstacle.maximum, shift)),
        ))
    return Environment3D(environment.bounds, obstacles)


def make_realizations(
    scenarios: Sequence[Scenario3D], *, sigmas: Sequence[float] = DEFAULT_SIGMAS,
    samples: int = 30, perturbation_seed: int = 20260922,
) -> list[PerturbationRealization]:
    """Generate iid translations across trials, paired across algorithms/sigmas."""
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not isinstance(perturbation_seed, int) or isinstance(perturbation_seed, bool):
        raise ValueError("perturbation_seed must be an integer")
    sigma_values = tuple(float(value) for value in sigmas)
    if not sigma_values or len(set(sigma_values)) != len(sigma_values):
        raise ValueError("sigmas must be non-empty and distinct")
    if any(not math.isfinite(value) or value < 0.0 for value in sigma_values):
        raise ValueError("sigmas must be finite and non-negative")
    names = [scenario.name for scenario in scenarios]
    if not names or len(set(names)) != len(names):
        raise ValueError("scenarios must be non-empty with distinct names")
    output = []
    for scenario in scenarios:
        for trial in range(samples):
            seed = _trial_seed(perturbation_seed, scenario.name, trial)
            rng = random.Random(seed)
            normal_shifts = [tuple(rng.gauss(0.0, 1.0) for _ in range(3)) for _ in scenario.environment.obstacles]
            for sigma in sigma_values:
                shifts = tuple(tuple(sigma * value for value in shift) for shift in normal_shifts)
                output.append(PerturbationRealization(scenario.name, sigma, trial, seed, shifts))
    return output


def _feasible(path: Path3D | None, scenario: Scenario3D, safety_margin: float) -> bool:
    if path is None:
        return False
    endpoints_match = math.dist(path.waypoints[0], scenario.start) <= 1e-8 and math.dist(path.waypoints[-1], scenario.goal) <= 1e-8
    return endpoints_match and path.collision_count(scenario.environment, safety_margin=safety_margin) == 0


def _clearance(path: Path3D | None, environment: Environment3D, safety_margin: float) -> float | None:
    # Null distinguishes obstacle-free maps and absent paths from zero clearance.
    if path is None or not environment.obstacles:
        return None
    return min(obstacle.segment_distance(a, b, margin=safety_margin)
               for a, b in zip(path.waypoints, path.waypoints[1:])
               for obstacle in environment.obstacles)


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def summarize_trials(trials: Sequence[RobustnessTrial]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate seeds inside scenarios before averaging scenarios equally.

    Wilson intervals appear only at fixed-path level. Shared perturbations and
    repeated planner seeds are never pooled into an iid inferential sample.
    """
    path_groups: dict[tuple[str, str, int, float], list[RobustnessTrial]] = {}
    for row in trials:
        path_groups.setdefault((row.algorithm, row.scenario, row.planner_seed, row.sigma), []).append(row)
    fixed = []
    for (algorithm, scenario, seed, sigma), rows in sorted(path_groups.items()):
        nominal = rows[0].nominal_feasible
        if any(row.nominal_feasible != nominal for row in rows):
            raise ValueError("nominal feasibility must remain constant for a frozen path")
        if len({row.trial for row in rows}) != len(rows):
            raise ValueError("duplicate perturbation trial for one frozen path")
        successes = sum(row.robust_success for row in rows)
        low, high = wilson_interval(successes, len(rows)) if nominal else (None, None)
        fixed.append({
            "algorithm": algorithm, "scenario": scenario, "planner_seed": seed, "sigma": sigma,
            "nominal_feasible": nominal, "trials": len(rows), "robust_successes": successes,
            "conditional_collision_free_rate": successes / len(rows) if nominal else None,
            "unconditional_collision_free_rate": successes / len(rows),
            "conditional_wilson95_low": low, "conditional_wilson95_high": high,
            "mean_clearance": _mean(row.clearance for row in rows) if nominal else None,
        })
    scenario_groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in fixed:
        scenario_groups.setdefault((row["algorithm"], row["scenario"], row["sigma"]), []).append(row)
    scenarios = []
    for (algorithm, scenario, sigma), rows in sorted(scenario_groups.items()):
        scenarios.append({
            "algorithm": algorithm, "scenario": scenario, "sigma": sigma,
            "nominal_plans": len(rows), "nominal_feasible_plans": sum(row["nominal_feasible"] for row in rows),
            "nominal_feasible_rate": _mean(float(row["nominal_feasible"]) for row in rows),
            "conditional_collision_free_rate": _mean(row["conditional_collision_free_rate"] for row in rows),
            "unconditional_collision_free_rate": _mean(row["unconditional_collision_free_rate"] for row in rows),
            "mean_clearance": _mean(row["mean_clearance"] for row in rows),
        })
    algorithm_groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in scenarios:
        algorithm_groups.setdefault((row["algorithm"], row["sigma"]), []).append(row)
    aggregates = []
    for (algorithm, sigma), rows in sorted(algorithm_groups.items()):
        aggregates.append({
            "algorithm": algorithm, "sigma": sigma, "scenarios": len(rows),
            "scenarios_with_nominal_feasible_paths": sum(row["nominal_feasible_plans"] > 0 for row in rows),
            **{metric: _mean(row[metric] for row in rows) for metric in (
                "nominal_feasible_rate", "conditional_collision_free_rate", "unconditional_collision_free_rate", "mean_clearance")},
        })
    return fixed, scenarios, aggregates


def evaluate_robustness(
    scenarios: Sequence[Scenario3D], nominal_plans: Sequence[NominalPlan], *,
    sigmas: Sequence[float] = DEFAULT_SIGMAS, samples: int = 30,
    perturbation_seed: int = 20260922, safety_margin: float = 0.0,
) -> RobustnessResult:
    """Public interface for evaluating existing paths without rerunning planners."""
    if not math.isfinite(safety_margin) or safety_margin < 0:
        raise ValueError("safety_margin must be finite and non-negative")
    realizations = make_realizations(scenarios, sigmas=sigmas, samples=samples, perturbation_seed=perturbation_seed)
    scenario_map = {scenario.name: scenario for scenario in scenarios}
    plans = list(nominal_plans)
    keys = [(plan.algorithm, plan.scenario, plan.planner_seed) for plan in plans]
    if not plans or len(set(keys)) != len(keys):
        raise ValueError("nominal plans must be non-empty and distinct by algorithm, scenario, planner seed")
    if any(plan.scenario not in scenario_map for plan in plans):
        raise ValueError("a nominal plan refers to an unknown scenario")
    # A rectangular scenario/seed design prevents algorithm-dependent weighting.
    design: dict[str, set[tuple[str, int]]] = {}
    for plan in plans:
        design.setdefault(plan.algorithm, set()).add((plan.scenario, plan.planner_seed))
    if any(cases != next(iter(design.values())) for cases in design.values()):
        raise ValueError("all algorithms must have the same scenario and planner-seed design")
    nominal_feasible = {key: _feasible(plan.path, scenario_map[plan.scenario], safety_margin) for key, plan in zip(keys, plans)}
    plan_groups: dict[str, list[NominalPlan]] = {}
    for plan in plans:
        plan_groups.setdefault(plan.scenario, []).append(plan)
    trials = []
    for realization in realizations:
        perturbed = translate_environment(scenario_map[realization.scenario].environment, realization.obstacle_shifts)
        for plan in plan_groups.get(realization.scenario, []):
            nominal = nominal_feasible[(plan.algorithm, plan.scenario, plan.planner_seed)]
            collision_free = plan.path is not None and plan.path.collision_count(perturbed, safety_margin=safety_margin) == 0
            trials.append(RobustnessTrial(
                plan.algorithm, plan.scenario, plan.planner_seed, realization.sigma, realization.trial,
                realization.perturbation_seed, plan.path is not None, nominal, collision_free,
                nominal and collision_free, _clearance(plan.path, perturbed, safety_margin),
                plan.path.length if plan.path is not None else None,
            ))
    fixed, scene_rows, aggregate = summarize_trials(trials)
    protocol = {
        "protocol_version": "obstacle-translation-v1", "perturbation_seed": perturbation_seed,
        "sigmas": list(sigmas), "samples_per_fixed_path_per_sigma": samples, "safety_margin": safety_margin,
        "perturbation": "independent isotropic Gaussian translation of each obstacle; sigma in scene coordinate units",
        "no_clipping_or_rejection": True, "world_bounds_fixed": True, "nominal_path_frozen": True,
        "common_random_numbers": "same realizations across algorithms/planner seeds; same standard normals across sigmas",
        "unconditional_definition": "nominal feasible AND collision free under perturbation; failed nominal plans count as failure",
        "conditional_definition": "collision free under perturbation among nominal feasible paths only",
        "interval_scope": "Wilson 95% per fixed nominal feasible path, conditional on iid modeled perturbations; no aggregate iid interval",
        "aggregation": "equal mean over planner seeds within scene, then equal mean over scenes; conditional means omit scenes with no nominal feasible path and report count",
        "clearance": "exact segment to axis-expanded obstacle distance; null for no path/no obstacles; summaries condition on nominal feasibility",
        "claim_limit": "empirical synthetic perturbation test; no certified probability bound or real sensor uncertainty claim",
        "scenarios": [scenario.to_dict() for scenario in scenarios],
    }
    return RobustnessResult(plans, realizations, trials, fixed, scene_rows, aggregate, protocol)


def plan_nominal_paths(
    scenarios: Sequence[Scenario3D], *, algorithms: Sequence[str], config: BenchmarkConfig,
    progress: Callable[[str], None] | None = None,
) -> list[NominalPlan]:
    """Run benchmark planners once per nominal case, preserving absent paths."""
    if not algorithms or len(set(algorithms)) != len(algorithms):
        raise ValueError("algorithms must be non-empty and distinct")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("planner seeds must be distinct")
    output = []
    for scenario in scenarios:
        for algorithm in algorithms:
            astar_result = _result_for_astar(scenario, config) if algorithm == "astar3d" else None
            for seed in config.seeds:
                result = astar_result if astar_result is not None else _result_for_continuous(scenario, algorithm, seed, config)
                internal_checks = int(result.metadata.get("internal_collision_checks", 0))
                output.append(NominalPlan(
                    algorithm, scenario.name, seed, result.path,
                    result.runtime_seconds, result.evaluations, internal_checks,
                ))
                if progress is not None:
                    progress(f"nominal {scenario.name} {algorithm} planner_seed={seed}")
    return output


def run_robustness(
    scenarios: Sequence[Scenario3D], *, algorithms: Sequence[str] = ("de", "pso", "astar3d", "jade", "shade", "lshade", "adaptive_de", "uad_de"),
    config: BenchmarkConfig | None = None, sigmas: Sequence[float] = DEFAULT_SIGMAS,
    samples: int = 30, perturbation_seed: int = 20260922,
    progress: Callable[[str], None] | None = None,
) -> RobustnessResult:
    """Plan nominal paths, then apply an independent common perturbation bank."""
    config = config or BenchmarkConfig()
    # Validate the perturbation protocol before spending time on optimization.
    make_realizations(scenarios, sigmas=sigmas, samples=samples, perturbation_seed=perturbation_seed)
    plans = plan_nominal_paths(scenarios, algorithms=algorithms, config=config, progress=progress)
    result = evaluate_robustness(scenarios, plans, sigmas=sigmas, samples=samples,
                                 perturbation_seed=perturbation_seed, safety_margin=config.safety_margin)
    result.protocol["benchmark_config"] = asdict(config)
    return result


def _save_rows(rows: list[dict[str, Any]], target: Path) -> None:
    target.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_robustness(result: RobustnessResult, output: str | Path) -> None:
    """Write raw trials, reproducible paths/realizations, and descriptive tables."""
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    _save_rows([asdict(row) for row in result.trials], target / "robustness_trials")
    _save_rows(result.fixed_path_summaries, target / "fixed_path_summary")
    _save_rows(result.scenario_summaries, target / "scenario_summary")
    _save_rows(result.aggregate_summaries, target / "aggregate_summary")
    nominal_rows = [{
        "algorithm": row.algorithm, "scenario": row.scenario, "planner_seed": row.planner_seed,
        "runtime_seconds": row.runtime_seconds, "evaluations": row.evaluations,
        "internal_collision_checks": row.internal_collision_checks,
        "waypoints": [list(point) for point in row.path.waypoints] if row.path is not None else None,
    } for row in result.nominal_plans]
    (target / "nominal_paths.json").write_text(json.dumps(nominal_rows, indent=2, allow_nan=False), encoding="utf-8")
    (target / "realizations.json").write_text(json.dumps([asdict(row) for row in result.realizations], indent=2, allow_nan=False), encoding="utf-8")
    (target / "protocol.json").write_text(json.dumps(result.protocol, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# 独立障碍扰动鲁棒性实验", "",
        "路径仅在名义地图上规划一次。每个障碍以三维各轴独立高斯位移平移，sigma 单位等于场景坐标单位；不截断、不拒绝碰到起终点的扰动，世界边界保持不变。", "",
        "同一场景的扰动 realization 对全部算法和规划种子共享；扰动随机源与规划随机源分开。不同 sigma 共用标准正态样本后缩放，因此各 sigma 结果也存在配对关系。", "",
        "unconditional 成功要求名义可行且扰动后无碰撞，所有失败规划进入分母；conditional 只在名义可行路径上计算。先在场景内平均规划种子，再对场景等权平均。没有名义可行路径的场景不进入 conditional 均值，表中报告可用场景数。", "",
        "Wilson 95% 区间只见 fixed_path_summary，描述给定固定路径、给定合成 iid 扰动模型下的采样不确定性。场景或算法汇总不将重复种子和共享扰动冒充独立样本。这里不提供概率安全保证，也不证明真实传感器不确定性性能。", "",
        "clearance 为线段到按 safety_margin 扩张后的 AABB 的精确欧氏距离；没有障碍或没有路径时记 null，不当作零距离。汇总只使用名义可行路径。", "",
        "| algorithm | sigma | scenes | conditional scenes | nominal feasible | conditional collision-free | unconditional collision-free | mean clearance |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.aggregate_summaries:
        values = [row[key] for key in ("algorithm", "sigma", "scenarios", "scenarios_with_nominal_feasible_paths", "nominal_feasible_rate", "conditional_collision_free_rate", "unconditional_collision_free_rate", "mean_clearance")]
        lines.append("| " + " | ".join("NA" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value) for value in values) + " |")
    lines.extend(["", "原始路径见 nominal_paths.json，障碍平移见 realizations.json，完整场景和协议见 protocol.json，逐扰动记录见 robustness_trials.json/CSV。", ""])
    (target / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")


__all__ = ["DEFAULT_SIGMAS", "NominalPlan", "PerturbationRealization", "RobustnessTrial", "RobustnessResult", "wilson_interval", "translate_environment", "make_realizations", "summarize_trials", "evaluate_robustness", "plan_nominal_paths", "run_robustness", "save_robustness"]
