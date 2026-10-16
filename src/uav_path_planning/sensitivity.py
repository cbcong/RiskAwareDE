"""Parameter sensitivity and convergence experiments.

The benchmark runner answers *which* planner is better.  This module answers
the next research question: *which settings and mechanisms make the proposed
planner better, and how quickly does it converge?*  Every row wraps the
existing :class:`ExperimentRecord`, so sensitivity results use exactly the
same objective and path metrics as the baseline table.

The implementation intentionally performs one-factor-at-a-time sweeps.  This
keeps the interpretation suitable for an initial paper experiment; a later
factorial or Bayesian study can reuse :func:`run_parameter_sensitivity`
without changing the record format.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

from .benchmarks import BenchmarkConfig, run_one
from .experiments import ExperimentRecord
from .scenarios import Scenario3D, make_benchmark_scenarios


@dataclass(frozen=True)
class SensitivityRecord:
    """One parameter value, scenario and seed, linked to an experiment row."""

    parameter: str
    value: Any
    record: ExperimentRecord

    @property
    def history_summary(self) -> dict[str, Any]:
        metadata = self.record.config.get("planner_metadata", {})
        summary = metadata.get("history_summary", {})
        return dict(summary) if isinstance(summary, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        row = self.record.to_dict()
        row["parameter"] = self.parameter
        row["parameter_value"] = self.value
        row["history_summary"] = self.history_summary
        return row


# Names are deliberately public: the CLI and notebooks can select a subset
# without copying the experiment design into a second place.
DEFAULT_SWEEP_VALUES: dict[str, tuple[Any, ...]] = {
    "population": (8, 12, 20, 32),
    "generations": (10, 20, 40, 80),
    "budget": (240, 480, 820, 1600),
    "mutation_factor": (0.4, 0.7, 1.0),
    "crossover_rate": (0.5, 0.7, 0.9, 1.0),
    "diversity_threshold": (0.02, 0.08, 0.2),
    "stagnation_generations": (4, 12, 24),
    "p_best_rate": (0.1, 0.2, 0.4),
    "uad_difficulty_gain": (0.0, 0.4, 0.8, 1.2),
    "uad_uncertainty_sigma": (0.5, 1.0, 2.0),
    "uad_collision_risk_weight": (0.5, 0.75, 1.0),
    "uad_clearance_risk_weight": (0.0, 0.25, 0.5),
}

_PARAMETER_TO_CONFIG_FIELD = {
    "population": "adaptive_population",
    "generations": "adaptive_generations",
    "budget": "adaptive_max_evaluations",
    "mutation_factor": "adaptive_mutation_factor",
    "crossover_rate": "adaptive_crossover_rate",
    "diversity_threshold": "adaptive_diversity_threshold",
    "stagnation_generations": "adaptive_stagnation_generations",
    "p_best_rate": "adaptive_p_best_rate",
    "uad_difficulty_gain": "uad_difficulty_gain",
    "uad_uncertainty_sigma": "uad_uncertainty_sigma",
    "uad_collision_risk_weight": "uad_collision_risk_weight",
    "uad_clearance_risk_weight": "uad_clearance_risk_weight",
}


def _validate_parameter(parameter: str) -> str:
    if parameter not in _PARAMETER_TO_CONFIG_FIELD:
        valid = ", ".join(sorted(_PARAMETER_TO_CONFIG_FIELD))
        raise ValueError(f"unknown sensitivity parameter {parameter!r}; choose one of: {valid}")
    return parameter


def _config_field(parameter: str, algorithm: str) -> str:
    """Resolve the budget field for the selected candidate planner."""

    if parameter == "budget":
        return "uad_max_evaluations" if algorithm.startswith("uad") else "adaptive_max_evaluations"
    return _PARAMETER_TO_CONFIG_FIELD[parameter]


def config_for_parameter(
    base: BenchmarkConfig,
    parameter: str,
    value: Any,
    *,
    algorithm: str = "adaptive_de",
) -> BenchmarkConfig:
    """Return a benchmark config for one adaptive setting.

    Population sweeps keep the evaluation cap fixed and extend the generation
    limit as needed to make the cap reachable.  Budget sweeps also allow enough
    generations to exhaust their requested cap.  Thus neither sweep stops
    early merely because the old generation limit was too small.
    """

    parameter = _validate_parameter(parameter)
    field_name = _config_field(parameter, algorithm)
    # ``replace`` invokes BenchmarkConfig.__post_init__, catching invalid
    # budgets/rates before an expensive batch run starts.
    updates = {field_name: value}
    budget_cap = base.uad_max_evaluations if algorithm.startswith("uad") else base.adaptive_max_evaluations
    if parameter == "population" and budget_cap is not None:
        if int(value) < 4:
            raise ValueError("population must be at least 4")
        updates["adaptive_generations"] = max(base.adaptive_generations, math.ceil(budget_cap / int(value)))
    elif parameter == "budget":
        updates["adaptive_generations"] = max(base.adaptive_generations, math.ceil(int(value) / base.adaptive_population))
    elif parameter != "generations" and budget_cap is not None:
        updates["adaptive_generations"] = max(base.adaptive_generations, math.ceil(budget_cap / base.adaptive_population))
    return replace(base, **updates)


def run_parameter_sensitivity(
    scenarios: Sequence[Scenario3D] | None = None,
    *,
    parameter: str,
    values: Iterable[Any],
    config: BenchmarkConfig | None = None,
    algorithm: str = "adaptive_de",
) -> list[SensitivityRecord]:
    """Run one-factor-at-a-time sensitivity trials.

    The scenario order, seed order and planner budget are deterministic.  A
    separate config is produced for each value, while all other settings are
    inherited from ``config``.  ``algorithm`` is normally ``adaptive_de``;
    accepting an algorithm label makes the function useful for sanity checks
    against the plain DE implementation too.
    """

    parameter = _validate_parameter(parameter)
    if parameter.startswith("uad_") and not algorithm.startswith("uad"):
        raise ValueError(f"parameter {parameter!r} requires a UAD algorithm, got {algorithm!r}")
    values = list(values)
    if not values:
        raise ValueError("values must contain at least one parameter setting")
    base = config or BenchmarkConfig()
    # Population/rate/threshold sweeps use a common evaluation cap.  Generation
    # and budget sweeps deliberately change compute and therefore belong to
    # the convergence/compute scaling study rather than equal-budget tuning.
    budget_field = _config_field("budget", algorithm)
    existing_budget = getattr(base, budget_field)
    if parameter != "budget" and existing_budget is None:
        # A config produced for the other candidate may already carry the
        # intended cap; reuse it when switching the sweep to UAD.
        fixed_budget = base.adaptive_max_evaluations if algorithm.startswith("uad") and base.adaptive_max_evaluations is not None else base.adaptive_population * (base.adaptive_generations + 1)
        if parameter == "population":
            fixed_budget = max(fixed_budget, max(int(value) for value in values))
        elif parameter == "generations":
            fixed_budget = max(fixed_budget, base.adaptive_population * (max(int(value) for value in values) + 1))
        base = replace(base, **{budget_field: fixed_budget})
    selected_scenarios = list(scenarios or make_benchmark_scenarios())
    rows: list[SensitivityRecord] = []
    for value in values:
        trial_config = config_for_parameter(base, parameter, value, algorithm=algorithm)
        for scenario in selected_scenarios:
            for seed in trial_config.seeds:
                rows.append(SensitivityRecord(parameter, value, run_one(scenario, algorithm, seed, trial_config)))
    return rows


def _numeric(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _mean(values: Iterable[float]) -> float:
    values = _numeric(values)
    return mean(values) if values else math.nan


def _std(values: Iterable[float]) -> float:
    values = _numeric(values)
    return stdev(values) if len(values) > 1 else 0.0


def _history_value(row: SensitivityRecord, name: str) -> float:
    value = row.history_summary.get(name, math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _resample(values: Sequence[float], count: int = 21) -> list[float]:
    """Linearly resample a bounded convergence trace to a common x-axis."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return [math.nan] * count
    if len(finite) == 1:
        return finite * count
    output: list[float] = []
    for index in range(count):
        position = index * (len(finite) - 1) / (count - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            output.append(finite[low])
        else:
            weight = position - low
            output.append(finite[low] * (1.0 - weight) + finite[high] * weight)
    return output


def summarize_convergence(rows: Sequence[SensitivityRecord], points: int = 21) -> list[dict[str, Any]]:
    """Aggregate sampled best-so-far histories on a normalized progress axis."""

    if points < 2:
        raise ValueError("points must be at least 2")
    grouped: dict[tuple[str, str], list[SensitivityRecord]] = {}
    values: dict[tuple[str, str], Any] = {}
    for row in rows:
        key = (row.parameter, repr(row.value))
        grouped.setdefault(key, []).append(row)
        values[key] = row.value
    output: list[dict[str, Any]] = []
    for key, items in grouped.items():
        traces = [_resample(item.history_summary.get("sampled", []), points) for item in items]
        for index in range(points):
            values_at_step = _numeric(trace[index] for trace in traces)
            output.append({
                "parameter": key[0],
                "value": values[key],
                "progress": index / (points - 1),
                "mean_best_fitness": _mean(values_at_step),
                "std_best_fitness": _std(values_at_step),
                "runs": len(values_at_step),
            })
    return output


def summarize_sensitivity(rows: Sequence[SensitivityRecord]) -> list[dict[str, Any]]:
    """Aggregate sensitivity rows by parameter value.

    The standard deviation fields are included for reporting variability.  A
    feasible-length statistic excludes infeasible paths, while objective and
    feasible-rate statistics include all runs.
    """

    groups: dict[tuple[str, str], list[SensitivityRecord]] = {}
    values: dict[tuple[str, str], Any] = {}
    for row in rows:
        key = (row.parameter, repr(row.value))
        groups.setdefault(key, []).append(row)
        values[key] = row.value
    summary: list[dict[str, Any]] = []
    for (parameter, _), items in groups.items():
        lengths = [r.record.metrics.length for r in items if r.record.metrics.feasible]
        objectives = [r.record.metrics.objective for r in items]
        runtimes = [r.record.runtime_seconds for r in items]
        evals = [float(r.record.evaluations) for r in items]
        gains = [_history_value(r, "relative_gain_pct") for r in items]
        finals = [_history_value(r, "final") for r in items]
        initial = [_history_value(r, "initial") for r in items]
        feasible_rate = sum(bool(r.record.metrics.feasible) for r in items) / len(items)
        value = values[(parameter, repr(items[0].value))]
        summary.append({
            "parameter": parameter,
            "value": value,
            "runs": len(items),
            "feasible_rate": feasible_rate,
            "mean_length_feasible": _mean(lengths),
            "std_length_feasible": _std(lengths),
            "mean_objective": _mean(objectives),
            "std_objective": _std(objectives),
            "mean_runtime_seconds": _mean(runtimes),
            "std_runtime_seconds": _std(runtimes),
            "mean_evaluations": _mean(evals),
            "mean_initial_fitness": _mean(initial),
            "mean_final_fitness": _mean(finals),
            "mean_convergence_gain_pct": _mean(gains),
        })
    # Keep output readable and stable even when values have mixed numeric
    # types.  Numeric settings use numeric order (so 8, 12, 20 is natural),
    # while categorical settings fall back to their text representation.
    order = {parameter: index for index, parameter in enumerate(DEFAULT_SWEEP_VALUES)}
    def value_key(value: Any) -> tuple[int, Any]:
        return (0, float(value)) if isinstance(value, (int, float)) else (1, str(value))
    return sorted(summary, key=lambda row: (order.get(str(row["parameter"]), 999), value_key(row["value"])))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def save_sensitivity_records(rows: Iterable[SensitivityRecord], path: str | Path) -> None:
    """Save detailed sensitivity records as JSON or a flat CSV."""

    rows = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".json":
        target.write_text(json.dumps([_json_safe(row.to_dict()) for row in rows], indent=2), encoding="utf-8")
        return
    if target.suffix.lower() != ".csv":
        raise ValueError("sensitivity records path must end with .json or .csv")
    fields = [
        "parameter", "parameter_value", "algorithm", "scenario", "seed", "runtime_seconds",
        "evaluations", "status", "length", "collision_segments", "collision_samples",
        "min_obstacle_distance", "smoothness", "altitude_violation", "feasible", "objective",
        "history_initial", "history_final", "history_best", "history_steps", "history_relative_gain_pct",
    ]
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            history = row.history_summary
            metrics = row.record.metrics.to_dict()
            writer.writerow({
                "parameter": row.parameter,
                "parameter_value": row.value,
                "algorithm": row.record.algorithm,
                "scenario": row.record.scenario,
                "seed": row.record.seed,
                "runtime_seconds": row.record.runtime_seconds,
                "evaluations": row.record.evaluations,
                "status": row.record.status,
                **metrics,
                "history_initial": history.get("initial", ""),
                "history_final": history.get("final", ""),
                "history_best": history.get("best", ""),
                "history_steps": history.get("steps", ""),
                "history_relative_gain_pct": history.get("relative_gain_pct", ""),
            })


def save_sensitivity_summary(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Save an aggregate sensitivity table as JSON or Markdown."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".json":
        target.write_text(json.dumps(_json_safe(list(rows)), indent=2), encoding="utf-8")
        return
    if target.suffix.lower() != ".md":
        raise ValueError("sensitivity summary path must end with .json or .md")
    fields = [
        "parameter", "value", "runs", "feasible_rate", "mean_length_feasible", "std_length_feasible",
        "mean_objective", "std_objective", "mean_evaluations", "mean_runtime_seconds",
        "mean_initial_fitness", "mean_final_fitness", "mean_convergence_gain_pct",
    ]
    lines = [
        "# Adaptive DE 参数敏感性与收敛实验",
        "",
        "每一组改变一个研究参数，保持场景、随机种子和目标函数一致。种群/变异/阈值实验采用共同评价上限；代数和 budget 实验用于计算量扩展研究，必须结合实际评价次数解释。长度、目标值和运行时间越低越好；可行率越高越好。",
        "收敛增益定义为 `(初始最好目标 - 全程最好目标) / |初始最好目标| × 100%`，仅作过程诊断，不替代最终路径指标。",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                cells.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "建议先用本表筛选稳定区间，再在独立测试场景上锁定参数。不要根据测试集上的最佳单点直接宣称算法改进。",
        "",
    ])
    target.write_text("\n".join(lines), encoding="utf-8")


def save_convergence_summary(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Save normalized convergence traces as JSON or CSV for plotting."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".json":
        target.write_text(json.dumps(_json_safe(list(rows)), indent=2), encoding="utf-8")
        return
    if target.suffix.lower() != ".csv":
        raise ValueError("convergence summary path must end with .json or .csv")
    fields = ["parameter", "value", "progress", "mean_best_fitness", "std_best_fitness", "runs"]
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "SensitivityRecord", "DEFAULT_SWEEP_VALUES", "config_for_parameter",
    "run_parameter_sensitivity", "summarize_sensitivity", "summarize_convergence",
    "save_sensitivity_records", "save_sensitivity_summary", "save_convergence_summary",
]
