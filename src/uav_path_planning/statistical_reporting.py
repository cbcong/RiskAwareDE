"""Statistical summaries for reproducible UAV path-planning experiments.

The benchmark runner deliberately stores one :class:`ExperimentRecord` per
``(scenario, algorithm, seed)``.  This module consumes those records without
making assumptions about a particular optimizer.  It provides two levels of
analysis used in a paper:

* descriptive statistics for every scenario/algorithm group (mean, median,
  sample standard deviation and IQR), and
* paired comparisons on the same scenario and seed, including relative
  improvement, win rate, a paired Wilcoxon signed-rank test and a percentile
  bootstrap confidence interval.

Only the Python standard library is used.  When SciPy is unavailable this is
therefore still reproducible.  The Wilcoxon implementation uses an exact sign
enumeration for up to ``exact_max_n`` non-zero pairs and a tie-corrected normal
approximation for larger samples.  The method is recorded in every result so
that the choice is visible in exported reports.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import csv
import itertools
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from .experiments import ExperimentRecord


DEFAULT_METRICS: tuple[str, ...] = (
    "length",
    "objective",
    "collision_segments",
    "collision_samples",
    "min_obstacle_distance",
    "smoothness",
    "altitude_violation",
    "runtime_seconds",
    "evaluations",
)

# Path length, turning smoothness, collision count and compute cost are
# minimization metrics.  Clearance and feasibility are maximization metrics.
# Keeping this table in one place prevents a report from accidentally claiming
# that a larger clearance is a regression.
HIGHER_IS_BETTER_METRICS = frozenset({"min_obstacle_distance", "clearance", "feasible"})


def metric_higher_is_better(metric: str) -> bool:
    """Return the default direction for a metric used in comparisons."""

    return metric in HIGHER_IS_BETTER_METRICS


def _record_feasible(record: ExperimentRecord) -> bool:
    """Combine path geometry with the planner's explicit no-path outcome."""

    return bool(record.metrics.feasible) and record.status != "no_path"


def _metric_value(record: ExperimentRecord, metric: str) -> float:
    """Extract a scalar metric from a record.

    ``feasible`` is represented as 0/1 for convenience.  Unknown metric names
    fail early instead of silently producing a misleading report.
    """

    if metric == "runtime_seconds":
        return float(record.runtime_seconds)
    if metric == "evaluations":
        return float(record.evaluations)
    if metric == "feasible":
        return float(_record_feasible(record))
    if metric == "clearance":
        metric = "min_obstacle_distance"
    if hasattr(record.metrics, metric):
        value = getattr(record.metrics, metric)
        if isinstance(value, bool):
            return float(value)
        return float(value)
    raise ValueError(f"unknown experiment metric: {metric}")


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2
    if n % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, matching common statistics packages."""

    if not values:
        return math.nan
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between zero and one")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


@dataclass(frozen=True)
class DescriptiveSummary:
    """Statistics for one algorithm/scenario/metric cell."""

    algorithm: str
    scenario: str
    metric: str
    n: int
    valid_n: int
    feasible_n: int
    feasible_rate: float
    mean: float
    median: float
    std: float
    iqr: float
    minimum: float
    maximum: float
    failure_n: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_by_group(
    records: Sequence[ExperimentRecord],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> list[DescriptiveSummary]:
    """Summarize every ``(algorithm, scenario, metric)`` group.

    ``n`` includes rows with non-finite metric values, while ``valid_n`` is the
    number used for numeric statistics.  This distinction matters for metrics
    such as obstacle clearance, which may be infinite in an obstacle-free
    scene.  Length and smoothness additionally exclude failed paths.  Explicit
    ``no_path`` outcomes count as failures even if fallback geometry is feasible.
    Feasibility and failure counts always retain all original records.
    """

    grouped: dict[tuple[str, str], list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault((record.algorithm, record.scenario), []).append(record)
    output: list[DescriptiveSummary] = []
    for (algorithm, scenario), items in sorted(grouped.items()):
        feasible_n = sum(_record_feasible(item) for item in items)
        feasible_rate = feasible_n / len(items) if items else math.nan
        for metric in metrics:
            metric_items = items
            if metric in {"length", "smoothness"}:
                metric_items = [
                    item for item in items
                    if _record_feasible(item)
                ]
            values = [_metric_value(item, metric) for item in metric_items]
            valid = _finite(values)
            output.append(
                DescriptiveSummary(
                    algorithm=algorithm,
                    scenario=scenario,
                    metric=metric,
                    n=len(items),
                    valid_n=len(valid),
                    feasible_n=feasible_n,
                    feasible_rate=feasible_rate,
                    mean=_mean(valid),
                    median=_median(valid),
                    std=_sample_std(valid),
                    iqr=_percentile(valid, 0.75) - _percentile(valid, 0.25) if valid else math.nan,
                    minimum=min(valid) if valid else math.nan,
                    maximum=max(valid) if valid else math.nan,
                    failure_n=len(items) - feasible_n,
                )
            )
    return output


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based mid-ranks, retaining ties at their average rank."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in order[cursor:end]:
            ranks[position] = rank
        cursor = end
    return ranks


@dataclass(frozen=True)
class WilcoxonResult:
    """Paired Wilcoxon signed-rank test result."""

    n: int
    statistic: float
    p_value: float
    method: str
    z_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wilcoxon_signed_rank(
    differences: Sequence[float],
    *,
    exact_max_n: int = 20,
    zero_tolerance: float = 1e-12,
) -> WilcoxonResult:
    """Perform a two-sided paired Wilcoxon signed-rank test without SciPy.

    Zero differences are removed (the usual ``wilcox`` convention).  For at
    most ``exact_max_n`` non-zero pairs, all sign assignments are enumerated;
    this remains exact conditional on the observed absolute values, including
    ties.  Larger samples use the standard tie-corrected normal approximation
    with continuity correction.  The output records which method was used.
    """

    finite = _finite(differences)
    if not finite:
        return WilcoxonResult(0, math.nan, math.nan, "none (no finite paired differences)", math.nan)
    nonzero = [value for value in finite if abs(value) > zero_tolerance]
    n = len(nonzero)
    if n == 0:
        return WilcoxonResult(0, 0.0, 1.0, "none (all paired differences are zero)", 0.0)
    ranks = _average_ranks([abs(value) for value in nonzero])
    observed = sum(rank for rank, difference in zip(ranks, nonzero) if difference > 0.0)
    total_rank = sum(ranks)
    center = total_rank / 2.0

    if n <= exact_max_n:
        # Conditional randomization over all signs.  This is exact even when
        # tied absolute values create fractional mid-ranks.
        extreme = 0
        tolerance = 1e-10
        for signs in itertools.product((0, 1), repeat=n):
            positive = sum(rank for rank, sign in zip(ranks, signs) if sign)
            if abs(positive - center) + tolerance >= abs(observed - center):
                extreme += 1
        p_value = extreme / float(2**n)
        return WilcoxonResult(n, min(observed, total_rank - observed), min(1.0, p_value), "exact sign enumeration", math.nan)

    # Tie-corrected normal approximation.
    # With mid-ranks, the null distribution is obtained by independently
    # assigning a plus/minus sign to every observed rank.  Its variance is
    # therefore exactly ``sum(rank**2) / 4``.  Computing it from the actual
    # mid-ranks avoids the easy-to-miss factor-of-two error in tie-correction
    # formulas (for example, 21 identical absolute differences have rank 11
    # and variance 21*11**2/4).
    variance = sum(rank * rank for rank in ranks) / 4.0
    if variance <= 0.0:
        return WilcoxonResult(n, min(observed, total_rank - observed), 1.0, "normal approximation (zero variance)", 0.0)
    direction = 1.0 if observed >= center else -1.0
    corrected = observed - center - direction * 0.5
    z_value = corrected / math.sqrt(variance)
    p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return WilcoxonResult(n, min(observed, total_rank - observed), min(1.0, p_value), "tie-corrected normal approximation", z_value)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval.

    The random generator is local to this function, so report generation does
    not alter planner seeds or depend on global random state.
    """

    clean = _finite(values)
    if not clean:
        return math.nan, math.nan
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if statistic == "mean":
        stat = _mean
    elif statistic == "median":
        stat = _median
    else:
        raise ValueError("statistic must be 'mean' or 'median'")
    if len(clean) == 1:
        value = stat(clean)
        return value, value
    generator = random.Random(seed)
    estimates = [stat([clean[generator.randrange(len(clean))] for _ in clean]) for _ in range(n_resamples)]
    alpha = (1.0 - confidence) / 2.0
    return _percentile(estimates, alpha), _percentile(estimates, 1.0 - alpha)


@dataclass(frozen=True)
class PairedComparison:
    """Comparison of two algorithms on exactly matching scenario/seed pairs."""

    proposed: str
    baseline: str
    metric: str
    n_pairs: int
    feasible_rate_proposed: float
    feasible_rate_baseline: float
    feasible_delta_pp: float
    mean_difference: float
    median_difference: float
    paired_cohens_d: float
    rank_biserial_effect: float
    mean_relative_improvement_pct: float
    median_relative_improvement_pct: float
    std_relative_improvement_pct: float
    iqr_relative_improvement_pct: float
    win_rate_pct: float
    bootstrap_ci_low_pct: float
    bootstrap_ci_high_pct: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    wilcoxon_method: str
    scenario: str = "__all__"
    # ``n_pairs`` counts matching raw (scenario, seed) keys.  The valid and
    # excluded counts remain on that raw-pair scale; ``n_units`` is the count
    # after optional scenario aggregation.
    n_valid_pairs: int = 0
    n_excluded_pairs: int = 0
    n_compared: int = 0
    n_units: int = 0
    n_scenarios: int = 0
    proposed_failures: int = 0
    baseline_failures: int = 0
    either_failed_pairs: int = 0
    proposed_failure_rate: float = math.nan
    baseline_failure_rate: float = math.nan
    feasible_only: bool = False
    unit: str = "seed"
    higher_is_better: bool = False
    wilcoxon_p_value_holm: float = math.nan

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pair_records(
    records: Sequence[ExperimentRecord],
    proposed: str,
    baseline: str,
    scenario: str | None = None,
) -> list[tuple[ExperimentRecord, ExperimentRecord]]:
    """Match by ``(scenario, seed)`` and reject duplicate keys."""

    left: dict[tuple[str, int], ExperimentRecord] = {}
    right: dict[tuple[str, int], ExperimentRecord] = {}
    for record in records:
        if scenario is not None and record.scenario != scenario:
            continue
        key = (record.scenario, int(record.seed))
        if record.algorithm == proposed:
            if key in left:
                raise ValueError(f"duplicate proposed record for pair {key}")
            left[key] = record
        if record.algorithm == baseline:
            if key in right:
                raise ValueError(f"duplicate baseline record for pair {key}")
            right[key] = record
    return [(left[key], right[key]) for key in sorted(left.keys() & right.keys())]


def compare_paired(
    records: Sequence[ExperimentRecord],
    *,
    proposed: str,
    baseline: str,
    metric: str = "objective",
    higher_is_better: bool | None = None,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
    scenario: str | None = None,
    feasible_only: bool | None = None,
    unit: str = "seed",
) -> PairedComparison:
    """Compare two algorithms using matching scenario/seed observations.

    For lower-is-better metrics, relative improvement is
    ``(baseline - proposed) / abs(baseline)``.  Set ``higher_is_better`` for
    metrics such as feasible rate or clearance.  If omitted, direction is
    inferred from :func:`metric_higher_is_better`.  Length and smoothness use
    only pairs where both paths are feasible by default.  ``unit='scenario'``
    aggregates seeds within each scenario, while the default ``unit='seed'``
    is retained for diagnostic analysis.  Zero baseline values have no
    meaningful relative percentage and are omitted from that percentage while
    retained in the paired difference and Wilcoxon test.
    """

    if unit not in {"scenario", "seed"}:
        raise ValueError("unit must be 'scenario' or 'seed'")
    direction = metric_higher_is_better(metric) if higher_is_better is None else bool(higher_is_better)
    if feasible_only is None:
        feasible_only = metric in {"length", "smoothness"}
    pairs = _pair_records(records, proposed, baseline, scenario=scenario)
    matched_n = len(pairs)
    proposed_failures = sum(not _record_feasible(left) for left, _ in pairs)
    baseline_failures = sum(not _record_feasible(right) for _, right in pairs)
    either_failed = sum(not (_record_feasible(left) and _record_feasible(right)) for left, right in pairs)
    raw_observations: list[tuple[str, float, float]] = []
    for proposed_record, baseline_record in pairs:
        proposed_ok = _record_feasible(proposed_record)
        baseline_ok = _record_feasible(baseline_record)
        if feasible_only and not (proposed_ok and baseline_ok):
            continue
        proposed_value = _metric_value(proposed_record, metric)
        baseline_value = _metric_value(baseline_record, metric)
        if math.isfinite(proposed_value) and math.isfinite(baseline_value):
            raw_observations.append((proposed_record.scenario, proposed_value, baseline_value))
    if unit == "scenario":
        grouped_values: dict[str, list[tuple[float, float]]] = {}
        for scene, proposed_value, baseline_value in raw_observations:
            grouped_values.setdefault(scene, []).append((proposed_value, baseline_value))
        observations = [
            (scene, _mean([pair[0] for pair in values]), _mean([pair[1] for pair in values]))
            for scene, values in sorted(grouped_values.items())
        ]
    else:
        observations = raw_observations
    valid_pairs = len(raw_observations)
    excluded_pairs = matched_n - valid_pairs
    proposed_values = [item[1] for item in observations]
    baseline_values = [item[2] for item in observations]
    differences: list[float] = []
    relative: list[float] = []
    wins = 0
    for proposed_value, baseline_value in zip(proposed_values, baseline_values):
        # Positive difference means an improvement regardless of direction.
        difference = proposed_value - baseline_value if direction else baseline_value - proposed_value
        differences.append(difference)
        if difference > 1e-12:
            wins += 1
        if abs(baseline_value) > 1e-12:
            relative.append(100.0 * difference / abs(baseline_value))
    test = wilcoxon_signed_rank(differences)
    if not matched_n:
        test = replace(test, method="none (no matched pairs)")
    elif not valid_pairs:
        test = replace(test, method="none (no valid pairs after filtering)")
    ci_low, ci_high = bootstrap_ci(relative, statistic="mean", n_resamples=bootstrap_resamples, seed=bootstrap_seed)
    difference_std = _sample_std(differences)
    if differences and difference_std > 0.0 and math.isfinite(difference_std):
        paired_cohens_d = _mean(differences) / difference_std
    elif differences and abs(_mean(differences)) > 0.0:
        paired_cohens_d = math.copysign(math.inf, _mean(differences))
    else:
        paired_cohens_d = 0.0 if differences else math.nan
    if differences:
        effect_ranks = _average_ranks([abs(value) for value in differences if abs(value) > 1e-12])
        effect_nonzero = [value for value in differences if abs(value) > 1e-12]
        if effect_nonzero:
            positive_rank_sum = sum(rank for rank, value in zip(effect_ranks, effect_nonzero) if value > 0.0)
            total_effect_rank = sum(effect_ranks)
            rank_biserial_effect = (2.0 * positive_rank_sum - total_effect_rank) / total_effect_rank if total_effect_rank else 0.0
        else:
            rank_biserial_effect = 0.0
    else:
        rank_biserial_effect = math.nan
    if relative:
        mean_relative = _mean(relative)
        median_relative = _median(relative)
        std_relative = _sample_std(relative)
        iqr_relative = _percentile(relative, 0.75) - _percentile(relative, 0.25)
    else:
        mean_relative = median_relative = std_relative = iqr_relative = math.nan
    proposed_feasible = [float(_record_feasible(record)) for record, _ in pairs]
    baseline_feasible = [float(_record_feasible(record)) for _, record in pairs]
    proposed_feasible_rate = _mean(proposed_feasible)
    baseline_feasible_rate = _mean(baseline_feasible)
    return PairedComparison(
        proposed=proposed,
        baseline=baseline,
        metric=metric,
        n_pairs=matched_n,
        n_valid_pairs=valid_pairs,
        n_excluded_pairs=excluded_pairs,
        feasible_rate_proposed=proposed_feasible_rate,
        feasible_rate_baseline=baseline_feasible_rate,
        feasible_delta_pp=(proposed_feasible_rate - baseline_feasible_rate) * 100.0,
        mean_difference=_mean(differences),
        median_difference=_median(differences),
        paired_cohens_d=paired_cohens_d,
        rank_biserial_effect=rank_biserial_effect,
        mean_relative_improvement_pct=mean_relative,
        median_relative_improvement_pct=median_relative,
        std_relative_improvement_pct=std_relative,
        iqr_relative_improvement_pct=iqr_relative,
        win_rate_pct=(wins / len(differences) * 100.0) if differences else math.nan,
        bootstrap_ci_low_pct=ci_low,
        bootstrap_ci_high_pct=ci_high,
        wilcoxon_statistic=test.statistic,
        wilcoxon_p_value=test.p_value,
        wilcoxon_method=test.method,
        scenario=scenario or "__all__",
        n_compared=len(differences),
        n_units=len(differences),
        n_scenarios=len({item[0] for item in observations}),
        proposed_failures=proposed_failures,
        baseline_failures=baseline_failures,
        either_failed_pairs=either_failed,
        proposed_failure_rate=(proposed_failures / matched_n) if matched_n else math.nan,
        baseline_failure_rate=(baseline_failures / matched_n) if matched_n else math.nan,
        feasible_only=bool(feasible_only),
        unit=unit,
        higher_is_better=direction,
    )


def pairwise_comparisons(
    records: Sequence[ExperimentRecord],
    *,
    proposed: str,
    baselines: Sequence[str],
    metrics: Sequence[str] = ("length", "objective"),
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
    unit: str = "scenario",
) -> list[PairedComparison]:
    """Create comparisons against each baseline and selected metric."""

    output: list[PairedComparison] = []
    for baseline in baselines:
        for metric in metrics:
            output.append(
                compare_paired(
                    records,
                    proposed=proposed,
                    baseline=baseline,
                    metric=metric,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                    unit=unit,
                )
            )
    return _apply_holm(output)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down family-wise error correction, preserving input order."""

    valid = [(index, float(value)) for index, value in enumerate(p_values) if math.isfinite(float(value))]
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(sorted(valid, key=lambda item: item[1])):
        candidate = min(1.0, (len(valid) - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _apply_holm(rows: Sequence[PairedComparison]) -> list[PairedComparison]:
    adjusted = holm_adjust([row.wilcoxon_p_value for row in rows])
    return [replace(row, wilcoxon_p_value_holm=value) for row, value in zip(rows, adjusted)]


def pairwise_comparisons_seed_diagnostic(
    records: Sequence[ExperimentRecord],
    *,
    proposed: str,
    baselines: Sequence[str],
    metrics: Sequence[str] = ("length", "objective"),
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> list[PairedComparison]:
    """Seed-level paired results, explicitly marked diagnostic."""

    return pairwise_comparisons(
        records,
        proposed=proposed,
        baselines=baselines,
        metrics=metrics,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        unit="seed",
    )


def pairwise_comparisons_by_scenario(
    records: Sequence[ExperimentRecord],
    *,
    proposed: str,
    baselines: Sequence[str],
    metrics: Sequence[str] = ("length", "objective"),
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> list[PairedComparison]:
    """Run the same paired comparison separately for every scenario.

    Rows with no matching proposed/baseline seed remain in the output with
    ``n_pairs == 0``.  Keeping those rows visible makes incomplete benchmark
    downloads or failed runs auditable in a paper supplement.
    """

    scenarios = sorted({record.scenario for record in records})
    output: list[PairedComparison] = []
    for scenario in scenarios:
        for baseline in baselines:
            for metric in metrics:
                output.append(
                    compare_paired(
                        records,
                        proposed=proposed,
                        baseline=baseline,
                        metric=metric,
                        bootstrap_resamples=bootstrap_resamples,
                        bootstrap_seed=bootstrap_seed,
                        scenario=scenario,
                    )
                )
    return _apply_holm(output)


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_statistical_report(
    records: Sequence[ExperimentRecord],
    output_dir: str | Path,
    *,
    proposed: str = "adaptive_de",
    baselines: Sequence[str] = ("de", "pso", "astar3d"),
    metrics: Sequence[str] = ("length", "objective"),
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
    stem: str = "statistical_report",
) -> tuple[list[DescriptiveSummary], list[PairedComparison]]:
    """Write machine-readable CSVs and a reviewer-readable Markdown report."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    descriptive = summarize_by_group(records, metrics=tuple(metrics) + ("feasible", "runtime_seconds", "evaluations"))
    comparisons = pairwise_comparisons(
        records,
        proposed=proposed,
        baselines=baselines,
        metrics=metrics,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    scenario_comparisons = pairwise_comparisons_by_scenario(
        records,
        proposed=proposed,
        baselines=baselines,
        metrics=metrics,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    _write_csv([row.to_dict() for row in descriptive], target / f"{stem}_descriptive.csv")
    _write_csv([row.to_dict() for row in comparisons], target / f"{stem}_paired.csv")
    _write_csv([row.to_dict() for row in scenario_comparisons], target / f"{stem}_paired_by_scenario.csv")

    lines = [
        "# 严格统计报告",
        "",
        "本报告按场景和算法分别汇总；总体推断先在每个场景内平均随机种子，因此地图是独立分析单位。路径长度和目标值越低越好。",
        "相对改进定义为 `(baseline - proposed) / |baseline| × 100%`；正值表示候选算法更好。",
        "Wilcoxon 使用小样本精确符号枚举，大样本使用带并列秩修正和连续性修正的正态近似。",
        "Bootstrap 区间为配对相对改进均值的百分位区间。",
        "长度和转弯平滑性只使用双方都可行的匹配运行；匹配总数、双方失败数和被排除配对仍单独报告。",
        "matched/valid/excluded pairs 均以原始 `(scenario, seed)` 配对计数，满足 matched = valid + excluded；compared maps 为有效配对按地图聚合后的检验样本数，不是被排除的配对数。",
        "没有匹配配对或全部配对被排除时，检验和效应量显示 NA；实际存在配对且差异全零时才报告 p = 1。",
        "seed 级配对结果仅用于诊断，主结果不把同一场景的多个随机种子当成独立地图。",
        "Wilcoxon p 值同时给出 Holm family-wise error 校正值。",
        "",
        "## 配对比较",
        "",
        "| proposed | baseline | metric | matched pairs | valid pairs | compared maps | excluded pairs | proposed failures % | baseline failures % | mean improvement % | median improvement % | 95% bootstrap CI | win rate % | paired d | rank-biserial | Wilcoxon p | Holm p | method |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        def fmt(value: Any) -> str:
            return "NA" if isinstance(value, float) and not math.isfinite(value) else f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(
            "| "
            + " | ".join(
                [
                    row.proposed,
                    row.baseline,
                    row.metric,
                    str(row.n_pairs),
                    str(row.n_valid_pairs),
                    str(row.n_units),
                    str(row.n_excluded_pairs),
                    fmt(100.0 * row.proposed_failure_rate),
                    fmt(100.0 * row.baseline_failure_rate),
                    fmt(row.mean_relative_improvement_pct),
                    fmt(row.median_relative_improvement_pct),
                    f"[{fmt(row.bootstrap_ci_low_pct)}, {fmt(row.bootstrap_ci_high_pct)}]",
                    fmt(row.win_rate_pct),
                    fmt(row.paired_cohens_d),
                    fmt(row.rank_biserial_effect),
                    fmt(row.wilcoxon_p_value),
                    fmt(row.wilcoxon_p_value_holm),
                    row.wilcoxon_method,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
        "## 可行率配对差异",
            "",
            "| proposed | baseline | metric | proposed feasible rate | baseline feasible rate | delta (percentage points) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row.proposed} | {row.baseline} | {row.metric} | {row.feasible_rate_proposed:.4f} | {row.feasible_rate_baseline:.4f} | {row.feasible_delta_pp:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 分场景配对相对改进",
            "",
            "详细机器可读结果见 `statistical_report_paired_by_scenario.csv`。以下列出每个场景和算法组合的均值相对改进、Bootstrap 区间及 Wilcoxon p 值。",
            "",
            "| scenario | proposed | baseline | metric | matched pairs | valid pairs | excluded pairs | mean improvement % | 95% bootstrap CI | win rate % | Wilcoxon p | Holm p |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scenario_comparisons:
        def fmt_scenario(value: Any) -> str:
            return "NA" if isinstance(value, float) and not math.isfinite(value) else f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(
            f"| {row.scenario} | {row.proposed} | {row.baseline} | {row.metric} | {row.n_pairs} | {row.n_valid_pairs} | {row.n_excluded_pairs} | {fmt_scenario(row.mean_relative_improvement_pct)} | [{fmt_scenario(row.bootstrap_ci_low_pct)}, {fmt_scenario(row.bootstrap_ci_high_pct)}] | {fmt_scenario(row.win_rate_pct)} | {fmt_scenario(row.wilcoxon_p_value)} | {fmt_scenario(row.wilcoxon_p_value_holm)} |"
        )
    lines.extend(
        [
            "",
            "## 复现实验注意事项",
            "",
            "* 每个配对由同一场景和同一随机种子组成；缺少匹配运行的记录不会被伪造或跨场景替代。",
            "* 描述统计同时保存 `n`、`valid_n` 和 `failure_n`；长度和平滑性剔除失败路径，非有限指标也不进入均值、标准差和区间。`no_path` 状态始终记为失败。",
            "* 统计显著性不能替代效应量。论文中应同时报告可行率、相对改进、置信区间和每个场景的结果。",
            "",
        ]
    )
    (target / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    return descriptive, comparisons


__all__ = [
    "DEFAULT_METRICS",
    "HIGHER_IS_BETTER_METRICS",
    "metric_higher_is_better",
    "DescriptiveSummary",
    "WilcoxonResult",
    "PairedComparison",
    "summarize_by_group",
    "wilcoxon_signed_rank",
    "bootstrap_ci",
    "compare_paired",
    "pairwise_comparisons",
    "pairwise_comparisons_seed_diagnostic",
    "pairwise_comparisons_by_scenario",
    "holm_adjust",
    "write_statistical_report",
]
