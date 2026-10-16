"""Uncertainty- and difficulty-aware DE path-planning extension.

``AdaptiveDiversityDE`` already adapts its operators to population diversity.
This module adds one separately testable signal: the current population's
path-risk profile.  Risk is estimated from collision samples and clearance to
obstacles, or from a caller-provided point uncertainty model.  When the
population spends effort near collisions or uncertain regions, the optimizer
increases its differential step and crossover within the same evaluation
budget.  The signal is read from the path evaluation cache, so it does not
silently add objective evaluations.

This is a research candidate, not a claim that the final paper contribution
has already been established.  The external ``uncertainty_fn`` hook is kept
explicit so a later DEM, sensor-error or dynamic-obstacle model can replace
the geometric proxy without rewriting the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Callable, Sequence

from .adaptive_de import AdaptiveDEConfig, AdaptiveDEResult, AdaptiveDiversityDE
from .continuous_planner import ContinuousPathProblem
from .models import Environment3D, Path3D, Point3D


UncertaintyFn = Callable[[Point3D], float]


@dataclass(frozen=True)
class UADConfig:
    """Configuration for the uncertainty-aware path planner.

    ``base`` controls the ordinary DE mechanisms.  ``difficulty_gain`` is the
    only additional algorithmic gain: zero disables UAD operator adaptation,
    which makes ``UADConfig(base=...)`` a direct mechanism ablation.
    ``uncertainty_sigma`` controls how quickly clearance risk decays with
    distance from an obstacle.
    """

    base: AdaptiveDEConfig = field(default_factory=lambda: AdaptiveDEConfig())
    uncertainty_sigma: float = 1.0
    risk_sampling_step: float = 1.0
    risk_gate: float = 0.10
    difficulty_gain: float = 0.8
    collision_risk_weight: float = 0.75
    clearance_risk_weight: float = 0.25
    uncertainty_fn: UncertaintyFn | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.uncertainty_sigma <= 0.0:
            raise ValueError("uncertainty_sigma must be positive")
        if self.risk_sampling_step <= 0.0:
            raise ValueError("risk_sampling_step must be positive")
        if not 0.0 <= self.risk_gate < 1.0:
            raise ValueError("risk_gate must be in [0, 1)")
        if self.difficulty_gain < 0.0:
            raise ValueError("difficulty_gain cannot be negative")
        if self.collision_risk_weight < 0.0 or self.clearance_risk_weight < 0.0:
            raise ValueError("risk weights cannot be negative")
        if self.collision_risk_weight + self.clearance_risk_weight <= 0.0:
            raise ValueError("at least one risk weight must be positive")


def _point_box_distance(point: Point3D, environment: Environment3D) -> float:
    if not environment.obstacles:
        return math.inf
    return min(
        math.sqrt(sum(max(lo - x, 0.0, x - hi) ** 2 for x, lo, hi in zip(point, box.minimum, box.maximum)))
        for box in environment.obstacles
    )


def path_risk(
    path: Path3D,
    environment: Environment3D,
    *,
    segment_step: float,
    uncertainty_sigma: float,
    uncertainty_fn: UncertaintyFn | None = None,
    collision_weight: float = 0.75,
    clearance_weight: float = 0.25,
) -> float:
    """Return a bounded risk score for one path without calling an objective."""

    if segment_step <= 0.0:
        raise ValueError("segment_step must be positive")
    occupied = 0
    total = 0
    clearance = math.inf
    external_uncertainty = 0.0
    for start, end in zip(path.waypoints, path.waypoints[1:]):
        samples = max(1, math.ceil(math.dist(start, end) / segment_step))
        for index in range(samples + 1):
            t = index / samples
            point = tuple(a + (b - a) * t for a, b in zip(start, end))
            point3d = (float(point[0]), float(point[1]), float(point[2]))
            total += 1
            occupied += int(not environment.is_free(point3d))
            clearance = min(clearance, _point_box_distance(point3d, environment))
            if uncertainty_fn is not None:
                external_uncertainty = max(external_uncertainty, float(uncertainty_fn(point3d)))
    collision_risk = occupied / max(1, total)
    clearance_risk = 0.0 if math.isinf(clearance) else math.exp(-max(0.0, clearance) / uncertainty_sigma)
    weights = collision_weight + clearance_weight
    geometric_risk = (collision_weight * collision_risk + clearance_weight * clearance_risk) / weights
    return max(0.0, min(1.0, max(geometric_risk, external_uncertainty)))


def plan_continuous_path_uad(
    problem: ContinuousPathProblem,
    config: UADConfig | None = None,
) -> tuple[Path3D, AdaptiveDEResult]:
    """Plan a path while adapting DE exploration to cached path-risk signals."""

    config = config or UADConfig()
    if not problem.environment.is_free(problem.start, problem.safety_margin) or not problem.environment.is_free(problem.goal, problem.safety_margin):
        raise ValueError("start and goal must be collision-free and within bounds")
    if not problem.bounds:
        value = float(problem.objective([]))
        return problem.decode([]), AdaptiveDEResult([], value, [value], [0.0], [], 1, difficulty_history=[0.0])

    risk_cache: dict[tuple[float, ...], float] = {}

    def cached_objective(vector: Sequence[float]) -> float:
        key = tuple(float(value) for value in vector)
        path = problem.decode(key)
        value = float(problem.objective(key))
        risk_cache[key] = path_risk(
            path,
            problem.environment,
            segment_step=max(problem.segment_step, config.risk_sampling_step),
            uncertainty_sigma=config.uncertainty_sigma,
            uncertainty_fn=config.uncertainty_fn,
            collision_weight=config.collision_risk_weight,
            clearance_weight=config.clearance_risk_weight,
        )
        return value

    def difficulty_fn(
        population: Sequence[Sequence[float]],
        fitness: Sequence[float],
        diversity: float,
        success_rate: float,
    ) -> float:
        del fitness
        risks = [risk_cache.get(tuple(float(value) for value in vector), 0.0) for vector in population]
        population_risk = sum(risks) / max(1, len(risks))
        # The risk gate is deliberate: an obstacle-free/open environment must
        # not pay an exploration penalty merely because DE is stagnating.  The
        # new mechanism activates only after geometric or external uncertainty
        # is materially present; this makes the ablation interpretable.
        if population_risk <= config.risk_gate:
            return 0.0
        return max(0.0, min(1.0, (population_risk - config.risk_gate) / (1.0 - config.risk_gate)))

    base = replace(
        config.base,
        difficulty_gain=config.difficulty_gain,
        difficulty_fn=difficulty_fn,
    )
    result = AdaptiveDiversityDE(problem.bounds, base).optimize(cached_objective)
    return problem.decode(result.best_vector), result


__all__ = ["UADConfig", "path_risk", "plan_continuous_path_uad"]
