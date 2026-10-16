"""Continuous waypoint planning using the baseline Differential Evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .de import DEConfig, DifferentialEvolution, DEResult
from .metrics import CostWeights, PathMetrics, evaluate_path
from .models import Environment3D, Path3D, Point3D


@dataclass
class ContinuousPathProblem:
    start: Point3D
    goal: Point3D
    environment: Environment3D
    intermediate_waypoints: int = 2
    collision_penalty: float | None = 10_000.0
    segment_step: float = 0.25
    safety_margin: float = 0.0
    weights: CostWeights | None = None
    min_altitude: float | None = None
    max_altitude: float | None = None
    evaluation_count: int = field(default=0, init=False)
    first_feasible_evaluation: int | None = field(default=None, init=False)
    evaluation_trace: list[tuple[int, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.intermediate_waypoints < 0:
            raise ValueError("intermediate_waypoints cannot be negative")
        if not math.isfinite(self.segment_step) or self.segment_step <= 0:
            raise ValueError("segment_step must be finite and positive")
        if self.collision_penalty is not None and (not math.isfinite(self.collision_penalty) or self.collision_penalty < 0):
            raise ValueError("collision_penalty must be finite and non-negative")
        if not math.isfinite(self.safety_margin) or self.safety_margin < 0:
            raise ValueError("safety_margin must be finite and non-negative")
        if self.min_altitude is not None and not math.isfinite(self.min_altitude):
            raise ValueError("min_altitude must be finite")
        if self.max_altitude is not None and not math.isfinite(self.max_altitude):
            raise ValueError("max_altitude must be finite")
        if self.min_altitude is not None and self.max_altitude is not None and self.min_altitude > self.max_altitude:
            raise ValueError("min_altitude cannot exceed max_altitude")
        if self.weights is not None and not isinstance(self.weights, CostWeights):
            raise TypeError("weights must be a CostWeights instance")

    @property
    def effective_weights(self) -> CostWeights:
        """Return the one objective definition shared by optimizers/reports."""
        if self.weights is not None:
            return self.weights
        return CostWeights(collision=self.collision_penalty if self.collision_penalty is not None else 0.0)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [axis for _ in range(self.intermediate_waypoints) for axis in zip(self.environment.bounds.minimum, self.environment.bounds.maximum)]

    def decode(self, vector: Sequence[float]) -> Path3D:
        expected = 3 * self.intermediate_waypoints
        if len(vector) != expected:
            raise ValueError(f"expected {expected} values, received {len(vector)}")
        points = [self.start]
        points.extend(tuple(float(x) for x in vector[i : i + 3]) for i in range(0, expected, 3))
        points.append(self.goal)
        return Path3D(points)

    def objective(self, vector: Sequence[float]) -> float:
        path = self.decode(vector)
        metrics = self.evaluate(path)
        self.evaluation_count += 1
        if metrics.feasible and self.first_feasible_evaluation is None:
            self.first_feasible_evaluation = self.evaluation_count
        if not self.evaluation_trace or metrics.objective < self.evaluation_trace[-1][1]:
            self.evaluation_trace.append((self.evaluation_count, metrics.objective))
        return metrics.objective

    def evaluate(self, path: Path3D) -> PathMetrics:
        """Evaluate a decoded path with exactly the optimization objective."""
        return evaluate_path(
            path,
            self.environment,
            weights=self.effective_weights,
            segment_step=self.segment_step,
            safety_margin=self.safety_margin,
            min_altitude=self.min_altitude,
            max_altitude=self.max_altitude,
        )


def plan_continuous_path(problem: ContinuousPathProblem, config: DEConfig | None = None) -> tuple[Path3D, DEResult]:
    """Optimize intermediate waypoint coordinates and return path plus optimizer result."""
    if not problem.environment.is_free(problem.start, problem.safety_margin) or not problem.environment.is_free(problem.goal, problem.safety_margin):
        raise ValueError("start and goal must be collision-free and within bounds")
    optimizer = DifferentialEvolution(problem.bounds or [(0.0, 1.0)], config=config)
    if not problem.bounds:
        # DE expects one dimension, but a zero-waypoint path has no variables.
        value = problem.objective([])
        result = DEResult([], value, [value], 1)
    else:
        result = optimizer.optimize(problem.objective)
    return problem.decode(result.best_vector), result
