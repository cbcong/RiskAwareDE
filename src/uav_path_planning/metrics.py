"""Objective components and evaluation metrics shared by all planners."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

from .models import Environment3D, Path3D


def _point_box_distance(point: tuple[float, float, float], obstacle: Any) -> float:
    # Euclidean distance to an axis-aligned box (zero inside).
    return math.sqrt(sum(max(lo - x, 0.0, x - hi) ** 2 for x, lo, hi in zip(point, obstacle.minimum, obstacle.maximum)))


@dataclass(frozen=True)
class PathMetrics:
    length: float
    collision_segments: int
    collision_samples: int
    min_obstacle_distance: float
    smoothness: float
    altitude_violation: float
    feasible: bool
    objective: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class CostWeights:
    length: float = 1.0
    collision: float = 1_000.0
    safety: float = 0.0
    smoothness: float = 0.0
    altitude: float = 0.0

    def __post_init__(self) -> None:
        values = (self.length, self.collision, self.safety, self.smoothness, self.altitude)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("all cost weights must be finite and non-negative")


def evaluate_path(path: Path3D, environment: Environment3D, *, weights: CostWeights | None = None, segment_step: float = 0.25, safety_margin: float = 0.0, min_altitude: float | None = None, max_altitude: float | None = None) -> PathMetrics:
    if not math.isfinite(segment_step) or segment_step <= 0:
        raise ValueError("segment_step must be finite and positive")
    if not math.isfinite(safety_margin) or safety_margin < 0.0:
        raise ValueError("safety_margin must be finite and non-negative")
    if min_altitude is not None and not math.isfinite(min_altitude):
        raise ValueError("min_altitude must be finite")
    if max_altitude is not None and not math.isfinite(max_altitude):
        raise ValueError("max_altitude must be finite")
    if min_altitude is not None and max_altitude is not None and min_altitude > max_altitude:
        raise ValueError("min_altitude cannot exceed max_altitude")
    weights = weights or CostWeights()
    collision_segments = path.collision_count(environment, step=segment_step, safety_margin=safety_margin)
    collision_samples = 0
    min_distance = math.inf
    altitude_violation = 0.0
    for point in path.waypoints:
        min_distance = min(min_distance, *(max(0.0, _point_box_distance(point, o) - safety_margin) for o in environment.obstacles)) if environment.obstacles else min_distance
        if min_altitude is not None:
            altitude_violation += max(0.0, min_altitude - point[2])
        if max_altitude is not None:
            altitude_violation += max(0.0, point[2] - max_altitude)
    for a, b in zip(path.waypoints, path.waypoints[1:]):
        # Collision is exact (slab intersection) and independent of the
        # historical ``segment_step`` argument. Keep collision_samples as a
        # count of colliding segment witnesses for backwards compatibility.
        if not environment.segment_is_free(a, b, step=segment_step, safety_margin=safety_margin):
            collision_samples += 1
        if environment.obstacles:
            min_distance = min(min_distance, *(o.segment_distance(a, b, margin=safety_margin) for o in environment.obstacles))
    smoothness = 0.0
    for a, b, c in zip(path.waypoints, path.waypoints[1:], path.waypoints[2:]):
        # Both vectors point forward along the path: a->b and b->c.
        u, v = tuple(y - x for x, y in zip(a, b)), tuple(y - x for x, y in zip(b, c))
        nu, nv = math.dist((0, 0, 0), u), math.dist((0, 0, 0), v)
        if nu and nv:
            smoothness += 1.0 - max(-1.0, min(1.0, sum(x * y for x, y in zip(u, v)) / (nu * nv)))
    if math.isinf(min_distance):
        min_distance = math.inf
    safety_penalty = 0.0 if math.isinf(min_distance) else 1.0 / max(min_distance, 1e-9)
    objective = (weights.length * path.length + weights.collision * collision_segments + weights.safety * safety_penalty + weights.smoothness * smoothness + weights.altitude * altitude_violation)
    return PathMetrics(path.length, collision_segments, collision_samples, min_distance, smoothness, altitude_violation, collision_segments == 0 and altitude_violation == 0.0, objective)
