"""Geometry and environment data structures used by planners.

The implementation deliberately uses standard-library types so experiments can
run on a small laptop without a GPU or a simulation stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

Point3D = tuple[float, float, float]


def _point(value: Sequence[float]) -> Point3D:
    if len(value) != 3:
        raise ValueError("a 3-D point must have exactly three coordinates")
    result = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(coordinate) for coordinate in result):
        raise ValueError("point coordinates must be finite")
    return result


def _validate_margin(margin: float) -> None:
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("safety margin must be finite and non-negative")


@dataclass(frozen=True)
class Bounds3D:
    """Closed axis-aligned rectangular search domain."""

    minimum: Point3D
    maximum: Point3D

    def __post_init__(self) -> None:
        lo, hi = _point(self.minimum), _point(self.maximum)
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError("each minimum coordinate must be smaller than maximum")
        object.__setattr__(self, "minimum", lo)
        object.__setattr__(self, "maximum", hi)

    def clamp(self, point: Sequence[float]) -> Point3D:
        p = _point(point)
        return tuple(max(a, min(b, x)) for x, a, b in zip(p, self.minimum, self.maximum))  # type: ignore[return-value]

    @property
    def dimensions(self) -> Point3D:
        return tuple(b - a for a, b in zip(self.minimum, self.maximum))  # type: ignore[return-value]


@dataclass(frozen=True)
class ObstacleBox:
    """Closed axis-aligned obstacle box."""

    minimum: Point3D
    maximum: Point3D

    def __post_init__(self) -> None:
        lo, hi = _point(self.minimum), _point(self.maximum)
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError("obstacle minimum cannot exceed maximum")
        object.__setattr__(self, "minimum", lo)
        object.__setattr__(self, "maximum", hi)

    def contains(self, point: Sequence[float], margin: float = 0.0) -> bool:
        _validate_margin(margin)
        p = _point(point)
        return all(a - margin <= x <= b + margin for x, a, b in zip(p, self.minimum, self.maximum))

    def intersects_segment(self, start: Sequence[float], end: Sequence[float], margin: float = 0.0) -> bool:
        """Exact slab test against the closed, axis-wise expanded obstacle.

        Boundary contact counts as collision. Parallel and zero-length
        segments are handled without a division-by-zero tolerance heuristic.
        """
        _validate_margin(margin)
        a, b = _point(start), _point(end)
        entry, exit_ = 0.0, 1.0
        for x, y, lower, upper in zip(a, b, self.minimum, self.maximum):
            lower, upper = lower - margin, upper + margin
            direction = y - x
            if direction == 0.0:
                if x < lower or x > upper:
                    return False
                continue
            near, far = (lower - x) / direction, (upper - x) / direction
            if near > far:
                near, far = far, near
            entry, exit_ = max(entry, near), min(exit_, far)
            if entry > exit_:
                return False
        return True

    def segment_distance(self, start: Sequence[float], end: Sequence[float], margin: float = 0.0) -> float:
        """Exact Euclidean distance from a segment to the expanded box.

        The squared distance is a convex, piecewise quadratic function of
        segment parameter t. Box-plane crossings divide [0, 1] into pieces;
        evaluating each stationary point and piece boundary finds its minimum.
        No spatial sampling resolution enters this clearance calculation.
        """
        _validate_margin(margin)
        a, b = _point(start), _point(end)
        if self.intersects_segment(a, b, margin):
            return 0.0
        direction = tuple(y - x for x, y in zip(a, b))
        lo = tuple(x - margin for x in self.minimum)
        hi = tuple(x + margin for x in self.maximum)
        breaks = {0.0, 1.0}
        for x, delta, lower, upper in zip(a, direction, lo, hi):
            if delta != 0.0:
                for boundary in (lower, upper):
                    t = (boundary - x) / delta
                    if 0.0 < t < 1.0:
                        breaks.add(t)
        ordered = sorted(breaks)

        def squared_distance(t: float) -> float:
            return sum(max(lower - (x + t * delta), 0.0, x + t * delta - upper) ** 2
                       for x, delta, lower, upper in zip(a, direction, lo, hi))

        minimum = min(squared_distance(t) for t in ordered)
        for left, right in zip(ordered, ordered[1:]):
            middle = (left + right) / 2.0
            numerator, denominator = 0.0, 0.0
            for x, delta, lower, upper in zip(a, direction, lo, hi):
                value = x + middle * delta
                boundary = lower if value < lower else upper if value > upper else None
                if boundary is not None:
                    numerator += delta * (x - boundary)
                    denominator += delta * delta
            if denominator:
                t = max(left, min(right, -numerator / denominator))
                minimum = min(minimum, squared_distance(t))
        return math.sqrt(max(0.0, minimum))


@dataclass
class Environment3D:
    """Continuous 3-D environment with optional box obstacles."""

    bounds: Bounds3D
    obstacles: list[ObstacleBox] = field(default_factory=list)

    def is_free(self, point: Sequence[float], safety_margin: float = 0.0) -> bool:
        _validate_margin(safety_margin)
        p = _point(point)
        if any(x < a or x > b for x, a, b in zip(p, self.bounds.minimum, self.bounds.maximum)):
            return False
        return not any(o.contains(p, safety_margin) for o in self.obstacles)

    def segment_is_free(
        self,
        start: Sequence[float],
        end: Sequence[float],
        step: float = 0.25,
        safety_margin: float = 0.0,
    ) -> bool:
        """Check a segment exactly against closed box obstacles and bounds.

        ``step`` is retained for API compatibility, but does not affect this
        exact collision predicate. Safety margin expands obstacles along all
        three axes; it does not shrink the rectangular world boundary.
        """
        if not math.isfinite(step) or step <= 0:
            raise ValueError("step must be finite and positive")
        _validate_margin(safety_margin)
        a, b = _point(start), _point(end)
        # A rectangular world is convex, so endpoint membership is enough.
        if any(x < lo or x > hi for p in (a, b)
               for x, lo, hi in zip(p, self.bounds.minimum, self.bounds.maximum)):
            return False
        return not any(obstacle.intersects_segment(a, b, safety_margin) for obstacle in self.obstacles)


@dataclass
class Path3D:
    """Ordered polyline in continuous 3-D space."""

    waypoints: list[Point3D]

    def __post_init__(self) -> None:
        self.waypoints = [_point(p) for p in self.waypoints]
        if len(self.waypoints) < 2:
            raise ValueError("a path requires at least a start and an end point")

    @property
    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.waypoints, self.waypoints[1:]))

    def collision_count(self, environment: Environment3D, step: float = 0.25, safety_margin: float = 0.0) -> int:
        return sum(
            not environment.segment_is_free(a, b, step=step, safety_margin=safety_margin)
            for a, b in zip(self.waypoints, self.waypoints[1:])
        )

    def as_flat(self) -> list[float]:
        return [coordinate for point in self.waypoints for coordinate in point]
