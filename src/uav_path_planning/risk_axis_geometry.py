"""Coordinate-restricted local shortening for the risk-aware DE optimizer.

The existing chord operator changes all three coordinates of a waypoint. This
extension can retain one or two detour coordinates while optimizing the others.
It uses geometry alone to propose one trial; the inherited nominal/risk
evaluation and comparator remain the only authority for acceptance.

For fixed coordinates other than j, let A and B be the perpendicular distances
to neighbors a and b. The minimizer of the adjacent length is
    x_j = (B * a_j + A * b_j) / (A + B).
If A+B=0, any coordinate between a_j and b_j minimizes the length. Moving a
fraction in [0,1] toward this minimizer cannot increase length by convexity.
Applying two such updates sequentially retains this property. This elementary
coordinate descent observation is not asserted to be a new mathematical result.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from .continuous_planner import ContinuousPathProblem
from .risk_aware_de import RiskAwareDE, RiskAwareDEConfig, RiskAwareDEResult, RiskEvaluation


def coordinate_minimizer(
    previous: Sequence[float], point: Sequence[float], following: Sequence[float], axis: int,
) -> float:
    """Return an exact unconstrained minimizer of adjacent length on one axis."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    left_radius = math.sqrt(sum((point[j] - previous[j]) ** 2 for j in range(3) if j != axis))
    right_radius = math.sqrt(sum((point[j] - following[j]) ** 2 for j in range(3) if j != axis))
    total = left_radius + right_radius
    if total == 0.0:
        lo, hi = sorted((previous[axis], following[axis]))
        return min(hi, max(lo, float(point[axis])))
    fraction = left_radius / total
    return float(previous[axis] + fraction * (following[axis] - previous[axis]))


class AxisGeometryRiskAwareDE(RiskAwareDE):
    """RiskAwareDE with an explicitly budgeted coordinate-shortening proposal."""

    def __init__(
        self, problem: ContinuousPathProblem, base_config: RiskAwareDEConfig, *,
        axis_count: int = 1, axis_scale: float = 1.0, chord_probability: float = 0.0,
    ):
        if isinstance(axis_count, bool) or axis_count not in (1, 2):
            raise ValueError("axis_count must be 1 or 2")
        if not math.isfinite(axis_scale) or not 0.0 < axis_scale <= 1.0:
            raise ValueError("axis_scale must lie in (0,1]")
        if not math.isfinite(chord_probability) or not 0.0 <= chord_probability <= 1.0:
            raise ValueError("chord_probability must lie in [0,1]")
        if base_config.local_refinement_mode != "partial_shortcut":
            raise ValueError("axis extension requires a partial_shortcut parent configuration")
        super().__init__(problem.bounds, problem, base_config)
        self.axis_count = axis_count
        self.axis_scale = axis_scale
        self.chord_probability = chord_probability
        self.events: list[dict[str, float | int | bool]] = []
        self._pending_event: dict[str, float | int | bool] | None = None
        self.extension_metadata = {
            "mechanism": "coordinate_adjacent_length_minimization",
            "axis_count": axis_count, "axis_scale": axis_scale,
            "chord_probability": chord_probability,
            "additional_collision_checks_per_proposal": 0,
            "additional_objective_evaluations_per_proposal": 0,
            "geometry_guarantee": "local polyline length nonincrease; not a safety guarantee",
        }

    def _refine_toward_direct(
        self, vector: Sequence[float], rng: random.Random, *, scale: float | None = None,
    ) -> list[float]:
        if not self.config.local_refinement or self.problem.intermediate_waypoints < 1:
            return list(vector)
        chord = self.chord_probability >= 1.0 or (
            self.chord_probability > 0.0 and rng.random() < self.chord_probability
        )
        if chord:
            output = super()._refine_toward_direct(vector, rng, scale=scale)
            axis_mask = 7
        else:
            waypoint = rng.randrange(self.problem.intermediate_waypoints)
            previous = self.problem.start if waypoint == 0 else vector[3 * (waypoint - 1):3 * waypoint]
            following = self.problem.goal if waypoint == self.problem.intermediate_waypoints - 1 else vector[3 * (waypoint + 1):3 * (waypoint + 2)]
            point = list(vector[3 * waypoint:3 * waypoint + 3])
            axes = rng.sample(range(3), self.axis_count)
            for axis in axes:
                minimizer = coordinate_minimizer(previous, point, following, axis)
                value = point[axis] + self.axis_scale * (minimizer - point[axis])
                lo, hi = self.bounds[3 * waypoint + axis]
                point[axis] = min(hi, max(lo, value))
            output = list(vector)
            output[3 * waypoint:3 * waypoint + 3] = point
            axis_mask = sum(1 << axis for axis in axes)
        gain = self.problem.decode(vector).length - self.problem.decode(output).length
        self._pending_event = {
            "proposal": len(self.events), "chord": chord, "axis_mask": axis_mask,
            "proposed_length_gain": gain, "nominal_feasible": False,
            "risk_guard_pass": False, "accepted": False, "accepted_length_gain": 0.0,
        }
        self.events.append(self._pending_event)
        return output

    def _refinement_allowed(
        self, candidate: RiskEvaluation, incumbent: RiskEvaluation, chance_alpha: float | None = None,
    ) -> bool:
        allowed = super()._refinement_allowed(candidate, incumbent, chance_alpha)
        if self._pending_event is not None:
            self._pending_event["nominal_feasible"] = candidate.nominal_feasible
            self._pending_event["risk_guard_pass"] = allowed
        return allowed

    def _accept_trial(
        self, candidate: RiskEvaluation, incumbent: RiskEvaluation, *, refinement: bool = False,
    ) -> bool:
        accepted = super()._accept_trial(candidate, incumbent, refinement=refinement)
        if refinement and self._pending_event is not None:
            self._pending_event["accepted"] = accepted
            self._pending_event["accepted_length_gain"] = self._pending_event["proposed_length_gain"] if accepted else 0.0
        return accepted

    def optimize(self) -> RiskAwareDEResult:
        self.events = []
        self._pending_event = None
        result = super().optimize()
        statistics = {
            "proposals": len(self.events),
            "axis_proposals": sum(not bool(row["chord"]) for row in self.events),
            "chord_proposals": sum(bool(row["chord"]) for row in self.events),
            "nominal_feasible": sum(bool(row["nominal_feasible"]) for row in self.events),
            "risk_guard_pass": sum(bool(row["risk_guard_pass"]) for row in self.events),
            "accepted": sum(bool(row["accepted"]) for row in self.events),
            "accepted_length_gain_sum": sum(float(row["accepted_length_gain"]) for row in self.events),
        }
        result.local_refinement_stats = statistics
        result.local_refinement_events = self.events
        self.extension_metadata["refinement_statistics"] = statistics
        return result


def create_optimizer(
    problem: ContinuousPathProblem, base_config: RiskAwareDEConfig, options: dict,
) -> AxisGeometryRiskAwareDE:
    """Plugin entry point for the shared complete-scene experiment runner."""
    allowed = {"axis_count", "axis_scale", "chord_probability"}
    unexpected = set(options) - allowed
    if unexpected:
        raise ValueError(f"unknown axis geometry options: {sorted(unexpected)}")
    return AxisGeometryRiskAwareDE(problem, base_config, **options)
