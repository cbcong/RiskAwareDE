"""Budget-reserved incumbent pattern search after risk-aware DE.

This experimental extension allocates, rather than adds, function evaluations.
The global phase uses the unchanged optimizer.  The final phase applies a
deterministic set of geometric directions to the current incumbent, keeping a
separate trust radius for each waypoint/direction. Rejection contracts that
radius; success expands it up to its initial size. Every proposal, including a
rejected one, receives the usual nominal and complete frozen-bank evaluation.

This is a derivative-free constrained local-search composition, not a claim of
a new general pattern-search algorithm or a probabilistic safety certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

from .continuous_planner import ContinuousPathProblem
from .risk_aware_de import RiskAwareDE, RiskAwareDEConfig, RiskAwareDEResult


@dataclass(frozen=True)
class PolishOptions:
    reserve_fraction: float = 0.20
    initial_fraction: float = 0.70
    coordinate_fraction: float = 0.025
    shrink: float = 0.5
    grow: float = 1.2
    minimum_fraction: float = 1e-6
    preserve_empirical_risk: bool = False
    coordinate_mode: str = 'signed_gradient'
    allocation: str = 'cyclic'
    direction_set: str = 'all'

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name in {'coordinate_mode', 'allocation', 'direction_set'}:
                continue
            if name == "preserve_empirical_risk":
                if not isinstance(self.preserve_empirical_risk, bool):
                    raise ValueError("preserve_empirical_risk must be boolean")
                continue
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.reserve_fraction < 1.0:
            raise ValueError("reserve_fraction must be in (0, 1)")
        if not 0.0 < self.initial_fraction <= 1.0:
            raise ValueError("initial_fraction must be in (0, 1]")
        if not 0.0 < self.coordinate_fraction <= 1.0:
            raise ValueError("coordinate_fraction must be in (0, 1]")
        if not 0.0 < self.shrink < 1.0 or self.grow <= 1.0:
            raise ValueError("shrink must be in (0, 1), grow must exceed 1")
        if not 0.0 < self.minimum_fraction <= min(self.initial_fraction, self.coordinate_fraction):
            raise ValueError("minimum_fraction must not exceed initial radii")
        if self.coordinate_mode not in {'signed_gradient', 'exact_minimizer'}:
            raise ValueError('unsupported coordinate_mode')
        if self.allocation not in {'cyclic', 'geometric_priority'}:
            raise ValueError('unsupported allocation')
        if self.direction_set not in {'all', 'chord_only', 'coordinates_only'}:
            raise ValueError('unsupported direction_set')


class RiskBudgetedPolish:
    """An incumbent-based local phase with fully charged evaluation costs."""

    def __init__(self, problem: ContinuousPathProblem, base_config: RiskAwareDEConfig,
                 options: PolishOptions | None = None):
        self.problem = problem
        self.config = base_config
        self.options = options or PolishOptions()
        if not problem.bounds:
            raise ValueError("budgeted polish requires at least one intermediate waypoint")
        if base_config.risk_gate_schedule != "fixed":
            raise ValueError("budgeted polish currently requires a fixed empirical risk gate")
        self.metadata: dict[str, object] = {}
        self.extension_metadata: dict[str, object] = {}

    def _proposal(self, vector: Sequence[float], waypoint: int,
                  direction: int, fraction: float) -> list[float]:
        """Use only geometry here; all feasibility/risk calls occur afterwards.

        Direction 0 contracts toward the nearest point of the neighbouring
        chord. Direction 1 follows the negative local length gradient. The
        remaining directions use one component of that gradient, permitting
        movement along a constraint when a direct chord crosses an obstacle.
        """
        points = self.problem.decode(vector).waypoints
        before, current, after = points[waypoint:waypoint + 3]
        gradient = [0.0, 0.0, 0.0]
        for neighbour in (before, after):
            distance = math.dist(current, neighbour)
            if distance > 1e-12:
                for axis in range(3):
                    gradient[axis] += (current[axis] - neighbour[axis]) / distance
        widths = [hi - lo for lo, hi in self.problem.bounds[3 * waypoint:3 * waypoint + 3]]
        if direction == 0:
            chord = [b - a for a, b in zip(before, after)]
            squared_norm = sum(value * value for value in chord)
            position = (sum((v - a) * d for v, a, d in zip(current, before, chord)) / squared_norm
                        if squared_norm > 1e-24 else 0.5)
            position = min(1.0, max(0.0, position))
            target = [a + position * d for a, d in zip(before, chord)]
            replacement = [v + fraction * (t - v) for v, t in zip(current, target)]
        elif direction == 1:
            norm = math.sqrt(sum(value * value for value in gradient))
            replacement = [v - fraction * width * component / max(norm, 1e-12)
                           for v, width, component in zip(current, widths, gradient)]
        else:
            axis = direction - 2
            replacement = list(current)
            if self.options.coordinate_mode == 'exact_minimizer':
                from .risk_axis_geometry import coordinate_minimizer
                target = coordinate_minimizer(before, current, after, axis)
                replacement[axis] += fraction * (target - replacement[axis])
            else:
                sign = 1.0 if gradient[axis] > 0.0 else -1.0 if gradient[axis] < 0.0 else 0.0
                replacement[axis] -= fraction * widths[axis] * sign
        output = list(vector)
        output[3 * waypoint:3 * waypoint + 3] = replacement
        return [max(lo, min(hi, value)) for value, (lo, hi) in zip(output, self.problem.bounds)]

    def _create_evaluator(self, config: RiskAwareDEConfig) -> RiskAwareDE:
        return RiskAwareDE(self.problem.bounds, self.problem, config)

    def optimize(self) -> RiskAwareDEResult:
        cfg, opt = self.config, self.options
        total_budget = cfg.base.max_evaluations
        if total_budget is None:
            total_budget = cfg.base.population_size * (cfg.base.generations + 1)
        reserved = max(1, int(round(total_budget * opt.reserve_fraction)))
        global_budget = max(cfg.base.population_size, total_budget - reserved)
        global_cfg = replace(cfg, base=replace(cfg.base, max_evaluations=global_budget))
        evaluator = self._create_evaluator(global_cfg)
        result = evaluator.optimize()
        global_used = result.evaluations
        global_best = result.best_fitness
        global_rate = result.best_evaluation.perturbed_collision_rate
        vector, record = list(result.best_vector), result.best_evaluation
        allowed_directions = {'all': range(5), 'chord_only': (0,), 'coordinates_only': (2,3,4)}[opt.direction_set]
        directions = [(waypoint, direction) for waypoint in range(self.problem.intermediate_waypoints)
                      for direction in allowed_directions]
        initial = [opt.initial_fraction if direction == 0 else opt.coordinate_fraction
                   for _, direction in directions]
        radii = list(initial)
        events: list[dict[str, float | int | bool]] = []
        accepted_count = 0
        while result.evaluations < total_budget:
            slot = len(events) % len(directions)
            if opt.allocation == 'geometric_priority' and len(events) % 4 != 0:
                # Rank pure geometric proposals at their current trust radii.
                # Every fourth attempt remains cyclic. Rejected directions
                # shrink, lowering their attainable geometric improvement.
                slot = min(range(len(directions)), key=lambda index: self.problem.decode(
                    self._proposal(vector, *directions[index], radii[index])).length)
            waypoint, direction = directions[slot]
            radius = radii[slot]
            proposal = self._proposal(vector, waypoint, direction, radius)
            evaluation = evaluator._evaluate(proposal, cfg.chance_alpha)
            result.evaluations += 1
            allowed = evaluator._refinement_allowed(evaluation, record, cfg.chance_alpha)
            # Geometric polishing is deliberately monotone in the nominal
            # objective, even if a feasibility/risk repair could rank better.
            accepted = (evaluation.objective <= record.objective
                        and allowed and evaluator._accept_trial(evaluation, record, refinement=True))
            if opt.preserve_empirical_risk:
                accepted = accepted and evaluation.perturbed_collision_rate <= record.perturbed_collision_rate
            previous = record.objective
            if accepted:
                vector, record = proposal, evaluation
                accepted_count += 1
                radii[slot] = min(initial[slot], radius * opt.grow)
            else:
                radii[slot] = max(opt.minimum_fraction, radius * opt.shrink)
            events.append({
                "nfe": result.evaluations, "waypoint": waypoint, "direction": direction,
                "radius": radius, "next_radius": radii[slot],
                "nominal_feasible": evaluation.nominal_feasible,
                "risk_guard_pass": allowed, "accepted": accepted,
                "proposed_objective_gain": previous - evaluation.objective,
                "accepted_objective_gain": previous - record.objective,
                "perturbed_collision_rate": evaluation.perturbed_collision_rate,
            })
            result.history.append(record.objective)
            result.key_history.append(record.key)
            result.gate_history.append(cfg.chance_alpha)
            result.mutation_factor_history.append(result.mutation_factor_history[-1])
            result.crossover_rate_history.append(result.crossover_rate_history[-1])
        result.best_vector = vector
        result.best_evaluation = record
        result.best_fitness = record.objective
        result.internal_collision_checks = evaluator._internal_collision_checks
        result.local_refinement_events.extend(events)
        result.local_refinement_stats.update({
            "polish_global_evaluations": global_used,
            "polish_evaluations": len(events), "polish_accepted": accepted_count,
            "polish_global_best_objective": global_best,
            "polish_global_collision_rate": global_rate,
            "polish_objective_improvement": global_best - result.best_fitness,
        })
        self.metadata = {
            "mechanism": "budget_reserved_incumbent_pattern_search",
            "reserve_fraction": opt.reserve_fraction, "global_evaluations": global_used,
            "preserve_empirical_risk": opt.preserve_empirical_risk,
            "polish_evaluations": len(events), "polish_accepted": accepted_count,
            "frozen_bank_reused": True, "additional_uncharged_collision_checks": 0,
            "coordinate_mode": opt.coordinate_mode, "allocation": opt.allocation,
            "direction_set": opt.direction_set,
            "directions": ["chord_projection", "negative_gradient", "coordinate_x", "coordinate_y", "coordinate_z"],
        }
        self.extension_metadata = dict(self.metadata)
        return result


def create_optimizer(problem: ContinuousPathProblem, base_config: RiskAwareDEConfig,
                     options: dict | None = None) -> RiskBudgetedPolish:
    return RiskBudgetedPolish(problem, base_config, PolishOptions(**(options or {})))
