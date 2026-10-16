"""Multi-start projected gradient descent for chance-constrained path planning.

This project-defined, non-evolutionary local-search baseline uses length
descent and tangent projection against one sampled risk constraint. It is not
an implementation of a complete Rosen active-set solver, SQP or SLSQP: it has
no nominal-obstacle feasibility restoration or general KKT stopping rule.
The evaluation budget, problem definition and conditional Gaussian risk
estimator match the evolutionary candidates when configured identically.

No population or mutation is used. It tests whether global evolutionary search
(DE/L-SHADE) helps this particular local heuristic. Poor performance cannot
establish that global evolutionary search is mathematically indispensable.
"""
from __future__ import annotations
import math
import random
from typing import Sequence
from .continuous_planner import ContinuousPathProblem
from .risk_aware_de import RiskAwareDEConfig, RiskAwareDEResult, RiskEvaluation
from .conditional_risk_de import ConditionalRiskDE
from .conditional_projected_polish import length_gradient_normalized


class MultiStartProjectedGradient:
    """Custom multi-start tangent-projected descent; no feasibility certificate."""

    def __init__(
        self,
        problem: ContinuousPathProblem,
        config: RiskAwareDEConfig,
        *,
        restarts: int = 4,
        active_margin: float = 0.02,
        initial_step: float = 0.04,
        axis: int = 2,
    ):
        self.problem = problem
        self.config = config
        if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
            raise ValueError('restarts must be a positive integer')
        if not math.isfinite(active_margin) or not 0 <= active_margin <= 1:
            raise ValueError('active_margin must be finite and lie in [0, 1]')
        if not math.isfinite(initial_step) or not 0 < initial_step <= 1:
            raise ValueError('initial_step must be finite and lie in (0, 1]')
        if config.risk_gate_schedule != 'fixed':
            raise ValueError('Fixed gate required by the projected-gradient baseline')
        self.restarts = restarts
        self.active_margin = active_margin
        self.initial_step = initial_step
        self.axis = axis
        self.evaluator = ConditionalRiskDE(problem, config, axis=axis)
        self.extension_metadata = {
            "algorithm": "multi_start_projected_gradient",
            "restarts": self.restarts,
            "axis": axis,
            "risk_evaluation_type": "conditional_gaussian_probability",
            "baseline_status": "project-defined tangent-descent heuristic, not a canonical SQP or Rosen solver",
            "nominal_feasibility_restoration": False,
            "safety_certificate": False,
        }

    def _sample_initial_vector(self, restart_index: int, rng: random.Random) -> list[float]:
        """Generate diverse starting points: straight chord, perturbed, and random."""
        start, goal = self.problem.start, self.problem.goal
        n_waypoints = self.problem.intermediate_waypoints
        vector = []

        if restart_index == 0:
            # Straight-line chord
            for i in range(1, n_waypoints + 1):
                frac = i / (n_waypoints + 1)
                for d in range(3):
                    lo, hi = self.problem.bounds[3 * (i - 1) + d]
                    val = start[d] + frac * (goal[d] - start[d])
                    vector.append(max(lo, min(hi, val)))
        elif restart_index == 1:
            # Perturbed straight-line chord with high z (altitude bias)
            for i in range(1, n_waypoints + 1):
                frac = i / (n_waypoints + 1)
                for d in range(3):
                    lo, hi = self.problem.bounds[3 * (i - 1) + d]
                    val = start[d] + frac * (goal[d] - start[d])
                    if d == 2:  # z-axis boost to test altitude bypass
                        val += 0.3 * (hi - lo)
                    else:
                        val += rng.uniform(-0.15, 0.15) * (hi - lo)
                    vector.append(max(lo, min(hi, val)))
        else:
            # Uniform random points in bounding box (standard DE initialization)
            for lo, hi in self.problem.bounds:
                vector.append(rng.uniform(lo, hi))
        return vector

    def optimize(self) -> RiskAwareDEResult:
        cfg = self.config
        total_budget = cfg.base.max_evaluations or cfg.base.population_size * (cfg.base.generations + 1)
        budget_per_restart = total_budget // self.restarts

        planner_seed = cfg.base.seed if cfg.base.seed is not None else 0
        rng = random.Random(planner_seed)

        global_best_vector: list[float] | None = None
        global_best_record: RiskEvaluation | None = None
        total_evaluations = 0
        history: list[float] = []
        key_history: list[tuple[float, ...]] = []
        gate_history: list[float] = []

        widths = [hi - lo for lo, hi in self.problem.bounds]

        def eval_candidate(v: Sequence[float]) -> RiskEvaluation:
            nonlocal total_evaluations
            total_evaluations += 1
            return self.evaluator._evaluate(v, cfg.chance_alpha)

        def record_trace(rec: RiskEvaluation):
            history.append(rec.objective)
            key_history.append(rec.key)
            gate_history.append(cfg.chance_alpha)

        for restart in range(self.restarts):
            if total_evaluations >= total_budget:
                break
            restart_budget_end = min(total_budget, total_evaluations + budget_per_restart)
            if restart == self.restarts - 1:
                restart_budget_end = total_budget

            vector = self._sample_initial_vector(restart, rng)
            record = eval_candidate(vector)
            record_trace(record)

            if global_best_record is None or record.key < global_best_record.key:
                global_best_vector = list(vector)
                global_best_record = record

            step = self.initial_step

            while total_evaluations < restart_budget_end:
                # 1. Negative length gradient (straightening direction)
                direction = [-x for x in length_gradient_normalized(self.problem, vector)]
                projected = False

                # 2. Tangent projection if near or violating the chance gate
                if (record.perturbed_collision_rate >= cfg.chance_alpha - self.active_margin
                        and restart_budget_end - total_evaluations > len(vector)):
                    risk_gradient = []
                    for idx, (lo, hi) in enumerate(self.problem.bounds):
                        probe = list(vector)
                        delta = 0.001 * widths[idx]
                        probe[idx] = min(hi, vector[idx] + delta)
                        if probe[idx] == vector[idx]:
                            probe[idx] = max(lo, vector[idx] - delta)
                        chg = (probe[idx] - vector[idx]) / widths[idx]
                        measured = eval_candidate(probe)
                        record_trace(record)
                        risk_gradient.append(
                            (measured.perturbed_collision_rate - record.perturbed_collision_rate) / chg
                        )
                        if total_evaluations >= restart_budget_end:
                            break

                    normal2 = sum(x * x for x in risk_gradient)
                    outward = sum(x * y for x, y in zip(direction, risk_gradient))
                    if normal2 > 1e-18 and outward > 0:
                        direction = [d - outward * g / normal2 for d, g in zip(direction, risk_gradient)]
                        projected = True

                norm = math.sqrt(sum(x * x for x in direction))
                if norm < 1e-15:
                    direction = [0.0] * len(vector)
                    direction[total_evaluations % len(vector)] = 1.0
                    norm = 1.0
                direction = [x / norm for x in direction]

                # 3. Line search along direction
                accepted = False
                for _ in range(8):
                    if total_evaluations >= restart_budget_end:
                        break
                    proposal = [
                        max(lo, min(hi, x + step * d * w))
                        for x, d, w, (lo, hi) in zip(vector, direction, widths, self.problem.bounds)
                    ]
                    measured = eval_candidate(proposal)
                    # Use standard lexicographic acceptance
                    if measured.key < record.key or (
                        measured.nominal_feasible and record.nominal_feasible
                        and measured.chance_violation <= record.chance_violation
                        and measured.objective < record.objective
                    ):
                        vector, record = proposal, measured
                        accepted = True
                        step = min(self.initial_step, step * 1.2)
                        record_trace(record)
                        if record.key < global_best_record.key:
                            global_best_vector = list(vector)
                            global_best_record = record
                        break
                    else:
                        step = max(1e-6, step * 0.5)
                        record_trace(record)

                if not accepted and step <= 1e-5:
                    step = self.initial_step * 0.25

        # Safety fallback
        if global_best_vector is None or global_best_record is None:
            vector = self._sample_initial_vector(0, rng)
            global_best_record = eval_candidate(vector)
            global_best_vector = vector

        # Pad evaluations and history to total_budget if early exit
        while total_evaluations < total_budget:
            eval_candidate(global_best_vector)
            record_trace(global_best_record)

        self.extension_metadata.update(self.evaluator.extension_metadata)
        self.extension_metadata['nominal_nfe'] = total_evaluations
        return RiskAwareDEResult(
            best_vector=global_best_vector,
            best_fitness=global_best_record.objective,
            history=history[:total_budget],
            evaluations=total_evaluations,
            internal_collision_checks=self.evaluator._internal_collision_checks,
            best_evaluation=global_best_record,
            key_history=key_history[:total_budget],
            gate_history=gate_history[:total_budget],
        )


def create_optimizer(problem, base_config, options):
    opts = dict(options)
    restarts = opts.pop("restarts", 4)
    active_margin = opts.pop("active_margin", 0.02)
    initial_step = opts.pop("initial_step", 0.04)
    axis = opts.pop("axis", 2)
    if opts:
        raise ValueError(f'Unknown projected-gradient options: {opts}')
    return MultiStartProjectedGradient(
        problem, base_config,
        restarts=restarts,
        active_margin=active_margin,
        initial_step=initial_step,
        axis=axis,
    )
