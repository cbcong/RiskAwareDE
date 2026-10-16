"""Research candidates that change different parts of the locked planner.

These are experimental mechanisms, not claims of proven novelty or safety.
Every optimizer evaluation uses the existing NFE accounting.  The candidates
are intentionally configured by a compact, serializable option dictionary so
the frozen full-run protocol can preserve their exact definitions.
"""
from __future__ import annotations

from dataclasses import replace
import math
import random
from typing import Sequence

from .conditional_projected_polish import ConditionalProjectedPolish
from .conditional_risk_de import conditional_path_collision
from .models import Environment3D, ObstacleBox
from .risk_aware_de import RiskAwareDE, RiskAwareDEConfig, RiskAwareDEResult, RiskEvaluation


def _environment_from_shifts(problem, shifts):
    obstacles = [
        ObstacleBox(tuple(a + d for a, d in zip(box.minimum, shift)),
                    tuple(b + d for b, d in zip(box.maximum, shift)))
        for box, shift in zip(problem.environment.obstacles, shifts)
    ]
    return Environment3D(problem.environment.bounds, obstacles)


def _distance_to_box(point, box):
    closest = tuple(min(hi, max(lo, x)) for x, lo, hi in zip(point, box.minimum, box.maximum))
    delta = tuple(x - y for x, y in zip(point, closest))
    distance = math.sqrt(sum(x * x for x in delta))
    if distance > 1e-12:
        return distance, tuple(x / distance for x in delta)
    # A point inside the closed box is pushed through its nearest face.
    choices = []
    for axis, (x, lo, hi) in enumerate(zip(point, box.minimum, box.maximum)):
        choices.append((x - lo, axis, -1.0))
        choices.append((hi - x, axis, 1.0))
    face_distance, axis, sign = min(choices)
    direction = [0.0, 0.0, 0.0]
    direction[axis] = sign
    return -face_distance, tuple(direction)


def _pathwise_collision_ledger(path, environments, step, safety_margin):
    """Estimate full-path collision probability and allocate it across hit segments.

    A perturbed map is one joint scenario. If several segments collide in one
    map, that scenario's unit event mass is divided evenly among them; segment
    contributions therefore sum exactly to the full-path collision rate.
    """
    segment_count = len(path.waypoints) - 1
    allocation = [0.0] * segment_count
    path_hits = 0
    for environment in environments:
        hit_segments = [
            not environment.segment_is_free(a, b, step=step, safety_margin=safety_margin)
            for a, b in zip(path.waypoints, path.waypoints[1:])
        ]
        count = sum(hit_segments)
        if count:
            path_hits += 1
            share = 1.0 / count
            allocation = [value + share if hit else value
                          for value, hit in zip(allocation, hit_segments)]
    denominator = max(1, len(environments))
    return path_hits / denominator, [value / denominator for value in allocation]


class ResearchRiskDE(RiskAwareDE):
    """Risk-aware search with selectable, independently testable mechanisms."""

    def __init__(
        self, problem, config: RiskAwareDEConfig, *, initialization="uniform",
        mutation="standard", risk_model="conditional_mean", risk_weight=0.5,
        beta=0.05,
    ):
        if initialization not in {"uniform", "topology_seeds"}:
            raise ValueError("unsupported candidate initialization")
        if mutation not in {"standard", "risk_localized"}:
            raise ValueError("unsupported candidate mutation")
        if risk_model not in {"conditional_mean", "conditional_tail", "uncertainty_score", "common_translation", "distribution_mixture", "pathwise_ledger"}:
            raise ValueError("unsupported candidate risk model")
        if not math.isfinite(risk_weight) or risk_weight < 0:
            raise ValueError("risk_weight must be finite and non-negative")
        if config.skip_risk_for_nominal_infeasible:
            raise ValueError("candidate suite requires a fixed K risk score for every NFE")
        if risk_model == "distribution_mixture" and config.perturbation_samples < 3:
            raise ValueError("distribution_mixture requires at least one sample per shift family")
        if risk_model == "uncertainty_score" and config.selection_mode != "ucb_budgeted":
            config = replace(config, selection_mode="ucb_budgeted")
        super().__init__(problem.bounds, problem, config)
        self.problem = problem
        self.initialization = initialization
        self.mutation = mutation
        self.risk_model = risk_model
        self.risk_weight = float(risk_weight)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        self.beta = float(beta)
        self.length_scale = math.dist(problem.start, problem.goal)
        self.axis = 2
        self._sample_shifts: list[list[tuple[float, float, float]]] = []
        self._model_groups: dict[str, list[Environment3D]] = {}
        self._ledger_by_vector = {}
        if risk_model in {"conditional_mean", "conditional_tail", "uncertainty_score", "pathwise_ledger"}:
            for environment in self._perturbed_banks[0]:
                self._sample_shifts.append([
                    tuple(moved.minimum[d] - original.minimum[d]
                          for d in range(3))
                    for original, moved in zip(problem.environment.obstacles, environment.obstacles)
                ])
        else:
            self._build_alternative_bank()
        self.extension_metadata = {
            "candidate_family": "research_mechanism_suite",
            "initialization": initialization,
            "mutation": mutation,
            "risk_model": risk_model,
            "risk_weight": risk_weight,
            "risk_score_semantics": {
                "conditional_mean": "mean conditional Gaussian path-collision probability",
                "conditional_tail": "mean of the largest quarter of conditional probabilities; heuristic tail score",
                "uncertainty_score": "mean and mean-plus-weighted-standard-deviation gate score; not a confidence certificate",
                "common_translation": "binary path collision rate when all obstacles share one Gaussian translation",
                "distribution_mixture": "worst empirical collision rate among Gaussian, uniform, and standardized Student-t3 sub-banks",
                "pathwise_ledger": "full-path collision event rate with unit event mass allocated across colliding segments",
            }[risk_model],
            "risk_samples_per_evaluation": config.perturbation_samples,
            "distribution_mixture_group_counts": ({name: len(group) for name, group in self._model_groups.items()}
                                                    if risk_model == "distribution_mixture" else None),
            "soft_risk_beta": self.beta,
            "evaluation_budget_note": "each risk sample corresponds to one frozen full-map perturbation; total checks are counted explicitly",
        }

    def _retarget(self, evaluation, chance_alpha):
        record = super()._retarget(evaluation, chance_alpha)
        if self.config.selection_mode == "ucb_budgeted":
            # The upper score, rather than the sample mean, is the operative
            # gate for this exploratory uncertainty-aware candidate.
            violation = max(0.0, record.risk_upper_bound - chance_alpha)
            record = replace(record, chance_violation=violation)
        if (self.beta and record.nominal_feasible and record.chance_violation <= 0.0):
            # Keep the locked main method's in-gate soft-risk preference in
            # every candidate; the candidate treatment is the declared
            # search/risk/terminal mechanism, not an accidental beta change.
            cost = record.objective + self.beta * self.length_scale * record.perturbed_collision_rate
            return replace(record, key=(*record.key[:-1], cost))
        return record

    def risk_gate_score(self, record):
        return record.risk_upper_bound if self.config.selection_mode == "ucb_budgeted" else record.perturbed_collision_rate

    def _refinement_allowed(self, candidate, incumbent, chance_alpha=None):
        if self.config.selection_mode != "ucb_budgeted":
            return super()._refinement_allowed(candidate, incumbent, chance_alpha)
        if not candidate.nominal_feasible or not incumbent.nominal_feasible:
            return candidate.nominal_feasible and not incumbent.nominal_feasible
        alpha = self.config.chance_alpha if chance_alpha is None else float(chance_alpha)
        candidate_score = self.risk_gate_score(candidate)
        incumbent_score = self.risk_gate_score(incumbent)
        if incumbent_score <= alpha and candidate_score > alpha:
            return False
        if incumbent_score > alpha and candidate_score <= alpha:
            return True
        if self.config.local_refinement_mode == "safe_shortcut":
            return candidate_score <= incumbent_score + 1e-15
        return max(0.0, candidate_score - alpha) <= max(0.0, incumbent_score - alpha) + 1e-15

    def _build_alternative_bank(self):
        cfg = self.config
        sigma = cfg.perturbation_sigma
        count = cfg.perturbation_samples
        seed = cfg.perturbation_seed
        if seed is None:
            seed = (cfg.base.seed or 0) ^ 0x5EEDC0DE
        rng = random.Random(seed)
        if self.risk_model == "common_translation":
            envs = []
            for _ in range(count):
                shift = tuple(rng.gauss(0.0, sigma) for _ in range(3))
                envs.append(_environment_from_shifts(self.problem, [shift] * len(self.problem.environment.obstacles)))
            self._model_groups = {"common_gaussian": envs}
        else:
            # A balanced fixed allocation makes this a distribution-shift
            # stress objective. It is deliberately a score, not a certificate.
            group_names = ("gaussian", "uniform", "student_t3")
            allocations = {name: [] for name in group_names}
            normalizer = math.sqrt(3.0)
            for index in range(count):
                model = group_names[index % len(group_names)]
                shifts = []
                for _ in self.problem.environment.obstacles:
                    if model == "gaussian":
                        shift = tuple(rng.gauss(0.0, sigma) for _ in range(3))
                    elif model == "uniform":
                        shift = tuple(sigma * rng.uniform(-normalizer, normalizer) for _ in range(3))
                    else:
                        # A variance-one Student-t with three degrees of freedom.
                        shift = tuple(sigma * rng.gauss(0.0, 1.0) / math.sqrt(rng.gammavariate(1.5, 2.0)) for _ in range(3))
                    shifts.append(shift)
                allocations[model].append(_environment_from_shifts(self.problem, shifts))
            self._model_groups = {name: values for name, values in allocations.items() if values}
        self._perturbed_banks = [sum(self._model_groups.values(), [])]
        self._perturbed = list(self._perturbed_banks[0])

    def _initial_population(self, rng, size, dim):
        population = super()._initial_population(rng, size, dim)
        if self.initialization != "topology_seeds" or self.problem.intermediate_waypoints < 1:
            return population
        start, goal = self.problem.start, self.problem.goal
        blockers = [box for box in self.problem.environment.obstacles
                    if box.intersects_segment(start, goal, self.problem.safety_margin)]
        if not blockers:
            blockers = list(self.problem.environment.obstacles)
        if not blockers:
            return population
        axis_directions = [(axis, sign) for axis in range(3) for sign in (-1.0, 1.0)]
        dimensions = [hi - lo for lo, hi in self.bounds]
        for slot, (axis, sign) in enumerate(axis_directions[:max(0, size - 1)], start=1):
            obstacle = blockers[(slot - 1) % len(blockers)]
            center = tuple((lo + hi) * .5 for lo, hi in zip(obstacle.minimum, obstacle.maximum))
            line = tuple(g - s for s, g in zip(start, goal))
            denom = sum(x * x for x in line) or 1.0
            t = min(1.0, max(0.0, sum((c - s) * d for c, s, d in zip(center, start, line)) / denom))
            extent = obstacle.maximum[axis] - obstacle.minimum[axis]
            amplitude = max(1.0, .5 * extent + .65)
            vector = []
            for waypoint in range(self.problem.intermediate_waypoints):
                fraction = (waypoint + 1) / (self.problem.intermediate_waypoints + 1)
                point = [s + fraction * d for s, d in zip(start, line)]
                envelope = max(0.0, 1.0 - abs(fraction - t) / .55)
                point[axis] += sign * amplitude * envelope
                for coordinate, (lo, hi) in zip(point, self.bounds[3 * waypoint:3 * waypoint + 3]):
                    vector.append(max(lo, min(hi, coordinate)))
            population[slot] = vector
        self.extension_metadata["topology_seed_count"] = min(len(axis_directions), max(0, size - 1))
        self.extension_metadata["topology_seed_rule"] = "six axis-side detours around nominal straight-line blockers; geometry-only initialization"
        return population

    def _risk_local_waypoint(self, vector):
        count = self.problem.intermediate_waypoints
        if count < 1 or not self.problem.environment.obstacles:
            return None
        if self.risk_model == "pathwise_ledger":
            segment_mass = self._ledger_by_vector.get(tuple(vector))
            if segment_mass:
                waypoint_mass = [segment_mass[index] + segment_mass[index + 1]
                                 for index in range(count)]
                if max(waypoint_mass, default=0.0) > 0.0:
                    return max(range(count), key=lambda index: waypoint_mass[index])
        scores = []
        for index in range(count):
            point = tuple(vector[3 * index:3 * index + 3])
            clearance = min(box.segment_distance(point, point, self.problem.safety_margin)
                            for box in self.problem.environment.obstacles)
            scores.append(1.0 / max(clearance, .05))
        return max(range(count), key=lambda i: scores[i])

    def _create_trial(self, rng, parents, frozen_archive, index, mutation_factor, crossover_rate, pbest_pool):
        if self.mutation != "risk_localized":
            return super()._create_trial(rng, parents, frozen_archive, index, mutation_factor, crossover_rate, pbest_pool)
        size, dim = len(parents), len(self.bounds)
        target = parents[index]
        if self.config.mutation_strategy == "rand1":
            choices = [j for j in range(size) if j != index]
            a_idx, b_idx, c_idx = rng.sample(choices, 3)
            mutant = [parents[a_idx][d] + mutation_factor * (parents[b_idx][d] - parents[c_idx][d]) for d in range(dim)]
        else:
            pbest = parents[rng.choice(pbest_pool)]
            r1_idx = rng.choice([j for j in range(size) if j != index])
            r2 = rng.choice([parents[j] for j in range(size) if j != index and j != r1_idx] + list(frozen_archive))
            mutant = [target[d] + mutation_factor * (pbest[d] - target[d])
                      + mutation_factor * (parents[r1_idx][d] - r2[d]) for d in range(dim)]
        mutant = [max(lo, min(hi, value)) for value, (lo, hi) in zip(mutant, self.bounds)]
        # Force one coordinate in the waypoint nearest a nominal obstacle.
        # Other coordinates retain ordinary binomial crossover.
        waypoint = self._risk_local_waypoint(target)
        if waypoint is None:
            return super()._create_trial(rng, parents, frozen_archive, index, mutation_factor, crossover_rate, pbest_pool)
        forced = rng.randrange(3 * waypoint, min(dim, 3 * waypoint + 3))
        return [mutant[d] if rng.random() < crossover_rate or d == forced else target[d] for d in range(dim)]

    def _evaluate(self, vector: Sequence[float], chance_alpha=None) -> RiskEvaluation:
        if self.risk_model in {"conditional_mean", "conditional_tail", "uncertainty_score"}:
            gate = self.config.chance_alpha if chance_alpha is None else float(chance_alpha)
            path = self.problem.decode(vector)
            metrics = self.problem.evaluate(path)
            if self.config.skip_risk_for_nominal_infeasible and not metrics.feasible:
                return self._retarget(RiskEvaluation(False, 1.0, 0.0, float(metrics.objective), (), 1.0), gate)
            values = [conditional_path_collision(path, self.problem.environment.obstacles, shifts,
                                                 self.config.perturbation_sigma, 2, self.problem.safety_margin)
                      for shifts in self._sample_shifts]
            self._internal_collision_checks += len(values)
            mean_risk = sum(values) / len(values)
            score = mean_risk
            upper = math.nan
            if self.risk_model == "conditional_tail":
                tail_n = max(1, math.ceil(.25 * len(values)))
                score = sum(sorted(values, reverse=True)[:tail_n]) / tail_n
            elif self.risk_model == "uncertainty_score":
                variance = sum((x - mean_risk) ** 2 for x in values) / max(1, len(values) - 1)
                upper = min(1.0, mean_risk + self.risk_weight * math.sqrt(variance))
            measured = RiskEvaluation(metrics.feasible, score if self.risk_model != "uncertainty_score" else mean_risk,
                                      0.0, float(metrics.objective), (), upper)
            self.extension_metadata["conditional_risk_evaluations"] = self._internal_collision_checks
            return self._retarget(measured, gate)

        if self.risk_model == "pathwise_ledger":
            gate = self.config.chance_alpha if chance_alpha is None else float(chance_alpha)
            path = self.problem.decode(vector)
            metrics = self.problem.evaluate(path)
            if self.config.skip_risk_for_nominal_infeasible and not metrics.feasible:
                return self._retarget(RiskEvaluation(False, 1.0, 0.0,
                                                     float(metrics.objective), (), 1.0), gate)
            rate, ledger = _pathwise_collision_ledger(
                path, self._perturbed_banks[0], self.problem.segment_step,
                self.problem.safety_margin,
            )
            self._internal_collision_checks += len(self._perturbed_banks[0])
            self._ledger_by_vector[tuple(vector)] = ledger
            self.extension_metadata["last_evaluated_pathwise_risk_ledger"] = {
                "segment_event_mass": ledger,
                "sum_segment_mass": sum(ledger),
                "whole_path_collision_rate": rate,
                "sample_count": len(self._perturbed_banks[0]),
                "event_allocation": "one full-map collision event per sample; divide event mass evenly across the colliding path segments",
            }
            return self._retarget(RiskEvaluation(metrics.feasible, rate, 0.0,
                                                 float(metrics.objective), (), math.nan), gate)

        gate = self.config.chance_alpha if chance_alpha is None else float(chance_alpha)
        path = self.problem.decode(vector)
        metrics = self.problem.evaluate(path)
        if self.config.skip_risk_for_nominal_infeasible and not metrics.feasible:
            return self._retarget(RiskEvaluation(False, 1.0, 0.0, float(metrics.objective), (), math.nan), gate)
        rates = {}
        for name, environments in self._model_groups.items():
            hits = sum(path.collision_count(environment, step=self.problem.segment_step,
                                             safety_margin=self.problem.safety_margin) > 0
                       for environment in environments)
            rates[name] = hits / len(environments)
        self._internal_collision_checks += sum(len(group) for group in self._model_groups.values())
        score = rates["common_gaussian"] if self.risk_model == "common_translation" else max(rates.values())
        self.extension_metadata["last_model_rates"] = rates
        self.extension_metadata["model_sample_counts"] = {name: len(group) for name, group in self._model_groups.items()}
        return self._retarget(RiskEvaluation(metrics.feasible, score, 0.0, float(metrics.objective), (), math.nan), gate)


class CrossEntropyRiskSearch(ResearchRiskDE):
    """Cross-entropy global search with the shared path-risk evaluator."""

    def optimize(self) -> RiskAwareDEResult:
        cfg = self.config.base
        rng = random.Random(cfg.seed)
        self._internal_collision_checks = 0
        size, dim = cfg.population_size, len(self.bounds)
        budget = cfg.max_evaluations if cfg.max_evaluations is not None else size * (cfg.generations + 1)
        population = self._initial_population(rng, size, dim)
        records = [self._evaluate(vector, self.config.chance_alpha) for vector in population]
        evaluations = size
        best_index = min(range(size), key=lambda i: records[i].key)
        best_vector, best_eval = list(population[best_index]), records[best_index]
        mean = [(lo + hi) * .5 for lo, hi in self.bounds]
        spread = [(hi - lo) * .28 for lo, hi in self.bounds]
        history, key_history = [best_eval.objective], [best_eval.key]
        gate_history = [self.config.chance_alpha]
        update_weight = .65
        generation = 0
        while evaluations < budget:
            count = min(size, budget - evaluations)
            candidates = []
            for _ in range(count):
                candidate = [max(lo, min(hi, rng.gauss(mu, max(sd, .015 * (hi - lo)))))
                             for mu, sd, (lo, hi) in zip(mean, spread, self.bounds)]
                # Preserve the existing internal shortcut as a proposal
                # transform, avoiding a hidden extra evaluation.
                if self.config.local_refinement and rng.random() < self.config.local_refinement_probability:
                    candidate = self._refine_toward_direct(candidate, rng)
                candidates.append(candidate)
            candidate_records = [self._evaluate(candidate, self.config.chance_alpha) for candidate in candidates]
            evaluations += len(candidates)
            elite_count = min(len(candidates), max(2, math.ceil(.2 * len(candidates))))
            elite_indices = sorted(range(len(candidates)), key=lambda i: candidate_records[i].key)[:elite_count]
            elites = [candidates[i] for i in elite_indices]
            target_mean = [sum(row[d] for row in elites) / len(elites) for d in range(dim)]
            target_spread = []
            for d, (lo, hi) in enumerate(self.bounds):
                variance = sum((row[d] - target_mean[d]) ** 2 for row in elites) / max(1, len(elites) - 1)
                target_spread.append(max(.015 * (hi - lo), math.sqrt(variance)))
                mean[d] = (1 - update_weight) * mean[d] + update_weight * target_mean[d]
                spread[d] = max(.015 * (hi - lo), (1 - update_weight) * spread[d] + update_weight * target_spread[d])
            local_best = min(range(len(candidates)), key=lambda i: candidate_records[i].key)
            if candidate_records[local_best].key < best_eval.key:
                best_vector, best_eval = list(candidates[local_best]), candidate_records[local_best]
            history.append(best_eval.objective)
            key_history.append(best_eval.key)
            gate_history.append(self.config.chance_alpha)
            generation += 1
        best_eval = self._retarget(best_eval, self.config.chance_alpha)
        self.extension_metadata.update({
            "global_optimizer": "cross_entropy_method",
            "elite_fraction": .2, "distribution_smoothing": update_weight,
            "minimum_std_fraction": .015,
            "proposal_transform": "existing internal shortcut applied without extra objective/risk evaluation",
            "cem_generations": generation,
        })
        return RiskAwareDEResult(best_vector, best_eval.objective, history, evaluations,
                                 self._internal_collision_checks, best_eval,
                                 key_history=key_history, gate_history=gate_history)


class ResearchCandidateOptimizer(ConditionalProjectedPolish):
    """Three-part candidate wrapper with selectable global/risk/local ideas."""

    def __init__(self, problem, config, *, global_search="de", initialization="uniform",
                 mutation="standard", risk_model="conditional_mean", risk_weight=.5,
                 terminal_method="tangent", risk_credit_fraction=.01, beta=.05, **options):
        if global_search not in {"de", "cem"}:
            raise ValueError("unsupported global search")
        if terminal_method not in {"tangent", "risk_credit"}:
            raise ValueError("unsupported terminal method")
        if not math.isfinite(risk_credit_fraction) or not 0 <= risk_credit_fraction <= .2:
            raise ValueError("risk_credit_fraction must lie in [0, .2]")
        self.global_search = global_search
        self.initialization = initialization
        self.mutation = mutation
        self.risk_model = risk_model
        self.risk_weight = risk_weight
        self.terminal_method = terminal_method
        self.risk_credit_fraction = risk_credit_fraction
        self.beta = beta
        if risk_model == "uncertainty_score" and config.selection_mode != "ucb_budgeted":
            config = replace(config, selection_mode="ucb_budgeted")
        super().__init__(problem, config, **options)

    def _create_evaluator(self, global_cfg):
        evaluator_type = CrossEntropyRiskSearch if self.global_search == "cem" else ResearchRiskDE
        evaluator = evaluator_type(self.problem, global_cfg, initialization=self.initialization,
                                   mutation=self.mutation, risk_model=self.risk_model,
                                   risk_weight=self.risk_weight, beta=self.beta)
        self._active_evaluator = evaluator
        return evaluator

    def _attach_selected_ledger(self, result, evaluator):
        if self.risk_model == "pathwise_ledger":
            ledger = evaluator._ledger_by_vector.get(tuple(result.best_vector))
            self.extension_metadata["selected_path_risk_ledger"] = ledger
            self.extension_metadata["selected_path_ledger_sum"] = sum(ledger) if ledger is not None else None

    def _risk_credit_direction(self, vector, record):
        dimensions = [hi - lo for lo, hi in self.problem.bounds]
        if self._gate_score(record) > self.config.chance_alpha and self.problem.environment.obstacles:
            best = None
            selected = (self._active_evaluator._risk_local_waypoint(vector)
                        if self.risk_model == "pathwise_ledger" else None)
            waypoint_indices = ([selected] if selected is not None
                                else range(self.problem.intermediate_waypoints))
            for index in waypoint_indices:
                point = tuple(vector[3 * index:3 * index + 3])
                for obstacle_index, box in enumerate(self.problem.environment.obstacles):
                    distance, away = _distance_to_box(point, box)
                    candidate = (distance, index, away)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
            if best is not None:
                _, index, away = best
                direction = [0.0] * len(vector)
                for axis in range(3):
                    direction[3 * index + axis] = away[axis] / dimensions[3 * index + axis]
                return direction
        # Once inside the gate, spend remaining evaluations shortening the
        # largest removable local detour.
        count = self.problem.intermediate_waypoints
        if count == 0:
            return [0.0] * len(vector)
        choices = []
        for index in range(count):
            previous = self.problem.start if index == 0 else tuple(vector[3 * (index - 1):3 * index])
            following = self.problem.goal if index == count - 1 else tuple(vector[3 * (index + 1):3 * (index + 2)])
            point = tuple(vector[3 * index:3 * index + 3])
            detour = math.dist(previous, point) + math.dist(point, following) - math.dist(previous, following)
            ledger_mass = 0.0
            if self.risk_model == "pathwise_ledger":
                ledger = getattr(self._active_evaluator, "_ledger_by_vector", {}).get(tuple(vector), [])
                if len(ledger) >= count + 1:
                    ledger_mass = ledger[index] + ledger[index + 1]
            # Under the gate, spend risk-ledger edits on the segment with the
            # largest attributed collision mass; otherwise shorten the largest
            # removable detour as usual.
            priority = (ledger_mass if self.risk_model == "pathwise_ledger"
                        and self._gate_score(record) > self.config.chance_alpha else detour)
            choices.append((priority, index, previous, following, point))
        if not choices:
            return [0.0] * len(vector)
        _, index, previous, following, point = max(choices)
        target = tuple(a + .5 * (b - a) for a, b in zip(previous, following))
        direction = [0.0] * len(vector)
        for axis in range(3):
            direction[3 * index + axis] = (target[axis] - point[axis]) / dimensions[3 * index + axis]
        return direction

    def _gate_score(self, record):
        if self.risk_model == "uncertainty_score":
            return record.risk_upper_bound
        return record.perturbed_collision_rate

    def _risk_credit_accept(self, candidate, incumbent, alpha):
        if not candidate.nominal_feasible:
            return False
        old_risk, new_risk = self._gate_score(incumbent), self._gate_score(candidate)
        length_scale = max(math.dist(self.problem.start, self.problem.goal), 1e-12)
        credit = self.risk_credit_fraction * length_scale
        if old_risk > alpha:
            return new_risk < old_risk - 1e-12 and candidate.objective <= incumbent.objective + credit
        return new_risk <= alpha + 1e-12 and candidate.objective < incumbent.objective - 1e-12

    def optimize(self):
        if self.terminal_method == "tangent":
            result = super().optimize()
            self.extension_metadata = {**self.extension_metadata,
                "global_search": self.global_search, "initialization": self.initialization,
                "mutation": self.mutation, "risk_model": self.risk_model,
                "terminal_method": "conditional_tangent_projection", "risk_credit_fraction": self.risk_credit_fraction}
            self._attach_selected_ledger(result, self._active_evaluator)
            return result
        cfg = self.config
        total = cfg.base.max_evaluations or cfg.base.population_size * (cfg.base.generations + 1)
        reserve = round(total * self.reserve_fraction)
        global_budget = max(cfg.base.population_size, total - reserve)
        global_cfg = replace(cfg, base=replace(cfg.base, max_evaluations=global_budget))
        evaluator = self._create_evaluator(global_cfg)
        self._active_evaluator = evaluator
        raw = evaluator.optimize()
        vector, record = list(raw.best_vector), raw.best_evaluation
        start = raw.evaluations
        events = []
        step = self.initial_step
        while raw.evaluations < total:
            direction = self._risk_credit_direction(vector, record)
            norm = math.sqrt(sum(x * x for x in direction))
            if norm <= 1e-15:
                direction = [0.0] * len(vector)
                direction[len(events) % len(vector)] = 1.0
                norm = 1.0
            direction = [x / norm for x in direction]
            accepted = False
            for _ in range(8):
                if raw.evaluations >= total:
                    break
                proposal = [max(lo, min(hi, x + step * d * (hi - lo)))
                            for x, d, (lo, hi) in zip(vector, direction, self.problem.bounds)]
                measured = evaluator._evaluate(proposal, cfg.chance_alpha)
                raw.evaluations += 1
                accepted = self._risk_credit_accept(measured, record, cfg.chance_alpha)
                events.append({"nfe": raw.evaluations, "accepted": accepted,
                               "path_length": measured.objective,
                               "risk_score": measured.perturbed_collision_rate,
                               "risk_credit_fraction": self.risk_credit_fraction})
                raw.history.append(measured.objective if accepted else record.objective)
                raw.key_history.append(measured.key if accepted else record.key)
                raw.gate_history.append(cfg.chance_alpha)
                if accepted:
                    vector, record = proposal, measured
                    step = min(self.initial_step, step * 1.15)
                    break
                step = max(1e-6, step * .5)
            if not accepted and step <= 1e-5:
                step = self.initial_step * .25
        raw.best_vector, raw.best_evaluation = vector, record
        raw.best_fitness = record.objective
        raw.internal_collision_checks = evaluator._internal_collision_checks
        raw.local_refinement_events = events
        raw.local_refinement_stats = {"risk_credit_global_evaluations": start,
                                      "risk_credit_trials": len(events),
                                      "risk_credit_accepted": sum(bool(x["accepted"]) for x in events)}
        self.extension_metadata = {**getattr(evaluator, "extension_metadata", {}),
            "global_search": self.global_search, "initialization": self.initialization,
            "mutation": self.mutation, "risk_model": self.risk_model,
            "terminal_method": "risk_credit_edit_search", "risk_credit_fraction": self.risk_credit_fraction,
            "risk_credit_trials": len(events), "risk_credit_accepted": sum(bool(x["accepted"]) for x in events),
            "global_stage_evaluations": start,
            "acceptance_rule": "above gate, accept measured risk reductions with bounded length credit; inside gate, accept only shorter paths that remain within gate",
            "safety_certificate": False}
        self._attach_selected_ledger(raw, evaluator)
        return raw


def create_optimizer(problem, base_config, options):
    allowed = {"global_search", "initialization", "mutation", "risk_model", "risk_weight",
               "terminal_method", "risk_credit_fraction", "reserve_fraction", "active_margin",
               "initial_step", "axis", "projection", "max_line_steps", "beta"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unknown candidate options: {sorted(unknown)}")
    return ResearchCandidateOptimizer(problem, base_config, **options)
