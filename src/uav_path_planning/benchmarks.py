"""Small, reproducible benchmark runner for baseline and research planners.

The runner keeps the experiment contract explicit: every algorithm sees the
same scenario, seed, waypoint dimension and objective resolution.  It writes
one :class:`~uav_path_planning.experiments.ExperimentRecord` per run; summary
code can therefore be replaced later without changing the planner code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
from typing import Iterable, Sequence

from .adaptive_de import AdaptiveDEConfig, plan_continuous_path_adaptive
from .astar import AStarConfig, GridAStarPlanner
from .continuous_planner import ContinuousPathProblem, plan_continuous_path
from .de import DEConfig
from .experiments import ExperimentRecord, save_records
from .grid import GridMap3D
from .metrics import CostWeights, PathMetrics, evaluate_path
from .models import Path3D
from .planning import PlannerResult
from .pso import PSOConfig, ParticleSwarmPlanner
from .scenarios import Scenario3D, make_benchmark_scenarios
from .uad_de import UADConfig, plan_continuous_path_uad
from .risk_aware_de import RiskAwareDEConfig, plan_continuous_path_risk_aware
from .jade_shade import JADEConfig, SHADEConfig, plan_continuous_path_jade, plan_continuous_path_shade
from .lshade import LSHADEConfig, plan_continuous_path_lshade


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fairness and budget settings shared by all benchmark runs."""

    seeds: tuple[int, ...] = (0, 1, 2)
    intermediate_waypoints: int = 3
    segment_step: float = 0.5
    grid_resolution: float = 1.0
    de_population: int = 20
    de_generations: int = 40
    de_max_evaluations: int | None = None
    jade_population: int = 20
    jade_generations: int = 40
    shade_population: int = 20
    shade_generations: int = 40
    jade_max_evaluations: int | None = None
    shade_max_evaluations: int | None = None
    lshade_population: int = 20
    lshade_min_population: int = 4
    lshade_generations: int = 40
    lshade_max_evaluations: int | None = None
    pso_particles: int = 20
    pso_iterations: int = 40
    pso_max_evaluations: int | None = None
    adaptive_population: int = 20
    adaptive_generations: int = 40
    # Research-variant knobs are explicit here so sensitivity runs can hold
    # every other part of the benchmark fixed.  They default to the values in
    # AdaptiveDEConfig and therefore preserve the original API.
    adaptive_mutation_factor: float = 0.7
    adaptive_crossover_rate: float = 0.9
    adaptive_p_best_rate: float = 0.2
    adaptive_diversity_threshold: float = 0.08
    adaptive_stagnation_generations: int = 12
    adaptive_restart_fraction: float = 0.25
    adaptive_max_evaluations: int | None = None
    uad_difficulty_gain: float = 0.8
    uad_uncertainty_sigma: float = 1.0
    uad_risk_sampling_step: float = 1.0
    uad_risk_gate: float = 0.10
    uad_collision_risk_weight: float = 0.75
    uad_clearance_risk_weight: float = 0.25
    uad_max_evaluations: int | None = None
    risk_aware_population: int = 20
    risk_aware_generations: int = 40
    risk_aware_max_evaluations: int | None = None
    risk_aware_perturbation_sigma: float = 0.3
    risk_aware_perturbation_sigmas: tuple[float, ...] | None = None
    risk_aware_perturbation_samples: int = 8
    risk_aware_risk_aggregation: str = "pooled"
    risk_aware_chance_alpha: float = 0.10
    risk_aware_perturbation_seed: int = 20260922
    risk_aware_selection_mode: str = "risk_budgeted"
    risk_aware_gate_schedule: str = "fixed"
    risk_aware_gate_start: float | None = None
    # Zero caps risk-repair mutation at no nominal cost increase; None
    # explicitly removes that cost cap for the corresponding ablation.
    risk_aware_tradeoff: float | None = 0.0
    risk_aware_nominal_tolerance: float = 0.0
    risk_aware_mutation_strategy: str = "rand1"
    risk_aware_p_best_rate: float = 0.20
    risk_aware_pbest_selection: str = "risk_first"
    risk_aware_pbest_length_fraction: float = 0.50
    risk_aware_archive_size: int | None = None
    risk_aware_parameter_adaptation: str = "fixed"
    risk_aware_adaptation_learning_rate: float = 0.10
    risk_aware_skip_risk_for_nominal_infeasible: bool = False
    risk_aware_local_refinement: bool = False
    risk_aware_local_refinement_probability: float = 0.15
    risk_aware_local_refinement_scale: float = 0.08
    risk_aware_local_refinement_min_scale: float = 0.05
    risk_aware_local_refinement_shrink: float = 0.5
    risk_aware_local_refinement_grow: float = 1.2
    risk_aware_local_refinement_mode: str = "direct"
    collision_penalty: float = 10_000.0
    safety_margin: float = 0.0

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.intermediate_waypoints < 0:
            raise ValueError("intermediate_waypoints cannot be negative")
        if self.segment_step <= 0 or self.grid_resolution <= 0:
            raise ValueError("segment_step and grid_resolution must be positive")
        if self.adaptive_max_evaluations is not None and self.adaptive_max_evaluations < self.adaptive_population:
            raise ValueError("adaptive_max_evaluations must cover the adaptive initial population")
        if (self.adaptive_population < 4 or self.pso_particles < 4 or self.de_population < 4
                or self.jade_population < 4 or self.shade_population < 4
                or self.lshade_population < 4 or self.lshade_min_population < 4
                or self.lshade_min_population > self.lshade_population
                or self.risk_aware_population < 4):
            raise ValueError("continuous optimizer populations must be at least 4")
        if self.lshade_max_evaluations is not None and self.lshade_max_evaluations < self.lshade_population:
            raise ValueError("lshade_max_evaluations must cover the L-SHADE initial population")
        if self.risk_aware_max_evaluations is not None and self.risk_aware_max_evaluations < self.risk_aware_population:
            raise ValueError("risk_aware_max_evaluations must cover the risk-aware initial population")
        if self.risk_aware_perturbation_sigma < 0.0 or self.risk_aware_perturbation_samples < 1:
            raise ValueError("risk-aware perturbation settings are invalid")
        if self.risk_aware_perturbation_sigmas is not None and (
                not self.risk_aware_perturbation_sigmas
                or any(float(value) < 0.0 or not math.isfinite(float(value))
                       for value in self.risk_aware_perturbation_sigmas)):
            raise ValueError("risk_aware_perturbation_sigmas must be non-empty and non-negative")
        if self.risk_aware_risk_aggregation not in {"pooled", "max"}:
            raise ValueError("risk_aware_risk_aggregation must be 'pooled' or 'max'")
        if not 0.0 <= self.risk_aware_chance_alpha <= 1.0:
            raise ValueError("risk_aware_chance_alpha must be in [0, 1]")
        if self.risk_aware_selection_mode not in {"risk_budgeted", "ucb_budgeted", "nominal_guarded", "lexicographic", "lexicographic_risk", "nominal", "risk_budgeted_tiebreak"}:
            raise ValueError("risk_aware_selection_mode must be 'risk_budgeted', 'ucb_budgeted', 'nominal_guarded', 'lexicographic', 'lexicographic_risk', 'nominal', or 'risk_budgeted_tiebreak'")
        if self.risk_aware_gate_schedule not in {"fixed", "linear"}:
            raise ValueError("risk_aware_gate_schedule must be 'fixed' or 'linear'")
        if self.risk_aware_gate_start is not None and not math.isfinite(self.risk_aware_gate_start):
            raise ValueError("risk_aware_gate_start must be finite or None")
        if self.risk_aware_gate_start is not None and not 0.0 <= self.risk_aware_gate_start <= 1.0:
            raise ValueError("risk_aware_gate_start must lie in [0, 1] or be None")
        if (self.risk_aware_gate_schedule == "linear"
                and self.risk_aware_gate_start is not None
                and self.risk_aware_gate_start < self.risk_aware_chance_alpha):
            raise ValueError("linear risk_aware_gate_start must be at least risk_aware_chance_alpha")
        if self.risk_aware_tradeoff is not None and (not math.isfinite(self.risk_aware_tradeoff) or self.risk_aware_tradeoff < 0.0):
            raise ValueError("risk_aware_tradeoff must be finite and non-negative or None")
        if not math.isfinite(self.risk_aware_nominal_tolerance) or self.risk_aware_nominal_tolerance < 0.0:
            raise ValueError("risk_aware_nominal_tolerance must be finite and non-negative")
        if self.risk_aware_mutation_strategy not in {"rand1", "current_to_pbest1"}:
            raise ValueError("risk_aware_mutation_strategy must be 'rand1' or 'current_to_pbest1'")
        if not math.isfinite(self.risk_aware_p_best_rate) or not 0.0 < self.risk_aware_p_best_rate <= 1.0:
            raise ValueError("risk_aware_p_best_rate must lie in (0, 1]")
        if self.risk_aware_pbest_selection not in {"risk_first", "risk_diverse"}:
            raise ValueError("risk_aware_pbest_selection must be 'risk_first' or 'risk_diverse'")
        if (not math.isfinite(self.risk_aware_pbest_length_fraction)
                or not 0.0 <= self.risk_aware_pbest_length_fraction <= 1.0):
            raise ValueError("risk_aware_pbest_length_fraction must lie in [0, 1]")
        if self.risk_aware_archive_size is not None and (
                isinstance(self.risk_aware_archive_size, bool)
                or not isinstance(self.risk_aware_archive_size, int)
                or self.risk_aware_archive_size < 0):
            raise ValueError("risk_aware_archive_size must be a non-negative integer or None")
        if self.risk_aware_parameter_adaptation not in {"fixed", "risk_success"}:
            raise ValueError("risk_aware_parameter_adaptation must be 'fixed' or 'risk_success'")
        if (not math.isfinite(self.risk_aware_adaptation_learning_rate)
                or not 0.0 < self.risk_aware_adaptation_learning_rate <= 1.0):
            raise ValueError("risk_aware_adaptation_learning_rate must lie in (0, 1]")
        if not isinstance(self.risk_aware_skip_risk_for_nominal_infeasible, bool):
            raise ValueError("risk_aware_skip_risk_for_nominal_infeasible must be a boolean")
        if not 0.0 <= self.risk_aware_local_refinement_probability <= 1.0:
            raise ValueError("risk_aware_local_refinement_probability must be in [0, 1]")
        if self.risk_aware_local_refinement_scale <= 0.0 or not math.isfinite(self.risk_aware_local_refinement_scale):
            raise ValueError("risk_aware_local_refinement_scale must be finite and positive")
        if self.risk_aware_local_refinement_mode not in {"direct", "shortcut", "partial_shortcut", "greedy_shortcut", "safe_shortcut", "mixed", "detour_weighted_partial", "greedy_partial", "adaptive_partial_shortcut"}:
            raise ValueError("unsupported risk_aware_local_refinement_mode")
        if self.risk_aware_local_refinement_mode in {"partial_shortcut", "detour_weighted_partial", "greedy_partial", "adaptive_partial_shortcut"} and self.risk_aware_local_refinement_scale > 1.0:
            raise ValueError("partial refinement risk_aware_local_refinement_scale must lie in (0, 1]")
        if self.risk_aware_local_refinement_mode == "adaptive_partial_shortcut":
            if not math.isfinite(self.risk_aware_local_refinement_min_scale) or not 0.0 < self.risk_aware_local_refinement_min_scale <= self.risk_aware_local_refinement_scale:
                raise ValueError("adaptive partial minimum scale must lie in (0, initial scale]")
            if not math.isfinite(self.risk_aware_local_refinement_shrink) or not 0.0 < self.risk_aware_local_refinement_shrink < 1.0:
                raise ValueError("adaptive partial shrink must lie in (0, 1)")
            if not math.isfinite(self.risk_aware_local_refinement_grow) or self.risk_aware_local_refinement_grow <= 1.0:
                raise ValueError("adaptive partial grow must exceed 1")
        if self.uad_difficulty_gain < 0.0 or self.uad_uncertainty_sigma <= 0.0 or self.uad_risk_sampling_step <= 0.0:
            raise ValueError("UAD gain must be non-negative and UAD risk scales positive")
        if not 0.0 <= self.uad_risk_gate < 1.0:
            raise ValueError("uad_risk_gate must be in [0, 1)")
        if self.uad_collision_risk_weight < 0.0 or self.uad_clearance_risk_weight < 0.0:
            raise ValueError("UAD risk weights cannot be negative")
        if self.uad_collision_risk_weight + self.uad_clearance_risk_weight <= 0.0:
            raise ValueError("at least one UAD risk weight must be positive")


def _problem(scenario: Scenario3D, config: BenchmarkConfig) -> ContinuousPathProblem:
    return ContinuousPathProblem(
        scenario.start,
        scenario.goal,
        scenario.environment,
        intermediate_waypoints=config.intermediate_waypoints,
        collision_penalty=config.collision_penalty,
        segment_step=config.segment_step,
        safety_margin=config.safety_margin,
        weights=CostWeights(collision=config.collision_penalty),
    )


def _result_for_continuous(scenario: Scenario3D, algorithm: str, seed: int, config: BenchmarkConfig) -> PlannerResult:
    problem = _problem(scenario, config)
    started = time.perf_counter()
    if algorithm == "de":
        path, raw = plan_continuous_path(
            problem,
            DEConfig(
                population_size=config.de_population,
                generations=config.de_generations,
                seed=seed,
                max_evaluations=(config.de_max_evaluations if config.de_max_evaluations is not None
                                 else config.de_population * (config.de_generations + 1)),
            ),
        )
        metadata = {"algorithm": algorithm, "optimizer": "DE/rand/1/bin"}
    elif algorithm == "pso":
        raw_result = ParticleSwarmPlanner(
            problem,
            PSOConfig(
                particles=config.pso_particles,
                iterations=config.pso_iterations,
                seed=seed,
                max_evaluations=(config.pso_max_evaluations
                                 if config.pso_max_evaluations is not None
                                 else config.pso_particles * (config.pso_iterations + 1)),
            ),
        ).plan()
        raw = raw_result
        path = raw_result.path
        metadata = dict(raw_result.metadata)
    elif algorithm == "jade":
        path, raw = plan_continuous_path_jade(
            problem,
            JADEConfig(
                population_size=config.jade_population,
                generations=config.jade_generations,
                seed=seed,
                max_evaluations=(config.jade_max_evaluations
                                 if config.jade_max_evaluations is not None
                                 else config.jade_population * (config.jade_generations + 1)),
            ),
        )
        metadata = {
            "algorithm": algorithm,
            "optimizer": "JADE current-to-pbest with archive",
            "mu_f_history": raw.mu_f_history,
            "mu_cr_history": raw.mu_cr_history,
            "success_history": raw.success_history,
            "archive_history": raw.archive_history,
        }
    elif algorithm == "shade":
        path, raw = plan_continuous_path_shade(
            problem,
            SHADEConfig(
                population_size=config.shade_population,
                generations=config.shade_generations,
                seed=seed,
                max_evaluations=(config.shade_max_evaluations
                                 if config.shade_max_evaluations is not None
                                 else config.shade_population * (config.shade_generations + 1)),
            ),
        )
        metadata = {
            "algorithm": algorithm,
            "optimizer": "SHADE success-history adaptive DE",
            "memory_f_history": raw.memory_f_history,
            "memory_cr_history": raw.memory_cr_history,
            "success_history": raw.success_history,
            "archive_history": raw.archive_history,
        }
    elif algorithm == "lshade":
        path, raw = plan_continuous_path_lshade(
            problem,
            LSHADEConfig(
                population_size=config.lshade_population,
                min_population_size=config.lshade_min_population,
                generations=config.lshade_generations,
                seed=seed,
                max_evaluations=(config.lshade_max_evaluations
                                 if config.lshade_max_evaluations is not None
                                 else config.lshade_population * (config.lshade_generations + 1)),
            ),
        )
        metadata = {
            "algorithm": algorithm,
            "optimizer": "L-SHADE success-history adaptive DE with linear population-size reduction",
            "population_size_history": raw.population_size_history,
            "memory_f_history": raw.memory_f_history,
            "memory_cr_history": raw.memory_cr_history,
            "success_history": raw.success_history,
            "archive_history": raw.archive_history,
        }
    elif algorithm.startswith("adaptive_de"):
        flags = {
            "adaptive_de_no_parameter_adaptation": {"adaptive_parameters": False},
            "adaptive_de_no_elite_diversity": {"elite_diversity": False},
            "adaptive_de_no_stagnation_restart": {"stagnation_restart": False},
            "adaptive_de_plain": {"adaptive_parameters": False, "elite_diversity": False, "stagnation_restart": False},
        }.get(algorithm, {})
        path, raw = plan_continuous_path_adaptive(
            problem,
            AdaptiveDEConfig(
                population_size=config.adaptive_population,
                generations=config.adaptive_generations,
                seed=seed,
                mutation_factor=config.adaptive_mutation_factor,
                crossover_rate=config.adaptive_crossover_rate,
                p_best_rate=config.adaptive_p_best_rate,
                diversity_threshold=config.adaptive_diversity_threshold,
                stagnation_generations=config.adaptive_stagnation_generations,
                restart_fraction=config.adaptive_restart_fraction,
                max_evaluations=(config.adaptive_max_evaluations
                                 if config.adaptive_max_evaluations is not None
                                 else config.adaptive_population * (config.adaptive_generations + 1)),
                **flags,
            ),
        )
        metadata = {
            "algorithm": algorithm,
            "optimizer": "adaptive multi-strategy DE",
            "ablation_flags": flags,
            "restart_count": raw.restart_count,
            "strategy_counts": raw.strategy_counts,
        }
    elif algorithm.startswith("uad_de"):
        uad_gain = 0.0 if algorithm == "uad_de_no_risk" else config.uad_difficulty_gain
        base = AdaptiveDEConfig(
            population_size=config.adaptive_population,
            generations=config.adaptive_generations,
            seed=seed,
            mutation_factor=config.adaptive_mutation_factor,
            crossover_rate=config.adaptive_crossover_rate,
            p_best_rate=config.adaptive_p_best_rate,
            diversity_threshold=config.adaptive_diversity_threshold,
            stagnation_generations=config.adaptive_stagnation_generations,
            restart_fraction=config.adaptive_restart_fraction,
            max_evaluations=(config.uad_max_evaluations
                             if config.uad_max_evaluations is not None
                             else config.adaptive_population * (config.adaptive_generations + 1)),
        )
        path, raw = plan_continuous_path_uad(
            problem,
            UADConfig(
                base=base,
                uncertainty_sigma=config.uad_uncertainty_sigma,
                risk_sampling_step=config.uad_risk_sampling_step,
                risk_gate=config.uad_risk_gate,
                difficulty_gain=uad_gain,
                collision_risk_weight=config.uad_collision_risk_weight,
                clearance_risk_weight=config.uad_clearance_risk_weight,
            ),
        )
        metadata = {
            "algorithm": algorithm,
            "optimizer": "uncertainty-aware difficulty-adaptive DE",
            "difficulty_gain": uad_gain,
            "restart_count": raw.restart_count,
            "strategy_counts": raw.strategy_counts,
            "difficulty_history": raw.difficulty_history,
        }
    elif algorithm in {"risk_aware_de", "risk_aware_de_nominal"} or algorithm.startswith("risk_aware_ablation_"):
        # Ablation aliases reuse the exact planner implementation while
        # changing one registered mechanism in the immutable benchmark
        # configuration.  Keeping the alias at this boundary lets nominal
        # and robustness runners evaluate the same variant independently.
        ablation_changes = {
            "risk_aware_no_risk_gate": {"risk_aware_selection_mode": "nominal"},
            "risk_aware_no_pbest_archive": {"risk_aware_mutation_strategy": "rand1", "risk_aware_archive_size": 0},
            "risk_aware_no_parameter_adaptation": {"risk_aware_parameter_adaptation": "fixed"},
            "risk_aware_no_gate_annealing": {"risk_aware_gate_schedule": "fixed", "risk_aware_gate_start": None},
            "risk_aware_no_local_refinement": {"risk_aware_local_refinement": False},
            "risk_aware_no_risk_tradeoff": {"risk_aware_tradeoff": None},
            "risk_aware_no_infeasible_shortcut": {"risk_aware_skip_risk_for_nominal_infeasible": False},
        }
        variant_name = algorithm.removeprefix("risk_aware_ablation_")
        risk_config = replace(config, **ablation_changes.get(variant_name, {}))
        is_nominal_control = algorithm == "risk_aware_de_nominal"
        path, raw = plan_continuous_path_risk_aware(
            problem,
            RiskAwareDEConfig(
                base=DEConfig(
                    population_size=risk_config.risk_aware_population,
                    generations=risk_config.risk_aware_generations,
                    seed=seed,
                    max_evaluations=(risk_config.risk_aware_max_evaluations
                                     if risk_config.risk_aware_max_evaluations is not None
                                     else risk_config.risk_aware_population * (risk_config.risk_aware_generations + 1)),
                ),
                perturbation_sigma=risk_config.risk_aware_perturbation_sigma,
                perturbation_sigmas=risk_config.risk_aware_perturbation_sigmas,
                perturbation_samples=risk_config.risk_aware_perturbation_samples,
                risk_aggregation=risk_config.risk_aware_risk_aggregation,
                chance_alpha=risk_config.risk_aware_chance_alpha,
                risk_gate_schedule=risk_config.risk_aware_gate_schedule,
                risk_gate_start=risk_config.risk_aware_gate_start,
                perturbation_seed=risk_config.risk_aware_perturbation_seed,
                selection_mode=("nominal" if is_nominal_control else risk_config.risk_aware_selection_mode),
                risk_tradeoff=risk_config.risk_aware_tradeoff,
                nominal_tolerance=risk_config.risk_aware_nominal_tolerance,
                mutation_strategy=risk_config.risk_aware_mutation_strategy,
                p_best_rate=risk_config.risk_aware_p_best_rate,
                pbest_selection=risk_config.risk_aware_pbest_selection,
                pbest_length_fraction=risk_config.risk_aware_pbest_length_fraction,
                archive_size=risk_config.risk_aware_archive_size,
                parameter_adaptation=risk_config.risk_aware_parameter_adaptation,
                adaptation_learning_rate=risk_config.risk_aware_adaptation_learning_rate,
                skip_risk_for_nominal_infeasible=risk_config.risk_aware_skip_risk_for_nominal_infeasible,
                local_refinement=risk_config.risk_aware_local_refinement,
                local_refinement_probability=risk_config.risk_aware_local_refinement_probability,
                local_refinement_scale=risk_config.risk_aware_local_refinement_scale,
                local_refinement_min_scale=risk_config.risk_aware_local_refinement_min_scale,
                local_refinement_shrink=risk_config.risk_aware_local_refinement_shrink,
                local_refinement_grow=risk_config.risk_aware_local_refinement_grow,
                local_refinement_mode=risk_config.risk_aware_local_refinement_mode,
            ),
        )
        metadata = {
            "algorithm": algorithm,
            "optimizer": "risk-aware DE with chance-budgeted selection",
            "perturbation_sigma": risk_config.risk_aware_perturbation_sigma,
            "perturbation_sigmas": risk_config.risk_aware_perturbation_sigmas,
            "perturbation_samples": risk_config.risk_aware_perturbation_samples,
            "risk_aggregation": risk_config.risk_aware_risk_aggregation,
            "chance_alpha": risk_config.risk_aware_chance_alpha,
            "risk_gate_schedule": risk_config.risk_aware_gate_schedule,
            "risk_gate_start": risk_config.risk_aware_gate_start,
            "perturbation_seed": risk_config.risk_aware_perturbation_seed,
            "internal_collision_checks": raw.internal_collision_checks,
            "nominal_feasible": raw.best_evaluation.nominal_feasible,
            "perturbed_collision_rate": raw.best_evaluation.perturbed_collision_rate,
            "risk_upper_bound": raw.best_evaluation.risk_upper_bound,
            "chance_violation": raw.best_evaluation.chance_violation,
            "lexicographic_key_history": raw.key_history,
            "risk_gate_history": raw.gate_history,
            "selection_mode": ("nominal" if is_nominal_control else risk_config.risk_aware_selection_mode),
            "pbest_selection": risk_config.risk_aware_pbest_selection,
            "pbest_length_fraction": risk_config.risk_aware_pbest_length_fraction,
            "risk_tradeoff": risk_config.risk_aware_tradeoff,
            "nominal_tolerance": risk_config.risk_aware_nominal_tolerance,
            "local_refinement": risk_config.risk_aware_local_refinement,
            "local_refinement_probability": risk_config.risk_aware_local_refinement_probability,
            "local_refinement_scale": risk_config.risk_aware_local_refinement_scale,
            "local_refinement_mode": risk_config.risk_aware_local_refinement_mode,
            "local_refinement_stats": raw.local_refinement_stats,
            "local_refinement_events": raw.local_refinement_events,
            "parameter_adaptation": risk_config.risk_aware_parameter_adaptation,
            "adaptation_learning_rate": risk_config.risk_aware_adaptation_learning_rate,
            "skip_risk_for_nominal_infeasible": risk_config.risk_aware_skip_risk_for_nominal_infeasible,
            "mutation_factor_history": raw.mutation_factor_history,
            "crossover_rate_history": raw.crossover_rate_history,
        }
    else:
        raise ValueError(f"unknown continuous algorithm: {algorithm}")

    if path is None:
        return PlannerResult(None, math.inf, False, runtime_seconds=time.perf_counter() - started, metadata=metadata)
    if hasattr(raw, "best_fitness"):
        cost = float(raw.best_fitness)
    elif hasattr(raw, "cost"):
        cost = float(raw.cost)
    else:
        cost = float(path.length)
    evaluations = int(getattr(raw, "evaluations", 0))
    history = list(getattr(raw, "history", []))
    if history and algorithm != "risk_aware_de":
        # Keep a compact convergence trace in every row.  Full histories are
        # available to sensitivity experiments through this sampled vector;
        # storing a bounded number of points keeps CSV/JSON manageable.
        sample_count = min(21, len(history))
        if sample_count == len(history):
            sampled_history = history
        else:
            indices = [round(i * (len(history) - 1) / (sample_count - 1)) for i in range(sample_count)]
            sampled_history = [history[index] for index in indices]
        metadata["history_summary"] = {
            "initial": float(history[0]),
            "final": float(history[-1]),
            "best": float(min(history)),
            "steps": len(history),
            "relative_gain_pct": ((history[0] - min(history)) / abs(history[0]) * 100.0) if history[0] else 0.0,
            "sampled": [float(value) for value in sampled_history],
        }
    metadata["first_feasible_evaluation"] = problem.first_feasible_evaluation
    metadata["objective_evaluation_trace"] = problem.evaluation_trace
    final_metrics = problem.evaluate(path)
    success = final_metrics.feasible
    metadata["collision_count"] = final_metrics.collision_segments
    return PlannerResult(path, cost, success, evaluations, time.perf_counter() - started, history, metadata)


def _grid_for_scenario(scenario: Scenario3D, resolution: float) -> tuple[GridMap3D, tuple[int, int, int], tuple[int, int, int], tuple[float, float, float]]:
    lo = scenario.environment.bounds.minimum
    hi = scenario.environment.bounds.maximum
    size = tuple(int(round((b - a) / resolution)) + 1 for a, b in zip(lo, hi))
    grid = GridMap3D.empty(size)  # type: ignore[arg-type]
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                point = tuple(lo[d] + (x, y, z)[d] * resolution for d in range(3))
                if not scenario.environment.is_free(point):
                    grid.set_occupied((x, y, z))
    start = tuple(int(round((value - lo[d]) / resolution)) for d, value in enumerate(scenario.start))
    goal = tuple(int(round((value - lo[d]) / resolution)) for d, value in enumerate(scenario.goal))
    return grid, start, goal, lo


def _result_for_astar(scenario: Scenario3D, config: BenchmarkConfig) -> PlannerResult:
    grid, start, goal, lo = _grid_for_scenario(scenario, config.grid_resolution)
    raw = GridAStarPlanner(
        grid,
        AStarConfig(
            include_diagonal=True,
            safety_margin=config.safety_margin,
            world_origin=lo,
            cell_size=(config.grid_resolution,) * 3,
        ),
    ).plan(start, goal, environment=scenario.environment)
    if raw.path is not None:
        translated = Path3D([
            tuple(lo[d] + point[d] * config.grid_resolution for d in range(3))
            for point in raw.path.waypoints
        ])
        raw.path = translated
        raw.metadata["grid_resolution"] = config.grid_resolution
    return raw


def run_one(scenario: Scenario3D, algorithm: str, seed: int, config: BenchmarkConfig | None = None) -> ExperimentRecord:
    """Run one algorithm and convert its output to the shared record schema."""

    config = config or BenchmarkConfig()
    started = time.perf_counter()
    if algorithm == "astar3d":
        result = _result_for_astar(scenario, config)
    else:
        result = _result_for_continuous(scenario, algorithm, seed, config)
    elapsed = result.runtime_seconds or (time.perf_counter() - started)
    evaluation_weights = CostWeights(collision=config.collision_penalty)
    if result.path is None:
        # A failed search remains a valid record.  A direct path supplies a
        # deterministic metric row while ``status`` preserves the failure.
        fallback = Path3D([scenario.start, scenario.goal])
        metrics = evaluate_path(fallback, scenario.environment, weights=evaluation_weights, segment_step=config.segment_step, safety_margin=config.safety_margin)
        status = "no_path"
    else:
        metrics = evaluate_path(result.path, scenario.environment, weights=evaluation_weights, segment_step=config.segment_step, safety_margin=config.safety_margin)
        status = "ok" if result.success else "infeasible"
    return ExperimentRecord(
        algorithm=algorithm,
        scenario=scenario.name,
        seed=seed,
        runtime_seconds=elapsed,
        evaluations=result.evaluations,
        metrics=metrics,
        status=status,
        config={"budget": config.__dict__, "planner_metadata": result.metadata},
    )


def run_benchmark(
    scenarios: Sequence[Scenario3D] | None = None,
    *,
    algorithms: Sequence[str] = ("de", "pso", "astar3d", "adaptive_de", "uad_de"),
    config: BenchmarkConfig | None = None,
) -> list[ExperimentRecord]:
    """Run all algorithm/scene/seed combinations in deterministic order."""

    config = config or BenchmarkConfig()
    scenarios = list(scenarios or make_benchmark_scenarios())
    records: list[ExperimentRecord] = []
    for scenario in scenarios:
        for algorithm in algorithms:
            for seed in config.seeds:
                # A* is deterministic; still emit one row per seed so the
                # comparison table has a rectangular design.
                records.append(run_one(scenario, algorithm, seed, config))
    return records


def run_ablation(
    scenarios: Sequence[Scenario3D] | None = None,
    *,
    config: BenchmarkConfig | None = None,
) -> list[ExperimentRecord]:
    """Run the proposed optimizer with one mechanism removed at a time.

    The budget, seed and objective are inherited from ``config``.  Variant
    labels are stored in ``ExperimentRecord.algorithm`` so they can be grouped
    directly beside the main benchmark records.
    """

    config = config or BenchmarkConfig()
    scenarios = list(scenarios or make_benchmark_scenarios())
    variants = {
        "adaptive_de_full": {},
        "adaptive_de_no_parameter_adaptation": {"adaptive_parameters": False},
        "adaptive_de_no_elite_diversity": {"elite_diversity": False},
        "adaptive_de_no_stagnation_restart": {"stagnation_restart": False},
        "adaptive_de_plain": {"adaptive_parameters": False, "elite_diversity": False, "stagnation_restart": False},
        "uad_de_full": {},
        "uad_de_no_risk": {},
    }
    records: list[ExperimentRecord] = []
    for label, _flags in variants.items():
        for scenario in scenarios:
            for seed in config.seeds:
                records.append(run_one(scenario, label, seed, config))
    return records


def run_risk_aware_ablation(
    scenarios: Sequence[Scenario3D] | None = None,
    *,
    config: BenchmarkConfig | None = None,
) -> list[ExperimentRecord]:
    """Run the complete, mechanism-level RiskAwareDE ablation matrix.

    Every row uses the same maps, seeds, objective and evaluation budget.  A
    variant is created by changing exactly one registered RiskAwareDE switch
    from the supplied configuration; this keeps the ablation interpretable
    and prevents the legacy UAD-DE ablation runner from being mistaken for a
    RiskAwareDE study.
    """

    config = config or BenchmarkConfig()
    scenarios = list(scenarios or make_benchmark_scenarios())
    variants = {
        "risk_aware_full": {},
        "risk_aware_no_risk_gate": {"risk_aware_selection_mode": "nominal"},
        "risk_aware_no_pbest_archive": {
            "risk_aware_mutation_strategy": "rand1",
            "risk_aware_archive_size": 0,
        },
        "risk_aware_no_parameter_adaptation": {"risk_aware_parameter_adaptation": "fixed"},
        "risk_aware_no_gate_annealing": {
            "risk_aware_gate_schedule": "fixed",
            "risk_aware_gate_start": None,
        },
        "risk_aware_no_local_refinement": {
            "risk_aware_local_refinement": False,
        },
        "risk_aware_no_risk_tradeoff": {"risk_aware_tradeoff": None},
        "risk_aware_no_infeasible_shortcut": {
            "risk_aware_skip_risk_for_nominal_infeasible": False,
        },
        "risk_aware_risk_tiebreak": {
            "risk_aware_selection_mode": "risk_budgeted_tiebreak",
        },
        "risk_aware_shortcut_refinement": {
            "risk_aware_local_refinement": True,
            "risk_aware_local_refinement_mode": "shortcut",
        },
    }
    records: list[ExperimentRecord] = []
    for label, changes in variants.items():
        variant_config = replace(config, **changes)
        for scenario in scenarios:
            for seed in variant_config.seeds:
                record = run_one(scenario, "risk_aware_de", seed, variant_config)
                record.algorithm = label
                record.config["risk_ablation_variant"] = label
                record.config["risk_ablation_changes"] = dict(changes)
                records.append(record)
    return records


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def summarize_records(
    records: Sequence[ExperimentRecord],
    proposed: str = "uad_de",
    baseline_algorithm: str = "de",
) -> list[dict[str, object]]:
    """Aggregate records by algorithm and calculate lower-is-better gains."""

    grouped: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault(record.algorithm, []).append(record)
    baseline = grouped.get(baseline_algorithm, [])
    feasible_record = lambda r: r.status == "ok" and r.metrics.feasible
    base_feasible = _mean(float(feasible_record(r)) for r in baseline)
    base_length = _mean(r.metrics.length for r in baseline if feasible_record(r))
    rows: list[dict[str, object]] = []
    for algorithm, items in grouped.items():
        feasible = _mean(float(feasible_record(r)) for r in items)
        length = _mean(r.metrics.length for r in items if feasible_record(r))
        rows.append({
            "algorithm": algorithm,
            "runs": len(items),
            "feasible_rate": feasible,
            "mean_length_feasible": length,
            "mean_objective": _mean(r.metrics.objective for r in items),
            "mean_runtime_seconds": _mean(r.runtime_seconds for r in items),
            "mean_evaluations": _mean(float(r.evaluations) for r in items),
            "improvement_vs_de_length_pct": (base_length - length) / base_length * 100.0 if base_length and math.isfinite(length) else math.nan,
            "feasible_rate_delta_vs_de_pct": (feasible - base_feasible) * 100.0 if math.isfinite(base_feasible) else math.nan,
            "is_proposed": algorithm == proposed,
        })
    return rows


def save_summary_markdown(rows: Sequence[dict[str, object]], path: str | Path, baseline_label: str = "DE") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["algorithm", "runs", "feasible_rate", "mean_length_feasible", "mean_objective", "mean_runtime_seconds", "mean_evaluations", "improvement_vs_de_length_pct", "feasible_rate_delta_vs_de_pct"]
    lines = ["# 基线与候选算法汇总", "", "按相同场景/种子配对。请核对每条记录的预算：候选评价次数、风险检查次数、A* 扩展节点不能视为同一计算成本。长度和目标值越低越好，可行率越高越好。", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            values.append(f"{value:.4f}" if isinstance(value, float) and math.isfinite(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", f"`improvement_vs_de_length_pct` 的定义是 `({baseline_label}平均可行路径长度 - 当前算法平均可行路径长度) / {baseline_label}平均可行路径长度 × 100%`。它是描述性指标，正式论文还应报告每个场景的配对统计和置信区间。", ""])
    target.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["BenchmarkConfig", "run_one", "run_benchmark", "run_ablation", "run_risk_aware_ablation", "summarize_records", "save_summary_markdown"]
