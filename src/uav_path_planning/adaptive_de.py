"""Research extension: adaptive multi-strategy differential evolution.

The implementation is deliberately independent from the paper used as the
initial reading reference.  It is a small, CPU-only optimizer that exposes
the mechanisms as switches so that later experiments can attribute gains to
individual components.  It optimizes any bounded continuous objective and
therefore can be used with :class:`ContinuousPathProblem.objective` directly.

The default variant combines two mutation strategies, adapts ``F`` and
``CR`` from population diversity and recent success, keeps a small archive of
replaced points, and re-seeds only the worst individuals after stagnation.
Each mechanism can be disabled for ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Sequence


Objective = Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class AdaptiveDEConfig:
    """Configuration of :class:`AdaptiveDiversityDE`.

    ``adaptive_parameters``, ``elite_diversity`` and ``stagnation_restart``
    are the three principal ablation switches.  A baseline can therefore be
    run with all three set to ``False`` while keeping the budget and seed
    identical to the proposed method.
    """

    population_size: int = 24
    generations: int = 100
    mutation_factor: float = 0.7
    crossover_rate: float = 0.9
    p_best_rate: float = 0.2
    archive_size: int | None = None
    diversity_threshold: float = 0.08
    stagnation_generations: int = 12
    restart_fraction: float = 0.25
    adaptive_parameters: bool = True
    elite_diversity: bool = True
    stagnation_restart: bool = True
    max_evaluations: int | None = None
    seed: int | None = 0
    difficulty_gain: float = 0.0
    difficulty_fn: Callable[[Sequence[Sequence[float]], Sequence[float], float, float], float] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.generations < 0:
            raise ValueError("generations cannot be negative")
        if not 0.0 < self.mutation_factor <= 2.0:
            raise ValueError("mutation_factor must be in (0, 2]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 < self.p_best_rate <= 1.0:
            raise ValueError("p_best_rate must be in (0, 1]")
        if self.archive_size is not None and self.archive_size < 0:
            raise ValueError("archive_size cannot be negative")
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError("diversity_threshold must be in [0, 1]")
        if self.stagnation_generations < 1:
            raise ValueError("stagnation_generations must be at least 1")
        if not 0.0 <= self.restart_fraction <= 1.0:
            raise ValueError("restart_fraction must be in [0, 1]")
        if self.max_evaluations is not None and self.max_evaluations < self.population_size:
            raise ValueError("max_evaluations must cover the initial population")
        if self.difficulty_gain < 0.0:
            raise ValueError("difficulty_gain cannot be negative")


@dataclass
class AdaptiveDEResult:
    """Auditable optimizer output used by the experiment runner."""

    best_vector: list[float]
    best_fitness: float
    history: list[float]
    diversity_history: list[float]
    success_rate_history: list[float]
    evaluations: int
    restart_count: int = 0
    strategy_counts: dict[str, int] = field(default_factory=dict)
    difficulty_history: list[float] = field(default_factory=list)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class AdaptiveDiversityDE:
    """Bounded minimizer with adaptive multi-strategy DE operators.

    The class has no NumPy dependency.  Randomness is local to one optimizer
    instance and controlled solely by ``config.seed``, making repeated
    benchmark rows reproducible.
    """

    def __init__(self, bounds: Sequence[tuple[float, float]], config: AdaptiveDEConfig | None = None):
        if not bounds:
            raise ValueError("at least one decision variable is required")
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        if any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi for lo, hi in self.bounds):
            raise ValueError("every bound must be finite and lower < upper")
        self.config = config or AdaptiveDEConfig()

    def _diversity(self, population: Sequence[Sequence[float]]) -> float:
        """Mean normalized coordinate spread in ``[0, 1]``."""

        if not population:
            return 0.0
        spreads: list[float] = []
        for d, (lo, hi) in enumerate(self.bounds):
            values = [float(x[d]) for x in population]
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            # A uniform population has sigma/range ~= .289.  Scale this to a
            # convenient [0, 1] diagnostic while retaining relative changes.
            spreads.append(min(1.0, math.sqrt(variance) / (hi - lo) * math.sqrt(12.0)))
        return sum(spreads) / len(spreads)

    @staticmethod
    def _sample_distinct(rng: random.Random, size: int, excluded: int, count: int) -> list[int]:
        choices = [index for index in range(size) if index != excluded]
        if len(choices) < count:
            raise ValueError("population is too small for the selected DE operator")
        return rng.sample(choices, count)

    def _parameters(self, diversity: float, success_rate: float, difficulty: float = 0.0) -> tuple[float, float]:
        if not self.config.adaptive_parameters and self.config.difficulty_gain <= 0.0:
            return self.config.mutation_factor, self.config.crossover_rate
        # Low diversity receives a stronger differential step and crossover;
        # an unsuccessful recent generation receives a small exploration boost.
        f = self.config.mutation_factor * (1.20 - 0.50 * diversity)
        f *= 1.0 + 0.15 * max(-1.0, min(1.0, 0.20 - success_rate))
        f *= 1.0 + self.config.difficulty_gain * max(0.0, min(1.0, difficulty))
        cr = self.config.crossover_rate + 0.20 * (1.0 - diversity)
        cr += 0.15 * self.config.difficulty_gain * max(0.0, min(1.0, difficulty))
        return _clip(f, 0.1, 2.0), _clip(cr, 0.0, 1.0)

    def _difficulty(
        self,
        population: Sequence[Sequence[float]],
        fitness: Sequence[float],
        diversity: float,
        success_rate: float,
    ) -> float:
        if self.config.difficulty_fn is None:
            return 0.0
        value = self.config.difficulty_fn(population, fitness, diversity, success_rate)
        if not math.isfinite(float(value)):
            return 0.0
        return _clip(float(value), 0.0, 1.0)

    def optimize(self, objective: Objective) -> AdaptiveDEResult:
        """Minimize ``objective`` under the constructor's box constraints."""

        rng = random.Random(self.config.seed)
        dim = len(self.bounds)
        size = self.config.population_size
        population = [[rng.uniform(lo, hi) for lo, hi in self.bounds] for _ in range(size)]
        fitness = [float(objective(x)) for x in population]
        evaluations = size
        budget = self.config.max_evaluations
        best_index = min(range(size), key=fitness.__getitem__)
        best_vector = list(population[best_index])
        best_fitness = fitness[best_index]
        history = [best_fitness]
        diversity_history = [self._diversity(population)]
        success_rate_history: list[float] = []
        difficulty_history = [self._difficulty(population, fitness, diversity_history[0], 0.0)]
        archive: list[list[float]] = []
        archive_limit = self.config.archive_size if self.config.archive_size is not None else size
        no_improvement = 0
        restart_count = 0
        strategy_counts = {"rand1": 0, "current_to_pbest1": 0}

        for _generation in range(self.config.generations):
            diversity = self._diversity(population)
            previous_best = best_fitness
            recent_success = success_rate_history[-1] if success_rate_history else 0.0
            difficulty = self._difficulty(population, fitness, diversity, recent_success)
            f, cr = self._parameters(diversity, recent_success, difficulty)
            order = sorted(range(size), key=fitness.__getitem__)
            p_count = max(2, int(math.ceil(self.config.p_best_rate * size)))
            successful = 0

            attempted = 0
            budget_exhausted = False
            for i, target in enumerate(population):
                if budget is not None and evaluations >= budget:
                    budget_exhausted = True
                    break
                # The two strategies have deliberately equal prior probability;
                # strategy selection remains visible in the result for analysis.
                use_pbest = rng.random() < 0.5
                if use_pbest:
                    pbest = population[rng.choice(order[:p_count])]
                    a_idx, b_idx = self._sample_distinct(rng, size, i, 2)
                    # A member of the archive can supply a genuinely old
                    # direction while still preserving feasibility by clipping.
                    donor_b = archive[rng.randrange(len(archive))] if archive and rng.random() < 0.35 else population[b_idx]
                    mutant = [
                        target[d] + f * (pbest[d] - target[d]) + f * (population[a_idx][d] - donor_b[d])
                        for d in range(dim)
                    ]
                    strategy_counts["current_to_pbest1"] += 1
                else:
                    a_idx, b_idx, c_idx = self._sample_distinct(rng, size, i, 3)
                    mutant = [population[a_idx][d] + f * (population[b_idx][d] - population[c_idx][d]) for d in range(dim)]
                    strategy_counts["rand1"] += 1
                mutant = [_clip(value, lo, hi) for value, (lo, hi) in zip(mutant, self.bounds)]
                forced = rng.randrange(dim)
                trial = [mutant[d] if rng.random() < cr or d == forced else target[d] for d in range(dim)]
                trial_fitness = float(objective(trial))
                evaluations += 1
                attempted += 1
                if trial_fitness <= fitness[i]:
                    if archive_limit:
                        archive.append(list(target))
                        if len(archive) > archive_limit:
                            # Remove one uniformly sampled stale point.  The
                            # previous implementation accidentally sampled
                            # only from the overflow count (usually index 0).
                            del archive[rng.randrange(len(archive))]
                    population[i], fitness[i] = trial, trial_fitness
                    successful += 1
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_vector = list(trial)

            success_rate = successful / max(1, attempted)
            success_rate_history.append(success_rate)

            # Preserve an elite while injecting a few bounded points if the
            # population has collapsed.  This switch is intentionally simple,
            # deterministic under the seed, and easy to remove in ablations.
            if self.config.elite_diversity and diversity < self.config.diversity_threshold:
                inject = max(1, int(round(size * 0.10)))
                worst = sorted(range(size), key=fitness.__getitem__, reverse=True)
                for index in worst[:inject]:
                    if budget is not None and evaluations >= budget:
                        budget_exhausted = True
                        break
                    if fitness[index] <= best_fitness:
                        continue
                    candidate = [rng.uniform(lo, hi) for lo, hi in self.bounds]
                    candidate_fitness = float(objective(candidate))
                    evaluations += 1
                    population[index], fitness[index] = candidate, candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_fitness, best_vector = candidate_fitness, list(candidate)

            if best_fitness < previous_best:
                no_improvement = 0
            else:
                no_improvement += 1
            if self.config.stagnation_restart and no_improvement >= self.config.stagnation_generations:
                restart_count += 1
                restart_count_size = max(1, int(round(size * self.config.restart_fraction)))
                worst = sorted(range(size), key=fitness.__getitem__, reverse=True)
                for index in worst[:restart_count_size]:
                    if budget is not None and evaluations >= budget:
                        budget_exhausted = True
                        break
                    if population[index] == best_vector:
                        continue
                    candidate = [rng.uniform(lo, hi) for lo, hi in self.bounds]
                    candidate_fitness = float(objective(candidate))
                    evaluations += 1
                    population[index], fitness[index] = candidate, candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_fitness, best_vector = candidate_fitness, list(candidate)
                no_improvement = 0

            history.append(best_fitness)
            diversity_history.append(self._diversity(population))
            difficulty_history.append(self._difficulty(population, fitness, diversity_history[-1], success_rate))
            if budget_exhausted or (budget is not None and evaluations >= budget):
                break

        return AdaptiveDEResult(
            best_vector=best_vector,
            best_fitness=best_fitness,
            history=history,
            diversity_history=diversity_history,
            success_rate_history=success_rate_history,
            evaluations=evaluations,
            restart_count=restart_count,
            strategy_counts=strategy_counts,
            difficulty_history=difficulty_history,
        )


def plan_continuous_path_adaptive(problem, config: AdaptiveDEConfig | None = None):
    """Optimize a :class:`ContinuousPathProblem` with the research variant.

    The import is intentionally local so this module remains a generic
    optimizer and does not create a circular dependency with the planner.
    """

    if not problem.environment.is_free(problem.start, problem.safety_margin) or not problem.environment.is_free(problem.goal, problem.safety_margin):
        raise ValueError("start and goal must be collision-free and within bounds")
    if not problem.bounds:
        value = float(problem.objective([]))
        result = AdaptiveDEResult([], value, [value], [0.0], [], 1, difficulty_history=[0.0])
    else:
        result = AdaptiveDiversityDE(problem.bounds, config).optimize(problem.objective)
    return problem.decode(result.best_vector), result
