"""L-SHADE single-objective minimizer.

This is an independent implementation of Tanabe and Fukunaga's L-SHADE:
success-history parameter adaptation and an external archive from SHADE are
combined with linear population-size reduction (LPSR).  See:

    R. Tanabe and A. S. Fukunaga, "Improving the Search Performance of SHADE
    Using Linear Population Size Reduction," IEEE CEC 2014,
    DOI: 10.1109/CEC.2014.6900380.

The implementation is deliberately isolated from the existing SHADE baseline.
That keeps comparisons auditable: an experiment can report SHADE and L-SHADE
as distinct algorithms while sharing the same objective and NFE budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Sequence

from .jade_shade import (
    _archive_vector,
    _evaluate,
    _repair_mutant,
    _sample_distinct,
    _sample_f_cauchy,
    _shade_p_count,
    _trim_archive,
    _weighted_lehmer,
    _validate_bounds,
)


Objective = Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class LSHADEConfig:
    """Configuration for the L-SHADE optimizer.

    ``population_size`` is N_init and ``min_population_size`` is N_min in the
    LPSR equation.  If no explicit budget is supplied, the configured number
    of generations is used.  With a strict budget, the population is resized
    from the number of evaluations already consumed, as in the paper's
    evaluation-based schedule.
    """

    population_size: int = 100
    min_population_size: int = 4
    generations: int = 100
    p_best_rate: float = 0.11
    memory_size: int = 6
    initial_f: float = 0.5
    initial_cr: float = 0.5
    archive_size: int | None = None
    max_evaluations: int | None = None
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.min_population_size < 4 or self.min_population_size > self.population_size:
            raise ValueError("min_population_size must be in [4, population_size]")
        if self.generations < 0:
            raise ValueError("generations cannot be negative")
        if not 0.0 < self.p_best_rate <= 1.0:
            raise ValueError("p_best_rate must be in (0, 1]")
        if self.memory_size < 1:
            raise ValueError("memory_size must be positive")
        if not 0.0 < self.initial_f <= 1.0 or not 0.0 <= self.initial_cr <= 1.0:
            raise ValueError("initial_f must be in (0, 1] and initial_cr in [0, 1]")
        if self.archive_size is not None and self.archive_size < 0:
            raise ValueError("archive_size cannot be negative")
        if self.max_evaluations is not None and self.max_evaluations < self.population_size:
            raise ValueError("max_evaluations must cover the initial population")


@dataclass
class LSHADEResult:
    best_vector: list[float]
    best_fitness: float
    history: list[float]
    evaluations: int
    population_size_history: list[int] = field(default_factory=list)
    memory_f_history: list[list[float]] = field(default_factory=list)
    memory_cr_history: list[list[float]] = field(default_factory=list)
    success_history: list[int] = field(default_factory=list)
    archive_history: list[int] = field(default_factory=list)


class LSHADE:
    """L-SHADE with strict objective-evaluation accounting."""

    def __init__(self, bounds: Sequence[tuple[float, float]], config: LSHADEConfig | None = None):
        self.bounds = _validate_bounds(bounds)
        self.config = config or LSHADEConfig()

    def _budget(self) -> int:
        cfg = self.config
        return cfg.max_evaluations or cfg.population_size * (cfg.generations + 1)

    def _target_population_size(self, evaluations: int, budget: int) -> int:
        """LPSR size NFE-schedule, rounded to nearest integer.

        The first generation is evaluated at ``N_init``.  Therefore the
        reduction schedule starts after that generation, as in Algorithm 1 of
        Tanabe and Fukunaga, rather than shrinking the initial population
        before it has had a full generation to produce offspring.
        """

        cfg = self.config
        if evaluations <= cfg.population_size:
            return cfg.population_size
        if budget <= cfg.population_size:
            return cfg.min_population_size
        fraction = min(1.0, max(0.0, evaluations / float(budget)))
        target = cfg.population_size + (cfg.min_population_size - cfg.population_size) * fraction
        rounded = math.floor(target + 0.5)
        return max(cfg.min_population_size, min(cfg.population_size, rounded))

    @staticmethod
    def _shrink_population(
        rng: random.Random,
        population: list[list[float]],
        fitness: list[float],
        target_size: int,
    ) -> tuple[list[list[float]], list[float]]:
        if target_size >= len(population):
            return population, fitness
        # L-SHADE removes the worst individuals after a reduction. Ties are
        # randomly ordered so equal-fitness populations do not bias indices.
        order = list(range(len(population)))
        rng.shuffle(order)
        order.sort(key=fitness.__getitem__)
        keep = order[:target_size]
        return [list(population[i]) for i in keep], [fitness[i] for i in keep]

    def optimize(self, objective: Objective) -> LSHADEresult:
        cfg = self.config
        rng = random.Random(cfg.seed)
        dim = len(self.bounds)
        population = [[rng.uniform(lo, hi) for lo, hi in self.bounds] for _ in range(cfg.population_size)]
        fitness = [_evaluate(objective, vector) for vector in population]
        evaluations = len(population)
        budget = self._budget()
        archive_limit = cfg.archive_size if cfg.archive_size is not None else cfg.population_size
        archive: list[list[float]] = []
        memory_f = [cfg.initial_f] * cfg.memory_size
        memory_cr = [cfg.initial_cr] * cfg.memory_size
        memory_index = 0
        history = [min(fitness)]
        population_history = [len(population)]
        memory_f_history = [list(memory_f)]
        memory_cr_history = [list(memory_cr)]
        success_history: list[int] = []
        archive_history = [0]

        generation = 0
        # An explicit NFE budget is the fairness contract used by the
        # benchmark.  Because LPSR shrinks the population, a fixed number of
        # generations can consume substantially fewer evaluations than DE or
        # SHADE.  When max_evaluations is supplied, keep producing generations
        # until that budget is exhausted; the generations field remains the
        # default-run stopping rule when no explicit budget is requested.
        while evaluations < budget and (
                cfg.max_evaluations is not None or generation < cfg.generations):
            size = len(population)
            parent_population = [list(vector) for vector in population]
            parent_fitness = list(fitness)
            parent_archive = [list(vector) for vector in archive]
            order = sorted(range(size), key=parent_fitness.__getitem__)
            successful_f: list[float] = []
            successful_cr: list[float] = []
            improvements: list[float] = []
            next_population = [list(vector) for vector in parent_population]
            next_fitness = list(parent_fitness)
            for i, target in enumerate(parent_population):
                if evaluations >= budget:
                    break
                memory_slot = rng.randrange(cfg.memory_size)
                cr = min(1.0, max(0.0, rng.gauss(memory_cr[memory_slot], 0.1)))
                f = _sample_f_cauchy(rng, memory_f[memory_slot])
                p_count = _shade_p_count(rng, size, cfg.p_best_rate)
                pbest = parent_population[rng.choice(order[:p_count])]
                r1_index = _sample_distinct(rng, size, {i})
                r1 = parent_population[r1_index]
                r2 = _archive_vector(rng, parent_archive, parent_population, {i, r1_index})
                mutant = [target[d] + f * (pbest[d] - target[d]) + f * (r1[d] - r2[d]) for d in range(dim)]
                mutant = _repair_mutant(mutant, self.bounds, target)
                forced = rng.randrange(dim)
                trial = [mutant[d] if rng.random() < cr or d == forced else target[d] for d in range(dim)]
                trial_fitness = _evaluate(objective, trial)
                evaluations += 1
                if trial_fitness <= parent_fitness[i]:
                    next_population[i], next_fitness[i] = trial, trial_fitness
                if trial_fitness < parent_fitness[i]:
                    improvement = parent_fitness[i] - trial_fitness
                    successful_f.append(f)
                    successful_cr.append(cr)
                    improvements.append(improvement)
                    if archive_limit:
                        archive.append(list(target))
            population, fitness = next_population, next_fitness
            _trim_archive(rng, archive, min(archive_limit, len(population)))
            if successful_f:
                memory_f[memory_index] = _weighted_lehmer(successful_f, improvements)
                total_improvement = sum(improvements)
                memory_cr[memory_index] = sum(w * value for w, value in zip(improvements, successful_cr)) / total_improvement
                memory_index = (memory_index + 1) % cfg.memory_size
            # LPSR is applied after a complete generation.  The reduced
            # population is then used by the following generation.
            target_size = self._target_population_size(evaluations, budget)
            population, fitness = self._shrink_population(rng, population, fitness, target_size)
            _trim_archive(rng, archive, min(archive_limit, len(population)))
            history.append(min(fitness))
            population_history.append(len(population))
            memory_f_history.append(list(memory_f))
            memory_cr_history.append(list(memory_cr))
            success_history.append(len(successful_f))
            archive_history.append(len(archive))
            generation += 1

        best = min(range(len(population)), key=fitness.__getitem__)
        return LSHADEResult(
            list(population[best]), fitness[best], history, evaluations,
            population_history, memory_f_history, memory_cr_history,
            success_history, archive_history,
        )


def plan_continuous_path_lshade(problem, config: LSHADEConfig | None = None):
    """Adapter matching the JADE/SHADE continuous-planner wrappers."""

    if not problem.bounds:
        value = _evaluate(problem.objective, [])
        cfg = config or LSHADEConfig()
        result = LSHADEResult([], value, [value], 1, [cfg.min_population_size], [[cfg.initial_f] * cfg.memory_size], [[cfg.initial_cr] * cfg.memory_size], [], [0])
        return problem.decode([]), result
    result = LSHADE(problem.bounds, config).optimize(problem.objective)
    return problem.decode(result.best_vector), result


__all__ = ["LSHADEConfig", "LSHADEResult", "LSHADE", "plan_continuous_path_lshade"]
