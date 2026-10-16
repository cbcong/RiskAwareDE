"""Paper-based JADE (2009) and SHADE (2013) minimizers.

Sources (this is an independent implementation, not verified official code):
* Zhang and Sanderson, JADE, DOI 10.1109/TEVC.2009.2014613,
  Sec. II (midpoint bound repair), Sec. IV, Eqs. (9), (11), (12), Table I.
* Tanabe and Fukunaga, SHADE, DOI 10.1109/CEC.2013.6557555,
  https://www.metahack.org/CEC2013-SHADE.pdf, Eqs. (13)--(20), Algorithm 1.

JADE uses unweighted successful CR/F means; SHADE uses improvement-weighted
means and per-target randomized p. Population and archive donors are frozen
for each generation. This follows SHADE Algorithm 1 and JADE Sec. IV-A's
description of adding parents after a generation (JADE Table I writes archive
insertion inside its loop, so that presentation is ambiguous).

Explicit implementation choices: fractional p*N is rounded up; JADE permits
one p-best candidate, SHADE at least two. For SHADE populations below 10, the
otherwise empty [2/N, 0.2] interval is collapsed to 2/N. The default population
20 serves this project's equal-budget runs, not the original CEC protocol.
H defaults to 100 as in SHADE 2013; custom H/p/archive sizes are variants.
Finite objective values are required. A budget may terminate a partial final
generation, whose evaluated offspring are selected and recorded normally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Sequence


Objective = Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class JADEConfig:
    population_size: int = 20
    generations: int = 100
    p_best_rate: float = 0.05
    mu_f: float = 0.5
    mu_cr: float = 0.5
    learning_rate: float = 0.1
    archive_size: int | None = None
    max_evaluations: int | None = None
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.generations < 0:
            raise ValueError("generations cannot be negative")
        if not 0.0 < self.p_best_rate <= 1.0:
            raise ValueError("p_best_rate must be in (0, 1]")
        if not 0.0 < self.mu_f <= 1.0 or not 0.0 <= self.mu_cr <= 1.0:
            raise ValueError("mu_f must be in (0, 1] and mu_cr in [0, 1]")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.archive_size is not None and self.archive_size < 0:
            raise ValueError("archive_size cannot be negative")
        if self.max_evaluations is not None and self.max_evaluations < self.population_size:
            raise ValueError("max_evaluations must cover the initial population")


@dataclass(frozen=True)
class SHADEConfig:
    population_size: int = 20
    generations: int = 100
    # Retained field name: this is the UPPER limit for random p_i, not fixed p.
    p_best_rate: float = 0.2
    memory_size: int = 100
    initial_f: float = 0.5
    initial_cr: float = 0.5
    archive_size: int | None = None
    max_evaluations: int | None = None
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
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
class JADEResult:
    best_vector: list[float]
    best_fitness: float
    history: list[float]
    evaluations: int
    mu_f_history: list[float] = field(default_factory=list)
    mu_cr_history: list[float] = field(default_factory=list)
    success_history: list[int] = field(default_factory=list)
    archive_history: list[int] = field(default_factory=list)


@dataclass
class SHADEResult:
    best_vector: list[float]
    best_fitness: float
    history: list[float]
    evaluations: int
    memory_f_history: list[list[float]] = field(default_factory=list)
    memory_cr_history: list[list[float]] = field(default_factory=list)
    success_history: list[int] = field(default_factory=list)
    archive_history: list[int] = field(default_factory=list)


def _validate_bounds(bounds: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not bounds:
        raise ValueError("at least one decision variable is required")
    output = [(float(lo), float(hi)) for lo, hi in bounds]
    if any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi for lo, hi in output):
        raise ValueError("every bound must be finite and lower < upper")
    return output


def _repair_mutant(vector: Sequence[float], bounds: Sequence[tuple[float, float]], parent: Sequence[float]) -> list[float]:
    """Repair a mutant using the midpoint of the violated bound and parent.

    JADE Sec. II, pp. 946--947; also used here for SHADE, whose 2013 paper
    does not separately specify a decision-variable bound repair rule.
    """

    repaired: list[float] = []
    for index, (value, (lo, hi)) in enumerate(zip(vector, bounds)):
        value = float(value)
        if value < lo:
            value = lo / 2.0 + float(parent[index]) / 2.0
        elif value > hi:
            value = hi / 2.0 + float(parent[index]) / 2.0
        repaired.append(value)
    return repaired


def _clip_vector(
    vector: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    parent: Sequence[float] | None = None,
) -> list[float]:
    """Compatibility wrapper for JADE's midpoint bound repair.

    ``parent`` is required for the paper's repair rule; accepting ``None``
    keeps this helper useful for callers that only need ordinary clipping.
    """
    if parent is None:
        return [min(float(hi), max(float(lo), float(value))) for value, (lo, hi) in zip(vector, bounds)]
    return _repair_mutant(vector, bounds, parent)


def _sample_f_cauchy(rng: random.Random, location: float) -> float:
    """Sample positive Cauchy value, as used by JADE/SHADE."""

    while True:
        value = location + 0.1 * math.tan(math.pi * (rng.random() - 0.5))
        if value > 0.0 and math.isfinite(value):
            return min(1.0, value)


def _sample_distinct(rng: random.Random, count: int, excluded: set[int]) -> int:
    choices = [index for index in range(count) if index not in excluded]
    if not choices:
        raise ValueError("population is too small for distinct sampling")
    return rng.choice(choices)


def _archive_vector(
    rng: random.Random,
    archive: list[list[float]],
    population: list[list[float]],
    excluded: set[int],
) -> list[float]:
    """Select from P union A, excluding the target and r1 population index."""

    candidates = [population[index] for index in range(len(population)) if index not in excluded]
    candidates.extend(archive)
    if not candidates:
        raise ValueError("no donor candidates available")
    return list(rng.choice(candidates))


def _weighted_lehmer(values: Sequence[float], weights: Sequence[float]) -> float:
    """Improvement-weighted Lehmer mean used by the original SHADE paper."""

    denominator = sum(weight * value for value, weight in zip(values, weights))
    return sum(weight * value * value for value, weight in zip(values, weights)) / denominator


def _evaluate(objective: Objective, vector: Sequence[float]) -> float:
    value = float(objective(vector))
    if not math.isfinite(value):
        raise ValueError("JADE/SHADE objective must return a finite value")
    return value


def _trim_archive(rng: random.Random, archive: list[list[float]], limit: int) -> None:
    while len(archive) > limit:
        del archive[rng.randrange(len(archive))]


def _shade_p_count(rng: random.Random, size: int, upper_rate: float) -> int:
    """SHADE Eq. (20), with documented handling for the empty small-N interval."""
    p_min = 2.0 / size
    p = rng.uniform(p_min, max(p_min, upper_rate))
    return min(size, max(2, math.ceil(p * size)))


class JADE:
    """JADE minimizer with strict budget and auditable parameter traces."""

    def __init__(self, bounds: Sequence[tuple[float, float]], config: JADEConfig | None = None):
        self.bounds = _validate_bounds(bounds)
        self.config = config or JADEConfig()

    def optimize(self, objective: Objective) -> JADEResult:
        rng = random.Random(self.config.seed)
        size = self.config.population_size
        dim = len(self.bounds)
        population = [[rng.uniform(lo, hi) for lo, hi in self.bounds] for _ in range(size)]
        fitness = [_evaluate(objective, vector) for vector in population]
        evaluations = size
        budget = self.config.max_evaluations or size * (self.config.generations + 1)
        archive_limit = self.config.archive_size if self.config.archive_size is not None else size
        archive: list[list[float]] = []
        mu_f, mu_cr = self.config.mu_f, self.config.mu_cr
        history = [min(fitness)]
        mu_f_history = [mu_f]
        mu_cr_history = [mu_cr]
        success_history: list[int] = []
        archive_history = [0]

        for _ in range(self.config.generations):
            if evaluations >= budget:
                break
            # Freeze the generation before creating any trial vectors.  This is
            # essential for DE: p-best and all donors must come from P_G, while
            # replacements form P_{G+1} only after the generation is complete.
            parent_population = [list(vector) for vector in population]
            parent_fitness = list(fitness)
            order = sorted(range(size), key=parent_fitness.__getitem__)
            p_count = max(1, int(math.ceil(self.config.p_best_rate * size)))
            parent_archive = [list(vector) for vector in archive]
            successful_f: list[float] = []
            successful_cr: list[float] = []
            next_population = [list(vector) for vector in parent_population]
            next_fitness = list(parent_fitness)
            for i, target in enumerate(parent_population):
                if evaluations >= budget:
                    break
                cr = min(1.0, max(0.0, rng.gauss(mu_cr, 0.1)))
                f = _sample_f_cauchy(rng, mu_f)
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
                # JADE Table I keeps the parent on ties.
                if trial_fitness < parent_fitness[i]:
                    successful_f.append(f)
                    successful_cr.append(cr)
                    if archive_limit:
                        archive.append(list(target))
                    next_population[i], next_fitness[i] = trial, trial_fitness
            population, fitness = next_population, next_fitness
            _trim_archive(rng, archive, archive_limit)
            # Original JADE uses arithmetic mean for CR and Lehmer mean for F.
            # Improvement weighting belongs to SHADE and later JADE extensions.
            if successful_f:
                mean_f = sum(value * value for value in successful_f) / sum(successful_f)
                mean_cr = sum(successful_cr) / len(successful_cr)
                c = self.config.learning_rate
                mu_f = (1.0 - c) * mu_f + c * mean_f
                mu_cr = (1.0 - c) * mu_cr + c * mean_cr
            history.append(min(fitness))
            mu_f_history.append(mu_f)
            mu_cr_history.append(mu_cr)
            success_history.append(len(successful_f))
            archive_history.append(len(archive))
        best = min(range(size), key=fitness.__getitem__)
        return JADEResult(list(population[best]), fitness[best], history, evaluations, mu_f_history, mu_cr_history, success_history, archive_history)


class SHADE:
    """SHADE minimizer with cyclic success-history memories."""

    def __init__(self, bounds: Sequence[tuple[float, float]], config: SHADEConfig | None = None):
        self.bounds = _validate_bounds(bounds)
        self.config = config or SHADEConfig()

    def optimize(self, objective: Objective) -> SHADEResult:
        rng = random.Random(self.config.seed)
        size = self.config.population_size
        dim = len(self.bounds)
        population = [[rng.uniform(lo, hi) for lo, hi in self.bounds] for _ in range(size)]
        fitness = [_evaluate(objective, vector) for vector in population]
        evaluations = size
        budget = self.config.max_evaluations or size * (self.config.generations + 1)
        archive_limit = self.config.archive_size if self.config.archive_size is not None else size
        archive: list[list[float]] = []
        memory_f = [self.config.initial_f] * self.config.memory_size
        memory_cr = [self.config.initial_cr] * self.config.memory_size
        memory_index = 0
        history = [min(fitness)]
        memory_f_history = [list(memory_f)]
        memory_cr_history = [list(memory_cr)]
        success_history: list[int] = []
        archive_history = [0]

        for _ in range(self.config.generations):
            if evaluations >= budget:
                break
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
                memory_slot = rng.randrange(self.config.memory_size)
                cr = min(1.0, max(0.0, rng.gauss(memory_cr[memory_slot], 0.1)))
                f = _sample_f_cauchy(rng, memory_f[memory_slot])
                p_count = _shade_p_count(rng, size, self.config.p_best_rate)
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
                # SHADE Algorithm 1 lines 21--24: ties do not teach memory or
                # enter the archive, even though they replace the parent.
                if trial_fitness < parent_fitness[i]:
                    improvement = parent_fitness[i] - trial_fitness
                    successful_f.append(f)
                    successful_cr.append(cr)
                    improvements.append(improvement)
                    if archive_limit:
                        archive.append(list(target))
            population, fitness = next_population, next_fitness
            _trim_archive(rng, archive, archive_limit)
            if successful_f:
                # SHADE's weighted means use normalized improvements; the
                # equivalent normalized form is invariant to common scaling.
                memory_f[memory_index] = _weighted_lehmer(successful_f, improvements)
                total_improvement = sum(improvements)
                memory_cr[memory_index] = sum(weight * value for weight, value in zip(improvements, successful_cr)) / total_improvement
                memory_index = (memory_index + 1) % self.config.memory_size
            history.append(min(fitness))
            memory_f_history.append(list(memory_f))
            memory_cr_history.append(list(memory_cr))
            success_history.append(len(successful_f))
            archive_history.append(len(archive))
        best = min(range(size), key=fitness.__getitem__)
        return SHADEResult(list(population[best]), fitness[best], history, evaluations, memory_f_history, memory_cr_history, success_history, archive_history)


def plan_continuous_path_jade(problem, config: JADEConfig | None = None):
    if not problem.bounds:
        value = _evaluate(problem.objective, [])
        return problem.decode([]), JADEResult([], value, [value], 1, [config.mu_f if config else 0.5], [config.mu_cr if config else 0.5], [], [0])
    result = JADE(problem.bounds, config).optimize(problem.objective)
    return problem.decode(result.best_vector), result


def plan_continuous_path_shade(problem, config: SHADEConfig | None = None):
    if not problem.bounds:
        value = _evaluate(problem.objective, [])
        cfg = config or SHADEConfig()
        return problem.decode([]), SHADEResult([], value, [value], 1, [[cfg.initial_f] * cfg.memory_size], [[cfg.initial_cr] * cfg.memory_size], [], [0])
    result = SHADE(problem.bounds, config).optimize(problem.objective)
    return problem.decode(result.best_vector), result


__all__ = [
    "JADEConfig", "JADEResult", "JADE", "SHADEConfig", "SHADEResult", "SHADE",
    "plan_continuous_path_jade", "plan_continuous_path_shade", "_clip_vector", "_weighted_lehmer",
]
