"""Minimal, deterministic Differential Evolution optimizer.

This is a research baseline. It exposes the common mutation/crossover/selection
hooks while keeping the implementation compact enough to modify in experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Sequence


Objective = Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class DEConfig:
    population_size: int = 20
    generations: int = 100
    mutation_factor: float = 0.8
    crossover_rate: float = 0.9
    seed: int | None = 0
    max_evaluations: int | None = None

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.generations < 0:
            raise ValueError("generations cannot be negative")
        if self.max_evaluations is not None and self.max_evaluations < self.population_size:
            raise ValueError("max_evaluations must cover initialization")
        if not 0 < self.mutation_factor <= 2:
            raise ValueError("mutation_factor must be in (0, 2]")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be in [0, 1]")


@dataclass
class DEResult:
    best_vector: list[float]
    best_fitness: float
    history: list[float]
    evaluations: int


class DifferentialEvolution:
    """Classic DE/rand/1/bin minimizer over box constraints."""

    def __init__(self, bounds: Sequence[tuple[float, float]], config: DEConfig | None = None):
        if not bounds:
            raise ValueError("at least one decision variable is required")
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        if any(lo >= hi for lo, hi in self.bounds):
            raise ValueError("every lower bound must be smaller than upper bound")
        self.config = config or DEConfig()

    def optimize(self, objective: Objective) -> DEResult:
        rng = random.Random(self.config.seed)
        dim = len(self.bounds)
        population = [[rng.uniform(lo, hi) for lo, hi in self.bounds] for _ in range(self.config.population_size)]
        fitness = [float(objective(x)) for x in population]
        evaluations = len(population)
        history = [min(fitness)]
        budget = self.config.max_evaluations or self.config.population_size * (self.config.generations + 1)

        for _ in range(self.config.generations):
            if evaluations >= budget:
                break
            parents = list(population)
            for i, target in enumerate(parents):
                if evaluations >= budget:
                    break
                choices = [j for j in range(len(population)) if j != i]
                a_idx, b_idx, c_idx = rng.sample(choices, 3)
                a, b, c = parents[a_idx], parents[b_idx], parents[c_idx]
                mutant = [a[d] + self.config.mutation_factor * (b[d] - c[d]) for d in range(dim)]
                mutant = [max(lo, min(hi, value)) for value, (lo, hi) in zip(mutant, self.bounds)]
                forced = rng.randrange(dim)
                trial = [
                    mutant[d] if rng.random() < self.config.crossover_rate or d == forced else target[d]
                    for d in range(dim)
                ]
                trial_fitness = float(objective(trial))
                evaluations += 1
                if trial_fitness <= fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
            history.append(min(fitness))

        best_index = min(range(len(population)), key=fitness.__getitem__)
        return DEResult(list(population[best_index]), fitness[best_index], history, evaluations)
