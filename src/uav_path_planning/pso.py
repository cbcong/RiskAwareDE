"""Compact particle swarm optimization baseline for continuous paths."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Sequence

from .continuous_planner import ContinuousPathProblem
from .models import Path3D
from .planning import PlannerResult


@dataclass(frozen=True)
class PSOConfig:
    """CPU-friendly PSO settings.

    ``particles * iterations`` is the dominant evaluation budget.  A fixed
    seed makes comparisons reproducible across planners.
    """

    particles: int = 24
    iterations: int = 100
    inertia: float = 0.72
    cognitive: float = 1.49
    social: float = 1.49
    velocity_fraction: float = 0.25
    seed: int | None = 0
    max_evaluations: int | None = None

    def __post_init__(self) -> None:
        if self.particles < 2:
            raise ValueError("particles must be at least 2")
        if self.iterations < 0:
            raise ValueError("iterations cannot be negative")
        if self.inertia < 0 or self.cognitive < 0 or self.social < 0:
            raise ValueError("PSO coefficients cannot be negative")
        if self.velocity_fraction <= 0:
            raise ValueError("velocity_fraction must be positive")
        if self.max_evaluations is not None and self.max_evaluations < self.particles:
            raise ValueError("max_evaluations must cover initialization")


class ParticleSwarmPlanner:
    """Particle swarm path planner over intermediate waypoint coordinates."""

    name = "pso"

    def __init__(self, problem: ContinuousPathProblem, config: PSOConfig | None = None):
        self.problem = problem
        self.config = config or PSOConfig()

    def plan(
        self,
        start: Sequence[float] | None = None,
        goal: Sequence[float] | None = None,
        *,
        environment=None,
        grid=None,
    ) -> PlannerResult:
        del environment, grid
        if start is not None and tuple(float(x) for x in start) != self.problem.start:
            raise ValueError("start does not match the configured ContinuousPathProblem")
        if goal is not None and tuple(float(x) for x in goal) != self.problem.goal:
            raise ValueError("goal does not match the configured ContinuousPathProblem")
        if not self.problem.environment.is_free(self.problem.start, self.problem.safety_margin) or not self.problem.environment.is_free(self.problem.goal, self.problem.safety_margin):
            raise ValueError("start and goal must be collision-free and within bounds")

        if not self.problem.bounds:
            path = self.problem.decode([])
            cost = self.problem.objective([])
            collisions = path.collision_count(self.problem.environment, self.problem.segment_step, self.problem.safety_margin)
            return PlannerResult(
                path,
                cost,
                collisions == 0,
                evaluations=1,
                history=[cost],
                metadata={"algorithm": self.name, "collision_count": collisions},
            )

        started = time.perf_counter()
        rng = random.Random(self.config.seed)
        bounds = self.problem.bounds
        dimension = len(bounds)
        vmax = [self.config.velocity_fraction * (hi - lo) for lo, hi in bounds]
        particles = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(self.config.particles)]
        velocities = [[rng.uniform(-v, v) for v in vmax] for _ in particles]
        fitness = [float(self.problem.objective(position)) for position in particles]
        evaluations = len(particles)
        personal_best = [list(position) for position in particles]
        personal_fitness = list(fitness)
        best_index = min(range(len(particles)), key=fitness.__getitem__)
        global_best = list(particles[best_index])
        global_fitness = fitness[best_index]
        history = [global_fitness]
        budget = self.config.max_evaluations or self.config.particles * (self.config.iterations + 1)

        for _ in range(self.config.iterations):
            if evaluations >= budget:
                break
            for i, position in enumerate(particles):
                if evaluations >= budget:
                    break
                velocity = velocities[i]
                for d, (lo, hi) in enumerate(bounds):
                    r1, r2 = rng.random(), rng.random()
                    velocity[d] = (
                        self.config.inertia * velocity[d]
                        + self.config.cognitive * r1 * (personal_best[i][d] - position[d])
                        + self.config.social * r2 * (global_best[d] - position[d])
                    )
                    velocity[d] = max(-vmax[d], min(vmax[d], velocity[d]))
                    position[d] = max(lo, min(hi, position[d] + velocity[d]))
                value = float(self.problem.objective(position))
                evaluations += 1
                if value <= personal_fitness[i]:
                    personal_best[i] = list(position)
                    personal_fitness[i] = value
                if value <= global_fitness:
                    global_best = list(position)
                    global_fitness = value
            history.append(global_fitness)

        path = self.problem.decode(global_best)
        return PlannerResult(
            path=path,
            cost=global_fitness,
            success=path.collision_count(self.problem.environment, self.problem.segment_step, self.problem.safety_margin) == 0,
            evaluations=evaluations,
            runtime_seconds=time.perf_counter() - started,
            history=history,
            metadata={
                "algorithm": self.name,
                "collision_count": path.collision_count(self.problem.environment, self.problem.segment_step, self.problem.safety_margin),
                "particles": self.config.particles,
                "iterations": self.config.iterations,
                "max_evaluations": budget,
            },
        )


def plan_pso(problem: ContinuousPathProblem, config: PSOConfig | None = None) -> tuple[Path3D, PlannerResult]:
    """Compatibility helper returning the path and shared result separately."""

    result = ParticleSwarmPlanner(problem, config).plan()
    if result.path is None:  # pragma: no cover - planner always returns a path for valid bounds
        raise RuntimeError("PSO did not produce a path")
    return result.path, result
