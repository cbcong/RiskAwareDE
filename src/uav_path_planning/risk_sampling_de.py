"""Gaussian bank designs for equal-cost risk-aware DE experiments.

Latin hypercube and antithetic sampling are established variance-reduction
tools, not new planning algorithms. Only the frozen bank changes here. The
optimizer's risk gate, search, geometry, and evaluation accounting are reused.
Rows of these banks are dependent, so their empirical rates must not be
interpreted using independent-binomial confidence certificates.
"""

from __future__ import annotations

import hashlib
import json
import random
from statistics import NormalDist

from .continuous_planner import ContinuousPathProblem
from .models import Environment3D, ObstacleBox
from .risk_aware_de import RiskAwareDE, RiskAwareDEConfig


def standardized_gaussian_bank(
    samples: int, dimensions: int, seed: int, design: str,
) -> list[list[float]]:
    """Generate one frozen bank, with independent coordinate randomization.

    ``latin_hypercube`` uses one independently jittered point from each of
    K equal-probability strata in every coordinate. ``antithetic_lhs`` pairs
    each row with its exact negative while retaining one point in every
    coordinate stratum. Random signs are independent per pair/coordinate;
    thus pairing does not force all obstacle coordinates to move together.
    Every individual row has independent standard-normal coordinates over
    randomized designs, but the rows within one bank are dependent.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 0:
        raise ValueError("dimensions must be a non-negative integer")
    if design not in {"latin_hypercube", "antithetic_lhs"}:
        raise ValueError("unsupported Gaussian bank design")
    if design == "antithetic_lhs" and samples % 2:
        raise ValueError("antithetic_lhs requires an even sample count")
    rng = random.Random(seed)
    normal = NormalDist()
    bank = [[0.0] * dimensions for _ in range(samples)]
    for dimension in range(dimensions):
        strata = list(range(samples if design == "latin_hypercube" else samples // 2))
        rng.shuffle(strata)
        for row, stratum in enumerate(strata):
            # random() can equal zero. Midpoint fallback only handles that
            # measure-zero numerical endpoint; it avoids an infinite quantile.
            jitter = rng.random()
            if jitter == 0.0:
                jitter = 0.5
            quantile = (stratum + jitter) / samples
            value = normal.inv_cdf(quantile)
            if design == "antithetic_lhs":
                if rng.random() < 0.5:
                    value = -value
                bank[row][dimension] = value
                bank[row + samples // 2][dimension] = -value
            else:
                bank[row][dimension] = value
    return bank


class SamplingRiskAwareDE(RiskAwareDE):
    """RiskAwareDE with a declared, reproducible Gaussian bank design."""

    def __init__(self, problem: ContinuousPathProblem, config: RiskAwareDEConfig, *, design: str):
        if config.selection_mode == "ucb_budgeted":
            raise ValueError("dependent bank samples cannot use the independent-binomial UCB gate")
        super().__init__(problem.bounds, problem, config)
        seed = config.perturbation_seed
        if seed is None:
            planner_seed = config.base.seed if config.base.seed is not None else 0
            seed = planner_seed ^ 0x5EEDC0DE
        self.standardized_bank = standardized_gaussian_bank(
            config.perturbation_samples, 3 * len(problem.environment.obstacles), seed, design,
        )
        sigmas = config.perturbation_sigmas
        if sigmas is None:
            sigmas = (config.perturbation_sigma,)
        self._perturbed_banks = []
        for sigma in sigmas:
            environments = []
            for row in self.standardized_bank:
                obstacles = []
                for index, obstacle in enumerate(problem.environment.obstacles):
                    shift = row[3 * index:3 * index + 3]
                    obstacles.append(ObstacleBox(
                        tuple(a + sigma * d for a, d in zip(obstacle.minimum, shift)),
                        tuple(b + sigma * d for b, d in zip(obstacle.maximum, shift)),
                    ))
                environments.append(Environment3D(problem.environment.bounds, obstacles))
            self._perturbed_banks.append(environments)
        self._perturbed = [env for bank in self._perturbed_banks for env in bank]
        encoded = json.dumps(self.standardized_bank, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.extension_metadata = {
            "extension": "risk_sampling_de", "bank_design": design,
            "bank_seed": seed, "samples_per_scale": config.perturbation_samples,
            "bank_dimensions": 3 * len(problem.environment.obstacles),
            "bank_sigmas": list(sigmas),
            "standardized_bank_sha256": hashlib.sha256(encoded).hexdigest(),
            "standardized_bank": self.standardized_bank,
            "gaussian_marginals": True, "independent_bank_rows": False,
            "additional_collision_checks": 0,
        }


def create_optimizer(
    problem: ContinuousPathProblem, base_config: RiskAwareDEConfig, options: dict,
) -> SamplingRiskAwareDE:
    unknown = set(options) - {"bank_design"}
    if unknown:
        raise ValueError(f"unknown sampling options: {sorted(unknown)}")
    return SamplingRiskAwareDE(
        problem, base_config, design=options.get("bank_design", "latin_hypercube"),
    )
