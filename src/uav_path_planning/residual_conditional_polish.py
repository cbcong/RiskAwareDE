"""Fixed-:math:`z` conditional risk with a residual LHS uncertainty bank.

This module is an experiment *adapter*, rather than a new optimizer.  The
conditional Gaussian integral already removes one translation coordinate
(the integrated ``z`` coordinate) analytically.  Sampling that coordinate as
well is therefore wasted randomisation.  ``ResidualConditionalRiskDE`` uses
the exact same conditional estimator and selection relation as
``ConditionalRiskDE`` but generates a frozen bank only for the two residual
coordinates per obstacle.  The residual bank can be ordinary Latin
hypercube or paired antithetic Latin hypercube.

The terminal projected-polish implementation is inherited unchanged.  The
only replacement is the evaluator factory, so every proposal, terminal
trial, chance gate and NFE accounting rule remains the established one.  The
``beta=.05`` option adds the already implemented soft in-gate risk tie-break;
``beta=0`` is an exact residual-bank control without that tie-break.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

from .conditional_projected_polish import ConditionalProjectedPolish
from .conditional_risk_de import ConditionalRiskDE
from .risk_aware_de import RiskAwareDEConfig, RiskEvaluation
from .risk_sampling_de import standardized_gaussian_bank


_DESIGNS = {"latin_hypercube", "antithetic_lhs"}
_ALLOWED_BETAS = (0.0, 0.05)


def _validate_design(design: str) -> str:
    if design not in _DESIGNS:
        raise ValueError(
            "bank_design must be 'latin_hypercube' or 'antithetic_lhs'"
        )
    return design


def _bank_seed(config: RiskAwareDEConfig) -> int:
    """Use the established uncertainty-seed fallback without new RNG draws."""

    if config.perturbation_seed is not None:
        return config.perturbation_seed
    planner_seed = config.base.seed if config.base.seed is not None else 0
    return planner_seed ^ 0x5EEDC0DE


def residual_standardized_gaussian_bank(
    samples: int,
    obstacles: int,
    seed: int,
    design: str = "latin_hypercube",
) -> list[list[float]]:
    """Return a deterministic bank with exactly two coordinates per obstacle.

    The helper deliberately delegates stratum construction to the existing
    ``standardized_gaussian_bank`` implementation.  Consequently the LHS and
    antithetic-LHS randomisation, including its seed stream and odd-sample
    validation, are exactly the already tested bank design.  The returned
    columns are ``x_1, y_1, ..., x_m, y_m``; the integrated ``z`` coordinate
    is not sampled.
    """

    if isinstance(obstacles, bool) or not isinstance(obstacles, int) or obstacles < 0:
        raise ValueError("obstacles must be a non-negative integer")
    _validate_design(design)
    return standardized_gaussian_bank(samples, 2 * obstacles, seed, design)


def _residual_shifts(
    bank: list[list[float]], obstacles: int, sigma: float
) -> list[list[list[float]]]:
    """Convert standardized residual draws to physical obstacle translations.

    The analytic z integral uses ``sigma`` separately, so its placeholder is
    zero. The sampled x/y coordinates must use that same uncertainty scale;
    leaving them standardized would instead model (1, 1, sigma) uncertainty.
    """

    return [
        [
            [sigma * row[2 * index], sigma * row[2 * index + 1], 0.0]
            for index in range(obstacles)
        ]
        for row in bank
    ]


class ResidualConditionalRiskDE(ConditionalRiskDE):
    """ConditionalRiskDE using a frozen two-dimensional residual bank."""

    def __init__(self, problem, config: RiskAwareDEConfig, *, bank_design: str = "latin_hypercube"):
        _validate_design(bank_design)
        if config.perturbation_sigmas is not None or config.selection_mode == "ucb_budgeted":
            raise ValueError("Conditional estimator currently needs one scale and no binomial UCB")
        # Reproduce the small state initialisation performed by RiskAwareDE,
        # but do not allocate its unused 3-D binary bank.  The inherited
        # ConditionalRiskDE._evaluate and all selection methods use only these
        # fields plus ``shifts`` below.  Avoiding the provisional bank matters
        # in a long factorial run: each optimizer instance constructs exactly
        # the declared two-residual-coordinate random stream.
        bounds = [(float(lo), float(hi)) for lo, hi in problem.bounds]
        if not bounds:
            raise ValueError("at least one decision variable is required")
        if any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi for lo, hi in bounds):
            raise ValueError("every bound must be finite and lower < upper")
        self.bounds = bounds
        self.problem = problem
        self.config = config
        self._perturbed_banks = []
        self._perturbed = []
        self._internal_collision_checks = 0
        self.axis = 2
        self.extension_metadata = {
            "risk_evaluation_type": "conditional_gaussian_probability",
            "integrated_axis": 2,
            "binary_collision_checks": 0,
            "cost_limit": (
                "internal_collision_checks is legacy field storing conditional "
                "risk evaluations here; not equal-cost binary checks"
            ),
            "nominal_nfe": config.base.max_evaluations,
            "sigma": config.perturbation_sigma,
        }
        seed = _bank_seed(config)
        bank = residual_standardized_gaussian_bank(
            config.perturbation_samples,
            len(problem.environment.obstacles),
            seed,
            bank_design,
        )
        self.standardized_residual_bank = bank
        self.residual_bank_design = bank_design
        self.residual_bank_seed = seed
        self.residual_bank_dimensions = 2 * len(problem.environment.obstacles)
        self.shifts = _residual_shifts(
            bank, len(problem.environment.obstacles), config.perturbation_sigma
        )
        encoded = json.dumps(bank, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.extension_metadata.update(
            risk_evaluation_type="conditional_gaussian_residual_probability",
            integrated_axis=2,
            bank_design=bank_design,
            bank_seed=seed,
            samples_per_scale=config.perturbation_samples,
            bank_dimensions=self.residual_bank_dimensions,
            residual_coordinates_per_obstacle=2,
            residual_shift_sigma=config.perturbation_sigma,
            integrated_coordinates_per_obstacle=1,
            standardized_residual_bank_sha256=hashlib.sha256(encoded).hexdigest(),
            standardized_residual_bank=bank,
            residual_bank_rows_dependent=bank_design in _DESIGNS,
            binary_collision_checks=0,
        )


# Short aliases make the adapter convenient in experiment specifications while
# retaining the explicit class name in metadata and paper tables.
ResidualConditionalDE = ResidualConditionalRiskDE


class ResidualSoftRiskConditionalDE(ResidualConditionalRiskDE):
    """Residual conditional evaluator with the existing soft beta tie-break."""

    def __init__(self, problem, config: RiskAwareDEConfig, *, beta: float = 0.05,
                 bank_design: str = "latin_hypercube"):
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if config.selection_mode != "risk_budgeted":
            raise ValueError("soft risk control requires risk_budgeted selection")
        self.beta = float(beta)
        self.length_scale = math.dist(problem.start, problem.goal)
        super().__init__(problem, config, bank_design=bank_design)
        self.extension_metadata.update(
            experimental_risk_regularization_beta=self.beta,
            risk_regularization_length_scale=self.length_scale,
            risk_regularization_scope=(
                "only nominally feasible and inside the fixed chance gate"
            ),
        )

    def _retarget(self, evaluation: RiskEvaluation, chance_alpha: float) -> RiskEvaluation:
        record = super()._retarget(evaluation, chance_alpha)
        if not self.beta or not record.nominal_feasible or record.chance_violation > 0:
            return record
        cost = record.objective + self.beta * self.length_scale * record.perturbed_collision_rate
        return replace(record, key=(*record.key[:-1], cost))


class ResidualConditionalProjectedPolish(ConditionalProjectedPolish):
    """Established terminal polish backed by the residual conditional bank."""

    def __init__(self, problem, config, *, bank_design: str = "latin_hypercube", **options: Any):
        _validate_design(bank_design)
        if options.get("axis", 2) != 2:
            raise ValueError("residual conditional polish is fixed-z and requires axis=2")
        if options.get("global_method", "risk_de") != "risk_de":
            raise ValueError("residual conditional polish requires global_method='risk_de'")
        if options.get("risk_estimator", "conditional") != "conditional":
            raise ValueError("residual conditional polish requires conditional risk")
        self.bank_design = bank_design
        super().__init__(problem, config, **options)

    def _create_evaluator(self, global_cfg):
        return ResidualConditionalRiskDE(
            self.problem, global_cfg, bank_design=self.bank_design
        )


class ResidualSoftRiskProjectedPolish(ConditionalProjectedPolish):
    """Established terminal polish plus the existing beta=.05 soft control."""

    def __init__(self, problem, config, *, beta: float = 0.05,
                 bank_design: str = "latin_hypercube", **options: Any):
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        _validate_design(bank_design)
        if options.get("axis", 2) != 2:
            raise ValueError("residual conditional polish is fixed-z and requires axis=2")
        if options.get("global_method", "risk_de") != "risk_de":
            raise ValueError("residual soft polish requires global_method='risk_de'")
        if options.get("risk_estimator", "conditional") != "conditional":
            raise ValueError("residual soft polish requires conditional risk")
        self.beta = float(beta)
        self.bank_design = bank_design
        super().__init__(problem, config, **options)

    def _create_evaluator(self, global_cfg):
        return ResidualSoftRiskConditionalDE(
            self.problem,
            global_cfg,
            beta=self.beta,
            bank_design=self.bank_design,
        )


def _normalise_beta(value: Any) -> float:
    try:
        beta = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta must be 0 or 0.05") from exc
    if not math.isfinite(beta) or not any(math.isclose(beta, allowed, abs_tol=1e-12) for allowed in _ALLOWED_BETAS):
        raise ValueError("beta must be 0 or 0.05")
    return 0.0 if math.isclose(beta, 0.0, abs_tol=1e-12) else 0.05


def create_optimizer(problem, base_config: RiskAwareDEConfig, options: dict | None = None):
    """Create one of the two locked residual-bank experiment variants.

    Supported options are ``bank_design`` (``latin_hypercube`` or
    ``antithetic_lhs``), ``beta`` (``0`` or ``.05``), and all ordinary
    ``ConditionalProjectedPolish`` terminal options such as
    ``reserve_fraction``, ``active_margin``, ``initial_step``, ``projection``
    and ``max_line_steps``.  The residual adapter is deliberately fixed-z;
    silently switching to another axis or global optimizer is rejected.
    """

    opts = dict(options or {})
    design = _validate_design(opts.pop("bank_design", "latin_hypercube"))
    beta = _normalise_beta(opts.pop("beta", 0.05))
    if opts.get("axis", 2) != 2:
        raise ValueError("residual conditional polish is fixed-z and requires axis=2")
    if opts.get("global_method", "risk_de") != "risk_de":
        raise ValueError("residual conditional polish requires global_method='risk_de'")
    if opts.get("risk_estimator", "conditional") != "conditional":
        raise ValueError("residual conditional polish requires conditional risk")
    common = {"bank_design": design, **opts}
    if beta == 0.0:
        return ResidualConditionalProjectedPolish(problem, base_config, **common)
    return ResidualSoftRiskProjectedPolish(problem, base_config, beta=beta, **common)


__all__ = [
    "ResidualConditionalRiskDE",
    "ResidualConditionalDE",
    "ResidualSoftRiskConditionalDE",
    "ResidualConditionalProjectedPolish",
    "ResidualSoftRiskProjectedPolish",
    "residual_standardized_gaussian_bank",
    "create_optimizer",
]
