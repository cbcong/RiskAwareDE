"""Native DE-family / PSO controls with a shared conditional-risk evaluator.

This is a project adaptation, not an author's published risk-aware algorithm.
The native optimizers receive one explicitly declared scalar merit; their
mutation, particle updates and success-history rules are not replaced by the
proposed RiskAwareDE search.  A best-observed archive uses the exact shared
feasibility/risk/length key for the returned path and terminal refinement.

The scalar merit is the legacy ConditionalControl merit.  Outside the risk
gate, the 1e-6 normalized-objective term can reverse ordering only for risks
separated by less than 1e-6 (ignoring floating-point rounding).  It is therefore
an approximation to lexicographic selection, not an exact scalar encoding.
For SHADE-family methods the scalar improvement also affects adaptation
weights; that is part of this explicitly identified risk adaptation.
"""
from __future__ import annotations

from dataclasses import asdict
import math

from .conditional_risk_de import ConditionalRiskDE
from .de import DEConfig, DifferentialEvolution
from .jade_shade import JADE, JADEConfig, SHADE, SHADEConfig
from .lshade import LSHADE, LSHADEConfig
from .pso import PSOConfig, ParticleSwarmPlanner
from .risk_aware_de import RiskAwareDEResult


class _ObjectiveProblem:
    """Forward geometry unchanged while supplying PSO's shared scalar merit."""

    def __init__(self, problem, objective):
        self._problem = problem
        self.objective = objective

    def __getattr__(self, name):
        return getattr(self._problem, name)


class ConditionalBaselineControl(ConditionalRiskDE):
    """Conditional risk + an unmodified native baseline optimizer.

    ``config.base`` supplies population, iteration cap, NFE and planner seed.
    Algorithm-specific coefficients use the same native defaults as nominal
    ``BenchmarkConfig`` baselines; RiskAwareDE mutation/adaptation/shortcut
    options do not leak into their global-search phase.  Inherited refinement
    acceptance remains available for the common terminal-polish experiment.
    """

    METHODS = ("de", "jade", "shade", "lshade", "pso")
    OBJECTIVE_TIE_WEIGHT = 1e-6

    def __init__(self, problem, config, *, axis=2, global_method="de"):
        if global_method not in self.METHODS:
            raise ValueError(f"global_method must be one of {self.METHODS}")
        if config.risk_gate_schedule != "fixed":
            raise ValueError("Conditional baseline controls require a fixed risk gate")
        if config.selection_mode != "risk_budgeted":
            raise ValueError("Conditional baseline controls require risk_budgeted selection")
        budget = config.base.max_evaluations or (
            config.base.population_size * (config.base.generations + 1)
        )
        if (global_method != "lshade"
                and budget > config.base.population_size * (config.base.generations + 1)):
            raise ValueError("generation cap cannot consume the configured NFE budget")
        super().__init__(problem, config, axis=axis)
        self.global_method = global_method
        self.extension_metadata.update(
            global_method=global_method,
            control_kind="native_baseline_with_shared_conditional_risk",
            control_merit=(
                "3*nominal_failure + objective/(scale+objective) inside gate; "
                "3*nominal_failure + 1 + risk + 1e-6*objective/(scale+objective) outside gate"
            ),
            outside_gate_ordering_approximation_bound=self.OBJECTIVE_TIE_WEIGHT,
            returned_path_rule="exact risk_budgeted key over every observed candidate",
            selection_rule="native optimizer uses scalar merit; final archive uses exact key",
            native_success_weights="scalar merit improvements, not nominal length improvements",
            acceptance_scope=(
                "Shared conditional estimator, risk gate and final archive key; global native scalar "
                "selection does not use RiskAwareDE._accept_trial or its nominal-length safeguard. "
                "Inherited _accept_trial applies only when the common terminal polish calls it."
            ),
            global_riskaware_shortcuts=False,
            global_riskaware_parameter_adaptation=False,
        )

    def _merit(self, record, scale):
        if not math.isfinite(record.objective) or record.objective < 0:
            raise ValueError("conditional merit requires a finite nonnegative objective")
        tail = record.objective / (scale + record.objective)
        if record.perturbed_collision_rate <= self.config.chance_alpha:
            component = tail
        else:
            component = 1.0 + record.perturbed_collision_rate + self.OBJECTIVE_TIE_WEIGHT * tail
        return (not record.nominal_feasible) * 3.0 + component

    def _native_optimizer(self, objective):
        base = self.config.base
        common = dict(population_size=base.population_size, generations=base.generations,
                      seed=base.seed, max_evaluations=base.max_evaluations)
        if self.global_method == "de":
            native_config = DEConfig(**common)
            optimizer = DifferentialEvolution(self.bounds, native_config)
        elif self.global_method == "jade":
            native_config = JADEConfig(**common)
            optimizer = JADE(self.bounds, native_config)
        elif self.global_method == "shade":
            native_config = SHADEConfig(**common)
            optimizer = SHADE(self.bounds, native_config)
        elif self.global_method == "lshade":
            native_config = LSHADEConfig(**common, min_population_size=4)
            optimizer = LSHADE(self.bounds, native_config)
        else:
            native_config = PSOConfig(
                particles=base.population_size, iterations=base.generations,
                seed=base.seed, max_evaluations=base.max_evaluations,
            )
            optimizer = ParticleSwarmPlanner(_ObjectiveProblem(self.problem, objective), native_config)
        self.extension_metadata["native_config"] = asdict(native_config)
        return optimizer.plan() if self.global_method == "pso" else optimizer.optimize(objective)

    def optimize(self):
        cfg = self.config
        scale = (self.problem.intermediate_waypoints + 1) * math.dist(
            self.problem.environment.bounds.minimum, self.problem.environment.bounds.maximum,
        )
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("the objective-normalization scale must be positive and finite")
        best = None
        best_vector = None
        best_merit = None
        history, keys = [], []
        evaluated = {}
        incumbent_disagreements = 0

        def objective(vector):
            nonlocal best, best_vector, best_merit, incumbent_disagreements
            record = self._evaluate(vector)
            merit = self._merit(record, scale)
            evaluated[tuple(vector)] = record
            if best is not None and ((record.key < best.key) != (merit < best_merit)):
                incumbent_disagreements += 1
            if best is None or record.key < best.key:
                best, best_vector, best_merit = record, list(vector), merit
            history.append(best.objective)
            keys.append(best.key)
            return merit

        raw = self._native_optimizer(objective)
        budget = cfg.base.max_evaluations or cfg.base.population_size * (cfg.base.generations + 1)
        if raw.evaluations != len(history) or raw.evaluations != budget:
            raise RuntimeError("native baseline did not consume its declared objective budget")
        if self.global_method == "pso":
            native_vector = tuple(coordinate for point in raw.path.waypoints[1:-1] for coordinate in point)
        else:
            native_vector = tuple(raw.best_vector)
        native_record = evaluated[native_vector]
        self.extension_metadata.update(
            control_global_evaluations=raw.evaluations,
            scalar_exact_incumbent_disagreements=incumbent_disagreements,
            scalar_native_return_key=list(native_record.key),
            exact_archive_return_key=list(best.key),
            scalar_native_return_differs_from_exact_archive=(native_record.key != best.key),
        )
        return RiskAwareDEResult(
            best_vector=best_vector, best_fitness=best.objective,
            history=history, evaluations=raw.evaluations,
            internal_collision_checks=self._internal_collision_checks,
            best_evaluation=best, key_history=keys,
            gate_history=[cfg.chance_alpha] * len(history),
        )


def create_optimizer(problem, base_config, options):
    return ConditionalBaselineControl(problem, base_config, **dict(options))
