"""Paper controls matched to the frozen soft_beta050 risk ranking.

These adapters are experiment interventions, not new main-method versions.
The frozen optimizer is imported unchanged. Native baselines keep their own
proposal/adaptation rules, while using the same within-gate soft cost for
scalar selection and the same exact ordering for the returned-path archive.
Their inherited outside-gate scalar approximation remains explicitly reported.
"""
from dataclasses import replace
import math

from .conditional_baseline_control import ConditionalBaselineControl
from .conditional_projected_polish import ConditionalProjectedPolish
from .conditional_risk_de import ConditionalRiskDE
from .risk_aware_de import RiskAwareDE
from .soft_risk_projected_polish import SoftRiskConditionalDE, SoftRiskProjectedPolish


class _SoftRiskKey:
    """The frozen SoftRiskConditionalDE._retarget rule, estimator-independent."""

    def _initialize_soft_key(self, problem, config, beta):
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if config.selection_mode != "risk_budgeted":
            raise ValueError("soft risk control requires risk_budgeted selection")
        self.beta = beta
        self.length_scale = math.dist(problem.start, problem.goal)

    def _retarget(self, evaluation, chance_alpha):
        record = super()._retarget(evaluation, chance_alpha)
        if not self.beta or not record.nominal_feasible or record.chance_violation > 0:
            return record
        cost = record.objective + self.beta * self.length_scale * record.perturbed_collision_rate
        return replace(record, key=(*record.key[:-1], cost))

    def _record_soft_key(self):
        self.extension_metadata.update(
            experimental_risk_regularization_beta=self.beta,
            risk_regularization_length_scale=self.length_scale,
            risk_regularization_scope="only nominally feasible and inside the fixed chance gate",
            experiment_role="matched control for frozen soft_beta050; not a new main method",
        )


class MatchedConditionalDE(_SoftRiskKey, ConditionalRiskDE):
    """Estimator-independent implementation checked against the frozen key."""

    def __init__(self, problem, config, *, beta=.05, axis=2):
        self._initialize_soft_key(problem, config, beta)
        super().__init__(problem, config, axis=axis)
        self._record_soft_key()


class MatchedBinaryDE(_SoftRiskKey, RiskAwareDE):
    """Replace only the risk estimator, retaining soft ranking and proposals."""

    def __init__(self, problem, config, *, beta=.05):
        self._initialize_soft_key(problem, config, beta)
        super().__init__(problem.bounds, problem, config)
        self.extension_metadata = {
            "risk_evaluation_type": "binary_iid_collision_frequency",
            "cost_limit": "Binary collision trials; not equal operations to conditional integration.",
            "estimator_interaction": (
                "The same finite-difference tangent machinery now receives discrete risk estimates; "
                "this is an end-to-end estimator replacement, not an isolated variance experiment."
            ),
        }
        self._record_soft_key()


class MatchedBaselineControl(_SoftRiskKey, ConditionalBaselineControl):
    """Native optimizer plus conditional evaluator and beta-matched merit."""

    def __init__(self, problem, config, *, beta=.05, axis=2, global_method="de"):
        self._initialize_soft_key(problem, config, beta)
        super().__init__(problem, config, axis=axis, global_method=global_method)
        self._record_soft_key()
        self.extension_metadata.update(
            control_merit=(
                "3*nominal_failure + soft_cost/(scale+soft_cost) for nominally feasible paths inside gate; "
                "otherwise the inherited scalar merit; exact shared soft key for archive and terminal acceptance"
            ),
            returned_path_rule="exact beta-matched risk_budgeted key over every observed candidate",
            native_success_weights="beta-matched scalar merit improvements",
        )

    def _merit(self, record, scale):
        # Outside the gate (or nominally infeasible), the frozen soft rule
        # leaves ordering unchanged. Preserve the established scalar adapter.
        if not record.nominal_feasible or record.chance_violation > 0 or not self.beta:
            return super()._merit(record, scale)
        cost = record.objective + self.beta * self.length_scale * record.perturbed_collision_rate
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("conditional merit requires a finite nonnegative soft cost")
        return cost / (scale + cost)


class MatchedProjectedPolish(ConditionalProjectedPolish):
    """Reuse the unmodified terminal procedure with the declared evaluator."""

    def __init__(self, problem, config, *, experiment_kind, family=None, beta=.05, **options):
        if experiment_kind not in {"projected_baseline", "binary"}:
            raise ValueError("MatchedProjectedPolish requires projected_baseline or binary")
        if "global_method" in options or "risk_estimator" in options:
            raise ValueError("experiment_kind and family determine global_method and risk_estimator")
        if experiment_kind == "projected_baseline":
            if family not in ConditionalBaselineControl.METHODS:
                raise ValueError("a supported native family is required")
            global_method, risk_estimator = "baseline_" + family, "conditional"
        else:
            if family is not None:
                raise ValueError("binary estimator replacement retains the main global search")
            global_method, risk_estimator = "risk_de", "binary"
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if config.selection_mode != "risk_budgeted":
            raise ValueError("soft risk control requires risk_budgeted selection")
        self.beta, self.family, self.experiment_kind = beta, family, experiment_kind
        super().__init__(problem, config, global_method=global_method,
                         risk_estimator=risk_estimator, **options)

    def _create_evaluator(self, global_cfg):
        if self.experiment_kind == "binary":
            return MatchedBinaryDE(self.problem, global_cfg, beta=self.beta)
        return MatchedBaselineControl(self.problem, global_cfg, beta=self.beta,
                                      axis=self.axis, global_method=self.family)


def create_optimizer(problem, base_config, options):
    """Create one explicitly declared control without changing the frozen main.

    ``experiment_kind``: risk_baseline, projected_baseline, no_terminal,
    binary, or full. Baseline kinds require ``family``. ``beta`` defaults to
    .05; terminal options use the original ConditionalProjectedPolish names.
    All provided options are consumed or rejected.
    """
    opts = dict(options)
    kind = opts.pop("experiment_kind")
    beta = opts.pop("beta", .05)
    if kind == "full":
        return SoftRiskProjectedPolish(problem, base_config, beta=beta, **opts)
    if kind in {"projected_baseline", "binary"}:
        return MatchedProjectedPolish(problem, base_config, experiment_kind=kind, beta=beta, **opts)
    axis = opts.pop("axis", 2)
    if kind == "risk_baseline":
        family = opts.pop("family")
        if opts:
            raise ValueError(f"Unsupported risk-only baseline options: {sorted(opts)}")
        return MatchedBaselineControl(problem, base_config, beta=beta, axis=axis, global_method=family)
    if kind == "no_terminal":
        if opts:
            raise ValueError(f"Unsupported no-terminal options: {sorted(opts)}")
        return SoftRiskConditionalDE(problem, base_config, beta=beta, axis=axis)
    raise ValueError(f"Unsupported matched experiment kind: {kind}")
