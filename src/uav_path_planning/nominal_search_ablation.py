"""Remove every perturbation-risk decision from the complete search method.

This is the E5 ablation, not a different setting of the empirical risk gate.
It retains the global proposal operators and adaptation schedule, but success
weights use nominal improvement only. Terminal descent keeps nominal collision
acceptance and never evaluates or projects onto a perturbation-risk gradient.
The core module depends only on the Python standard library.
"""

from dataclasses import replace
import math

from .conditional_projected_polish import ConditionalProjectedPolish
from .risk_aware_de import RiskAwareDE, RiskEvaluation


def _nominal_config(config):
    return replace(config, selection_mode="nominal", risk_tradeoff=None,
                   risk_gate_schedule="fixed", risk_gate_start=None,
                   pbest_selection="risk_first")


class NominalSearchDE(RiskAwareDE):
    """Same proposal machinery as RiskAwareDE, with no uncertainty access.

    The legacy ``parameter_adaptation='risk_success'`` flag still switches on
    parameter adaptation, but this subclass replaces its success weight with
    ``1 + min(1, relative_nominal_improvement)``. Zero risk fields in the shared
    result structure are compatibility placeholders, not measurements; callers
    must use ``risk_evaluation_type='none'`` and report internal risk as absent.
    """

    def __init__(self, problem, config):
        # Deliberately do not call RiskAwareDE.__init__: that creates shifted
        # environments even when selection_mode is 'nominal'.
        self.bounds = [(float(lo), float(hi)) for lo, hi in problem.bounds]
        if not self.bounds:
            raise ValueError("at least one decision variable is required")
        if any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi
               for lo, hi in self.bounds):
            raise ValueError("every bound must be finite and lower < upper")
        self.problem = problem
        self.config = _nominal_config(config)
        self._internal_collision_checks = 0
        self.extension_metadata = {
            "risk_evaluation_type": "none",
            "internal_risk_measured": False,
            "uncertainty_bank_constructed": False,
            "binary_collision_checks": 0,
            "conditional_risk_evaluations": 0,
            "selection": "nominal feasibility, then nominal objective",
            "parameter_adaptation": (
                "nominal_success" if config.parameter_adaptation == "risk_success"
                else "fixed"
            ),
            "success_weight": "1 + min(1, relative nominal objective improvement)",
            "risk_record_fields": "zero placeholders only; report as not measured",
        }

    def _evaluate(self, vector, chance_alpha=None):
        metrics = self.problem.evaluate(self.problem.decode(vector))
        objective = float(metrics.objective)
        key = (0.0 if metrics.feasible else 1.0, objective)
        return RiskEvaluation(metrics.feasible, 0.0, 0.0, objective, key, math.nan)

    def _retarget(self, evaluation, chance_alpha):
        return evaluation

    def _gate_alpha(self, completed, total):
        return 0.0

    def _accept_trial(self, candidate, incumbent, *, refinement=False):
        return candidate.key < incumbent.key

    def _refinement_allowed(self, candidate, incumbent, chance_alpha=None):
        return candidate.nominal_feasible

    def _success_weight(self, parent, trial):
        gain = max(0.0, parent.objective - trial.objective)
        gain /= max(abs(parent.objective), 1e-12)
        return 1.0 + min(1.0, gain)

    def optimize(self):
        result = super().optimize()
        # Empty means no risk gate exists in this ablation. A zero-valued gate
        # trace could misleadingly look like a very strict chance constraint.
        result.gate_history = []
        return result


class NominalProjectedPolish(ConditionalProjectedPolish):
    """Global nominal search plus the same budgeted geometric line search."""

    def __init__(self, problem, config, *, reserve_fraction=.2,
                 initial_step=.03, max_line_steps=1):
        super().__init__(problem, _nominal_config(config),
                         reserve_fraction=reserve_fraction,
                         initial_step=initial_step,
                         max_line_steps=max_line_steps,
                         projection="length_only")

    def _create_evaluator(self, global_cfg):
        return NominalSearchDE(self.problem, global_cfg)

    def optimize(self):
        result = super().optimize()
        result.gate_history = []
        for event in result.local_refinement_events:
            event.pop("risk", None)
        self.extension_metadata.update({
            "mechanism": "nominal_only_search_and_geometric_length_descent",
            "global_method": "nominal_proposal_matched_de",
            "risk_estimator": "none",
            "risk_evaluation_type": "none",
            "internal_risk_measured": False,
            "cost_limit": "Nominal evaluations only; no uncertainty bank or risk checks.",
        })
        self.extension_metadata["global_stage"]["estimated_risk"] = None
        return result


def create_optimizer(problem, base_config, options):
    return NominalProjectedPolish(problem, base_config, **options)
