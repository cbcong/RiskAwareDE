"""Experimental within-gate risk regularization; never selected by default.

The chance gate still has priority. Inside that gate, global search ranks
``length objective + beta * straight-line distance * estimated risk``.
Terminal proposals must additionally retain the original non-increasing length
guard. This is a weighted-objective control, not a new probability result or
a claim of free safety improvement. beta=0 exactly recovers the original key.
"""
from dataclasses import replace
import math

from .conditional_projected_polish import ConditionalProjectedPolish
from .conditional_risk_de import ConditionalRiskDE


class SoftRiskConditionalDE(ConditionalRiskDE):
    def __init__(self, problem, config, *, beta=0.0, axis=2):
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if config.selection_mode != "risk_budgeted":
            raise ValueError("soft risk control requires risk_budgeted selection")
        self.beta = beta
        self.length_scale = math.dist(problem.start, problem.goal)
        super().__init__(problem, config, axis=axis)
        self.extension_metadata.update(
            experimental_risk_regularization_beta=beta,
            risk_regularization_length_scale=self.length_scale,
            risk_regularization_scope="only nominally feasible and inside the fixed chance gate",
        )

    def _retarget(self, evaluation, chance_alpha):
        record = super()._retarget(evaluation, chance_alpha)
        if not self.beta or not record.nominal_feasible or record.chance_violation > 0:
            return record
        cost = record.objective + self.beta * self.length_scale * record.perturbed_collision_rate
        return replace(record, key=(*record.key[:-1], cost))


class SoftRiskProjectedPolish(ConditionalProjectedPolish):
    def __init__(self, problem, config, *, beta=0.0, **options):
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        self.beta = beta
        super().__init__(problem, config, **options)
        if self.global_method != "risk_de" or self.risk_estimator != "conditional" or self.axis == "adaptive":
            raise ValueError("experimental soft control supports fixed-axis conditional RiskAwareDE only")

    def _create_evaluator(self, global_cfg):
        return SoftRiskConditionalDE(self.problem, global_cfg, beta=self.beta, axis=self.axis)


def create_optimizer(problem, base_config, options):
    return SoftRiskProjectedPolish(problem, base_config, **options)
