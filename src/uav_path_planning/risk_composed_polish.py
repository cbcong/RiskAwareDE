"""Combine a declared Gaussian bank design with budget-reserved polishing.

Composition uses the exact same bank for the global phase and local phase.
Its purpose is factorial testing; established component ideas are not claimed
as new on their own. No risk evaluation is hidden in a geometric proposal.
"""
from __future__ import annotations
from .risk_budgeted_polish import RiskBudgetedPolish, PolishOptions
from .risk_sampling_de import SamplingRiskAwareDE


class SamplingPolish(RiskBudgetedPolish):
    def __init__(self, problem, config, options, bank_design):
        super().__init__(problem, config, options)
        self.bank_design = bank_design

    def _create_evaluator(self, config):
        return SamplingRiskAwareDE(self.problem, config, design=self.bank_design)

    def optimize(self):
        result = super().optimize()
        self.extension_metadata['bank_design'] = self.bank_design
        return result


def create_optimizer(problem, base_config, options):
    options = dict(options)
    design = options.pop('bank_design', 'antithetic_lhs')
    return SamplingPolish(problem, base_config, PolishOptions(**options), design)
