"""Strong control: L-SHADE or DE with identical risk bank and local polishing.

This deliberately transfers the proposed local phase to existing optimizers
to test whether any measured gain requires the RiskAwareDE global search.
Scalar risk encoding is the same as the declared shared-risk controls.
"""
from __future__ import annotations
import math
from .de import DifferentialEvolution
from .lshade import LSHADE, LSHADEConfig
from .risk_aware_de import RiskAwareDE, RiskAwareDEResult
from .risk_sampling_de import SamplingRiskAwareDE
from .risk_budgeted_polish import RiskBudgetedPolish, PolishOptions


class ControlEvaluator(RiskAwareDE):
    def __init__(self, problem, config, global_method, bank_design):
        super().__init__(problem.bounds, problem, config)
        if bank_design != 'iid':
            sampler = SamplingRiskAwareDE(problem, config, design=bank_design)
            self._perturbed_banks = sampler._perturbed_banks
            self._perturbed = sampler._perturbed
        self.global_method = global_method

    def _merit(self, record, scale):
        count = self.config.perturbation_samples
        collision_count = round(record.perturbed_collision_rate * count)
        code = 0 if record.perturbed_collision_rate <= self.config.chance_alpha else collision_count
        return (not record.nominal_feasible)*(count+1) + code + record.objective/(scale+record.objective)

    def optimize(self):
        cfg = self.config
        if cfg.perturbation_sigmas is not None:
            raise ValueError('Control scalar risk encoding requires one scale')
        count = cfg.perturbation_samples
        scale = (self.problem.intermediate_waypoints + 1) * math.dist(
            self.problem.environment.bounds.minimum, self.problem.environment.bounds.maximum)
        best = None
        vector = None
        history = []
        keys = []
        def objective(x):
            nonlocal best, vector
            record = self._evaluate(x)
            if best is None or record.key < best.key:
                best, vector = record, list(x)
            history.append(best.objective); keys.append(best.key)
            return self._merit(record, scale)
        if self.global_method == 'lshade':
            optimizer = LSHADE(self.bounds, LSHADEConfig(
                population_size=cfg.base.population_size, min_population_size=4,
                generations=cfg.base.generations, seed=cfg.base.seed, max_evaluations=cfg.base.max_evaluations))
        elif self.global_method == 'de':
            optimizer = DifferentialEvolution(self.bounds, cfg.base)
        else:
            raise ValueError('Unsupported global control')
        raw = optimizer.optimize(objective)
        return RiskAwareDEResult(vector, best.objective, history, raw.evaluations, self._internal_collision_checks, best,
            key_history=keys, gate_history=[cfg.chance_alpha]*len(history),
            mutation_factor_history=[cfg.base.mutation_factor], crossover_rate_history=[cfg.base.crossover_rate])


class ControlPolish(RiskBudgetedPolish):
    def __init__(self, problem, config, options, global_method, bank_design):
        super().__init__(problem, config, options)
        self.global_method = global_method
        self.bank_design = bank_design

    def _create_evaluator(self, config):
        return ControlEvaluator(self.problem, config, self.global_method, self.bank_design)

    def optimize(self):
        raw = super().optimize()
        self.extension_metadata.update(global_method=self.global_method, bank_design=self.bank_design,
            claim_limit='Project adaptation control; same risk information and same local refinement; scalar weights affect L-SHADE history.')
        return raw


def create_optimizer(problem, base_config, options):
    opts = dict(options)
    global_method = opts.pop('global_method','lshade')
    bank_design = opts.pop('bank_design','iid')
    return ControlPolish(problem, base_config, PolishOptions(**opts), global_method, bank_design)
