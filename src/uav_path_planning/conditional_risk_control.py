"""L-SHADE adaptation using the same conditional Gaussian risk information."""
from .conditional_risk_de import ConditionalRiskDE
from .risk_control_polish import ControlEvaluator
from .risk_budgeted_polish import RiskBudgetedPolish, PolishOptions


class ConditionalControl(ConditionalRiskDE):
    def __init__(self,problem,config,axis=2,global_method='lshade'):
        super().__init__(problem,config,axis=axis)
        self.global_method=global_method
        self.extension_metadata['control_merit']='nominal_failure*3 + (objective/(scale+objective) if risk passes else 1+probability+1e-6*objective/(scale+objective)); outside gate is approximate risk ordering with explicit tiny objective tie-break'

    def _merit(self,record,scale):
        tail=record.objective/(scale+record.objective)
        component=tail if record.perturbed_collision_rate<=self.config.chance_alpha else 1.+record.perturbed_collision_rate+1e-6*tail
        return (not record.nominal_feasible)*3.+component

    def optimize(self):
        return ControlEvaluator.optimize(self)


class ConditionalControlPolish(RiskBudgetedPolish):
    def __init__(self,problem,config,options,axis,global_method):
        super().__init__(problem,config,options)
        self.axis,self.global_method=axis,global_method
    def _create_evaluator(self,config):
        self.conditional_evaluator=ConditionalControl(self.problem,config,self.axis,self.global_method)
        return self.conditional_evaluator
    def optimize(self):
        raw=super().optimize()
        self.extension_metadata.update(self.conditional_evaluator.extension_metadata)
        return raw


def create_optimizer(problem,base_config,options):
    opts=dict(options);axis=opts.pop('axis',2);global_method=opts.pop('global_method','lshade');polish=opts.pop('polish',False)
    if polish:return ConditionalControlPolish(problem,base_config,PolishOptions(**opts),axis,global_method)
    if opts:raise ValueError(f'Unknown conditional control options {opts}')
    return ConditionalControl(problem,base_config,axis,global_method)
