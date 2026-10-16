"""Budgeted length descent projected onto a sampled risk-constraint tangent.

Conditional Gaussian risk is continuous enough to estimate directional
derivatives. Every finite-difference probe and line-search trial is charged to
the same nominal/risk budget. Projection is only a proposal heuristic; nominal
and empirical risk acceptance remain mandatory. No convergence or safety
certificate is claimed for this finite-budget, piecewise-smooth problem.
"""
from dataclasses import replace
import math
from .conditional_risk_de import ConditionalRiskDE


def length_gradient_normalized(problem, vector):
    points=problem.decode(vector).waypoints
    gradient=[]
    for i,current in enumerate(points[1:-1]):
        for axis in range(3):
            value=0.
            for neighbour in (points[i],points[i+2]):
                distance=math.dist(current,neighbour)
                if distance>1e-12:value+=(current[axis]-neighbour[axis])/distance
            lo,hi=problem.bounds[3*i+axis]
            gradient.append(value*(hi-lo))
    return gradient


class ConditionalProjectedPolish:
    def __init__(self,problem,config,*,reserve_fraction=.2,active_margin=.02,initial_step=.03,axis=2,global_method='risk_de',projection='tangent',risk_estimator='conditional',max_line_steps=1):
        if not 0<reserve_fraction<1 or not 0<initial_step<=1 or not 0<=active_margin<=1:
            raise ValueError('Invalid projected-polish parameters')
        if config.risk_gate_schedule!='fixed':raise ValueError('Fixed gate required')
        if isinstance(max_line_steps,bool) or not isinstance(max_line_steps,int) or max_line_steps<1:
            raise ValueError('max_line_steps must be a positive integer')
        if axis not in (0,1,2,'adaptive') or isinstance(axis,bool):
            raise ValueError("axis must be 0, 1, 2, or 'adaptive'")
        self.problem,self.config=problem,config
        self.reserve_fraction,self.active_margin,self.initial_step,self.axis=reserve_fraction,active_margin,initial_step,axis
        self.max_line_steps=max_line_steps
        if global_method not in ('risk_de','lshade','baseline_de','baseline_jade','baseline_shade','baseline_lshade','baseline_pso'):
            raise ValueError('Unsupported global_method')
        if axis=='adaptive' and global_method!='risk_de':
            raise ValueError('Adaptive-axis L-SHADE is not implemented; refusing to substitute RiskAwareDE')
        self.global_method=global_method
        if projection not in ('tangent','ignore_measured_gradient','length_only'):
            raise ValueError('Unsupported projection mode')
        self.projection=projection
        if risk_estimator not in ('conditional','binary'):
            raise ValueError('Unsupported risk estimator')
        if risk_estimator=='binary' and global_method!='risk_de':
            raise ValueError('Binary removal ablation currently uses RiskAwareDE global search')
        if risk_estimator=='binary' and axis=='adaptive':
            raise ValueError('An adaptive integration axis requires conditional risk estimation')
        self.risk_estimator=risk_estimator
        self.extension_metadata={}

    def _create_evaluator(self,global_cfg):
        """Construct the global stage without changing historical RNG streams.

        Explicit ``baseline_`` names select the common-module comparison
        adapters; the historical ``lshade`` name retains its original adapter.
        Nominal removal ablations override this factory so that they never
        construct an uncertainty bank.
        """
        if self.risk_estimator=='binary':
            from .risk_aware_de import RiskAwareDE
            return RiskAwareDE(self.problem.bounds,self.problem,global_cfg)
        elif self.axis=='adaptive':
            from .adaptive_axis_conditional_de import AdaptiveAxisConditionalDE
            return AdaptiveAxisConditionalDE(self.problem,global_cfg,mode='min_extent')
        elif self.global_method=='risk_de':
            return ConditionalRiskDE(self.problem,global_cfg,axis=self.axis)
        elif self.global_method.startswith('baseline_'):
            from .conditional_baseline_control import ConditionalBaselineControl
            return ConditionalBaselineControl(self.problem,global_cfg,axis=self.axis,
                global_method=self.global_method.removeprefix('baseline_'))
        else:
            from .conditional_risk_control import ConditionalControl
            return ConditionalControl(self.problem,global_cfg,self.axis,self.global_method)

    def optimize(self):
        cfg=self.config
        total=cfg.base.max_evaluations or cfg.base.population_size*(cfg.base.generations+1)
        reserve=round(total*self.reserve_fraction)
        global_budget=max(cfg.base.population_size,total-reserve)
        global_cfg=replace(cfg,base=replace(cfg.base,max_evaluations=global_budget))
        evaluator=self._create_evaluator(global_cfg)
        raw=evaluator.optimize()
        vector,record=list(raw.best_vector),raw.best_evaluation
        global_stage={'evaluations':raw.evaluations,'best_vector':list(vector),
            'objective':record.objective,'path_length':self.problem.decode(vector).length,
            'nominal_feasible':record.nominal_feasible,
            'estimated_risk':record.perturbed_collision_rate}
        widths=[hi-lo for lo,hi in self.problem.bounds]
        start=raw.evaluations;gradient_probes=0;accepted_count=0;events=[]
        step=self.initial_step
        def evaluate(candidate):
            raw.evaluations+=1
            return evaluator._evaluate(candidate,cfg.chance_alpha)
        def risk_score(evaluation):
            score=getattr(evaluator,'risk_gate_score',None)
            return score(evaluation) if callable(score) else evaluation.perturbed_collision_rate
        def trace():
            raw.history.append(record.objective);raw.key_history.append(record.key);raw.gate_history.append(cfg.chance_alpha)
        while raw.evaluations<total:
            direction=[-x for x in length_gradient_normalized(self.problem,vector)]
            projected=False
            risk_gradient=[]
            # Leave at least one evaluation for a proposal. No derivatives
            # are inferred without paying for all measured probes.
            if self.projection!='length_only' and risk_score(record)>=cfg.chance_alpha-self.active_margin and total-raw.evaluations>len(vector):
                for index,(lo,hi) in enumerate(self.problem.bounds):
                    probe=list(vector)
                    delta=.001*widths[index]
                    probe[index]=min(hi,vector[index]+delta)
                    if probe[index]==vector[index]:probe[index]=max(lo,vector[index]-delta)
                    change=(probe[index]-vector[index])/widths[index]
                    measured=evaluate(probe);gradient_probes+=1
                    risk_gradient.append((risk_score(measured)-risk_score(record))/change)
                    trace()
                normal2=sum(x*x for x in risk_gradient)
                outward=sum(x*y for x,y in zip(direction,risk_gradient))
                if self.projection=='tangent' and normal2>1e-18 and outward>0:
                    direction=[d-outward*g/normal2 for d,g in zip(direction,risk_gradient)]
                    projected=True
            norm=math.sqrt(sum(x*x for x in direction))
            if norm<1e-15:
                # A stalled first-order model gets a deterministic coordinate
                # move, still fully evaluated. Never return a free proposal.
                direction=[0.]*len(vector);direction[len(events)%len(vector)]=1.;norm=1.
            direction=[x/norm for x in direction]
            consecutive_steps=0
            # Reuse only the proposal direction. The risk and nominal objective
            # of EVERY new point are reevaluated before accepting that point.
            while consecutive_steps<self.max_line_steps and raw.evaluations<total:
                accepted=False
                for attempt in range(8):
                    if raw.evaluations>=total:break
                    proposal=[max(lo,min(hi,x+step*d*w)) for x,d,w,(lo,hi) in zip(vector,direction,widths,self.problem.bounds)]
                    measured=evaluate(proposal)
                    accepted=(measured.objective<=record.objective and evaluator._refinement_allowed(measured,record,cfg.chance_alpha)
                        and evaluator._accept_trial(measured,record,refinement=True))
                    gain=record.objective-measured.objective if accepted else 0.
                    events.append({'nfe':raw.evaluations,'gradient_projected':projected,'step':step,'accepted':accepted,'accepted_objective_gain':gain,'risk':risk_score(measured)})
                    if accepted:
                        vector,record=proposal,measured;accepted_count+=1
                        step=min(self.initial_step,step*1.2)
                        consecutive_steps+=1
                        trace()
                        break
                    else:
                        step=max(1e-6,step*.5)
                        trace()
                if not accepted:break
            if not accepted and step<=1e-5:
                step=self.initial_step*.25
        raw.best_vector=vector;raw.best_evaluation=record;raw.best_fitness=record.objective
        raw.internal_collision_checks=evaluator._internal_collision_checks
        raw.local_refinement_events=events
        raw.local_refinement_stats={'projected_global_evaluations':start,'projected_gradient_probes':gradient_probes,'projected_line_trials':len(events),'projected_accepted':accepted_count}
        self.extension_metadata={**getattr(evaluator,'extension_metadata',{}),'mechanism':'risk_tangent_projected_length_descent','reserve_fraction':self.reserve_fraction,
            'gradient_probes':gradient_probes,'line_search_evaluations':len(events),'global_method':self.global_method,'projection':self.projection,
            'risk_estimator':self.risk_estimator,'max_line_steps':self.max_line_steps,
            'global_stage':global_stage,
            'gradient_reuse_policy':'up to max_line_steps accepted trials; every trial reevaluated',
            'safety_certificate':False,
            'cost_limit':'Every derivative probe charged; conditional integrations differ from binary collision checks.'}
        return raw


def create_optimizer(problem,base_config,options):
    return ConditionalProjectedPolish(problem,base_config,**options)
