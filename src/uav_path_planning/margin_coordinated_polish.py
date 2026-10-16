"""Experimental changes inside the existing search and terminal-polish stages.

This candidate changes neither the conditional estimator nor the uncertainty
bank. Bounded repair lets a globally proposed, nominally feasible path accept a
small nominal cost increase when its empirical risk violation improves. A
terminal gate retains a fixed fraction of the global winner's empirical risk
slack. Neither rule is a safety certificate; neither is claimed to be a new
optimization theorem. External evaluation results are never used by the rules.

The terminal loop is intentionally isolated from ConditionalProjectedPolish so
that the frozen main algorithm stays untouched. It is copied from that module,
with only the explicitly recorded empirical gate changed. With repair_limit=0
and slack_fraction=1, the candidate is numerically equivalent to the frozen
fixed-z method, including random streams and evaluation accounting.
"""
from dataclasses import replace
import math

from .conditional_projected_polish import (
    ConditionalProjectedPolish, length_gradient_normalized,
)
from .conditional_risk_de import ConditionalRiskDE


def terminal_risk_gate(search_gate, global_risk, slack_fraction):
    """Spend a chosen fraction of internal slack, without external feedback.

    An already violated gate is not relaxed. An exact identity branch at one
    preserves historical floating-point decisions for the disabled candidate.
    """
    if slack_fraction == 1.0 or global_risk >= search_gate:
        return search_gate
    return min(search_gate, global_risk + slack_fraction * (search_gate-global_risk))


class RepairConditionalDE(ConditionalRiskDE):
    """Bounded mutation-only cost relaxation for strict risk-key improvements."""

    def __init__(self, problem, config, *, repair_limit=0.0):
        if not math.isfinite(repair_limit) or not 0.0 <= repair_limit <= 1.0:
            raise ValueError('repair_limit must be finite and in [0, 1]')
        if config.selection_mode != 'risk_budgeted':
            raise ValueError('Repair candidate requires risk_budgeted selection')
        super().__init__(problem, config, axis=2)
        self.repair_limit = float(repair_limit)
        self.repair_stats = {
            'opportunities': 0, 'accepted': 0, 'rejected': 0,
            'maximum_accepted_relative_cost_increase': 0.0,
        }

    def _accept_trial(self, candidate, incumbent, *, refinement=False):
        inherited = super()._accept_trial(candidate, incumbent, refinement=refinement)
        if inherited:
            return True
        eligible = (
            not refinement
            and candidate.key < incumbent.key
            and candidate.nominal_feasible and incumbent.nominal_feasible
            and candidate.objective > incumbent.objective
            and incumbent.chance_violation > 0.0
            and candidate.chance_violation < incumbent.chance_violation
        )
        if not eligible:
            return False
        self.repair_stats['opportunities'] += 1
        relative_cost = ((candidate.objective-incumbent.objective)
                         / max(abs(incumbent.objective), 1e-12))
        fraction_repaired = min(1.0, (
            incumbent.chance_violation-candidate.chance_violation
        ) / max(incumbent.chance_violation, 1e-12))
        accepted = (self.repair_limit > 0.0
                    and relative_cost <= self.repair_limit * fraction_repaired)
        self.repair_stats['accepted' if accepted else 'rejected'] += 1
        if accepted:
            self.repair_stats['maximum_accepted_relative_cost_increase'] = max(
                self.repair_stats['maximum_accepted_relative_cost_increase'],
                relative_cost,
            )
        return accepted


class MarginCoordinatedPolish(ConditionalProjectedPolish):
    """Fixed-z search repair plus a slack-limited terminal empirical gate."""

    def __init__(self, problem, config, *, repair_limit=0.0, slack_fraction=1.0,
                 reserve_fraction=.2, max_line_steps=1):
        if not math.isfinite(repair_limit) or not 0.0 <= repair_limit <= 1.0:
            raise ValueError('repair_limit must be finite and in [0, 1]')
        if not math.isfinite(slack_fraction) or not 0.0 <= slack_fraction <= 1.0:
            raise ValueError('slack_fraction must be finite and in [0, 1]')
        if config.selection_mode != 'risk_budgeted':
            raise ValueError('Margin candidate requires risk_budgeted selection')
        super().__init__(problem, config, reserve_fraction=reserve_fraction,
                         max_line_steps=max_line_steps)
        self.repair_limit = float(repair_limit)
        self.slack_fraction = float(slack_fraction)

    def _create_evaluator(self, global_cfg):
        return RepairConditionalDE(self.problem, global_cfg, repair_limit=self.repair_limit)

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
        terminal_gate = terminal_risk_gate(
            cfg.chance_alpha, record.perturbed_collision_rate, self.slack_fraction)
        record = evaluator._retarget(record, terminal_gate)
        widths=[hi-lo for lo,hi in self.problem.bounds]
        start=raw.evaluations;gradient_probes=0;accepted_count=0;events=[]
        step=self.initial_step
        def evaluate(candidate):
            raw.evaluations+=1
            return evaluator._evaluate(candidate,terminal_gate)
        def trace():
            raw.history.append(record.objective);raw.key_history.append(record.key);raw.gate_history.append(terminal_gate)
        while raw.evaluations<total:
            direction=[-x for x in length_gradient_normalized(self.problem,vector)]
            projected=False
            risk_gradient=[]
            # Leave at least one evaluation for a proposal. No derivatives
            # are inferred without paying for all measured probes.
            if self.projection!='length_only' and record.perturbed_collision_rate>=terminal_gate-self.active_margin and total-raw.evaluations>len(vector):
                for index,(lo,hi) in enumerate(self.problem.bounds):
                    probe=list(vector)
                    delta=.001*widths[index]
                    probe[index]=min(hi,vector[index]+delta)
                    if probe[index]==vector[index]:probe[index]=max(lo,vector[index]-delta)
                    change=(probe[index]-vector[index])/widths[index]
                    measured=evaluate(probe);gradient_probes+=1
                    risk_gradient.append((measured.perturbed_collision_rate-record.perturbed_collision_rate)/change)
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
                    accepted=(measured.objective<=record.objective and evaluator._refinement_allowed(measured,record,terminal_gate)
                        and evaluator._accept_trial(measured,record,refinement=True))
                    gain=record.objective-measured.objective if accepted else 0.
                    events.append({'nfe':raw.evaluations,'gradient_projected':projected,'step':step,'accepted':accepted,'accepted_objective_gain':gain,'risk':measured.perturbed_collision_rate})
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
        self.extension_metadata={**getattr(evaluator,'extension_metadata',{}),'mechanism':'experimental_margin_coordinated_projected_descent','reserve_fraction':self.reserve_fraction,
            'gradient_probes':gradient_probes,'line_search_evaluations':len(events),'global_method':self.global_method,'projection':self.projection,
            'risk_estimator':self.risk_estimator,'max_line_steps':self.max_line_steps,
            'global_stage':global_stage,
            'terminal_empirical_gate':terminal_gate,
            'original_search_gate':cfg.chance_alpha,
            'slack_fraction':self.slack_fraction,
            'repair_limit':self.repair_limit,
            'risk_repair':dict(evaluator.repair_stats),
            'evidence_status':'experimental heuristic; no new optimality or safety theorem',
            'gradient_reuse_policy':'up to max_line_steps accepted trials; every trial reevaluated',
            'safety_certificate':False,
            'cost_limit':'Every derivative probe charged; conditional integrations differ from binary collision checks.'}
        return raw


def create_optimizer(problem, base_config, options):
    return MarginCoordinatedPolish(problem, base_config, **options)
