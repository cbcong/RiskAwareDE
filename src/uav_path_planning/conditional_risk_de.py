"""Conditional Gaussian risk integration for box-obstacle polyline planning.

Given two translation coordinates per obstacle, a segment intersects that box
exactly when the remaining translation coordinate lies in an interval. Union
the intervals across path segments, integrate the Gaussian CDF, and multiply
obstacle survival probabilities (independent obstacle translations). Averaging
these conditional probabilities is Rao-Blackwellized Monte Carlo, not a new
probability theorem. It is not an adaptively selected path safety certificate.

One risk sample here is a conditional probability calculation, NOT one ordinary
binary collision check. Both nominal budget and operation type are reported;
equal sample count must not be described as equal runtime or equal operations.
"""
from __future__ import annotations
import math
from .risk_aware_de import RiskAwareDE, RiskAwareDEConfig, RiskEvaluation
from .risk_budgeted_polish import RiskBudgetedPolish, PolishOptions


def segment_shift_interval(a, b, lower, upper, shifts, axis, margin=0.):
    """Integrating-axis shifts for which a segment intersects the shifted box."""
    left, right = 0., 1.
    for d in range(3):
        if d == axis:
            continue
        lo, hi = lower[d]+shifts[d]-margin, upper[d]+shifts[d]+margin
        delta = b[d]-a[d]
        if delta == 0.:
            if a[d] < lo or a[d] > hi:
                return None
        else:
            t0, t1 = (lo-a[d])/delta, (hi-a[d])/delta
            left, right = max(left,min(t0,t1)), min(right,max(t0,t1))
            if left > right:
                return None
    z0, z1 = a[axis]+left*(b[axis]-a[axis]), a[axis]+right*(b[axis]-a[axis])
    return min(z0,z1)-upper[axis]-margin, max(z0,z1)-lower[axis]+margin


def normal_interval_union_probability(intervals, sigma):
    if not intervals:
        return 0.
    if sigma == 0.:
        return float(any(lo <= 0. <= hi for lo,hi in intervals))
    intervals=sorted(intervals)
    merged=[]
    for lo,hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1]=(merged[-1][0],max(merged[-1][1],hi))
        else:
            merged.append((lo,hi))
    den=sigma*math.sqrt(2.)
    def cdf(x):return .5*math.erfc(-x/den)
    return min(1.,max(0.,sum(cdf(hi)-cdf(lo) for lo,hi in merged)))


def conditional_path_collision(path, obstacles, shifts, sigma, axis=2, margin=0.):
    survival=1.
    points=path.waypoints
    for obstacle,shift in zip(obstacles,shifts):
        intervals=[]
        for a,b in zip(points,points[1:]):
            interval=segment_shift_interval(a,b,obstacle.minimum,obstacle.maximum,shift,axis,margin)
            if interval is not None:
                intervals.append(interval)
        collision=normal_interval_union_probability(intervals,sigma)
        survival*=1.-collision
    return 1.-survival


class ConditionalRiskDE(RiskAwareDE):
    def __init__(self,problem,config,*,axis=2):
        if axis not in (0,1,2):raise ValueError('axis must be 0,1,2')
        if config.perturbation_sigmas is not None or config.selection_mode=='ucb_budgeted':
            raise ValueError('Conditional estimator currently needs one scale and no binomial UCB')
        super().__init__(problem.bounds,problem,config)
        self.axis=axis
        original=problem.environment.obstacles
        self.shifts=[[[moved.minimum[d]-ob.minimum[d] for d in range(3)] for moved,ob in zip(env.obstacles,original)] for env in self._perturbed_banks[0]]
        self.extension_metadata={'risk_evaluation_type':'conditional_gaussian_probability',
          'integrated_axis':axis,'binary_collision_checks':0,
          'cost_limit':'internal_collision_checks is legacy field storing conditional risk evaluations here; not equal-cost binary checks',
          'nominal_nfe':config.base.max_evaluations,'sigma':config.perturbation_sigma}

    def _evaluate(self,vector,chance_alpha=None):
        path=self.problem.decode(vector)
        metrics=self.problem.evaluate(path)
        if self.config.skip_risk_for_nominal_infeasible and not metrics.feasible:
            measured=RiskEvaluation(False,1.,0.,float(metrics.objective),(),math.nan)
        else:
            probability=sum(conditional_path_collision(path,self.problem.environment.obstacles,shift,
                self.config.perturbation_sigma,self.axis,self.problem.safety_margin) for shift in self.shifts)/len(self.shifts)
            self._internal_collision_checks+=len(self.shifts)
            self.extension_metadata['conditional_risk_evaluations']=self._internal_collision_checks
            self.extension_metadata['segment_box_interval_tests']=self._internal_collision_checks*len(self.problem.environment.obstacles)*(len(path.waypoints)-1)
            measured=RiskEvaluation(metrics.feasible,probability,0.,float(metrics.objective),(),math.nan)
        return self._retarget(measured,self.config.chance_alpha if chance_alpha is None else chance_alpha)


class ConditionalPolish(RiskBudgetedPolish):
    def __init__(self,problem,config,options,axis):
        super().__init__(problem,config,options)
        self.axis=axis
    def _create_evaluator(self,config):
        self.conditional_evaluator=ConditionalRiskDE(self.problem,config,axis=self.axis)
        return self.conditional_evaluator
    def optimize(self):
        result=super().optimize()
        self.extension_metadata.update(self.conditional_evaluator.extension_metadata)
        return result


def create_optimizer(problem,base_config,options):
    opts=dict(options)
    axis=opts.pop('axis',2)
    polish=opts.pop('polish',False)
    if polish:
        return ConditionalPolish(problem,base_config,PolishOptions(**opts),axis)
    if opts:raise ValueError(f'Unknown conditional options: {opts}')
    return ConditionalRiskDE(problem,base_config,axis=axis)
