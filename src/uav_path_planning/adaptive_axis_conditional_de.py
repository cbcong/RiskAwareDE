"""Conditional Gaussian integration with one geometry-selected axis per box.

The minimum-extent choice is a deterministic heuristic, not a theorem that it
minimizes estimator variance. Each obstacle keeps the same integration axis
across every path segment and every sample. This is essential: intervals on
different coordinates cannot be merged into one univariate Gaussian union.
"""

import math
from uav_path_planning.risk_aware_de import RiskAwareDE, RiskEvaluation
from uav_path_planning.conditional_risk_de import segment_shift_interval, normal_interval_union_probability

def adaptive_conditional_path_collision(path, obstacles, shifts, sigma, mode, margin=0.):
    if mode != 'min_extent':
        raise ValueError("Unknown mode or unsupported mode: only 'min_extent' is valid; "
                         "'travel_orthogonal' mixed coordinate axes across segments")
    if len(obstacles) != len(shifts):
        raise ValueError('Each obstacle must have exactly one translation sample')
    survival = 1.
    points = path.waypoints
    for obstacle, shift in zip(obstacles, shifts):
        intervals = []
        extents = [obstacle.maximum[d] - obstacle.minimum[d] for d in range(3)]
        axis = extents.index(min(extents))
            
        for a, b in zip(points, points[1:]):
            interval = segment_shift_interval(a, b, obstacle.minimum, obstacle.maximum, shift, axis, margin)
            if interval is not None:
                intervals.append(interval)
        collision = normal_interval_union_probability(intervals, sigma)
        survival *= 1. - collision
    return 1. - survival


class AdaptiveAxisConditionalDE(RiskAwareDE):
    def __init__(self, problem, config, mode='min_extent'):
        if mode != 'min_extent':
            raise ValueError(f"Unknown mode or unsupported mode: {mode}; only 'min_extent' "
                             "keeps a valid fixed integration coordinate per obstacle")
        if config.perturbation_sigmas is not None or config.selection_mode == 'ucb_budgeted':
            raise ValueError('Conditional estimator currently needs one scale and no binomial UCB')
        super().__init__(problem.bounds, problem, config)
        self.mode = mode
        
        original = problem.environment.obstacles
        self.shifts = [
            [[moved.minimum[d] - ob.minimum[d] for d in range(3)]
             for moved, ob in zip(env.obstacles, original)]
            for env in self._perturbed_banks[0]
        ]
        
        self.extension_metadata = {
            'risk_evaluation_type': 'adaptive_conditional_gaussian_probability',
            'integration_mode': mode,
            'integrated_axis_by_obstacle': [
                min(range(3), key=lambda d: ob.maximum[d] - ob.minimum[d])
                for ob in original
            ],
            'axis_selection_rule': 'deterministic obstacle geometry only; ties use first coordinate',
            'axis_optimality_claim': False,
            'binary_collision_checks': 0,
            'cost_limit': 'internal_collision_checks is legacy field storing conditional risk evaluations here; not equal-cost binary checks',
            'nominal_nfe': config.base.max_evaluations,
            'sigma': config.perturbation_sigma
        }

    def _evaluate(self, vector, chance_alpha=None):
        path = self.problem.decode(vector)
        metrics = self.problem.evaluate(path)
        if self.config.skip_risk_for_nominal_infeasible and not metrics.feasible:
            measured = RiskEvaluation(False, 1., 0., float(metrics.objective), (), math.nan)
        else:
            probability = sum(
                adaptive_conditional_path_collision(
                    path, self.problem.environment.obstacles, shift,
                    self.config.perturbation_sigma, self.mode, self.problem.safety_margin
                ) for shift in self.shifts
            ) / len(self.shifts)
            
            self._internal_collision_checks += len(self.shifts)
            self.extension_metadata['conditional_risk_evaluations'] = self._internal_collision_checks
            self.extension_metadata['segment_box_interval_tests'] = self._internal_collision_checks * len(self.problem.environment.obstacles) * (len(path.waypoints) - 1)
            
            measured = RiskEvaluation(metrics.feasible, probability, 0., float(metrics.objective), (), math.nan)
        return self._retarget(measured, self.config.chance_alpha if chance_alpha is None else chance_alpha)


def create_optimizer(problem, base_config, options):
    opts = dict(options)
    mode = opts.pop('mode', 'min_extent')
    if opts:
        raise ValueError(f"Unknown adaptive axis options: {opts}")
    return AdaptiveAxisConditionalDE(problem, base_config, mode=mode)
