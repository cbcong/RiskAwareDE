"""A new ranking rule must not spend free evaluations or weaken the hard gate."""
import pytest
from uav_path_planning import Bounds3D, ContinuousPathProblem, DEConfig, Environment3D, ObstacleBox
from uav_path_planning.risk_aware_de import RiskAwareDEConfig, RiskEvaluation
from uav_path_planning.conditional_projected_polish import ConditionalProjectedPolish
from uav_path_planning.soft_risk_projected_polish import SoftRiskConditionalDE, SoftRiskProjectedPolish


def problem_config():
    problem = ContinuousPathProblem((1, 1, 1), (9, 9, 1), Environment3D(
        Bounds3D((0, 0, 0), (10, 10, 5)), [ObstacleBox((4, 3, 0), (6, 7, 3))]), intermediate_waypoints=2)
    config = RiskAwareDEConfig(base=DEConfig(population_size=8, generations=30, max_evaluations=100, seed=7),
        perturbation_sigma=.5, perturbation_samples=4, perturbation_seed=43, chance_alpha=.2,
        mutation_strategy='current_to_pbest1', archive_size=0, local_refinement=True,
        local_refinement_mode='partial_shortcut', local_refinement_probability=.7, local_refinement_scale=.7)
    return problem, config


def test_zero_beta_exactly_recovers_frozen_method():
    p, c = problem_config()
    original = ConditionalProjectedPolish(p, c).optimize()
    p, c = problem_config()
    experiment = SoftRiskProjectedPolish(p, c, beta=0).optimize()
    assert experiment == original


def test_positive_weight_keeps_gate_and_nominal_feasibility_primary():
    p, c = problem_config()
    evaluator = SoftRiskConditionalDE(p, c, beta=100.)
    def score(feasible, risk, objective):
        return evaluator._retarget(RiskEvaluation(feasible, risk, 0., objective, ()), .2)
    feasible = score(True, .19, 50.)
    violating = score(True, .2001, 1.)
    nominal_collision = score(False, 0., 0.)
    assert feasible.key < violating.key < nominal_collision.key
    assert score(True, .05, 20.).key < score(True, .10, 20.).key


def test_terminal_acceptance_and_cost_accounting_remain_enforced():
    p, c = problem_config()
    optimizer = SoftRiskProjectedPolish(p, c, beta=.02)
    result = optimizer.optimize()
    assert result.evaluations == 100
    assert result.internal_collision_checks == 400
    independent = SoftRiskConditionalDE(p, c, beta=.02)._evaluate(result.best_vector)
    assert independent.key == result.best_evaluation.key
    assert all(event['accepted_objective_gain'] >= 0 for event in result.local_refinement_events)
    assert result.best_fitness == p.evaluate(p.decode(result.best_vector)).objective


@pytest.mark.parametrize('beta', [-1, float('nan'), float('inf')])
def test_invalid_beta_rejected(beta):
    p, c = problem_config()
    with pytest.raises(ValueError):
        SoftRiskProjectedPolish(p, c, beta=beta)
