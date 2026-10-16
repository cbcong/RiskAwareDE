"""Meaningful matching checks, including real full-budget obstacle problems."""
from dataclasses import replace

import pytest

from uav_path_planning import Bounds3D, ContinuousPathProblem, DEConfig, Environment3D, ObstacleBox
from uav_path_planning.conditional_baseline_control import ConditionalBaselineControl
from uav_path_planning.paper_matched_controls import (
    MatchedBaselineControl, MatchedBinaryDE, MatchedConditionalDE, create_optimizer,
)
from uav_path_planning.risk_aware_de import RiskAwareDEConfig, RiskEvaluation
from uav_path_planning.soft_risk_projected_polish import SoftRiskConditionalDE, SoftRiskProjectedPolish


def problem_config(budget=820):
    problem = ContinuousPathProblem((1, 1, 1), (9, 9, 1), Environment3D(
        Bounds3D((0, 0, 0), (10, 10, 5)), [ObstacleBox((4, 3, 0), (6, 7, 3))]),
        intermediate_waypoints=3)
    config = RiskAwareDEConfig(
        base=DEConfig(population_size=20, generations=40, max_evaluations=budget, seed=7),
        perturbation_sigma=.6, perturbation_samples=16, perturbation_seed=6301,
        chance_alpha=.1, mutation_strategy="current_to_pbest1", p_best_rate=.15,
        archive_size=0, parameter_adaptation="risk_success", risk_tradeoff=0.,
        local_refinement=True, local_refinement_mode="partial_shortcut",
        local_refinement_probability=.7, local_refinement_scale=.7)
    return problem, config


def record(evaluator, risk, objective, feasible=True):
    return evaluator._retarget(RiskEvaluation(feasible, risk, 0., objective, ()), .1)


def test_estimator_independent_conditional_implementation_matches_frozen_nontrivial_run():
    problem, config = problem_config()
    expected = SoftRiskConditionalDE(problem, config, beta=.05).optimize()
    actual = MatchedConditionalDE(problem, config, beta=.05).optimize()
    assert actual == expected
    assert actual.evaluations == 820
    assert actual.internal_collision_checks == 13120
    assert actual.best_evaluation.nominal_feasible
    assert actual.history[0] != actual.history[-1]


def test_full_factory_keeps_frozen_path_history_and_budget_exactly():
    problem, config = problem_config()
    expected = SoftRiskProjectedPolish(problem, config, beta=.05).optimize()
    optimizer = create_optimizer(problem, config, {"experiment_kind": "full", "beta": .05})
    actual = optimizer.optimize()
    assert actual == expected
    assert actual.local_refinement_stats["projected_global_evaluations"] == 656
    assert actual.evaluations == 820
    assert actual.internal_collision_checks == 13120


@pytest.mark.parametrize("kind", ["no_terminal", "binary"])
def test_changed_ablation_preserves_soft_ranking_and_total_budget(kind):
    problem, config = problem_config()
    optimizer = create_optimizer(problem, config, {"experiment_kind": kind, "beta": .05})
    actual = optimizer.optimize()
    evaluator = (MatchedBinaryDE(problem, config, beta=.05) if kind == "binary"
                 else SoftRiskConditionalDE(problem, config, beta=.05))
    assert actual.best_evaluation == evaluator._evaluate(actual.best_vector)
    assert actual.evaluations == 820
    assert actual.internal_collision_checks == 13120
    if kind == "no_terminal":
        assert "projected_global_evaluations" not in actual.local_refinement_stats
    else:
        assert actual.local_refinement_stats["projected_global_evaluations"] == 656
        assert all(event["accepted_objective_gain"] >= 0 for event in actual.local_refinement_events)
        assert actual.best_evaluation.perturbed_collision_rate * 16 == pytest.approx(
            round(actual.best_evaluation.perturbed_collision_rate * 16))


@pytest.mark.parametrize("family", ConditionalBaselineControl.METHODS)
@pytest.mark.parametrize("kind", ["risk_baseline", "projected_baseline"])
def test_each_native_family_receives_soft_cost_and_exact_archive(kind, family):
    problem, config = problem_config(budget=101)
    optimizer = create_optimizer(problem, config, {
        "experiment_kind": kind, "family": family, "beta": .05})
    actual = optimizer.optimize()
    evaluator = SoftRiskConditionalDE(problem, config, beta=.05)
    assert actual.best_evaluation == evaluator._evaluate(actual.best_vector)
    assert actual.evaluations == 101
    assert actual.internal_collision_checks == 1616
    assert optimizer.extension_metadata["experimental_risk_regularization_beta"] == .05
    assert not optimizer.extension_metadata["global_riskaware_shortcuts"]
    if kind == "projected_baseline":
        assert actual.local_refinement_stats["projected_global_evaluations"] == 81
        assert all(event["accepted_objective_gain"] >= 0 for event in actual.local_refinement_events)


def test_beta_affects_native_selection_not_only_returned_archive():
    problem, config = problem_config()
    soft = MatchedBaselineControl(problem, config, beta=.05)
    original = ConditionalBaselineControl(problem, config)
    # A slightly shorter but substantially riskier feasible path changes order.
    shorter = record(soft, .09, 10.)
    safer = record(soft, .01, 10.02)
    assert safer.key < shorter.key
    assert soft._merit(safer, 100.) < soft._merit(shorter, 100.)
    assert original._merit(shorter, 100.) < original._merit(safer, 100.)
    # Both kinds of infeasibility still outrank all soft-cost preferences.
    violating = record(soft, .1001, .01)
    collision = record(soft, 0., .001, feasible=False)
    assert shorter.key < violating.key < collision.key
    assert soft._merit(shorter, 100.) < soft._merit(violating, 100.) < soft._merit(collision, 100.)


def test_binary_and_conditional_share_frozen_key_at_all_gate_boundaries():
    problem, config = problem_config()
    binary = MatchedBinaryDE(problem, config, beta=.05)
    conditional = SoftRiskConditionalDE(problem, config, beta=.05)
    for feasible in (True, False):
        for risk in (0., .05, .1, .10001, 1.):
            assert record(binary, risk, 10., feasible) == record(conditional, risk, 10., feasible)


def test_zero_beta_native_control_preserves_existing_result():
    problem, config = problem_config(budget=101)
    expected = ConditionalBaselineControl(problem, config, global_method="lshade").optimize()
    actual = MatchedBaselineControl(problem, config, beta=0., global_method="lshade").optimize()
    assert actual == expected


@pytest.mark.parametrize("beta", [-.1, float("inf"), float("nan")])
def test_invalid_beta_rejected(beta):
    problem, config = problem_config()
    with pytest.raises(ValueError):
        create_optimizer(problem, config, {"experiment_kind": "binary", "beta": beta})


def test_unused_options_cannot_silently_change_experiment():
    problem, config = problem_config()
    with pytest.raises(ValueError, match="Unsupported no-terminal"):
        create_optimizer(problem, config, {"experiment_kind": "no_terminal", "reserve_fraction": .2})
    with pytest.raises(ValueError, match="determine"):
        create_optimizer(problem, config, {"experiment_kind": "binary", "risk_estimator": "conditional"})
    with pytest.raises(ValueError, match="risk_budgeted"):
        create_optimizer(problem, replace(config, selection_mode="nominal"), {"experiment_kind": "binary"})
