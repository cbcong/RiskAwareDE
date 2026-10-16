from dataclasses import replace
from statistics import NormalDist

import pytest

from uav_path_planning import Bounds3D, ContinuousPathProblem, DEConfig, Environment3D, ObstacleBox
from uav_path_planning.risk_aware_de import RiskAwareDEConfig
from uav_path_planning.risk_sampling_de import create_optimizer, standardized_gaussian_bank


@pytest.mark.parametrize("design", ["latin_hypercube", "antithetic_lhs"])
def test_gaussian_probability_strata_and_reproducibility(design):
    bank = standardized_gaussian_bank(16, 12, 6301, design)
    assert bank == standardized_gaussian_bank(16, 12, 6301, design)
    assert bank != standardized_gaussian_bank(16, 12, 6302, design)
    normal = NormalDist()
    for dimension in range(12):
        strata = sorted(int(16 * normal.cdf(row[dimension])) for row in bank)
        assert strata == list(range(16))
    if design == "antithetic_lhs":
        for i in range(8):
            assert bank[i] == [-value for value in bank[i + 8]]
        assert any(row[0] * row[1] < 0 for row in bank)


@pytest.mark.parametrize("design", ["latin_hypercube", "antithetic_lhs"])
def test_exact_budget_and_repeatable_optimizer(design):
    environment = Environment3D(Bounds3D((0, 0, 0), (10, 10, 5)), [ObstacleBox((4, 3, 0), (6, 7, 3))])
    problem = ContinuousPathProblem((1, 1, 1), (9, 9, 1), environment, intermediate_waypoints=2)
    config = RiskAwareDEConfig(
        base=DEConfig(population_size=8, generations=4, seed=7, max_evaluations=17),
        perturbation_sigma=0.6, perturbation_samples=16, perturbation_seed=6301,
    )
    first = create_optimizer(problem, config, {"bank_design": design})
    second = create_optimizer(problem, config, {"bank_design": design})
    assert first._internal_collision_checks == 0
    a, b = first.optimize(), second.optimize()
    assert a.evaluations == 17
    assert a.internal_collision_checks == 17 * 16
    assert a.best_evaluation == b.best_evaluation
    assert first.extension_metadata == second.extension_metadata
    with pytest.raises(ValueError, match="UCB"):
        create_optimizer(problem, replace(config, selection_mode="ucb_budgeted"), {"bank_design": design})


def test_antithetic_requires_pairs_and_rejects_unknown_design():
    with pytest.raises(ValueError, match="even"):
        standardized_gaussian_bank(15, 3, 6301, "antithetic_lhs")
    with pytest.raises(ValueError, match="unsupported"):
        standardized_gaussian_bank(16, 3, 6301, "invalid")
