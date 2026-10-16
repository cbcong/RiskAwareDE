import math
import random
import pytest
from uav_path_planning import Bounds3D, ContinuousPathProblem, DEConfig, Environment3D, ObstacleBox
from uav_path_planning.models import Path3D
from uav_path_planning.risk_aware_de import RiskAwareDEConfig
from uav_path_planning.conditional_risk_de import segment_shift_interval,normal_interval_union_probability,conditional_path_collision,create_optimizer


def test_integrated_axis_intervals_agree_with_independent_segment_collision_predicate():
    rng=random.Random(109)
    for axis in range(3):
        for _ in range(100):
            a,b=[tuple(rng.uniform(-3,3) for _ in range(3)) for _ in range(2)]
            lo=(-.7,-.4,-.5);hi=(.6,.8,.9)
            shift=[rng.gauss(0,.5) for _ in range(3)]
            for margin in (0.,.15):
                interval=segment_shift_interval(a,b,lo,hi,shift,axis,margin)
                for value in (-3.,-.7,0.,.4,2.1):
                    shift[axis]=value
                    ob=ObstacleBox(tuple(x+y for x,y in zip(lo,shift)),tuple(x+y for x,y in zip(hi,shift)))
                    env=Environment3D(Bounds3D((-10,-10,-10),(10,10,10)),[ob])
                    expected=Path3D([a,b]).collision_count(env,safety_margin=margin)>0
                    actual=interval is not None and interval[0]<=value<=interval[1]
                    assert actual==expected


def test_overlapping_intervals_are_not_double_counted_and_zero_sigma_exact():
    p=normal_interval_union_probability([(-1,1),(-.5,.5)],1.)
    assert p==pytest.approx(math.erf(1/math.sqrt(2)))
    assert normal_interval_union_probability([(1,2)],0.)==0.
    assert normal_interval_union_probability([(-1,0)],0.)==1.


def test_conditional_probability_matches_closed_form_path_and_obstacle_union():
    path=Path3D([(0,0,0),(10,0,0)])
    ob=ObstacleBox((4,-1,1),(6,1,2))
    p=.5*(math.erfc(1/math.sqrt(2))-math.erfc(2/math.sqrt(2)))
    assert conditional_path_collision(path,[ob],[[0,0,99]],1.)==pytest.approx(p)
    assert conditional_path_collision(path,[ob,ob],[[0,0,0],[0,0,0]],1.)==pytest.approx(1-(1-p)**2)


@pytest.mark.parametrize('polish',[False,True])
def test_budget_and_returned_conditional_risk_agree(polish):
    problem=ContinuousPathProblem((1,1,1),(9,9,1),Environment3D(Bounds3D((0,0,0),(10,10,5)),[ObstacleBox((4,3,0),(6,7,3))]),intermediate_waypoints=2)
    cfg=RiskAwareDEConfig(base=DEConfig(population_size=8,generations=30,max_evaluations=100,seed=7),perturbation_sigma=.5,perturbation_samples=4,perturbation_seed=43,chance_alpha=.2,mutation_strategy='current_to_pbest1',archive_size=0,local_refinement=True,local_refinement_mode='partial_shortcut',local_refinement_probability=.7,local_refinement_scale=.7)
    result=create_optimizer(problem,cfg,{'polish':polish}).optimize()
    evaluated=create_optimizer(problem,cfg,{})._evaluate(result.best_vector)
    assert result.evaluations==100 and result.internal_collision_checks==400
    assert result.best_evaluation.key==evaluated.key
    assert result.best_evaluation.perturbed_collision_rate==evaluated.perturbed_collision_rate
