import pytest
from uav_path_planning import Bounds3D,ContinuousPathProblem,DEConfig,Environment3D,ObstacleBox
from uav_path_planning.risk_aware_de import RiskAwareDEConfig
from uav_path_planning.conditional_risk_de import ConditionalRiskDE
from uav_path_planning.conditional_projected_polish import create_optimizer,length_gradient_normalized

def fixture():
    p=ContinuousPathProblem((1,1,1),(9,9,1),Environment3D(Bounds3D((0,0,0),(10,10,5)),[ObstacleBox((4,3,0),(6,7,3))]),intermediate_waypoints=2)
    c=RiskAwareDEConfig(base=DEConfig(population_size=8,generations=30,max_evaluations=100,seed=7),perturbation_sigma=.5,perturbation_samples=4,perturbation_seed=43,chance_alpha=.2,mutation_strategy='current_to_pbest1',archive_size=0,local_refinement=True,local_refinement_mode='partial_shortcut',local_refinement_probability=.7,local_refinement_scale=.7)
    return p,c

def test_normalized_length_gradient_matches_independent_finite_difference():
    p,_=fixture();v=[2.,4.,3.,6.,8.,2.]
    g=length_gradient_normalized(p,v)
    for i,(lo,hi) in enumerate(p.bounds):
        minus=list(v);plus=list(v);h=1e-6
        minus[i]-=h*(hi-lo);plus[i]+=h*(hi-lo)
        numeric=(p.decode(plus).length-p.decode(minus).length)/(2*h)
        assert g[i]==pytest.approx(numeric,abs=1e-7)

@pytest.mark.parametrize('method',['risk_de','lshade'])
@pytest.mark.parametrize('projection',['tangent','ignore_measured_gradient','length_only'])
def test_all_gradient_and_trial_costs_are_charged_and_returned_path_consistent(method,projection):
    p,c=fixture();o=create_optimizer(p,c,{'global_method':method,'projection':projection})
    r=o.optimize();e=ConditionalRiskDE(p,c)._evaluate(r.best_vector)
    s=r.local_refinement_stats
    assert s['projected_global_evaluations']+s['projected_gradient_probes']+s['projected_line_trials']==100
    assert r.evaluations==100 and r.internal_collision_checks==400
    assert r.best_evaluation.key==e.key
    if projection=='length_only':assert s['projected_gradient_probes']==0


def test_adaptive_axis_and_amortized_line_search():
    p, c = fixture()
    o = create_optimizer(p, c, {'global_method': 'risk_de', 'projection': 'tangent', 'axis': 'adaptive', 'max_line_steps': 3})
    r = o.optimize()
    s = r.local_refinement_stats
    assert s['projected_global_evaluations'] + s['projected_gradient_probes'] + s['projected_line_trials'] == 100
    assert r.evaluations == 100
    assert r.best_fitness == p.evaluate(p.decode(r.best_vector)).objective
    assert all(x['accepted_objective_gain'] >= 0 for x in r.local_refinement_events)
    from uav_path_planning.adaptive_axis_conditional_de import AdaptiveAxisConditionalDE
    assert r.best_evaluation.key == AdaptiveAxisConditionalDE(p, c)._evaluate(r.best_vector).key
    assert r.internal_collision_checks == 400
    assert o.extension_metadata['max_line_steps'] == 3


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_amortization_cannot_skip_the_line_search_or_stall(value):
    p, c = fixture()
    with pytest.raises(ValueError, match='max_line_steps'):
        create_optimizer(p, c, {'max_line_steps': value})


def test_adaptive_lshade_cannot_silently_run_de():
    p, c = fixture()
    with pytest.raises(ValueError, match='refusing to substitute'):
        create_optimizer(p, c, {'axis': 'adaptive', 'global_method': 'lshade'})


def test_binary_risk_cannot_claim_adaptive_integration():
    p, c = fixture()
    with pytest.raises(ValueError, match='requires conditional'):
        create_optimizer(p, c, {'axis': 'adaptive', 'risk_estimator': 'binary'})
