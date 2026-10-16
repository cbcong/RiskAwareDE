"""Freeze the resource-bounded first external comparison before new-map results."""
from pathlib import Path
import datetime
import itertools
import json
import platform
import random
import subprocess
import sys
from common import HERE,REPO,dump,sha,generate_bank

destination=HERE/'runs/confirmatory_v1'
assert not (destination/'protocol.json').exists(),'Protocol already frozen; never overwrite'
decision={
    'date':'2026-10-15','stage':'protocol locked before test-map generation',
    'scientific_question':'How do a frozen low-sample geometric planner and two implemented external risk formulations trade path quality, empirical reliability and measured compute on new instances?',
    'methods':['ours','cc_vpsto','saa','astar'],
    'scales':'Primary sigma=.6 only; .9 pressure deferred to preserve resources',
    'maps':24,'families':3,'maps_per_family':8,'stochastic_seeds':[101,102,103,104,105],
    'expected_jobs':384,
    'resource_decision':'24 new maps x5 stochastic seeds x3 risk planners plus24 deterministic A* =384. Reduced from exploratory60x20x4 plan BEFORE test observations. Map remains inference unit.',
    'budget':'Frozen RiskAwareDE820 calls; external risk formulations6s including setup/initialization; A* completion. NOT equal-time superiority or unmodified native-system comparison.',
    'comparators':'CC-VPSTO paper reimplementation preserving official VP-STO OBF/CMA; SAA-AVaR explicit geometric adaptation preserving AVaR/SCP/OSQP, nominal A* initialization; grid A* descriptive reference.',
    'development':'Three separate pilot maps, seeds41/42. CC n_via4 sigma_init40 selected by6/6 internal+nominal feasibility at6s; SAA nominal A* +>=5free chosen for initialization plateau on pilot, no external CFR used for these settings.',
    'external_evaluation':'20000 independent common Gaussian worlds per map, fixed obstacle translation along entire path; point robot. Conservative cubic chord error<=.002; exact slab predicate.',
    'failure_policy':'No path=zero joint CFR/nominal success, length missing; nominally infeasible retained as zero joint CFR. Also record raw CFR of any incumbent and its method-specific training feasibility separately.',
    'risk_target':'External CFR>=.825 for joint nominal-and-target pass. Per-path Wilson interval reported. Internal risk parameters are method-specific, not equated.',
    'statistics':'24 map means,5 seeds/map; two main-vs-risk-comparator contrasts xCFR/length=4 Holm family; length on pairwise nominally feasible seeds, counts disclosed. A*descriptive. 20000 stratified-by-family bootstrap resamples; all effects retained.',
    'interpretation':'New held-out instances from established generator families. Limited geometric adaptation study, not a ranking of original full robotics systems; no dynamical-flight claim.',
    'stopping':'Fixed384 jobs; no optional stopping on effect/pvalue. Pause/resume allowed for machine availability. Any implementation fix after first test requires explicit invalidation/new batch, never silently mixed.'}
dump(destination/'design_before_maps.json',decision)
sources=[HERE/'common.py',HERE/'run_study.py',HERE/'freeze_confirmatory.py',HERE/'saa/adapter.py',HERE/'saa/initialized_adapter.py',HERE/'cc_vpsto/adapter.py']
sources+=list((HERE/'cc_vpsto/vp-sto/vpsto').glob('*.py'))
sources+=list((REPO/'src/uav_path_planning').glob('*.py'))
sources+=[REPO/'scripts/run_extension_candidate.py',REPO/'configs/risk_aware_main_20261009_soft050.json']
import os
source_hashes={Path(os.path.relpath(p,HERE)).as_posix():sha(p) for p in sources}
bank=generate_bank(HERE/'banks/confirmatory_v1',202610150701,8,'heldout')
jobs=[];blocks=list(itertools.product(bank['maps'],decision['stochastic_seeds']))
random.Random(202610150702).shuffle(blocks)
methods=['ours','cc_vpsto','saa'];orders=list(itertools.permutations(methods))
for block,(m,seed) in enumerate(blocks):
    for method in orders[block%len(orders)]:
        jobs.append({'id':f'{method}_{m["name"]}_s{seed}','method':method,'map_file':m['file'],
            'map':m['name'],'family':m['family'],'map_sha256':m['sha256'],'seed':seed,'block':block})
    if seed==101:
        jobs.append({'id':f'astar_{m["name"]}','method':'astar','map_file':m['file'],'map':m['name'],
            'family':m['family'],'map_sha256':m['sha256'],'seed':101,'block':block})
assert len(jobs)==384
protocol={**decision,'study':'confirmatory_v1','evidence_stage':'prospective geometric adaptation comparison',
    'bank':'banks/confirmatory_v1','external_budget_seconds':6.,'external_samples':20000,
    'external_root':202610150703,'sigmas':[.6],'curve_tolerance':.002,'saa_grid_initialization':True,
    'method_options':{'cc_vpsto':{'n_via':4,'sigma_init':40.,'n_samples':100,'eta':.175,'beta':.05},
                      'saa':{'alpha':.1,'samples':32,'min_free_waypoints':5}},
    'source_sha256':source_hashes,'jobs':jobs,'map_manifest_sha256':sha(HERE/'banks/confirmatory_v1/manifest.json')}
dump(destination/'protocol.json',protocol)
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze','--disable-pip-version-check'],text=True)
(destination/'environment_freeze.txt').write_text(freeze,encoding='utf8')
print('Frozen24 new maps,384 jobs, protocol SHA256',sha(destination/'protocol.json'))
