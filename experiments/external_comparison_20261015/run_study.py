"""Serial, resumable external-planner comparison with separate pilot/test banks.

Run --prepare pilot, then --run pilot. Formal protocols are written separately
after implementation checks; a protocol hash binds every completed job.
"""
from pathlib import Path
import argparse
import importlib.util
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import dump,sha,generate_bank,boxes_of,score,bezier_chords,ppoly_to_bezier
import numpy as np


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def load_adapter(name,initialized=False):
    filename='initialized_adapter.py' if name=='saa' and initialized else 'adapter.py'
    spec=importlib.util.spec_from_file_location('external_'+name,HERE/name/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    return x


def ours(scene,seed):
    from uav_path_planning.benchmarks import BenchmarkConfig,_problem
    from uav_path_planning.scenarios import Scenario3D
    from uav_path_planning.soft_risk_projected_polish import SoftRiskProjectedPolish
    sys.path.insert(0,str(HERE.parents[1]/'scripts'))
    from run_extension_candidate import risk_config
    locked=read(HERE.parents[1]/'configs/risk_aware_main_20261009_soft050.json')
    cfg=BenchmarkConfig(**locked['benchmark_config'])
    start=time.perf_counter()
    problem=_problem(Scenario3D.from_dict(scene),cfg)
    optimizer=SoftRiskProjectedPolish(problem,risk_config(cfg,seed),**locked['extension_options'])
    result=optimizer.optimize()
    points=np.asarray(problem.decode(result.best_vector).waypoints,float)
    elapsed=time.perf_counter()-start
    return {'method':'RiskAwareDE-frozen','points':points,'success':result.best_evaluation.nominal_feasible,
            'stats':{'wall_seconds':elapsed,'objective_evaluations':result.evaluations,
                     'training_risk':result.best_evaluation.perturbed_collision_rate,
                     'budget_type':'unchanged 820 evaluations; not an equal-time adaptation'}}


def astar(scene,seed):
    from uav_path_planning.benchmarks import BenchmarkConfig,_result_for_astar
    from uav_path_planning.scenarios import Scenario3D
    start=time.perf_counter()
    result=_result_for_astar(Scenario3D.from_dict(scene),BenchmarkConfig(grid_resolution=1.,safety_margin=0.))
    return {'method':'Grid-A* (26 neighbors, unit resolution)','points':np.asarray(result.path.waypoints) if result.path else None,
            'success':result.success,'stats':{'wall_seconds':time.perf_counter()-start,
            'expanded_nodes':result.evaluations,'budget_type':'until completion; descriptive nominal reference'}}


def worker(study,job_id):
    directory=HERE/'runs'/study; protocol=read(directory/'protocol.json')
    job=next(j for j in protocol['jobs'] if j['id']==job_id)
    for relative,expected in protocol.get('source_sha256',{}).items():
        assert sha(HERE/relative)==expected,f'Frozen implementation changed: {relative}'
    scene=read(HERE/protocol['bank']/job['map_file'])
    assert sha(HERE/protocol['bank']/job['map_file'])==job['map_sha256']
    method=job['method'];budget=protocol['external_budget_seconds']
    if method=='ours':result=ours(scene,job['seed'])
    elif method=='astar':result=astar(scene,job['seed'])
    else:
        adapter=load_adapter(method,protocol.get('saa_grid_initialization',False))
        result=adapter.plan(scene['start'],scene['goal'],boxes_of(scene),
                            np.asarray([scene['bounds']['minimum'],scene['bounds']['maximum']]),
                            seed=job['seed'],budget_seconds=budget,sigma=.6,
                            **protocol['method_options'].get(method,{}))
    result=jsonable(result)
    points=result.get('points');error=0.;native_bounds=None
    if points is None:
        metrics={'nominal_success':False,'chord_length':None,'length':None,
                 'external_samples':0,'failure':'no_path'};bits={}
        for sigma in protocol['sigmas']:
            metrics[f'cfr_{sigma}']=metrics[f'joint_cfr_{sigma}']=0.
    else:
        if method=='cc_vpsto':
            native=ppoly_to_bezier(result['coefficients'],result['breaks'])
            points,error=bezier_chords(native,protocol['curve_tolerance'])
            # Arc-length quadrature uses the native cubic derivative, not its drawing samples.
            coef=np.asarray(result['coefficients']);lengths=[];native_bounds=True
            bounds=np.asarray([scene['bounds']['minimum'],scene['bounds']['maximum']])
            for count in [32,64]:
                x,w=np.polynomial.legendre.leggauss(count);length=0.
                for i,h in enumerate(np.diff(result['breaks'])):
                    t=(x+1)*h/2
                    velocity=3*coef[0,i]*t[:,None]**2+2*coef[1,i]*t[:,None]+coef[2,i]
                    length+=float(np.dot(w,np.linalg.norm(velocity,axis=1))*h/2)
                    if count==32:
                        for axis in range(3):
                            c=coef[:,i,axis];roots=np.roots([3*c[0],2*c[1],c[2]])
                            times=[0.,h]+[float(r.real) for r in roots if abs(r.imag)<1e-10 and 0<r.real<h]
                            values=np.polyval(c,times)
                            native_bounds &= bool(values.min()>=bounds[0,axis]-1e-9 and values.max()<=bounds[1,axis]+1e-9)
                lengths.append(length)
            result['native_arc_length_quadrature']=lengths[-1]
            result['arc_length_32_vs_64_difference']=abs(lengths[1]-lengths[0])
            result['native_bounds_extrema_checked']=native_bounds
        metrics,bits=score(scene,points,samples=protocol['external_samples'],external_root=protocol['external_root'],
                           sigmas=tuple(protocol['sigmas']),curve_error=error,native_within_bounds=native_bounds)
        metrics['length']=result.get('native_arc_length_quadrature',metrics['chord_length'])
    stats=result.get('stats',{})
    reported_success=bool(result.get('success',result.get('status')=='feasible'))
    if method=='ours':training_feasible=bool(stats['training_risk']<=.1 and reported_success)
    elif method=='saa':training_feasible=bool(result['risk_feasible'] and not result['nominal_collision'])
    elif method=='cc_vpsto':training_feasible=reported_success
    else:training_feasible=reported_success
    metrics['planner_reported_success']=reported_success
    metrics['internal_constraints_satisfied']=training_feasible
    metrics['success_semantics']='Internal definitions differ; external nominal feasibility and raw CFR are shared comparisons.'
    for sigma in protocol['sigmas']:
        metrics[f'returned_success_cfr_{sigma}']=metrics.get(f'cfr_{sigma}',0.) if training_feasible and metrics['nominal_success'] else 0.
    elapsed=stats.get('wall_seconds',result.get('elapsed_seconds'))
    record={'job':job,'protocol_sha256':sha(directory/'protocol.json'),'result':result,
            'metrics':metrics,'planning_seconds':elapsed,'evidence_stage':protocol['evidence_stage']}
    if bits:
        np.savez_compressed(directory/'results'/f'{job_id}_external_bits.npz',**bits)
    dump(directory/'results'/f'{job_id}.json',record)
    print(json.dumps({'id':job_id,'planning_seconds':elapsed,'nominal_success':metrics['nominal_success'],
                      'cfr':metrics.get('cfr_0.6')},ensure_ascii=False),flush=True)


def prepare_pilot():
    directory=HERE/'runs/pilot'
    if (directory/'protocol.json').exists():return
    bank=generate_bank(HERE/'banks/pilot',202610150101,1,'pilot')
    methods=['ours','cc_vpsto','saa','astar'];jobs=[]
    blocks=list(itertools.product(bank['maps'],[41,42]))
    random.Random(2026101502).shuffle(blocks)
    for bi,(m,seed) in enumerate(blocks):
        order=methods[bi%4:]+methods[:bi%4]
        for method in order:
            if method=='astar' and seed!=41:continue
            jobs.append({'id':f'{method}_{m["name"]}_s{seed}','method':method,'map_file':m['file'],
                         'map':m['name'],'family':m['family'],'map_sha256':m['sha256'],'seed':seed})
    protocol={'study':'pilot','evidence_stage':'development pilot; excluded from confirmatory results',
              'bank':'banks/pilot','external_budget_seconds':3.,'external_samples':1000,'external_root':2026101503,
              'sigmas':[.6],'curve_tolerance':.002,'method_options':{},'jobs':jobs,
              'timing':'RiskAwareDE original820 calls; external methods timed3s; A* completes. Costs descriptive, not strict equal-time.',
              'statistics':'No inferential claims from this development pilot'}
    dump(directory/'protocol.json',protocol)


def run(study,methods=None,limit=None):
    import psutil
    directory=HERE/'runs'/study;protocol=read(directory/'protocol.json')
    (directory/'results').mkdir(parents=True,exist_ok=True);(directory/'logs').mkdir(exist_ok=True)
    jobs=[j for j in protocol['jobs'] if not methods or j['method'] in methods]
    if limit:jobs=jobs[:limit]
    stop=directory/'STOP'; receipts=[]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    dump(directory/'environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'numpy':np.__version__,'logical_cpu_count':os.cpu_count(),'total_memory_bytes':psutil.virtual_memory().total,
        'threads':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}})
    for index,job in enumerate(jobs):
        output=directory/'results'/f'{job["id"]}.json'
        if output.exists():
            assert read(output)['protocol_sha256']==sha(directory/'protocol.json'),'Do not mix protocols'
            continue
        if stop.exists():break
        if psutil.virtual_memory().available<800*1024**2:
            dump(directory/'status.json',{'status':'paused_low_memory','next_job':job,'completed':len(list((directory/'results').glob('*.json')))})
            return
        start=time.perf_counter();peak=0; timedout=False
        with (directory/'logs'/f'{job["id"]}.log').open('w',encoding='utf8') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',study,job['id']],
                                     cwd=HERE,stdout=log,stderr=subprocess.STDOUT,env=env)
            monitor=psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    tree=[monitor]+monitor.children(recursive=True)
                    rss=0
                    for item in tree:
                        try:rss+=item.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,rss)
                except psutil.NoSuchProcess:pass
                if time.perf_counter()-start>120:
                    try:
                        for child in monitor.children(recursive=True):
                            try:child.kill()
                            except psutil.NoSuchProcess:pass
                    except psutil.NoSuchProcess:pass
                    process.kill();timedout=True;break
                dump(directory/'status.json',{'status':'running','current':job['id'],'ordinal':index+1,'queue_total':len(jobs),
                    'completed':len(list((directory/'results').glob('*.json'))),'elapsed_job_seconds':time.perf_counter()-start,
                    'peak_worker_rss_bytes':peak})
                time.sleep(.5)
            code=process.wait()
        receipt={'id':job['id'],'exit_code':code,'hard_timeout':timedout,'process_seconds':time.perf_counter()-start,'peak_process_tree_rss_bytes':peak}
        with (directory/'process_log.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(receipt)+'\n')
        print(json.dumps(receipt),flush=True)
        if code:
            dump(directory/'status.json',{'status':'stopped_on_error','failed_job':job['id'],'receipt':receipt})
            raise SystemExit(f'Worker failed; inspect {directory}/logs/{job["id"]}.log')
    dump(directory/'status.json',{'status':'paused_by_stop_file' if stop.exists() else 'queue_complete',
         'completed':len(list((directory/'results').glob('*.json'))),'protocol_expected_jobs':len(protocol['jobs'])})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',choices=['pilot']);parser.add_argument('--run')
    parser.add_argument('--worker',nargs=2);parser.add_argument('--methods',nargs='+');parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.prepare:prepare_pilot()
    if args.worker:worker(*args.worker)
    if args.run:run(args.run,args.methods,args.limit)
