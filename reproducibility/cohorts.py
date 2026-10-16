"""Reproduce the complete frozen E or P6 queue with one shared scientific engine.

Every task commits atomically and resumes by validated checkpoint. Smoke runs
exercise a small explicit subset and never produce full-study statistics.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for name in ('src','scripts'):
    sys.path.insert(0,str(ROOT/name))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

from reproducibility.atomic_io import write_json as write


def checksum(bundle):
    return hashlib.sha256(json.dumps({k:v for k,v in bundle.items() if k!='checkpoint_sha256'},
        ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()

def verify_sources(reference):
    manifest=read(Path(reference)/'execution_manifest.json')
    expected=manifest['source_sha256']
    # Historical launch/configuration files are archived provenance, not runtime
    # inputs. All current scientific modules and imported helpers remain exact.
    required=[p.relative_to(ROOT).as_posix() for p in (ROOT/'src/uav_path_planning').glob('*.py')]
    required += ['scripts/'+n+'.py' for n in ('seven_category_protocol','run_extension_candidate',
        'run_frozen_stress_suite','run_six_category_comparison','summarize_seven_category_run',
        'summarize_p6_run','summarize_six_category_comparison','ood_category_protocol','ood_exposure','ood_method_lock')]
    missing=[x for x in required if x not in expected or not (ROOT/x).is_file() or sha(ROOT/x)!=expected[x]]
    if missing:
        raise ValueError('Scientific source differs from reference: '+', '.join(missing))
    return {name:expected[name] for name in required}

def scientific_output(bundle):
    row={k:v for k,v in bundle['row'].items() if k not in ('runtime_sec','cpu_seconds')}
    return dict(row=row,waypoints=bundle['detail']['waypoints'],
        internal_risk=bundle['detail']['internal_risk'],external_trials=bundle['external_trials'],
        distribution_external_trials=bundle.get('distribution_external_trials',{}))

def equivalent(a,b):
    """Strict IDs/bits; floating arithmetic tolerance far below score precision."""
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and isinstance(b,(float,int)) and not isinstance(b,bool):
        return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
    return a==b

def execute(job,protocol):
    from seven_category_protocol import execute_job,minimalvalidate_result
    bundle=execute_job(job,protocol)
    minimalvalidate_result(job,protocol,bundle)
    bundle['checkpoint_sha256']=checksum(bundle)
    return bundle

def stopped(output):
    return any((p/'STOP').exists() or (p/'STOP_REQUESTED.json').exists() for p in (output,output.parent))

def protect_output(output,reference):
    output=Path(output).resolve();reference=Path(reference).resolve()
    if output==reference or reference in output.parents:
        raise ValueError('Use a separate output outside reference evidence.')
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Inside this repository, reproduction writes are restricted to outputs/.')
    return output,reference

def run(cohort,output,reference,smoke=False,workers=1):
    from seven_category_protocol import minimalvalidate_result
    from seven_category_control import FileLock
    from .runtimes import fixed_runtime_plan, RuntimePool
    from .fixed_environment import identity as environment_identity
    output,reference=protect_output(output,reference)
    if workers<1:raise ValueError('workers must be positive')
    source_hashes=verify_sources(reference)
    protocol=read(reference/'protocol.json');all_jobs=read(reference/'jobs.json')
    if cohort=='p6' and protocol.get('profile')!='p6':raise ValueError('Not the frozen P6 protocol')
    if cohort=='e' and protocol.get('profile')!='manuscript':raise ValueError('Not the frozen manuscript E protocol')
    selected_methods=['full','bare_de','binary_risk'] if cohort=='e' else ['full','risk_polish_de','bare_lshade']
    jobs=[next(j for j in all_jobs if j['method']==m) for m in selected_methods] if smoke else all_jobs
    if smoke:
        if cohort=='e':
            for root in protocol['map_roots']:
                for method in ('full','bare_lshade','risk_polish_lshade'):
                    jobs.append(next(j for j in all_jobs if j['map_root']==root and j['method']==method))
                for method in ('full','bare_lshade'):
                    jobs.append(next(j for j in all_jobs if j['map_root']==root and j['method']==method
                        and j['scene']==protocol['scenarios'][str(root)][0]['name']
                        and j['seed']==min(protocol['planner_seeds'])))
                jobs.append(min((j for j in all_jobs if j['map_root']==root and j['method']=='full'),
                                key=lambda j:(j['scene'],j['seed'])))
            jobs.append(next(j for j in all_jobs if j['id']=='r20260923_m06_nominal_search_s0014'))
        else:
            jobs.append(next(j for j in all_jobs if j['method']=='risk_polish_lshade'))
        jobs=list({j['id']:j for j in jobs}.values())
    runtime_plan,environments=fixed_runtime_plan(all_jobs)
    environment_sha=environment_identity()
    contract={'schema':'RiskAwareDE-single-environment-v1','cohort':cohort,'smoke':smoke,
        'reference_protocol_sha256':sha(reference/'protocol.json'),'reference_jobs_sha256':sha(reference/'jobs.json'),
        'scientific_source_sha256':source_hashes,'job_ids':[j['id'] for j in jobs],
        'full_matrix_jobs':len(all_jobs),'reference_path':str(reference),
        'environment_lock_sha256':environment_sha,
        'scope':'Fresh recomputation of all methods in one fixed environment on previously observed maps; historical differences are reported, not used as a stopping or selection rule.'}
    output.mkdir(parents=True,exist_ok=True)
    with FileLock(output/'worker.lock'):
        if (output/'reproduction.json').exists():
            if read(output/'reproduction.json')!=contract:
                raise ValueError('Output already belongs to different inputs, mode, or source. Choose a new output.')
        else:
            unexpected=[p for p in output.iterdir() if p.name!='worker.lock']
            if unexpected:raise ValueError('Nonempty output has no matching reproduction contract.')
            write(output/'reproduction.json',contract)
        # Initialization is recoverable if interrupted between its atomic files.
        for filename in ('protocol.json','jobs.json'):
            if not (output/filename).exists():
                temporary=output/(filename+'.tmp')
                shutil.copy2(reference/filename,temporary)
                temporary.replace(output/filename)
        if not (output/'execution_manifest.json').exists():
            original=read(reference/'execution_manifest.json')
            original.update(reproduction_created_at=now(),reproduction_of=str(reference),
                source_root=str(ROOT),reproduction_contract_sha256=sha(output/'reproduction.json'),
                environment_lock_sha256=environment_sha,all_jobs_recomputed=True,
                python_version=list(sys.version_info[:3]),numpy_version=environments['fixed']['numpy_version'])
            write(output/'execution_manifest.json',original)
        if sha(output/'protocol.json')!=contract['reference_protocol_sha256'] or sha(output/'jobs.json')!=contract['reference_jobs_sha256']:
            raise ValueError('Frozen output protocol/jobs changed; refusing mixed resume.')
        (output/'results').mkdir(exist_ok=True)
        done=[];historical_differences=[]
        allowed={j['id'] for j in jobs}
        extra={p.stem for p in (output/'results').glob('*.json')}-allowed
        if extra:raise ValueError('Unexpected checkpoints: '+','.join(sorted(extra)))
        for job in jobs:
            path=output/'results'/(job['id']+'.json')
            if not path.exists():continue
            saved=read(path)
            if saved.get('checkpoint_sha256')!=checksum(saved):raise ValueError('Checkpoint hash mismatch: '+job['id'])
            minimalvalidate_result(job,protocol,saved)
            if saved.get('environment_lock_sha256')!=environment_sha:
                raise ValueError('Checkpoint was not computed in the current fixed environment: '+job['id'])
            if not equivalent(scientific_output(saved),scientific_output(read(reference/'results'/path.name))):
                historical_differences.append(job['id'])
            done.append(job['id'])
        prior=read(output/'state.json') if (output/'state.json').exists() else {}
        runtime_contract={'schema':2,'plan':runtime_plan,
            'environment_lock_sha256':environment_sha,
            'dispatcher_source_sha256':{name:sha(ROOT/'reproducibility'/name)
                for name in ('cohorts.py','runtimes.py','runtime_worker.py','fixed_environment.py','atomic_io.py')}}
        runtime_path=output/'runtime_contract.json'
        if runtime_path.exists() and read(runtime_path)!=runtime_contract:
            raise ValueError('Runtime provenance or dispatcher changed; refusing mixed resume.')
        write(runtime_path,runtime_contract)
        write(output/'runtime_environments.json',environments)
        print(f'{cohort.upper()}: validated {len(done)} saved checkpoints; all tasks use the fixed Python 3.12.4 environment',flush=True)
        started=time.perf_counter();prior_seconds=prior.get('active_seconds',0)
        state=dict(status='running',cohort=cohort,smoke=smoke,total=len(jobs),completed=len(done),
            full_matrix_total=len(all_jobs),workers=workers,created_at=prior.get('created_at',now()),updated_at=now(),
            historical_differences=len(historical_differences),environment_lock_sha256=environment_sha)
        def update(**values):
            state.update(values,updated_at=now(),active_seconds=prior_seconds+time.perf_counter()-started)
            write(output/'state.json',state)
        def commit(job,bundle):
            expected_environment=environments[runtime_plan['jobs'][job['id']]]
            if (bundle.get('reproduction_runtime')!=expected_environment or bundle.get('checkpoint_sha256')!=checksum(bundle)
                    or bundle.get('environment_lock_sha256')!=environment_sha):
                raise ValueError('Native worker runtime/checkpoint verification failed: '+job['id'])
            reference_bundle=read(reference/'results'/(job['id']+'.json'))
            if not equivalent(scientific_output(bundle),scientific_output(reference_bundle)):
                historical_differences.append(job['id'])
                write(output/'historical_comparisons'/(job['id']+'.json'),{
                    'job_id':job['id'],'comparison':'different under the unified environment',
                    'historical_row':reference_bundle['row'],'new_row':bundle['row'],
                    'reference_sha256':sha(reference/'results'/(job['id']+'.json'))})
            write(output/'results'/(job['id']+'.json'),bundle)
            done.append(job['id'])
            update(completed=len(done),current_job=None,historical_differences=len(historical_differences))
            if smoke or len(done)%25==0 or len(done)==len(jobs):
                print(f'{cohort.upper()}: {len(done)}/{len(jobs)}; validated; {len(historical_differences)} historical differences',flush=True)
        update()
        pending=[j for j in jobs if j['id'] not in set(done)]
        with RuntimePool(runtime_plan,environments,reference/'protocol.json',output/'runtime_logs') as runtime_pool:
            try:
                if workers==1:
                    for job in pending:
                        if stopped(output):update(status='paused');return 130
                        update(current_job=job['id'])
                        commit(job,runtime_pool.execute(job))
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        queue=iter(pending);futures={}
                        while True:
                            while len(futures)<workers and not stopped(output):
                                job=next(queue,None)
                                if job is None:break
                                futures[pool.submit(runtime_pool.execute,job)]=job
                            if not futures:break
                            finished,_=wait(futures,return_when=FIRST_COMPLETED)
                            for future in finished:commit(futures.pop(future),future.result())
                        if len(done)!=len(jobs):update(status='paused');return 130
                if not smoke:
                    update(status='summarizing')
                    if cohort=='e':from summarize_seven_category_run import summarize_run
                    else:from summarize_p6_run import summarize_run
                    summary=summarize_run(output)
                else:
                    summary={'scope':'Smoke only; no complete-cohort analysis or significance tests.', 'jobs':len(done)}
                write(output/'completion_receipt.json',{'status':'complete_smoke' if smoke else 'complete',
                    'completed_at':now(),'jobs':len(done),'all_scientific_outputs_match_reference':not historical_differences,
                    'historical_differences':len(historical_differences),'historical_difference_job_ids':historical_differences,
                    'all_jobs_recomputed_in_one_environment':True,'environment_lock_sha256':environment_sha,
                    'timing_excluded_from_equality':True,'float_absolute_tolerance':1e-10,
                    'float_relative_tolerance':1e-12,'historical_comparison_uses_exact_bits_and_ids':True,
                    'runtime_contract_sha256':sha(runtime_path),'summary':summary})
                update(status='complete_smoke' if smoke else 'complete',current_job=None)
                return 0
            except KeyboardInterrupt:
                update(status='paused',current_job=None);return 130
            except BaseException as exc:
                update(status='failed',error=repr(exc));raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=['e','p6'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    return run(args.cohort,args.output,args.reference,args.smoke,args.workers)

if __name__=='__main__':raise SystemExit(main())
