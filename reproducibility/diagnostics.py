"""Run P0-P5 diagnostics using the current frozen E cohort and one shared engine.

Completed stages resume without repeating work. P3 resumes individual jobs;
other short stages retain interrupted attempts and restart only that stage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from .cohorts import ROOT,read,write,sha,now,verify_sources,stopped,protect_output

STAGES=[
 ('p0','p0_estimator.py','--source'),
 ('p1_replay','p1_replay.py','--source'),
 ('p3','p3_sensitivity.py','--source-run'),
 ('p1_p2_p4_p5','saved_diagnostics.py','--run-dir'),
]

def verify_stage(directory,runner_sha256,smoke):
    directory=Path(directory).resolve()
    marker=directory/'stage_complete.json'
    if not marker.is_file():raise ValueError('Completed stage marker missing: '+str(marker))
    completed=read(marker)
    if completed.get('runner_sha256')!=runner_sha256 or completed.get('smoke')!=smoke:
        raise ValueError('Completed stage uses a different adapter or smoke mode: '+str(directory))
    hashes=completed.get('artifacts_sha256')
    if not hashes:raise ValueError('Completed stage contains no recorded artifact hashes')
    for relative,expected in hashes.items():
        path=(directory/relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha(path)!=expected:
            raise ValueError('Completed stage artifact changed or missing: '+str(path))
    return dict(output=str(directory),completed_at=now(),
        artifacts_sha256={**hashes,'stage_complete.json':sha(marker)})

def run(reference,output,smoke=False):
    reference=Path(reference)
    if str(reference)=='e':reference=ROOT/'data/reference/e'
    output,reference=protect_output(output,reference)
    from .fixed_environment import identity
    environment_sha=identity()
    scientific=verify_sources(reference)
    if read(reference/'reproduction.json').get('environment_lock_sha256')!=environment_sha:
        raise ValueError('Diagnostics require current run fixed-environment E results')
    if read(reference/'state.json').get('status')!=('complete_smoke' if smoke else 'complete'):raise ValueError('Diagnostics require the complete E reference or a completed fresh E run')
    scripts={name:sha(ROOT/'reproducibility/internal'/script) for name,script,_ in STAGES}
    contract={'schema':'RiskAwareDE-P0-P5-reproduction-v270','smoke':smoke,
        'reference':str(reference),'environment_lock_sha256':environment_sha,'protocol_sha256':sha(reference/'protocol.json'),
        'scientific_source_sha256':scientific,'adapter_sha256':scripts,
        'scope':'Existing E maps and locked settings; diagnostics are not a new blind validation'}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'reproduction.json').exists():
        if read(output/'reproduction.json')!=contract:raise ValueError('P output already has different inputs or mode')
    else:
        if any(output.iterdir()):raise ValueError('Nonempty P output has no matching contract')
        write(output/'reproduction.json',contract)
    status=read(output/'state.json') if (output/'state.json').exists() else {'status':'running','completed_stages':{},'attempts':{}}
    environment=os.environ.copy();environment.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        for name,script,source_flag in STAGES:
            if name in status['completed_stages']:
                record=status['completed_stages'][name]
                directory=Path(record['output'])
                if not directory.is_dir():raise ValueError('Completed stage output missing: '+name)
                verify_stage(directory,scripts[name],smoke)
                for relative,expected in record.get('artifacts_sha256',{}).items():
                    path=directory/relative
                    if not path.is_file() or sha(path)!=expected:
                        raise ValueError('Completed stage artifact changed: '+str(path))
                print('P diagnostics: already complete '+name,flush=True);continue
            if stopped(output):status.update(status='paused',updated_at=now());write(output/'state.json',status);return 130
            base=output/name
            attempts=status['attempts'].setdefault(name,[])
            if attempts and (Path(attempts[-1])/'stage_complete.json').exists():
                record=verify_stage(Path(attempts[-1]),scripts[name],smoke)
                record['recovered_after_parent_interruption']=True
                status['completed_stages'][name]=record
                write(output/'state.json',status)
                print('P diagnostics: recovered completed '+name,flush=True)
                continue
            resumable=bool(name=='p3' and attempts and (Path(attempts[-1])/'protocol.json').exists())
            if resumable:destination=Path(attempts[-1])
            else:
                destination=base/f'attempt_{len(attempts)+1:02d}'
                attempts.append(str(destination))
            cmd=[sys.executable,str(ROOT/'reproducibility/internal'/script),source_flag,str(reference),'--output',str(destination)]
            if resumable:cmd.append('--resume')
            if smoke:cmd.append('--smoke')
            status.update(status='running',stage=name,updated_at=now());write(output/'state.json',status)
            print('P diagnostics: running '+name,flush=True)
            log=output/(name+f'_attempt{len(attempts):02d}.log')
            with log.open('w',encoding='utf-8') as stream:
                completed=subprocess.run(cmd,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f'{name} exited {completed.returncode}; retained log {log}')
            status['completed_stages'][name]=verify_stage(destination,scripts[name],smoke)
            status['completed_stages'][name]['log']=str(log)
            write(output/'state.json',status)
            print('P diagnostics: complete '+name,flush=True)
        status.update(status='complete_smoke' if smoke else 'complete',stage=None,updated_at=now())
        write(output/'state.json',status)
        write(output/'completion_receipt.json',{'status':status['status'],'stages':status['completed_stages'],
            'parts':['P0','P1','P2','P3','P4','P5'],'smoke':smoke,
            'resume':'Completed stages cached; P3 per-job resume; interrupted P0/P1/saved-analysis attempts retained and restarted.'})
        return 0
    except KeyboardInterrupt:
        status.update(status='paused',updated_at=now());write(output/'state.json',status);return 130
    except BaseException as exc:
        status.update(status='failed',error=repr(exc),updated_at=now());write(output/'state.json',status);raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('e'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    return run(args.reference,args.output,args.smoke)

if __name__=='__main__':raise SystemExit(main())
