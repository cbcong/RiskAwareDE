"""Bounded, descriptive follow-up using unchanged external planners at 1/3 s.

prepare freezes the full queue before running. run is serial and resumable;
analyze fails closed until every planned result has a successful receipt.
The earlier six-second runs are a historical reference, not rerun here.
"""
from pathlib import Path
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timezone
import itertools
import json
import os
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import dump, sha
from run_study import read
import numpy as np


def prepare():
    path = HERE/'budget_protocol.json'
    if path.exists():
        raise SystemExit('Protocol already frozen; use run or analyze.')
    original = read(ROOT/'runs/confirmatory_v1/protocol.json')
    for relative, expected in original['source_sha256'].items():
        assert sha(ROOT/relative) == expected, relative
    selected = [j for j in original['jobs'] if j['method'] in ('cc_vpsto','saa') and j['seed'] in (101,102,103)]
    assert len(selected) == 144
    studies = {}
    for budget in (1,3):
        name = f'budget_v260_{budget}s'
        study = dict(original, study=name, external_budget_seconds=float(budget), jobs=selected,
                     expected_jobs=len(selected), stochastic_seeds=[101,102,103],
                     methods=['cc_vpsto','saa'],
                     stage='post hoc budget sensitivity on previously assessed maps',
                     evidence_stage='descriptive lower-budget follow-up; original frozen adapters',
                     statistics='Descriptive map means and family summaries; no new significance tests.',
                     budget=f'{budget} second stopping allowance including method initialization; final in-flight operation completes.',
                     resource_decision='24 existing maps x first3 fixed seeds x2 methods x2 new budgets =288 new optimizations.',
                     interpretation='Lower-budget follow-up with an archived six-second reference. No equal-time cross-planner ranking or fresh held-out evidence.')
        p = ROOT/'runs'/name/'protocol.json'
        assert not p.exists()
        dump(p,study)
        (p.parent/'results').mkdir()
        (p.parent/'logs').mkdir()
        studies[name] = sha(p)
    blocks = sorted({(j['map'],j['seed']) for j in selected})
    rng = random.Random(202610150902)
    rng.shuffle(blocks)
    orders = list(itertools.permutations([(b,m) for b in (1,3) for m in ('cc_vpsto','saa')]))
    rng.shuffle(orders)
    lookup = {(j['map'],j['seed'],j['method']):j for j in selected}
    queue = []
    for i,(map_name,seed) in enumerate(blocks):
        for budget,method in orders[i%len(orders)]:
            j=lookup[(map_name,seed,method)]
            queue.append({'study':f'budget_v260_{budget}s','budget_seconds':budget,'job':j})
    original_files = {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'runs/confirmatory_v1').rglob('*') if p.is_file()}
    protocol = {'created_utc':datetime.now(timezone.utc).isoformat(),
        'question':'How sensitive are the two implemented risk formulations to a shorter stopping allowance?',
        'planning_sigma':.6,'evaluation_sigma':.6,'samples_per_path':20000,
        'external_root':original['external_root'],
        'external_worlds':'Same fixed worlds as the archived six-second comparison; never used during optimization.',
        'selection':'All24 maps; seeds101,102,103 fixed before obtaining new-budget outputs. No map or seed selection by outcome.',
        'randomization':'72 map-seed blocks shuffled; all24 permutations of the4 budget-method combinations repeated3times.',
        'new_optimizations':len(queue),'stopping':'Fixed288 jobs; failures retained; pause/resume for machine availability only.',
        'analysis':'All24 maps weighted equally; length conditional on nominal feasibility with map/output counts; shared CFR includes nominal failures as zero. Report raw CFR and internal pass separately. No p-values or fresh confirmatory claims.',
        'reference':'Archived6s outputs restricted to same101-103seeds. Frozen main method on same72 cases is a separate820-evaluation reference. Historical vs new batch is disclosed.',
        'source_sha256':original['source_sha256'],'controller_sha256':sha(__file__),
        'source_study_sha256':sha(ROOT/'runs/confirmatory_v1/protocol.json'),
        'original_files_sha256':original_files,'studies_sha256':studies,'queue':queue}
    dump(path,protocol)
    print(json.dumps({'protocol_sha256':sha(path),'jobs':len(queue)}),flush=True)


def verify(protocol):
    assert sha(__file__) == protocol['controller_sha256']
    for relative, expected in protocol['source_sha256'].items():
        assert sha(ROOT/relative) == expected, relative
    for relative, expected in protocol['original_files_sha256'].items():
        assert sha(ROOT/relative) == expected, relative
    for study, expected in protocol['studies_sha256'].items():
        assert sha(ROOT/'runs'/study/'protocol.json') == expected, study


def run():
    import psutil
    protocol = read(HERE/'budget_protocol.json')
    verify(protocol)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1')
    receipts_path = HERE/'budget_process_log.jsonl'
    receipts = [json.loads(x) for x in receipts_path.read_text().splitlines()] if receipts_path.exists() else []
    successful = {(x['study'],x['id']) for x in receipts if x['exit_code']==0 and not x['hard_timeout']}
    for i,q in enumerate(protocol['queue']):
        study,j=q['study'],q['job']; directory=ROOT/'runs'/study
        output=directory/'results'/f'{j["id"]}.json'
        if output.exists():
            assert read(output)['protocol_sha256'] == protocol['studies_sha256'][study]
            assert (study,j['id']) in successful, 'Result without a successful process receipt'
            continue
        if (HERE/'BUDGET_STOP').exists() or psutil.virtual_memory().available<800*1024**2:
            dump(HERE/'budget_status.json',{'status':'paused','completed':len(successful),'next':q})
            return
        start=time.perf_counter(); peak=0; timedout=False
        with (directory/'logs'/f'{j["id"]}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'run_study.py'),'--worker',study,j['id']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            monitor=psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    rss=0
                    for item in [monitor]+monitor.children(recursive=True):
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
                dump(HERE/'budget_status.json',{'status':'running','ordinal':i+1,'total':len(protocol['queue']),'completed':len(successful),'study':study,'job':j['id']})
                time.sleep(.5)
            code=process.wait()
        receipt={'study':study,'id':j['id'],'exit_code':code,'hard_timeout':timedout,'process_seconds':time.perf_counter()-start,'peak_process_tree_rss_bytes':peak}
        with receipts_path.open('a',encoding='utf-8') as f:f.write(json.dumps(receipt)+'\n')
        if code:
            dump(HERE/'budget_status.json',{'status':'failed','receipt':receipt})
            raise SystemExit(f'Failed worker: {study}/{j["id"]}')
        successful.add((study,j['id']))
        if (i+1)%12==0:print(json.dumps({'completed':len(successful),'total':len(protocol['queue'])}),flush=True)
    verify(protocol)
    dump(HERE/'budget_status.json',{'status':'complete','completed':len(successful),'total':len(protocol['queue'])})


def analyze():
    from analyze_study import write_csv
    protocol=read(HERE/'budget_protocol.json');verify(protocol)
    receipts=[json.loads(x) for x in (HERE/'budget_process_log.jsonl').read_text().splitlines()]
    assert len(receipts)==len(protocol['queue'])==288
    assert all(x['exit_code']==0 and not x['hard_timeout'] for x in receipts)
    assert len({(x['study'],x['id']) for x in receipts})==288
    rows=[]
    sources=[(q['study'],q['job'],q['budget_seconds'],False) for q in protocol['queue']]
    old=read(ROOT/'runs/confirmatory_v1/protocol.json')
    sources += [('confirmatory_v1',j,6 if j['method']!='ours' else 0,True) for j in old['jobs'] if j['method'] in ('cc_vpsto','saa','ours') and j['seed'] in (101,102,103)]
    for study,j,budget,archived in sources:
        d=ROOT/'runs'/study; record=read(d/'results'/f'{j["id"]}.json')
        assert record['job']==j and record['protocol_sha256']==sha(d/'protocol.json')
        assert sha(ROOT/old['bank']/j['map_file'])==j['map_sha256']
        m=record['metrics']; bits_path=d/'results'/f'{j["id"]}_external_bits.npz'
        if m.get('external_samples',0):
            with np.load(bits_path) as bits:
                raw=np.unpackbits(bits['0.6'],bitorder='little')[:20000].mean()
                assert abs(raw-m['cfr_0.6'])<1e-12
        assert m['joint_cfr_0.6']==(m['cfr_0.6'] if m['nominal_success'] else 0.)
        rows.append({'method':j['method'],'budget_seconds':budget,'archived':archived,'map':j['map'],'family':j['family'],'seed':j['seed'],
            'nominal':bool(m['nominal_success']),'joint_cfr_pct':100*m['joint_cfr_0.6'],'raw_cfr_pct':100*m['cfr_0.6'],
            'internal_pass':bool(m['internal_constraints_satisfied']),'length':m['length'] if m['nominal_success'] else None,'seconds':record['planning_seconds']})
    groups=defaultdict(list)
    for r in rows:groups[(r['method'],r['budget_seconds'])].append(r)
    summaries=[];family_summaries=[];map_rows=[]
    def summarize(rs):
        maps=defaultdict(list)
        for r in rs:maps[r['map']].append(r)
        lengths=[np.mean([r['length'] for r in group if r['length'] is not None]) for group in maps.values() if any(r['length'] is not None for r in group)]
        return {'outputs':len(rs),'maps':len(maps),'nominal_count':sum(r['nominal'] for r in rs),
            'nominal_pct':100*np.mean([r['nominal'] for r in rs]),'internal_pass_count':sum(r['internal_pass'] for r in rs),
            'joint_cfr_pct':float(np.mean([np.mean([r['joint_cfr_pct'] for r in g]) for g in maps.values()])),
            'raw_cfr_pct':float(np.mean([np.mean([r['raw_cfr_pct'] for r in g]) for g in maps.values()])),
            'conditional_length':float(np.mean(lengths)) if lengths else None,'conditional_length_maps':len(lengths),
            'median_seconds':float(np.median([r['seconds'] for r in rs])),'p95_seconds':float(np.quantile([r['seconds'] for r in rs],.95))}
    for (method,budget),rs in sorted(groups.items()):
        assert len(rs)==72 and set(Counter(r['family'] for r in rs).values())=={24}
        base={'method':method,'budget_seconds':budget,'archived':rs[0]['archived']}
        summaries.append(dict(base,**summarize(rs)))
        for family in sorted({r['family'] for r in rs}):
            family_summaries.append(dict(base,family=family,**summarize([r for r in rs if r['family']==family])))
        for map_name in sorted({r['map'] for r in rs}):
            group=[r for r in rs if r['map']==map_name]
            assert len(group)==3
            map_rows.append(dict(base,map=map_name,family=group[0]['family'],**summarize(group)))
    output=HERE/'budget_analysis';output.mkdir(exist_ok=True)
    write_csv(output/'row_level.csv',rows);write_csv(output/'map_level.csv',map_rows)
    write_csv(output/'family_summary.csv',family_summaries);write_csv(output/'method_summary.csv',summaries)
    summary={'audit_passed':True,'new_optimizations':288,'archived_reference_outputs':216,'paired_maps':24,'seeds_per_map':3,
             'protocol_sha256':sha(HERE/'budget_protocol.json'),'all_frozen_sources_and_original_records_unchanged':True,
             'process_seconds_sum':sum(x['process_seconds'] for x in receipts),'maximum_rss_mib':max(x['peak_process_tree_rss_bytes'] for x in receipts)/1024**2,
             'scope':protocol['analysis'],'reference':protocol['reference'],'methods':summaries,'families':family_summaries}
    dump(output/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','analyze']);args=p.parse_args()
    globals()[args.action]()
