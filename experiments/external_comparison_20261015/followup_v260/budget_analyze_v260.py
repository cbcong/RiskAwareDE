"""Analysis amendment: retain one documented output-free failed execution attempt.

Original frozen planning controller remains unchanged; all raw receipts retained.
"""
from budget_sensitivity import HERE, ROOT, read, verify, np, defaultdict, Counter, dump, sha
import json
def analyze():
    from analyze_study import write_csv
    protocol=read(HERE/'budget_protocol.json');verify(protocol)
    all_receipts=[json.loads(x) for x in (HERE/'budget_process_log.jsonl').read_text().splitlines()]
    failed=[r for r in all_receipts if r['exit_code']!=0 or r['hard_timeout']]
    amendment=read(HERE/'budget_execution_amendment.json')
    assert not failed or (failed==[amendment['failed_attempt']] and amendment['output_existed'] is False)
    receipts=[r for r in all_receipts if r['exit_code']==0 and not r['hard_timeout']]
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
             'process_seconds_sum':sum(x['process_seconds'] for x in all_receipts),'maximum_rss_mib':max(x['peak_process_tree_rss_bytes'] for x in receipts)/1024**2,
             'failed_execution_attempts':len(failed),'retry_policy':amendment['decision'],'scope':protocol['analysis'],'reference':protocol['reference'],'methods':summaries,'families':family_summaries}
    dump(output/'summary.json',summary)
    print(json.dumps(summary,indent=2))



if __name__=='__main__':analyze()
