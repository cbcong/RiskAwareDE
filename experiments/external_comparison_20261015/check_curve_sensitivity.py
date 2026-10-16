"""Re-evaluate every saved CC cubic at tighter tolerance, without replanning."""
from pathlib import Path
import argparse
import json
import numpy as np
from common import HERE, dump, sha, ppoly_to_bezier, bezier_chords, score


def read(path):
    return Path(path).read_text(encoding='utf-8')


def main(study):
    directory=HERE/'runs'/study
    protocol=json.loads(read(directory/'protocol.json'))
    paths=[directory/'results'/(j['id']+'.json') for j in protocol['jobs']]
    if not all(p.exists() for p in paths):
        raise SystemExit('Wait for the complete frozen cohort before sensitivity analysis.')
    output=HERE/'analysis'/study
    rows=[]
    for job,path in zip(protocol['jobs'],paths):
        if job['method']!='cc_vpsto':continue
        record=json.loads(read(path))
        assert record['protocol_sha256']==sha(directory/'protocol.json')
        native=record['result']; original=record['metrics']
        if native.get('points') is None:continue
        scene=json.loads(read(HERE/protocol['bank']/job['map_file']))
        points,error=bezier_chords(ppoly_to_bezier(native['coefficients'],native['breaks']),.0005)
        new,_=score(scene,points,samples=protocol['external_samples'],
                    external_root=protocol['external_root'],sigmas=(.6,),curve_error=error,
                    native_within_bounds=native['native_bounds_extrema_checked'])
        coef=np.asarray(native['coefficients']);x,w=np.polynomial.legendre.leggauss(128)
        length=0.
        for i,h in enumerate(np.diff(native['breaks'])):
            t=(x+1)*h/2
            v=3*coef[0,i]*t[:,None]**2+2*coef[1,i]*t[:,None]+coef[2,i]
            length+=float(np.dot(w,np.linalg.norm(v,axis=1))*h/2)
        rows.append({'id':job['id'],'family':job['family'],'nominal_original':original['nominal_success'],
                     'nominal_tightened':new['nominal_success'],
                     'raw_cfr_original':original['cfr_0.6'],'raw_cfr_tightened':new['cfr_0.6'],
                     'joint_cfr_original':original['joint_cfr_0.6'],
                     'joint_cfr_tightened':new['joint_cfr_0.6'],
                     'raw_cfr_delta_pp':100*(new['cfr_0.6']-original['cfr_0.6']),
                     'joint_cfr_delta_pp':100*(new['joint_cfr_0.6']-original['joint_cfr_0.6']),
                     'original_32_vs_64_length_difference':native['arc_length_32_vs_64_difference'],
                     'length_64_vs_128_difference':abs(native['native_arc_length_quadrature']-length),
                     'tightened_chord_error':error,'tightened_chords':len(points)-1})
    summary={'study':study,'protocol_sha256':sha(directory/'protocol.json'),
             'type':'post hoc numerical sensitivity of saved outputs; no replanning or replacement of primary scores',
             'paths':len(rows),'original_tolerance':.002,'tightened_tolerance':.0005,
             'nominal_classification_changes':sum(r['nominal_original']!=r['nominal_tightened'] for r in rows),
             'mean_raw_cfr_delta_pp':float(np.mean([r['raw_cfr_delta_pp'] for r in rows])),
             'max_abs_raw_cfr_delta_pp':max(abs(r['raw_cfr_delta_pp']) for r in rows),
             'mean_joint_cfr_delta_pp':float(np.mean([r['joint_cfr_delta_pp'] for r in rows])),
             'max_abs_joint_cfr_delta_pp':max(abs(r['joint_cfr_delta_pp']) for r in rows),
             'max_length_32_vs_64_difference':max(r['original_32_vs_64_length_difference'] for r in rows),
             'max_length_64_vs_128_difference':max(r['length_64_vs_128_difference'] for r in rows),
             'rows':rows}
    dump(output/'curve_sensitivity.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',default='confirmatory_v1')
    main(parser.parse_args().study)
