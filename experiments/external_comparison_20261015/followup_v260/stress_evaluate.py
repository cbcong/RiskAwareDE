"""Paired uncertainty stress test of frozen paths; never invokes a planner.

Prepare the dated protocol before running; audit-only mode rechecks every
packed bit count and input hash.  Original experiment outputs are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))
from common import (bezier_chords, collision_free, collision_free_compiled,
                    dump, ppoly_to_bezier, score, seed_for, sha, wilson)
from analyze_study import audit_and_rows, write_csv

STUDY = EXPERIMENT / 'runs' / 'confirmatory_v1'
PROTOCOL = HERE / 'stress_protocol.json'
OUT = HERE / 'stress_results'
METHODS = ['ours', 'cc_vpsto', 'saa', 'astar']
LABELS = {'ours': 'RiskAwareDE', 'cc_vpsto': 'CC-VPSTO-3D',
          'saa': 'SAA-AVaR-3D', 'astar': 'Grid A*'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def synthetic_checks():
    """Check sigma-zero geometry, shared random worlds, and compiled parity."""
    scene = {'name': 'synthetic_stress_check',
             'start': [0., 0., 0.], 'goal': [4., 4., 4.],
             'bounds': {'minimum': [0., 0., 0.], 'maximum': [4., 4., 4.]},
             'obstacles': [{'minimum': [1., 1., 1.], 'maximum': [2., 2., 2.]}]}
    hit = np.array([[0., 0., 0.], [4., 4., 4.]])
    free = np.array([[0., 0., 0.], [0., 0., 4.], [0., 4., 4.], [4., 4., 4.]])
    m0, b0 = score(scene, hit, samples=101, external_root=7, sigmas=(0.,))
    m1, b1 = score(scene, free, samples=101, external_root=7, sigmas=(0.,))
    require(not m0['nominal_success'] and m0['cfr_0.0'] == 0., 'Sigma-zero collision failed')
    require(m1['nominal_success'] and m1['cfr_0.0'] == 1., 'Sigma-zero free path failed')
    together, bits = score(scene, free, samples=101, external_root=7, sigmas=(.6, .9))
    standard = np.random.default_rng(seed_for('external-evaluation-v1', scene['name'], 7)).standard_normal((101, 1, 3))
    boxes = np.array([[[1., 1., 1.], [2., 2., 2.]]])
    for sigma in (.6, .9):
        separate, isolated = score(scene, free, samples=101, external_root=7, sigmas=(sigma,))
        expected = collision_free(free, boxes, standard*sigma)
        compiled = collision_free_compiled(free, boxes, standard*sigma)
        require(np.array_equal(expected, compiled), 'Compiled collision predicate differs')
        require(np.array_equal(bits[str(sigma)], isolated[str(sigma)]), 'Sigma stream changed')
        require(np.array_equal(np.unpackbits(bits[str(sigma)], bitorder='little')[:101], expected), 'Shared normal bank failed')
        require(together[f'cfr_{sigma}'] == separate[f'cfr_{sigma}'], 'Paired score mismatch')
    return {'passed': True, 'checks': ['sigma_zero_hit', 'sigma_zero_free',
             'paired_standard_normal_bank', 'standalone_sigma_identity',
             'compiled_vs_vectorized_closed_AABB', 'non_byte_aligned_packing']}


def prepare():
    require(not PROTOCOL.exists(), 'A protocol already exists; never overwrite a completed or running protocol.')
    require(not OUT.exists(), 'Results already exist before protocol; refuse retrospective overwrite.')
    checks = synthetic_checks()
    original, _, audit = audit_and_rows(STUDY)
    require(original['expected_jobs'] == 384 and original['maps'] == 24, 'Unexpected original cohort')
    dump(HERE / 'stress_input_audit.json', audit)
    protocol = {
        'created_at_utc': now(),
        'stage': 'Post hoc follow-up planned after the primary sigma=.6 outcomes; before follow-up scoring.',
        'question': 'How does empirical collision-free reliability of the same fixed paths change when obstacle translation sigma increases from .6 to .9?',
        'design': 'Saved-path stress evaluation; zero new planner runs and zero method retuning.',
        'original_study': 'runs/confirmatory_v1',
        'original_protocol_sha256': sha(STUDY / 'protocol.json'),
        'input_audit_sha256': sha(HERE / 'stress_input_audit.json'),
        'code_sha256': sha(__file__),
        'expected_paths': 384, 'maps': 24, 'maps_per_family': 8,
        'methods': METHODS, 'seeds_per_risk_planner_per_map': 5,
        'astar_paths_per_map': 1,
        'external_samples': 20000,
        'external_root': 202610150901,
        'sigmas': [.6, .9],
        'randomization': 'The same standard-normal array is shared by all paths of a map and both sigma values; shifts = sigma * Z. The new root is independent of the original evaluation bank.',
        'curve_tolerance': .002,
        'geometry': 'Original continuous-polyline closed-box predicate; native CC cubic is converted to conservative chords with maximum coordinate error .002. Exact native extrema determine workspace bounds.',
        'failure_policy': 'Every original job remains in the denominator. Nominally infeasible or absent paths have joint CFR zero at both scales. Raw incumbent CFR is retained separately.',
        'aggregation': 'Average seeds within each map first; average 24 map means for overall results and 8 map means for each family. No seed-level pseudoreplication.',
        'primary_descriptions': 'Per-method joint CFR at .6 and .9, signed change (.9 minus .6) in percentage points, nominal-and-CFR>=.825 counts, and family-specific mean CFR.',
        'uncertainty_description': 'Within-path paired Monte Carlo standard error; descriptive 95% percentile bootstrap CI for each method mean stress change using 20000 map resamples stratified by family. No hypothesis tests or superiority claims.',
        'bootstrap_replicates': 20000, 'bootstrap_root': 202610150902,
        'interpretation': 'A distribution-shift stress test at fixed planning assumptions sigma=.6, not a re-optimized sigma=.9 benchmark or an independent set of 384 planner runs. CFR need not decrease monotonically for every path.',
        'stopping': 'Evaluate every one of the 384 prespecified saved paths at both sigma values. Resume allowed only with matching protocol and source hashes. No outcome-based exclusions.',
        'synthetic_checks': checks,
    }
    dump(PROTOCOL, protocol)
    print(json.dumps({'prepared': str(PROTOCOL), 'protocol_sha256': sha(PROTOCOL), 'paths': 384}, indent=2), flush=True)


def load_verified_protocol():
    protocol = read(PROTOCOL)
    require(sha(__file__) == protocol['code_sha256'], 'Follow-up code changed after protocol freeze')
    require(sha(HERE / 'stress_input_audit.json') == protocol['input_audit_sha256'], 'Input audit changed')
    require(sha(STUDY / 'protocol.json') == protocol['original_protocol_sha256'], 'Original protocol changed')
    original, _, audit = audit_and_rows(STUDY)
    baseline = read(HERE / 'stress_input_audit.json')
    for key in ['source_sha256', 'map_manifest_sha256', 'result_file_sha256', 'bit_file_sha256',
                'protocol_sha256', 'frozen_environment_file_sha256']:
        require(audit[key] == baseline[key], f'Original immutable inputs changed: {key}')
    return protocol, original, audit


def native_bounds(result, scene):
    """Exact extrema of each coordinate of every cubic segment."""
    coef = np.asarray(result['coefficients'], float)
    bound = np.array([scene['bounds']['minimum'], scene['bounds']['maximum']], float)
    valid = True
    for i, h in enumerate(np.diff(result['breaks'])):
        for axis in range(3):
            c = coef[:, i, axis]
            roots = np.roots([3*c[0], 2*c[1], c[2]])
            interior = [float(t.real) for t in roots if abs(t.imag) <= 1e-12 and 0. < t.real < h]
            values = np.polyval(c, np.array([0., h] + interior))
            valid &= bool(values.min() >= bound[0, axis]-1e-9 and values.max() <= bound[1, axis]+1e-9)
    return bool(valid)


def score_one(job, protocol, original):
    record = read(STUDY / 'results' / (job['id']+'.json'))
    native = record['result']
    scene = read(EXPERIMENT / original['bank'] / job['map_file'])
    error = 0.
    bounds_valid = None
    points = native.get('points')
    if points is None:
        n = protocol['external_samples']
        bits = {str(s): np.zeros((n+7)//8, np.uint8) for s in protocol['sigmas']}
        metrics = {'nominal_success': False, 'external_samples': n, 'path_absent': True}
        for s in protocol['sigmas']:
            metrics.update({f'cfr_{s}': 0., f'joint_cfr_{s}': 0., f'cfr_ci_{s}': wilson(0,n)})
    else:
        if job['method'] == 'cc_vpsto':
            points, error = bezier_chords(ppoly_to_bezier(native['coefficients'], native['breaks']), protocol['curve_tolerance'])
            bounds_valid = native_bounds(native, scene)
            require(bounds_valid == native['native_bounds_extrema_checked'], 'Native bounds status differs')
            require(abs(error-record['metrics']['curve_chord_error_bound']) <= 1e-14, 'CC curve tolerance changed')
        metrics, bits = score(scene, points, samples=protocol['external_samples'],
                              external_root=protocol['external_root'], sigmas=tuple(protocol['sigmas']),
                              curve_error=error, native_within_bounds=bounds_valid)
    require(metrics['nominal_success'] == record['metrics']['nominal_success'], 'Nominal classification changed')
    decoded = {k: np.unpackbits(v, bitorder='little')[:protocol['external_samples']].astype(float) for k,v in bits.items()}
    changes = decoded['0.9']-decoded['0.6'] if metrics['nominal_success'] else np.zeros(protocol['external_samples'])
    metrics['joint_change_pp'] = 100*float(changes.mean())
    metrics['paired_mc_standard_error_pp'] = 100*float(changes.std(ddof=1)/math.sqrt(len(changes)))
    metrics['original_primary_joint_cfr_0.6'] = record['metrics']['joint_cfr_0.6']
    metrics['new_minus_original_sigma_0.6_pp'] = 100*(metrics['joint_cfr_0.6']-record['metrics']['joint_cfr_0.6'])
    bitpath = OUT / 'paths' / (job['id']+'_bits.npz')
    bitpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bitpath, **bits)
    dump(OUT / 'paths' / (job['id']+'.json'), {
        'job': job, 'followup_protocol_sha256': sha(PROTOCOL),
        'original_result_sha256': sha(STUDY / 'results' / (job['id']+'.json')),
        'evaluated_at_utc': now(), 'metrics': metrics, 'bits_sha256': sha(bitpath)})


def run():
    protocol, original, _ = load_verified_protocol()
    OUT.mkdir(exist_ok=True)
    started = time.perf_counter()
    for i, job in enumerate(original['jobs']):
        output = OUT / 'paths' / (job['id']+'.json')
        if output.exists():
            old = read(output)
            require(old['followup_protocol_sha256'] == sha(PROTOCOL), 'Resume protocol differs')
            continue
        memory = psutil.virtual_memory().available
        require(memory >= 600*1024**2, 'Less than 600 MiB available; resume later without altering results.')
        score_one(job, protocol, original)
        if (i+1) % 24 == 0 or i+1 == len(original['jobs']):
            print(json.dumps({'evaluated': i+1, 'total': len(original['jobs']),
                              'elapsed_seconds': round(time.perf_counter()-started, 2),
                              'available_mib': round(memory/1024**2)}), flush=True)
    verify_and_summarize()


def stratified_bootstrap_change(mapped, method, protocol):
    rng = np.random.default_rng(seed_for('stress-map-bootstrap-v1', method, protocol['bootstrap_root']))
    sums = np.zeros(protocol['bootstrap_replicates'])
    for family in sorted({m['family'] for m in mapped}):
        data = np.array([m['joint_change_pp'] for m in mapped if m['family'] == family])
        draw = rng.integers(0, len(data), size=(len(sums), len(data)))
        sums += data[draw].sum(axis=1)
    return np.quantile(sums/len(mapped), [.025, .975]).tolist()


def verify_and_summarize():
    protocol, original, input_audit = load_verified_protocol()
    expected = {j['id'] for j in original['jobs']}
    require({p.stem for p in (OUT/'paths').glob('*.json')} == expected, 'Follow-up job IDs incomplete or extra')
    rows, hashes, bit_hashes = [], {}, {}
    for job in original['jobs']:
        path = OUT / 'paths' / (job['id']+'.json')
        record = read(path)
        require(record['job'] == job and record['followup_protocol_sha256'] == sha(PROTOCOL), 'Follow-up identity mismatch')
        require(record['original_result_sha256'] == input_audit['result_file_sha256'][job['id']+'.json'], 'Saved path changed')
        bitpath = OUT / 'paths' / (job['id']+'_bits.npz')
        require(sha(bitpath) == record['bits_sha256'], 'Follow-up bit file changed')
        metric = record['metrics']
        n = protocol['external_samples']
        row = {'id': job['id'], 'method': job['method'], 'map': job['map'], 'family': job['family'],
               'seed': job['seed'], 'nominal_success': metric['nominal_success']}
        decoded = {}
        with np.load(bitpath, allow_pickle=False) as packed:
            require(set(packed.files) == {'0.6','0.9'}, 'Unexpected sigma in packed file')
            for sigma in protocol['sigmas']:
                key = str(sigma)
                blob = packed[key]
                require(blob.dtype == np.uint8 and blob.shape == ((n+7)//8,), 'Invalid packed bit structure')
                allbits = np.unpackbits(blob, bitorder='little')
                require(not allbits[n:].any(), 'Nonzero padding')
                decoded[key] = allbits[:n].astype(float)
                raw = float(decoded[key].mean())
                require(abs(raw-metric[f'cfr_{key}']) <= 1e-14, 'CFR differs from bits')
                joint = raw if metric['nominal_success'] else 0.
                require(joint == metric[f'joint_cfr_{key}'], 'Nominal gate inconsistent')
                require(np.allclose(metric[f'cfr_ci_{key}'], wilson(int(decoded[key].sum()),n), rtol=0, atol=1e-12), 'Wilson interval mismatch')
                row[f'raw_cfr_{key}'] = raw
                row[f'joint_cfr_{key}'] = joint
                row[f'nominal_and_target_pass_{key}'] = bool(metric['nominal_success'] and raw >= .825)
        difference = decoded['0.9']-decoded['0.6'] if metric['nominal_success'] else np.zeros(n)
        change = 100*float(difference.mean())
        se = 100*float(difference.std(ddof=1)/math.sqrt(n))
        require(abs(change-metric['joint_change_pp']) <= 1e-12, 'Paired change mismatch')
        require(abs(se-metric['paired_mc_standard_error_pp']) <= 1e-12, 'Paired MC standard error mismatch')
        row.update({'joint_change_pp': change, 'paired_mc_standard_error_pp': se,
                    'new_minus_original_sigma_0.6_pp': metric['new_minus_original_sigma_0.6_pp']})
        rows.append(row)
        hashes[path.name], bit_hashes[bitpath.name] = sha(path), sha(bitpath)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'],row['map'])].append(row)
    maps = []
    for (method, name), group in sorted(groups.items()):
        expected_count = 1 if method == 'astar' else 5
        require(len(group) == expected_count and len({r['seed'] for r in group}) == expected_count, 'Wrong per-map seed counts')
        maps.append({'method': method, 'map': name, 'family': group[0]['family'], 'paths': len(group),
                     **{key: float(np.mean([r[key] for r in group])) for key in
                        ['joint_cfr_0.6','joint_cfr_0.9','raw_cfr_0.6','raw_cfr_0.9','joint_change_pp','new_minus_original_sigma_0.6_pp']}})
    summary_rows = []
    for family in ['all']+sorted({r['family'] for r in rows}):
        for method in METHODS:
            selected = [r for r in rows if r['method'] == method and (family == 'all' or r['family'] == family)]
            mapped = [r for r in maps if r['method'] == method and (family == 'all' or r['family'] == family)]
            summary_rows.append({'family': family, 'method': method, 'method_label': LABELS[method],
                'maps': len(mapped), 'paths': len(selected), 'nominal_feasible_paths': sum(r['nominal_success'] for r in selected),
                'joint_cfr_0.6_percent': 100*float(np.mean([r['joint_cfr_0.6'] for r in mapped])),
                'joint_cfr_0.9_percent': 100*float(np.mean([r['joint_cfr_0.9'] for r in mapped])),
                'raw_cfr_0.6_percent': 100*float(np.mean([r['raw_cfr_0.6'] for r in mapped])),
                'raw_cfr_0.9_percent': 100*float(np.mean([r['raw_cfr_0.9'] for r in mapped])),
                'joint_change_pp': float(np.mean([r['joint_change_pp'] for r in mapped])),
                'joint_change_descriptive_bootstrap_ci95_pp': stratified_bootstrap_change(mapped,method,protocol),
                'nominal_and_target_pass_0.6': sum(r['nominal_and_target_pass_0.6'] for r in selected),
                'nominal_and_target_pass_0.9': sum(r['nominal_and_target_pass_0.9'] for r in selected),
                'mean_new_minus_original_sigma_0.6_pp': float(np.mean([r['new_minus_original_sigma_0.6_pp'] for r in mapped])),
                'max_abs_path_new_minus_original_sigma_0.6_pp': max(abs(r['new_minus_original_sigma_0.6_pp']) for r in selected)})
    summary = {'completed_at_utc': now(), 'protocol_sha256': sha(PROTOCOL),
               'evidence_stage': protocol['stage'], 'design': protocol['design'],
               'paths': len(rows), 'maps': 24, 'sigma_evaluations': 2*len(rows),
               'path_world_tests': 2*len(rows)*protocol['external_samples'],
               'new_planner_runs': 0, 'summary': summary_rows,
               'interpretation': protocol['interpretation']}
    dump(OUT/'summary.json', summary)
    write_csv(OUT/'path_level.csv', sorted(rows,key=lambda r:r['id']))
    write_csv(OUT/'map_level.csv', maps)
    write_csv(OUT/'method_family_summary.csv', summary_rows)
    dump(OUT/'audit.json', {'passed': True, 'verified_at_utc': now(), 'followup_protocol_sha256': sha(PROTOCOL),
         'original_input_audit_sha256': sha(HERE/'stress_input_audit.json'), 'frozen_source_files_unchanged': len(input_audit['source_sha256']),
         'original_result_files_unchanged': len(input_audit['result_file_sha256']),
         'original_bit_files_unchanged': len(input_audit['bit_file_sha256']),
         'followup_paths': len(rows), 'packed_collision_arrays_audited': 2*len(rows),
         'no_exclusions': True, 'path_record_sha256': hashes, 'path_bits_sha256': bit_hashes})
    print(json.dumps({'audit_passed': True, 'paths': len(rows), 'summary': summary_rows[:4]}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare', action='store_true')
    operation.add_argument('--run', action='store_true')
    operation.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        verify_and_summarize()
