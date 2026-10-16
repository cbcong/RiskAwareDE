"""Release complete-study aggregates only after audit, per analysis_plan.md.

Default: runs/confirmatory_v1 -> analysis/confirmatory_v1.
An incomplete queue fails before any result JSON is read. --self-test uses
only synthetic inputs and never opens a study result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
METHODS = ['ours', 'cc_vpsto', 'saa', 'astar']
LABELS = {
    'ours': 'RiskAwareDE (frozen)',
    'cc_vpsto': 'CC-VPSTO-3D adaptation',
    'saa': 'SAA-AVaR-3D (A*-initialized adaptation)',
    'astar': 'Grid A* (descriptive reference)',
}
ANALYSIS_SEED = 202610150704
BOOTSTRAP_REPLICATES = 20000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8-sig')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def exact_signed_rank(differences):
    """Two-sided conditional random-sign distribution, retaining average ties.

    Zero means exactly zero in the saved floating-point map difference; no
    data-dependent near-zero/tie tolerance is introduced. Ranks are doubled
    to integers and the full 2**n distribution is computed by dynamic programming.
    """
    d = np.asarray(differences, float)
    if not np.isfinite(d).all():
        raise ValueError('Finite paired differences are required')
    nonzero = d[d != 0]
    n = len(nonzero)
    if not n:
        return {'p_value': 1.0, 'n_nonzero': 0, 'n_zero': len(d),
                'positive_rank_sum': 0.0, 'negative_rank_sum': 0.0,
                'statistic': 0.0, 'sign_patterns': 1, 'ties_present': False}
    if n > 60:
        raise ValueError('int64 sign-count arithmetic is bounded to n<=60')
    ranks = rankdata(np.abs(nonzero), method='average')
    doubled = np.rint(2*ranks).astype(int)
    total = int(doubled.sum())
    ways = np.zeros(total+1, dtype=np.int64)
    ways[0] = 1
    reached = 0
    for r in doubled:
        previous = ways[:reached+1].copy()
        ways[r:r+reached+1] += previous
        reached += int(r)
    assert int(ways.sum()) == 2**n
    observed = int(doubled[nonzero > 0].sum())
    extreme = np.abs(2*np.arange(total+1)-total) >= abs(2*observed-total)
    p_value = int(ways[extreme].sum()) / 2**n
    return {'p_value': float(p_value), 'n_nonzero': n, 'n_zero': len(d)-n,
            'positive_rank_sum': observed/2, 'negative_rank_sum': (total-observed)/2,
            'statistic': min(observed, total-observed)/2, 'sign_patterns': 2**n,
            'ties_present': len(np.unique(np.abs(nonzero))) < n}


def holm_adjust(p_values):
    p = np.asarray(p_values, float)
    if np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError('P-values must be finite probabilities')
    order = np.argsort(p, kind='stable')
    sorted_adjusted = np.minimum(1.0, np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    out = np.empty_like(p)
    out[order] = sorted_adjusted
    return out.tolist()


def bootstrap_paired(rows, rng, scale=1.0, replicates=BOOTSTRAP_REPLICATES):
    """Resample whole paired map means separately within each available family.

    Each family retains its original available-map count; absent families
    contribute no artificial observations. Thus original available-map weights
    are retained, including for conditional length contrasts.
    """
    if not rows:
        return {'difference_ci95': None, 'percentage_change_ci95': None,
                'family_map_counts': {}}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['family']].append(row)
    sums_a = np.zeros(replicates)
    sums_b = np.zeros(replicates)
    family_counts = {}
    for family in sorted(grouped):
        group = sorted(grouped[family], key=lambda row: row['map'])
        a = np.asarray([row['ours'] for row in group], float)
        b = np.asarray([row['comparator'] for row in group], float)
        draw = rng.integers(0, len(group), size=(replicates, len(group)))
        sums_a += a[draw].sum(axis=1)
        sums_b += b[draw].sum(axis=1)
        family_counts[family] = len(group)
    differences = scale*(sums_a-sums_b)/len(rows)
    percentage = np.divide(sums_a, sums_b, out=np.full(replicates, np.nan), where=sums_b != 0)
    percentage = 100*(percentage-1)
    return {'difference_ci95': np.quantile(differences, [.025, .975]).tolist(),
            'percentage_change_ci95': np.quantile(percentage, [.025,.975]).tolist()
            if np.isfinite(percentage).all() else None,
            'family_map_counts': family_counts}


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    den = 1+z*z/n
    mid = (p+z*z/(2*n))/den
    rad = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-rad, mid+rad]


def audit_and_rows(study):
    protocol_path = study/'protocol.json'
    protocol = read(protocol_path)
    expected = {job['id']: job for job in protocol['jobs']}
    if len(expected) != len(protocol['jobs']) or len(expected) != protocol['expected_jobs']:
        raise RuntimeError('Protocol has duplicate IDs or inconsistent expected job count')
    actual = {p.stem: p for p in (study/'results').glob('*.json')}
    missing = sorted(set(expected)-set(actual))
    extras = sorted(set(actual)-set(expected))
    # This check deliberately precedes every result read or method aggregate.
    if missing or extras:
        raise RuntimeError(f'Complete-study analysis refused: {len(actual)}/{len(expected)} files; '
                           f'{len(missing)} missing, {len(extras)} unexpected. No effects calculated.')
    receipts = defaultdict(list)
    receipt_path = study/'process_log.jsonl'
    for line in receipt_path.read_text(encoding='utf-8-sig').splitlines():
        receipt = json.loads(line)
        receipts[receipt['id']].append(receipt)
    if set(receipts) != set(expected):
        raise RuntimeError('Process receipts do not cover exactly the frozen jobs')
    failed_attempts = []
    for job_id, attempts in receipts.items():
        successes = [r for r in attempts if r['exit_code'] == 0 and not r['hard_timeout']]
        if len(successes) != 1 or attempts[-1] != successes[0]:
            raise RuntimeError(f'Expected one successful final attempt, without selective successful reruns: {job_id}')
        failed_attempts.extend([r for r in attempts if r not in successes])
        if 'peak_process_tree_rss_bytes' not in successes[0]:
            raise RuntimeError(f'Only launcher RSS, not process-tree RSS recorded: {job_id}')
    protocol_hash = sha(protocol_path)
    source_actual = {}
    for name, expected_hash in protocol['source_sha256'].items():
        current = sha(HERE/name)
        if current != expected_hash:
            raise RuntimeError(f'Frozen implementation changed: {name}')
        source_actual[name] = current
    bank = HERE/protocol['bank']
    if sha(bank/'manifest.json') != protocol['map_manifest_sha256']:
        raise RuntimeError('Map manifest changed')
    manifest = read(bank/'manifest.json')
    map_items = {item['name']: item for item in manifest['maps']}
    for item in map_items.values():
        if sha(bank/item['file']) != item['sha256']:
            raise RuntimeError(f'Map hash mismatch: {item["name"]}')
    if len(map_items) != protocol['maps']:
        raise RuntimeError('Unexpected number of maps')
    family_maps = Counter(item['family'] for item in map_items.values())
    if len(family_maps) != protocol['families'] or set(family_maps.values()) != {protocol['maps_per_family']}:
        raise RuntimeError('Family map counts do not match the frozen design')
    counts = defaultdict(list)
    for job in expected.values():
        counts[(job['map'], job['method'])].append(job['seed'])
        item = map_items[job['map']]
        if job['map_sha256'] != item['sha256'] or job['map_file'] != item['file'] or job['family'] != item['family']:
            raise RuntimeError('Job/map manifest mismatch')
    for name in map_items:
        for method in METHODS:
            seeds = counts[(name, method)]
            want = 1 if method == 'astar' else len(protocol['stochastic_seeds'])
            if len(seeds) != want or len(set(seeds)) != want:
                raise RuntimeError(f'Wrong seed count: {name}/{method}')
            if method != 'astar' and set(seeds) != set(protocol['stochastic_seeds']):
                raise RuntimeError(f'Wrong stochastic seeds: {name}/{method}')
    rows = []
    packed_audits = 0
    absent_paths = []
    result_hashes = {}
    bit_hashes = {}
    for job_id in sorted(expected):
        job = expected[job_id]
        record = read(actual[job_id])
        if record['job'] != job or record['protocol_sha256'] != protocol_hash:
            raise RuntimeError(f'Wrong job/protocol identity: {job_id}')
        if record['evidence_stage'] != protocol['evidence_stage']:
            raise RuntimeError(f'Wrong evidence stage: {job_id}')
        metrics = record['metrics']
        result = record['result']
        points = result.get('points')
        absent = points is None
        bitpath = study/'results'/f'{job_id}_external_bits.npz'
        packed = None
        if absent:
            absent_paths.append(job_id)
            if metrics['nominal_success'] or metrics['external_samples'] != 0 or metrics.get('length') is not None:
                raise RuntimeError(f'No-path failure fields inconsistent: {job_id}')
            if bitpath.exists():
                raise RuntimeError(f'Unexpected stale collision bits for absent path: {job_id}')
        else:
            p = np.asarray(points, float)
            if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
                raise RuntimeError(f'Invalid finite path: {job_id}')
            if metrics['external_samples'] != protocol['external_samples']:
                raise RuntimeError(f'External sample count changed: {job_id}')
            expected_seed = int.from_bytes(hashlib.sha256(
                f'external-evaluation-v1|{protocol["external_root"]}|{job["map"]}'.encode()).digest()[:8], 'little')
            if metrics['external_seed'] != expected_seed:
                raise RuntimeError(f'External random-stream mismatch: {job_id}')
            if not bitpath.exists():
                raise RuntimeError(f'Missing collision bits: {job_id}')
            packed = np.load(bitpath, allow_pickle=False)
            if set(packed.files) != {str(sigma) for sigma in protocol['sigmas']}:
                raise RuntimeError(f'Collision-bit sigma mismatch: {job_id}')
            bit_hashes[bitpath.name] = sha(bitpath)
        for sigma in protocol['sigmas']:
            key = str(sigma)
            raw = float(metrics[f'cfr_{key}'])
            joint = float(metrics[f'joint_cfr_{key}'])
            returned = float(metrics[f'returned_success_cfr_{key}'])
            if not 0 <= raw <= 1 or not 0 <= joint <= 1:
                raise RuntimeError(f'Nonprobability CFR: {job_id}')
            if absent:
                if raw != 0 or joint != 0 or returned != 0:
                    raise RuntimeError(f'No-path failure omitted from denominator: {job_id}')
            else:
                n = protocol['external_samples']
                blob = packed[key]
                if blob.dtype != np.uint8 or blob.ndim != 1 or len(blob) != (n+7)//8:
                    raise RuntimeError(f'Invalid packed collision-array structure: {job_id}')
                decoded = np.unpackbits(blob, bitorder='little')
                if decoded[n:].any():
                    raise RuntimeError(f'Nonzero unused packed bits: {job_id}')
                k = int(decoded[:n].sum())
                if not math.isclose(raw, k/n, rel_tol=0, abs_tol=1e-14):
                    raise RuntimeError(f'Collision-bit count differs from reported CFR: {job_id}')
                if not np.allclose(metrics[f'cfr_ci_{key}'], wilson(k,n), rtol=0, atol=1e-12):
                    raise RuntimeError(f'Wilson interval mismatch: {job_id}')
                packed_audits += 1
            wanted_joint = raw if metrics['nominal_success'] else 0.
            wanted_returned = wanted_joint if metrics['internal_constraints_satisfied'] else 0.
            if joint != wanted_joint or returned != wanted_returned:
                raise RuntimeError(f'Success/CFR semantics mismatch: {job_id}')
        if packed is not None:
            packed.close()
        receipt = receipts[job_id][-1]
        elapsed = record['planning_seconds']
        if elapsed is None or not np.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError(f'Missing actual planning time: {job_id}')
        length = metrics.get('length')
        if not absent and (length is None or not np.isfinite(length) or length < 0):
            raise RuntimeError(f'Missing incumbent length: {job_id}')
        s = result.get('stats', {})
        row = {
            'job_id': job_id, 'method': job['method'], 'method_label': LABELS[job['method']],
            'map': job['map'], 'family': job['family'], 'seed': job['seed'],
            'path_returned': not absent, 'nominal_success': bool(metrics['nominal_success']),
            'planner_reported_success': bool(metrics['planner_reported_success']),
            'internal_constraints_satisfied': bool(metrics['internal_constraints_satisfied']),
            'length': length, 'planning_seconds': float(elapsed),
            'budget_overrun_seconds': float(s.get('budget_overrun_seconds', result.get('overrun_seconds',0.))),
            'process_seconds': float(receipt['process_seconds']),
            'observed_peak_process_tree_rss_bytes': int(receipt['peak_process_tree_rss_bytes']),
            'external_samples': int(metrics['external_samples']),
            'external_samples_requested': int(protocol['external_samples']),
            'curve_chord_error_bound': metrics.get('curve_chord_error_bound',0.),
            'arc_length_32_vs_64_difference': result.get('arc_length_32_vs_64_difference'),
            'free_waypoints': result.get('grid_initialization', {}).get('free_waypoints',
                                s.get('n_via',1)-1 if job['method']=='cc_vpsto' else (3 if job['method']=='ours' else None)),
        }
        for sigma in protocol['sigmas']:
            key = str(sigma)
            for prefix in ['cfr', 'joint_cfr', 'returned_success_cfr']:
                row[f'{prefix}_{key}'] = float(metrics[f'{prefix}_{key}'])
            row[f'nominal_and_target_pass_{key}'] = bool(metrics['nominal_success'] and metrics[f'cfr_{key}'] >= .825)
            row[f'cfr_ci_low_{key}'] = None if absent else metrics[f'cfr_ci_{key}'][0]
            row[f'cfr_ci_high_{key}'] = None if absent else metrics[f'cfr_ci_{key}'][1]
        rows.append(row)
        result_hashes[actual[job_id].name] = sha(actual[job_id])
    audit = {
        'passed': True, 'expected_jobs': len(expected), 'actual_jobs': len(actual),
        'maps': len(map_items), 'family_map_counts': dict(family_maps),
        'packed_collision_arrays_audited': packed_audits, 'no_path_job_ids': absent_paths,
        'failed_process_attempts_retained': failed_attempts,
        'source_immutability_verified': True, 'source_sha256': source_actual,
        'protocol_sha256': protocol_hash, 'map_manifest_sha256': sha(bank/'manifest.json'),
        'analysis_plan_sha256': sha(HERE/'analysis_plan.md'), 'analysis_code_sha256': sha(__file__),
        'frozen_environment_file_sha256': sha(study/'environment_freeze.txt'),
        'result_file_sha256': result_hashes, 'bit_file_sha256': bit_hashes,
        'not_recomputed': 'Audit checks packed counts and immutable evaluation code; it does not re-run planning or every collision predicate.',
    }
    return protocol, rows, audit


def map_aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'], row['map'])].append(row)
    output = []
    for (method, name), group in sorted(groups.items()):
        feasible = [r['length'] for r in group if r['nominal_success'] and r['length'] is not None]
        item = {'method': method, 'method_label': LABELS[method], 'map': name,
                'family': group[0]['family'], 'runs': len(group),
                'nominal_feasible_count': sum(r['nominal_success'] for r in group),
                'nominal_feasible_length_mean': float(np.mean(feasible)) if feasible else None}
        for key in ['nominal_success', 'internal_constraints_satisfied', 'cfr_0.6', 'joint_cfr_0.6',
                    'returned_success_cfr_0.6', 'nominal_and_target_pass_0.6', 'planning_seconds']:
            item[key+'_mean'] = float(np.mean([r[key] for r in group]))
        output.append(item)
    return output


def descriptions(rows, maps):
    output = []
    families = sorted({r['family'] for r in rows})
    for family in ['all']+families:
        for method in METHODS:
            subset = [r for r in rows if r['method']==method and (family=='all' or r['family']==family)]
            mapped = [r for r in maps if r['method']==method and (family=='all' or r['family']==family)]
            times = np.asarray([r['planning_seconds'] for r in subset])
            feasible = [r['length'] for r in subset if r['nominal_success'] and r['length'] is not None]
            map_lengths = [r['nominal_feasible_length_mean'] for r in mapped if r['nominal_feasible_length_mean'] is not None]
            output.append({
                'family': family, 'method': method, 'method_label': LABELS[method],
                'maps': len(mapped), 'runs': len(subset),
                'nominal_feasible_runs': sum(r['nominal_success'] for r in subset),
                'internal_feasible_runs': sum(r['internal_constraints_satisfied'] for r in subset),
                'nominal_and_target_pass_runs': sum(r['nominal_and_target_pass_0.6'] for r in subset),
                'no_path_runs': sum(not r['path_returned'] for r in subset),
                'joint_cfr_percent': 100*float(np.mean([r['joint_cfr_0.6_mean'] for r in mapped])),
                'raw_cfr_percent': 100*float(np.mean([r['cfr_0.6_mean'] for r in mapped])),
                'nominal_feasible_percent': 100*float(np.mean([r['nominal_success_mean'] for r in mapped])),
                'internal_feasible_percent': 100*float(np.mean([r['internal_constraints_satisfied_mean'] for r in mapped])),
                'nominal_and_target_pass_percent': 100*float(np.mean([r['nominal_and_target_pass_0.6_mean'] for r in mapped])),
                'nominal_feasible_length_map_mean': float(np.mean(map_lengths)) if map_lengths else None,
                'nominal_feasible_length_pooled_mean': float(np.mean(feasible)) if feasible else None,
                'nominal_feasible_length_maps': len(map_lengths),
                'planning_seconds_mean': float(times.mean()),
                'planning_seconds_median': float(np.median(times)),
                'planning_seconds_p95': float(np.quantile(times,.95)),
                'planning_seconds_max': float(times.max()),
                'budget_overrun_seconds_max': max(r['budget_overrun_seconds'] for r in subset),
                'observed_peak_process_tree_rss_mib': max(r['observed_peak_process_tree_rss_bytes'] for r in subset)/1024**2,
            })
    return output


def primary_contrasts(rows, rng):
    indexed = {(r['method'], r['map'], r['seed']): r for r in rows}
    maps = sorted({r['map'] for r in rows})
    paired_rows, contrasts = [], []
    for comparator in ['cc_vpsto', 'saa']:
        for outcome in ['joint_cfr', 'length']:
            group = []
            excluded = []
            for name in maps:
                a = sorted([r for r in rows if r['method']=='ours' and r['map']==name], key=lambda r:r['seed'])
                pairs = [(r, indexed[(comparator,name,r['seed'])]) for r in a]
                if outcome == 'length':
                    pairs = [(x,y) for x,y in pairs if x['nominal_success'] and y['nominal_success']]
                if not pairs:
                    excluded.append(name)
                    continue
                key = 'joint_cfr_0.6' if outcome=='joint_cfr' else 'length'
                ours = float(np.mean([x[key] for x,y in pairs]))
                other = float(np.mean([y[key] for x,y in pairs]))
                difference = ours-other
                if outcome == 'joint_cfr' and all('external_samples_requested' in x for x,y in pairs):
                    # Joint CFR has an exact integer-world numerator. Preserve
                    # genuine rank ties instead of float-subtraction roundoff.
                    n_worlds = pairs[0][0]['external_samples_requested']
                    assert all(x['external_samples_requested']==n_worlds and y['external_samples_requested']==n_worlds for x,y in pairs)
                    delta_count = sum(round(x[key]*n_worlds)-round(y[key]*n_worlds) for x,y in pairs)
                    difference = delta_count/(n_worlds*len(pairs))
                item = {'comparison': f'ours_minus_{comparator}', 'outcome': outcome,
                        'map': name, 'family': a[0]['family'], 'paired_seeds': len(pairs),
                        'eligible_seeds': [x['seed'] for x,y in pairs], 'ours': ours,
                        'comparator': other, 'difference': difference}
                group.append(item)
                paired_rows.append(item)
            differences = [r['difference'] for r in group]
            test = exact_signed_rank(differences)
            scale = 100. if outcome=='joint_cfr' else 1.
            ci = bootstrap_paired(group, rng, scale)
            avg_a = float(np.mean([r['ours'] for r in group])) if group else None
            avg_b = float(np.mean([r['comparator'] for r in group])) if group else None
            contrasts.append({
                'comparison': f'RiskAwareDE minus {LABELS[comparator]}', 'comparator': comparator,
                'outcome': outcome, 'available': bool(group), 'maps_available': len(group),
                'paired_seeds_total': sum(r['paired_seeds'] for r in group),
                'excluded_no_common_feasible_map_ids': excluded, 'family_map_counts': ci['family_map_counts'],
                'effect_unit': 'percentage points' if outcome=='joint_cfr' else 'coordinate units',
                'mean_paired_difference': scale*float(np.mean(differences)) if group else None,
                'difference_ci95': ci['difference_ci95'],
                'paired_ours_mean': avg_a, 'paired_comparator_mean': avg_b,
                'percentage_length_change': 100*(avg_a/avg_b-1) if outcome=='length' and avg_b else None,
                'percentage_length_change_ci95': ci['percentage_change_ci95'] if outcome=='length' else None,
                'signed_rank': test, 'p_value': test['p_value'],
                'test_interpretation': 'Conditional symmetric random-sign distribution of map differences; not method randomization.',
            })
    adjusted = holm_adjust([c['p_value'] for c in contrasts])
    for item, adjusted_p in zip(contrasts, adjusted):
        item['holm_adjusted_p'] = adjusted_p
        item['holm_family_size'] = 4
    return contrasts, paired_rows


def fmt(value, digits=3):
    return 'NA' if value is None else f'{value:.{digits}f}'


def render_markdown(summary):
    out = ['# Confirmatory geometric-adaptation study', '',
           'All frozen jobs and independent collision-bit counts passed the completeness and integrity audit. '
           'This comparison uses 24 new maps with five stochastic seeds per risk planner; A* has one run per map.', '',
           'The main reliability endpoint is nominal-and-perturbed joint CFR. Failed/absent paths stay in the denominator. '
           'Internal feasibility uses method-specific risk functionals and is not an equivalent safety guarantee.', '',
           '| Method | Runs | Joint CFR (%) | Raw CFR (%) | Nominal feasible (%) | Target pass (%) | Feasible length | Time median / P95 (s) | Peak tree RSS (MiB) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['descriptive']:
        if r['family'] != 'all':
            continue
        out.append(f"| {r['method_label']} | {r['runs']} | {r['joint_cfr_percent']:.2f} | {r['raw_cfr_percent']:.2f} | "
                   f"{r['nominal_feasible_percent']:.2f} | {r['nominal_and_target_pass_percent']:.2f} | "
                   f"{fmt(r['nominal_feasible_length_map_mean'])} | {r['planning_seconds_median']:.3f} / {r['planning_seconds_p95']:.3f} | "
                   f"{r['observed_peak_process_tree_rss_mib']:.1f} |")
    out += ['', 'Feasible length is the mean of map-level means among nominally feasible outputs; it is descriptive, '
            'and its eligible map/run counts are in method_summary.csv. Target pass means nominal feasibility and external CFR point estimate >=82.5%; '
            'it is not an individual-path certificate. Process-tree RSS is the observed 0.5-second-sampled peak including process overhead/evaluation, not optimizer-only allocated memory.', '',
            '## Four prespecified primary contrasts', '',
            '| Comparator | Outcome | Maps / paired seeds | Mean paired change [95% map-bootstrap CI] | Raw p | Holm p |',
            '|---|---|---:|---:|---:|---:|']
    for c in summary['primary_contrasts']:
        ci = c['difference_ci95']
        effect = 'NA' if ci is None else f"{c['mean_paired_difference']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        out.append(f"| {LABELS[c['comparator']]} | {c['outcome']} ({c['effect_unit']}) | "
                   f"{c['maps_available']} / {c['paired_seeds_total']} | {effect} | {c['p_value']:.6g} | {c['holm_adjusted_p']:.6g} |")
    out += ['', 'Differences are RiskAwareDE minus comparator. Positive reliability change favors RiskAwareDE; negative length change favors RiskAwareDE. '
            'Length is conditional on paired nominal feasibility and is not an unconditional quality score.', '',
            'Tests use the exact random-sign distribution of average-tied Wilcoxon ranks after dropping zero map differences. '
            'They require symmetry of differences. All four p-values share one Holm family. Percentile confidence intervals use 20,000 whole-map resamples '
            'within each generator family with fixed RNG seed 202610150704; the original available-map weights are retained.', '']
    for c in summary['primary_contrasts']:
        if c['outcome']=='length':
            out.append(f"- {LABELS[c['comparator']]}: percentage paired mean-length change {fmt(c['percentage_length_change'])}%; "
                       f"excluded maps without a common nominally feasible seed: {', '.join(c['excluded_no_common_feasible_map_ids']) or 'none'}.")
    out += ['', '## Scope and audit', '',
            'RiskAwareDE uses its frozen 820-evaluation budget; both external adaptations use six-second allowances and A* runs to completion. '
            'Actual times/overruns are recorded. This is not equal-time superiority, unmodified source-paper replication, a state-of-the-art system ranking, '
            'or a dynamic-flight comparison. Curves are conservatively checked with the frozen chord tolerance.', '',
            'The analysis specification was written at 88/384 completed jobs before method aggregates were inspected; '
            'it is explicitly not represented as preceding data collection. Audit/source/result hashes are in audit.json. '
            'Complete row data, family summaries, conditional-pair membership and excluded maps are supplied alongside this report.', '']
    return '\n'.join(out)


def render_tex(summary):
    label = {'ours': r'\RiskAwareDE{} (frozen)',
             'cc_vpsto': r'CC-VPSTO-3D\cite{brudermueller2026ccvpsto,jankowski2023vpsto}',
             'saa': r'SAA--AVaR-3D\cite{lew2024saa}',
             'astar': r'Grid A*\cite{hart1968astar}'}
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{Prospective geometric-adaptation comparison on 24 new maps. All numerical entries were computed in this study; citations identify source formulations. Joint CFR counts nominally infeasible or absent paths as failures. Length is conditional on nominal feasibility. The timing settings differ: the frozen main planner uses 820 evaluations, external adaptations receive six-second allowances, and A* completes.}',
             r'\label{tab:external-adaptation}', r'\begin{tabular}{lrrrrr}',r'\toprule',
             r'Method & Joint CFR (\%) & Nominal (\%) & Target (\%) & Length & Time median / P95 (s) \\',r'\midrule']
    for r in summary['descriptive']:
        if r['family']=='all':
            lines.append(f"{label[r['method']]} & {r['joint_cfr_percent']:.2f} & {r['nominal_feasible_percent']:.2f} & "
                         f"{r['nominal_and_target_pass_percent']:.2f} & {fmt(r['nominal_feasible_length_map_mean'])} & "
                         f"{r['planning_seconds_median']:.3f} / {r['planning_seconds_p95']:.3f} " + r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',r'\begin{table}[t]',r'\centering',r'\small',
              r'\caption{Map-paired contrasts, main method minus external adaptation. Confidence intervals use family-stratified map bootstraps. The four exact signed-rank tests share one Holm correction. Length contrasts retain only seeds nominally feasible for both methods.}',
              r'\label{tab:external-adaptation-paired}',r'\begin{tabular}{llrrrr}',r'\toprule',
              r'Comparator & Outcome & Maps & Pairs & Change [95\% CI] & Holm $p$ \\',r'\midrule']
    for c in summary['primary_contrasts']:
        ci = c['difference_ci95']
        effect = 'NA' if ci is None else f"{c['mean_paired_difference']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        outcome = 'CFR (pp)' if c['outcome']=='joint_cfr' else 'Length'
        method = 'CC-VPSTO-3D' if c['comparator']=='cc_vpsto' else 'SAA--AVaR-3D'
        lines.append(f"{method} & {outcome} & {c['maps_available']} & {c['paired_seeds_total']} & {effect} & "
                     f"{c['holm_adjusted_p']:.4g} " + r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '']
    return '\n'.join(lines)


def self_test():
    # Independent brute-force sign enumeration, including tied ranks/zeros.
    examples = [[1,2,3], [1,-2,3,-4], [0,1,-1,2,-2,2], [0,0,0], [-1]*6]
    for example in examples:
        d = np.asarray(example, float)
        nonzero = d[d != 0]
        if not len(nonzero):
            assert exact_signed_rank(d)['p_value'] == 1
            continue
        ranks = rankdata(np.abs(nonzero), method='average')
        observed = abs(float(ranks @ np.sign(nonzero)))
        values = [abs(float(ranks @ signs)) for signs in itertools.product([-1,1], repeat=len(nonzero))]
        brute = sum(value >= observed for value in values)/len(values)
        assert exact_signed_rank(d)['p_value'] == brute
    assert exact_signed_rank(np.ones(24))['p_value'] == 2/2**24
    assert np.allclose(holm_adjust([.01,.04,.03,1.]), [.04,.09,.09,1.])
    synthetic = [{'map': f'm{i}', 'family': 'a' if i<2 else 'b', 'ours': float(i+2),
                  'comparator': float(i+1)} for i in range(5)]
    x = bootstrap_paired(synthetic,np.random.default_rng(10),replicates=500)
    assert np.allclose(x['difference_ci95'],[1,1])
    assert x['family_map_counts'] == {'a':2,'b':3}
    assert x == bootstrap_paired(synthetic,np.random.default_rng(10),replicates=500)
    # Conditional selection and missing maps with no jointly feasible seed.
    fake=[]
    for method in METHODS:
        for map_index in range(3):
            for seed in ([1] if method=='astar' else [1,2]):
                fake.append({'method':method,'map':f'm{map_index}','family':f'f{map_index}',
                             'seed':seed,'nominal_success':not(method=='saa' and map_index==2),
                             'joint_cfr_0.6':.8 if method=='ours' else .7,
                             'length':10. if method=='ours' else 11.})
    contrasts,pairs=primary_contrasts(fake,np.random.default_rng(ANALYSIS_SEED))
    assert len(contrasts)==4 and contrasts[-1]['maps_available']==2
    assert contrasts[-1]['excluded_no_common_feasible_map_ids']==['m2']
    assert contrasts[-1]['paired_seeds_total']==4
    print('PASS: exact signed-rank DP vs exhaustive signs; ties/zeros; 24-map tails; Holm; deterministic stratified bootstrap; conditional-pair exclusions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='confirmatory_v1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    study = HERE/'runs'/args.study
    protocol,rows,audit = audit_and_rows(study)
    maps = map_aggregate(rows)
    descriptive = descriptions(rows,maps)
    contrasts,paired = primary_contrasts(rows,np.random.default_rng(ANALYSIS_SEED))
    summary = {
        'study': args.study, 'analysis_timestamp_asia_shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'protocol_sha256': audit['protocol_sha256'], 'analysis_plan_sha256': audit['analysis_plan_sha256'],
        'analysis_code_sha256': audit['analysis_code_sha256'], 'analysis_rng_seed': ANALYSIS_SEED,
        'bootstrap_replicates': BOOTSTRAP_REPLICATES, 'audit_passed': True,
        'primary_metric': 'joint_cfr_0.6; nominal failure/no path=0',
        'inference_unit': 'map', 'descriptive': descriptive, 'primary_contrasts': contrasts,
        'timing_scope': protocol['budget'], 'interpretation': protocol['interpretation'],
        'analysis_plan_timing': 'Written at 88/384 completed jobs before method aggregates; not before data collection.',
    }
    out = args.output or HERE/'analysis'/args.study
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'audit.json',audit)
    dump(out/'summary.json',summary)
    write_csv(out/'row_level.csv',rows)
    write_csv(out/'map_level.csv',maps)
    write_csv(out/'method_summary.csv',[r for r in descriptive if r['family']=='all'])
    write_csv(out/'family_summary.csv',[r for r in descriptive if r['family']!='all'])
    write_csv(out/'paired_map_effects.csv',paired)
    write_csv(out/'primary_contrasts.csv',contrasts)
    (out/'report.md').write_text(render_markdown(summary),encoding='utf-8')
    (out/'tables.tex').write_text(render_tex(summary),encoding='utf-8')
    print(json.dumps({'study':args.study,'audited_jobs':len(rows),'maps':protocol['maps'],
                      'output':str(out),'primary_tests':len(contrasts)},ensure_ascii=False))


if __name__ == '__main__':
    main()
