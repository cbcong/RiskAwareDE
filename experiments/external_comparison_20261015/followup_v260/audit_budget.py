"""Independent, read-only validation of the completed 2.6 budget follow-up.

No planners, scoring kernels, or Monte Carlo draws are run.  The only output is
budget_analysis/independent_audit.json after all checks pass.  Expected relative
directory layout matches the portable experiment archive.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL_PATH = HERE / 'budget_protocol.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def relative(path):
    return str(Path(path).relative_to(ROOT)).replace('\\', '/')


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    den = 1+z*z/n
    mid = (p+z*z/(2*n))/den
    rad = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-rad, mid+rad]


def audit():
    p = read(PROTOCOL_PATH)
    require(sha(HERE/'budget_sensitivity.py') == p['controller_sha256'], 'Frozen budget controller changed')
    original_path = ROOT/'runs'/'confirmatory_v1'/'protocol.json'
    require(sha(original_path) == p['source_study_sha256'], 'Original study protocol changed')
    original = read(original_path)
    require(p['source_sha256'] == original['source_sha256'], 'Frozen source set differs from original')
    require(len(p['source_sha256']) == 52, 'Expected 52 frozen implementation files')
    for name, expected in p['source_sha256'].items():
        require(sha(ROOT/name) == expected, f'Frozen source changed: {name}')
    for name, expected in p['original_files_sha256'].items():
        require(sha(ROOT/name) == expected, f'Archived study input changed: {name}')
    require(p['samples_per_path'] == 20000 and p['planning_sigma'] == .6 and p['evaluation_sigma'] == .6, 'Unexpected evaluation settings')
    require(p['external_root'] == original['external_root'], 'Archived/new worlds are not paired')
    original_jobs = {j['id']: j for j in original['jobs']}
    queue = p['queue']
    require(len(queue) == p['new_optimizations'] == 288, 'Unexpected budget queue size')
    expected_pairs = {(q['study'],q['job']['id']) for q in queue}
    require(len(expected_pairs) == 288, 'Duplicate study/job queue identity')
    require({q['budget_seconds'] for q in queue} == {1,3}, 'Unexpected stopping allowance')
    require({q['job']['method'] for q in queue} == {'cc_vpsto','saa'}, 'Unexpected new method')
    grouped = defaultdict(list)
    for q in queue:
        j = q['job']
        require(original_jobs.get(j['id']) == j, 'New queue job differs from archived map/seed/method')
        require(j['seed'] in (101,102,103), 'Unexpected planner seed')
        require(q['study'] == f'budget_v260_{q["budget_seconds"]}s', 'Study/allowance identity differs')
        grouped[(q['budget_seconds'],j['method'],j['map'])].append(j['seed'])
    require(len(grouped) == 96, 'Expected 24 maps for each of four conditions')
    require(all(sorted(v) == [101,102,103] for v in grouped.values()), 'Map-condition seeds incomplete or duplicated')
    condition_counts = Counter((q['budget_seconds'],q['job']['method']) for q in queue)
    require(set(condition_counts.values()) == {72}, 'Each method/allowance must contain 72 outputs')
    condition_families = Counter((q['budget_seconds'],q['job']['method'],q['job']['family']) for q in queue)
    require(len(condition_families) == 12 and set(condition_families.values()) == {24}, 'Family balance differs')
    orders = []
    block_ids = []
    for start in range(0, len(queue), 4):
        block = queue[start:start+4]
        ids = {(q['job']['map'],q['job']['seed']) for q in block}
        require(len(ids) == 1, 'Randomized queue block mixes map/seed identities')
        block_ids.append(next(iter(ids)))
        orders.append(tuple((q['budget_seconds'],q['job']['method']) for q in block))
    require(len(set(block_ids)) == 72, 'Duplicate map/seed randomized blocks')
    permutations = Counter(orders)
    wanted = set(itertools.permutations([(b,m) for b in (1,3) for m in ('cc_vpsto','saa')]))
    require(set(permutations) == wanted and set(permutations.values()) == {3}, '24 order permutations are not repeated exactly three times')

    studies = {}
    for study, expected in p['studies_sha256'].items():
        path = ROOT/'runs'/study/'protocol.json'
        require(sha(path) == expected, f'Frozen follow-up study protocol changed: {study}')
        study_protocol = read(path)
        wanted_jobs = {q['job']['id']:q['job'] for q in queue if q['study'] == study}
        actual_jobs = {j['id']:j for j in study_protocol['jobs']}
        require(actual_jobs == wanted_jobs and len(study_protocol['jobs']) == 144, 'Per-study jobs differ from queue')
        require(study_protocol['expected_jobs'] == 144 and study_protocol['stochastic_seeds'] == [101,102,103], 'Per-study cohort count differs')
        require(study_protocol['source_sha256'] == original['source_sha256'], 'Per-study sources changed')
        require(study_protocol['external_samples'] == 20000 and study_protocol['external_root'] == p['external_root'], 'Per-study external worlds differ')
        require(study_protocol['sigmas'] == [.6] and study_protocol['curve_tolerance'] == original['curve_tolerance'], 'Per-study geometry changed')
        require(study_protocol['method_options'] == original['method_options'] and study_protocol['saa_grid_initialization'] == original['saa_grid_initialization'], 'Method settings changed beyond stopping allowance')
        require(study_protocol['external_budget_seconds'] == next(q['budget_seconds'] for q in queue if q['study'] == study), 'Stopping allowance differs')
        actual_files = {path.stem for path in (ROOT/'runs'/study/'results').glob('*.json')}
        require(actual_files == set(wanted_jobs), f'Incomplete or extra result files: {study}')
        studies[study] = study_protocol
    require(set(studies) == {q['study'] for q in queue}, 'Missing or unexpected follow-up study')

    receipt_path = HERE/'budget_process_log.jsonl'
    raw_receipt_bytes = receipt_path.read_bytes()
    receipts = [json.loads(line) for line in raw_receipt_bytes.decode('utf-8-sig').splitlines() if line.strip()]
    successful = [r for r in receipts if r['exit_code'] == 0 and not r['hard_timeout']]
    failed = [r for r in receipts if r['exit_code'] != 0 or r['hard_timeout']]
    require(len(successful) == 288, 'Expected exactly 288 successful worker receipts')
    actual_pairs = {(r['study'],r['id']) for r in successful}
    require(actual_pairs == expected_pairs and len(actual_pairs) == len(successful), 'Successful receipt identities do not match the entire queue')
    amendment_hash = None
    failed_log_hash = None
    if failed:
        require(len(failed) == 1 and len(receipts) == 289, 'Only the single documented pre-output execution interruption is permitted')
        amendment_path = HERE/'budget_execution_amendment.json'
        amendment = read(amendment_path)
        require(failed[0] == amendment['failed_attempt'], 'Failed receipt does not exactly match the recorded amendment')
        require(failed[0]['exit_code'] == 3221225658 and not failed[0]['hard_timeout'], 'Unexpected failure type; cannot silently apply this exception')
        failed_pair = (failed[0]['study'],failed[0]['id'])
        require(failed_pair in expected_pairs, 'Failed attempt was not part of the planned queue')
        attempts_for_job = [r for r in receipts if (r['study'],r['id']) == failed_pair]
        require(len(attempts_for_job) == 2 and attempts_for_job[0] == failed[0] and attempts_for_job[1]['exit_code'] == 0,
                'Expected one documented failed attempt followed by one successful identical-job retry')
        require(amendment['output_existed'] is False and amendment['log_bytes'] == 0, 'Exception was not documented before any output existed')
        failed_log = HERE/'budget_failed_startup.log'
        require(failed_log.stat().st_size == 0 and sha(failed_log) == amendment['log_sha256'], 'Preserved failed-attempt log differs')
        prefix = bytearray()
        for line in raw_receipt_bytes.splitlines(keepends=True):
            prefix.extend(line)
            if line.strip() and json.loads(line.decode('utf-8-sig')) == failed[0]:
                break
        require(hashlib.sha256(prefix).hexdigest() == amendment['raw_process_log_before_retry_sha256'],
                'Raw receipt prefix recorded before retry has changed')
        amendment_hash, failed_log_hash = sha(amendment_path), sha(failed_log)
    else:
        require(len(receipts) == 288, 'Unexpected extra attempts despite no documented failure')
    for r in receipts:
        require(not r['hard_timeout'], f'Hard timeout is not covered by the limited execution amendment: {r["study"]}/{r["id"]}')
        require(math.isfinite(r['process_seconds']) and r['process_seconds'] > 0, 'Invalid process duration')
        require(r['peak_process_tree_rss_bytes'] > 0, 'Missing process-tree memory measurement')

    archived = [j for j in original['jobs'] if j['method'] in ('ours','cc_vpsto','saa') and j['seed'] in (101,102,103)]
    require(len(archived) == 216, 'Archived reference subset incomplete')
    archived_groups = defaultdict(list)
    for j in archived:
        archived_groups[(j['method'],j['map'])].append(j['seed'])
    require(len(archived_groups) == 72 and all(sorted(v) == [101,102,103] for v in archived_groups.values()), 'Archived method/map matching differs')
    studies['confirmatory_v1'] = original
    all_records = [(q['study'],q['job']) for q in queue] + [('confirmatory_v1',j) for j in archived]
    record_hashes, bit_hashes, map_hashes = {}, {}, {}
    no_path, nominal_failures, initialization_exhausted = [], [], []
    expected_bit_files = defaultdict(set)
    sampled_records = 0
    for study, j in all_records:
        sp = studies[study]
        result_path = ROOT/'runs'/study/'results'/(j['id']+'.json')
        record = read(result_path)
        require(record['job'] == j and record['protocol_sha256'] == sha(ROOT/'runs'/study/'protocol.json'), 'Result/protocol identity differs')
        require(record['evidence_stage'] == sp['evidence_stage'], 'Result evidence stage differs')
        map_path = ROOT/sp['bank']/j['map_file']
        require(sha(map_path) == j['map_sha256'], 'Scenario geometry changed')
        scene = read(map_path)
        require(scene['name'] == j['map'], 'Scenario name does not match external stream key')
        map_hashes[relative(map_path)] = sha(map_path)
        result, m = record['result'], record['metrics']
        bits_path = result_path.with_name(j['id']+'_external_bits.npz')
        elapsed = record['planning_seconds']
        require(isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and elapsed >= 0, 'Invalid planning duration')
        requested = result.get('stats',{}).get('budget_seconds',result.get('budget_seconds'))
        if requested is not None and j['method'] != 'ours':
            require(requested == sp['external_budget_seconds'], 'Recorded requested allowance differs')
        if result.get('status') == 'initialization_budget_exhausted':
            initialization_exhausted.append(f'{study}/{j["id"]}')
        points = result.get('points')
        if points is None:
            require(not m['nominal_success'] and not m['internal_constraints_satisfied'], 'Absent path marked feasible')
            require(m['external_samples'] == 0 and m['length'] is None and not bits_path.exists(), 'Absent path fields or stale bits inconsistent')
            require(m['cfr_0.6'] == m['joint_cfr_0.6'] == m['returned_success_cfr_0.6'] == 0., 'Absent path omitted from zero denominator')
            no_path.append(f'{study}/{j["id"]}')
        else:
            array = np.asarray(points,float)
            require(array.ndim == 2 and array.shape[1] == 3 and len(array) >= 2 and np.isfinite(array).all(), 'Invalid returned path')
            require(m['external_samples'] == 20000, 'External probability denominator is not 20000')
            expected_seed = int.from_bytes(hashlib.sha256(f'external-evaluation-v1|{p["external_root"]}|{j["map"]}'.encode()).digest()[:8],'little')
            require(m['external_seed'] == expected_seed, 'External random stream differs across allowance/method')
            endpoints = bool(np.allclose(array[0],scene['start'],rtol=0,atol=1e-7) and np.allclose(array[-1],scene['goal'],rtol=0,atol=1e-7))
            require(m['endpoints_correct'] == endpoints, 'Endpoint gate differs from saved path')
            if j['method'] == 'cc_vpsto':
                require(m['within_world_bounds'] == result['native_bounds_extrema_checked'], 'Native cubic bounds gate differs')
                require(0 <= m['curve_chord_error_bound'] <= sp['curve_tolerance'], 'Cubic approximation bound exceeds protocol')
            length = m['length']
            require(isinstance(length,(int,float)) and math.isfinite(length) and length >= 0, 'Invalid incumbent path length')
            with np.load(bits_path,allow_pickle=False) as packed:
                require(set(packed.files) == {'0.6'}, 'Unexpected external scale arrays')
                bits = packed['0.6']
                require(bits.dtype == np.uint8 and bits.shape == (2500,), 'Packed collision bits must encode exactly 20000 worlds')
                decoded = np.unpackbits(bits,bitorder='little')
                require(len(decoded) == 20000, 'Wrong decoded external denominator')
                count = int(decoded.sum())
            raw = count/20000
            require(abs(m['cfr_0.6']-raw) <= 1e-14, 'Reported raw CFR differs from packed count')
            require(np.allclose(m['cfr_ci_0.6'],wilson(count,20000),rtol=0,atol=1e-12), 'Reported Wilson interval differs')
            if not endpoints or not m['within_world_bounds']:
                require(raw == 0 and not m['nominal_success'], 'Invalid endpoint/workspace path retains nonzero reliability')
            expected_bit_files[study].add(bits_path.name)
            bit_hashes[relative(bits_path)] = sha(bits_path)
            sampled_records += 1
        raw = m['cfr_0.6']
        require(math.isfinite(raw) and 0 <= raw <= 1, 'CFR outside probability range')
        wanted_joint = raw if m['nominal_success'] else 0.
        wanted_returned = wanted_joint if m['internal_constraints_satisfied'] else 0.
        require(m['joint_cfr_0.6'] == wanted_joint, 'Nominal failure was excluded rather than scored zero')
        require(m['returned_success_cfr_0.6'] == wanted_returned, 'Internal feasibility gate differs')
        if not m['nominal_success']:
            nominal_failures.append(f'{study}/{j["id"]}')
        record_hashes[relative(result_path)] = sha(result_path)
    for study in p['studies_sha256']:
        actual_bit_files = {path.name for path in (ROOT/'runs'/study/'results').glob('*_external_bits.npz')}
        require(actual_bit_files == expected_bit_files[study], 'Unexpected or missing follow-up collision-bit files')
    require(len(record_hashes) == 504 and len(map_hashes) == 24, 'Final comparison record/map count differs')

    result = {
        'passed': True, 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'audit_code_sha256': sha(__file__), 'followup_protocol_sha256': sha(PROTOCOL_PATH),
        'frozen_controller_sha256': p['controller_sha256'],
        'worker_receipts_sha256': sha(receipt_path),
        'original_study_protocol_sha256': sha(original_path),
        'frozen_source_files_verified': len(p['source_sha256']),
        'archived_files_unchanged': len(p['original_files_sha256']),
        'new_queue_jobs': len(queue), 'new_successful_worker_receipts': len(successful),
        'total_worker_attempts': len(receipts), 'failed_execution_attempts_retained': failed,
        'execution_amendment_sha256': amendment_hash, 'preserved_failed_log_sha256': failed_log_hash,
        'retry_scope': 'Zero failures, or exactly the single documented pre-output interruption followed by one identical-job retry; all raw attempts retained. No output-based reruns are permitted.',
        'archived_reference_outputs': len(archived), 'total_output_records': len(record_hashes),
        'maps': len(map_hashes), 'seeds_per_map_per_condition': 3,
        'order_permutations': len(permutations), 'repeats_per_order_permutation': 3,
        'packed_bit_arrays_verified': sampled_records, 'samples_per_present_path': 20000,
        'no_path_outputs': no_path, 'nominal_failure_outputs_retained': nominal_failures,
        'initialization_budget_exhausted_outputs_retained': initialization_exhausted,
        'checks': ['exact_queue_receipt_identity', 'all_24_maps_and_three_fixed_seeds',
                   'four_conditions_balanced_over_24_orders', 'frozen_adapter_and_original_result_hashes',
                   'new_result_protocol_and_map_identity', 'matched_external_stream_and_exact_20000_denominator',
                   'raw_probability_and_Wilson_interval_from_bits', 'nominal_and_internal_failure_gates',
                   'no_path_outputs_zero_and_retained', 'finite_incumbent_lengths'],
        'not_recomputed': 'No planning, path collision scoring, or independent runtime benchmark is performed by this audit. It checks immutable inputs, recorded outputs, and bit-level arithmetic.',
        'scope': p['analysis'], 'reference': p['reference'],
        'source_sha256': p['source_sha256'], 'map_sha256': map_hashes,
        'output_record_sha256': record_hashes, 'output_bits_sha256': bit_hashes,
    }
    output = HERE/'budget_analysis'/'independent_audit.json'
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','map_sha256','output_record_sha256','output_bits_sha256','nominal_failure_outputs_retained','initialization_budget_exhausted_outputs_retained')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    audit()
