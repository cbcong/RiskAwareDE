"""Run the frozen six-category comparison on the two exposed research map banks.

All optimization is rerun. External perturbations are shared, and never enter
search or selection. A* is a separate deterministic geometric reference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from uav_path_planning import make_research_scenarios, split_research_scenarios
from uav_path_planning.benchmarks import BenchmarkConfig, _problem, _result_for_continuous, _result_for_astar
from uav_path_planning.conditional_projected_polish import ConditionalProjectedPolish
from uav_path_planning.conditional_risk_de import ConditionalRiskDE
from uav_path_planning.robustness import translate_environment
from run_extension_candidate import risk_config
from run_frozen_stress_suite import collision_free_batch, make_shift_bank, save_csv

FAMILIES = ('de', 'jade', 'shade', 'lshade', 'pso')
SIGMAS = (.3, .6, .9)
NFE = 820
TARGET_ALPHA = .175


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method_specifications():
    methods = {'full': {'category': 'reference', 'label': '完整方法',
                        'kind': 'projected', 'options': {'reserve_fraction': .2},
                        'risk_estimator': 'conditional-z'}}
    for family in FAMILIES:
        label = {'lshade': 'L-SHADE', 'jade': 'JADE', 'shade': 'SHADE', 'de': 'DE', 'pso': 'PSO'}[family]
        methods['bare_' + family] = {'category': 'E1', 'label': label,
            'kind': 'bare', 'family': family, 'risk_estimator': None}
        methods['risk_' + family] = {'category': 'E2', 'label': label + ' + 条件风险',
            'kind': 'risk_control', 'family': family, 'risk_estimator': 'conditional-z'}
        methods['risk_polish_' + family] = {'category': 'E3', 'label': label + ' + 条件风险 + 切向精修',
            'kind': 'projected', 'options': {'reserve_fraction': .2, 'global_method': 'baseline_' + family},
            'risk_estimator': 'conditional-z'}
    methods['no_terminal_polish'] = {'category': 'E4', 'label': '完整方法去掉末端精修',
        'kind': 'no_terminal_polish', 'risk_estimator': 'conditional-z'}
    methods['no_risk'] = {'category': 'E5', 'label': '完整方法去掉全部风险处理',
        'kind': 'no_risk', 'risk_estimator': None,
        'adaptation': 'nominal-success; nominal length polish; no perturbation checks'}
    methods['binary_risk'] = {'category': 'E6', 'label': '完整方法改用普通IID风险采样',
        'kind': 'projected', 'options': {'reserve_fraction': .2, 'risk_estimator': 'binary'},
        'risk_estimator': 'binary-IID'}
    methods['bare_astar'] = {'category': 'E1-reference', 'label': 'A* 3D（单独几何参照）',
        'kind': 'astar', 'risk_estimator': None, 'deterministic': True}
    return methods


def run_scene(task):
    rootseed, scene, method, spec, config_dict, seeds, output, samples, noise_seed = task
    cfg = BenchmarkConfig(**config_dict)
    problem = _problem(scene, cfg)
    bank = make_shift_bank([scene], 'gaussian', samples, noise_seed)[scene.name]
    seeds = (0,) if spec['kind'] == 'astar' else seeds
    rows, details, trials = [], [], []
    for seed in seeds:
        optimizer = raw = None
        cpu_start, start = time.process_time(), time.perf_counter()
        if spec['kind'] == 'bare':
            result = _result_for_continuous(scene, spec['family'], seed, cfg)
            path, evaluations = result.path, result.evaluations
            risk_checks, internal_risk, refinement, metadata = 0, None, {}, result.metadata
        elif spec['kind'] == 'astar':
            result = _result_for_astar(scene, cfg)
            path, evaluations = result.path, result.evaluations
            risk_checks, internal_risk, refinement, metadata = 0, None, {}, result.metadata
        else:
            rcfg = risk_config(cfg, seed)
            if spec['kind'] == 'projected':
                optimizer = ConditionalProjectedPolish(problem, rcfg, **spec['options'])
            elif spec['kind'] == 'risk_control':
                from uav_path_planning.conditional_baseline_control import ConditionalBaselineControl
                optimizer = ConditionalBaselineControl(problem, rcfg, axis=2, global_method=spec['family'])
            elif spec['kind'] == 'no_terminal_polish':
                optimizer = ConditionalRiskDE(problem, rcfg, axis=2)
            elif spec['kind'] == 'no_risk':
                from uav_path_planning.nominal_search_ablation import NominalProjectedPolish
                optimizer = NominalProjectedPolish(problem, rcfg, reserve_fraction=.2)
            else:
                raise ValueError('Unknown optimizer kind')
            raw = optimizer.optimize()
            path, evaluations = problem.decode(raw.best_vector), raw.evaluations
            risk_checks = raw.internal_collision_checks
            internal_risk = raw.best_evaluation.perturbed_collision_rate if spec['risk_estimator'] else None
            refinement = raw.local_refinement_stats
            metadata = getattr(optimizer, 'extension_metadata', {})
        runtime, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu_start
        if spec['kind'] != 'astar' and evaluations != NFE:
            raise AssertionError(f'{method}/{rootseed}/{scene.name}/{seed}: NFE {evaluations} != {NFE}')
        expected_checks = NFE * 16 if spec['risk_estimator'] else 0
        if risk_checks != expected_checks:
            raise AssertionError(f'{method}: risk checks {risk_checks} != {expected_checks}')
        metrics = problem.evaluate(path) if path is not None else None
        feasible = bool(metrics is not None and metrics.feasible)
        row = {'map_root': rootseed, 'scene': scene.name, 'method': method, 'seed': seed,
               'length': metrics.length if metrics else None, 'nominal_success': feasible,
               'evaluations': evaluations, 'risk_checks': risk_checks,
               'runtime_sec': runtime, 'cpu_seconds': cpu_seconds}
        trial_rows = []
        for sigma in SIGMAS:
            free = collision_free_batch(path, scene, sigma * bank, margin=cfg.safety_margin)
            if seed == seeds[0] and path is not None:
                for index in (0, samples - 1):
                    expected = path.collision_count(translate_environment(scene.environment, sigma * bank[index]), safety_margin=cfg.safety_margin) == 0
                    if bool(free[index]) != expected:
                        raise AssertionError('External scalar and batched collision predicates disagree')
            success = free & feasible
            rate = float(success.mean())
            row['cfr_' + str(sigma)] = rate
            row['pass_' + str(sigma)] = bool(feasible and rate >= 1 - TARGET_ALPHA - 1e-12)
            trial_rows.append(success)
        rows.append(row)
        trial_detail = {'row': row, 'waypoints': list(path.waypoints) if path else None,
                        'internal_risk': internal_risk, 'local_refinement_stats': refinement,
                        'extension_metadata': metadata}
        if raw is not None:
            trial_detail['objective_history'] = raw.history
            trial_detail['risk_gate_history'] = raw.gate_history if spec['risk_estimator'] else None
        details.append(trial_detail)
        trials.append(trial_rows)
    directory = Path(output) / str(rootseed) / method
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / (scene.name + '_trials.npz'),
                        success=np.asarray(trials), seeds=np.asarray(seeds), sigmas=np.asarray(SIGMAS))
    write_json(directory / (scene.name + '_paths.json'), details)
    write_json(directory / (scene.name + '.json'), rows)
    return {'map_root': rootseed, 'method': method, 'scene': scene.name, 'plans': len(rows),
            'ended_at': datetime.now(timezone.utc).isoformat()}


def collect_rows(output, roots, methods, scenes_by_root):
    rows = []
    for rootseed in roots:
        for method in methods:
            for scene in scenes_by_root[rootseed]:
                source = output / str(rootseed) / method / (scene.name + '.json')
                if source.exists():
                    rows.extend(json.loads(source.read_text(encoding='utf-8')))
    write_json(output / 'run_rows.json', rows)
    if rows:
        save_csv(output / 'run_rows.csv', rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--roots', type=int, nargs='+', default=[20260923, 20261005])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(9, 29)))
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--external-seed', type=int, default=196102)
    parser.add_argument('--scene-limit', type=int, help='Smoke-only: fewer maps; never a full result')
    args = parser.parse_args()
    specs = method_specifications()
    if args.methods:
        if any(m not in specs for m in args.methods):
            parser.error('Unknown method')
        specs = {m: specs[m] for m in args.methods}
    if args.workers < 1 or args.samples < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Invalid workers, samples or seeds')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Refusing to overwrite: ' + str(args.output))
    bundle_path = ROOT / 'configs/risk_aware_conservative_gate_20261002.json'
    bundle = json.loads(bundle_path.read_text(encoding='utf-8-sig'))
    config = asdict(replace(BenchmarkConfig(**bundle['benchmark_config']), seeds=tuple(args.seeds)))
    scenes_by_root = {r: list(split_research_scenarios(make_research_scenarios(r), protocol='research24_6_6_12').test)[:args.scene_limit] for r in args.roots}
    source_paths = sorted((ROOT / 'src/uav_path_planning').glob('*.py'))
    source_paths += [Path(__file__), ROOT / 'scripts/run_extension_candidate.py', ROOT / 'scripts/run_frozen_stress_suite.py', bundle_path]
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in source_paths}
    with zipfile.ZipFile(args.output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, str(path.relative_to(ROOT)))
    manifest = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
                'purpose': __doc__, 'methods': specs, 'benchmark_config': config,
                'map_roots': args.roots, 'planner_seeds': args.seeds, 'NFE': NFE, 'K': 16,
                'search_alpha': .1, 'target_alpha': TARGET_ALPHA, 'sigmas': SIGMAS,
                'external_samples': args.samples, 'external_seed': args.external_seed,
                'workers': args.workers, 'reuse': False, 'smoke_test': bool(args.scene_limit),
                'python': sys.version, 'platform': platform.platform(),
                'source_sha256': hashes, 'snapshot_sha256': sha256(args.output / 'source_snapshot.zip'),
                'scenarios': {str(r): [s.to_dict() for s in scenes] for r, scenes in scenes_by_root.items()},
                'limits': ['Both map banks exposed; not blind/OOD.', 'Empirical pass threshold is not a confidence certificate.',
                           'Equal NFE is not equal computation. Report risk calls and CPU/wall time.',
                           'Native scalar baseline risk adapters approximate continuous risk priority outside gate; document merit.',
                           'A* runs once per map, different representation and expansion budget.',
                           'All baseline cores are local reproductions/adaptations, not author-reported numbers.'],
                'primary_analysis': {'unit': 'map mean of paired seed outcomes; 12 maps per bank',
                    'contrasts': [['full', 'risk_polish_lshade', 'length'], ['full', 'bare_lshade', 'cfr_0.6'], ['full', 'no_terminal_polish', 'length']],
                    'family': '3 contrasts x 2 banks = 6 two-sided exact signed-rank tests, Holm adjustment',
                    'timing': 'Specified before this rerun, but maps and method previously explored; not public preregistration'}}
    tasks = []
    for rootseed, scenes in scenes_by_root.items():
        bank_file = args.output / f'common_unit_noise_{rootseed}.npz'
        np.savez_compressed(bank_file, **make_shift_bank(scenes, 'gaussian', args.samples, args.external_seed))
        manifest.setdefault('noise_sha256', {})[str(rootseed)] = sha256(bank_file)
        for method, spec in specs.items():
            for scene in scenes:
                tasks.append((rootseed, scene, method, spec, config, tuple(args.seeds), str(args.output), args.samples, args.external_seed))
    # Interleave method/map tasks to reduce systematic timing/order confounding.
    random.Random(20261003).shuffle(tasks)
    manifest['expected_plans'] = sum(1 if t[3]['kind'] == 'astar' else len(args.seeds) for t in tasks)
    manifest['expected_scene_tasks'] = len(tasks)
    write_json(args.output / 'execution_manifest.json', manifest)
    print(json.dumps({'tasks': len(tasks), 'plans': manifest['expected_plans'], 'workers': args.workers, 'output': str(args.output)}), flush=True)
    completed = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(run_scene, task): task for task in tasks}
            for future in as_completed(pending):
                completed.append(future.result())
                write_json(args.output / 'progress.json', {'completed_scene_tasks': len(completed), 'total_scene_tasks': len(tasks), 'completed_plans': sum(x['plans'] for x in completed), 'completed': completed})
                if len(completed) % 20 == 0 or len(completed) == len(tasks):
                    print(json.dumps({'done': len(completed), 'total': len(tasks), 'last': completed[-1]}, ensure_ascii=False), flush=True)
        rows = collect_rows(args.output, args.roots, specs, scenes_by_root)
        if len(rows) != manifest['expected_plans']:
            raise AssertionError('Incomplete run matrix')
        changed = [p for p, digest in hashes.items() if sha256(ROOT / p) != digest]
        if changed:
            raise RuntimeError('Source changed during experiment: ' + ', '.join(changed))
        manifest.update(status='complete', ended_at=datetime.now(timezone.utc).isoformat(), actual_plans=len(rows))
        write_json(args.output / 'completion_receipt.json', {'status': 'complete', 'plans': len(rows),
            'scene_tasks': len(completed), 'source_hashes_unchanged': True, 'run_rows_sha256': sha256(args.output / 'run_rows.json'),
            'ended_at': manifest['ended_at']})
    except BaseException as exc:
        collect_rows(args.output, args.roots, specs, scenes_by_root)
        manifest.update(status='failed', error=repr(exc), ended_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write_json(args.output / 'execution_manifest.json', manifest)
    print(json.dumps({'status': 'complete', 'plans': len(rows), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
