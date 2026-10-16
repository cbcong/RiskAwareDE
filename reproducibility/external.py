"""Recompute the frozen external comparison and follow-ups in an isolated tree.

Fresh output: 384 optimizations, 384 saved-path stress evaluations (two scales),
and 288 lower-budget optimizations. Reusing an output resumes its own records.
Smoke output is deliberately incomplete and cannot be reused as a full run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
RELATIVE_EXPERIMENT = Path('experiments/external_comparison_20261015')
SOURCE = REPO / RELATIVE_EXPERIMENT
METHODS = ('ours', 'cc_vpsto', 'saa', 'astar')
ENVIRONMENT = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from reproducibility.atomic_io import write_json as dump


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


class Paused(RuntimeError):
    """A requested stop is a resumable pause, never a completed experiment."""


def check_output(output):
    output = Path(output).resolve()
    require(not REPO.is_relative_to(output), 'Output cannot be the repository or its ancestor.')
    if output.is_relative_to(REPO):
        root = REPO / 'outputs'
        require(output.is_relative_to(root) and output != root,
                'Inside this repository, --output must be a child of outputs/. '
                'Source, data, reference, config and experiment directories are protected.')
    return output


def reject_redirects(root):
    """Never write through an existing symbolic link or Windows junction."""
    pending = [root] if root.exists() else []
    while pending:
        path = pending.pop()
        require(not path.is_symlink() and not getattr(path, 'is_junction', lambda: False)(),
                f'Output contains a redirected path: {path}')
        if path.is_dir():
            pending.extend(path.iterdir())


def collect_inputs():
    """Use an explicit allowlist; never copy historical outputs or environments."""
    protocol_path = SOURCE / 'runs/confirmatory_v1/protocol.json'
    protocol = read(protocol_path)
    require(len(protocol['jobs']) == 384, 'Unexpected frozen comparison size')
    files = {}
    for relative, expected in protocol['source_sha256'].items():
        source = (SOURCE / relative).resolve()
        require(source.is_relative_to(REPO), f'Frozen source escapes repository: {relative}')
        require(sha(source) == expected, f'Frozen source changed: {relative}')
        files[source.relative_to(REPO).as_posix()] = expected
    additional = [protocol_path, SOURCE / 'analysis_plan.md', SOURCE / 'analyze_study.py']
    additional += list((SOURCE / protocol['bank']).glob('*.json'))
    additional += [SOURCE / 'followup_v260' / name for name in
                   ('stress_evaluate.py', 'budget_sensitivity.py', 'audit_budget.py')]
    for source in additional:
        require(source.is_file(), f'Required input is missing: {source}')
        files[source.relative_to(REPO).as_posix()] = sha(source)
    return protocol, files


def prepare(output, smoke):
    from .fixed_environment import identity
    environment_sha = identity()
    output = check_output(output)
    reject_redirects(output)
    protocol, inputs = collect_inputs()
    marker = output / 'reproduction_manifest.json'
    tree = output / 'worktree/repo'
    if marker.exists():
        manifest = read(marker)
        require(manifest['schema'] == 1 and manifest['smoke'] == smoke,
                'Output belongs to a different mode. Use a separate --output for smoke/full.')
        require(manifest['inputs_sha256'] == inputs,
                'Input code/protocol changed since this output was created. Use a new output.')
        require(manifest.get('environment_lock_sha256') == environment_sha,
                'External comparison environment changed. Use a new output.')
    else:
        require(not output.exists() or not any(output.iterdir()),
                'New output must be empty; existing files are never overwritten.')
        output.mkdir(parents=True, exist_ok=True)
        manifest = {'schema': 1, 'created_utc': now(), 'smoke': smoke,
                    'environment_lock_sha256': environment_sha,
                    'source_repository': str(REPO), 'worktree': str(tree),
                    'inputs_sha256': inputs,
                    'expected_full': {'comparison_plans': 384, 'stress_paths': 384,
                                      'stress_scales_per_path': 2, 'budget_plans': 288},
                    'scope': 'Fresh numerical reproduction on original fixed maps/seeds; '
                             'time-limited planner outcomes and timings can vary by machine. '
                             'No archived result records are copied.'}
        dump(marker, manifest)
    for relative, expected in inputs.items():
        destination = tree / relative
        if destination.exists():
            require(sha(destination) == expected, f'Isolated immutable input changed: {relative}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, destination)
    experiment = tree / RELATIVE_EXPERIMENT
    frozen_environment = experiment / 'runs/confirmatory_v1/environment_freeze.txt'
    if not frozen_environment.exists():
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--disable-pip-version-check'],
                                capture_output=True, text=True, env=ENVIRONMENT, check=True)
        frozen_environment.write_text(result.stdout, encoding='utf-8')
    return experiment, protocol


class OutputLock:
    """Prevent concurrent controllers; retain dead-run evidence for resume."""
    def __init__(self, output):
        self.path = output / '.external.lock'

    def __enter__(self):
        import psutil
        if self.path.exists():
            previous = read(self.path)
            for key in ('pid', 'child_pid'):
                pid = previous.get(key)
                require(not pid or not psutil.pid_exists(pid),
                        f'Another controller/worker is active (PID {pid}); output is locked.')
            needle = str(self.path.parent / 'worktree').casefold()
            for process in psutil.process_iter(['pid', 'cmdline']):
                try:
                    command = ' '.join(process.info['cmdline'] or []).casefold()
                    require(needle not in command,
                            f'An orphan worker still uses this output (PID {process.pid}); wait before resuming.')
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
            self.path.unlink()
        with self.path.open('x', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid(), 'child_pid': None, 'created_utc': now()}, stream)
        return self

    def child(self, pid):
        value = read(self.path)
        value['child_pid'] = pid
        dump(self.path, value)

    def __exit__(self, *unused):
        import psutil
        needle = str(self.path.parent / 'worktree').casefold()
        for process in psutil.process_iter(['pid', 'cmdline']):
            try:
                if needle in ' '.join(process.info['cmdline'] or []).casefold():
                    # Retain the lock when a child could not be terminated.
                    return
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
        self.path.unlink(missing_ok=True)


def terminate_tree(process):
    """Terminate and wait for descendants too, including an in-flight planner."""
    import psutil
    try:
        parent = psutil.Process(process.pid)
        targets = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=3)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    require(not alive, 'A child process is still alive; do not resume this output yet.')
    process.wait(timeout=3)


def execute(experiment, lock, script, *arguments):
    command = [sys.executable, str(experiment / script), *map(str, arguments)]
    print('RUN:', subprocess.list2cmdline(command), flush=True)
    options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
    process = subprocess.Popen(command, cwd=experiment, env=ENVIRONMENT, **options)
    lock.child(process.pid)
    stop = None
    if script == 'run_study.py' and '--run' in arguments:
        study = arguments[arguments.index('--run') + 1]
        stop = experiment / 'runs' / study / 'STOP'
    elif script == 'followup_v260/budget_sensitivity.py' and 'run' in arguments:
        stop = experiment / 'followup_v260/BUDGET_STOP'
    requested = False
    created_stop = False
    stop_started = None
    try:
        while process.poll() is None:
            try:
                if any((base / 'STOP').exists() for base in (lock.path.parent, lock.path.parent.parent)):
                    requested = True
                if requested:
                    if stop is None:
                        terminate_tree(process)
                        break
                    if stop_started is None:
                        stop_started = time.monotonic()
                        if not stop.exists():
                            stop.touch()
                            created_stop = True
                        print('Pause requested; finishing the current worker and its receipt.', flush=True)
                    elif time.monotonic() - stop_started > 130:
                        terminate_tree(process)
                        break
                time.sleep(.2)
            except KeyboardInterrupt:
                if requested:
                    terminate_tree(process)
                    break
                requested = True
        code = process.wait()
    except BaseException:
        terminate_tree(process)
        raise
    finally:
        if created_stop:
            stop.unlink(missing_ok=True)
        lock.child(None)
    if requested:
        raise Paused('Paused. Remove output/STOP if present and rerun the same command to resume.')
    require(code == 0, f'Command failed ({code}); inspect this output and rerun to resume.')


def check_resume_records(experiment, study, budget=False):
    """Do not infer success from a path file or rerun already observed outcomes."""
    directory = experiment / 'runs' / study
    receipt_path = (experiment / 'followup_v260/budget_process_log.jsonl') if budget else directory / 'process_log.jsonl'
    receipts = [json.loads(x) for x in receipt_path.read_text().splitlines() if x.strip()] if receipt_path.exists() else []
    if budget:
        receipts = [r for r in receipts if r['study'] == study]
    groups = {}
    for row in receipts:
        groups.setdefault(row['id'], []).append(row)
    for path in (directory / 'results').glob('*.json'):
        attempts = groups.get(path.stem, [])
        successes = [r for r in attempts if r['exit_code'] == 0 and not r['hard_timeout']]
        require(len(successes) == 1 and attempts[-1] == successes[0],
                f'{study}/{path.stem}: a result exists without one final successful process receipt. '
                'Evidence is preserved; do not delete, fabricate a receipt, or silently rerun this outcome.')
    for name, attempts in groups.items():
        path = directory / 'results' / (name + '.json')
        successes = [r for r in attempts if r['exit_code'] == 0 and not r['hard_timeout']]
        require(not successes or path.exists(), f'Successful result was removed: {study}/{name}')
        failed = [r for r in attempts if r['exit_code'] != 0 or r['hard_timeout']]
        if failed and not successes:
            # Preserve any failed process log before the frozen comparison controller overwrites it.
            log = directory / 'logs' / (name + '.log')
            saved = experiment / 'execution_failures' / study / (name + f'_attempt{len(attempts)}.log')
            if log.exists() and not saved.exists():
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(log, saved)
            require(not budget, 'A budget worker failed before producing a result. Raw evidence is preserved. '
                    'The frozen budget audit does not authorize arbitrary failed-attempt retries; '
                    'inspect the execution failure before starting a new output.')
    # An interrupted worker can leave a bit file before its atomic result JSON.
    # Preserve that incomplete artifact and its log; never let it masquerade as
    # the new attempt's output when the new attempt returns no path.
    for bitpath in (directory / 'results').glob('*_external_bits.npz'):
        name = bitpath.name.removesuffix('_external_bits.npz')
        if not (directory / 'results' / (name + '.json')).exists():
            target = experiment / 'execution_failures' / study / (name + '_incomplete_' + sha(bitpath)[:12] + '.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(sha(target) == sha(bitpath), 'Incomplete artifact archive differs')
                bitpath.unlink()
            else:
                bitpath.replace(target)
            dump(target.with_suffix('.json'), {'id': name, 'study': study, 'preserved_utc': now(),
                 'reason': 'Packed bits existed without a completed result JSON; no successful outcome is discarded.',
                 'sha256': sha(target), 'retry': 'Same fixed job, without parameter or outcome-based selection.'})
    for log in (directory / 'logs').glob('*.log'):
        if not (directory / 'results' / (log.stem + '.json')).exists():
            target = experiment / 'execution_failures' / study / (log.stem + '_incomplete_' + sha(log)[:12] + '.log')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(log, target)


def validate_records(experiment, study, jobs):
    """Validate genuine receipts, identities and packed counts for a subset/full set."""
    import numpy as np
    directory = experiment / 'runs' / study
    protocol_hash = sha(directory / 'protocol.json')
    receipts_file = directory / 'process_log.jsonl'
    receipts = [json.loads(line) for line in receipts_file.read_text().splitlines() if line.strip()]
    for job in jobs:
        record = read(directory / 'results' / (job['id'] + '.json'))
        require(record['job'] == job and record['protocol_sha256'] == protocol_hash,
                f'Record identity changed: {study}/{job["id"]}')
        successful = [r for r in receipts if r['id'] == job['id'] and r['exit_code'] == 0
                      and not r['hard_timeout']]
        require(len(successful) == 1, f'Expected one actual successful execution: {job["id"]}')
        metrics = record['metrics']
        if metrics['external_samples']:
            with np.load(directory / 'results' / (job['id'] + '_external_bits.npz')) as packed:
                require(packed['0.6'].shape == (2500,), 'External sample count changed')
                value = float(np.unpackbits(packed['0.6'], bitorder='little').mean())
            require(abs(value - metrics['cfr_0.6']) < 1e-14, 'CFR differs from packed samples')
        require(metrics['joint_cfr_0.6'] == (metrics['cfr_0.6'] if metrics['nominal_success'] else 0.),
                'Failure handling changed')


def comparison(experiment, protocol, smoke, lock):
    study = experiment / 'runs/confirmatory_v1'
    check_resume_records(experiment, 'confirmatory_v1')
    if smoke:
        jobs = [next(j for j in protocol['jobs'] if j['method'] == method) for method in METHODS]
        for job in jobs:
            if not (study / 'results' / (job['id'] + '.json')).exists():
                execute(experiment, lock, 'run_study.py', '--run', 'confirmatory_v1',
                        '--methods', job['method'], '--limit', '1')
        validate_records(experiment, 'confirmatory_v1', jobs)
        return jobs
    # Do not rewrite completed study status/environment after a follow-up has bound its hashes.
    expected = {j['id'] for j in protocol['jobs']}
    found = {p.stem for p in (study / 'results').glob('*.json')}
    require(found <= expected, 'Unexpected comparison records')
    if found != expected:
        execute(experiment, lock, 'run_study.py', '--run', 'confirmatory_v1')
        found = {p.stem for p in (study / 'results').glob('*.json')}
        require(found == expected, 'Comparison paused or incomplete. Resolve memory/STOP and resume.')
    execute(experiment, lock, 'analyze_study.py', '--study', 'confirmatory_v1')
    return protocol['jobs']


def smoke_stress(experiment, protocol, jobs):
    """Execute the same stress scorer without pretending its 384-path audit passed."""
    import importlib.util
    import numpy as np
    follow = experiment / 'followup_v260'
    spec = importlib.util.spec_from_file_location('external_smoke_stress', follow / 'stress_evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL = follow / 'smoke_stress_protocol.json'
    module.OUT = follow / 'smoke_stress_results'
    settings = {'mode': 'smoke_only_not_a_complete_experiment', 'external_samples': 20000,
                'external_root': 202610150901, 'sigmas': [.6, .9], 'curve_tolerance': .002,
                'jobs': jobs, 'expected_paths': len(jobs)}
    if module.PROTOCOL.exists():
        require(read(module.PROTOCOL) == settings, 'Smoke stress settings changed')
    else:
        dump(module.PROTOCOL, settings)
    for job in jobs:
        path = module.OUT / 'paths' / (job['id'] + '.json')
        if not path.exists():
            module.score_one(job, settings, protocol)
        result = read(path)
        bitpath = module.OUT / 'paths' / (job['id'] + '_bits.npz')
        require(result['followup_protocol_sha256'] == sha(module.PROTOCOL), 'Smoke protocol mismatch')
        require(result['bits_sha256'] == sha(bitpath), 'Smoke packed bits changed')
        require(result['original_result_sha256'] == sha(module.STUDY / 'results' / (job['id']+'.json')),
                'Smoke input path changed')
        with np.load(bitpath) as packed:
            for sigma in ('0.6', '0.9'):
                require(packed[sigma].shape == (2500,), 'Smoke stress sample count changed')
                actual = float(np.unpackbits(packed[sigma], bitorder='little').mean())
                require(abs(actual - result['metrics']['cfr_' + sigma]) < 1e-14, 'Smoke stress CFR mismatch')
    dump(module.OUT / 'smoke_validation.json', {'passed': True, 'paths_executed': len(jobs),
         'full_experiment': False, 'full_expected_paths': 384, 'samples_per_path_per_scale': 20000})


def stress(experiment, protocol, jobs, smoke, lock):
    if smoke:
        smoke_stress(experiment, protocol, jobs)
        return
    script = 'followup_v260/stress_evaluate.py'
    if not (experiment / 'followup_v260/stress_protocol.json').exists():
        execute(experiment, lock, script, '--prepare')
    execute(experiment, lock, script, '--run')


def budget(experiment, smoke, lock):
    script = 'followup_v260/budget_sensitivity.py'
    follow = experiment / 'followup_v260'
    if not (follow / 'budget_protocol.json').exists():
        execute(experiment, lock, script, 'prepare')
    if smoke:
        for seconds in (1, 3):
            study = f'budget_v260_{seconds}s'
            settings = read(experiment / 'runs' / study / 'protocol.json')
            check_resume_records(experiment, study)
            for method in ('cc_vpsto', 'saa'):
                job = next(j for j in settings['jobs'] if j['method'] == method)
                if not (experiment / 'runs' / study / 'results' / (job['id'] + '.json')).exists():
                    execute(experiment, lock, 'run_study.py', '--run', study, '--methods', method, '--limit', '1')
                validate_records(experiment, study, [job])
        dump(follow / 'smoke_budget_validation.json', {'passed': True, 'plans_executed': 4,
             'full_experiment': False, 'full_expected_plans': 288})
        return
    for seconds in (1, 3):
        check_resume_records(experiment, f'budget_v260_{seconds}s', budget=True)
    execute(experiment, lock, script, 'run')
    require(read(follow / 'budget_status.json')['status'] == 'complete',
            'Budget study paused. Resolve memory/BUDGET_STOP and resume.')
    execute(experiment, lock, script, 'analyze')
    execute(experiment, lock, 'followup_v260/audit_budget.py')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true', help='Four comparison + four budget plans; incomplete by design.')
    parser.add_argument('--phase', choices=('all', 'comparison', 'stress', 'budget'), default='all')
    args = parser.parse_args(argv)
    output = check_output(args.output.expanduser())
    experiment, protocol = prepare(output, args.smoke)
    with OutputLock(output) as lock:
        dump(output / 'status.json', {'status': 'running', 'phase': args.phase, 'smoke': args.smoke,
                                      'updated_utc': now(), 'worktree': str(experiment)})
        try:
            if any((base / 'STOP').exists() for base in (output, output.parent)):
                raise Paused('Remove output/STOP before resuming.')
            # Stress/budget need this output's own fresh comparison, never archived records.
            jobs = comparison(experiment, protocol, args.smoke, lock)
            if args.phase in ('all', 'stress'):
                stress(experiment, protocol, jobs, args.smoke, lock)
            if args.phase in ('all', 'budget'):
                budget(experiment, args.smoke, lock)
        except BaseException as error:
            dump(output / 'status.json', {'status': 'paused' if isinstance(error, Paused) else 'interrupted_or_failed', 'phase': args.phase,
                 'smoke': args.smoke, 'updated_utc': now(), 'error': str(error),
                 'resume': 'Run the identical command. Inspect frozen worker logs before retrying execution failures.'})
            raise
        dump(output / 'status.json', {'status': 'smoke_passed' if args.smoke else 'requested_phase_complete',
             'phase': args.phase, 'smoke': args.smoke, 'updated_utc': now(),
             'comparison_plans': len(jobs), 'stress_paths': len(jobs) if args.phase in ('all', 'stress') else 0,
             'budget_plans': (4 if args.smoke else 288) if args.phase in ('all', 'budget') else 0,
             'full_experiment': not args.smoke and args.phase == 'all', 'worktree': str(experiment)})
    print('Outputs:', output, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Paused as error:
        print(str(error), flush=True)
        raise SystemExit(130)
