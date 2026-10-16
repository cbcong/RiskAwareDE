"""Start, resume, cooperatively stop, and inspect the frozen E0-E6 study.

Each completed planning seed is committed atomically. A run executes its own
source snapshot so later workspace edits cannot silently change resumed work.
No process-name killing is used. Only the selected run's stop flag is written.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = ROOT / 'configs/seven_category_launch.json'
SHANGHAI = timezone(timedelta(hours=8))
ACTIVE_PHASES = {'validating', 'running', 'summarizing'}


def now():
    return datetime.now(SHANGHAI).isoformat(timespec='seconds')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class RunLocked(RuntimeError):
    pass


class FileLock:
    """OS-owned nonblocking lock: a crash releases it without PID guessing."""
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.stream.write(b'0')
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise RunLocked('该实验已有进程运行；不要重复启动。') from exc
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


def is_running(directory):
    try:
        with FileLock(Path(directory)/'worker.lock'):
            return False
    except RunLocked:
        return True


def load_settings(path):
    settings = read_json(path)
    for key in ('output_base', 'control_directory'):
        if key not in settings or not Path(settings[key]).is_absolute():
            raise ValueError(f'{key} 必须是绝对路径')
    if settings.get('workers', 1) != 1:
        raise ValueError('当前入口固定单进程，避免此前内存不足；workers必须为1。')
    return settings


def selected_run(settings, explicit=None):
    if explicit:
        directory = Path(explicit).resolve()
    else:
        pointer = Path(settings['control_directory'])/'latest_run.json'
        if not pointer.exists():
            return None
        directory = Path(read_json(pointer)['run_directory']).resolve()
    if not (directory/'protocol.json').is_file():
        raise ValueError('运行目录不存在或协议不完整：' + str(directory))
    return directory


def append_event(directory, kind, **values):
    with (Path(directory)/'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time': now(), 'event': kind, **values}, ensure_ascii=False, allow_nan=False)+'\n')


def prepare_run(repo, settings, profile='full', new_run=False):
    """Prepare a concrete reproducible run; this function never optimizes."""
    import numpy as np
    from seven_category_protocol import build_protocol, make_jobs
    repo = Path(repo).resolve()
    control = Path(settings['control_directory'])
    with FileLock(control/'launch.lock'):
        previous = selected_run(settings)
        if previous is not None:
            state = read_json(previous/'state.json')
            if is_running(previous) or (state['status'] != 'complete' and not new_run):
                raise RuntimeError('已有未完成实验，请使用“续跑”，避免重复全量计算：' + str(previous))
        protocol = build_protocol(repo, profile=profile)
        jobs = make_jobs(protocol)
        stamp = datetime.now(SHANGHAI)
        suffix = ('定稿主方法七类补齐' if profile == 'manuscript' else
                  '候选全量比较' if profile == 'candidate' else '七类完整比较')
        directory = Path(settings['output_base'])/stamp.strftime('%Y%m%d')/(stamp.strftime('%Y%m%d_%H%M%S')+'_'+suffix+'_'+uuid.uuid4().hex[:6])
        directory.mkdir(parents=True, exist_ok=False)
        (directory/'results').mkdir()
        snapshot = directory/'frozen_code'
        sources = list((repo/'src').rglob('*.py')) + list((repo/'scripts').glob('*.py'))
        sources += list((repo/'configs').glob('*.json'))
        hashes = {}
        for source in sorted(sources):
            relative = source.relative_to(repo)
            target = snapshot/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = digest(target)
        write_json(directory/'protocol.json', protocol)
        write_json(directory/'jobs.json', jobs)
        manifest = {'schema': 1, 'created_at': now(), 'run_directory': str(directory),
            'repository_at_start': str(repo), 'profile': profile, 'expected_plans': len(jobs),
            'python_version': list(sys.version_info[:3]), 'numpy_version': np.__version__,
            'source_sha256': hashes, 'protocol_sha256': digest(directory/'protocol.json'),
            'jobs_sha256': digest(directory/'jobs.json'), 'workers': 1,
            'main_method': protocol['main_method'],
            'analytic_gradient_candidate_included': False,
            'checkpoint_unit': 'one map/method/planner-seed; A* once per map',
            'evidence_scope': ('Both map banks already exposed; candidate profile uses locked soft_beta050 as full and does not use these maps as blind OOD evidence.' if profile == 'candidate' else
                               'Both map banks already exposed; exploratory matched rerun, not new blind testing.')}
        write_json(directory/'execution_manifest.json', manifest)
        state = {'status':'prepared', 'created_at':now(), 'updated_at':now(), 'completed':0,
                 'total':len(jobs), 'completed_by_method':{}, 'current_job':None,
                 'active_seconds':0., 'recent_job_seconds':[], 'attempts':0}
        if protocol.get('reuse'):
            from paper_study_protocol import import_reused_results
            completed, counts = import_reused_results(directory, protocol, jobs)
            state.update(completed=completed, completed_by_method=counts,
                         reused_plans=completed, new_plans_required=len(jobs)-completed)
        write_json(directory/'state.json', state)
        append_event(directory, 'prepared', plans=len(jobs), profile=profile)
        write_json(control/'latest_run.json', {'run_directory':str(directory), 'created_at':now()})
        return directory


def verify_frozen_run(directory):
    import numpy as np
    manifest = read_json(directory/'execution_manifest.json')
    for name, expected in manifest['source_sha256'].items():
        path = directory/'frozen_code'/name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError('冻结源码被修改，拒绝混合版本续跑：' + name)
    for name in ('protocol', 'jobs'):
        if digest(directory/(name+'.json')) != manifest[name+'_sha256']:
            raise RuntimeError('冻结协议/任务清单改变：' + name)
    if tuple(manifest['python_version'][:2]) != sys.version_info[:2] or manifest['numpy_version'] != np.__version__:
        raise RuntimeError('Python主次版本或NumPy版本发生变化，请恢复原环境后续跑。')
    return manifest


def checkpoint_digest(bundle):
    return canonical_digest({k:v for k,v in bundle.items() if k != 'checkpoint_sha256'})


def worker(directory, executor=None, validator=None, summarizer=None, skip_snapshot_check=False):
    """Execute sequentially; dependency injection is only for control tests."""
    directory = Path(directory).resolve()
    if not skip_snapshot_check and Path(__file__).resolve() != (directory/'frozen_code/scripts/seven_category_control.py').resolve():
        raise RuntimeError('worker只能从本实验的冻结源码启动；请使用start或resume固定入口。')
    with FileLock(directory/'worker.lock'):
        state = read_json(directory/'state.json')
        if state['status'] == 'complete':
            print('实验已完成，不重复计算。', flush=True)
            return 0
        state.update(status='validating', pid=os.getpid(), updated_at=now(), current_job=None)
        write_json(directory/'state.json',state)
        clock_start, prior_active = time.monotonic(), state.get('active_seconds',0.)
        try:
            if not skip_snapshot_check:
                verify_frozen_run(directory)
            if executor is None or validator is None:
                from seven_category_protocol import execute_job, minimalvalidate_result
                executor, validator = execute_job, minimalvalidate_result
            protocol, jobs = read_json(directory/'protocol.json'), read_json(directory/'jobs.json')
            if summarizer is None:
                if protocol.get('profile') == 'candidate':
                    from summarize_candidate_run import summarize_run
                else:
                    from summarize_seven_category_run import summarize_run
                summarizer = summarize_run
            if len(jobs) != state['total'] or len({j['id'] for j in jobs}) != len(jobs):
                raise ValueError('冻结任务数量或唯一性错误')
            complete, counts = set(), {}
            for job in jobs:
                file = directory/'results'/(job['id']+'.json')
                if not file.exists():
                    continue
                try:
                    bundle = read_json(file)
                    if bundle.get('checkpoint_sha256') != checkpoint_digest(bundle):
                        raise ValueError('checkpoint checksum differs')
                    validator(job, protocol, bundle)
                except (ValueError, KeyError, TypeError, OSError, AssertionError) as exc:
                    dest = directory/'quarantine'/(uuid.uuid4().hex+'_'+file.name)
                    dest.parent.mkdir(exist_ok=True)
                    shutil.move(str(file),str(dest))
                    append_event(directory, 'invalid_checkpoint_saved_for_inspection', job=job['id'], reason=str(exc), archive=str(dest))
                    continue
                complete.add(job['id'])
                counts[job['method']] = counts.get(job['method'],0)+1
            state.update(status='running', completed=len(complete), completed_by_method=counts,
                         attempts=state.get('attempts',0)+1, error=None)
            recent = deque(state.get('recent_job_seconds',[]), maxlen=30)

            def save_state(**updates):
                state.update(updates, updated_at=now(), active_seconds=prior_active+time.monotonic()-clock_start,
                             recent_job_seconds=list(recent))
                write_json(directory/'state.json',state)

            save_state()
            append_event(directory,'worker_started',pid=os.getpid(),reused=len(complete),total=len(jobs))
            print(f"已保存 {len(complete)}/{len(jobs)} 条；仅执行剩余任务。\n结果目录：{directory}",flush=True)
            for job in jobs:
                if job['id'] in complete:
                    continue
                if (directory/'STOP_REQUESTED.json').exists():
                    save_state(status='paused',current_job=None,pid=None)
                    append_event(directory,'paused',completed=len(complete))
                    print('已安全暂停。已完成路径全部保留，使用续跑文件继续。',flush=True)
                    return 0
                save_state(current_job=job)
                before = time.monotonic()
                bundle = executor(job,protocol)
                validator(job,protocol,bundle)
                bundle['checkpoint_sha256'] = checkpoint_digest(bundle)
                write_json(directory/'results'/(job['id']+'.json'),bundle)
                recent.append(time.monotonic()-before)
                complete.add(job['id'])
                counts[job['method']] = counts.get(job['method'],0)+1
                save_state(completed=len(complete),completed_by_method=counts,current_job=None)
                if len(complete)%25 == 0 or len(complete)==len(jobs):
                    print(f"进度 {len(complete)}/{len(jobs)} ({100*len(complete)/len(jobs):.2f}%)",flush=True)
            if (directory/'STOP_REQUESTED.json').exists():
                save_state(status='paused',current_job=None,pid=None)
                append_event(directory,'paused_before_summary',completed=len(complete))
                print('全部规划已保存；按停跑请求暂停，续跑时仅生成汇总。',flush=True)
                return 0
            save_state(status='summarizing',current_job=None)
            append_event(directory,'summary_started')
            print('全部规划完成，正在生成七类对照表与地图层统计。',flush=True)
            summary = summarizer(directory)
            final_manifest = verify_frozen_run(directory) if not skip_snapshot_check else {}
            source_amendments = final_manifest.get('source_amendments', [])
            write_json(directory/'completion_receipt.json', {'status':'complete','completed_at':now(),
                'plans':len(complete),'protocol_sha256':digest(directory/'protocol.json'),
                'jobs_sha256':digest(directory/'jobs.json'),'summary':summary,
                'source_snapshot_unchanged':not skip_snapshot_check and not source_amendments,
                'source_snapshot_matches_current_manifest':not skip_snapshot_check,
                'source_amendments':source_amendments})
            save_state(status='complete',pid=None,current_job=None)
            append_event(directory,'complete',plans=len(complete))
            print('实验与汇总全部完成：'+str(directory/'analysis'),flush=True)
            return 0
        except KeyboardInterrupt:
            state.update(status='paused',pid=None,current_job=None,updated_at=now(),
                         active_seconds=prior_active+time.monotonic()-clock_start)
            write_json(directory/'state.json',state)
            append_event(directory,'keyboard_interrupt',note='Committed paths retained; uncommitted path reruns on resume.')
            print('已中断；保存过的路径会复用，当前未提交的一条会在续跑时重算。',flush=True)
            return 130
        except BaseException as exc:
            state.update(status='failed',pid=None,error=repr(exc),updated_at=now(),
                         active_seconds=prior_active+time.monotonic()-clock_start)
            write_json(directory/'state.json',state)
            error_path=directory/('error_'+uuid.uuid4().hex[:8]+'.txt')
            error_path.write_text(traceback.format_exc(),encoding='utf-8')
            append_event(directory,'failed',error=repr(exc),details=str(error_path))
            raise


def request_stop(directory):
    state = read_json(directory/'state.json')
    if state['status'] == 'complete':
        return '实验已经完成，无需停跑。'
    write_json(directory/'STOP_REQUESTED.json', {'requested_at':now(),'mode':'finish_current_path_then_pause'})
    if not is_running(directory):
        return '当前没有运行进程。已完成结果保留，使用续跑即可继续。'
    if state['status'] == 'summarizing':
        return '已记录停跑请求；规划均已完成，当前汇总会完成并保存后退出。'
    return '已请求安全停跑：完成并保存当前这一条路径后暂停，不启动下一条。'


def status(directory):
    state = read_json(directory/'state.json')
    active = is_running(directory)
    saved = len(list((directory/'results').glob('*.json')))
    recent = state.get('recent_job_seconds',[])
    label = {'prepared':'已准备，未开始','validating':'核对断点','running':'运行中',
             'paused':'已暂停','failed':'出错，已保存断点','summarizing':'生成汇总中','complete':'全部完成'}.get(state['status'],state['status'])
    if state['status'] in ACTIVE_PHASES and not active:
        label = '运行进程已退出，可断点续跑'
    print('\n实验目录：'+str(directory))
    print(f"状态：{label}；进程{'正在运行' if active else '未运行'}")
    print(f"已保存：{saved}/{state['total']} 条 ({100*saved/state['total']:.2f}%)；续跑会校验这些检查点。")
    if (directory/'STOP_REQUESTED.json').exists() and active:
        print('停跑请求：已提交，等待当前操作安全结束。')
    if state.get('current_job') and active:
        job=state['current_job']
        print(f"当前：{job['method']} / 图库{job['map_root']} / {job['scene']} / 种子{job['seed']}")
    if recent and saved < state['total']:
        seconds=(state['total']-saved)*sum(recent)/len(recent)
        print(f'按最近任务粗估剩余计算 {seconds/60:.1f} 分钟；不同算法速度不同，此值仅供参考。')
    print('最近状态更新时间：'+state['updated_at'])
    for method,count in sorted(state.get('completed_by_method',{}).items()):
        print(f'  {method}: {count} 条已核对保存')
    if state.get('error'):
        print('最近错误：'+state['error'])
    if state['status']=='complete':
        report = '候选全量对照.md' if read_json(directory/'protocol.json').get('profile') == 'candidate' else '七类完整对照.md'
        print('结果表：'+str(directory/'analysis'/report))
    return {'state':state,'running':active,'saved_files':saved}


def launch_worker(directory, resume=False):
    # The short dispatch lock closes the race between two resume/start clicks.
    with FileLock(directory/'dispatch.lock'):
        if is_running(directory):
            raise RunLocked('该实验正在运行；可打开进度文件查看，不需要续跑。')
        if read_json(directory/'state.json')['status']=='complete':
            print('实验已完成，不重复计算。')
            return 0
        if resume:
            stop=directory/'STOP_REQUESTED.json'
            if stop.exists():
                stop.unlink()
        command=[sys.executable,str(directory/'frozen_code/scripts/seven_category_control.py'),
                 'worker','--run-dir',str(directory)]
        process=subprocess.Popen(command)
        try:
            return process.wait()
        except KeyboardInterrupt:
            print(request_stop(directory),flush=True)
            return process.wait()


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8',errors='replace')
        sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','resume','stop','status','worker','plan'])
    parser.add_argument('--settings',type=Path,default=DEFAULT_SETTINGS)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--profile',choices=['full','smoke','candidate','manuscript'],default=None)
    parser.add_argument('--prepare-only',action='store_true',help='Create snapshot/queue without running a planner')
    parser.add_argument('--new-run',action='store_true',help='Explicitly create a new run while preserving an older inactive unfinished run')
    args=parser.parse_args()
    if args.action=='worker':
        if args.run_dir is None:
            parser.error('worker requires --run-dir')
        return worker(args.run_dir)
    settings=load_settings(args.settings)
    args.profile = args.profile or settings.get('default_profile', 'full')
    if args.action=='plan':
        from seven_category_protocol import build_protocol,make_jobs
        p=build_protocol(ROOT,profile=args.profile)
        print(json.dumps({'methods':p['methods'],'categories':p['categories'],
                          'plans':len(make_jobs(p)),'run_started':False},ensure_ascii=False,indent=2))
        return 0
    if args.action=='start':
        directory=prepare_run(ROOT,settings,args.profile,new_run=args.new_run)
        print('已冻结本次配置、地图和源码：'+str(directory),flush=True)
        if args.prepare_only:
            print('仅准备，未执行任何规划。')
            return 0
        return launch_worker(directory)
    directory=selected_run(settings,args.run_dir)
    if directory is None:
        print('尚未启动七类实验。双击01开始文件可创建并执行新实验。')
        return 0
    if args.action=='resume':
        return launch_worker(directory,resume=True)
    if args.action=='stop':
        print(request_stop(directory))
        return 0
    status(directory)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,ImportError) as exc:
        print('未能继续：'+str(exc),file=sys.stderr)
        raise SystemExit(1)
