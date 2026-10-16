"""Run every task in the single locked environment. Historical runtime routing is archived."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_runtime_plan(jobs):
    from .fixed_environment import check, identity
    import numpy
    import platform
    check()
    environment = dict(executable=sys.executable, implementation=platform.python_implementation(),
                       python_version=list(sys.version_info[:3]), numpy_version=numpy.__version__)
    return ({"schema": 2, "jobs": {j["id"]: "fixed" for j in jobs},
             "environment_lock_sha256": identity(), "job_counts": {"fixed": len(jobs)}},
            {"fixed": environment})


class NativeWorker:
    """Persistent JSON-lines process; the scientific code is never monkey-patched."""

    def __init__(self, environment, protocol_path, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self.log = log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [environment["executable"], "-u", "-m", "reproducibility.runtime_worker",
             "--protocol", str(protocol_path)], cwd=ROOT, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8",
            env=dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="1",
                     OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"))
        try:
            response = self.receive()
            if response.get("environment") != environment:
                raise RuntimeError("Native worker did not start in the verified environment")
            self.environment = response["environment"]
        except BaseException:
            self.close()
            raise

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Native worker exited unexpectedly; inspect {self.log_path}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"Native worker failed: {response.get('error')}; inspect {self.log_path}")
        return response

    def execute(self, job):
        self.process.stdin.write(json.dumps(job, allow_nan=False) + "\n")
        self.process.stdin.flush()
        return self.receive()["bundle"]

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        if self.process.stdout:
            self.process.stdout.close()
        self.log.close()


class RuntimePool:
    """One active native worker per calling thread; idle runtimes can be reused."""

    def __init__(self, plan, environments, protocol_path, log_directory):
        self.plan, self.environments = plan, environments
        self.protocol_path, self.log_directory = Path(protocol_path), Path(log_directory)
        self.workers = {}
        self.lock = threading.Lock()

    def __enter__(self):
        return self

    def execute(self, job):
        runtime = self.plan["jobs"][job["id"]]
        key = (threading.get_ident(), runtime)
        with self.lock:
            if key not in self.workers:
                self.workers[key] = NativeWorker(self.environments[runtime], self.protocol_path,
                    self.log_directory / f"{runtime}-{key[0]}.log")
            worker = self.workers[key]
        return worker.execute(job)

    def __exit__(self, *args):
        for worker in self.workers.values():
            worker.close()
