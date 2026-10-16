"""One isolated, version-locked environment for all new experimental results."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "environment.lock.json"
DEFAULT_RUN = ROOT / "outputs/fixed_env_20261016"
THREADS = {name: "1" for name in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect():
    prefix = Path(sys.prefix).resolve()
    config = prefix / "pyvenv.cfg"
    if prefix != (ROOT / ".venv").resolve() or not config.is_file():
        raise RuntimeError("Use this repository's .venv/Scripts/python.exe (or a root run_*.py launcher).")
    if "include-system-site-packages = false" not in config.read_text(encoding="utf-8"):
        raise RuntimeError("The fixed environment must not inherit global site-packages.")
    packages = {}
    for line in (ROOT / "requirements.lock.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        distribution = metadata.distribution(name)
        if distribution.version != expected or not Path(distribution.locate_file("")).resolve().is_relative_to(prefix):
            raise RuntimeError(f"Fixed environment dependency changed or escaped .venv: {name}")
        packages[name] = distribution.version
    return {"schema": 1, "python_version": list(sys.version_info[:3]),
            "implementation": platform.python_implementation(), "architecture": platform.machine(),
            "system": platform.system(), "python_build": list(platform.python_build()),
            "base_executable_sha256": sha(sys._base_executable),
            "requirements_sha256": sha(ROOT / "requirements.lock.txt"),
            "isolated_site_packages": True, "packages": packages, "thread_limits": THREADS}


def check():
    actual = inspect()
    expected = json.loads(LOCK.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("Experimental environment differs from environment.lock.json; restore the pinned environment.")
    return actual


def identity():
    check()
    return sha(LOCK)


def subprocess_environment():
    return dict(os.environ, **THREADS, PYTHONUTF8="1", PYTHONNOUSERSITE="1")
