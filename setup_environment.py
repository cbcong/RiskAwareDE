"""Create the single pinned environment locally; never copy another PC's venv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, help="Optional directory of downloaded wheels for an offline installation")
    args = parser.parse_args()
    expected = json.loads((ROOT / "environment.lock.json").read_text(encoding="utf-8"))
    required = tuple(expected["python_version"])
    if (sys.version_info[:3] != required or platform.python_implementation() != "CPython"
            or platform.system() != "Windows" or platform.machine() != "AMD64"):
        parser.error("Install official CPython 3.12.4 for Windows x64, then run: py -3.12 setup_environment.py. Only one Python environment is needed.")
    from prepare_data import check as check_data
    try:
        check_data(ROOT)
    except ValueError as exc:
        parser.error(str(exc))
    bundled_wheels = ROOT / "vendor/wheels"
    wheels = args.wheelhouse.resolve() if args.wheelhouse else (bundled_wheels if bundled_wheels.is_dir() else None)
    if wheels is not None and not wheels.is_dir():
        parser.error(f"Wheel directory not found: {wheels}")
    preferred = ROOT / ".venv/Scripts/python.exe"
    pending = ROOT / ".venv/.setup_incomplete"
    if not preferred.exists():
        if (ROOT / ".venv").exists():
            parser.error("An incomplete .venv already exists. Move it aside before creating a new environment.")
        subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True, cwd=ROOT)
        pending.write_text("Installation started by setup_environment.py\n", encoding="utf-8")
    if pending.exists():
        command = [str(preferred), "-m", "pip", "--disable-pip-version-check", "install"]
        if wheels is not None:
            command += ["--no-index", "--find-links", str(wheels)]
        command += ["-r", str(ROOT / "requirements.lock.txt")]
        subprocess.run(command, check=True, cwd=ROOT)
    else:
        print("Checking existing project environment; installed packages are not changed.", flush=True)
    subprocess.run([str(preferred), "-m", "pip", "--disable-pip-version-check", "check"], check=True, cwd=ROOT)
    subprocess.run([str(preferred), str(ROOT / "run_all.py"), "--check"], check=True, cwd=ROOT)
    pending.unlink(missing_ok=True)
    print("Ready. Run: .venv/Scripts/python.exe run_all.py --smoke", flush=True)
    print("For the complete study: .venv/Scripts/python.exe run_all.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
