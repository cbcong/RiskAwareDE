"""Shared launcher; public entry files are deliberately small."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.7.0"
from .fixed_environment import DEFAULT_RUN, check as check_environment, identity as environment_identity, subprocess_environment
PARTS = {
    "e": "E: 9,624 frozen-matrix planning jobs",
    "p6": "P6: 3,840 shifted-map planning jobs",
    "diagnostics": "P0-P5: estimator, direction replay, sensitivity and saved-path diagnostics",
    "external": "384 external-comparison plans + 384 two-scale stress evaluations + 288 budget plans",
    "validation": "1,440 paths x 20,000 worlds; 360 measured plans; statistical robustness",
    "figures": "9 main figures + supplementary S1 from this run",
}
GROUPS = {"all": list(PARTS), "e": ["e"], "p": ["p6", "diagnostics"],
          "external": ["external"], "validation": ["validation"], "figures": ["figures"]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


from reproducibility.atomic_io import write_json as write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap():
    preferred = ROOT / ".venv/Scripts/python.exe"
    if not preferred.is_file():
        raise RuntimeError("Missing project .venv. Install official Python 3.12.4 (Windows x64), then run: py -3.12 setup_environment.py. Do not copy another computer's .venv.")
    if preferred.is_file() and Path(sys.executable).resolve() != preferred.resolve():
        return subprocess.call([str(preferred), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]], cwd=ROOT, env=subprocess_environment())
    return None


def check():
    from prepare_data import check as check_data
    check_data(ROOT)
    import importlib
    versions = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "PIL", "cma", "osqp", "shapely", "numba", "psutil"):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "installed")
    from .cohorts import verify_sources
    fixed_environment = check_environment()
    for cohort, total in (("e", 9624), ("p6", 3840)):
        reference = ROOT / "data/reference" / cohort
        verify_sources(reference)
        jobs = read(reference / "jobs.json")
        if len(jobs) != total or len(list((reference / "results").glob("*.json"))) != total:
            raise RuntimeError(f"Incomplete reference cohort: {cohort}")
    from .external import collect_inputs
    collect_inputs()
    manifest = ROOT / "provenance/current_files.json"
    if manifest.exists():
        for relative, expected in read(manifest)["sha256"].items():
            candidate = ROOT / relative
            if not candidate.is_file() or sha(candidate) != expected:
                raise RuntimeError(f"Changed or missing current release file: {relative}")
    print("Environment and frozen input checks passed: " + json.dumps(versions), flush=True)
    print("Fixed environment: Python 3.12.4; isolated .venv; 30 pinned packages; " + environment_identity(), flush=True)
    return versions


def command(part, output, smoke, workers, figure):
    target = output / part
    cmd = [sys.executable, "-m"]
    if part in ("e", "p6"):
        cmd += ["reproducibility.cohorts", "--cohort", part, "--reference", str(ROOT / "data/reference" / part),
                "--output", str(target), "--workers", str(workers)]
    elif part == "diagnostics":
        fresh = output / "e"
        reference = fresh
        cmd += ["reproducibility.diagnostics", "--reference", str(reference), "--output", str(target)]
    elif part == "external":
        cmd += ["reproducibility.external", "--output", str(target)]
    elif part == "validation":
        cmd += ["reproducibility.validation", "--output", str(target), "--run-root", str(output)]
    else:
        cmd += ["reproducibility.reporting", "--run-root", str(output), "--figure", figure]
    if smoke:
        cmd.append("--smoke")
    return cmd


def main(group="all"):
    forwarded = bootstrap()
    if forwarded is not None:
        return forwarded
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=f"RiskAwareDE: {group}. Default performs real experiments; repeat the command to resume.")
    parser.add_argument("--output", type=Path, help="Shared run directory (default outputs/fixed_env_20261016, or outputs/fixed_env_smoke)")
    parser.add_argument("--smoke", action="store_true", help="Small real computation; never a full reproduction")
    parser.add_argument("--plan", action="store_true", help="Print selected stages without computing")
    parser.add_argument("--check", action="store_true", help="Check dependencies, sources and evidence, without experiments")
    parser.add_argument("--status", action="store_true", help="Print existing status without starting")
    parser.add_argument("--workers", type=int, default=1, help="E/P6 workers only; timed comparisons remain serial")
    parser.add_argument("--only", nargs="+", choices=list(PARTS), help="Select stages within this launcher")
    parser.add_argument("--figure", choices=["all", "s1", *map(str, range(1,10))], default="all")
    args = parser.parse_args()
    stages = args.only or GROUPS[group]
    if not set(stages) <= set(GROUPS[group]):
        parser.error("--only must belong to this launcher; use run_all.py for any stage")
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = (args.output or (ROOT / "outputs/fixed_env_smoke" if args.smoke else DEFAULT_RUN)).resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "outputs"):
        parser.error("Within the repository, --output must be under outputs/")
    if output == ROOT / "outputs":
        parser.error("Use a named run below outputs/, e.g. outputs/full")
    if args.plan:
        print(f"Mode: {'SMOKE' if args.smoke else 'FULL'}; output: {output}")
        for part in stages:
            print(f"  {part}: {PARTS[part]}")
        print("Full reruns use the fixed paper maps and seeds. Timing-limited external planners can produce different outcomes.")
        return 0
    if args.status:
        if (output / "pipeline.json").exists():
            print(json.dumps(read(output / "pipeline.json"), ensure_ascii=False, indent=2))
        else:
            print("This run has not started.")
        for part in stages:
            for name in ("state.json", "status.json", "completion_receipt.json", "reproduction_completion.json"):
                file = output / part / name
                if file.exists():
                    print(part, name, json.dumps(read(file), ensure_ascii=False)[:1600])
                    break
        return 0
    check()
    if args.check:
        return 0
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(version=VERSION, smoke=args.smoke, execution_policy="single-environment-v1", environment_lock_sha256=environment_identity())
    marker = output / "pipeline_identity.json"
    from scripts.seven_category_control import FileLock
    with FileLock(output / "pipeline.lock"):
        if marker.exists() and read(marker) != contract:
            raise RuntimeError("Different version/mode already uses this output. Select a new --output.")
        write(marker, contract)
        status_path = output / "pipeline.json"
        status = read(status_path) if status_path.exists() else dict(version=VERSION, smoke=args.smoke, stages={})
        env = subprocess_environment()
        # Stage runners validate their own checkpoints even when marked complete.
        for part in stages:
            if (output / "STOP").exists():
                status.update(status="paused", reason="STOP file exists")
                write(status_path, status)
                print("Paused: delete the STOP file, then repeat this command to resume.")
                return 130
            cmd = command(part, output, args.smoke, args.workers, args.figure)
            status.update(status="running", current_stage=part, updated_utc=datetime.now(timezone.utc).isoformat())
            status["stages"][part] = dict(status="running", command=cmd)
            write(status_path, status)
            print(f"\n>>> {part}: {PARTS[part]}", flush=True)
            try:
                result = subprocess.run(cmd, cwd=ROOT, env=env)
            except KeyboardInterrupt:
                status.update(status="paused", current_stage=part)
                write(status_path, status)
                print("Interrupted. Checkpoints retained; repeat the same command to resume.", flush=True)
                return 130
            status["stages"][part]["exit_code"] = result.returncode
            status["stages"][part]["status"] = "complete_smoke" if args.smoke and result.returncode == 0 else "complete" if result.returncode == 0 else "failed"
            if part == "figures" and args.figure != "all" and result.returncode == 0:
                status["stages"][part].update(status="complete_selected_figure", selected_figure=args.figure)
            if result.returncode:
                status.update(status="paused" if result.returncode == 130 else "failed", current_stage=part, updated_utc=datetime.now(timezone.utc).isoformat())
                write(status_path, status)
                return result.returncode
            write(status_path, status)
        status.update(status="complete_smoke" if args.smoke else "complete_selected_stages", current_stage=None,
                      selected_stages=stages, updated_utc=datetime.now(timezone.utc).isoformat())
        if not args.smoke and set(GROUPS["all"]) <= {k for k,v in status["stages"].items() if v["status"] == "complete"}:
            status["status"] = "complete_full_pipeline"
        write(status_path, status)
        print(f"\n{status['status']}: {output}", flush=True)
    return 0
