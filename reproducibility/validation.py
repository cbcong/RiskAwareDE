"""Independent 20,000-world reassessment and the fixed-work timing cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/reference"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


from reproducibility.atomic_io import write_json as write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pause_requested(output):
    return any((p / "STOP").exists() for p in (output, output.parent, output.parent.parent))


def precision(output, smoke=False, cohort=None):
    import numpy as np
    from .fixed_environment import identity
    from .reporting import require_complete
    require_complete(cohort, smoke)
    engine = ROOT / "reproducibility/internal/precision_engine.py"
    if digest(engine) != read(REFERENCE / "high_precision/protocol.json")["script_sha256_at_freeze"]:
        raise RuntimeError("Independent precision evaluator changed")
    spec = importlib.util.spec_from_file_location("precision_engine", engine)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    experiment.HERE, experiment.SOURCE, experiment.REPO = output, Path(cohort), ROOT
    # Only file persistence is supplied by the shared robust writer. The
    # independent collision predicate and statistics are the frozen functions.
    experiment.dump = write
    output.mkdir(parents=True, exist_ok=True)
    if pause_requested(output):
        raise SystemExit(130)
    contract = dict(environment_lock_sha256=identity(), cohort=str(Path(cohort).resolve()),
                    cohort_contract_sha256=digest(Path(cohort)/"reproduction.json"), smoke=smoke,
                    source_result_sha256={p.name:digest(p) for p in sorted((Path(cohort)/"results").glob("*.json"))})
    if (output/"source_contract.json").exists() and read(output/"source_contract.json") != contract:
        raise RuntimeError("Precision input cohort or environment changed")
    write(output/"source_contract.json", contract)
    if smoke:
        protocol = read(Path(cohort)/"protocol.json")
        selected_ids=set(read(Path(cohort)/"reproduction.json")["job_ids"])
        job = next(j for j in read(Path(cohort)/"jobs.json") if j["method"]=="full" and j["id"] in selected_ids)
        scene = next(s for s in protocol["scenarios"][str(job["map_root"])] if s["name"]==job["scene"])
        record = read(Path(cohort)/"results"/(job["id"]+".json"))
        shifts = experiment.bank_for(scene)
        free = experiment.collision_free(record["detail"]["waypoints"], scene, shifts, protocol["benchmark_config"]["safety_margin"])
        from seven_category_protocol import _scene_and_noise
        from run_frozen_stress_suite import collision_free_batch
        from uav_path_planning.scenarios import Scenario3D
        from uav_path_planning.models import Path3D
        points = Path3D(record["detail"]["waypoints"])
        check_indices = [0,1,117,10001,19999]
        expected = collision_free_batch(points,Scenario3D.from_dict(scene),shifts[check_indices],protocol["benchmark_config"]["safety_margin"])
        np.testing.assert_array_equal(free[check_indices], expected)
        if not record["row"]["nominal_success"]:
            free[:] = False
        report = dict(mode="smoke", paths=1, worlds_per_path=20000, cfr=float(free.mean()),
                      production_predicate_checks_passed=len(check_indices),source_result_sha256=digest(Path(cohort)/"results"/(job["id"]+".json")))
        write(output/"smoke_verification.json",report)
        print(json.dumps(report),flush=True)
        return
    if not (output/"protocol.json").exists():
        experiment.prepare()
    inputs = read(output/"frozen_inputs.json")
    for record in inputs["records"]:
        if digest(Path(cohort)/"results"/record["source_file"]) != record["source_sha256"]:
            raise RuntimeError("New P6 path changed after precision inputs were fixed")
    if not (output/"validation_pilot.json").exists():
        experiment.validate_pilot()
    if not (output/"completion.json").exists():
        experiment.run()
    for name,expected in read(output/"completion.json")["files"].items():
        if digest(output/name) != expected:
            raise RuntimeError("Precision output changed: "+name)
    # Describe historical differences without forcing new paths to reproduce
    # old counts. All paths and all 20,000 draws are retained.
    changed=[]
    with np.load(REFERENCE/"high_precision/collision_free_bits.npz") as old, np.load(output/"collision_free_bits.npz") as new:
        if set(old.files)!=set(new.files):
            raise AssertionError("Precision path identity set differs from the fixed design")
        changed=[key for key in new.files if not np.array_equal(new[key],old[key])]
    report=dict(mode="full",paths=1440,worlds_per_path=20000,source_cohort=str(cohort),
                changed_bit_arrays_vs_historical=len(changed),environment_lock_sha256=identity())
    write(output/"reproduction_verification.json",report)
    print(json.dumps(report),flush=True)


def runtime(output, smoke=False):
    import numpy as np
    source = REFERENCE / "runtime"
    manifest = read(source / "SOURCE_MANIFEST.json")
    verified = 0
    for relative, expected in manifest["files"].items():
        if relative.startswith("frozen_code/"):
            candidate = ROOT / relative.removeprefix("frozen_code/")
        elif relative.startswith("configs/") or relative.startswith("reference/"):
            candidate = source / Path(relative).name
        else:
            continue  # old launcher is replaced by this orchestration layer
        if digest(candidate) != expected:
            raise RuntimeError(f"Runtime input/source mismatch: {relative}")
        verified += 1
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    from seven_category_protocol import execute_job, minimalvalidate_result
    from .fixed_environment import identity as environment_identity
    protocol = read(source / "runtime_protocol.json")
    schedule = read(source / "job_schedule.json")
    references = read(source / "frozen_outputs.json")
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(mode="smoke" if smoke else "full", protocol_sha256=digest(source / "runtime_protocol.json"),
                    schedule_sha256=digest(source / "job_schedule.json"), environment_lock_sha256=environment_identity())
    if (output / "identity.json").exists() and read(output / "identity.json") != identity:
        raise RuntimeError("Output belongs to a different runtime protocol or mode")
    write(output / "identity.json", identity)
    write(output / "environment.json", dict(python=sys.version, numpy=np.__version__, platform=platform.platform()))
    def execute_checked(job):
        result = execute_job(job, protocol)
        minimalvalidate_result(job, protocol, result)
        actual = dict(row={k:v for k,v in result["row"].items() if k not in ["runtime_sec", "cpu_seconds"]},
                      waypoints=result["detail"]["waypoints"], internal_risk=result["detail"]["internal_risk"],
                      external_trials=result["external_trials"], distribution_external_trials=result["distribution_external_trials"])
        matches = json.loads(json.dumps(actual)) == references[job["id"]]
        return dict(row=dict(result["row"], id=job["id"], exact_reference_match=matches), scientific=actual)
    jobs = [next(j for j in schedule if j["id"] == "r202610101835_m00_full_s0009")] if smoke else schedule
    records = output / "records"
    records.mkdir(exist_ok=True)
    def seal(payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    def verified_record(job):
        payload = read(records / f'{job["id"]}.json')
        expected_hash = payload.pop("checkpoint_sha256")
        if seal(payload) != expected_hash:
            raise RuntimeError(f"Runtime checkpoint changed: {job['id']}")
        row = payload["row"]
        if row["id"] != job["id"]:
            raise RuntimeError("Runtime checkpoint identity mismatch")
        for key, value in payload["scientific"]["row"].items():
            if row[key] != value:
                raise RuntimeError(f"Runtime metric mismatch: {job['id']}/{key}")
        return row
    for job in jobs:
        if (records / f'{job["id"]}.json').exists():
            verified_record(job)
    pending = [j for j in jobs if not (records / f'{j["id"]}.json').exists()]
    if pause_requested(output):
        raise SystemExit(130)
    if pending and not smoke:
        warmups = []
        for index in (0, 8, 16):
            for method in protocol["methods"]:
                job = next(j for j in schedule if j["id"] == f"r202610101835_m{index:02d}_{method}_s0009")
                warmups.append(execute_checked(job)["row"])
        write(output / f"warmups_excluded_{time.time_ns()}.json", warmups)
    started = time.perf_counter()
    for index, job in enumerate(pending, 1):
        if pause_requested(output):
            raise SystemExit(130)
        payload = execute_checked(job)
        payload["checkpoint_sha256"] = seal(payload)
        write(records / f'{job["id"]}.json', payload)
        if smoke or index % 3 == 0:
            print(f"runtime: {len(jobs)-len(pending)+index}/{len(jobs)}; validated fixed-environment results", flush=True)
    rows = [verified_record(j) for j in jobs]
    with (output / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = dict(status="complete", mode=identity["mode"], measured_plans=len(rows),
                  exact_reference_match=all(r["exact_reference_match"] for r in rows), source_hashes_verified=verified,
                  seconds_this_invocation=time.perf_counter()-started, note="Runtime varies with hardware/load; 9 warmups are excluded on each resumed invocation with pending jobs.")
    write(output / "completion.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "precision", "runtime", "statistics"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    from .fixed_environment import check
    from .reporting import cohort_tables
    check()
    destination = args.output.resolve()
    if destination == ROOT or ROOT.is_relative_to(destination) or (destination.is_relative_to(ROOT) and not destination.is_relative_to(ROOT / "outputs")):
        parser.error("Use a new output below outputs/ or outside the repository")
    if args.phase in ("all", "precision"):
        precision(args.output.resolve() / "precision", args.smoke, args.run_root / "p6")
    if args.phase in ("all", "runtime"):
        runtime(args.output.resolve() / "runtime", args.smoke)
    if args.phase in ("all", "statistics"):
        if args.smoke:
            write(args.output / "statistics/smoke_scope.json",dict(status="smoke_only",formal_statistics=False))
            return
        tables = args.run_root / "statistics_inputs"
        cohort_tables(args.run_root,tables)
        statistics = args.output.resolve() / "statistics"
        statistics.mkdir(parents=True, exist_ok=True)
        with (statistics / "analysis.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, str(ROOT / "reproducibility/internal/analyze_frozen_robustness.py"),
                            "--data", str(tables), "--output", str(statistics)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("Statistical robustness: all 106 prescribed contrasts recomputed from this run; details in " + str(statistics), flush=True)


if __name__ == "__main__":
    main()
