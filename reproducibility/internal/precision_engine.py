"""Independent external reassessment of frozen P6 paths; no optimizer calls.

First run --prepare, then --validate-pilot, then --run. Archived inputs,
protocol, and bit-packed path outcomes allow portable reanalysis with NumPy.
The source repository is needed for production-predicate validation only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import NormalDist
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(r"D:\WorkSpace\pycharm\uav_path_planning")
SOURCE = Path(r"C:\Users\17613\Desktop\论文5\实验记录\20261010\20261010_191848_OOD七类完整比较_459cdf")
METHODS = ("full", "risk_polish_lshade", "risk_polish_de")
LABELS = {"full": "RiskAwareDE", "risk_polish_lshade": "RT-LS", "risk_polish_de": "RT-DE"}
SAMPLES = 20000
SIGMA = 0.6
TARGET = 0.825
BASE_SEED = 202610142017


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def seed_for(scene):
    payload = json.dumps(["P6-independent-high-precision-v1", BASE_SEED, scene]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def bank_for(scene):
    rng = np.random.Generator(np.random.PCG64(seed_for(scene["name"])))
    return SIGMA * rng.normal(size=(SAMPLES, len(scene["obstacles"]), 3))


def collision_free(points, scene, shifts, margin=0.0):
    """Exact closed axis-aligned-box segment predicate, not point sampling."""
    if points is None:
        return np.zeros(len(shifts), dtype=bool)
    points = np.asarray(points, dtype=float)
    bounds = scene["bounds"]
    if np.any(points < np.asarray(bounds["minimum"])) or np.any(points > np.asarray(bounds["maximum"])):
        return np.zeros(len(shifts), dtype=bool)
    if not scene["obstacles"]:
        return np.ones(len(shifts), dtype=bool)
    lo = np.asarray([o["minimum"] for o in scene["obstacles"]])[None, :, None, :] + shifts[:, :, None, :] - margin
    hi = np.asarray([o["maximum"] for o in scene["obstacles"]])[None, :, None, :] + shifts[:, :, None, :] + margin
    starts = points[:-1][None, None, :, :]
    direction = np.diff(points, axis=0)[None, None, :, :]
    parallel = direction == 0.0
    outside = np.any(parallel & ((starts < lo) | (starts > hi)), axis=-1)
    denominator = np.where(parallel, 1.0, direction)
    first, second = (lo - starts) / denominator, (hi - starts) / denominator
    near = np.where(parallel, -np.inf, np.minimum(first, second))
    far = np.where(parallel, np.inf, np.maximum(first, second))
    enter = np.maximum(0., np.max(near, axis=-1))
    leave = np.minimum(1., np.min(far, axis=-1))
    return ~np.any((~outside) & (enter <= leave), axis=(1, 2))


def prepare():
    if (HERE / "protocol.json").exists():
        raise RuntimeError("Protocol already frozen; refusing to overwrite it")
    p = read(SOURCE / "protocol.json")
    assert p["main_method"]["options"]["beta"] == .05
    assert p["external_samples"] == 500 and p["target_alpha"] == .175
    roots = p["map_roots"]
    assert len(roots) == 1
    scenes = p["scenarios"][str(roots[0])]
    records = []
    for i, scene in enumerate(scenes):
        for method in METHODS:
            for seed in p["planner_seeds"]:
                name = f"r{roots[0]}_m{i:02d}_{method}_s{seed:04d}.json"
                path = SOURCE / "results" / name
                r = read(path)
                row = r["row"]
                assert (row["scene"], row["method"], row["seed"]) == (scene["name"], method, seed)
                records.append({"source_file": name, "source_sha256": digest(path),
                                "row": row, "waypoints": r["detail"]["waypoints"]})
    assert len(scenes) == 24 and len(records) == 1440
    assert len({(r["row"]["scene"], r["row"]["method"], r["row"]["seed"]) for r in records}) == 1440
    inputs = {"scenarios": scenes, "records": records, "original_protocol": p}
    dump(HERE / "frozen_inputs.json", inputs)
    protocol = {
        "protocol": "P6 independent external precision reassessment v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reduce finite-Monte-Carlo uncertainty in frozen-path risk and empirical target classifications; not new-map validation or replanning.",
        "methods": list(METHODS), "maps": 24, "planner_seeds": p["planner_seeds"],
        "paths": len(records), "samples_per_path": SAMPLES, "sigma": SIGMA,
        "target_cfr": TARGET, "safety_margin": p["benchmark_config"]["safety_margin"],
        "distribution": "Independent zero-mean Gaussian translation for each obstacle and coordinate, standard deviation sigma; no clipping or endpoint-rejection.",
        "random_generator": "NumPy PCG64, Generator.normal", "base_seed": BASE_SEED,
        "stream_derivation": "SHA256(JSON(['P6-independent-high-precision-v1', base_seed, scene_name])), first16bytes unsigned big endian",
        "pairing": "One fresh 20000-draw bank per map, shared across methods and planner seeds; independent of original external and internal streams.",
        "outcomes": "Unconditional CFR includes nominal failures as zero; each fixed feasible path gets an individual 95% Wilson interval; compare old M500 and new M20000 empirical CFR >=0.825 classifications.",
        "statistics": "24 paired map means; two fixed CFR contrasts full-vs-RT-LS and full-vs-RT-DE; two-sided exact signed-rank with zeros <=1e-12 discarded and average ranks for ties; one two-test Holm family; 20000 within-family stratified map bootstraps, seed202610142018, descriptive95%percentile intervals. No length test.",
        "interpretation": "Post hoc supplementary reassessment of existing paths. Individual Wilson intervals concern Monte Carlo error for one fixed path, not simultaneous certification or uncertainty across maps. Shared draws are not treated as independent planner repetitions.",
        "source_archive": str(SOURCE), "source_protocol_sha256": digest(SOURCE / "protocol.json"),
        "frozen_inputs_sha256": digest(HERE / "frozen_inputs.json"),
        "script_sha256_at_freeze": digest(__file__), "python": sys.version,
        "numpy": np.__version__, "platform": platform.platform(),
        "stop_rule": "All 1440 paths, M=20000, no outcome-dependent stopping or path exclusion.",
        "storage": "All per-path integer counts and bit-packed collision-free outcomes; map-bank seeds and SHA256 of exact float64 shifts."
    }
    dump(HERE / "protocol.json", protocol)
    print(json.dumps({"prepared": len(records), "samples": SAMPLES, "path_realizations": SAMPLES * len(records)}), flush=True)


def load():
    p = read(HERE / "protocol.json")
    assert digest(HERE / "frozen_inputs.json") == p["frozen_inputs_sha256"]
    assert p["samples_per_path"] == SAMPLES and p["base_seed"] == BASE_SEED
    return p, read(HERE / "frozen_inputs.json")


def validate_pilot():
    p, inputs = load()
    sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
    from run_frozen_stress_suite import collision_free_batch, make_shift_bank
    from uav_path_planning.models import Path3D, Bounds3D, Environment3D, ObstacleBox
    from uav_path_planning.robustness import translate_environment
    from uav_path_planning.scenarios import Scenario3D
    margin = p["safety_margin"]
    scalar_checks, replay_checks = 0, 0
    byscene = defaultdict(list)
    for record in inputs["records"]:
        byscene[record["row"]["scene"]].append(record)
    for sd in inputs["scenarios"]:
        scene = Scenario3D.from_dict(sd)
        bank = bank_for(sd)
        old_bank = .6 * make_shift_bank([scene], "gaussian", 500, inputs["original_protocol"]["external_seed"])[scene.name]
        for method in METHODS:
            r = next(r for r in byscene[scene.name] if r["row"]["method"] == method and r["row"]["seed"] == 9)
            path = Path3D(r["waypoints"])
            short = bank[[0, 1, 117, 10001, 19999]]
            a = collision_free(r["waypoints"], sd, short, margin)
            np.testing.assert_array_equal(a, collision_free_batch(path, scene, short, margin))
            scalar = [path.collision_count(translate_environment(scene.environment, x), safety_margin=margin) == 0 for x in short]
            np.testing.assert_array_equal(a, scalar)
            scalar_checks += len(short)
            old_cfr = float(np.mean(collision_free(r["waypoints"], sd, old_bank, margin))) if r["row"]["nominal_success"] else 0.
            assert abs(old_cfr - r["row"]["cfr_0.6"]) < 1e-14
            replay_checks += 1
    box = {"minimum": [4., 4., 4.], "maximum": [6., 6., 6.]}
    sd = {"bounds": {"minimum": [0., 0., 0.], "maximum": [10., 10., 10.]}, "obstacles": [box]}
    env = Environment3D(Bounds3D((0, 0, 0), (10, 10, 10)), [ObstacleBox(box["minimum"], box["maximum"])])
    cases = [([(0,4,5),(10,4,5)], False), ([(0,3.99,5),(10,3.99,5)], True),
             ([(0,0,0),(4,4,4)], False), ([(5,5,5),(5,5,5)], False),
             ([(2,2,2),(2,2,2)], True), ([(-1,2,2),(2,2,2)], False),
             ([(0,5,5),(10,5,5)], False), ([(0,0,0),(0,0,10)], True)]
    for points, expected in cases:
        b = collision_free(points, sd, np.zeros((1,1,3)))[0]
        scalar = Path3D(points).collision_count(env) == 0
        assert b == scalar == expected
        scalar_checks += 1
    sd = inputs["scenarios"][0]
    bank = bank_for(sd)
    start = time.perf_counter()
    for r in byscene[sd["name"]][:3]:
        collision_free(r["waypoints"], sd, bank, margin)
    seconds = time.perf_counter() - start
    result = {"scalar_checks": scalar_checks, "old_bank_path_reproductions": replay_checks,
              "edge_cases": len(cases), "pilot_paths": 3, "samples": SAMPLES,
              "pilot_elapsed_seconds": seconds, "estimated_full_seconds": seconds / 3 * 1440,
              "production_collision_source_sha256": digest(REPO / "src/uav_path_planning/models.py"),
              "production_vector_source_sha256": digest(REPO / "scripts/run_frozen_stress_suite.py"),
              "passed": True}
    dump(HERE / "validation_pilot.json", result)
    print(json.dumps(result), flush=True)


def wilson(k, n):
    z = NormalDist().inv_cdf(.975)
    center = (k / n + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n)) / (1+z*z/n)
    return max(0., center-half), min(1., center+half)


def exact_signed_rank(values):
    values = [float(x) for x in values if abs(x) > 1e-12]
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    ranks = [0] * len(values)
    a = 0
    while a < len(order):
        b = a + 1
        while b < len(order) and abs(values[order[b]]) == abs(values[order[a]]):
            b += 1
        for i in order[a:b]:
            ranks[i] = a + 1 + b
        a = b
    total = sum(ranks)
    counts = [1] + [0] * total
    reach = 0
    for rank in ranks:
        for score in range(reach, -1, -1):
            counts[score + rank] += counts[score]
        reach += rank
    observed = sum(r for r, v in zip(ranks, values) if v > 0)
    return sum(n for i, n in enumerate(counts) if abs(2*i-total) >= abs(2*observed-total)) / 2**len(values)


def summarize(rows, maps):
    maprows = []
    methodrows = []
    for sd in maps:
        for method in METHODS:
            rs = [r for r in rows if r["scene"] == sd["name"] and r["method"] == method]
            assert len(rs) == 20
            maprows.append({"scene": sd["name"], "method": method, "paths": 20,
                            "cfr_old": float(np.mean([r["old_cfr"] for r in rs])),
                            "cfr_new": float(np.mean([r["new_cfr"] for r in rs])),
                            "pass_old": float(np.mean([r["old_pass"] for r in rs])),
                            "pass_new": float(np.mean([r["new_pass"] for r in rs]))})
    for method in METHODS:
        rs = [r for r in rows if r["method"] == method]
        mr = [r for r in maprows if r["method"] == method]
        methodrows.append({"method": method, "label": LABELS[method], "paths": len(rs),
                           "cfr_old": float(np.mean([r["cfr_old"] for r in mr])),
                           "cfr_new": float(np.mean([r["cfr_new"] for r in mr])),
                           "pass_old_count": sum(r["old_pass"] for r in rs),
                           "pass_new_count": sum(r["new_pass"] for r in rs),
                           "old_interval_contains_target": sum(r["old_wilson_low"] <= TARGET <= r["old_wilson_high"] for r in rs),
                           "new_interval_contains_target": sum(r["new_wilson_low"] <= TARGET <= r["new_wilson_high"] for r in rs),
                           "pass_to_fail": sum(r["old_pass"] and not r["new_pass"] for r in rs),
                           "fail_to_pass": sum(not r["old_pass"] and r["new_pass"] for r in rs),
                           "new_interval_entirely_above_target": sum(r["new_wilson_low"] > TARGET for r in rs),
                           "new_interval_entirely_below_target": sum(r["new_wilson_high"] < TARGET for r in rs),
                           "mean_old_wilson_width_pp": 100*float(np.mean([r["old_wilson_high"]-r["old_wilson_low"] for r in rs])),
                           "mean_new_wilson_width_pp": 100*float(np.mean([r["new_wilson_high"]-r["new_wilson_low"] for r in rs]))})
    lookup = {(r["scene"], r["method"]): r for r in maprows}
    names = [sd["name"] for sd in maps]
    rng = np.random.default_rng(BASE_SEED+1)
    idx = np.concatenate([rng.integers(j,j+8,size=(20000,8)) for j in (0,8,16)], axis=1)
    contrasts = []
    for comparator in METHODS[1:]:
        d = np.array([100*(lookup[(s,"full")]["cfr_new"]-lookup[(s,comparator)]["cfr_new"]) for s in names])
        lo, hi = np.quantile(d[idx].mean(axis=1), [.025,.975])
        contrasts.append({"reference": "full", "comparator": comparator, "maps": 24,
                          "mean_cfr_difference_pp": float(d.mean()), "bootstrap95_low_pp": float(lo),
                          "bootstrap95_high_pp": float(hi), "wins": int((d>1e-12).sum()),
                          "ties": int((abs(d)<=1e-12).sum()), "losses": int((d < -1e-12).sum()),
                          "exact_two_sided_p": exact_signed_rank(d)})
    order = sorted(range(2), key=lambda i: contrasts[i]["exact_two_sided_p"])
    previous = 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., contrasts[i]["exact_two_sided_p"]*(2-rank)))
        contrasts[i]["holm_p_two_test_family"] = previous
    csv_write(HERE / "map_summary.csv", maprows)
    csv_write(HERE / "method_summary.csv", methodrows)
    csv_write(HERE / "paired_contrasts.csv", contrasts)
    dump(HERE / "summary.json", {"methods": methodrows, "contrasts": contrasts})
    return methodrows, contrasts


def run():
    p, inputs = load()
    assert read(HERE / "validation_pilot.json")["passed"]
    start = time.perf_counter()
    rows, packed, bank_hashes = [], {}, {}
    for mi, sd in enumerate(inputs["scenarios"]):
        bank = bank_for(sd)
        bank_hashes[sd["name"]] = {"seed": seed_for(sd["name"]), "float64_shape": list(bank.shape),
                                   "sha256": hashlib.sha256(bank.tobytes()).hexdigest()}
        for record in inputs["records"]:
            old = record["row"]
            if old["scene"] != sd["name"]:
                continue
            free = collision_free(record["waypoints"], sd, bank, p["safety_margin"])
            if not old["nominal_success"]:
                free[:] = False
            k = int(free.sum())
            n = len(free)
            low, high = wilson(k,n)
            old_k = round(old["cfr_0.6"]*500)
            oldlo, oldhi = wilson(old_k,500)
            key = f'{old["scene"]}__{old["method"]}__{old["seed"]}'
            packed[key] = np.packbits(free, bitorder="little")
            rows.append({"scene": old["scene"], "method": old["method"], "planner_seed": old["seed"],
                         "nominal_success": bool(old["nominal_success"]), "old_samples": 500,
                         "old_successes": old_k, "old_cfr": old["cfr_0.6"],
                         "old_pass": bool(old["pass_0.6"]), "old_wilson_low": oldlo, "old_wilson_high": oldhi,
                         "new_samples": n, "new_successes": k, "new_cfr": k/n, "new_pass": k/n >= TARGET,
                         "new_wilson_low": low, "new_wilson_high": high,
                         "original_length": old["length"], "source_sha256": record["source_sha256"]})
        print(f'map {mi+1}/24: {sd["name"]}; completed_paths={len(rows)}; elapsed={time.perf_counter()-start:.1f}s', flush=True)
    assert len(rows) == 1440
    csv_write(HERE / "path_counts.csv", rows)
    np.savez_compressed(HERE / "collision_free_bits.npz", **packed)
    dump(HERE / "bank_provenance.json", bank_hashes)
    methodrows, contrasts = summarize(rows, inputs["scenarios"])
    receipt = {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.perf_counter()-start, "paths": len(rows), "samples_per_path": SAMPLES,
               "total_path_realizations": SAMPLES*len(rows), "independent_map_banks": 24,
               "files": {f: digest(HERE/f) for f in ("protocol.json", "frozen_inputs.json", "path_counts.csv", "collision_free_bits.npz", "bank_provenance.json", "map_summary.csv", "method_summary.csv", "paired_contrasts.csv")},
               "script_sha256_used": digest(__file__), "note": "Validation and timing pilot outcomes were not used for tuning; full prespecified batch completed."}
    dump(HERE / "completion.json", receipt)
    print(json.dumps({"completion": receipt, "methods": methodrows, "contrasts": contrasts}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--validate-pilot", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare()
    elif args.validate_pilot:
        validate_pilot()
    else:
        run()
