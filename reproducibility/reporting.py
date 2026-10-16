"""Build paper inputs and figures exclusively from a completed new run."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .atomic_io import write_json
from .cohorts import ROOT, read, sha, equivalent
from .fixed_environment import identity, subprocess_environment


def require_complete(directory, smoke=False):
    directory = Path(directory)
    expected = "complete_smoke" if smoke else "complete"
    if not (directory / "state.json").is_file() or read(directory / "state.json")["status"] != expected:
        raise ValueError(f"This stage needs {expected} results from the same run: {directory}")
    if read(directory / "reproduction.json").get("environment_lock_sha256") != identity():
        raise ValueError("Source cohort does not use the current fixed environment")
    receipt = read(directory / "completion_receipt.json")
    if (receipt.get("status") != expected or receipt.get("environment_lock_sha256") != identity()
            or not receipt.get("all_jobs_recomputed_in_one_environment")):
        raise ValueError("Missing verified fixed-environment completion receipt")


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cohort_tables(run_root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for cohort, prefix in (("e", "E"), ("p6", "P6")):
        require_complete(run_root / cohort)
        for name in ("run_rows", "map_summary", "paired_statistics"):
            source = run_root / cohort / "analysis" / (name + ".csv")
            shutil.copy2(source, destination / (prefix + "_" + source.name))


def assemble(run_root):
    import numpy as np
    run_root = Path(run_root).resolve()
    destination = run_root / "paper_inputs"
    cohort_tables(run_root, destination)
    state = read(run_root / "diagnostics/state.json")
    if state["status"] != "complete":
        raise ValueError("Complete new diagnostics before building manuscript inputs")
    diagnostic_contract = read(run_root / "diagnostics/reproduction.json")
    if (Path(diagnostic_contract["reference"]).resolve() != (run_root / "e").resolve()
            or diagnostic_contract["environment_lock_sha256"] != identity()):
        raise ValueError("Diagnostics must derive from this run's E cohort")
    stages = state["completed_stages"]
    from .diagnostics import STAGES, verify_stage
    for name, script, _ in STAGES:
        verify_stage(stages[name]["output"], sha(ROOT / "reproducibility/internal" / script), False)
    copies = {
        "P0_metrics.csv": ("p0", "fixed_path_metrics.csv"),
        "P0_estimator_summary.csv": ("p0", "summary.csv"),
        "P1_replay_rows.csv": ("p1_replay", "run_rows.csv"),
        "P2_per_map.csv": ("p1_p2_p4_p5", "P2_per_map.csv"),
        "P3_local_sensitivity_summary.csv": ("p3", "P3_summary.csv"),
    }
    source_files = {}
    for target, (stage, name) in copies.items():
        source = Path(stages[stage]["output"]) / name
        shutil.copy2(source, destination / target)
        source_files[str(source)] = sha(source)
    # Author-supplied conceptual diagram is data-independent.
    for extension in ("pdf", "png", "svg", "drawio"):
        name = "fig01_framework." + extension
        shutil.copy2(ROOT / "data/plot_inputs" / name, destination / name)

    # Preserve the predeclared example IDs and composition, replace all paths
    # and annotations with this run's actual results (no outcome selection).
    grid = copy.deepcopy(read(ROOT / "data/plot_inputs/path_grid_examples.json"))
    for example in grid["examples"]:
        for method, path in example["paths"].items():
            filename = path["source_file"]
            cohort = "p6" if (run_root / "p6/results" / filename).is_file() else "e"
            source = run_root / cohort / "results" / filename
            result = read(source)
            if (result["row"]["scene"], result["row"]["seed"], result["row"]["method"]) != (example["scene"], example["seed"], method):
                raise ValueError("Path illustration source identity mismatch")
            path.update(waypoints=result["detail"]["waypoints"], length=result["row"]["length"],
                        cfr06=result["row"]["cfr_0.6"], nominal_success=result["row"]["nominal_success"],
                        source_sha256=sha(source))
            source_files[str(source)] = sha(source)
    grid["source_protocol_sha256"] = {c: sha(run_root / c / "protocol.json") for c in ("e", "p6")}
    grid["source_rows_sha256"] = {c: sha(destination / f"{c}_run_rows.csv") for c in ("E", "P6")}
    grid["environment_lock_sha256"] = identity()
    write_json(destination / "path_grid_examples.json", grid)

    protocol = read(run_root / "p6/protocol.json")
    root = protocol["map_roots"][0]
    scenes = protocol["scenarios"][str(root)]
    seeds = protocol["planner_seeds"]
    methods = ["full", "risk_polish_de", "risk_polish_lshade"]
    traces = np.empty((3, 24, 20, 165), dtype=float)
    provenance = []
    for mi, method in enumerate(methods):
        for si, scene in enumerate(scenes):
            for ki, seed in enumerate(seeds):
                filename = f"r{root}_m{si:02d}_{method}_s{seed:04d}.json"
                source = run_root / "p6/results" / filename
                result = read(source)
                stage = result["detail"]["extension_metadata"]["global_stage"]
                trace = result["detail"]["objective_history"][-165:]
                if (len(trace) != 165 or stage["evaluations"] != 656
                        or not equivalent(trace[0], stage["path_length"])
                        or not equivalent(trace[-1], result["row"]["length"])):
                    raise ValueError("Terminal trace does not match its actual recorded endpoints: " + filename)
                traces[mi, si, ki] = trace
                provenance.append(dict(method=method, scene=scene["name"], seed=seed, receipt=filename,
                    sha256=sha(source), global_nfe=656, terminal_nfe=164,
                    start_length=trace[0], final_length=trace[-1]))
    np.savez_compressed(destination / "P6_terminal_traces.npz", methods=np.array(methods),
                        scenes=np.array([s["name"] for s in scenes]), seeds=np.array(seeds),
                        traces=traces, terminal_nfe=np.arange(165))
    csv_write(destination / "P6_terminal_trace_provenance.csv", provenance)
    reductions = 100 * (1 - traces / traces[..., :1])
    map_means = reductions.mean(axis=2)
    curves = []
    for family in ("ood_dense_clutter", "ood_highrise_urban", "ood_narrow_barriers"):
        selected = [i for i, s in enumerate(scenes) if s["name"].rsplit("_", 1)[0] == family]
        indices = np.random.default_rng(20261011).integers(8, size=(5000, 8))
        for mi, method in enumerate(methods):
            maps = map_means[mi, selected]
            low, high = np.quantile(maps[indices].mean(axis=1), [.025, .975], axis=0)
            curves.append(dict(family=family, method=method, maps=8, paths=160, trace_points=165,
                               mean_terminal_reduction_pct=float(maps.mean(axis=0)[-1]),
                               pointwise_map_bootstrap_95_low=float(low[-1]),
                               pointwise_map_bootstrap_95_high=float(high[-1])))
    csv_write(destination / "P6_terminal_curve_summary.csv", curves)
    manifest = {"schema": 1, "source_run": str(run_root), "environment_lock_sha256": identity(),
                "scope": "All numerical inputs from the completed current run. Figure 1 is the author's unchanged conceptual diagram.",
                "inputs": {p.name: {"sha256": sha(p)} for p in destination.iterdir() if p.is_file() and p.name != "manifest.json"},
                "diagnostic_and_example_sources": source_files,
                "cohort_receipts": {c: sha(run_root / c / "completion_receipt.json") for c in ("e", "p6")}}
    write_json(destination / "manifest.json", manifest)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--figure", default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        write_json(args.run_root / "figures/smoke_scope.json", {
            "status": "smoke_only", "numerical_figures_rendered": False,
            "reason": "Partial cohorts cannot produce complete-study statistics or figures."})
        print("Figures deferred until the complete new cohorts and diagnostics exist.", flush=True)
        return
    inputs = assemble(args.run_root)
    subprocess.run([sys.executable, str(ROOT / "plotting/render.py"), "--input", str(inputs),
                    "--output", str(args.run_root / "figures"), "--figure", args.figure],
                   env=subprocess_environment(), check=True)


if __name__ == "__main__":
    main()
