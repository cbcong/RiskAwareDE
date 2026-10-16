"""Render the nine manuscript figures and Supplement Figure S1.

Examples:
    python plotting/render.py --figure all --output outputs/figures
    python plotting/render.py --figure 7 --output outputs/figure7

Inputs are the committed, deduplicated files in data/plot_inputs. Each figure
runs in an isolated Python process to preserve its established font and style.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PLOTTING = Path(__file__).resolve().parent
INPUTS = ROOT / "data" / "plot_inputs"
FIGURES = {
    "1": {"stem": "fig01_framework", "module": None, "function": None,
          "inputs": ["fig01_framework.pdf", "fig01_framework.png", "fig01_framework.svg"]},
    "2": {"stem": "fig06_path_grid", "module": "paths_3d", "function": "draw_horizontal",
          "inputs": ["path_grid_examples.json"]},
    "3": {"stem": "fig02_tradeoff", "module": "statistical", "function": "fig02_tradeoff",
          "inputs": ["E_map_summary.csv", "P6_map_summary.csv"]},
    "4": {"stem": "fig03_p6_effects", "module": "statistical", "function": "fig03_p6_effects",
          "inputs": ["P6_paired_statistics.csv"]},
    "5": {"stem": "fig08_estimator", "module": "estimator_sensitivity", "function": "estimator",
          "inputs": ["P0_metrics.csv", "P0_estimator_summary.csv"]},
    "6": {"stem": "fig04_mechanisms", "module": "statistical", "function": "fig04_mechanisms",
          "inputs": ["P2_per_map.csv", "P0_metrics.csv", "P1_replay_rows.csv"]},
    "7": {"stem": "fig07_convergence", "module": "statistical", "function": "fig07_convergence",
          "inputs": ["P6_terminal_traces.npz", "P6_terminal_trace_provenance.csv",
                     "P6_terminal_curve_summary.csv"]},
    "8": {"stem": "fig09_sensitivity", "module": "estimator_sensitivity", "function": "sensitivity",
          "inputs": ["P3_local_sensitivity_summary.csv"]},
    "9": {"stem": "fig05_generalization", "module": "statistical", "function": "fig05_generalization",
          "inputs": ["E_run_rows.csv", "P6_run_rows.csv"]},
    "s1": {"stem": "figS01_map_heatmap", "module": "map_heatmap", "function": "main",
           "inputs": ["E_map_summary.csv", "P6_map_summary.csv"]},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(figure: str, output: Path) -> None:
    import importlib
    spec = FIGURES[figure]
    if figure == "1":
        for extension in ("pdf", "png", "svg"):
            shutil.copy2(INPUTS / f"{spec['stem']}.{extension}", output)
        return
    module = importlib.import_module(spec["module"])
    if spec["module"] == "statistical":
        module.DATA, module.FIGS = INPUTS, output
    else:
        module.OUTPUT = output
        if spec["module"] == "paths_3d":
            module.DATA = json.loads((INPUTS / "path_grid_examples.json").read_text(encoding="utf-8"))
        elif spec["module"] == "estimator_sensitivity":
            module.D = module.S = INPUTS
        elif spec["module"] == "map_heatmap":
            module.DATA = INPUTS
    getattr(module, spec["function"])()


def main() -> None:
    global INPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument("--figure", choices=["all", *FIGURES], default="all")
    parser.add_argument("--input", type=Path, help="Verified inputs from a completed new run")
    parser.add_argument("--worker", choices=list(FIGURES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.input:
        INPUTS = args.input.resolve()
    output = args.output.resolve()
    if output == INPUTS.resolve() or INPUTS.resolve() in output.parents:
        raise ValueError("Choose an output directory outside the frozen plotting inputs")
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker, output)
        return

    expected = json.loads(((INPUTS if args.input else PLOTTING) / "manifest.json").read_text(encoding="utf-8"))
    chosen = list(FIGURES) if args.figure == "all" else [args.figure]
    needed = sorted({name for key in chosen for name in FIGURES[key]["inputs"]})
    for name in needed:
        if sha256(INPUTS / name) != expected["inputs"][name]["sha256"]:
            raise RuntimeError(f"Frozen plotting input changed: {name}")
    results = []
    for figure in chosen:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", figure,
                        "--output", str(output), *(["--input", str(INPUTS)] if args.input else [])], check=True, cwd=ROOT)
        exports = {}
        for extension in ("pdf", "png", "svg"):
            name = f"{FIGURES[figure]['stem']}.{extension}"
            path = output / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing figure export: {path}")
            exports[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        results.append({"figure": figure, "inputs": FIGURES[figure]["inputs"], "exports": exports})
        print(f"Figure {figure}: PDF, PNG, SVG complete", flush=True)
    for name in needed:
        if sha256(INPUTS / name) != expected["inputs"][name]["sha256"]:
            raise RuntimeError(f"Rendering modified an input: {name}")
    report = {"status": "passed", "manuscript_version": "3.1.0", "figures": results,
              "input_directory": str(INPUTS), "fresh_run_inputs": bool(args.input),
              "input_hashes": {name: sha256(INPUTS / name) for name in needed},
              "source_hashes": {path.name: sha256(path) for path in sorted(PLOTTING.glob("*.py"))},
              "input_files_unchanged": True, "python": sys.version,
              "figure7_band_alpha": 0.30, "figure9_background_grid": False,
              "panel_labels": "below each panel; metric labels combined where appropriate",
              "caption_axis_context": {
                  "3": "Horizontal axis: mean feasible path length (scenario units).",
                  "7": "Horizontal axis: terminal objective evaluations."}}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"Verified {len(chosen)} figures; manifest: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
