"""Adapter tests using archived fixtures; these are not new experiment results."""
import json
from pathlib import Path
import shutil

import numpy as np

from reproducibility import reporting

ROOT = Path(__file__).resolve().parents[1]


def test_fresh_input_assembly_preserves_complete_trace_data(tmp_path, monkeypatch):
    # Mock completion gates only for this unit fixture. The live pipeline must
    # pass the gates against completed results and the environment fingerprint.
    monkeypatch.setattr(reporting, "require_complete", lambda *args: None)
    from reproducibility import diagnostics
    monkeypatch.setattr(diagnostics, "verify_stage", lambda *args: None)
    data = ROOT / "data/plot_inputs"
    references = ROOT / "data/reference"
    for cohort, prefix in (("e", "E"), ("p6", "P6")):
        directory = tmp_path / cohort
        (directory / "analysis").mkdir(parents=True)
        (directory / "results").mkdir()
        for name in ("protocol.json", "completion_receipt.json"):
            shutil.copy2(references / cohort / name, directory / name)
        for name in ("run_rows", "map_summary", "paired_statistics"):
            shutil.copy2(data / f"{prefix}_{name}.csv", directory / "analysis" / (name + ".csv"))
    for source in (references / "p6/results").glob("*.json"):
        if any("_" + method + "_s" in source.stem for method in ("full", "risk_polish_de", "risk_polish_lshade")):
            shutil.copy2(source, tmp_path / "p6/results" / source.name)
    grid = json.loads((data / "path_grid_examples.json").read_text())
    for example in grid["examples"]:
        for record in example["paths"].values():
            name = record["source_file"]
            if not (tmp_path / "p6/results" / name).exists():
                cohort = "p6" if (references / "p6/results" / name).exists() else "e"
                shutil.copy2(references / cohort / "results" / name, tmp_path / cohort / "results" / name)
    stages = {}
    copies = {
        "p0": {"P0_metrics.csv": "fixed_path_metrics.csv", "P0_estimator_summary.csv": "summary.csv"},
        "p1_replay": {"P1_replay_rows.csv": "run_rows.csv"},
        "p3": {"P3_local_sensitivity_summary.csv": "P3_summary.csv"},
        "p1_p2_p4_p5": {"P2_per_map.csv": "P2_per_map.csv"},
    }
    for stage, files in copies.items():
        directory = tmp_path / "diagnostics" / stage
        directory.mkdir(parents=True)
        for source, target in files.items():
            shutil.copy2(data / source, directory / target)
        stages[stage] = {"output": str(directory)}
    (tmp_path / "diagnostics/state.json").write_text(json.dumps({"status": "complete", "completed_stages": stages}))
    (tmp_path / "diagnostics/reproduction.json").write_text(json.dumps({
        "reference": str(tmp_path / "e"), "environment_lock_sha256": reporting.identity()}))
    result = reporting.assemble(tmp_path)
    with np.load(result / "P6_terminal_traces.npz") as actual, np.load(data / "P6_terminal_traces.npz") as expected:
        for method in expected["methods"]:
            a = actual["methods"].tolist().index(method)
            e = expected["methods"].tolist().index(method)
            np.testing.assert_array_equal(actual["traces"][a], expected["traces"][e])
    assert len(json.loads((result / "manifest.json").read_text())["inputs"]) == 19
    # The plotting source must be the freshly assembled directory even when
    # the fixture's values happen to equal the older published inputs.
    assert json.loads((result / "manifest.json").read_text())["source_run"] == str(tmp_path)
