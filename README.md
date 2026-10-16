# RiskAwareDE

Implementation of **Low-variance risk estimation for 3D path planning under obstacle uncertainty**, by Bocong Chen and Shuai Wu.

RiskAwareDE plans three-dimensional paths through uncertain obstacle environments using three components:

1. **Global search:** differential evolution searches for a feasible, short path.
2. **Risk estimation:** conditional integration estimates whole-path collision risk under obstacle-position uncertainty.
3. **Local refinement:** budgeted tangent refinement shortens the path using the estimated risk direction.

The repository provides the planning algorithms, experimental protocols, external-method comparisons, statistical analyses, and plotting code used in the paper.

## Requirements

- Windows x64 and **official CPython 3.12.4**.
- CPU execution; a GPU is not required.
- Dependencies are pinned in `requirements.lock.txt` and installed automatically by the setup script.

## Installation

Clone the repository, download the reference data, and create the environment:

```powershell
git clone https://github.com/cbcong/RiskAwareDE.git
cd RiskAwareDE
py -3.12 prepare_data.py
py -3.12 setup_environment.py
```

`prepare_data.py` downloads the data asset from [Releases](https://github.com/cbcong/RiskAwareDE/releases/latest) and verifies each file against its recorded SHA-256. If the data are already present, it only verifies them. `setup_environment.py` creates an isolated `.venv` and installs the pinned dependencies.

If automatic download is unavailable, download `RiskAwareDE_Data.zip` from Releases and import it manually:

```powershell
py -3.12 prepare_data.py --archive "<path-to-RiskAwareDE_Data.zip>"
```

If the Windows `py` launcher is unavailable, invoke these scripts with the installed Python 3.12.4 executable. Run the following commands from the repository root.

## Run the experiments

First run a small computation covering every experimental stage:

```powershell
.venv/Scripts/python.exe run_all.py --smoke
```

A successful smoke run reports `complete_smoke`. It checks the workflow with a small subset of tasks; it does not produce the complete experimental results.

To reproduce the complete study:

```powershell
.venv/Scripts/python.exe run_all.py
```

The launcher executes the stages in dependency order and reports `complete_full_pipeline` when they have all completed.

| Entry point | Experiments |
|---|---|
| `run_all.py` | Complete experimental pipeline, statistical analysis, and figures |
| `run_e.py` | E: 9,624 planning tasks across 21 methods, including main comparisons and ablations |
| `run_p.py` | P6: 3,840 shifted-map planning tasks; P0–P5: estimator, replay, sensitivity, calibration, and stress diagnostics |
| `run_external.py` | 384 external-method comparison plans, two-scale evaluation of 384 saved paths, and 288 lower-budget plans |
| `run_validation.py` | Independent evaluation of 1,440 paths using 20,000 perturbations per path, 360 timing runs, and statistical robustness analysis |
| `run_figures.py` | Figures generated from the completed experiment outputs |

External methods are evaluated using the common geometric scenarios and adaptations described in the paper. Method implementations and upstream provenance are included in `experiments/`.

The individual entry points share the same output directory. Diagnostics require completed E results; validation and figure assembly require the preceding stages. Use `run_all.py` for the first complete reproduction.

## Reproduce the figures

The saved numerical inputs and figures are available in `paper_results/`. To redraw all nine main-paper figures and Supplementary Figure S1 without rerunning the planning experiments:

```powershell
.venv/Scripts/python.exe run_figures.py --published
```

The resulting PDF, PNG, and SVG files are written to `outputs/published_figures/`. To redraw one figure:

```powershell
.venv/Scripts/python.exe run_figures.py --published --figure 7
```

Omit `--published` to generate figures from a completed new experimental run.

## Check progress and resume

```powershell
# Verify the environment, source files, and reference data:
.venv/Scripts/python.exe run_all.py --check

# List the experimental stages:
.venv/Scripts/python.exe run_all.py --plan

# Inspect progress:
.venv/Scripts/python.exe run_all.py --status

# Start an independent run in a separate directory:
.venv/Scripts/python.exe run_all.py --output outputs/new_run
```

Repeat the original command to resume validated checkpoints on the same computer, in the same location and environment. Do not run two processes into the same output directory. `--workers 2` parallelizes E/P6 tasks; timed comparisons remain serial.

To pause at a safe boundary, create an empty `STOP` file in the run directory. Remove it before resuming. An interrupted diagnostic stage may repeat its unfinished work.

## Data and reproducibility

The repository includes synthetic map geometry, fixed seeds, method configurations, evaluation budgets, and experimental protocols. The release data asset contains per-run reference records and associated summaries. Checksums verify the scientific source and data before execution.

New experimental results are written under `outputs/`. The default complete-run directory is `outputs/fixed_env_20261016/`; smoke runs use `outputs/fixed_env_smoke/`. Saved reference records are kept separate from newly computed results.

Execution time depends on hardware and system load. Outcomes of methods stopped by elapsed time can therefore vary across computers. Fresh results and their differences from the reference are retained for inspection.

## Repository structure

| Directory | Contents |
|---|---|
| `src/uav_path_planning/` | Search algorithms, geometry, risk estimation, refinement, and statistics |
| `reproducibility/` | Experiment scheduling, checkpoints, validation, and result assembly |
| `experiments/` | External-method adaptations, upstream source, maps, and protocols |
| `scripts/`, `configs/` | Experimental protocol helpers and method configurations |
| `plotting/` | Shared plotting functions for new results and saved paper inputs |
| `data/reference/` | Frozen reference records used for audit and comparison |
| `data/plot_inputs/` | Diagram assets and predefined example-path selections |
| `paper_results/paper_inputs/` | Numerical inputs for the paper figures |
| `paper_results/figures/`, `paper_results/tables/` | Saved figures and numerical summaries |
| `provenance/` | File-integrity and data-download manifests |
| `tests/` | Tests for geometry, risk estimation, evaluation budgets, checkpoints, and input validation |

The main planner is implemented in `soft_risk_projected_polish.py`. Conditional risk estimation and tangent refinement are implemented in `conditional_risk_de.py`, `conditional_projected_polish.py`, and `risk_axis_geometry.py`. Matched controls are in `paper_matched_controls.py`.

To run the tests after installation:

```powershell
.venv/Scripts/python.exe -m pytest -q
```
