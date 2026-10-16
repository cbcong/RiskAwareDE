"""Run an explicitly configured optimizer extension on the original map bank.

Extensions provide create_optimizer(problem, RiskAwareDEConfig, options).
All objective and risk evaluations must be counted in their returned result.
The old map bank remains exposed development data, irrespective of seed count.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields, replace
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uav_path_planning import make_research_scenarios, split_research_scenarios
from uav_path_planning.benchmarks import BenchmarkConfig, _problem
from uav_path_planning.de import DEConfig
from uav_path_planning.experiments import ExperimentRecord, save_records
from uav_path_planning.risk_aware_de import RiskAwareDE, RiskAwareDEConfig
from uav_path_planning.robustness import NominalPlan, evaluate_robustness, save_robustness
from uav_path_planning.run_artifacts import recorded_run


def risk_config(config: BenchmarkConfig, seed: int) -> RiskAwareDEConfig:
    aliases = {"risk_gate_schedule": "gate_schedule", "risk_gate_start": "gate_start", "risk_tradeoff": "tradeoff"}
    values = {}
    for field in fields(RiskAwareDEConfig):
        if field.name == "base":
            continue
        values[field.name] = getattr(config, "risk_aware_" + aliases.get(field.name, field.name))
    return RiskAwareDEConfig(base=DEConfig(
        population_size=config.risk_aware_population,
        generations=config.risk_aware_generations,
        max_evaluations=config.risk_aware_max_evaluations,
        seed=seed), **values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--changes", type=Path)
    p.add_argument("--bundle", type=Path, help="Frozen JSON containing module, benchmark_config and extension_options")
    p.add_argument("--label", required=True)
    p.add_argument("--module", default="standard")
    p.add_argument("--options", type=Path)
    p.add_argument("--split", choices=("test", "validation"), default="test")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--nfe", type=int, default=820)
    p.add_argument("--external-samples", type=int, default=100)
    p.add_argument("--external-seed", type=int, default=5101)
    a = p.parse_args()
    if (a.changes is None) == (a.bundle is None):
        p.error('Use exactly one of --changes and --bundle')
    cfg = BenchmarkConfig(**json.loads((ROOT / "configs/risk_aware_partial070_noarchive_20260930.json").read_text(encoding="utf-8-sig")))
    bundle = json.loads(a.bundle.read_text(encoding='utf-8-sig')) if a.bundle else None
    changes = bundle['benchmark_config'] if bundle else json.loads(a.changes.read_text(encoding="utf-8-sig"))
    cfg = replace(cfg, **changes)
    seeds = tuple(a.seeds or ([4, 5, 6, 7, 8] if a.split == "test" else [2, 3]))
    cfg = replace(cfg, seeds=seeds, risk_aware_max_evaluations=a.nfe)
    if bundle:
        if a.options or a.module != 'standard':
            p.error('A bundle supplies its module and options; do not override them')
        a.module = bundle['module']
    options = bundle['extension_options'] if bundle else (json.loads(a.options.read_text(encoding="utf-8-sig")) if a.options else {})
    module = None if a.module == "standard" else importlib.import_module(a.module)
    if module is None and options:
        p.error('The standard optimizer does not accept extension options; select the explicit extension module')
    scenes = list(getattr(split_research_scenarios(make_research_scenarios(20260923), protocol="research24_6_6_12"), a.split))
    protocol = {"purpose": __doc__, "root_seed": 20260923, "split": a.split,
        "label": a.label, "module": a.module, "extension_options": options,
        "benchmark_config": asdict(cfg), "scenarios": [s.to_dict() for s in scenes],
        "external_seed": a.external_seed, "external_samples": a.external_samples,
        "evidence_status": "known original maps; new planner seeds do not make new geometry"}
    with recorded_run(a.output, protocol):
        records, plans = [], []
        for scene in scenes:
            for seed in seeds:
                problem = _problem(scene, cfg)
                rcfg = risk_config(cfg, seed)
                started = time.perf_counter()
                optimizer = RiskAwareDE(problem.bounds, problem, rcfg) if module is None else module.create_optimizer(problem, rcfg, options)
                raw = optimizer.optimize()
                elapsed = time.perf_counter() - started
                if raw.evaluations != a.nfe:
                    raise AssertionError(f"Expected {a.nfe} nominal evaluations; got {raw.evaluations}")
                expected = a.nfe * rcfg.perturbation_samples * len(rcfg.perturbation_sigmas or (rcfg.perturbation_sigma,))
                if not rcfg.skip_risk_for_nominal_infeasible and raw.internal_collision_checks != expected:
                    raise AssertionError(f"Expected {expected} internal checks; got {raw.internal_collision_checks}")
                path = problem.decode(raw.best_vector)
                metrics = problem.evaluate(path)
                metadata = {"budget": asdict(cfg), "planner_metadata": {
                    "module": a.module, "extension_options": options,
                    "extension_metadata": getattr(optimizer, "extension_metadata", {}),
                    "internal_collision_checks": raw.internal_collision_checks,
                    "nominal_feasible": raw.best_evaluation.nominal_feasible,
                    "perturbed_collision_rate": raw.best_evaluation.perturbed_collision_rate,
                    "chance_violation": raw.best_evaluation.chance_violation,
                    "best_risk_evaluation": asdict(raw.best_evaluation),
                    "objective_history": raw.history, "lexicographic_key_history": raw.key_history,
                    "risk_gate_history": raw.gate_history,
                    "local_refinement_stats": raw.local_refinement_stats,
                    "local_refinement_events": raw.local_refinement_events}}
                records.append(ExperimentRecord(a.label, scene.name, seed, elapsed, raw.evaluations,
                    metrics, "ok" if metrics.feasible else "infeasible", metadata))
                plans.append(NominalPlan(a.label, scene.name, seed, path, elapsed, raw.evaluations, raw.internal_collision_checks))
            save_records(records, a.output / "benchmark_records.json")
            print(f"{a.label}: {scene.name} ({len(records)} plans)", flush=True)
        save_records(records, a.output / "benchmark_records.csv")
        (a.output / "nominal_paths_checkpoint.json").write_text(json.dumps([asdict(x) for x in plans]), encoding="utf-8")
        robust = evaluate_robustness(scenes, plans, sigmas=(0., .3, .6), samples=a.external_samples,
            perturbation_seed=a.external_seed, safety_margin=cfg.safety_margin)
        robust.protocol["benchmark_config"] = asdict(cfg)
        robust.protocol["research_scope"] = protocol
        save_robustness(robust, a.output / "robustness")
        good = [x for x in records if x.metrics.feasible]
        cfr = {x["sigma"]: x["unconditional_collision_free_rate"] for x in robust.aggregate_summaries}
        summary = {"algorithm": a.label, "module": a.module, "root_seed": 20260923,
            "split": a.split, "plans": len(records), "mean_length": sum(x.metrics.length for x in good)/len(good) if good else None,
            "feasible_rate": len(good)/len(records), "cfr03": cfr[.3], "cfr06": cfr[.6],
            "mean_nfe": sum(x.evaluations for x in records)/len(records),
            "mean_internal_checks": sum(x.internal_collision_checks for x in plans)/len(plans),
            "mean_runtime_seconds": sum(x.runtime_seconds for x in plans)/len(plans)}
        (a.output / "candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (a.output / "candidate_summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
