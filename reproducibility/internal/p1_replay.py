"""P1: replay three terminal rules from 120 predeclared saved global states.

Uses the source run's immutable optimizer files and its internal/external noise
settings. No global search is rerun and no main-method selection is performed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from reproducibility.cohorts import equivalent
import time

SOURCE = CODE_ROOT / "data/reference/e"
MODES = ("tangent", "ignore_measured_gradient", "length_only")
SEEDS = (9, 10, 11, 12, 13)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from reproducibility.atomic_io import write_json as write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a replay experiment")
    manifest, protocol = read(source / "execution_manifest.json"), read(source / "protocol.json")
    if read(source / "state.json")["status"] != ("complete_smoke" if args.smoke else "complete"):
        raise ValueError("Source must be complete")
    frozen = CODE_ROOT
    for name, expected in manifest["source_sha256"].items():
        if name.startswith("src/") and sha(frozen / name) != expected:
            raise ValueError("Frozen scientific source changed: " + name)
    sys.path[:0] = [str(frozen / "src"), str(frozen / "scripts")]
    from uav_path_planning.benchmarks import BenchmarkConfig, _problem
    from uav_path_planning.scenarios import Scenario3D
    from uav_path_planning.soft_risk_projected_polish import SoftRiskConditionalDE, SoftRiskProjectedPolish
    from uav_path_planning.risk_aware_de import RiskAwareDEResult
    from run_extension_candidate import risk_config
    from run_frozen_stress_suite import collision_free_batch, make_shift_bank, save_csv
    from summarize_six_category_comparison import bootstrap_maps, holm, wilcoxon_signed_rank
    from seven_category_control import checkpoint_digest
    cfg = BenchmarkConfig(**protocol["benchmark_config"])
    jobs = [job for job in read(source / "jobs.json") if job["method"] == "full" and job["seed"] in SEEDS]
    jobs.sort(key=lambda job: (job["map_root"], job["scene"], job["seed"]))
    if len(jobs) != 120 or len({(j["map_root"], j["scene"]) for j in jobs}) != 24:
        raise ValueError("Expected all 24 maps and the first five declared seeds")
    if args.smoke:
        jobs = [next(j for j in jobs if j["map_root"] == root) for root in protocol["map_roots"]]
    inputs = {job["id"]: sha(source / "results" / (job["id"] + ".json")) for job in jobs}
    declaration = {"status": "running", "declared_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_protocol_sha256": sha(source / "protocol.json"),
        "script_sha256": sha(__file__), "input_sha256": inputs,
        "selection": "All 24 maps, fixed first five planner seeds 9..13 before any replay outcomes",
        "planner_seeds": SEEDS, "global_states": 120, "terminal_replays": 360,
        "modes": MODES, "reused_global_nfe": 656, "terminal_nfe": 164, "beta": .05,
        "analysis": "2 controls x 2 metrics (length,CFR.6) x 2 banks = 8 exploratory Holm tests",
        "setup": "One evaluator construction and one initial-record restoration check per replay; not used as a search decision; counted separately from 164 terminal NFE",
        "limits": "Known maps and selected main; not blind confirmation. Ignore-gradient retains gradient probe cost. Length-only reallocates probes to proposals, so it is a whole-terminal strategy comparison."}
    output.mkdir(parents=True)
    declaration.update(smoke=args.smoke, global_states=len(jobs), terminal_replays=3*len(jobs))
    write(output / "execution_manifest.json", declaration)

    class Replay(SoftRiskProjectedPolish):
        def __init__(self, *a, saved_global, **kw):
            self.saved_global = saved_global
            super().__init__(*a, **kw)

        def _create_evaluator(self, global_cfg):
            evaluator = SoftRiskConditionalDE(self.problem, global_cfg, beta=self.beta, axis=self.axis)
            stage = self.saved_global
            if stage["evaluations"] != global_cfg.base.max_evaluations:
                raise AssertionError("Saved global budget mismatch")
            vector = list(stage["best_vector"])
            record = evaluator._evaluate(vector)
            if not equivalent(record.objective, stage["objective"]) or not equivalent(record.perturbed_collision_rate, stage["estimated_risk"]):
                raise AssertionError("Saved global risk/objective cannot be restored exactly")
            evaluator._internal_collision_checks = stage["evaluations"] * global_cfg.perturbation_samples
            raw = RiskAwareDEResult(best_vector=vector, best_fitness=record.objective,
                history=[record.objective], evaluations=stage["evaluations"],
                internal_collision_checks=evaluator._internal_collision_checks,
                best_evaluation=record, key_history=[record.key], gate_history=[global_cfg.chance_alpha])
            evaluator.optimize = lambda: raw
            return evaluator

    scenes = { (root, data["name"]): Scenario3D.from_dict(data)
               for root in protocol["map_roots"] for data in protocol["scenarios"][str(root)] }
    noise = {key: make_shift_bank([scene], "gaussian", protocol["external_samples"],
              protocol["external_seed"])[scene.name] for key, scene in scenes.items()}
    rows = []
    try:
        random.Random(20261010).shuffle(jobs)
        for index, job in enumerate(jobs):
            saved = read(source / "results" / (job["id"] + ".json"))
            if saved["checkpoint_sha256"] != checkpoint_digest(saved):
                raise ValueError("Source checkpoint digest mismatch")
            scene = scenes[job["map_root"], job["scene"]]
            problem = _problem(scene, cfg)
            rcfg = risk_config(cfg, job["seed"])
            bundle = []
            for mode in MODES:
                options = dict(protocol["methods"]["full"]["options"])
                options["projection"] = mode
                optimizer = Replay(problem, rcfg,
                    saved_global=saved["detail"]["extension_metadata"]["global_stage"], **options)
                started = time.perf_counter()
                raw = optimizer.optimize()
                elapsed = time.perf_counter() - started
                path = problem.decode(raw.best_vector)
                metrics = problem.evaluate(path)
                exact = equivalent([list(p) for p in path.waypoints], saved["detail"]["waypoints"])
                if raw.evaluations != 820 or raw.internal_collision_checks != 13120:
                    raise AssertionError("Replay budget mismatch")
                if mode == "tangent" and not (exact and equivalent(metrics.length, saved["row"]["length"])
                        and equivalent(raw.best_evaluation.perturbed_collision_rate, saved["row"]["internal_risk"])
                        and raw.local_refinement_stats == saved["detail"]["local_refinement_stats"]):
                    raise AssertionError("Original tangent replay differs from source")
                row = {"map_root": job["map_root"], "scene": job["scene"], "seed": job["seed"],
                    "method": mode, "length": metrics.length, "nominal_success": bool(metrics.feasible),
                    "internal_risk": raw.best_evaluation.perturbed_collision_rate,
                    "terminal_nfe": 164, "terminal_risk_terms": 2624,
                    "restore_check_nfe": 1, "restore_check_risk_terms": 16,
                    "runtime_including_setup_sec": elapsed,
                    "gradient_probes": raw.local_refinement_stats["projected_gradient_probes"],
                    "line_trials": raw.local_refinement_stats["projected_line_trials"],
                    "accepted_steps": raw.local_refinement_stats["projected_accepted"],
                    "projected_trials": sum(event["gradient_projected"] for event in raw.local_refinement_events),
                    "main_reproduction_within_numeric_tolerance": exact if mode == "tangent" else None}
                trials = {}
                for sigma in protocol["sigmas"]:
                    free = collision_free_batch(path, scene, sigma * noise[job["map_root"], job["scene"]], cfg.safety_margin)
                    bits = "".join("1" if bool(bit) and metrics.feasible else "0" for bit in free)
                    rate = bits.count("1") / len(bits)
                    row[f"cfr_{sigma}"] = rate
                    row[f"pass_{sigma}"] = bool(metrics.feasible and rate >= 1-protocol["target_alpha"]-1e-12)
                    trials[str(sigma)] = bits
                    if mode == "tangent" and bits != saved["external_trials"][str(sigma)]:
                        raise AssertionError("External original replay mismatch")
                bundle.append({"row": row, "waypoints": [list(p) for p in path.waypoints], "external_trials": trials,
                    "local_refinement_events": raw.local_refinement_events})
                rows.append(row)
            write(output / "results" / (job["id"] + ".json"), bundle)
            write(output / "progress.json", {"states": index+1, "total_states": len(jobs), "replays": len(rows), "smoke":args.smoke})
            if (index+1) % 20 == 0:
                print(f"P1 replay states {index+1}/120, replays {len(rows)}/360", flush=True)
        write(output / "run_rows.json", rows)
        save_csv(output / "run_rows.csv", rows)
        if args.smoke:
            declaration.update(status="complete_smoke", actual_replays=len(rows), floating_comparison="abs=1e-10, rel=1e-12; bits exact")
            write(output / "completion_receipt.json", {"status":"complete_smoke", "states":len(jobs), "replays":len(rows), "scope":"Smoke only; no formal statistics"})
            return
        tests = []
        lookup = {(r["map_root"],r["scene"],r["seed"],r["method"]):r for r in rows}
        for root in protocol["map_roots"]:
            for control in MODES[1:]:
                for metric in ("length", "cfr_0.6"):
                    maps = defaultdict(list)
                    for key, ref in lookup.items():
                        if key[0] == root and key[3] == "tangent":
                            cand = lookup[key[:3]+(control,)]
                            if metric != "length" or ref["nominal_success"] and cand["nominal_success"]:
                                maps[key[1]].append((cand[metric],ref[metric]))
                    pairs = [(statistics.fmean(a for a,b in v), statistics.fmean(b for a,b in v)) for _,v in sorted(maps.items())]
                    diffs = [a-b for a,b in pairs]
                    result = wilcoxon_signed_rank(diffs)
                    low,high = bootstrap_maps(pairs, metric, 5000, int(root)+len(tests))
                    effect = (100*(statistics.fmean(a for a,b in pairs)/statistics.fmean(b for a,b in pairs)-1)
                              if metric == "length" else 100*statistics.fmean(diffs))
                    tests.append({"map_root":root,"control":control,"metric":metric,"maps":len(pairs),
                        "effect_control_minus_main":effect,"ci95_low":low,"ci95_high":high,
                        "p_value":result.p_value,"test":result.method})
        holm(tests)
        save_csv(output / "paired_statistics.csv", tests)
        summaries = []
        for mode in MODES:
            group=[r for r in rows if r["method"]==mode]
            summaries.append({"method":mode,"plans":len(group), **{key:statistics.fmean(r[key] for r in group)
                for key in ("length","nominal_success","cfr_0.3","cfr_0.6","cfr_0.9","pass_0.6",
                            "gradient_probes","line_trials","accepted_steps","projected_trials")}})
        write(output / "summary.json", summaries)
        save_csv(output / "summary.csv", summaries)
        lines=["# P1 相同全局状态的末端投影对照", "",
            "提前固定全部24张已知地图与种子9—13，共120个全局状态，360次末端重放；不是360次全新规划或盲测。",
            "主方法未修改。tangent全部120次在数值容差内复现（碰撞位串完全相同）源记录的路径、风险、精修计数以及外部500次评分向量。",
            "每个状态复用全局656 NFE，三种策略末端均164 NFE。恢复时各额外1次评价/16风险项仅用于检验，并单列；不参与搜索。",
            "ignore_measured_gradient保留有限差分探针成本，只不使用投影方向；length_only关闭风险梯度探针，将同预算用于更多提案。所有提案仍通过原主方法风险和长度接受规则。", "",
            "| 末端策略 | 路径↓ | 名义成功 | CFR .3 | CFR .6 | CFR .9 | .6达标 | 探针 | 提案 | 接受步 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in summaries:
            rates=" | ".join(f"{100*r[k]:.2f}%" for k in ("nominal_success","cfr_0.3","cfr_0.6","cfr_0.9","pass_0.6"))
            lines.append(f"| {r['method']} | {r['length']:.4f} | {rates} | {r['gradient_probes']:.2f} | {r['line_trials']:.2f} | {r['accepted_steps']:.2f} |")
        lines += ["", "每图库按12张地图独立配对，先平均5个种子；两对照×路径/CFR.6×两图库共8项统一探索性Holm校正。效应为对照−主方法，长度负值更短、CFR正值更高。区间为逐项区间，不显著不是等价。", "",
            "| 图库 | 对照 | 指标 | 效应 | 95%区间 | Holm p |", "|---|---|---|---:|---|---:|"]
        for r in tests:
            lines.append(f"| {r['map_root']} | {r['control']} | {r['metric']} | {r['effect_control_minus_main']:+.5f} | [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] | {r['p_holm']:.6f} |")
        lines += ["", "此实验隔离相同全局状态的末端策略，不能代替全820预算的E4。无论对照好坏均保留结果，不自动改变主方法或选择新的种子。"]
        (output / "P1_投影方向对照.md").write_text("\n".join(lines),encoding="utf-8")
        for job_id,expected in inputs.items():
            if sha(source / "results" / (job_id+".json")) != expected:
                raise ValueError("Source checkpoint changed")
        declaration.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                           actual_replays=len(rows), main_reproduction_within_numeric_tolerances=120)
        write(output / "completion_receipt.json", {"status":"complete","replays":len(rows),
              "main_reproduction_within_numeric_tolerances":120,"source_results_unchanged":True,"rows_sha256":sha(output/"run_rows.json")})
        print(json.dumps({"status":"complete","output":str(output),"summary":summaries},ensure_ascii=False),flush=True)
    except BaseException as exc:
        declaration.update(status="failed", error=repr(exc))
        raise
    finally:
        write(output / "execution_manifest.json", declaration)


if __name__ == "__main__":
    _outcome = main()
    if _outcome not in (None, 0):
        raise SystemExit(_outcome)
    from reproducibility.cohorts import write as _atomic_write, sha as _artifact_sha
    _destination = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    _smoke = "--smoke" in sys.argv
    _artifacts = {p.relative_to(_destination).as_posix(): _artifact_sha(p)
                  for p in _destination.rglob("*") if p.is_file() and p.name != "stage_complete.json"}
    _atomic_write(_destination / "stage_complete.json", {
        "schema": "RiskAwareDE-diagnostic-stage-v270", "status": "complete_smoke" if _smoke else "complete",
        "smoke": _smoke, "runner_sha256": _artifact_sha(Path(__file__)), "artifacts_sha256": _artifacts,
        "scope": "Smoke only; reduced computation and no complete-study conclusions" if _smoke else "Complete declared diagnostic stage"})
