"""Describe saved locked-main evidence without rerunning or tuning any planner.

P1: paired terminal-stage diagnostics; P2: in-search/out-of-search risk gap;
P4: frozen-path perturbation stress; P5: geometric and map-level diagnostics.
Uses the run's own frozen evaluator to reconstruct/scored saved paths.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from reproducibility.cohorts import equivalent


DEFAULT_RUN = CODE_ROOT / "data/reference/e"
DEFAULT_OUTPUT_BASE = CODE_ROOT / "outputs"
METHODS = ("full", "bare_de", "bare_jade", "bare_shade", "bare_lshade", "bare_pso", "bare_astar",
           "risk_de", "risk_jade", "risk_shade", "risk_lshade", "risk_pso",
           "risk_polish_de", "risk_polish_jade", "risk_polish_shade", "risk_polish_lshade", "risk_polish_pso",
           "cand_tail_risk")


from reproducibility.atomic_io import write_json as write_json


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def avg(values):
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.fmean(finite) if finite else None


def group_means(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys, key)), "planning_attempts": len(group),
             **{metric: avg(row.get(metric) for row in group) for metric in metrics}}
            for key, group in sorted(groups.items())]


def summaries(map_rows, metrics):
    # First average within maps, then give maps equal weight in each scope.
    result = []
    roots = sorted({str(row["map_root"]) for row in map_rows})
    for scope in [*roots, "pooled_known_banks"]:
        selected = [row for row in map_rows if scope == "pooled_known_banks" or str(row["map_root"]) == scope]
        by_method = defaultdict(list)
        for row in selected:
            by_method[row["method"]].append(row)
        for method, group in sorted(by_method.items()):
            result.append({"scope": scope, "method": method, "maps": len(group),
                           "planning_attempts": sum(row["planning_attempts"] for row in group),
                           **{metric: avg(row.get(metric) for row in group) for metric in metrics}})
    return result


def geometry(waypoints):
    if not waypoints or len(waypoints) < 2:
        return {"has_path": False, "defined_turns": 0, "zero_length_segments": 0,
                "mean_turn_deg": None, "max_turn_deg": None, "max_pitch_deg": None,
                "has_turn_gt45": None, "has_turn_gt60": None, "has_turn_gt90": None}
    vectors = [tuple(b[i] - a[i] for i in range(3)) for a, b in zip(waypoints, waypoints[1:])]
    norms = [math.sqrt(sum(x*x for x in v)) for v in vectors]
    turns = []
    for u, v, nu, nv in zip(vectors, vectors[1:], norms, norms[1:]):
        if nu > 1e-12 and nv > 1e-12:
            cosine = max(-1., min(1., sum(x*y for x, y in zip(u, v)) / (nu*nv)))
            turns.append(math.degrees(math.acos(cosine)))
    pitches = [math.degrees(math.asin(min(1., abs(v[2])/n))) for v, n in zip(vectors, norms) if n > 1e-12]
    maximum = max(turns) if turns else None
    return {"has_path": True, "defined_turns": len(turns), "zero_length_segments": sum(n <= 1e-12 for n in norms),
            "mean_turn_deg": avg(turns), "max_turn_deg": maximum, "max_pitch_deg": max(pitches) if pitches else None,
            **{f"has_turn_gt{threshold}": (maximum > threshold + 1e-9) if maximum is not None else None
               for threshold in (45, 60, 90)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = args.output or DEFAULT_OUTPUT_BASE / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_定稿主方法P实验")
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite existing evidence")
    out.mkdir(parents=True, exist_ok=True)
    frozen = CODE_ROOT
    if not (frozen / "scripts/seven_category_protocol.py").exists():
        raise ValueError("The source run must contain its frozen evaluator")
    sys.path.insert(0, str(frozen / "src"))
    sys.path.insert(0, str(frozen / "scripts"))
    from seven_category_protocol import _scene_data, _scene_and_noise
    from run_frozen_stress_suite import collision_free_batch
    from uav_path_planning.benchmarks import BenchmarkConfig, _problem
    from uav_path_planning.models import Path3D

    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8-sig"))
    if protocol["main_method"]["module"] != "uav_path_planning.soft_risk_projected_polish" or protocol["main_method"]["options"].get("beta") != .05:
        raise ValueError("This analysis requires the locked soft_beta050 main")
    if protocol["NFE"] != 820 or protocol["K"] != 16:
        raise ValueError("Unexpected frozen NFE/K contract")
    jobs = json.loads((run / "jobs.json").read_text(encoding="utf-8-sig"))
    selected = [job for job in jobs if job["method"] in METHODS]
    if args.smoke:
        selected=[next(j for j in selected if j["map_root"]==root and j["method"]==method) for root in protocol["map_roots"] for method in ("full","risk_polish_lshade")]
    cfg = BenchmarkConfig(**protocol["benchmark_config"])
    p1, p2, p4, p5, receipts = [], [], [], [], []
    scene_cache = {}
    final_length_checks = final_sigma06_checks = 0
    for job in selected:
        source = run / "results" / (job["id"] + ".json")
        content = source.read_bytes()
        receipts.append({"result_file": source.name, "sha256": hashlib.sha256(content).hexdigest()})
        bundle = json.loads(content)
        row, detail = bundle["row"], bundle["detail"]
        identity = {k: row[k] for k in ("map_root", "scene", "method", "seed")}
        if any(identity[k] != job[k] for k in identity):
            raise ValueError(f"Job identity mismatch: {source.name}")
        key = (job["map_root"], job["scene"])
        if key not in scene_cache:
            scene_data = _scene_data(job, protocol)
            scene, noise = _scene_and_noise(json.dumps(scene_data, sort_keys=True, ensure_ascii=False),
                                            protocol["external_samples"], protocol["external_seed"])
            scene_cache[key] = scene, noise, _problem(scene, cfg)
        scene, noise, problem = scene_cache[key]
        waypoints = detail["waypoints"]
        path = Path3D(waypoints) if waypoints else None
        if path is not None:
            independent_length = sum(math.dist(a, b) for a, b in zip(waypoints, waypoints[1:]))
            if not math.isclose(independent_length, row["length"], abs_tol=1e-10, rel_tol=1e-12):
                raise ValueError(f"Saved path length mismatch: {source.name}")
            final_length_checks += 1
        risk = row.get("internal_risk")
        success = bool(row["nominal_success"])
        # Exclude alternative tail-risk aggregation from a mean-probability calibration claim.
        if risk is not None and job["method"] != "cand_tail_risk":
            p2.append({**identity, "nominal_success": success, "internal_collision_estimate": risk,
                       "external_collision_fraction_06": 1-row["cfr_0.6"],
                       "external_minus_internal": (1-row["cfr_0.6"])-risk if success else None,
                       "internal_gate_met": success and risk <= protocol["search_alpha"] + 1e-12,
                       "external_empirical_target_met": row["pass_0.6"],
                       "internal_gate_met_external_target_missed": success and risk <= protocol["search_alpha"]+1e-12 and not row["pass_0.6"],
                       "external_trials": protocol["external_samples"]})
        stress = {k: v for k, v in row.items() if k.startswith("cfr_") or k.startswith("pass_")}
        p4.append({**identity, "nominal_success": success,
                   "length_feasible": row["length"] if success else None, **stress})
        p5.append({**identity, "nominal_success": success, "cfr_06": row["cfr_0.6"],
                   "external_target_missed": not row["pass_0.6"],
                   "length_feasible": row["length"] if success else None,
                   **geometry(waypoints)})
        if job["method"] not in ("full", "risk_polish_lshade"):
            continue
        before = detail["extension_metadata"]["global_stage"]
        prior_path = problem.decode(before["best_vector"])
        prior_metrics = problem.evaluate(prior_path)
        if not math.isclose(prior_path.length, before["path_length"], abs_tol=1e-10, rel_tol=1e-12):
            raise ValueError("Decoded pre-polish path disagrees with saved metadata")
        if bool(prior_metrics.feasible) != bool(before["nominal_feasible"]):
            raise ValueError("Pre-polish nominal feasibility disagrees with saved metadata")
        after_free = collision_free_batch(path, scene, .6 * noise, margin=cfg.safety_margin) & success
        after_bits = "".join("1" if x else "0" for x in after_free)
        if after_bits != bundle["external_trials"]["0.6"]:
            raise ValueError("Reconstructed external noise/evaluation differs from original result")
        final_sigma06_checks += 1
        pairs = {}
        for sigma in protocol["sigmas"]:
            free = collision_free_batch(prior_path, scene, sigma * noise, margin=cfg.safety_margin) & bool(prior_metrics.feasible)
            prior_cfr = float(free.mean())
            pairs[f"cfr_before_{sigma}"] = prior_cfr
            pairs[f"cfr_after_{sigma}"] = row[f"cfr_{sigma}"]
            pairs[f"cfr_delta_{sigma}"] = row[f"cfr_{sigma}"]-prior_cfr
        refinement = detail["local_refinement_stats"]
        p1.append({**identity, "before_nfe": before["evaluations"], "after_nfe": row["evaluations"],
                   "nominal_before": bool(prior_metrics.feasible), "nominal_after": success,
                   "length_before": prior_path.length, "length_after": row["length"],
                   "length_reduction": prior_path.length-row["length"],
                   "relative_length_reduction_percent": 100*(prior_path.length-row["length"])/prior_path.length,
                   "estimated_risk_before": before["estimated_risk"], "estimated_risk_after": risk,
                   "gradient_probes": refinement["projected_gradient_probes"],
                   "gradient_probes_used": refinement["projected_gradient_probes"] > 0,
                   "line_trials": refinement["projected_line_trials"],
                   "accepted_steps": refinement["projected_accepted"], **pairs})
    # Confirm the fixed-main planning matrix; A* legitimately has one seed per map.
    counts = defaultdict(int)
    for row in p4:
        counts[row["method"]] += 1
    expected = len(scene_cache)*len(protocol["planner_seeds"])
    if not args.smoke and (counts["full"] != expected or expected != 480):
        raise ValueError("The selected main does not contain the complete 24x20 matrix")
    metrics = {
        "P1": [k for k in p1[0] if k not in ("map_root", "scene", "method", "seed")],
        "P2": [k for k in p2[0] if k not in ("map_root", "scene", "method", "seed")],
        "P4": [k for k in p4[0] if k not in ("map_root", "scene", "method", "seed")],
        "P5": [k for k in p5[0] if k not in ("map_root", "scene", "method", "seed")],
    }
    outputs = {}
    for label, data in (("P1", p1), ("P2", p2), ("P4", p4), ("P5", p5)):
        by_map = group_means(data, ("map_root", "scene", "method"), metrics[label])
        summary = summaries(by_map, metrics[label])
        write_csv(out / f"{label}_per_path.csv", data)
        write_csv(out / f"{label}_per_map.csv", by_map)
        write_csv(out / f"{label}_summary.csv", summary)
        outputs[label] = summary
    write_csv(out / "source_result_hashes.csv", receipts)
    source_manifest = {"generated_at_local": datetime.now().isoformat(), "smoke":args.smoke, "source_run": str(run),
        "main_method": protocol["main_method"], "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest(),
        "jobs_sha256": hashlib.sha256((run / "jobs.json").read_bytes()).hexdigest(),
        "source_results": len(receipts), "selected_methods": sorted(counts), "new_optimization_calls": 0,
        "main_plans": counts["full"], "map_count": len(scene_cache), "external_trials_per_path_per_condition": protocol["external_samples"],
        "validation": {"independent_final_length_checks": final_length_checks, "original_sigma06_bitstring_matches": final_sigma06_checks},
        "frozen_evaluator_sources_sha256": {name: hashlib.sha256((frozen / name).read_bytes()).hexdigest() for name in (
            "scripts/seven_category_protocol.py", "scripts/run_frozen_stress_suite.py", "src/uav_path_planning/models.py", "src/uav_path_planning/benchmarks.py")},
        "limits": ["Both map banks were previously exposed; this is descriptive post-selection evidence.",
                   "P1 compares actual global-stage 656 NFE and final 820 NFE, not equal-budget removal of polishing.",
                   "P2 uses selected-path in-search estimates; independent external scoring has Monte Carlo error; not unbiased estimator calibration or a safety certificate.",
                   "Maps are the grouping unit. Planner seeds and perturbation trials are nested/reused, not independent experimental maps.",
                   "P4 reuses already observed perturbations; noise-model stress is not an unseen geometry evaluation.",
                   "P5 describes polyline geometry without time, velocities, accelerations or flight-controller validation.",
                   "Equal NFE is not equal wall-clock cost. No timing superiority inferred from existing run times.",
                   "P0, P3 and P6 remain separate experiments, not results produced here."]}
    write_json(out / "evidence_manifest.json", source_manifest)
    if args.smoke:
        write_json(out / "completion_receipt.json", {"status":"complete_smoke", "source_results":len(receipts), "new_optimizations":0, "scope":"Smoke only; four saved paths; no study conclusions"})
        return
    lookup = {(label, row["method"], row["scope"]): row for label, rows in outputs.items() for row in rows}
    scope = "pooled_known_banks"
    main1, main2, main5 = (lookup[(label, "full", scope)] for label in ("P1", "P2", "P5"))
    lines = ["# 定稿主方法：P1、P2、P4、P5 已保存结果分析", "",
        "主方法为锁定 soft_beta050。读取既有结果，不重新运行或调参任何优化算法。", "",
        "24 张已使用地图，两个图库各 12 张；每个随机方法各 480 次规划（每图库 240 次），A* 每图库 12 条独立路径、合计24条。所有随机方法 NFE=820，内部样本 K=16；每条最终路径每种外部条件已有500次扰动。", "",
        "所有汇总先对同一地图内规划种子平均，再对地图等权平均。CSV 保留每条路径、每张地图、每个图库及合并结果，不将480条当成480张独立地图。", "",
        "## P1 末端精修实际做了什么", "",
        "这是同一次运行中全局搜索结束（656 NFE）与精修结束（820 NFE）的配对过程分析；并非相同820预算下去掉精修的E4反事实。精修前路径用冻结代码恢复，独立评分复用相同外部扰动；精修后σ=.6的500位碰撞记录逐位复算匹配。", "",
        "| 方法 | 前路径 | 后路径 | 缩短 | CFR .6前 | CFR .6后 | 风险梯度调用比例 | 平均接受步数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("full", "risk_polish_lshade"):
        r=lookup[("P1",method,scope)]
        lines.append(f"| {method} | {r['length_before']:.4f} | {r['length_after']:.4f} | {r['length_reduction']:.4f} | {100*r['cfr_before_0.6']:.2f}% | {100*r['cfr_after_0.6']:.2f}% | {100*r['gradient_probes_used']:.2f}% | {r['accepted_steps']:.2f} |")
    lines += ["", f"主方法平均精修风险梯度探测 {main1['gradient_probes']:.2f} 次，路径试探 {main1['line_trials']:.2f} 次。所有探测均计入总预算；终端缩短不构成真实风险不升的保证。", "",
        "## P2 搜索风险与外部碰撞的差距", "",
        f"主方法搜索内估计碰撞率均值 {100*main2['internal_collision_estimate']:.3f}%，外部σ=.6碰撞比例 {100*main2['external_collision_fraction_06']:.3f}%，外部减内部均值 {100*main2['external_minus_internal']:.3f} 个百分点。", "",
        f"搜索内满足α_search=.10门限的占比 {100*main2['internal_gate_met']:.2f}%，外部达到CFR≥82.5%的经验占比 {100*main2['external_empirical_target_met']:.2f}%。两个门限含义不同；不能把内部门限通过率当作安全认证。", "",
        "这里只描述优化后被选中路径的估计落差，包含优化选择效应与外部500次蒙特卡洛误差；不等同于固定路径重新独立采样的无偏性/方差实验（P0）。不将共享扰动和同图规划种子当成独立地图计算显著性。", "",
        "## P4 扰动强度与分布变化", "",
        "| 方法 | 高斯 .3 | 高斯 .6 | 高斯 .9 | 均匀 .6 | t3 .6 | 共同平移 .6 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ("full", "risk_polish_lshade", "risk_lshade", "risk_de", "bare_de", "bare_lshade", "cand_tail_risk"):
        if ("P4",method,scope) not in lookup:
            continue
        r=lookup[("P4",method,scope)]
        keys=("cfr_0.3","cfr_0.6","cfr_0.9","cfr_uniform_0.6","cfr_student_t3_0.6","cfr_common_translation_0.6")
        lines.append("| "+method+" | "+" | ".join(f"{100*r[k]:.2f}%" for k in keys)+" |")
    lines += ["", "所有列的分母保留全部规划尝试，名义失败记0；路径不重新规划。均匀与t3按单位方差缩放，共同平移改变障碍之间的相关性。这是合成扰动失配分析，不是真实传感器建模，也不是第三图库盲测。", "",
        "## P5 几何形状与失败地图分布", "",
        "| 方法 | 名义成功 | 有定义转角的路径 | 平均转角/度 | 平均每路径最大转角/度 | >60度路径占比 | >90度路径占比 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ("full","risk_polish_lshade","bare_de","bare_lshade","bare_astar"):
        r=lookup[("P5",method,scope)]
        actual=sum(x["defined_turns"] > 0 for x in p5 if x["method"]==method)
        lines.append(f"| {method} | {100*r['nominal_success']:.2f}% | {actual} | {r['mean_turn_deg']:.2f} | {r['max_turn_deg']:.2f} | {100*r['has_turn_gt60']:.2f}% | {100*r['has_turn_gt90']:.2f}% |")
    lines += ["", "转角阈值比例表示一条路径至少有一个转角超过阈值，分母是转角有定义的返回路径，包含名义不可行返回路径；缺失路径或无定义转角不补零。各路径平均转角先在路径内计算。上表最大转角列是各路径最大值的地图等权平均，并非全数据极值。", "",
        "这些量只说明折线几何，不验证速度、加速度、最小转弯半径、跟踪或真实飞行。坐标长度不宣称米。", "",
        "P5_per_map.csv 列出每张图的名义成功率、CFR和未达外部目标比例，完整保留不利地图。", "",
        "## 尚未由本分析完成的实验", "",
        "P0：固定路径独立重复采样，比较条件估计与IID估计的误差、方差及成本。P3：围绕冻结主方法的预算与β敏感性，需要新规划。P6：冻结最终协议后新的分布偏移图库确认，旧OOD先导图库不可重新称盲测。", "",
        "本次使用当前正式E协议的匹配控制，不按这些描述性诊断更改主方法。", "",
        "原始输入哈希见source_result_hashes.csv与evidence_manifest.json；本目录CSV和文字均由同一脚本生成。"]
    (out/"P实验_已完成证据.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),"new_optimization_calls":0,"source_results":len(receipts),
                      "main_plans":counts["full"],"validation":source_manifest["validation"],
                      "main_P1":main1,"main_P2":main2},ensure_ascii=False))


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
