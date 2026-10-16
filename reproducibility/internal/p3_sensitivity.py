"""P3 one-factor sensitivity of the frozen soft_beta050 method; never select a replacement.

Six cells on six mechanically selected known maps and seeds 9..13. Reuse 30
locked-main plans, execute 150 new plans, checkpoint each job atomically.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from reproducibility.cohorts import equivalent


SOURCE = CODE_ROOT / "data/reference/e"
OUTPUT_BASE = CODE_ROOT / "outputs"
CELLS = (("beta000_nfe820", 0., 820), ("beta025_nfe820", .025, 820),
         ("beta050_nfe820", .05, 820), ("beta100_nfe820", .1, 820),
         ("beta050_nfe420", .05, 420), ("beta050_nfe1640", .05, 1640))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


from reproducibility.atomic_io import write_json as atomic_json


def canonical(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def csv_out(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_protocols(source):
    origin = json.loads((source / "protocol.json").read_text(encoding="utf-8-sig"))
    if origin["methods"]["full"]["kind"] != "soft_projected" or origin["methods"]["full"]["options"]["beta"] != .05:
        raise ValueError("Expected frozen soft_beta050 source")
    if origin["search_alpha"] != .1 or origin["K"] != 16:
        raise ValueError("Unexpected locked search-risk contract")
    protocols, jobs = {}, []
    old_jobs = json.loads((source / "jobs.json").read_text(encoding="utf-8-sig"))
    source_ids = {(j["map_root"], j["scene"], j["seed"]): j["id"] for j in old_jobs if j["method"] == "full"}
    for label, beta, nfe in CELLS:
        p = copy.deepcopy(origin)
        p["methods"] = {"full": copy.deepcopy(origin["methods"]["full"])}
        p["methods"]["full"]["options"]["beta"] = beta
        p["methods"]["full"]["label"] = label
        p["main_method"]["options"] = copy.deepcopy(p["methods"]["full"]["options"])
        p["main_method"]["NFE"] = nfe
        p["NFE"] = nfe
        p["planner_seeds"] = list(range(9, 14))
        p["distribution_external_models"] = []
        p["distribution_external_sigma"] = None
        p["profile"] = "locked_main_P3_one_factor_sensitivity"
        p["protocol_version"] = "locked-main-P3-v1"
        p["scenarios"] = {str(root): p["scenarios"][str(root)][:3] for root in p["map_roots"]}
        p["benchmark_config"]["seeds"] = list(range(9, 14))
        # A larger stopping ceiling is needed at 1640. Under the fixed gate,
        # generations is only a ceiling; it does not schedule risk thresholds.
        for key in list(p["benchmark_config"]):
            if key.endswith("_max_evaluations") and p["benchmark_config"][key] is not None:
                p["benchmark_config"][key] = nfe
        for prefix in ("risk_aware", "de", "jade", "shade", "lshade", "adaptive"):
            genkey, popkey = prefix + "_generations", prefix + "_population"
            if genkey in p["benchmark_config"]:
                population = p["benchmark_config"].get(popkey, 20)
                p["benchmark_config"][genkey] = max(origin["benchmark_config"][genkey], math.ceil((nfe-population)/population))
        p["benchmark_config"]["pso_iterations"] = max(origin["benchmark_config"]["pso_iterations"], math.ceil((nfe-20)/20))
        p["expected_plans"] = 30
        protocols[label] = p
        for root in p["map_roots"]:
            for scene_index, scene in enumerate(p["scenarios"][str(root)]):
                for seed in p["planner_seeds"]:
                    jobs.append({"id": f"{label}_r{root}_m{scene_index:02d}_s{seed:04d}",
                                 "map_root": root, "scene": scene["name"], "method": "full", "seed": seed,
                                 "cell": label, "beta": beta, "nfe": nfe,
                                 "reuse": label == "beta050_nfe820", "source_id": source_ids[(root, scene["name"], seed)]})
    random.Random(2026101003).shuffle(jobs)
    return protocols, jobs


def summarize(output, jobs, protocols):
    rows = []
    hashes = []
    for job in jobs:
        path = output / "results" / (job["id"] + ".json")
        item = json.loads(path.read_text(encoding="utf-8-sig"))
        r = item["row"]
        rows.append({"cell":job["cell"], "beta":job["beta"], "nfe":job["nfe"],
                     "map_root":job["map_root"], "scene":job["scene"], "seed":job["seed"],
                     "reused":job["reuse"], "nominal_success":r["nominal_success"],
                     "length_feasible":r["length"] if r["nominal_success"] else None,
                     "cfr03":r["cfr_0.3"], "cfr06":r["cfr_0.6"], "cfr09":r["cfr_0.9"],
                     "pass06":r["pass_0.6"], "internal_risk":r["internal_risk"], "risk_checks":r["risk_checks"],
                     "runtime_new_only_sec":r["runtime_sec"] if not job["reuse"] else None,
                     "terminal_nfe":job["nfe"]-item["detail"]["extension_metadata"]["global_stage"]["evaluations"]})
        hashes.append({"result_file":path.name,"sha256":digest(path)})
    metrics = ("nominal_success","length_feasible","cfr03","cfr06","cfr09","pass06","internal_risk","risk_checks","terminal_nfe")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["cell"],row["map_root"],row["scene"])].append(row)
    by_map=[]
    for (cell,root,scene), vals in sorted(grouped.items()):
        output_row={"cell":cell,"map_root":root,"scene":scene,"plans":len(vals)}
        for metric in metrics:
            values=[float(v[metric]) for v in vals if v[metric] is not None]
            output_row[metric]=statistics.fmean(values) if values else None
        by_map.append(output_row)
    summary=[]
    for label,beta,nfe in CELLS:
        maps=[r for r in by_map if r["cell"]==label]
        r={"cell":label,"beta":beta,"nfe":nfe,"maps":len(maps),"plans":sum(x["plans"] for x in maps),"reused_plans":30 if label=="beta050_nfe820" else 0}
        for metric in metrics:
            vals=[m[metric] for m in maps if m[metric] is not None]
            r[metric]=statistics.fmean(vals) if vals else None
        summary.append(r)
    csv_out(output/"P3_per_path.csv",rows)
    csv_out(output/"P3_per_map.csv",by_map)
    csv_out(output/"P3_summary.csv",summary)
    csv_out(output/"P3_result_hashes.csv",hashes)
    text=["# P3 定稿主方法的β与预算敏感性", "",
          "这是围绕已锁定soft_beta050的局部敏感性研究，不选择新的主方法、不据此改参。所有预声明的六个配置均报告。", "",
          "固定取两个已曝光图库各自冻结顺序的前三张图，规划种子9–13：6图×5种子=30次/配置。地图和种子选择不依赖本轮结果，仍属开发证据，不能当作未见地图确认。β=.05、NFE820的30条直接复用上一轮，其余五配置共150次新规划。", "",
          "固定K=16、σ_search=.6、α_search=.10、固定z条件估计、精修预算比例20%、内部几何捷径以及其他算法设置。外部每条路径500次扰动，σ=.3/.6/.9，沿用原冻结随机扰动种子。", "",
          "预算改变时维持20%精修比例；1640预算将generation上限提高到81，以免40代提前截断。风险门限固定，因此代数仅为不提前终止的上限，不增加门限退火机制。", "",
          "| β | 总NFE | 平均路径 | 名义成功率 | CFR .3 | CFR .6 | CFR .9 | 外部经验达标 | 内部风险项数 | 结果来源 |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary:
        text.append(f"| {r['beta']:.3f} | {r['nfe']} | {r['length_feasible']:.4f} | {100*r['nominal_success']:.2f}% | {100*r['cfr03']:.2f}% | {100*r['cfr06']:.2f}% | {100*r['cfr09']:.2f}% | {100*r['pass06']:.2f}% | {r['risk_checks']:.0f} | {'复用30条' if r['reused_plans'] else '新跑30条'} |")
    text += ["", "平均路径先对每张图名义成功的路径平均，再对地图等权平均；失败的CFR记0，所有规划保留在成功率与CFR分母中。外部经验达标指CFR≥82.5%，不是安全证书。", "",
             "等NFE不等于等墙钟。旧主方法计时来自旧执行环境，故不与本次新跑配置做耗时排名。每路径CSV仅保留新规划的实测时间用于追溯，不能据此作效率优越性结论。不同预算从头规划，非一次轨迹的不同截点。", "",
             "本研究不做观察结果后挑选显著性检验；六张图样本有限，只展示敏感性，不据此证明稳健等价、最优参数或大范围泛化。", "",
             "协议、源代码哈希、任务矩阵在protocol.json；每个任务原子保存，--resume读取并验证已有记录后只补缺失任务。"]
    (output/"P3_敏感性结果.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--source-run",type=Path,default=SOURCE)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    source=args.source_run.resolve()
    output=(args.output or OUTPUT_BASE/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_P3定稿敏感性")).resolve()
    frozen=CODE_ROOT
    scientific={str(p.relative_to(frozen)).replace("\\","/"):digest(p) for p in sorted((frozen/"src/uav_path_planning").glob("*.py"))}
    for name in ("scripts/seven_category_protocol.py","scripts/run_extension_candidate.py","scripts/run_frozen_stress_suite.py"):
        scientific[name]=digest(frozen/name)
    sys.path.insert(0,str(frozen/"src"))
    sys.path.insert(0,str(frozen/"scripts"))
    from seven_category_protocol import execute_job,minimalvalidate_result
    protocols,jobs=make_protocols(source)
    if args.smoke:
        origin=json.loads((source/"protocol.json").read_text(encoding="utf-8-sig"))
        root=origin["map_roots"][0]
        scene=origin["scenarios"][str(root)][0]["name"]
        jobs=[next(j for j in jobs if j["cell"]==label and j["map_root"]==root
                   and j["scene"]==scene and j["seed"]==9) for label,_,_ in CELLS]
    contract={"schema":"locked-main-P3-v1", "smoke":args.smoke,"source_run":str(source),"source_protocol_sha256":digest(source/"protocol.json"),
              "runner_sha256":digest(Path(__file__)),"frozen_scientific_source_sha256":scientific,
              "selection_rule":"first three frozen-order maps per known bank; planner seeds9..13; no outcome-based selection",
              "protocols":protocols,"jobs":jobs,"reused_plans":sum(j["reuse"] for j in jobs),"new_plans":sum(not j["reuse"] for j in jobs),
              "purpose":"one-factor sensitivity only; all cells reported; main-method lock unchanged"}
    if args.resume:
        saved=json.loads((output/"protocol.json").read_text(encoding="utf-8-sig"))
        if canonical(saved)!=canonical(contract):
            raise ValueError("Resume protocol, source or runner changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output already contains files; use --resume with unchanged protocol")
        (output/"results").mkdir(parents=True,exist_ok=True)
        atomic_json(output/"protocol.json",contract)
    done=new_count=reused_count=0
    atomic_json(output/"state.json",{"status":"running","done":done,"total":len(jobs),"started_or_resumed_at":datetime.now().isoformat()})
    print(json.dumps({"output":str(output),"total":len(jobs),"new_plans":sum(not j["reuse"] for j in jobs),"reused_plans":sum(j["reuse"] for j in jobs),"smoke":args.smoke},ensure_ascii=False),flush=True)
    for job in jobs:
        p=protocols[job["cell"]]
        result_path=output/"results"/(job["id"]+".json")
        if result_path.exists():
            bundle=json.loads(result_path.read_text(encoding="utf-8-sig"))
        elif job["reuse"]:
            source_path=source/"results"/(job["source_id"]+".json")
            bundle=json.loads(source_path.read_text(encoding="utf-8-sig"))
            bundle["distribution_external_trials"]={}
            allowed={f"{prefix}_{sigma}" for sigma in p["sigmas"] for prefix in ("cfr","pass")}
            for key in list(bundle["row"]):
                if (key.startswith("cfr_") or key.startswith("pass_")) and key not in allowed:
                    del bundle["row"][key]
            bundle["detail"]["row"]=copy.deepcopy(bundle["row"])
            bundle.pop("checkpoint_sha256",None)
            bundle["P3_provenance"]={"reused":True,"source_result":str(source_path),"source_sha256":digest(source_path)}
            minimalvalidate_result(job,p,bundle)
            atomic_json(result_path,bundle)
            reused_count+=1
        else:
            bundle=execute_job(job,p)
            bundle["P3_provenance"]={"reused":False,"cell":job["cell"]}
            atomic_json(result_path,bundle)
            new_count+=1
        minimalvalidate_result(job,p,bundle)
        done+=1
        atomic_json(output/"state.json",{"status":"running","done":done,"total":len(jobs),"new_plans_this_invocation":new_count,"reused_plans_this_invocation":reused_count,"updated_at":datetime.now().isoformat()})
        if done%15==0:
            print(f"P3 completed {done}/{len(jobs)}",flush=True)
    if args.smoke:
        atomic_json(output/"state.json", {"status":"complete_smoke", "done":done, "total":len(jobs), "new_plans":new_count, "reused_plans":reused_count, "scope":"Smoke only; one task per sensitivity cell"})
        return
    summary=summarize(output,jobs,protocols)
    atomic_json(output/"state.json",{"status":"complete","done":done,"total":len(jobs),"new_plans_this_invocation":new_count,"reused_plans_this_invocation":reused_count,"completed_at":datetime.now().isoformat()})
    print(json.dumps({"output":str(output),"status":"complete","summary":summary},ensure_ascii=False),flush=True)


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
