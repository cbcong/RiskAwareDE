"""Complete E0--E6 for soft_beta050, reusing compatible committed evidence.

This is a control repair, not a main-method or candidate-search revision.
Never modify the source run. Every reused checkpoint retains its provenance.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

SETTINGS = "configs/paper_evidence_study.json"
REUSED = ("full", "nominal_search", "no_risk", "bare_de", "bare_jade",
          "bare_shade", "bare_lshade", "bare_pso", "bare_astar")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_protocol(repo):
    from seven_category_protocol import build_protocol as base_protocol
    repo = Path(repo)
    settings = read(repo / SETTINGS)
    source = Path(settings["reuse_run"]).resolve()
    old = read(source / "protocol.json")
    if read(source / "state.json")["status"] != "complete":
        raise ValueError("Evidence reuse requires a completed source run")
    protocol = base_protocol(repo, "candidate")
    protocol["methods"] = {k: v for k, v in protocol["methods"].items()
                           if v["category"] != "CANDIDATE"}
    terminal = {k: v for k, v in protocol["methods"]["full"]["options"].items()
                if k not in {"global_method", "risk_estimator"}}
    beta = terminal["beta"]
    for family in ("de", "jade", "shade", "lshade", "pso"):
        for prefix, experiment_kind in (("risk_", "risk_baseline"),
                                        ("risk_polish_", "projected_baseline")):
            spec = protocol["methods"][prefix + family]
            opts = terminal if experiment_kind == "projected_baseline" else {"axis": 2, "beta": beta}
            spec.update(kind="paper_matched", options={**opts, "family": family,
                        "experiment_kind": experiment_kind},
                        adaptation="same conditional estimator/gate/within-gate soft cost; native optimizer; exact shared archive key; outside-gate scalar adapter")
            spec["label"] += "（软风险权重与主方法一致）"
    protocol["methods"]["no_terminal_polish"].update(
        kind="paper_matched", options={"experiment_kind": "no_terminal", "axis": 2, "beta": beta},
        adaptation="main global search at full 820 NFE, including beta=.05; no terminal allocation")
    protocol["methods"]["binary_risk"].update(
        kind="paper_matched", options={**terminal, "experiment_kind": "binary"},
        adaptation="only replace conditional estimator by binary IID; beta and finite-difference terminal preserved; includes estimator-terminal interaction")
    # Reuse only identical input experiments; labels do not establish equality.
    for key in ("benchmark_config", "map_roots", "planner_seeds", "scenarios",
                "external_samples", "external_seed", "sigmas", "NFE", "K",
                "search_alpha", "target_alpha", "distribution_external_models",
                "distribution_external_sigma", "main_method_bundle_sha256"):
        if protocol[key] != old[key]:
            raise ValueError(f"Source run is incompatible for reuse: {key}")
    for method in REUSED:
        if protocol["methods"][method] != old["methods"][method]:
            raise ValueError(f"Source method is incompatible for reuse: {method}")
    source_manifest = read(source / "execution_manifest.json")
    # All pre-existing core scientific implementations must be unchanged.
    for name, expected in source_manifest["source_sha256"].items():
        if name.startswith("src/") and sha(repo / name) != expected:
            raise ValueError(f"Scientific implementation changed; cannot silently reuse: {name}")
    protocol.update(protocol_version="paper-matched-soft050-v1", profile="manuscript",
                    study_kind="locked_main_paper_controls",
                    reuse={"run": str(source), "methods": list(REUSED),
                           "protocol_sha256": sha(source / "protocol.json"),
                           "manifest_sha256": sha(source / "execution_manifest.json")})
    protocol["limits"] = [x for x in protocol["limits"] if "Candidate profile" not in x]
    protocol["limits"].extend([
        "Main remains exactly soft_beta050; only matched controls are repaired.",
        "Native baseline proposal/adaptation mechanics are retained. Outside-gate scalarization differs from DE lexicographic selection; E3 compares global-search packages, not a single mutation rule.",
        "E5 removes the risk-dependent soft term by definition and retains nominal terminal polish.",
        "E6 includes the interaction of a discrete estimator with unchanged finite-difference polish; fixed-path variance is tested separately in P0.",
        "Some results are inherited from the recorded source run. Cross-session runtime is descriptive, not an efficiency superiority test.",
    ])
    protocol["statistical_analysis"]["multiple_testing"] = "23 category contrasts x 2 metrics x 2 banks, one 92-test exploratory Holm family; no candidate contrasts"
    protocol["expected_plans"] = sum(
        len(protocol["scenarios"][str(root)]) *
        sum(1 if spec["kind"] == "astar" else len(protocol["planner_seeds"])
            for spec in protocol["methods"].values()) for root in protocol["map_roots"])
    return deepcopy(protocol)


def import_reused_results(directory, protocol, jobs):
    """Copy immutable checkpoints once; resume then uses this run's own files."""
    from seven_category_control import checkpoint_digest, write_json
    from seven_category_protocol import minimalvalidate_result
    directory = Path(directory)
    source = Path(protocol["reuse"]["run"])
    if sha(source / "protocol.json") != protocol["reuse"]["protocol_sha256"]:
        raise ValueError("Source protocol changed after reuse was planned")
    if sha(source / "execution_manifest.json") != protocol["reuse"]["manifest_sha256"]:
        raise ValueError("Source execution manifest changed after reuse was planned")
    receipts, counts = [], {}
    for job in jobs:
        if job["method"] not in protocol["reuse"]["methods"]:
            continue
        path = source / "results" / (job["id"] + ".json")
        bundle = read(path)
        if bundle.get("checkpoint_sha256") != checkpoint_digest(bundle):
            raise ValueError("Invalid source checkpoint: " + job["id"])
        minimalvalidate_result(job, protocol, bundle)
        target = directory / "results" / path.name
        shutil.copy2(path, target)
        receipts.append({"job_id": job["id"], "source": str(path), "sha256": sha(target)})
        counts[job["method"]] = counts.get(job["method"], 0) + 1
    write_json(directory / "reuse_receipt.json", {"source": str(source),
               "reused_plans": len(receipts), "remaining_plans": len(jobs) - len(receipts),
               "source_manifest_sha256": protocol["reuse"]["manifest_sha256"],
               "checkpoints": receipts})
    return len(receipts), counts
