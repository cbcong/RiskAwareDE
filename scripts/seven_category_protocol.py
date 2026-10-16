"""Frozen E0--E6 and diverse-candidate protocols, independent of lifecycle.

The ordinary profile retains the historical ConditionalProjectedPolish study.
The candidate profile loads the verified soft_beta050 lock as `full`, then
compares the declared mechanism suite without selecting a winner in advance.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
import json
import hashlib
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from uav_path_planning import make_research_scenarios, split_research_scenarios
from uav_path_planning.benchmarks import (
    BenchmarkConfig, _problem, _result_for_continuous, _result_for_astar,
)
from uav_path_planning.conditional_projected_polish import ConditionalProjectedPolish
from uav_path_planning.conditional_risk_de import ConditionalRiskDE
from uav_path_planning.models import Path3D
from uav_path_planning.nominal_search_ablation import NominalSearchDE, NominalProjectedPolish
from uav_path_planning.robustness import translate_environment
from uav_path_planning.scenarios import Scenario3D
from run_extension_candidate import risk_config
from run_frozen_stress_suite import collision_free_batch, make_shift_bank
from run_six_category_comparison import method_specifications

FAMILIES = ("de", "jade", "shade", "lshade", "pso")
PROTOCOL_VERSION = "seven-categories-v1"
CANDIDATE_PROTOCOL_VERSION = "candidate-comparison-v2-diverse-mechanisms"


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an explicit positive integer")
    return value


def _unique_integers(values, name: str) -> list[int]:
    if (not isinstance(values, list) or not values or
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values) or
            len(set(values)) != len(values)):
        raise ValueError(f"{name} must be a nonempty list of distinct nonnegative integers")
    return list(values)


def _validated_study(repo_root: Path, *, profile: str = "full") -> tuple[dict, dict, Path]:
    study = json.loads((repo_root / "configs/seven_category_study.json").read_text(encoding="utf-8-sig"))
    if profile == "candidate":
        pointer_file = (repo_root / "configs/main_method_lock.json").resolve()
        pointer = json.loads(pointer_file.read_text(encoding="utf-8-sig"))
        canonical_lock_file = Path(pointer["lock_file"]).resolve()
        if not canonical_lock_file.is_file():
            raise ValueError("Canonical soft_beta050 lock file is missing")
        if hashlib.sha256(canonical_lock_file.read_bytes()).hexdigest() != pointer.get("lock_sha256"):
            raise ValueError("Canonical soft_beta050 lock file digest does not match its local pointer")
        lock = json.loads(canonical_lock_file.read_text(encoding="utf-8-sig"))
        if lock.get("method_id") != "soft_beta050" or lock.get("module") != "uav_path_planning.soft_risk_projected_polish":
            raise ValueError("Candidate matrix requires the frozen soft_beta050 main-method lock")
        bundle_file = (repo_root / pointer["bundle"]).resolve()
        bundle_digest = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
        if bundle_digest != lock.get("bundle_sha256") or bundle_digest != pointer.get("bundle_sha256"):
            raise ValueError("Locked main-method bundle digest does not match main_method_lock.json")
        lock_file = canonical_lock_file
    else:
        lock_file = None
        bundle_file = (repo_root / study["bundle_path"]).resolve()
    if bundle_file.parent != (repo_root / "configs").resolve() or bundle_file.suffix != ".json":
        raise ValueError("bundle_path must select a JSON directly inside this repository's configs directory")
    bundle = json.loads(bundle_file.read_text(encoding="utf-8-sig"))
    allowed_modules = {"uav_path_planning.conditional_projected_polish"}
    if profile == "candidate":
        allowed_modules.add("uav_path_planning.soft_risk_projected_polish")
    if bundle.get("module") not in allowed_modules:
        raise ValueError("Changing the optimizer module requires an explicit protocol revision")
    options = bundle.get("extension_options", {})
    allowed = {"reserve_fraction", "active_margin", "initial_step", "axis", "global_method",
               "projection", "risk_estimator", "max_line_steps"}
    if profile == "candidate":
        allowed.add("beta")
    if set(options) - allowed:
        raise ValueError("Unsupported optimizer options require an explicit protocol revision")
    for key, expected in {"axis": 2, "global_method": "risk_de", "projection": "tangent",
                          "risk_estimator": "conditional"}.items():
        if options.get(key, expected) != expected:
            raise ValueError(f"Protocol v1 requires {key}={expected!r}; revise the protocol for this mechanism change")
    for key, default, minimum, maximum, inclusive_min in (
        ("reserve_fraction", .2, 0, 1, False),
        ("initial_step", .03, 0, 1, False),
        ("active_margin", .02, 0, 1, True),
    ):
        value = options.get(key, default)
        valid_min = value >= minimum if inclusive_min else value > minimum
        if not math.isfinite(value) or not valid_min or value > maximum or (key == "reserve_fraction" and value == 1):
            raise ValueError(f"Invalid terminal option: {key}")
    _positive_integer(options.get("max_line_steps", 1), "max_line_steps")
    if profile == "candidate" and options.get("beta") != .05:
        raise ValueError("Candidate protocol requires the frozen soft_beta050 beta=.05")
    _positive_integer(study["external_samples"], "external_samples")
    for key in ("external_seed", "task_shuffle_seed"):
        _unique_integers([study[key]], key)
    sigmas = study["sigmas"]
    if (not isinstance(sigmas, list) or not sigmas or
            any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0 for value in sigmas) or
            len(set(sigmas)) != len(sigmas) or set(sigmas) != {.3, .6, .9}):
        raise ValueError("Protocol v1 requires exactly sigmas 0.3, 0.6, 0.9; other stress levels need a protocol revision")
    if profile == "candidate":
        study["main_method_lock"] = str(lock_file)
        study["main_method_lock_sha256"] = hashlib.sha256(lock_file.read_bytes()).hexdigest()
        study["main_method_bundle_sha256"] = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
    return study, bundle, bundle_file


def _candidate_methods(methods: dict, options: dict) -> None:
    """Declare a diverse mechanism matrix against the frozen full method."""
    terminal = {key: options[key] for key in (
        "reserve_fraction", "active_margin", "initial_step", "axis",
        "projection", "max_line_steps",
    ) if key in options}

    def candidate(label, *, global_search="de", initialization="uniform",
                  mutation="standard", risk_model="conditional_mean",
                  risk_weight=.5, terminal_method="tangent", credit=.01):
        return {
            "category": "CANDIDATE", "label": label,
            "kind": "research_candidate",
            "options": {**terminal, "global_search": global_search,
                        "initialization": initialization, "mutation": mutation,
                        "risk_model": risk_model, "risk_weight": risk_weight,
                        "terminal_method": terminal_method,
                        "risk_credit_fraction": credit, "beta": .05},
            "risk_estimator": "candidate-path-risk-score",
            "adaptation": "same NFE/K/external scoring; exploratory mechanism candidate; no safety certificate",
        }

    methods.update({
        "cand_topology": candidate("候选：六向绕障拓扑播种", initialization="topology_seeds"),
        "cand_local_mutation": candidate("候选：障碍邻近航点定向变异", mutation="risk_localized"),
        "cand_cem": candidate("候选：交叉熵全局搜索替换 DE", global_search="cem"),
        "cand_tail_risk": candidate("候选：条件风险上尾排序", risk_model="conditional_tail"),
        "cand_uncertainty": candidate("候选：风险均值加不确定性惩罚", risk_model="uncertainty_score", risk_weight=.5),
        "cand_common_shift": candidate("候选：整张地图共模位移风险", risk_model="common_translation"),
        "cand_distribution_robust": candidate("候选：高斯/均匀/t 分布最坏风险", risk_model="distribution_mixture"),
        "cand_risk_credit": candidate("候选：风险额度驱动末端路径编辑", terminal_method="risk_credit"),
        "cand_pathwise_ledger": candidate("候选：路径级联合碰撞风险账本",
                                           risk_model="pathwise_ledger"),
        "cand_ledger_mutation": candidate("组合：路径风险账本指导全局变异",
                                           risk_model="pathwise_ledger",
                                           mutation="risk_localized"),
        "cand_ledger_credit": candidate("组合：路径风险账本指导风险额度精修",
                                         risk_model="pathwise_ledger",
                                         terminal_method="risk_credit"),
        "cand_integrated_ledger": candidate("整合：风险账本贯穿全局搜索与末端精修",
                                             risk_model="pathwise_ledger",
                                             mutation="risk_localized",
                                             terminal_method="risk_credit"),
        "cand_cem_ledger_credit": candidate("整合：交叉熵 + 路径风险账本 + 风险额度精修",
                                             global_search="cem",
                                             risk_model="pathwise_ledger",
                                             mutation="risk_localized",
                                             terminal_method="risk_credit"),
        "cand_topology_local": candidate("组合：拓扑播种 + 障碍邻近变异",
                                           initialization="topology_seeds", mutation="risk_localized"),
        "cand_cem_tail": candidate("组合：交叉熵搜索 + 条件风险上尾",
                                    global_search="cem", risk_model="conditional_tail"),
        "cand_topology_mixture_credit": candidate("组合：拓扑搜索 + 分布鲁棒风险 + 风险额度编辑",
                                                   initialization="topology_seeds",
                                                   risk_model="distribution_mixture",
                                                   terminal_method="risk_credit"),
        "cand_cem_mixture_credit": candidate("组合：交叉熵搜索 + 分布鲁棒风险 + 风险额度编辑",
                                              global_search="cem",
                                              risk_model="distribution_mixture",
                                              terminal_method="risk_credit"),
    })


def build_protocol(repo_root: Path, profile: str = "full") -> dict:
    """Freeze all scientific choices; this function never starts optimization."""
    if profile == "manuscript":
        from paper_study_protocol import build_protocol as paper_protocol
        return paper_protocol(repo_root)
    if profile not in ("full", "smoke", "candidate"):
        raise ValueError("profile must be full, smoke, or candidate")
    repo_root = Path(repo_root).resolve()
    study, bundle, bundle_file = _validated_study(repo_root, profile=profile)
    seeds = _unique_integers(study["planner_seeds"], "planner_seeds")
    roots = _unique_integers(study["map_roots"], "map_roots")
    if profile == "smoke":
        seeds, roots = seeds[:1], roots[:1]
    cfg = asdict(replace(BenchmarkConfig(**bundle["benchmark_config"]), seeds=tuple(seeds)))
    budget_keys = [family + "_max_evaluations" for family in FAMILIES] + ["risk_aware_max_evaluations"]
    budgets = {_positive_integer(cfg[key], key) for key in budget_keys}
    if len(budgets) != 1:
        raise ValueError("All five baseline and proposed-method maximum evaluation budgets must be equal")
    nfe = budgets.pop()
    k = _positive_integer(cfg["risk_aware_perturbation_samples"], "risk_aware_perturbation_samples")
    if cfg["risk_aware_local_refinement"] is not True or cfg["risk_aware_local_refinement_mode"] != "partial_shortcut":
        raise ValueError("E0 explicitly retains the internal partial shortcut; disable/change it in a separate protocol")
    if cfg["risk_aware_gate_schedule"] != "fixed" or cfg["risk_aware_selection_mode"] != "risk_budgeted":
        raise ValueError("This protocol requires a fixed empirical gate and risk_budgeted selection")
    if cfg["risk_aware_skip_risk_for_nominal_infeasible"]:
        raise ValueError("Skipping risk evaluations would invalidate the shared NFE*K budget")
    search_alpha, target_alpha = cfg["risk_aware_chance_alpha"], bundle["target_alpha"]
    if any(not math.isfinite(value) or not 0 < value < 1 for value in (search_alpha, target_alpha)):
        raise ValueError("Search and external target alpha must be finite values strictly between zero and one")
    for family in FAMILIES + ("risk_aware",):
        population = cfg["pso_particles"] if family == "pso" else cfg[family + "_population"]
        generations = cfg["pso_iterations"] if family == "pso" else cfg[family + "_generations"]
        _positive_integer(population, family + " population")
        if nfe < population or (family != "lshade" and nfe > population * (generations + 1)):
            raise ValueError(f"{family}: population/generation cap cannot consume the requested NFE budget")
    options = dict(bundle.get("extension_options", {}))
    locked_beta = options.pop("beta", None) if profile == "candidate" else None
    nominal_options = {key: options[key] for key in ("reserve_fraction", "initial_step", "max_line_steps") if key in options}
    nominal_options.setdefault("reserve_fraction", .2)
    methods = method_specifications()
    methods["full"]["options"] = dict(options)
    methods["binary_risk"]["options"] = {**options, "risk_estimator": "binary"}
    methods["no_risk"]["options"] = nominal_options
    for family in FAMILIES:
        methods["risk_polish_" + family]["options"] = {**options, "global_method": "baseline_" + family}
    methods["nominal_search"] = {
        "category": "E0", "label": "仅全局搜索（保留内部捷径）",
        "kind": "nominal_search", "risk_estimator": None,
        "adaptation": "nominal-success; internal shortcut retained; no terminal polish",
    }
    if profile == "candidate":
        # The paired reference is the currently locked soft_beta050 method.
        # Keep its stable protocol id (`full`) so E1--E6 and the candidate
        # report point to exactly the same method.
        methods["full"].update(
            label="锁定主方法：soft_beta050（软风险 + 条件风险 + 切向精修）",
            kind="soft_projected",
            options={**options, "beta": locked_beta},
            risk_estimator="conditional-z",
            adaptation="locked main method; fixed gate; beta=.05; original terminal polish",
        )
        _candidate_methods(methods, options)
    bare = ["bare_" + name for name in FAMILIES] + ["bare_astar"]
    categories = {
        "E0": {"label": "搜索底座诊断（保留内部几何捷径；无风险处理、无末端精修）",
               "reference": "nominal_search", "comparators": bare},
        "E1": {"label": "完整方法与普通基线", "reference": "full", "comparators": bare},
        "E2": {"label": "相同条件风险对照", "reference": "full",
               "comparators": ["risk_" + name for name in FAMILIES]},
        "E3": {"label": "相同条件风险加相同末端精修", "reference": "full",
               "comparators": ["risk_polish_" + name for name in FAMILIES]},
        "E4": {"label": "去掉末端精修", "reference": "full", "comparators": ["no_terminal_polish"]},
        "E5": {"label": "去掉全部风险处理", "reference": "full", "comparators": ["no_risk"]},
        "E6": {"label": "条件估计替换为普通 IID", "reference": "full", "comparators": ["binary_risk"]},
    }
    scenes = {}
    for root in roots:
        bank = list(split_research_scenarios(
            make_research_scenarios(root), protocol="research24_6_6_12",
        ).test)
        if profile == "smoke":
            bank = bank[:1]
        scenes[str(root)] = [scene.to_dict() for scene in bank]
    protocol = {
        "protocol_version": CANDIDATE_PROTOCOL_VERSION if profile == "candidate" else PROTOCOL_VERSION,
        "profile": profile,
        "smoke_test": profile == "smoke", "methods": methods,
        "categories": categories, "benchmark_config": cfg,
        "map_roots": roots, "planner_seeds": seeds, "scenarios": scenes,
        "external_samples": study["external_samples"], "external_seed": study["external_seed"],
        "distribution_external_models": (["uniform", "student_t3", "common_translation"]
                                         if profile == "candidate" else []),
        "distribution_external_sigma": .6 if profile == "candidate" else None,
        "sigmas": study["sigmas"], "NFE": nfe, "K": k,
        "search_alpha": search_alpha, "target_alpha": target_alpha,
        "task_shuffle_seed": study["task_shuffle_seed"],
        "study_config": "configs/seven_category_study.json",
        "main_method_lock": study.get("main_method_lock"),
        "main_method_lock_sha256": study.get("main_method_lock_sha256"),
        "main_method_bundle_sha256": study.get("main_method_bundle_sha256"),
        "main_method": {
            "id": "full",
            "module": "uav_path_planning.soft_risk_projected_polish" if profile == "candidate" else "uav_path_planning.conditional_projected_polish",
            "class": "SoftRiskProjectedPolish" if profile == "candidate" else "ConditionalProjectedPolish",
            "conditional_axis": 2,
            "reserve_fraction": options.get("reserve_fraction", .2), "terminal_gradient": "original finite differences",
            "internal_shortcut": True, "experimental_analytic_gradient": False,
            "config_file": bundle_file.relative_to(repo_root).as_posix(),
            "options": methods["full"]["options"] if profile == "candidate" else options,
            "NFE": nfe, "K": k, "search_alpha": search_alpha,
        },
        "limits": [
            "Both map banks have been exposed; this is a reproducibility comparison, not a new blind or OOD test.",
            "Baseline implementations are local reproductions/adaptations, not original-author published values.",
            "A* is a deterministic geometric reference, once per map; its representation and expansion budget differ.",
            "Equal NFE does not imply equal runtime or primitive operation counts; risk terms and planning time are reported.",
            "Empirical external pass rate is not a finite-sample confidence certificate.",
            "E0 has no uncertainty bank and no terminal polish; E5 retains nominal terminal polish.",
            ("Candidate profile uses the verified soft_beta050 lock as full; all declared mechanisms run before any replacement decision." if profile == "candidate" else
             "The original full method is frozen; recent analytic-gradient and margin candidates are not included."),
        ],
        "statistical_analysis": {
            "unit": "map-level paired seed means within each bank",
            "metrics": ["length", "cfr_0.6"],
            "multiple_testing": (f"E0-E6: 23 stochastic category contrasts x 2 metrics x {len(roots)} banks; candidate suite: all candidate x two references x two metrics x {len(roots)} banks in a separate Holm family."
                                 if profile == "candidate" else
                                 f"All 23 stochastic category contrasts x 2 metrics x {len(roots)} banks; {46 * len(roots)}-test Holm family."),
            "astar": "descriptive only; no repeated artificial seeds",
            "interpretation": "Exploratory known-map comparison; nonsignificance is not equivalence.",
        },
    }
    protocol["expected_plans"] = sum(
        len(scenes[str(root)]) * sum(1 if spec["kind"] == "astar" else len(seeds)
                                   for spec in methods.values()) for root in roots
    )
    # Ensure the persisted contract contains only portable JSON data types.
    return json.loads(json.dumps(protocol, ensure_ascii=False, allow_nan=False))


def make_jobs(protocol: dict) -> list[dict]:
    """Return a stable interleaved job matrix; one file can checkpoint one plan."""
    jobs = []
    for root in protocol["map_roots"]:
        for scene_index, scene in enumerate(protocol["scenarios"][str(root)]):
            for method, spec in protocol["methods"].items():
                seeds = [0] if spec["kind"] == "astar" else protocol["planner_seeds"]
                for seed in seeds:
                    jobs.append({
                        "id": f"r{root}_m{scene_index:02d}_{method}_s{seed:04d}",
                        "map_root": root, "scene": scene["name"], "method": method, "seed": seed,
                    })
    if len(jobs) != protocol["expected_plans"] or len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Protocol has an inconsistent or duplicate job matrix")
    random.Random(protocol["task_shuffle_seed"]).shuffle(jobs)
    return jobs


def _scene_data(job: dict, protocol: dict) -> dict:
    matches = [scene for scene in protocol["scenarios"][str(job["map_root"]) ]
               if scene["name"] == job["scene"]]
    if len(matches) != 1:
        raise ValueError("Job scene is absent or duplicated in the frozen protocol")
    return matches[0]


@lru_cache(maxsize=48)
def _scene_and_noise(serialized_scene: str, samples: int, noise_seed: int):
    scene = Scenario3D.from_dict(json.loads(serialized_scene))
    noise = make_shift_bank([scene], "gaussian", samples, noise_seed)[scene.name]
    return scene, noise


@lru_cache(maxsize=192)
def _scene_and_distribution_noise(serialized_scene: str, samples: int, noise_seed: int, model: str):
    scene = Scenario3D.from_dict(json.loads(serialized_scene))
    noise = make_shift_bank([scene], model, samples, noise_seed)[scene.name]
    return scene, noise


def execute_job(job: dict, protocol: dict) -> dict:
    """Execute one plan and independent fixed-path scoring; write no files."""
    scene, noise = _scene_and_noise(
        json.dumps(_scene_data(job, protocol), sort_keys=True, ensure_ascii=False),
        protocol["external_samples"], protocol["external_seed"],
    )
    cfg = BenchmarkConfig(**protocol["benchmark_config"])
    problem = _problem(scene, cfg)
    spec = protocol["methods"][job["method"]]
    # Candidate protocols may vary only the declared risk-search controls
    # (for example p-best or alpha).  Apply those changes to the frozen
    # benchmark config before constructing the per-seed risk config.  The
    # original E0--E6 protocol has no config_changes entries and therefore
    # follows the exact historical path below.
    if spec.get("config_changes"):
        cfg = replace(cfg, **spec["config_changes"])
        problem = _problem(scene, cfg)
    kind, seed = spec["kind"], job["seed"]
    raw = optimizer = None
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    if kind in ("bare", "astar"):
        result = (_result_for_astar(scene, cfg) if kind == "astar" else
                  _result_for_continuous(scene, spec["family"], seed, cfg))
        path, evaluations = result.path, result.evaluations
        risk_checks, internal_risk, refinement, metadata = 0, None, {}, result.metadata
    else:
        rcfg = risk_config(cfg, seed)
        if kind == "projected":
            optimizer = ConditionalProjectedPolish(problem, rcfg, **spec["options"])
        elif kind == "risk_control":
            from uav_path_planning.conditional_baseline_control import ConditionalBaselineControl
            optimizer = ConditionalBaselineControl(problem, rcfg, axis=2, global_method=spec["family"])
        elif kind == "no_terminal_polish":
            optimizer = ConditionalRiskDE(problem, rcfg, axis=2)
        elif kind == "no_risk":
            optimizer = NominalProjectedPolish(problem, rcfg, **spec["options"])
        elif kind == "nominal_search":
            optimizer = NominalSearchDE(problem, rcfg)
        elif kind == "soft_projected":
            from uav_path_planning.soft_risk_projected_polish import SoftRiskProjectedPolish
            optimizer = SoftRiskProjectedPolish(problem, rcfg, **spec["options"])
        elif kind == "paper_matched":
            from uav_path_planning.paper_matched_controls import create_optimizer
            optimizer = create_optimizer(problem, rcfg, dict(spec["options"]))
        elif kind == "residual_projected":
            from uav_path_planning.residual_conditional_polish import create_optimizer
            optimizer = create_optimizer(problem, rcfg, dict(spec["options"]))
        elif kind == "research_candidate":
            from uav_path_planning.novel_candidates import create_optimizer
            optimizer = create_optimizer(problem, rcfg, dict(spec["options"]))
        else:
            raise ValueError(f"Unsupported kind: {kind}")
        raw = optimizer.optimize()
        path, evaluations = problem.decode(raw.best_vector), raw.evaluations
        risk_checks = raw.internal_collision_checks
        internal_risk = raw.best_evaluation.perturbed_collision_rate if spec["risk_estimator"] else None
        refinement = raw.local_refinement_stats
        metadata = getattr(optimizer, "extension_metadata", {})
    runtime, cpu_seconds = time.perf_counter() - wall_start, time.process_time() - cpu_start
    if kind != "astar" and evaluations != protocol["NFE"]:
        raise AssertionError(f"Unexpected NFE: {evaluations}, job={job['id']}")
    expected_risk = protocol["NFE"] * protocol["K"] if spec["risk_estimator"] else 0
    if risk_checks != expected_risk:
        raise AssertionError(f"Unexpected risk count: {risk_checks}, job={job['id']}")
    metrics = problem.evaluate(path) if path is not None else None
    feasible = bool(metrics is not None and metrics.feasible)
    row = {key: job[key] for key in ("map_root", "scene", "method", "seed")}
    row.update(length=metrics.length if metrics else None, nominal_success=feasible,
               evaluations=evaluations, risk_checks=risk_checks, internal_risk=internal_risk,
               runtime_sec=runtime, cpu_seconds=cpu_seconds)
    external_trials = {}
    for sigma in protocol["sigmas"]:
        free = collision_free_batch(path, scene, sigma * noise, margin=cfg.safety_margin)
        if path is not None:
            # A cheap independent scalar check catches accidental evaluator drift.
            for index in (0, protocol["external_samples"] - 1):
                scalar = path.collision_count(
                    translate_environment(scene.environment, sigma * noise[index]),
                    safety_margin=cfg.safety_margin,
                ) == 0
                if bool(free[index]) != scalar:
                    raise AssertionError("External scalar and batch predicates disagree")
        success = free & feasible
        rate = float(success.mean())
        row["cfr_" + str(sigma)] = rate
        row["pass_" + str(sigma)] = bool(feasible and rate >= 1 - protocol["target_alpha"] - 1e-12)
        external_trials[str(sigma)] = "".join("1" if value else "0" for value in success)
    distribution_external_trials = {}
    for model in protocol.get("distribution_external_models", []):
        serialized_scene = json.dumps(_scene_data(job, protocol), sort_keys=True, ensure_ascii=False)
        external_scene, unit_shifts = _scene_and_distribution_noise(
            serialized_scene, protocol["external_samples"], protocol["external_seed"], model,
        )
        sigma = protocol["distribution_external_sigma"]
        shifts = sigma * unit_shifts
        free = collision_free_batch(path, external_scene, shifts, margin=cfg.safety_margin)
        if path is not None:
            for index in (0, protocol["external_samples"] - 1):
                environment = translate_environment(external_scene.environment, shifts[index])
                scalar = path.collision_count(
                    environment, step=problem.segment_step, safety_margin=cfg.safety_margin,
                ) == 0
                if bool(free[index]) != scalar:
                    raise AssertionError(f"External distribution predicates disagree: {model}/{sigma}/{job['id']}")
        success = free & feasible
        key = f"{model}_{sigma}"
        rate = float(success.mean())
        row["cfr_" + key] = rate
        row["pass_" + key] = bool(feasible and rate >= 1 - protocol["target_alpha"] - 1e-12)
        distribution_external_trials[key] = "".join("1" if value else "0" for value in success)
    detail = {
        "row": row, "waypoints": list(path.waypoints) if path is not None else None,
        "internal_risk": internal_risk, "local_refinement_stats": refinement,
        "extension_metadata": metadata,
    }
    if raw is not None:
        detail["objective_history"] = raw.history
        detail["risk_gate_history"] = raw.gate_history if spec["risk_estimator"] else None
    bundle = {"row": row, "detail": detail, "external_trials": external_trials,
              "distribution_external_trials": distribution_external_trials}
    minimalvalidate_result(job, protocol, bundle)
    return bundle


def minimalvalidate_result(job: dict, protocol: dict, bundle: dict) -> bool:
    """Reject mismatched, partial or internally inconsistent resumable artifacts.

    Does not replan or regenerate stochastic scores. Controller-level source and
    artifact digests supply provenance; this checks the scientific row contract.
    """
    row, detail, trials = bundle["row"], bundle["detail"], bundle["external_trials"]
    for key in ("map_root", "scene", "method", "seed"):
        if row[key] != job[key]:
            raise ValueError(f"Result job mismatch: {key}")
    if detail["row"] != row:
        raise ValueError("Row and detailed row disagree")
    spec = protocol["methods"][job["method"]]
    if spec["kind"] == "astar" and job["seed"] != 0:
        raise ValueError("A* must be recorded once with seed 0")
    if spec["kind"] != "astar" and row["evaluations"] != protocol["NFE"]:
        raise ValueError("Wrong evaluation budget")
    if not isinstance(row["evaluations"], int) or row["evaluations"] < 0:
        raise ValueError("Invalid evaluation count")
    expected_risk = protocol["NFE"] * protocol["K"] if spec["risk_estimator"] else 0
    if row["risk_checks"] != expected_risk:
        raise ValueError("Wrong risk budget")
    for key in ("runtime_sec", "cpu_seconds"):
        if not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError(f"Invalid timing: {key}")
    risk = row["internal_risk"]
    if detail["internal_risk"] != risk:
        raise ValueError("Internal risk fields disagree")
    if spec["risk_estimator"]:
        if risk is None or not math.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid measured internal risk")
    elif risk is not None:
        raise ValueError("Nominal method must report unmeasured internal risk as null")
    points = detail["waypoints"]
    if points is None:
        if row["length"] is not None or row["nominal_success"]:
            raise ValueError("Absent path cannot have a length or be successful")
    else:
        if len(points) < 2 or any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in points):
            raise ValueError("Invalid path waypoints")
        scene = Scenario3D.from_dict(_scene_data(job, protocol))
        if tuple(points[0]) != tuple(scene.start) or tuple(points[-1]) != tuple(scene.goal):
            raise ValueError("Path endpoints disagree with the task")
        metrics = _problem(scene, BenchmarkConfig(**protocol["benchmark_config"])).evaluate(Path3D(points))
        if row["length"] is None or not math.isclose(row["length"], metrics.length, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Stored path length disagrees with waypoints")
        if row["nominal_success"] != bool(metrics.feasible):
            raise ValueError("Stored nominal feasibility disagrees with geometry")
    if set(trials) != {str(sigma) for sigma in protocol["sigmas"]}:
        raise ValueError("Missing or extra perturbation levels")
    for sigma in protocol["sigmas"]:
        bits = trials[str(sigma)]
        if not isinstance(bits, str) or len(bits) != protocol["external_samples"] or set(bits) - {"0", "1"}:
            raise ValueError("Incomplete or invalid external trial vector")
        rate = bits.count("1") / len(bits)
        if not row["nominal_success"] and rate:
            raise ValueError("Failed nominal plans must contribute zero unconditional CFR")
        if not math.isclose(row["cfr_" + str(sigma)], rate, rel_tol=0, abs_tol=1e-12):
            raise ValueError("External rate disagrees with trial outcomes")
        expected_pass = bool(row["nominal_success"] and rate >= 1 - protocol["target_alpha"] - 1e-12)
        if row["pass_" + str(sigma)] != expected_pass:
            raise ValueError("External pass indicator disagrees with the declared gate")
    distribution_trials = bundle.get("distribution_external_trials", {})
    expected_distribution_keys = {
        f"{model}_{protocol['distribution_external_sigma']}"
        for model in protocol.get("distribution_external_models", [])
    }
    if set(distribution_trials) != expected_distribution_keys:
        raise ValueError("Missing or extra distribution-shift external trial vectors")
    for key, bits in distribution_trials.items():
        if not isinstance(bits, str) or len(bits) != protocol["external_samples"] or set(bits) - {"0", "1"}:
            raise ValueError("Incomplete or invalid distribution-shift external trial vector")
        rate = bits.count("1") / len(bits)
        if not row["nominal_success"] and rate:
            raise ValueError("Failed nominal plans must contribute zero unconditional shifted CFR")
        if not math.isclose(row["cfr_" + key], rate, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Stored shifted CFR disagrees with its trial vector")
        expected_pass = bool(row["nominal_success"] and rate >= 1 - protocol["target_alpha"] - 1e-12)
        if row["pass_" + key] != expected_pass:
            raise ValueError("Shifted external pass indicator disagrees with the declared gate")
    json.dumps(bundle, allow_nan=False)
    return True
