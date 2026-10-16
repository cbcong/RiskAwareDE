"""Frozen E0--E6 protocol for the domain-shifted OOD map bank.

This module reuses the scientific method definitions and executor from
``seven_category_protocol`` while replacing only the scenario source with a
pre-frozen, geometry-only manifest.  No planner is called while a manifest is
loaded or validated.  The resulting protocol embeds every scenario, so a
resumed run cannot silently read a mutable map file.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from uav_path_planning.scenarios import Scenario3D  # noqa: E402
from ood_exposure import exposure_state  # noqa: E402
from seven_category_protocol import (  # noqa: E402
    build_protocol as _build_base_protocol,
    execute_job,
    make_jobs,
    minimalvalidate_result,
)

OOD_PROTOCOL_VERSION = "seven-categories-ood-v2"
OOD_STUDY_KIND = "ood_v2_domain_shift_confirmatory"
OOD_PILOT_STUDY_KIND = "ood_v2_domain_shift_pilot"
P6_METHODS = ("full", "risk_polish_de", "risk_polish_jade", "risk_polish_shade",
              "risk_polish_lshade", "risk_polish_pso", "bare_de", "bare_lshade")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_manifest(manifest_path: Path) -> tuple[dict, list[dict], list[Scenario3D]]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file() or manifest_path.name != "manifest.json":
        raise ValueError("map_manifest must point to a frozen manifest.json")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "frozen_geometry_only":
        raise ValueError("OOD manifest is not frozen_geometry_only")
    if manifest.get("generator_version") != "research-scenarios-v2-ood":
        raise ValueError("unexpected OOD generator version")
    if manifest.get("domain_shift") != "density-height-topology":
        raise ValueError("manifest is not the declared density-height-topology shift")
    ledger = exposure_state(manifest_path.parent)
    if ledger.get("planner_called_during_generation") or ledger.get("robustness_scored_during_generation"):
        raise ValueError("OOD map generation used a planner or robustness scorer")
    entries = manifest.get("maps")
    if not isinstance(entries, list) or not entries:
        raise ValueError("OOD manifest has no map entries")
    expected_count = int(manifest.get("map_count", len(entries)))
    if len(entries) != expected_count:
        raise ValueError("OOD map_count does not match entries")
    scenarios = []
    seen = set()
    for entry in entries:
        name = entry.get("name")
        rel = entry.get("relative_path")
        if not isinstance(name, str) or name in seen or not isinstance(rel, str):
            raise ValueError("OOD manifest contains duplicate or malformed map entries")
        seen.add(name)
        map_path = (manifest_path.parent / rel).resolve()
        if manifest_path.parent not in map_path.parents or map_path.suffix != ".json":
            raise ValueError("OOD map path escapes the frozen manifest directory")
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or sha256(map_path) != expected_hash:
            raise ValueError(f"OOD map hash mismatch: {name}")
        scenario = Scenario3D.from_dict(_read_json(map_path))
        if scenario.name != name:
            raise ValueError(f"OOD map name mismatch: {name}")
        metadata = scenario.metadata
        if metadata.get("generator") != "research-scenarios-v2-ood":
            raise ValueError(f"OOD map {name} has an unexpected generator")
        if metadata.get("planner_dependent_filtering") is not False:
            raise ValueError(f"OOD map {name} lacks planner-independent provenance")
        scenarios.append(scenario)
    manifest_digest = manifest.get("maps_digest")
    if not isinstance(manifest_digest, str) or not manifest_digest:
        raise ValueError("OOD manifest lacks maps_digest")
    canonical = json.dumps(entries, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest_digest:
        raise ValueError("OOD maps_digest disagrees with the map entries")
    manifest["current_exposure_ledger"] = ledger or manifest.get("exposure_ledger", {})
    return manifest, entries, scenarios


def build_protocol(repo_root: Path, manifest_path: Path, profile: str = "full") -> dict:
    """Build a portable OOD protocol without starting any optimizer.

    ``profile=smoke`` is deliberately limited to one map per OOD family and is
    intended only for controller/integration checks.  It must never be used as
    the paper's OOD result.
    """
    if profile not in ("full", "smoke", "p6"):
        raise ValueError("profile must be full, smoke or p6")
    repo_root = Path(repo_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest, entries, scenarios = _load_manifest(manifest_path)
    ledger = manifest["current_exposure_ledger"]
    blind_eligible = ledger["blind_eligible"]
    if manifest.get("purpose") == "blind":
        if profile == "smoke":
            raise ValueError("A blind bank cannot be used for smoke diagnostics; use a separate pilot bank")
        lock_relative = manifest.get("method_lock_relative_path")
        if lock_relative != "method_lock.json":
            raise ValueError("Blind manifest has no frozen method lock")
        lock_file = manifest_path.parent / lock_relative
        if sha256(lock_file) != manifest.get("method_lock_sha256"):
            raise ValueError("Blind map method lock hash mismatch")
        from ood_method_lock import validate_method_lock
        method_lock = validate_method_lock(lock_file, repo_root)
        # The frozen lock is the authoritative scientific definition. Building
        # an unconditional legacy "full" protocol here would silently replace
        # the locked soft main and matched E2/E3/E4/E6 controls.
        base = deepcopy(method_lock["protocol"])
        base_profile = method_lock.get("protocol_profile", "full")
        method_protocol_digest = method_lock["protocol_sha256"]
    else:
        # Existing pilot/smoke behavior remains backwards compatible.
        base = _build_base_protocol(repo_root, profile="full")
        base_profile = "full"
        method_protocol_digest = None
    if profile == "p6" and (manifest.get("purpose") != "blind" or base_profile != "manuscript"):
        raise ValueError("P6 requires a pre-generation blind bank and manuscript-profile method lock")
    # Known-map checkpoints have no experimental identity on a new map bank.
    # Even a manuscript-profile lock must plan every OOD job from scratch.
    base.pop("reuse", None)
    by_family = {}
    for scenario in scenarios:
        family = scenario.metadata.get("family")
        if family in by_family:
            by_family[family].append(scenario)
        else:
            by_family[family] = [scenario]
    if profile == "smoke":
        selected = []
        seen_families = set()
        for scenario in scenarios:
            family = scenario.metadata.get("family")
            if family not in seen_families:
                selected.append(scenario)
                seen_families.add(family)
        scenarios = selected
        base["planner_seeds"] = base["planner_seeds"][:1]
        base["benchmark_config"]["seeds"] = list(base["planner_seeds"])
    root_seed = int(manifest["root_seed"])
    base["protocol_version"] = OOD_PROTOCOL_VERSION
    base["study_kind"] = OOD_STUDY_KIND if blind_eligible else OOD_PILOT_STUDY_KIND
    base["blind_eligible_at_freeze"] = blind_eligible
    base["method_protocol_profile"] = base_profile
    base["method_protocol_sha256"] = method_protocol_digest
    base["new_map_checkpoint_policy"] = "All OOD maps require new planning; no known-map result reuse."
    base["profile"] = profile
    base["smoke_test"] = profile == "smoke"
    base["map_roots"] = [root_seed]
    base["scenarios"] = {str(root_seed): [scenario.to_dict() for scenario in scenarios]}
    base["ood_manifest"] = {
        "path_at_freeze": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "maps_digest": manifest["maps_digest"],
        "generator_version": manifest["generator_version"],
        "domain_shift": manifest["domain_shift"],
        "map_count_manifest": len(entries),
        "map_count_protocol": len(scenarios),
        "families": sorted({str(s.metadata.get("family")) for s in scenarios}),
        "exposure_ledger": manifest["current_exposure_ledger"],
    }
    base["evidence_scope"] = (
        "Prospective blind OOD evaluation: method/protocol/source locked before bank generation; no outcomes previously exposed."
        if blind_eligible else
        "DESCRIPTIVE OOD PILOT ONLY: this bank is pilot, exposed, or lacks a pre-generation method lock; never confirmatory blind evidence."
    )
    base["limits"] = [
        "This bank shifts obstacle density, height and topology while retaining bounds and endpoints.",
        "The map generator performs geometry-only checks and never calls a planner or robustness scorer.",
        "Do not tune parameters or select a method after inspecting OOD outcomes.",
        "Baseline implementations are local reproductions/adaptations, not original-author published values.",
        "Empirical external pass rate is not a finite-sample confidence certificate.",
        "A* is deterministic and descriptive; it is not included in stochastic paired tests.",
    ]
    if base_profile == "manuscript":
        base["limits"].extend([
            "Locked main is soft_beta050; matched E2/E3/E4/E6 retain beta=.05 and the frozen matched-control definitions.",
            "E3 compares complete global-search packages; native baseline scalar adaptation outside the gate differs from DE lexicographic selection.",
            "E5 removes risk-dependent soft cost by definition; E6 includes the discrete-estimator and finite-difference interaction.",
            "Every checkpoint is newly computed on OOD maps; the known-map reuse declaration in the method lock is inactive here.",
        ])
    base["statistical_analysis"] = {
        "unit": "map-level paired seed means within the frozen OOD bank",
        "metrics": ["length", "cfr_0.6"],
        "multiple_testing": "All planned E0-E6 contrasts and two metrics within this one OOD bank are Holm-adjusted.",
        "astar": "descriptive only; no repeated artificial seeds",
        "interpretation": base["evidence_scope"] + " Nonsignificance is not equivalence.",
    }
    base["expected_plans"] = sum(
        len(scenarios) * (1 if spec["kind"] == "astar" else len(base["planner_seeds"]))
        for spec in base["methods"].values()
    )
    if profile == "p6":
        # P6 is an external-validity supplement, not a duplicate E0--E6 matrix.
        base["methods"] = {name: base["methods"][name] for name in P6_METHODS}
        base.pop("categories", None)
        base["p6_comparisons"] = [{"reference": "full", "comparator": name}
                                  for name in P6_METHODS if name != "full"]
        base["study_kind"] = "p6_ood_locked_main_confirmatory" if blind_eligible else "p6_ood_locked_main_exposed"
        base["statistical_analysis"]["multiple_testing"] = "Seven fixed main-vs-baseline contrasts x two metrics; one 14-test Holm family."
        base["statistical_analysis"]["purpose"] = "External validity; no repeated E0/E4/E5/E6 component analysis on OOD."
        base["expected_plans"] = len(scenarios) * len(base["methods"]) * len(base["planner_seeds"])
    return json.loads(json.dumps(base, ensure_ascii=False, allow_nan=False))


__all__ = [
    "OOD_PROTOCOL_VERSION", "OOD_STUDY_KIND", "build_protocol", "execute_job",
    "make_jobs", "minimalvalidate_result",
]
